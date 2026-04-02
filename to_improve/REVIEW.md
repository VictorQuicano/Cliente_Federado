# DDPG Music Recommender - Training Flow Review

Full review of the DDPG actor-critic training pipeline. The model shows **misleadingly good training metrics** (reward 0.999, precision@k = 1.0) but **poor generalization** (test reward 0.758, volatile test metrics). Below are the root causes, ordered by severity.

---

## Summary of Symptoms

| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| Avg Reward | 0.9992 | 0.9691 | 0.7588 |
| Precision@5 | 1.0000 | 1.0000 | 0.4000 |
| NDCG@10 | 0.9825 | 0.9443 | 0.9217 |
| Critic Loss | 0.0000 | - | - |
| Epsilon (epoch 50) | 0.70 | - | - |

The training metrics are artificially inflated by multiple compounding issues. The critic loss reaching exactly 0.0 and precision being perfect 1.0 on training are red flags, not signs of success.

---

## Critical Issues

These are bugs that fundamentally break the training algorithm.

### 1. `interaction_ratio` cast to `int` destroys reward signal

**File:** `libs/client.py:154`

```python
listened_complete = int(row['interaction_ratio'])
```

If `interaction_ratio` is a float between 0 and 1 (e.g. 0.75 meaning 75% of a song was listened), `int(0.75)` becomes **0**. This means the reward function's beta component:

```python
recompensa = alpha * count_factor + beta * interaction_ratio
#                                    0.4  *  0  = always 0!
```

**Impact:** 40% of the reward signal is destroyed. The model only learns from interaction counts, ignoring how much of each song was actually listened to. Every track with ratio < 1.0 gets zero contribution from the ratio term.

**Fix:** Change to `float(row['interaction_ratio'])`.

---

### 2. Actor learning rate is 1000x too low

**File:** `libs/model.py:51`

```python
actor_lr: float = 1e-6   # Actor
critic_lr: float = 1e-3  # Critic
```

The actor learning rate (`1e-6`) is the same order of magnitude as typical weight decay penalties. The actor policy essentially doesn't update - gradient steps are negligible.

**Impact:** The actor remains close to its random initialization for the entire training run. The "policy" is basically random noise processed through a frozen network. This explains why high epsilon (random exploration) still produces decent rewards - the actor's actual output barely matters.

**Fix:** Use `actor_lr=1e-4` (100x increase). Standard DDPG uses actor LR 1-10x lower than critic, not 1000x.

---

### 3. Exploration noise added to states, not actions (wrong DDPG)

**File:** `libs/model.py:200-203`

```python
def _update_actor(self, batch):
    states = torch.cat([b['state'] for b in batch]).to(self.device)
    
    noise = torch.randn_like(states) * 0.1       # Noise shaped like STATES
    noisy_states = states + noise                  # Added to STATES
    actions = self.actor.forward_from_state(noisy_states)  # Actor sees corrupted states
```

Standard DDPG exploration adds noise to **actions**: `action = actor(state) + noise`. This code adds noise to **state representations** before passing them to the actor's decoder. This corrupts the learned state representation rather than promoting action-space exploration.

**Impact:** The actor update optimizes a policy that receives perturbed state inputs, which doesn't correspond to how the actor is actually used during inference. The critic evaluates `(clean_state, noisy_action)` pairs, creating a mismatch.

**Fix:**
```python
actions = self.actor.forward_from_state(states)
noise = torch.randn_like(actions) * 0.1
noisy_actions = actions + noise
q_values = self.critic(states, noisy_actions)
```

---

### 4. Critic weight initialization too conservative

**File:** `libs/critic.py:50-57`

```python
gain = 0.01  # Extremely small

# Output layer:
nn.init.uniform_(module.weight, -0.003, 0.003)
```

With gain=0.01, Xavier uniform initialization produces weights ~100x smaller than default. The output layer weights are in [-0.003, 0.003]. Combined with Q-value clamping to [-10, 10] (line 125), the critic:

1. Starts outputting values very close to 0
2. Has tiny gradients flowing back through the network
3. Reaches a "loss = 0.0" equilibrium quickly because predictions and targets are both near-zero

**Impact:** The critic loss going to 0.0000 is not convergence - it's the network being stuck near zero. The actor maximizes `-Q` which is also near zero, providing no meaningful policy gradient.

**Fix:** Use `gain=1.0` (default Xavier) and `uniform(-0.01, 0.01)` for output. Remove or widen the Q-value clamp to [-100, 100].

---

### 5. Actor weight initialization too conservative

**File:** `libs/actor.py:74`

```python
nn.init.xavier_uniform_(module.weight, gain=0.1)
```

The actor uses gain=0.1, which is 10x smaller than the default Xavier initialization. The decoder has 3 linear layers (288->512->256->64) each with this conservative init, compounding the effect.

**Impact:** Vanishing gradients through the decoder. Even if the learning rate were fixed, the initial weights are so small that outputs barely respond to input changes. Combined with Tanh activation on the output (which saturates near 0 for small inputs), the actor produces near-zero action vectors initially.

**Fix:** Use `gain=1.0` (default) for hidden layers. Optionally keep a smaller gain (0.1-0.3) only for the output layer.

---

### 6. Double Q-network is implemented but never activated

**File:** `libs/critic.py:117-120`

```python
if return_min:
    q_value = torch.min(q1, q2)  # TD3-style overestimation reduction
else:
    q_value = q1  # Only Q1 used (DEFAULT)
```

The critic has two full Q-networks (doubling parameters and computation), but `return_min=False` is the default and is **never overridden** in any call from `model.py`:

```python
# model.py:174 - target computation
next_q_values = self.critic_target(next_states, next_actions)  # return_min=False

# model.py:180 - current Q
current_q_values = self.critic(states, actions)  # return_min=False

# model.py:204 - actor update
q_values = self.critic(states, actions)  # return_min=False
```

**Impact:** The second Q-network trains but its output is never used. Q-value overestimation (a known DDPG failure mode) is not mitigated. Extra GPU memory and computation wasted.

**Fix:** Use `return_min=True` for target computation (line 174). This is the standard TD3 improvement.

---

## High-Severity Issues

Design flaws that severely degrade training quality and metric reliability.

### 7. Evaluation feeds its own predictions back as history

**File:** `libs/model.py:400-410`

```python
# During evaluation:
if recommendations:
    recommended = recommendations[0]
    recommended_item = {
        'embedding': recommended['embedding'],
        'context': context,
        'reward': recommended['reward'],
        'track_id': recommended['track_id']
    }
    user_history.append(recommended_item)  # Model's OWN prediction becomes history
```

During evaluation, the model builds `user_history` from its **own recommendations**, not from ground truth items. This is autoregressive evaluation where errors compound: a bad recommendation distorts the state, which produces another bad recommendation, and so on.

**Impact:** Evaluation metrics don't reflect how the model would perform on real user sequences. The metrics are measuring the model's ability to be self-consistent, not its ability to predict actual user behavior.

**Fix:** During evaluation, append the actual ground truth item (`item`) to `user_history`, not the model's recommendation. Use recommendations only for metric computation.

---

### 8. Evaluation recommends from the same split it's evaluating

**File:** `libs/model.py:392-397`

```python
recommendations = self.recommender.recommend(
    action_vector=action_vec.squeeze(0).cpu(),
    context=context,
    n=10,
    split_type=split_type,  # SplitType.TEST or SplitType.VALIDATION
)
```

When evaluating on the test set, the recommender searches for items **within the test set itself**. This means:

- The search space is limited to ~34k items (test) instead of ~159k (train)
- The model can only recommend items that are in the ground truth
- Precision/recall are artificially inflated because the candidate pool IS the answer set

**Impact:** Metrics like precision@k = 1.0 on validation become meaningless. The model is essentially choosing from a curated list of "correct" answers.

**Fix:** Always recommend from `SplitType.TRAIN` (the known catalog) regardless of which split is being evaluated.

---

### 9. Metrics computed on accumulated `user_history`, not recommendation lists

**File:** `libs/model.py:335, 421`

```python
# After training loop:
metrics = self._compute_metrics_comprehensive(user_history, train_items, SplitType.TRAIN)

# After evaluation loop:
metrics = self._compute_metrics_comprehensive(user_history, eval_items, split_type)
```

`user_history` is the accumulated list of items the model "visited" (up to ~1000 items). This is passed as "recommendations" to the metrics function, which compares it against the full split (158k+ items).

**Impact:** The metrics compare a sequence of ~1000 accumulated items against ~158k ground truth items. `precision@k` compares the first k items of this accumulated history against the full dataset's track IDs. This is not standard recommendation evaluation.

**Fix:** Collect `all_recommendations` as a separate list during the loop. Compute metrics on each step's top-k recommendations against the ground truth at that step.

---

### 10. Raw dot product instead of cosine similarity in recommender

**File:** `libs/recommender.py:56`

```python
scores = torch.mm(action_vector, embeddings_tensor.t()).squeeze(0)  # [n]
```

The actor outputs Tanh-bounded vectors in [-1, 1], but item embeddings from the dataset have arbitrary magnitudes. Raw dot product score depends on both direction AND magnitude, creating a bias toward embeddings with larger norms regardless of relevance.

**Impact:** The recommender systematically favors items with high-magnitude embeddings over items that are directionally similar to the action vector. The actor cannot learn to compensate for this because its output is bounded by Tanh.

**Fix:** Normalize both vectors before computing the dot product (cosine similarity):
```python
action_norm = F.normalize(action_vector, p=2, dim=-1)
embed_norm = F.normalize(embeddings_tensor, p=2, dim=-1)
scores = torch.mm(action_norm, embed_norm.t()).squeeze(0)
```

---

### 11. Epsilon barely decays - model explores 70% of the time at epoch 50

**Configuration:** `epsilon_start=0.9, epsilon_decay=0.995, num_epochs=50`

```
epsilon(50) = 0.9 * 0.995^50 = 0.9 * 0.778 = 0.700
```

After 50 epochs of training, the model still makes random exploratory actions **70% of the time**. It would take ~460 epochs to reach epsilon=0.1 (the target).

**Impact:** The learned policy is almost never used. The model's actual recommendations during training are dominated by random noise, not the actor's output. Reward metrics during training reflect random-exploration quality, not policy quality.

**Fix:** Use per-step decay instead of per-epoch, or use a much faster decay rate (e.g., `epsilon_decay=0.95` per epoch to reach 0.1 in ~43 epochs).

---

### 12. `_prepare_state` returns wrong number of values on empty history

**File:** `libs/model.py:99-100`

```python
if len(recent_items) == 0:
    return None, None       # Returns 2 values
# ...
return item_embeddings, item_contexts, last_context  # Returns 3 values
```

Callers unpack 3 values on line 257: `item_embeddings, item_contexts, last_context = state_rep`. This is currently guarded by a check on line 253 (`if state_rep[0] is None`), but the inconsistent return signature is fragile and will break if any caller forgets the check.

**Impact:** Low immediate impact (the guard works), but a maintenance hazard.

**Fix:** `return None, None, None`

---

## Medium Issues

Suboptimal choices that reduce training effectiveness.

### 13. Training sees only 0.63% of data per epoch

**File:** `libs/model.py:228` (default `max_steps=1000`)

With 158,615 training items and max_steps=1000, each epoch processes only 1000 items (0.63%). Over 50 epochs: 50,000 total steps, meaning most training items are never seen. The model overfits to the first 1000 items of the temporal sequence each epoch.

**Fix:** Either increase `max_steps` significantly, or shuffle the training data and iterate through all of it.

---

### 14. Evaluation computes metrics on 1.47% of eval data

**File:** `libs/model.py:355` (default `max_steps=500`)

With 33,989 test items and max_steps=500, evaluation metrics are computed on 500 items (1.47%). This tiny sample makes metrics noisy and unreliable, explaining the volatility in test metrics across epochs.

**Fix:** Increase `max_steps` for evaluation or evaluate on the full set.

---

### 15. Actor Tanh output mismatched with embedding space

**File:** `libs/actor.py:64`

```python
nn.Linear(256, embedding_dim),
nn.Tanh()  # Output bounded to [-1, 1]
```

The actor output is bounded to [-1, 1] per dimension, but item embeddings from the dataset have no such constraint. When computing dot products, the bounded action vector has a fundamental disadvantage in matching unbounded embeddings.

**Fix:** Either (a) remove Tanh and normalize the output to unit length, or (b) normalize all item embeddings to unit length during preprocessing, or (c) use cosine similarity (Issue #10 fix).

---

### 16. Weight decay is inverted between actor and critic

**File:** `libs/model.py:78-79`

```python
self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr, weight_decay=1e-4)   # Stronger
self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr, weight_decay=1e-6) # Weaker
```

The actor has 100x stronger weight decay than the critic. Typically the critic needs more regularization because it can diverge when estimating Q-values. The actor's already-tiny learning rate (1e-6) combined with strong weight decay (1e-4) means weight decay is a significant fraction of the learning signal.

**Fix:** Swap or equalize: `actor_weight_decay=1e-6`, `critic_weight_decay=1e-4`.

---

## Recommended Fix Priority

The issues compound on each other. Fix in this order:

1. **Fix `int()` cast on `interaction_ratio`** (Issue 1) - restores 40% of reward signal
2. **Increase actor LR to 1e-4** (Issue 2) - lets actor actually learn
3. **Move noise from states to actions** (Issue 3) - correct DDPG exploration
4. **Fix critic/actor initialization** (Issues 4, 5) - enable gradient flow
5. **Activate double-Q (`return_min=True`)** (Issue 6) - reduce Q overestimation
6. **Fix evaluation: use ground truth history, recommend from TRAIN** (Issues 7, 8) - get reliable metrics
7. **Fix metric computation** (Issue 9) - measure what matters
8. **Use cosine similarity in recommender** (Issue 10) - fixes magnitude bias and Tanh mismatch
9. **Fix epsilon decay rate** (Issue 11) - let exploitation happen during training
10. **Increase max_steps** (Issues 13, 14) - see more data
