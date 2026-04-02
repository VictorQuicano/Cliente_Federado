# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Music recommender system using a DDPG (Deep Deterministic Policy Gradient) actor-critic architecture, designed for federated learning via Flower (`flwr`). Each client represents a single user's listening history. The codebase and comments are primarily in Spanish.

## Running

The main entry point is `to_improve.ipynb`. It requires:
- A user CSV at a path like `Dataset/processed_users/{user_id}_processed.csv` with columns: `track_id`, `interaction_count`, `interaction_ratio`, `year`, `month`, `day`, `hour`, `minute`, `second`, `day_of_week`
- An embeddings CSV (e.g. `music_4_all_compress_64.csv`) with columns: `track_id, dim1, dim2, ...` (64-dim)
- Optional: `.env` with `API_URL` for fetching embeddings on cache miss

The notebook imports the library as `libs_to_improve` (mapped to the `libs/` directory). There is no `__init__.py` — the notebook adjusts the import path at runtime.

Dependencies: `torch`, `flwr` (Flower), `pandas`, `numpy`, `tqdm`, `matplotlib`, `requests`, `python-dotenv`. Python 3.11.

## Architecture

### Data Flow

```
User CSV → Client (load, split, embed) → Actor (history → action vector) → Recommender (dot product → top-N) → Reward
                                           ↕                                                                      ↓
                                        Critic (state, action → Q-value) ← DDPG update ← Experience Replay Buffer
```

### Key Dimensions

| Constant | Value | Where |
|---|---|---|
| `EMBEDDING_DIM` | 64 | Item embeddings |
| `CONTEXT_DIM` | 32 | Temporal context (day_of_week 8d + month 8d + is_workday 8d + hour sinusoidal 8d) |
| `HIDDEN_DIM` | 256 | Actor/Critic hidden layers |
| `state_dim` | 288 | HIDDEN_DIM + CONTEXT_DIM (Critic input) |
| `HISTORY_LENGTH` | 10 | Sliding window of recent items fed to Actor |

### Module Roles (`libs/`)

- **`client.py`** — `Client` class: loads user CSV, manages embedding cache (JSON + CSV sources), computes rewards via a pluggable `recompensa_func`, splits data temporally into train/test/val. Each item is a dict with `embedding`, `context`, `reward`, `track_id`.

- **`actor.py`** — `ContextAwareActor`: takes a window of item embeddings + per-item temporal context, processes through linear layers + multi-head attention, then decodes to a next-item embedding vector (the "action"). Has `forward()` (full pipeline from embeddings) and `forward_from_state()` (decoder-only, used in DDPG updates).

- **`critic.py`** — `ContextAwareCritic`: twin Q-networks (TD3-style double Q) that evaluate (state, action) pairs. Q-values clamped to [-10, 10]. Very conservative weight initialization (gain=0.01) to prevent value explosion.

- **`recommender.py`** — `Recommender`: given an action vector from Actor, finds the closest items by dot product against all embeddings in a context-filtered subset. Returns top-N scored items. Caches context-filtered lookups.

- **`model.py`** — `RecommenderTrainer`: orchestrates DDPG training. Manages experience replay buffer (capacity 10k), soft target network updates (tau=0.005), epsilon-greedy exploration with Gaussian noise. Computes comprehensive evaluation metrics: Precision@k, Recall@k, MAP@k, NDCG@k, embedding diversity, context coverage, temporal entropy. Also handles plotting and model save/load.

- **`federated_client.py`** — `FlowerClient`: wraps `RecommenderTrainer` as a Flower `NumPyClient` for federated learning. Supports dynamic user reassignment via `config["target_user_id"]` during `fit()`/`evaluate()`. Serializes/deserializes both Actor and Critic weights.

### Training Loop (DDPG)

1. Iterate over user's training items chronologically
2. Build state from last 10 items' embeddings + temporal context
3. Actor produces action vector; add epsilon-greedy Gaussian noise
4. Recommender finds best matching item via dot product
5. Store (state, action, reward, next_state) in replay buffer
6. Sample mini-batch from buffer, update Critic (MSE on TD target) then Actor (-Q maximization)
7. Soft-update target networks every `target_update_freq` steps

### Reward Function

Reward is `alpha * log1p(interaction_count)/log1p(10) + beta * interaction_ratio`, clipped to [0, 1]. Default alpha=0.6, beta=0.4.
