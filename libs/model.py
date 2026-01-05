from .actor import ContextAwareActor
from .critic import ContextAwareCritic
from .client import Client, SplitType
from .recommender import Recommender

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import copy
from collections import defaultdict, deque
import math
from tqdm import tqdm

import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List
import numpy as np
from datetime import datetime
import os
import logging


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import copy
from collections import defaultdict, deque
import math
from tqdm import tqdm

import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List
import numpy as np
from datetime import datetime
import os

class RecommenderTrainer:
    def __init__(self,
                 actor: ContextAwareActor,
                 critic: ContextAwareCritic,
                 client: Client,
                 recommender: Recommender,
                 gamma: float = 0.9,
                 tau: float = 0.005,
                 actor_lr: float = 1e-6,
                 critic_lr: float = 1e-4,
                 batch_size: int = 64,
                 state_size: int = 10,
                 device: torch.device = torch.device('cpu'),
                 target_update_freq: int = 100):
        
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        
        self.actor_target = copy.deepcopy(actor).to(device)
        self.critic_target = copy.deepcopy(critic).to(device)
        
        for param in self.actor_target.parameters():
            param.requires_grad = False
        for param in self.critic_target.parameters():
            param.requires_grad = False
        
        self.client = client
        self.recommender = recommender
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.state_size = state_size
        self.device = device
        self.target_update_freq = target_update_freq
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr, weight_decay=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr, weight_decay=1e-6)
        
        self.memory = []
        self.memory_capacity = 10000
        
        self.training_history = {
            'actor_loss': [],
            'critic_loss': [],
            'train_rewards': [],
            'train_metrics': defaultdict(list),
            'val_metrics': defaultdict(list),
            'test_metrics': defaultdict(list)
        }
        
        self.step_count = 0

    def _prepare_state(self, user_items: List[Dict], context: Dict) -> Tuple[torch.Tensor, Dict]:
        recent_items = user_items[-self.state_size:]
        
        if len(recent_items) == 0:
            return None, None
        
        item_embeddings = []
        item_contexts = {
            'day_of_week': [],
            'hour_of_day': [],
            'is_workday': [],
            'month': []
        }
        
        for item in recent_items:
            item_embeddings.append(item['embedding'].unsqueeze(0))
            item_contexts['day_of_week'].append(item['context']['day_of_week'])
            item_contexts['hour_of_day'].append(item['context']['hour_of_day'])
            item_contexts['is_workday'].append(item['context']['is_workday'])
            item_contexts['month'].append(item['context']['month'])
        
        item_embeddings = torch.cat(item_embeddings, dim=0).unsqueeze(0)
        
        for key in item_contexts:
            item_contexts[key] = torch.tensor(item_contexts[key], dtype=torch.long).unsqueeze(0)
        
        last_context = {
            'day_of_week': torch.tensor([context['day_of_week']], dtype=torch.long),
            'hour_of_day': torch.tensor([context['hour_of_day']], dtype=torch.float32),
            'is_workday': torch.tensor([context['is_workday']], dtype=torch.long),
            'month': torch.tensor([context['month']], dtype=torch.long)
        }
        
        return item_embeddings, item_contexts, last_context

    def _add_to_memory(self, state_vec: torch.Tensor, action_vec: torch.Tensor, 
                      reward: float, next_state_vec: torch.Tensor, done: bool):
        if len(self.memory) >= self.memory_capacity:
            self.memory.pop(0)
        self.memory.append({
            'state': state_vec.detach().cpu(),
            'action': action_vec.detach().cpu(),
            'reward': reward,
            'next_state': next_state_vec.detach().cpu() if next_state_vec is not None else None,
            'done': done
        })

    def _sample_batch(self) -> List[Dict]:
        if len(self.memory) < self.batch_size:
            return []
        indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        return [self.memory[i] for i in indices]

    def _update_critic(self, batch: List[Dict]) -> float:
        # Preparar batch
        states = torch.cat([b['state'] for b in batch]).to(self.device)
        actions = torch.cat([b['action'] for b in batch]).to(self.device)
        rewards = torch.tensor([b['reward'] for b in batch], 
                            dtype=torch.float32).to(self.device).unsqueeze(1)
        dones = torch.tensor([b['done'] for b in batch], 
                            dtype=torch.float32).to(self.device).unsqueeze(1)
        next_states = torch.cat([b['next_state'] if b['next_state'] is not None 
                                else torch.zeros_like(b['state']) 
                                for b in batch]).to(self.device)
        
        with torch.no_grad():
            # 1. Obtener acciones para los siguientes estados
            next_actions = self.actor_target.forward_from_state(next_states)
            
            # 2. Calcular valores Q para los siguientes estados
            next_q_values = self.critic_target(next_states, next_actions)
            
            # 3. Calcular target Q-values (Q-learning con estados terminales)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # 4. Calcular valores Q actuales
        current_q_values = self.critic(states, actions)
        
        # 5. Calcular pérdida (MSE entre Q actual y target)
        critic_loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # 6. Optimizar
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        
        # 7. Clip de gradientes (opcional pero recomendado)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        
        self.critic_optimizer.step()
        
        return critic_loss.item()

    def _update_actor(self, batch: List[Dict]) -> float:
        states = torch.cat([b['state'] for b in batch]).to(self.device)
        
        # Añadir ruido para exploración (importante en DDPG)
        noise = torch.randn_like(states) * 0.1
        noisy_states = states + noise
        
        actions = self.actor.forward_from_state(noisy_states)
        q_values = self.critic(states, actions)
        
        # Actor loss debería minimizar -Q (maximizar Q)
        actor_loss = -q_values.mean()
        
        # Añadir regularización de entropía
        action_std = actions.std()
        entropy_regularization = -0.01 * action_std  # Promover diversidad
        actor_loss += entropy_regularization
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        return actor_loss.item()

    def _soft_update_targets(self):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def train_epoch(self, epsilon: float = 0.1, max_steps: int = 1000, print_logs: bool = False, 
                exploration_noise: float = 0.2) -> Dict[str, float]:
        train_items = self.client.get_split(SplitType.TRAIN)
        if not train_items:
            return {}
        
        epoch_rewards = []
        epoch_actor_loss = []
        epoch_critic_loss = []
        
        user_history = []
        step = 0
        
        if print_logs:
            pbar = tqdm(total=min(len(train_items), max_steps), desc="Training")
        else:
            pbar = None
        
        for i, item in enumerate(train_items):
            if step >= max_steps:
                break
            
            context = item['context']
            
            state_rep = self._prepare_state(user_history, context)
            if state_rep[0] is None:
                user_history.append(item)
                continue
            
            item_embeddings, item_contexts, last_context = state_rep
            
            with torch.no_grad():
                state_vec, action_vec = self.actor(
                    item_embeddings.to(self.device),
                    {k: v.to(self.device) for k, v in item_contexts.items()},
                    {k: v.to(self.device) for k, v in last_context.items()}
                )
            
            if np.random.random() < epsilon:
                noise = torch.randn_like(action_vec) * exploration_noise
                action_vec = action_vec + noise
                # Normalizar si es necesario
                action_vec = torch.tanh(action_vec)
            
            recommendations = self.recommender.recommend(
                action_vector=action_vec.squeeze(0).cpu(),
                context=context,
                n=1,
                split_type=SplitType.TRAIN
            )
            
            if recommendations:
                recommended = recommendations[0]
                reward = recommended['reward']
                recommended_item = {
                    'embedding': recommended['embedding'],
                    'context': context,
                    'reward': reward,
                    'track_id': recommended['track_id']
                }
                
                next_user_history = user_history + [recommended_item]
                next_state_rep = self._prepare_state(next_user_history, context)
                
                if next_state_rep[0] is not None:
                    next_item_embeddings, next_item_contexts, _ = next_state_rep
                    with torch.no_grad():
                        next_state_vec, _ = self.actor(
                            next_item_embeddings.to(self.device),
                            {k: v.to(self.device) for k, v in next_item_contexts.items()},
                            {k: v.to(self.device) for k, v in last_context.items()}
                        )
                else:
                    next_state_vec = None
                
                self._add_to_memory(
                    state_vec.cpu(),
                    action_vec.cpu(),
                    reward,
                    next_state_vec.cpu() if next_state_vec is not None else None,
                    done=False
                )
                
                epoch_rewards.append(reward)
                user_history = next_user_history
            else:
                user_history.append(item)
                continue
            
            batch = self._sample_batch()
            if batch:
                critic_loss = self._update_critic(batch)
                actor_loss = self._update_actor(batch)
                
                epoch_critic_loss.append(critic_loss)
                epoch_actor_loss.append(actor_loss)
            
            self.step_count += 1
            if self.step_count % self.target_update_freq == 0:
                self._soft_update_targets()
            if pbar:
                pbar.update(1)
            step += 1

        if pbar:   
            pbar.close()
        
        metrics = self._compute_metrics_comprehensive(user_history, train_items, SplitType.TRAIN)
        
        if epoch_actor_loss:
            self.training_history['actor_loss'].append(np.mean(epoch_actor_loss))
        if epoch_critic_loss:
            self.training_history['critic_loss'].append(np.mean(epoch_critic_loss))
        if epoch_rewards:
            self.training_history['train_rewards'].append(np.mean(epoch_rewards))
        
        for key, value in metrics.items():
            self.training_history['train_metrics'][key].append(value)
        
        return {
            'actor_loss': np.mean(epoch_actor_loss) if epoch_actor_loss else 0,
            'critic_loss': np.mean(epoch_critic_loss) if epoch_critic_loss else 0,
            'avg_reward': np.mean(epoch_rewards) if epoch_rewards else 0,
            **metrics
        }

    def evaluate(self, split_type: SplitType = SplitType.VALIDATION, 
                max_steps: int = 500, print_logs: bool = False) -> Dict[str, float]:
        self.actor.eval()
        self.critic.eval()
        
        eval_items = self.client.get_split(split_type)
        if not eval_items:
            return {}
        
        user_history = []
        all_recommendations = []
        step = 0
        
        if print_logs:
            pbar = tqdm(total=min(len(eval_items), max_steps), desc=f"Evaluating {split_type.value}")
        else:
            pbar = None

        with torch.no_grad():
            for i, item in enumerate(eval_items):
                if step >= max_steps:
                    break
                
                context = item['context']
                
                state_rep = self._prepare_state(user_history, context)
                if state_rep[0] is None:
                    user_history.append(item)
                    continue
                
                item_embeddings, item_contexts, last_context = state_rep
                
                state_vec, action_vec = self.actor(
                    item_embeddings.to(self.device),
                    {k: v.to(self.device) for k, v in item_contexts.items()},
                    {k: v.to(self.device) for k, v in last_context.items()}
                )
                
                recommendations = self.recommender.recommend(
                    action_vector=action_vec.squeeze(0).cpu(),
                    context=context,
                    n=10,
                    split_type=split_type,
                    return_all=False
                )
                
                if recommendations:
                    all_recommendations.extend(recommendations)
                    
                    recommended = recommendations[0]
                    recommended_item = {
                        'embedding': recommended['embedding'],
                        'context': context,
                        'reward': recommended['reward'],
                        'track_id': recommended['track_id']
                    }
                    user_history.append(recommended_item)
                else:
                    user_history.append(item)
                
                if pbar:
                    pbar.update(1)
                step += 1
        
        if pbar:
            pbar.close()
        
        metrics = self._compute_metrics_comprehensive(user_history, eval_items, split_type)
        
        for key, value in metrics.items():
            if split_type == SplitType.VALIDATION:
                self.training_history['val_metrics'][key].append(value)
            elif split_type == SplitType.TEST:
                self.training_history['test_metrics'][key].append(value)
        
        self.actor.train()
        self.critic.train()
        
        return metrics

    def _compute_metrics(self, recommendations: List[Dict], 
                    ground_truth: List[Dict],
                    split_type: SplitType) -> Dict[str, float]:
        """Calcula métricas de evaluación corregidas y validadas"""
        
        if not recommendations or not ground_truth:
            return {}
        
        k_values = [1, 5, 10]
        metrics = {}
        max_items = min(100, len(recommendations), len(ground_truth))
        
        # ============================================
        # 1. METADATAS BÁSICAS Y VALIDACIÓN
        # ============================================
        
        # Limitar a max_items para consistencia
        rec_slice = recommendations[:max_items]
        gt_slice = ground_truth[:max_items]
        
        # 1.1 Recompensas
        rec_rewards = [r.get('reward', 0) for r in rec_slice]
        gt_rewards = [r.get('reward', 0) for r in gt_slice]
        
        metrics['avg_reward'] = float(np.mean(rec_rewards)) if rec_rewards else 0.0
        metrics['gt_avg_reward'] = float(np.mean(gt_rewards)) if gt_rewards else 0.0
        metrics['reward_ratio'] = metrics['avg_reward'] / metrics['gt_avg_reward'] if metrics['gt_avg_reward'] > 0 else 0.0
        
        # 1.2 IDs para matching exacto
        rec_track_ids = [r.get('track_id') for r in rec_slice if r.get('track_id') is not None]
        gt_track_ids = [r.get('track_id') for r in gt_slice if r.get('track_id') is not None]
        
        # Conjunto de ítems relevantes (ground truth)
        relevant_set = set(gt_track_ids)
        
        # ============================================
        # 2. PRECISION, RECALL Y MAP - CORREGIDOS
        # ============================================
        
        for k in k_values:
            if len(rec_track_ids) >= k and relevant_set:
                # Tomar primeros k recomendados
                k_recommended = rec_track_ids[:k]
                
                # Contar cuántos están en el ground truth
                relevant_count = sum(1 for item_id in k_recommended if item_id in relevant_set)
                
                # Precision@k
                precision = relevant_count / k
                metrics[f'precision@{k}'] = float(precision)
                
                # Recall@k - normalizado por min(k, |relevant_set|)
                recall_denominator = min(k, len(relevant_set))
                recall = relevant_count / recall_denominator if recall_denominator > 0 else 0.0
                metrics[f'recall@{k}'] = float(recall)
                
                # MAP@k (Mean Average Precision)
                if k >= 2:  # MAP solo tiene sentido para k > 1
                    map_score = self._calculate_map_at_k_corrected(k_recommended, relevant_set, k)
                    metrics[f'map@{k}'] = float(map_score)
            
            elif rec_track_ids and gt_track_ids:
                # Si no hay suficientes recomendaciones pero hay datos
                metrics[f'precision@{k}'] = 0.0
                metrics[f'recall@{k}'] = 0.0
                if k >= 2:
                    metrics[f'map@{k}'] = 0.0
        
        # ============================================
        # 3. NDCG - CORREGIDO CRÍTICAMENTE
        # ============================================
        
        if rec_rewards and gt_rewards:
            for k in k_values:
                k_valid = min(k, len(rec_rewards), len(gt_rewards))
                if k_valid > 0:
                    # Obtener recompensas de las recomendaciones
                    rec_k_rewards = rec_rewards[:k_valid]
                    
                    # Calcular DCG
                    dcg = 0.0
                    for i, reward in enumerate(rec_k_rewards):
                        # i+2 porque i empieza en 0 y log2(1)=0 sería división por 0
                        dcg += reward / math.log2(i + 2)
                    
                    # Calcular IDCG: recompensas IDEALES ordenadas descendentemente
                    # ¡IMPORTANTE! Usar las mejores recompensas posibles, no ground_truth tal cual
                    ideal_rewards = sorted(gt_rewards, reverse=True)[:k_valid]
                    idcg = 0.0
                    for i, reward in enumerate(ideal_rewards):
                        idcg += reward / math.log2(i + 2)
                    
                    # NDCG = DCG / IDCG (DEBE estar entre 0 y 1)
                    if idcg > 0:
                        ndcg = dcg / idcg
                    else:
                        ndcg = 0.0
                    
                    # VALIDACIÓN CRÍTICA
                    if ndcg > 1.0:
                        # ¡ERROR! Forzar a 1.0 y loguear advertencia
                        print(f"⚠️  [ERROR NDCG@{k}] Valor {ndcg:.4f} > 1.0 - Corrigiendo a 1.0")
                        print(f"   DCG: {dcg:.4f}, IDCG: {idcg:.4f}")
                        print(f"   Recompensas rec: {rec_k_rewards}")
                        print(f"   Recompensas ideal: {ideal_rewards}")
                        ndcg = 1.0
                    elif ndcg < 0:
                        ndcg = 0.0
                    
                    metrics[f'ndcg@{k}'] = float(ndcg)
        
        # ============================================
        # 4. MÉTRICAS BASADAS EN SIMILITUD (fallback)
        # ============================================
        
        # Si no hay track_ids o los resultados parecen sospechosos
        if (not rec_track_ids or not gt_track_ids) or \
        (metrics.get('precision@1', 0) == 1.0 and len(relevant_set) > 1):
            
            # Calcular métricas basadas en similitud de embeddings
            similarity_metrics = self._compute_similarity_based_metrics(rec_slice, gt_slice, k_values)
            
            # Mezclar con métricas existentes o reemplazar si son sospechosas
            for key, value in similarity_metrics.items():
                # Solo reemplazar si la métrica actual es sospechosa
                if key.startswith('precision@') and metrics.get(key, 0) == 1.0:
                    metrics[f'sim_{key}'] = value  # Guardar como alternativa
                elif key not in metrics or metrics[key] == 0:
                    metrics[key] = value
        
        # ============================================
        # 5. COBERTURA DE CONTEXTO - MEJORADA
        # ============================================
        
        context_coverage = self._compute_context_coverage_improved(rec_slice)
        metrics.update(context_coverage)
        
        # ============================================
        # 6. MÉTRICAS DE DIVERSIDAD Y NOVELTY
        # ============================================
        
        diversity_metrics = self._compute_diversity_metrics_improved(rec_slice)
        metrics.update(diversity_metrics)
        
        # ============================================
        # 7. MÉTRICAS DE DIAGNÓSTICO
        # ============================================
        
        diagnostic_metrics = self._compute_diagnostic_metrics(rec_slice, gt_slice)
        metrics.update(diagnostic_metrics)
        
        # ============================================
        # 8. VALIDACIÓN FINAL DE MÉTRICAS
        # ============================================
        
        self._validate_metrics(metrics, split_type)
        
        return metrics

    def _calculate_map_at_k_corrected(self, recommended_ids: List[str], 
                                 relevant_set: set, k: int) -> float:
        """Calcula Mean Average Precision@k corregido"""
        
        if not recommended_ids or not relevant_set or k <= 1:
            return 0.0
        
        # Tomar solo k recomendaciones
        recommended_k = recommended_ids[:k]
        
        # Calcular precision en cada posición relevante
        average_precision = 0.0
        num_relevant_found = 0
        
        for i, item_id in enumerate(recommended_k, 1):  # i empieza en 1
            if item_id in relevant_set:
                num_relevant_found += 1
                precision_at_i = num_relevant_found / i
                average_precision += precision_at_i
        
        # MAP = average_precision / número de relevantes en el top-k ideal
        # Usamos min(k, len(relevant_set)) como normalización
        map_score = average_precision / min(k, len(relevant_set))
        
        # Validar rango
        return max(0.0, min(1.0, map_score))

    def _compute_similarity_based_metrics(self, recommendations: List[Dict], 
                                        ground_truth: List[Dict], 
                                        k_values: List[int]) -> Dict[str, float]:
        """Calcula métricas basadas en similitud coseno entre embeddings"""
        
        metrics = {}
        
        try:
            # Extraer embeddings (asegurar que existen)
            rec_embeddings = []
            for r in recommendations:
                if 'embedding' in r and isinstance(r['embedding'], torch.Tensor):
                    rec_embeddings.append(r['embedding'])
            
            gt_embeddings = []
            for g in ground_truth:
                if 'embedding' in g and isinstance(g['embedding'], torch.Tensor):
                    gt_embeddings.append(g['embedding'])
            
            if not rec_embeddings or not gt_embeddings:
                return metrics
            
            # Convertir a tensores
            rec_tensor = torch.stack(rec_embeddings)
            gt_tensor = torch.stack(gt_embeddings)
            
            # Calcular matriz de similitud coseno
            # Normalizar embeddings primero
            rec_normalized = F.normalize(rec_tensor, p=2, dim=1)
            gt_normalized = F.normalize(gt_tensor, p=2, dim=1)
            
            similarity_matrix = torch.mm(rec_normalized, gt_normalized.t())
            
            for k in k_values:
                k_valid = min(k, similarity_matrix.size(0))
                
                # Para cada recomendación, tomar la máxima similitud con cualquier ground truth
                max_similarities, _ = similarity_matrix[:k_valid].max(dim=1)
                
                # Precision basada en umbral de similitud
                threshold = 0.6  # Ajustable
                relevant_count = (max_similarities > threshold).sum().item()
                
                precision = relevant_count / k_valid if k_valid > 0 else 0.0
                metrics[f'sim_precision@{k}'] = float(precision)
                
                # NDCG usando similitudes
                relevance_scores = max_similarities.tolist()
                if relevance_scores:
                    ideal_scores = sorted(relevance_scores, reverse=True)
                    
                    dcg = sum(score / math.log2(i + 2) for i, score in enumerate(relevance_scores))
                    idcg = sum(score / math.log2(i + 2) for i, score in enumerate(ideal_scores[:k_valid]))
                    
                    ndcg = dcg / idcg if idcg > 0 else 0.0
                    metrics[f'sim_ndcg@{k}'] = float(min(1.0, ndcg))
        
        except Exception as e:
            print(f"Error en similitud: {e}")
        
        return metrics

    def _compute_context_coverage_improved(self, recommendations: List[Dict]) -> Dict[str, float]:
        """Cálculo mejorado de cobertura de contexto"""
        
        if not recommendations:
            return {}
        
        contexts = [r.get('context', {}) for r in recommendations]
        
        # Días de la semana (0-6)
        days = [c.get('day_of_week', -1) for c in contexts]
        unique_days = len(set([d for d in days if 0 <= d <= 6]))
        
        # Horas del día (0-23) - analizar distribución
        hours = [c.get('hour_of_day', -1) for c in contexts]
        valid_hours = [h for h in hours if 0 <= h <= 23]
        
        if valid_hours:
            # Cobertura por rangos de 3 horas (8 rangos totales)
            hour_bins = [int(h // 3) for h in valid_hours]
            unique_hour_bins = len(set(hour_bins))
            
            # Entropía horaria (diversidad)
            hour_counts = np.bincount(hour_bins, minlength=8)
            hour_probs = hour_counts / hour_counts.sum()
            hour_entropy = -np.sum(hour_probs * np.log2(hour_probs + 1e-10))
            hour_entropy_normalized = hour_entropy / math.log2(len(hour_probs))
        else:
            unique_hour_bins = 0
            hour_entropy_normalized = 0.0
        
        # Meses (1-12)
        months = [c.get('month', -1) for c in contexts]
        unique_months = len(set([m for m in months if 1 <= m <= 12]))
        
        # Días laborables vs fines de semana
        workdays = [c.get('is_workday', -1) for c in contexts]
        workday_coverage = len(set([w for w in workdays if w in [0, 1]])) / 2.0
        
        return {
            'context_coverage_days': unique_days / 7.0,
            'context_coverage_hours': unique_hour_bins / 8.0,
            'context_coverage_months': unique_months / 12.0,
            'workday_coverage': workday_coverage,
            'temporal_entropy': float(hour_entropy_normalized),
            'context_variety': (unique_days + unique_hour_bins + unique_months) / (7 + 8 + 12)
        }

    def _compute_diversity_metrics_improved(self, recommendations: List[Dict]) -> Dict[str, float]:
        """Métricas mejoradas de diversidad"""
        
        if len(recommendations) < 2:
            return {}
        
        metrics = {}
        
        # 1. Diversidad basada en embeddings
        embeddings = []
        for r in recommendations:
            if 'embedding' in r and isinstance(r['embedding'], torch.Tensor):
                embeddings.append(r['embedding'].unsqueeze(0))
        
        if len(embeddings) >= 2:
            emb_matrix = torch.cat(embeddings, dim=0)
            
            # Distancia promedio entre todos los pares
            pairwise_dist = torch.cdist(emb_matrix, emb_matrix, p=2)
            # Excluir diagonal
            mask = ~torch.eye(pairwise_dist.size(0), dtype=torch.bool)
            avg_distance = pairwise_dist[mask].mean().item() if mask.any() else 0.0
            
            # Similitud coseno promedio
            cos_sim = F.cosine_similarity(emb_matrix.unsqueeze(1), emb_matrix.unsqueeze(0), dim=2)
            avg_cosine_sim = cos_sim[mask].mean().item() if mask.any() else 1.0
            
            metrics['embedding_diversity'] = avg_distance
            metrics['cosine_diversity'] = 1.0 - avg_cosine_sim  # 0 = idénticos, 1 = diversos
        
        # 2. Diversidad de contexto (complemento a cobertura)
        context_metrics = self._compute_context_coverage_improved(recommendations)
        metrics.update({f'div_{k}': v for k, v in context_metrics.items() 
                    if 'coverage' in k or 'entropy' in k or 'variety' in k})
        
        # 3. Novelty (proporción de ítems únicos)
        track_ids = [r.get('track_id') for r in recommendations if r.get('track_id') is not None]
        if track_ids:
            unique_ratio = len(set(track_ids)) / len(track_ids)
            metrics['novelty_ratio'] = unique_ratio
            
            # Popularity bias (asumiendo que track_id numéricos más bajos son más populares)
            try:
                numeric_ids = [int(tid) for tid in track_ids if tid.isdigit()]
                if numeric_ids:
                    avg_id = np.mean(numeric_ids)
                    metrics['popularity_bias'] = avg_id / 10000.0  # Normalizar
            except:
                pass
        
        return metrics

    def _compute_diagnostic_metrics(self, recommendations: List[Dict], 
                                ground_truth: List[Dict]) -> Dict[str, float]:
        """Métricas para diagnóstico de problemas"""
        
        metrics = {}
        
        # 1. Tamaños de las listas
        metrics['rec_length'] = len(recommendations)
        metrics['gt_length'] = len(ground_truth)
        
        # 2. Distribución de recompensas
        rec_rewards = [r.get('reward', 0) for r in recommendations]
        gt_rewards = [g.get('reward', 0) for g in ground_truth]
        
        if rec_rewards:
            metrics['rec_reward_mean'] = float(np.mean(rec_rewards))
            metrics['rec_reward_std'] = float(np.std(rec_rewards))
            metrics['rec_reward_min'] = float(np.min(rec_rewards))
            metrics['rec_reward_max'] = float(np.max(rec_rewards))
        
        if gt_rewards:
            metrics['gt_reward_mean'] = float(np.mean(gt_rewards))
            metrics['gt_reward_std'] = float(np.std(gt_rewards))
        
        # 3. Overlap exacto
        rec_ids = set(r.get('track_id') for r in recommendations if r.get('track_id'))
        gt_ids = set(g.get('track_id') for g in ground_truth if g.get('track_id'))
        
        if rec_ids and gt_ids:
            overlap = len(rec_ids.intersection(gt_ids))
            metrics['exact_overlap'] = overlap / len(rec_ids) if rec_ids else 0.0
            metrics['recall_coverage'] = overlap / len(gt_ids) if gt_ids else 0.0
        
        # 4. Sesgo temporal
        rec_hours = [r.get('context', {}).get('hour_of_day', 12) for r in recommendations]
        if rec_hours:
            hour_mean = np.mean(rec_hours)
            hour_std = np.std(rec_hours)
            metrics['hour_bias'] = abs(hour_mean - 12) / 12.0  # Sesgo respecto al mediodía
            metrics['hour_variance'] = hour_std / 24.0
        
        return metrics

    def _validate_metrics(self, metrics: Dict[str, float], split_type: SplitType):
        """Valida que las métricas estén en rangos razonables"""
        
        validation_issues = []
        
        # Rangos esperados para diferentes métricas
        expected_ranges = {
            'precision@': (0.0, 1.0),
            'recall@': (0.0, 1.0),
            'map@': (0.0, 1.0),
            'ndcg@': (0.0, 1.0),
            'context_coverage_': (0.0, 1.0),
            'avg_reward': (0.0, 1.0),  # Asumiendo recompensas normalizadas
            'gt_avg_reward': (0.0, 1.0),
        }
        
        for metric_name, value in metrics.items():
            # Verificar NaN o infinito
            if np.isnan(value) or np.isinf(value):
                validation_issues.append(f"{metric_name}: {value} (NaN/Inf)")
                metrics[metric_name] = 0.0  # Corregir
            
            # Verificar rangos específicos
            for prefix, (min_val, max_val) in expected_ranges.items():
                if metric_name.startswith(prefix):
                    if value < min_val or value > max_val:
                        validation_issues.append(f"{metric_name}: {value} fuera de rango [{min_val}, {max_val}]")
                        # Corregir valores fuera de rango
                        metrics[metric_name] = max(min_val, min(max_val, value))
                    break
        
        # Verificar consistencia entre métricas relacionadas
        if 'precision@1' in metrics and metrics['precision@1'] == 1.0:
            if 'recall@1' in metrics and metrics['recall@1'] != 1.0:
                validation_issues.append("Inconsistencia: precision@1=1.0 pero recall@1!=1.0")
        
        # Log issues si hay muchas
        if validation_issues and split_type == SplitType.TRAIN:
            if len(validation_issues) > 3:  # Solo loggear si hay varios problemas
                print(f"[VALIDACIÓN {split_type.value}] Issues encontradas:")
                for issue in validation_issues[:5]:  # Mostrar solo primeros 5
                    print(f"  - {issue}")
        
        return len(validation_issues) == 0

    def _compute_metrics_comprehensive(self, recommendations: List[Dict], 
                                 ground_truth: List[Dict],
                                 split_type: SplitType) -> Dict[str, float]:
        """Wrapper que incluye limpieza de datos antes de calcular métricas"""
        
        # 1. Limpiar y validar datos de entrada
        clean_recommendations = []
        for rec in recommendations:
            if rec and isinstance(rec, dict):
                # Asegurar que tiene los campos mínimos
                if 'embedding' not in rec or not isinstance(rec['embedding'], torch.Tensor):
                    continue
                clean_recommendations.append(rec)
        
        clean_ground_truth = []
        for gt in ground_truth:
            if gt and isinstance(gt, dict):
                clean_ground_truth.append(gt)
        
        # 2. Calcular métricas principales
        metrics = self._compute_metrics(clean_recommendations, clean_ground_truth, split_type)
        
        # 3. Añadir estadísticas básicas
        metrics['num_recommendations'] = len(clean_recommendations)
        metrics['num_ground_truth'] = len(clean_ground_truth)
        metrics['coverage_ratio'] = len(clean_recommendations) / len(clean_ground_truth) if clean_ground_truth else 0.0
        
        return metrics
    
    def _compute_context_coverage(self, recommendations: List[Dict]) -> Dict[str, float]:
        if not recommendations:
            return {}
        
        contexts = [r.get('context', {}) for r in recommendations]
        
        unique_days = len(set(c.get('day_of_week', -1) for c in contexts))
        unique_hours = len(set(c.get('hour_of_day', -1) for c in contexts))
        unique_months = len(set(c.get('month', -1) for c in contexts))
        
        return {
            'context_coverage_days': unique_days / 7.0,
            'context_coverage_hours': unique_hours / 24.0,
            'context_coverage_months': unique_months / 12.0
        }

    def _calculate_map_at_k(self, recommended_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """Calcula Mean Average Precision@k"""
        if not recommended_ids or not relevant_ids or k <= 1:
            return 0.0
        
        # Tomar solo k recomendaciones
        recommended_k = recommended_ids[:k]
        
        # Convertir a sets para búsqueda rápida
        relevant_set = set(relevant_ids[:k])  # Consideramos relevantes los primeros k de ground truth
        
        # Calcular precision en cada posición donde hay un ítem relevante
        average_precision = 0.0
        num_relevant = 0
        
        for i, item_id in enumerate(recommended_k):
            if item_id in relevant_set:
                num_relevant += 1
                # Precision en esta posición: relevantes encontrados hasta ahora / (posición + 1)
                precision_at_i = num_relevant / (i + 1)
                average_precision += precision_at_i
        
        # MAP = suma de precisiones / número de relevantes en ground truth
        num_relevant_in_gt = len(relevant_set)
        map_score = average_precision / num_relevant_in_gt if num_relevant_in_gt > 0 else 0.0
        
        return map_score

    def _compute_diversity_metrics(self, recommendations: List[Dict]) -> Dict[str, float]:
        """Calcula métricas de diversidad"""
        if not recommendations:
            return {}
        
        # Diversidad basada en contexto
        contexts = [r.get('context', {}) for r in recommendations]
        
        # Días únicos
        days = [c.get('day_of_week', -1) for c in contexts]
        unique_days = len(set(days))
        
        # Horas únicas (agrupadas por rangos de 3 horas)
        hours = [c.get('hour_of_day', -1) for c in contexts]
        hour_bins = [int(h // 3) for h in hours]  # 8 bins de 3 horas
        unique_hour_bins = len(set(hour_bins))
        
        # Meses únicos
        months = [c.get('month', -1) for c in contexts]
        unique_months = len(set(months))
        
        # Entropía temporal (medida de diversidad)
        if len(days) > 0:
            day_counts = np.bincount([d for d in days if d >= 0])
            day_probs = day_counts / day_counts.sum()
            day_entropy = -np.sum(day_probs * np.log2(day_probs + 1e-10))
            day_entropy_normalized = day_entropy / math.log2(len(day_probs) + 1e-10)
        else:
            day_entropy_normalized = 0.0
        
        return {
            'context_coverage_days': unique_days / 7.0,
            'context_coverage_hours': unique_hour_bins / 8.0,  # 8 bins de 3 horas
            'context_coverage_months': unique_months / 12.0,
            'temporal_entropy': float(day_entropy_normalized)
        }

    def _compute_item_diversity(self, recommendations: List[Dict]) -> Dict[str, float]:
        """Calcula diversidad de ítems (si tienes información de géneros/artistas)"""
        if not recommendations:
            return {}
        
        # Si tienes información de géneros o artistas
        genres = []
        artists = []
        
        for rec in recommendations:
            if 'genre' in rec:
                genres.append(rec['genre'])
            if 'artist_id' in rec:
                artists.append(rec['artist_id'])
        
        metrics = {}
        
        if genres:
            unique_genres = len(set(genres))
            genre_counts = np.bincount([hash(g) % 100 for g in genres])  # Ejemplo simplificado
            genre_probs = genre_counts / genre_counts.sum()
            genre_entropy = -np.sum(genre_probs * np.log2(genre_probs + 1e-10))
            
            metrics.update({
                'unique_genres': unique_genres,
                'genre_entropy': float(genre_entropy),
                'genre_coverage': unique_genres / len(set(genres)) if genres else 0.0
            })
        
        if artists:
            unique_artists = len(set(artists))
            artist_counts = np.bincount([hash(a) % 100 for a in artists])
            artist_probs = artist_counts / artist_counts.sum()
            artist_entropy = -np.sum(artist_probs * np.log2(artist_probs + 1e-10))
            
            metrics.update({
                'unique_artists': unique_artists,
                'artist_entropy': float(artist_entropy),
                'artist_coverage': unique_artists / len(set(artists)) if artists else 0.0
            })
        
        # Novelty: proporción de ítems no vistos antes
        # (necesitarías mantener un historial de ítems ya recomendados)
        
        return metrics
    
    def train(self, num_epochs: int, epsilon_start: float = 0.9, 
             epsilon_end: float = 0.1, epsilon_decay: float = 0.995,
             eval_freq: int = 1, save_path: Optional[str] = None, print_logs: bool = True) -> Dict:
        print(f"Starting training for {num_epochs} epochs in {self.device}...")
        epsilon = epsilon_start
        
        if not print_logs:
            pbar_t = tqdm(total=num_epochs, desc="Training Epochs")
        else:
            pbar_t = None
            
        for epoch in range(num_epochs):
            if print_logs:
                print(f"\n{'='*50}")
                print(f"Epoch {epoch+1}/{num_epochs}")
                print(f"{'='*50}")
            
            train_metrics = self.train_epoch(epsilon=epsilon, print_logs=print_logs)

            if print_logs:
                print(f"Train Metrics:")
                for key, value in train_metrics.items():
                    print(f"  {key}: {value:.4f}")
            
            if (epoch + 1) % eval_freq == 0:
                val_metrics = self.evaluate(SplitType.VALIDATION)
                if print_logs:
                    print(f"\nValidation Metrics:")
                    for key, value in val_metrics.items():
                        print(f"  {key}: {value:.4f}")
                
                test_metrics = self.evaluate(SplitType.TEST)
                if print_logs:
                    print(f"\nTest Metrics:")
                    for key, value in test_metrics.items():
                        print(f"  {key}: {value:.4f}")
            
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            
            if save_path and (epoch + 1) % 10 == 0:
                self.save_model(f"{save_path}_epoch_{epoch+1}.pt")
            
            if pbar_t:
                pbar_t.update(1)

        if pbar_t:
            pbar_t.close()

        if save_path:
            self.save_model(f"{save_path}_final.pt")
            self.save_training_history(f"{save_path}_training_history.json")

        return self.training_history

    def save_model(self, path: str):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'training_history': self.training_history,
            'step_count': self.step_count
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.training_history = checkpoint['training_history']
        self.step_count = checkpoint['step_count']
        print(f"Model loaded from {path}")

    def get_training_history(self) -> Dict:
        return self.training_history

    def save_training_history(self, path: str):
        import json
        with open(path, 'w') as f:
            json.dump(self.training_history, f, indent=4)
        print(f"Training history saved to {path}")

    def plot_training_history(self):
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            if self.training_history['actor_loss']:
                axes[0, 0].plot(self.training_history['actor_loss'])
                axes[0, 0].set_title('Actor Loss')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Loss')
            
            if self.training_history['critic_loss']:
                axes[0, 1].plot(self.training_history['critic_loss'])
                axes[0, 1].set_title('Critic Loss')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('Loss')
            
            if self.training_history['train_rewards']:
                axes[0, 2].plot(self.training_history['train_rewards'])
                axes[0, 2].set_title('Training Rewards')
                axes[0, 2].set_xlabel('Epoch')
                axes[0, 2].set_ylabel('Avg Reward')
            
            train_keys = ['precision@5', 'recall@5', 'ndcg@5']
            for i, key in enumerate(train_keys):
                if key in self.training_history['train_metrics']:
                    axes[1, 0].plot(self.training_history['train_metrics'][key], label=f'Train {key}')
            axes[1, 0].set_title('Training Metrics')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].legend()
            
            for i, key in enumerate(train_keys):
                if key in self.training_history['val_metrics']:
                    axes[1, 1].plot(self.training_history['val_metrics'][key], label=f'Val {key}')
            axes[1, 1].set_title('Validation Metrics')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].legend()
            
            for i, key in enumerate(train_keys):
                if key in self.training_history['test_metrics']:
                    axes[1, 2].plot(self.training_history['test_metrics'][key], label=f'Test {key}')
            axes[1, 2].set_title('Test Metrics')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Score')
            axes[1, 2].legend()
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not installed. Skipping plots.")
    
    def plot_individual_metrics(self, save_dir: str = None, figsize: tuple = (10, 6)):
        """
        Crea gráficas individuales para cada métrica, comparando train/val/test.
        
        Args:
            save_dir: Directorio donde guardar las gráficas (si None, solo muestra)
            figsize: Tamaño de cada figura individual
        """
        try:
            # Crear directorio si se especifica
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            # Configurar estilo
            plt.style.use('seaborn-v0_8-darkgrid')
            
            # Definir colores y estilos
            split_config = {
                'train': {'color': '#1f77b4', 'linestyle': '-', 'marker': 'o', 'label': 'Train'},
                'val': {'color': '#ff7f0e', 'linestyle': '--', 'marker': 's', 'label': 'Validation'},
                'test': {'color': '#2ca02c', 'linestyle': ':', 'marker': '^', 'label': 'Test'}
            }
            
            # 1. GRÁFICA DE RECOMPENSAS
            self._plot_single_metric(
                metric_name='Recompensas',
                train_data=self.training_history.get('train_rewards', []),
                val_data=self.training_history.get('val_metrics', {}).get('avg_reward', []),
                test_data=self.training_history.get('test_metrics', {}).get('avg_reward', []),
                split_config=split_config,
                ylabel='Recompensa Promedio',
                ylim=(0, 1.1),
                figsize=figsize,
                save_path=os.path.join(save_dir, 'recompensas.png') if save_dir else None
            )
            
            # 2. GRÁFICA DE PÉRDIDAS
            self._plot_losses(
                actor_loss=self.training_history.get('actor_loss', []),
                critic_loss=self.training_history.get('critic_loss', []),
                figsize=figsize,
                save_path=os.path.join(save_dir, 'perdidas.png') if save_dir else None
            )
            
            # 3. GRÁFICAS DE PRECISIÓN @k
            for k in [1, 5, 10]:
                self._plot_single_metric(
                    metric_name=f'Precisión@{k}',
                    train_data=self.training_history.get('train_metrics', {}).get(f'precision@{k}', []),
                    val_data=self.training_history.get('val_metrics', {}).get(f'precision@{k}', []),
                    test_data=self.training_history.get('test_metrics', {}).get(f'precision@{k}', []),
                    split_config=split_config,
                    ylabel=f'Precisión@{k}',
                    ylim=(0, 1.1),
                    figsize=figsize,
                    save_path=os.path.join(save_dir, f'precision@{k}.png') if save_dir else None
                )
            
            # 4. GRÁFICAS DE RECALL @k
            for k in [5, 10]:
                self._plot_single_metric(
                    metric_name=f'Recall@{k}',
                    train_data=self.training_history.get('train_metrics', {}).get(f'recall@{k}', []),
                    val_data=self.training_history.get('val_metrics', {}).get(f'recall@{k}', []),
                    test_data=self.training_history.get('test_metrics', {}).get(f'recall@{k}', []),
                    split_config=split_config,
                    ylabel=f'Recall@{k}',
                    ylim=(0, 1.1),
                    figsize=figsize,
                    save_path=os.path.join(save_dir, f'recall@{k}.png') if save_dir else None
                )
            
            # 5. GRÁFICAS DE MAP @k
            for k in [5, 10]:
                self._plot_single_metric(
                    metric_name=f'MAP@{k}',
                    train_data=self.training_history.get('train_metrics', {}).get(f'map@{k}', []),
                    val_data=self.training_history.get('val_metrics', {}).get(f'map@{k}', []),
                    test_data=self.training_history.get('test_metrics', {}).get(f'map@{k}', []),
                    split_config=split_config,
                    ylabel=f'MAP@{k}',
                    ylim=(0, 1.1),
                    figsize=figsize,
                    save_path=os.path.join(save_dir, f'map@{k}.png') if save_dir else None
                )
            
            # 6. GRÁFICAS DE NDCG @k
            for k in [1, 5, 10]:
                self._plot_single_metric(
                    metric_name=f'NDCG@{k}',
                    train_data=self.training_history.get('train_metrics', {}).get(f'ndcg@{k}', []),
                    val_data=self.training_history.get('val_metrics', {}).get(f'ndcg@{k}', []),
                    test_data=self.training_history.get('test_metrics', {}).get(f'ndcg@{k}', []),
                    split_config=split_config,
                    ylabel=f'NDCG@{k}',
                    ylim=(0, 1.1),
                    figsize=figsize,
                    save_path=os.path.join(save_dir, f'ndcg@{k}.png') if save_dir else None
                )
            
            # 7. GRÁFICAS DE COBERTURA DE CONTEXTO
            context_metrics = ['context_coverage_days', 'context_coverage_hours', 'context_coverage_months']
            context_labels = ['Cobertura de Días', 'Cobertura de Horas', 'Cobertura de Meses']
            
            for metric, label in zip(context_metrics, context_labels):
                self._plot_single_metric(
                    metric_name=label,
                    train_data=self.training_history.get('train_metrics', {}).get(metric, []),
                    val_data=self.training_history.get('val_metrics', {}).get(metric, []),
                    test_data=self.training_history.get('test_metrics', {}).get(metric, []),
                    split_config=split_config,
                    ylabel=label,
                    ylim=(0, 1.1),
                    figsize=figsize,
                    save_path=os.path.join(save_dir, f'{metric}.png') if save_dir else None
                )
            
            # 8. GRÁFICA DE ENTROPÍA TEMPORAL (si existe)
            if 'temporal_entropy' in self.training_history.get('train_metrics', {}):
                self._plot_single_metric(
                    metric_name='Entropía Temporal',
                    train_data=self.training_history['train_metrics'].get('temporal_entropy', []),
                    val_data=self.training_history['val_metrics'].get('temporal_entropy', []),
                    test_data=self.training_history['test_metrics'].get('temporal_entropy', []),
                    split_config=split_config,
                    ylabel='Entropía Temporal',
                    ylim=(0, 1.1),
                    figsize=figsize,
                    save_path=os.path.join(save_dir, 'entropia_temporal.png') if save_dir else None
                )
            
            print(f"✅ Gráficas individuales creadas{' y guardadas en ' + save_dir if save_dir else ''}")
            
        except Exception as e:
            print(f"Error al crear gráficas individuales: {e}")
            import traceback
            traceback.print_exc()
    
    def _plot_single_metric(self, metric_name: str, train_data: List[float], 
                           val_data: List[float], test_data: List[float],
                           split_config: Dict, ylabel: str, ylim: tuple,
                           figsize: tuple, save_path: Optional[str] = None):
        """
        Crea una gráfica individual para una métrica específica.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot train data
        if train_data:
            x_train = range(1, len(train_data) + 1)
            ax.plot(x_train, train_data, 
                   color=split_config['train']['color'],
                   linestyle=split_config['train']['linestyle'],
                   marker=split_config['train']['marker'],
                   markersize=6,
                   linewidth=2,
                   label=split_config['train']['label'])
        
        # Plot validation data (puede tener menos puntos)
        if val_data:
            eval_freq = max(1, len(train_data) // max(1, len(val_data)))
            x_val = [(i + 1) * eval_freq for i in range(len(val_data))]
            ax.plot(x_val, val_data,
                   color=split_config['val']['color'],
                   linestyle=split_config['val']['linestyle'],
                   marker=split_config['val']['marker'],
                   markersize=6,
                   linewidth=2,
                   label=split_config['val']['label'])
        
        # Plot test data (puede tener menos puntos)
        if test_data:
            eval_freq = max(1, len(train_data) // max(1, len(test_data)))
            x_test = [(i + 1) * eval_freq for i in range(len(test_data))]
            ax.plot(x_test, test_data,
                   color=split_config['test']['color'],
                   linestyle=split_config['test']['linestyle'],
                   marker=split_config['test']['marker'],
                   markersize=6,
                   linewidth=2,
                   label=split_config['test']['label'])
        
        # Configurar gráfica
        ax.set_title(f'{metric_name} - Evolución', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Época', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_ylim(ylim)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        # Añadir línea horizontal en y=0.5 para referencia
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        # Añadir línea horizontal en y=0.7 para buena performance
        ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.3, linewidth=1)
        
        # Añadir línea horizontal en y=0.3 para mala performance
        ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.3, linewidth=1)
        
        # Añadir estadísticas en el gráfico
        if train_data:
            final_value = train_data[-1]
            ax.text(0.02, 0.98, f'Final: {final_value:.3f}',
                   transform=ax.transAxes,
                   fontsize=10,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def _plot_losses(self, actor_loss: List[float], critic_loss: List[float],
                    figsize: tuple, save_path: Optional[str] = None):
        """Gráfica especial para las pérdidas del actor y crítico."""
        
        # Crear figura con dos subgráficas
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Colores y estilos consistentes
        colors = {
            'actor': '#d62728',  # Rojo
            'critic': '#9467bd'  # Morado
        }
        
        # Gráfica del Actor
        ax1 = axes[0]
        if actor_loss:
            x = range(1, len(actor_loss) + 1)
            ax1.plot(x, actor_loss,
                    color=colors['actor'],
                    linestyle='-',
                    marker='o',
                    markersize=5,
                    linewidth=2,
                    label='Pérdida Actor')
            
            # Calcular estadísticas del Actor
            initial_actor = actor_loss[0]
            final_actor = actor_loss[-1]
            improvement_actor = ((initial_actor - final_actor) / abs(initial_actor)) * 100
            
            # Texto de estadísticas
            stats_text = f'Inicial: {initial_actor:.4f}\n'
            stats_text += f'Final: {final_actor:.4f}\n'
            stats_text += f'Mejora: {improvement_actor:+.1f}%'
            
            ax1.text(0.02, 0.98, stats_text,
                    transform=ax1.transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax1.set_title('Pérdida del Actor', fontsize=14, fontweight='bold', pad=10)
        ax1.set_xlabel('Época', fontsize=11)
        ax1.set_ylabel('Valor de Pérdida', fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', fontsize=10)
        ax1.tick_params(axis='both', which='major', labelsize=9)
        
        # Escala logarítmica si es necesario
        if actor_loss and len(actor_loss) > 1:
            min_val = min(actor_loss)
            max_val = max(actor_loss)
            if min_val > 0 and max_val / min_val > 100:
                ax1.set_yscale('log')
                ax1.set_ylabel('Valor de Pérdida (log)', fontsize=11)
        
        # Gráfica del Crítico
        ax2 = axes[1]
        if critic_loss:
            x = range(1, len(critic_loss) + 1)
            ax2.plot(x, critic_loss,
                    color=colors['critic'],
                    linestyle='--',
                    marker='s',
                    markersize=5,
                    linewidth=2,
                    label='Pérdida Crítico')
            
            # Calcular estadísticas del Crítico
            initial_critic = critic_loss[0]
            final_critic = critic_loss[-1]
            improvement_critic = ((initial_critic - final_critic) / abs(initial_critic)) * 100
            
            # Texto de estadísticas
            stats_text = f'Inicial: {initial_critic:.4f}\n'
            stats_text += f'Final: {final_critic:.4f}\n'
            stats_text += f'Mejora: {improvement_critic:+.1f}%'
            
            ax2.text(0.02, 0.98, stats_text,
                    transform=ax2.transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax2.set_title('Pérdida del Crítico', fontsize=14, fontweight='bold', pad=10)
        ax2.set_xlabel('Época', fontsize=11)
        ax2.set_ylabel('Valor de Pérdida', fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best', fontsize=10)
        ax2.tick_params(axis='both', which='major', labelsize=9)
        
        # Escala logarítmica si es necesario
        if critic_loss and len(critic_loss) > 1:
            min_val = min(critic_loss)
            max_val = max(critic_loss)
            if min_val > 0 and max_val / min_val > 100:
                ax2.set_yscale('log')
                ax2.set_ylabel('Valor de Pérdida (log)', fontsize=11)
        
        plt.suptitle('Evolución de Pérdidas - Actor y Crítico', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            # Asegurar que la ruta tenga extensión .png si no la tiene
            if not save_path.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf', '.svg')):
                save_path += '.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_metric_comparison_grid(self, save_path: str = None):
        """
        Crea un grid de comparación de métricas finales entre train/val/test.
        """
        try:
            # Obtener métricas finales
            df_summary = self.generate_metrics_summary_table()
            
            if df_summary.empty:
                print("No hay métricas para crear el grid de comparación.")
                return
            
            # Seleccionar métricas principales
            main_metrics = ['avg_reward', 'precision@5', 'recall@5', 
                           'map@5', 'ndcg@5', 'context_coverage_days']
            
            # Filtrar métricas disponibles
            available_metrics = [m for m in main_metrics if m in df_summary.columns]
            
            if not available_metrics:
                available_metrics = [col for col in df_summary.columns if col != 'Split']
            
            n_metrics = len(available_metrics)
            n_cols = min(3, n_metrics)
            n_rows = (n_metrics + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            axes = axes.flatten() if n_metrics > 1 else [axes]
            
            # Colores para cada split
            colors = {'Training': '#1f77b4', 'Validation': '#ff7f0e', 'Test': '#2ca02c'}
            
            for idx, metric in enumerate(available_metrics):
                ax = axes[idx]
                
                # Obtener valores para cada split
                values = []
                labels = []
                bar_colors = []
                
                for split in ['Training', 'Validation', 'Test']:
                    if split in df_summary['Split'].values:
                        value = df_summary[df_summary['Split'] == split][metric].values[0]
                        values.append(value)
                        labels.append(split)
                        bar_colors.append(colors[split])
                
                # Crear barras
                bars = ax.bar(labels, values, color=bar_colors, edgecolor='black', linewidth=1.5)
                
                # Añadir valores encima de las barras
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=10)
                
                # Configurar gráfica
                metric_name_display = metric.replace('_', ' ').replace('@', '@').title()
                ax.set_title(f'{metric_name_display}', fontsize=12, fontweight='bold')
                ax.set_ylabel('Valor', fontsize=10)
                ax.set_ylim([0, 1.1])
                ax.grid(True, alpha=0.3, axis='y')
                ax.tick_params(axis='x', rotation=45)
                
                # Añadir líneas de referencia
                ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
                ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.3)
                ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.3)
            
            # Ocultar ejes no utilizados
            for idx in range(len(available_metrics), len(axes)):
                axes[idx].set_visible(False)
            
            plt.suptitle('Comparación Final de Métricas por Split', 
                        fontsize=16, fontweight='bold', y=1.02)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Grid de comparación guardado en: {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"Error al crear grid de comparación: {e}")
            import traceback
            traceback.print_exc()
    
    def _get_train_val_gap(self, df_summary: pd.DataFrame) -> float:
        """Calcula el gap entre training y validation."""
        if 'Training' not in df_summary['Split'].values or 'Validation' not in df_summary['Split'].values:
            return 0.0
        
        train_row = df_summary[df_summary['Split'] == 'Training'].iloc[0]
        val_row = df_summary[df_summary['Split'] == 'Validation'].iloc[0]
        
        if 'avg_reward' in df_summary.columns:
            return abs(train_row['avg_reward'] - val_row['avg_reward'])
        return 0.0
    