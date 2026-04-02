import torch
import numpy as np
from typing import List, Dict, Any, Optional
from .client import Client, SplitType

class Recommender:
    def __init__(self, client: Client):
        """
        Inicializa el recomendador con un cliente.
        
        Args:
            client: Instancia de Client que contiene los datos
        """
        self.client = client
        self.cache = {}  # Cache para resultados de búsqueda por contexto
        
    def _compute_dot_product(self, 
                           action_vector: torch.Tensor, 
                           items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Calcula el producto punto entre el vector de acción y los embeddings de los ítems.
        
        Args:
            action_vector: Tensor de tamaño [1, 512] o [512]
            items: Lista de ítems con embedding
        
        Returns:
            Lista de ítems con score añadido
        """
        # Asegurar que el vector de acción tenga la forma correcta
        if action_vector.dim() == 1:
            action_vector = action_vector.unsqueeze(0)  # [512] -> [1, 512]
        
        # Crear batch de embeddings
        embeddings_list = []
        valid_indices = []
        
        for i, item in enumerate(items):
            if 'embedding' in item and item['embedding'] is not None:
                embedding = item['embedding']
                # Asegurar que el embedding tenga la forma correcta
                if embedding.dim() == 1:
                    embedding = embedding.unsqueeze(0)  # [512] -> [1, 512]
                embeddings_list.append(embedding)
                valid_indices.append(i)
        
        if not embeddings_list:
            return []
        
        # Apilar embeddings
        embeddings_tensor = torch.cat(embeddings_list, dim=0)  # [n, 512]
        
        # Calcular producto punto
        # action_vector: [1, 512], embeddings_tensor: [n, 512]
        # Transponer embeddings_tensor para multiplicación: [512, n]
        scores = torch.mm(action_vector, embeddings_tensor.t()).squeeze(0)  # [n]
        
        # Añadir scores a los ítems válidos
        result_items = []
        for idx, score in zip(valid_indices, scores):
            item_copy = items[idx].copy()
            item_copy['score'] = float(score)
            result_items.append(item_copy)
        
        return result_items
    
    def _get_context_key(self, context: Dict[str, Any]) -> str:
        """
        Genera una clave única para el contexto para usar en cache.
        
        Args:
            context: Diccionario de contexto
        
        Returns:
            Clave string
        """
        return f"{context.get('day_of_week', '')}_{context.get('hour_of_day', '')}_" \
               f"{context.get('is_workday', '')}_{context.get('month', '')}"
    
    def get_context_items(self, 
                         context: Dict[str, Any],
                         split_type: SplitType = SplitType.TRAIN,
                         use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Obtiene ítems filtrados por contexto.
        
        Args:
            context: Diccionario con al menos uno de los siguientes:
                    - day_of_week: int (0-6)
                    - hour_of_day: int (0-23)
                    - is_workday: int (0 o 1)
                    - month: int (1-12)
            split_type: Tipo de split a usar
            use_cache: Si es True, usa cache para resultados
        
        Returns:
            Lista de ítems filtrados por contexto
        """
        # Generar clave de cache
        cache_key = self._get_context_key(context) + f"_{split_type.value}"
        
        # Verificar cache
        if use_cache and cache_key in self.cache:
            return self.cache[cache_key]
        
        # Obtener ítems por contexto
        items = self.client.get_items_by_context_dict(context, split_type)
        
        # Guardar en cache si está habilitado
        if use_cache:
            self.cache[cache_key] = items
        
        return items
    
    def recommend(self,
                 action_vector: torch.Tensor,
                 context: Dict[str, Any],
                 n: int = 10,
                 split_type: SplitType = SplitType.TRAIN,
                 return_all: bool = False,
                 min_score: Optional[float] = None,
                 use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Recomienda ítems basados en un vector de acción y contexto.
        
        Args:
            action_vector: Tensor de tamaño [1, 512] o [512]
            context: Diccionario con al menos uno de:
                    - day_of_week: int (0-6)
                    - hour_of_day: int (0-23)
                    - is_workday: int (0 o 1)
                    - month: int (1-12)
            n: Número de recomendaciones a retornar
            split_type: Tipo de split a usar para buscar ítems
            return_all: Si es True, retorna todos los ítems ordenados en lugar de solo n
            min_score: Score mínimo para incluir en recomendaciones (opcional)
            use_cache: Si es True, usa cache para búsquedas por contexto
        
        Returns:
            Lista de recomendaciones con formato:
            {
                'embedding': torch.Tensor,
                'score': float,
                'reward': float,
                'track_id': int,
                'context': dict,
                'original_data': dict (opcional)
            }
        """
        # 1. Buscar ítems por contexto
        context_items = self.get_context_items(context, split_type, use_cache)
        
        if not context_items:
            print(f"No se encontraron ítems para el contexto: {context}")
            return []
        
        # 2. Calcular scores (producto punto)
        scored_items = self._compute_dot_product(action_vector, context_items)
        
        if not scored_items:
            print("No se pudieron calcular scores para los ítems")
            return []
        
        # 3. Filtrar por score mínimo si se especifica
        if min_score is not None:
            scored_items = [item for item in scored_items if item['score'] >= min_score]
        
        # 4. Ordenar por score descendente
        scored_items.sort(key=lambda x: x['score'], reverse=True)
        
        # 5. Retornar resultados
        result_items = []
        
        # Determinar cuántos ítems retornar
        n_items = len(scored_items) if return_all else min(n, len(scored_items))
        
        for i in range(n_items):
            item = scored_items[i]
            
            # Crear resultado con la información requerida
            result = {
                'embedding': item.get('embedding'),
                'score': item.get('score', 0.0),
                'reward': item.get('reward', 0.0),
                'track_id': item.get('track_id'),
                'context': item.get('context', {}),
                'original_data': item.get('original_row', {}) if 'original_row' in item else None
            }
            result_items.append(result)
        
        return result_items
    
    def batch_recommend(self,
                       action_vectors: torch.Tensor,
                       contexts: List[Dict[str, Any]],
                       n: int = 10,
                       split_type: SplitType = SplitType.TRAIN,
                       verbose: bool = False) -> List[List[Dict[str, Any]]]:
        """
        Realiza recomendaciones por lotes.
        
        Args:
            action_vectors: Tensor de tamaño [batch_size, 512]
            contexts: Lista de diccionarios de contexto (uno por acción)
            n: Número de recomendaciones por acción
            split_type: Tipo de split a usar
            verbose: Si es True, muestra progreso
        
        Returns:
            Lista de listas de recomendaciones (una por acción)
        """
        batch_size = action_vectors.size(0)
        if len(contexts) != batch_size:
            raise ValueError(f"Número de contexts ({len(contexts)}) debe coincidir con batch_size ({batch_size})")
        
        all_recommendations = []
        
        for i in range(batch_size):
            if verbose and i % 10 == 0:
                print(f"Procesando recomendación {i+1}/{batch_size}")
            
            recommendations = self.recommend(
                action_vector=action_vectors[i],
                context=contexts[i],
                n=n,
                split_type=split_type,
                use_cache=True
            )
            all_recommendations.append(recommendations)
        
        return all_recommendations
    
    def evaluate_recommendations(self,
                                recommendations: List[Dict[str, Any]],
                                k: int = 10) -> Dict[str, float]:
        """
        Evalúa un conjunto de recomendaciones.
        
        Args:
            recommendations: Lista de recomendaciones
            k: Número de recomendaciones a considerar para métricas
        
        Returns:
            Diccionario con métricas de evaluación
        """
        if not recommendations:
            return {
                'avg_reward': 0.0,
                'avg_score': 0.0,
                'precision_at_k': 0.0,
                'recall_at_k': 0.0,
                'num_recommendations': 0
            }
        
        # Considerar solo las primeras k recomendaciones
        k_recommendations = recommendations[:k]
        
        # Métricas básicas
        rewards = [item['reward'] for item in k_recommendations]
        scores = [item['score'] for item in k_recommendations]
        
        # Calcular precisión y recall (simplificado)
        # Aquí podrías añadir lógica más compleja si tienes ground truth
        
        avg_reward = float(np.mean(rewards)) if rewards else 0.0
        avg_score = float(np.mean(scores)) if scores else 0.0
        
        # Si tienes datos de relevancia binaria, podrías calcular:
        # precision_at_k = sum(relevance[:k]) / k
        # Para este ejemplo, usamos recompensa normalizada como proxy de relevancia
        
        if rewards and max(rewards) > 0:
            normalized_rewards = [r / max(rewards) for r in rewards]
            precision_at_k = float(np.mean(normalized_rewards))
        else:
            precision_at_k = 0.0
        
        return {
            'avg_reward': avg_reward,
            'avg_score': avg_score,
            'precision_at_k': precision_at_k,
            'recall_at_k': precision_at_k,  # En este caso simplificado, es igual
            'num_recommendations': len(recommendations),
            'k_used': min(k, len(recommendations))
        }
    
    def find_similar_items(self,
                          item_embedding: torch.Tensor,
                          context: Optional[Dict[str, Any]] = None,
                          n: int = 10,
                          split_type: SplitType = SplitType.TRAIN) -> List[Dict[str, Any]]:
        """
        Encuentra ítems similares a un embedding dado.
        
        Args:
            item_embedding: Tensor de embedding [512] o [1, 512]
            context: Contexto opcional para filtrar
            n: Número de ítems similares a retornar
            split_type: Tipo de split a usar
        
        Returns:
            Lista de ítems similares ordenados por similitud
        """
        # Usar el mismo embedding como vector de acción
        return self.recommend(
            action_vector=item_embedding,
            context=context or {},
            n=n,
            split_type=split_type,
            return_all=False
        )
    
    def get_recommendation_statistics(self,
                                     recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Obtiene estadísticas de un conjunto de recomendaciones.
        
        Args:
            recommendations: Lista de recomendaciones
        
        Returns:
            Diccionario con estadísticas
        """
        if not recommendations:
            return {'empty': True}
        
        rewards = [item['reward'] for item in recommendations]
        scores = [item['score'] for item in recommendations]
        track_ids = [item['track_id'] for item in recommendations if 'track_id' in item]
        
        # Distribución de contexto en recomendaciones
        contexts = [item['context'] for item in recommendations]
        
        stats = {
            'count': len(recommendations),
            'unique_tracks': len(set(track_ids)),
            'reward_stats': {
                'mean': float(np.mean(rewards)) if rewards else 0.0,
                'std': float(np.std(rewards)) if rewards else 0.0,
                'min': float(np.min(rewards)) if rewards else 0.0,
                'max': float(np.max(rewards)) if rewards else 0.0,
                'median': float(np.median(rewards)) if rewards else 0.0
            },
            'score_stats': {
                'mean': float(np.mean(scores)) if scores else 0.0,
                'std': float(np.std(scores)) if scores else 0.0,
                'min': float(np.min(scores)) if scores else 0.0,
                'max': float(np.max(scores)) if scores else 0.0,
                'median': float(np.median(scores)) if scores else 0.0
            },
            'context_distribution': {
                'day_of_week': {},
                'hour_of_day': {},
                'is_workday': {},
                'month': {}
            }
        }
        
        # Calcular distribución de contexto
        for key in stats['context_distribution'].keys():
            values = [context.get(key) for context in contexts if key in context]
            if values:
                unique, counts = np.unique(values, return_counts=True)
                stats['context_distribution'][key] = dict(zip(unique.tolist(), counts.tolist()))
        
        return stats
    
    def clear_cache(self):
        """Limpia la cache de búsquedas por contexto."""
        self.cache.clear()