import pandas as pd
import torch
from typing import List, Dict, Any, Optional, Callable, Iterator
import numpy as np
#from folders import *

import pandas as pd

import torch
from typing import List, Dict, Any, Optional, Callable, Iterator, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum

class SplitType(Enum):
    """Tipos de división disponibles."""
    TRAIN = "train"
    TEST = "test"
    VALIDATION = "validation"
    ALL = "all"

@dataclass
class SplitIndices:
    """Estructura para almacenar índices de división."""
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    val_start: int
    val_end: int

class Client:
    def __init__(self, 
                 path: str, 
                 recompensa_func: Callable[[int, int], float],
                 get_embedding_func: Callable[[str], torch.Tensor],
                 batch_size: Optional[int] = None,
                 connect_with_server:bool=False,
                 split_ratios: Optional[Tuple[float, float, float]] = None):
        """
        Inicializa el cliente para procesar datos de recomendación.
        
        Args:
            path: Ruta al archivo CSV con los datos
            recompensa_func: Función que calcula la recompensa (count, listened_complete) -> float
            get_embedding_func: Función que obtiene el embedding para un track_id
            batch_size: Tamaño del batch para iteración (opcional)
            split_ratios: Tupla con ratios (train, test, validation). Ej: (0.7, 0.15, 0.15)
                          Si es None, no se divide el dataset
        """
        self.path = path
        self.recompensa_func = recompensa_func
        self.get_embedding_func = get_embedding_func
        self.batch_size = batch_size
        self.split_ratios = split_ratios
        
        # Leer y procesar los datos
        self.df = pd.read_csv(path)
        self._ensure_temporal_order()
        self.all_items = self._process_data()
        
        # Inicializar splits
        self.train_items = []
        self.test_items = []
        self.val_items = []
        self.split_indices = None
        
        if split_ratios is not None:
            self._create_splits(split_ratios)
    
    def _ensure_temporal_order(self):
        """Asegura que los datos estén ordenados temporalmente."""
        # Crear columna de timestamp para ordenamiento temporal
        if all(col in self.df.columns for col in ['year', 'month', 'day', 'hour', 'minute', 'second']):
            # Crear datetime para ordenamiento
            self.df['datetime'] = pd.to_datetime(
                self.df[['year', 'month', 'day', 'hour', 'minute', 'second']]
            )
            # Ordenar por fecha (más antigua a más reciente)
            self.df = self.df.sort_values('datetime').reset_index(drop=True)
            # Eliminar columna temporal si no se necesita
            self.df = self.df.drop(columns=['datetime'])
        else:
            # Si no hay columnas temporales completas, ordenar por las que existan
            temporal_cols = ['year', 'month', 'day', 'hour', 'minute', 'second', 'millisecond']
            available_cols = [col for col in temporal_cols if col in self.df.columns]
            if available_cols:
                self.df = self.df.sort_values(available_cols).reset_index(drop=True)
    
    def _process_data(self) -> List[Dict[str, Any]]:
        
        """Procesa el DataFrame y crea la lista de ítems."""
        items = []
        
        for _, row in self.df.iterrows():
            try:
                # Obtener embedding del track_id
                track_id = str(row['track_id'])

                embedding = self.get_embedding_func(track_id)
                
                # Calcular recompensa
                count = int(row['interaction_count'])
                listened_complete = int(row['interaction_ratio'])

                reward = self.recompensa_func(count, listened_complete)
                
                # Determinar si es día laboral (lunes a viernes = 1, fin de semana = 0)
                day_of_week = int(row['day_of_week'])
                is_workday = 1 if 0 <= day_of_week <= 4 else 0  # 0=Lunes, 4=Viernes
                
                # Crear ítem
                item = {
                    "embedding": embedding,
                    "context": {
                        'day_of_week': day_of_week,
                        'hour_of_day': int(row['hour']),
                        'is_workday': is_workday,
                        'month': int(row['month'])
                    },
                    "reward": reward,
                    "track_id": track_id,
                    "original_row": row.to_dict(),
                    "temporal_index": _  # Guardar índice temporal
                }
                items.append(item)
                
            except Exception as e:
                print(f"Error procesando fila {_}: {e}")
                continue
        
        return items
    
    def _create_splits(self, split_ratios: Tuple[float, float, float]):
        """
        Divide los datos secuencialmente en train, test y validation.
        
        Args:
            split_ratios: Tupla (train_ratio, test_ratio, validation_ratio)
        """
        train_ratio, test_ratio, val_ratio = split_ratios
        
        # Verificar que los ratios sumen 1 (aproximadamente)
        total = train_ratio + test_ratio + val_ratio
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Los ratios deben sumar 1.0, pero suman {total}")
        
        n_total = len(self.all_items)
        
        # Calcular índices de forma secuencial
        train_end = int(n_total * train_ratio)
        test_end = train_end + int(n_total * test_ratio)
        
        # Asegurar que todos los datos se usen
        val_end = n_total
        
        # Crear splits
        self.train_items = self.all_items[:train_end]
        self.test_items = self.all_items[train_end:test_end]
        self.val_items = self.all_items[test_end:val_end]
        
        # Guardar índices para referencia
        self.split_indices = SplitIndices(
            train_start=0,
            train_end=train_end,
            test_start=train_end,
            test_end=test_end,
            val_start=test_end,
            val_end=val_end
        )
        
        print(f"División creada: Train={len(self.train_items)}, "
              f"Test={len(self.test_items)}, Validation={len(self.val_items)}")
    
    def get_split(self, split_type: SplitType = SplitType.ALL) -> List[Dict[str, Any]]:
        """
        Obtiene los ítems de un split específico.
        
        Args:
            split_type: Tipo de split a obtener
            
        Returns:
            Lista de ítems del split solicitado
        """
        if split_type == SplitType.TRAIN:
            return self.train_items if self.split_ratios else self.all_items
        elif split_type == SplitType.TEST:
            return self.test_items if self.split_ratios else []
        elif split_type == SplitType.VALIDATION:
            return self.val_items if self.split_ratios else []
        elif split_type == SplitType.ALL:
            return self.all_items
        else:
            raise ValueError(f"Tipo de split no válido: {split_type}")
    
    def __len__(self) -> int:
        """Devuelve el número total de ítems."""
        return len(self.all_items)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Accede a un ítem por índice."""
        return self.all_items[idx]
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Itera sobre todos los ítems."""
        return iter(self.all_items)
    
    def iter_batches(self, 
                    batch_size: Optional[int] = None,
                    split_type: SplitType = SplitType.ALL,
                    shuffle: bool = False) -> Iterator[List[Dict[str, Any]]]:
        """
        Itera sobre los ítems en lotes.
        
        Args:
            batch_size: Tamaño del batch (si None, usa el batch_size del constructor)
            split_type: Tipo de split sobre el que iterar
            shuffle: Si es True, mezcla los datos dentro del split (manteniendo la división original)
        
        Yields:
            Lista de ítems por batch
        """
        size = batch_size or self.batch_size
        if size is None:
            raise ValueError("batch_size debe ser especificado en el constructor o en este método")
        
        items = self.get_split(split_type)
        
        # Crear copia para no modificar el original
        items_to_iterate = items.copy()
        
        if shuffle:
            # Mezclar pero manteniendo la división temporal
            # Podemos usar una semilla fija para reproducibilidad
            np.random.seed(42)  # Semilla fija para reproducibilidad
            np.random.shuffle(items_to_iterate)
        
        for i in range(0, len(items_to_iterate), size):
            yield items_to_iterate[i:i + size]
    
    def get_items_by_context(self, 
                            day_of_week: Optional[int] = None,
                            hour_of_day: Optional[int] = None,
                            is_workday: Optional[int] = None,
                            month: Optional[int] = None,
                            split_type: SplitType = SplitType.ALL) -> List[Dict[str, Any]]:
        """
        Obtiene ítems que coinciden con los valores de contexto especificados.
        
        Args:
            day_of_week: Día de la semana (0-6)
            hour_of_day: Hora del día (0-23)
            is_workday: Es día laboral (0 o 1)
            month: Mes (1-12)
            split_type: Tipo de split en el que buscar
        
        Returns:
            Lista de ítems que coinciden con todos los criterios especificados
        """
        items = self.get_split(split_type)
        filtered_items = items
        
        # Filtrar por cada criterio especificado
        if day_of_week is not None:
            filtered_items = [item for item in filtered_items 
                            if item['context']['day_of_week'] == day_of_week]
        
        if hour_of_day is not None:
            filtered_items = [item for item in filtered_items 
                            if item['context']['hour_of_day'] == hour_of_day]
        
        if is_workday is not None:
            filtered_items = [item for item in filtered_items 
                            if item['context']['is_workday'] == is_workday]
        
        if month is not None:
            filtered_items = [item for item in filtered_items 
                            if item['context']['month'] == month]
        
        return filtered_items
    
    def get_items_by_context_dict(self, 
                                 context_dict: Dict[str, Any],
                                 split_type: SplitType = SplitType.ALL) -> List[Dict[str, Any]]:
        """
        Obtiene ítems que coinciden con un diccionario de contexto.
        
        Args:
            context_dict: Diccionario con claves de contexto
            split_type: Tipo de split en el que buscar
        
        Returns:
            Lista de ítems que coinciden con los criterios
        """
        return self.get_items_by_context(
            day_of_week=context_dict.get('day_of_week'),
            hour_of_day=context_dict.get('hour_of_day'),
            is_workday=context_dict.get('is_workday'),
            month=context_dict.get('month'),
            split_type=split_type
        )
    
    def get_batch_tensors(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Convierte un batch de ítems en tensores para entrenamiento.
        
        Args:
            batch: Lista de ítems (output de iter_batches o get_items_by_context)
        
        Returns:
            Diccionario con tensores organizados
        """
        embeddings = torch.stack([item['embedding'] for item in batch])
        rewards = torch.tensor([item['reward'] for item in batch], dtype=torch.float32)
        
        # Contexto como tensor
        contexts = torch.tensor([
            [
                item['context']['day_of_week'],
                item['context']['hour_of_day'],
                item['context']['is_workday'],
                item['context']['month']
            ]
            for item in batch
        ], dtype=torch.float32)
        
        return {
            'embeddings': embeddings,
            'contexts': contexts,
            'rewards': rewards
        }
    
    def get_split_statistics(self, split_type: SplitType = SplitType.ALL) -> Dict[str, Any]:
        """
        Devuelve estadísticas sobre los datos de un split específico.
        
        Args:
            split_type: Tipo de split para calcular estadísticas
        
        Returns:
            Diccionario con estadísticas
        """
        items = self.get_split(split_type)
        
        if not items:
            return {
                'split_type': split_type.value,
                'total_items': 0,
                'message': 'Split vacío'
            }
        
        rewards = [item['reward'] for item in items]
        contexts = [item['context'] for item in items]
        
        # Calcular rango temporal si está disponible
        temporal_info = {}
        if 'temporal_index' in items[0]:
            indices = [item['temporal_index'] for item in items]
            temporal_info = {
                'min_index': min(indices),
                'max_index': max(indices),
                'range': max(indices) - min(indices) + 1
            }
        
        return {
            'split_type': split_type.value,
            'total_items': len(items),
            'temporal_info': temporal_info,
            'reward_stats': {
                'mean': np.mean(rewards),
                'std': np.std(rewards),
                'min': np.min(rewards),
                'max': np.max(rewards),
                'median': np.median(rewards)
            },
            'context_distribution': {
                'day_of_week': pd.Series([c['day_of_week'] for c in contexts]).value_counts().sort_index().to_dict(),
                'hour_of_day': pd.Series([c['hour_of_day'] for c in contexts]).value_counts().sort_index().to_dict(),
                'is_workday': pd.Series([c['is_workday'] for c in contexts]).value_counts().sort_index().to_dict(),
                'month': pd.Series([c['month'] for c in contexts]).value_counts().sort_index().to_dict()
            }
        }
    
    def get_all_statistics(self) -> Dict[str, Any]:
        """
        Devuelve estadísticas de todos los splits.
        
        Returns:
            Diccionario con estadísticas de cada split
        """
        stats = {
            'all': self.get_split_statistics(SplitType.ALL),
            'has_splits': self.split_ratios is not None
        }
        
        if self.split_ratios is not None:
            stats['train'] = self.get_split_statistics(SplitType.TRAIN)
            stats['test'] = self.get_split_statistics(SplitType.TEST)
            stats['validation'] = self.get_split_statistics(SplitType.VALIDATION)
            stats['split_ratios'] = self.split_ratios
            stats['split_indices'] = {
                'train': (self.split_indices.train_start, self.split_indices.train_end),
                'test': (self.split_indices.test_start, self.split_indices.test_end),
                'validation': (self.split_indices.val_start, self.split_indices.val_end)
            }
        
        return stats
    
    def visualize_split_distribution(self):
        """
        Visualiza la distribución de los splits.
        
        Nota: Requiere matplotlib instalado.
        """
        try:
            import matplotlib.pyplot as plt
            
            if self.split_ratios is None:
                print("No hay splits definidos")
                return
            
            # Crear figura
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Distribución de Splits', fontsize=16)
            
            splits = ['Train', 'Test', 'Validation']
            sizes = [len(self.train_items), len(self.test_items), len(self.val_items)]
            colors = ['#2ecc71', '#e74c3c', '#3498db']
            
            # Gráfico de torta
            axes[0, 0].pie(sizes, labels=splits, autopct='%1.1f%%', colors=colors, startangle=90)
            axes[0, 0].set_title('Distribución Proporcional')
            
            # Gráfico de barras
            axes[0, 1].bar(splits, sizes, color=colors)
            axes[0, 1].set_title('Número de Ítems por Split')
            axes[0, 1].set_ylabel('Cantidad')
            
            # Distribución temporal
            if self.split_indices:
                x_positions = ['Train', 'Test', 'Validation']
                x_ranges = [
                    (self.split_indices.train_start, self.split_indices.train_end),
                    (self.split_indices.test_start, self.split_indices.test_end),
                    (self.split_indices.val_start, self.split_indices.val_end)
                ]
                
                axes[1, 0].barh(x_positions, [r[1]-r[0] for r in x_ranges], 
                               left=[r[0] for r in x_ranges], color=colors)
                axes[1, 0].set_title('Distribución Temporal')
                axes[1, 0].set_xlabel('Índice Temporal')
            
            # Distribución de recompensas
            for i, (split_name, split_items) in enumerate(zip(
                ['Train', 'Test', 'Validation'],
                [self.train_items, self.test_items, self.val_items]
            )):
                if split_items:
                    rewards = [item['reward'] for item in split_items]
                    axes[1, 1].hist(rewards, alpha=0.5, label=split_name, color=colors[i])
            
            axes[1, 1].set_title('Distribución de Recompensas')
            axes[1, 1].set_xlabel('Recompensa')
            axes[1, 1].set_ylabel('Frecuencia')
            axes[1, 1].legend()
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Para visualizar la distribución, instala matplotlib: pip install matplotlib")