import flwr as fl
import torch
import numpy as np
from collections import OrderedDict
from typing import List, Dict, Tuple
from .model import RecommenderTrainer

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainer: RecommenderTrainer):
        self.trainer = trainer
        self.device = trainer.device

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        """Extrae los pesos del Actor y Crítico."""
        params = []
        # Pesos del Actor
        for val in self.trainer.actor.state_dict().values():
            params.append(val.cpu().numpy())
        # Pesos del Crítico
        for val in self.trainer.critic.state_dict().values():
            params.append(val.cpu().numpy())
        return params

    def set_parameters(self, parameters: List[np.ndarray]):
        """Carga los pesos recibidos del servidor en el Actor y Crítico."""
        # Separar parámetros para Actor y Crítico
        actor_params_len = len(self.trainer.actor.state_dict())
        actor_params = parameters[:actor_params_len]
        critic_params = parameters[actor_params_len:]

        # Cargar en Actor
        actor_state_dict = OrderedDict({
            k: torch.tensor(v) for k, v in zip(self.trainer.actor.state_dict().keys(), actor_params)
        })
        self.trainer.actor.load_state_dict(actor_state_dict)

        # Cargar en Crítico
        critic_state_dict = OrderedDict({
            k: torch.tensor(v) for k, v in zip(self.trainer.critic.state_dict().keys(), critic_params)
        })
        self.trainer.critic.load_state_dict(critic_state_dict)

        # Sincronizar target networks (soft update o hard copy inicial)
        self.trainer.actor_target.load_state_dict(self.trainer.actor.state_dict())
        self.trainer.critic_target.load_state_dict(self.trainer.critic.state_dict())

    def _reload_data(self, user_id: str):
        """Recarga los datos del entrenador para un nuevo usuario."""
        import os
        # Asumiendo que la ruta base está configurada en el cliente original o es predecible
        base_path = "/mnt/ssd/Carrera/5th_Year/X_SEMESTER/PFC_3/Dataset/processed_users/"
        new_path = os.path.join(base_path, f"{user_id}_processed.csv")
        
        if os.path.exists(new_path):
            print(f"🔄 Recargando datos para el usuario: {user_id}")
            # Actualizamos el cliente de datos dentro del entrenador
            self.trainer.client.load_user_data(new_path)
            # También necesitamos actualizar el Recommender si tiene referencias cacheadas
            self.trainer.recommender.client = self.trainer.client
        else:
            print(f"⚠️ Error: No se encontró el archivo para el usuario {user_id} en {new_path}")

    def fit(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[List[np.ndarray], int, Dict]:
        """Entrenamiento local con asignación dinámica de usuario."""
        self.set_parameters(parameters)
        
        # Obtener user_id asignado por el servidor
        target_user_id = config.get("target_user_id")
        if target_user_id:
            self._reload_data(target_user_id)

        epochs = int(config.get("epochs", 1))
        epsilon_start = float(config.get("epsilon_start", 0.1))
        
        # Entrenar una época
        metrics = self.trainer.train_epoch(epsilon=epsilon_start, print_logs=True)
        
        # Número de ejemplos usados para el promedio ponderado en el servidor
        num_examples = len(self.trainer.client.get_split(self.trainer.client.SplitType.TRAIN))
        
        # Añadir user_id a las métricas para tracking si es necesario
        metrics["user_id"] = target_user_id
        
        return self.get_parameters(config={}), num_examples, metrics

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[float, int, Dict]:
        """Evaluación local."""
        self.set_parameters(parameters)
        
        # También asegurar que evaluación use el usuario correcto si se pasa en config
        target_user_id = config.get("target_user_id")
        if target_user_id:
            self._reload_data(target_user_id)
            
        metrics = self.trainer.evaluate(self.trainer.client.SplitType.VALIDATION)
        
        loss = metrics.get("critic_loss", 0.0) 
        num_examples = len(self.trainer.client.get_split(self.trainer.client.SplitType.VALIDATION))
        
        return float(loss), num_examples, metrics
