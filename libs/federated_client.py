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

    def fit(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[List[np.ndarray], int, Dict]:
        """Entrenamiento local."""
        self.set_parameters(parameters)
        
        epochs = int(config.get("epochs", 1))
        epsilon_start = float(config.get("epsilon_start", 0.1))
        
        # Entrenar una época
        metrics = self.trainer.train_epoch(epsilon=epsilon_start, print_logs=True)
        
        # Número de ejemplos usados para el promedio ponderado en el servidor
        num_examples = len(self.trainer.client.get_split(self.trainer.client.SplitType.TRAIN))
        
        return self.get_parameters(config={}), num_examples, metrics

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[float, int, Dict]:
        """Evaluación local."""
        self.set_parameters(parameters)
        
        metrics = self.trainer.evaluate(self.trainer.client.SplitType.VALIDATION)
        
        loss = metrics.get("critic_loss", 0.0) # Flower usa 'loss' como métrica principal
        num_examples = len(self.trainer.client.get_split(self.trainer.client.SplitType.VALIDATION))
        
        return float(loss), num_examples, metrics
