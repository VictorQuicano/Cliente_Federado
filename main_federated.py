import logging
import os
import argparse
from typing import Dict, Tuple, List, Optional
from collections import OrderedDict
import copy

import torch
import numpy as np
import requests
from dotenv import load_dotenv
import flwr as fl

# Importaciones locales (asegurarse de que libs esté en el path)
from libs.client import Client, SplitType
from libs.model import ContextAwareActor, ContextAwareCritic, Recommender, RecommenderTrainer

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('cliente_federado.log')
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

# Constantes por defecto
USER_ID = "user_55239"
EMBEDDING_DIM = 64

USER_HISTORIES_PATH = os.getenv("USER_HISTORIES_PATH", "/mnt/ssd/Carrera/5th_Year/X_SEMESTER/PFC_3/Dataset/processed_users/")
METADATA_PATH = os.getenv("METADATA_PATH", "/mnt/ssd/Carrera/Datasets/Music4all-Onion/")


SELECTED_USER_COMPLETE_PATH = f"{USER_HISTORIES_PATH}/{USER_ID}_processed.csv"
EMBEDDINGS_PATH = f"{METADATA_PATH}/music_4_all_compress_64.csv"


SERVER_IP = os.getenv("SERVER_URL", "127.0.0.1")



EMBEDDING_URL = f"http://{SERVER_IP}:8072/info"
MONITORING_API = f"http://{SERVER_IP}:8083"
SERVER_URL = f"{SERVER_IP}:8080"

class MonitoringClient:
    def __init__(self, api_url=MONITORING_API):
        self.api_url = api_url
    
    def send_heartbeat(self, user_id: str, status: str, current_round: int = 0, session_id: int = None):
        try:
            payload = {
                "user_id": user_id, 
                "status": status,
                "current_round": current_round
            }
            if session_id:
                payload["session_id"] = session_id
                
            requests.post(
                f"{self.api_url}/client/heartbeat",
                json=payload,
                timeout=1
            )
        except Exception:
            pass # No bloquear si falla el monitoreo

    def log_metrics(self, session_id: int, user_id: str, round_num: int, metrics: Dict):
        try:
            requests.post(
                f"{self.api_url}/training/{session_id}/client/metrics",
                json={
                    "user_id": user_id,
                    "round_number": round_num,
                    "metrics": metrics
                },
                timeout=1
            )
        except Exception:
            pass

class DDPGFlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        user_id: str,
        data_path: str,
        embedding_dim: int = 64,
        local_epochs: int = 10,
        batch_size: int = 32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        embedding_cache: str = "cache_client.json",
        embeddings_path: Optional[str] = None
    ):
        self.user_id = user_id
        self.data_path = data_path
        self.embedding_dim = embedding_dim
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.embedding_cache = embedding_cache
        self.embeddings_path = embeddings_path
        
        self.monitor = MonitoringClient()
        self.monitor.send_heartbeat(user_id, "IDLE")
        
        # Inicializar componentes del modelo DDPG
        self.actor = None
        self.critic = None
        self.recommender = None
        self.recommender_trainer = None
        self.client_data = None
        
        self._initialize_components()
        
        logging.info(f"Cliente federado inicializado para usuario: {user_id}")
    
    def _initialize_components(self):
        """Inicializar todos los componentes necesarios para el entrenamiento DDPG"""
        # Configurar función de recompensa
        def calcular_recompensa_normalizada(interaction_count: int, interaction_ratio: float) -> float:
            count_factor = np.log1p(interaction_count) / np.log1p(10)
            alpha = 0.6
            beta = 0.4
            recompensa = alpha * count_factor + beta * interaction_ratio
            return float(np.clip(recompensa, 0.0, 1.0))
        
        # Configurar función para obtener embeddings
        def ejemplo_get_embedding(track_id: str) -> torch.Tensor:
            try:
                import requests
                response = requests.get(
                    f"{EMBEDDING_URL}/{track_id}?info_type=embedding",
                    timeout=5
                )
                if response.status_code == 200:
                    embedding = response.json().get("data", {}).get("embedding", [])
                    return torch.tensor(embedding, dtype=torch.float32)
            except Exception as e:
                logging.warning(f"Error obteniendo embedding: {e}")
            return torch.zeros(self.embedding_dim, dtype=torch.float32)
        
        # Inicializar cliente de datos
        self.monitor.send_heartbeat(self.user_id, "LOADING_DATA")
        logging.info(f"Cargando datos desde {self.data_path}")
        self.client_data = Client(
            path=self.data_path,
            recompensa_func=calcular_recompensa_normalizada,
            get_embedding_func=ejemplo_get_embedding,
            batch_size=self.batch_size,
            split_ratios=(0.7, 0.15, 0.15),
            cache_path=self.embedding_cache,
            embeddings_path=self.embeddings_path
        )
        
        # Inicializar modelos
        self.actor = ContextAwareActor(embedding_dim=self.embedding_dim).to(self.device)
        self.critic = ContextAwareCritic(action_dim=64).to(self.device)
        
        # Inicializar recomendador y entrenador
        self.recommender = Recommender(client=self.client_data)
        self.recommender_trainer = RecommenderTrainer(
            actor=self.actor,
            critic=self.critic,
            client=self.client_data,
            recommender=self.recommender,
            device=self.device
        )
    
    def get_parameters(self, config: Dict):
        """Obtener parámetros del modelo para enviar al servidor"""
        params = []
        
        # Obtener parámetros del actor
        actor_params = [val.cpu().numpy() for _, val in self.actor.state_dict().items()]
        # Obtener parámetros del crítico
        critic_params = [val.cpu().numpy() for _, val in self.critic.state_dict().items()]
        
        # Combinar parámetros
        params.extend(actor_params)
        params.extend(critic_params)
        
        return params
    
    def set_parameters(self, parameters: List[np.ndarray], config: Dict):
        """Establecer parámetros recibidos del servidor"""
        # Separar parámetros de actor y crítico
        actor_keys = list(self.actor.state_dict().keys())
        critic_keys = list(self.critic.state_dict().keys())
        
        actor_len = len(actor_keys)
        critic_len = len(critic_keys)
        
        # Extraer parámetros del actor
        actor_params = parameters[:actor_len]
        actor_state_dict = OrderedDict({
            k: torch.tensor(v) for k, v in zip(actor_keys, actor_params)
        })
        
        # Extraer parámetros del crítico
        critic_params = parameters[actor_len:actor_len + critic_len]
        critic_state_dict = OrderedDict({
            k: torch.tensor(v) for k, v in zip(critic_keys, critic_params)
        })
        
        # Aplicar parámetros a los modelos
        self.actor.load_state_dict(actor_state_dict)
        self.critic.load_state_dict(critic_state_dict)
        
        # Mover modelos al dispositivo
        self.actor.to(self.device)
        self.critic.to(self.device)
        
        # Sincronizar modelos target para entrenamiento estable
        if self.recommender_trainer:
            self.recommender_trainer._soft_update_targets()
    
    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """Entrenar el modelo localmente"""
        # Extraer info de monitoreo
        session_id = config.get("session_id", -1)
        server_round = config.get("server_round", 1)
        
        self.monitor.send_heartbeat(self.user_id, "TRAINING", server_round, session_id)
        
        # Establecer parámetros globales
        self.set_parameters(parameters, config)
        
        # Extraer configuración de entrenamiento
        local_epochs = config.get("local_epochs", self.local_epochs)
        epsilon_start = config.get("epsilon_start", 0.9)
        epsilon_end = config.get("epsilon_end", 0.1)
        epsilon_decay = config.get("epsilon_decay", 0.995)
        
        logging.info(f"Iniciando entrenamiento local para usuario {self.user_id} - {local_epochs} épocas")
        
        # Entrenar localmente
        # history es un dict con listas de métricas (definido en RecommenderTrainer)
        history = self.recommender_trainer.train(
            num_epochs=local_epochs,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            eval_freq=1,
            save_path=None,  # No guardar durante entrenamiento federado
            print_logs=True
        )
        
        # Obtener métricas para reportar (tomamos el último valor de la época)
        # Nota: history["critic_loss"] es una lista de floats
        # history["val_metrics"] es un defaultdict(list)
        
        train_loss = 0.0
        if history.get("critic_loss"):
            train_loss = float(history["critic_loss"][-1])
            
        actor_loss = 0.0
        if history.get("actor_loss"):
            actor_loss = float(history["actor_loss"][-1])
            
        val_reward = 0.0
        if history.get("val_metrics") and history["val_metrics"].get("avg_reward"):
            val_reward = float(history["val_metrics"]["avg_reward"][-1])
        
        local_metrics = {
            "train_loss": train_loss,
            "actor_loss": actor_loss,
            "val_reward": val_reward,
            "user_id": self.user_id,
            "samples_trained": len(self.client_data.train_items) if self.client_data else 0
        }
        
        # Obtener parámetros actualizados
        updated_params = self.get_parameters(config)
        
        # Reportar métricas completas al monitor
        if session_id != -1:
            self.monitor.log_metrics(session_id, self.user_id, server_round, history)
            
        self.monitor.send_heartbeat(self.user_id, "IDLE", server_round, session_id)
        
        return updated_params, local_metrics["samples_trained"], local_metrics
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """Evaluar el modelo localmente"""
        # Extraer info de monitoreo
        session_id = config.get("session_id", -1)
        server_round = config.get("server_round", 1)
        
        self.monitor.send_heartbeat(self.user_id, "EVALUATING", server_round, session_id)
        
        # Establecer parámetros globales
        self.set_parameters(parameters, config)
        
        logging.info(f"Evaluando modelo para usuario {self.user_id}")
        
        try:
            eval_metrics = {}
            # Usar el método evaluate del entrenador si existe
            if hasattr(self.recommender_trainer, 'evaluate'):
                eval_metrics = self.recommender_trainer.evaluate(split_type=SplitType.VALIDATION)
            else:
                eval_metrics = {"avg_reward": 0.0}
            
            # Métricas para el servidor
            server_metrics = {
                "val_reward": float(eval_metrics.get("avg_reward", 0.0)),
                "precision@5": float(eval_metrics.get("precision@5", 0.0)),
                "ndcg@5": float(eval_metrics.get("ndcg@5", 0.0)),
                "user_id": self.user_id,
                "samples_evaluated": len(self.client_data.val_items) if self.client_data else 0
            }
            
            # Usar recompensa de validación como métrica principal para la pérdida
            # (Flower minimiza la pérdida, así que 1 - reward es una buena proxy)
            loss = 1.0 - server_metrics["val_reward"]
            
            # Reportar métricas
            if session_id != -1:
                self.monitor.log_metrics(session_id, self.user_id, server_round, server_metrics)
            
            self.monitor.send_heartbeat(self.user_id, "IDLE", server_round, session_id)
            
            return float(loss), server_metrics["samples_evaluated"], server_metrics
            
        except Exception as e:
            logging.error(f"Error en evaluación: {e}")
            return 1.0, 0, {"error": str(e)}

def get_client_fn(data_paths: Dict[str, str]):
    """Función de fábrica para crear clientes (opcional si se usa con Simulation)"""
    def client_fn(cid: str) -> fl.client.Client:
        user_id = f"user_{cid}"
        data_path = data_paths.get(user_id, data_paths.get("default"))
        return DDPGFlowerClient(
            user_id=user_id,
            data_path=data_path,
            embedding_dim=EMBEDDING_DIM,
            local_epochs=5
        ).to_client()
    return client_fn

# Script principal para ejecutar el cliente
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cliente Federado DDPG")
    parser.add_argument("--server-address", type=str, default=SERVER_URL,
                       help="Dirección del servidor Flower")
    parser.add_argument("--user-id", type=str, default=USER_ID,
                       help="ID del usuario")
    parser.add_argument("--data-path", type=str, default=SELECTED_USER_COMPLETE_PATH,
                       help="Ruta al archivo de datos del usuario")
    parser.add_argument("--embedding-dim", type=int, default=EMBEDDING_DIM,
                       help="Dimensión de los embeddings")
    parser.add_argument("--embedding-cache", type=str, default="cache_client.json",
                       help="Ruta al archivo de cache de embeddings")
    parser.add_argument("--embeddings-path", type=str, 
                       default=EMBEDDINGS_PATH,
                       help="Ruta al CSV con todos los embeddings")

    args = parser.parse_args()
    
    logger.info(f"Iniciando cliente federado para {args.user_id}...")
    
    # Crear instancia del cliente
    client = DDPGFlowerClient(
        user_id=args.user_id,
        data_path=args.data_path,
        embedding_dim=args.embedding_dim,
        local_epochs=5,
        batch_size=32,
        embedding_cache=args.embedding_cache,
        embeddings_path=args.embeddings_path
    )
    
    logger.info(f"Conectando cliente {args.user_id} a {args.server_address}")
    
    fl.client.start_client(
        server_address=args.server_address,
        client=client.to_client()
    )