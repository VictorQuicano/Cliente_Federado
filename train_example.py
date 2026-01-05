import torch
import numpy as np
import logging
from libs.client import Client, SplitType
from libs.model import ContextAwareActor, ContextAwareCritic, Recommender, RecommenderTrainer

import os
from dotenv import load_dotenv

load_dotenv()

# Configuración de logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

USER_ID = "user_55239"
SELECTED_USER_COMPLETE_PATH = f"/mnt/ssd/Carrera/5th_Year/X_SEMESTER/PFC_3/Dataset/processed_users/{USER_ID}_processed.csv"
EMBEDDING_DIM = 64


API_URL = os.getenv("API_URL")
EMBEDDING_URL= f"{API_URL}/info"

def calcular_recompensa_normalizada(interaction_count: int, interaction_ratio: float) -> float:
    """Recompensa acotada entre 0 y 1 para estabilidad."""
    count_factor = np.log1p(interaction_count) / np.log1p(10)
    alpha = 0.6
    beta = 0.4
    recompensa = alpha * count_factor + beta * interaction_ratio
    return float(np.clip(recompensa, 0.0, 1.0))

def ejemplo_get_embedding(track_id: str) -> torch.Tensor:
    import requests
    response = requests.get(f"{EMBEDDING_URL}/{track_id}?info_type=embedding")
    if response.status_code == 200:
        embedding = response.json().get("data", {}).get("embedding", [])
        return torch.tensor(embedding, dtype=torch.float32)
    else:
        logging.error(f"Error al obtener embedding para track_id {track_id}: {response.status_code}")
        return torch.zeros(EMBEDDING_DIM, dtype=torch.float32)
    

logging.info(f"Iniciando configuración para el usuario: {USER_ID}")

logging.info(f"Cargando datos del cliente desde: {SELECTED_USER_COMPLETE_PATH}")
client = Client(
    path=SELECTED_USER_COMPLETE_PATH,
    recompensa_func=calcular_recompensa_normalizada,
    get_embedding_func=ejemplo_get_embedding,
    batch_size=32,
    split_ratios=(0.7, 0.15, 0.15),
    cache_path="cache_client.json",
    embeddings_path="/mnt/ssd/Carrera/Datasets/Music4all-Onion/music_4_all_compress_64.csv"
)

DEVICE = torch.device('cpu')
logging.info(f"Dispositivo detectado: {DEVICE}")

logging.info("Inicializando modelos ContextAwareActor y ContextAwareCritic...")
actor = ContextAwareActor(embedding_dim=EMBEDDING_DIM).to(DEVICE)
critic = ContextAwareCritic(action_dim=EMBEDDING_DIM).to(DEVICE)

logging.info("Configurando Recommender y RecommenderTrainer...")
recommender = Recommender(client=client)
recommender_trainer = RecommenderTrainer(
    actor=actor,
    critic=critic,
    client=client,
    recommender=recommender,
    device=DEVICE
)

logging.info("Iniciando entrenamiento...")
try:
    history = recommender_trainer.train(
        num_epochs=2,
        epsilon_start=0.9,
        epsilon_end=0.1,
        epsilon_decay=0.995,
        eval_freq=1,
        save_path="modelo_recomendacion"
    )
    logging.info("Entrenamiento completado satisfactoriamente.")
except Exception as e:
    logging.error(f"Error durante el entrenamiento: {e}")