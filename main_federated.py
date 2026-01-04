import torch
import numpy as np
import logging
import os
import flwr as fl
from google.cloud import storage
from dotenv import load_dotenv
from libs.client import Client, SplitType
from libs.model import ContextAwareActor, ContextAwareCritic, Recommender, RecommenderTrainer
from libs.federated_client import FlowerClient

load_dotenv()

# Configuración de logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

USER_ID = os.getenv("USER_ID", "pending")
# Ruta temporal, será actualizada por el servidor en la primera ronda
SELECTED_USER_COMPLETE_PATH = f"/mnt/ssd/Carrera/5th_Year/X_SEMESTER/PFC_3/Dataset/processed_users/user_{USER_ID}_processed.csv"
EMBEDDING_DIM = 64

API_URL = os.getenv("API_URL", "http://localhost:5000")

EMBEDDING_URL = f"{API_URL}/info"
SERVER_ADDRESS = os.getenv("FEDERATED_SERVER_ADDRESS", "127.0.0.1:8080")

def calcular_recompensa_normalizada(interaction_count: int, interaction_ratio: float) -> float:
    """Recompensa acotada entre 0 y 1 para estabilidad."""
    count_factor = np.log1p(interaction_count) / np.log1p(10)
    alpha = 0.6
    beta = 0.4
    recompensa = alpha * count_factor + beta * interaction_ratio
    return float(np.clip(recompensa, 0.0, 1.0))

def ejemplo_get_embedding(track_id: str) -> torch.Tensor:
    import requests
    try:
        response = requests.get(f"{EMBEDDING_URL}/{track_id}?info_type=embedding")
        if response.status_code == 200:
            embedding = response.json().get("data", {}).get("embedding", [])
            return torch.tensor(embedding, dtype=torch.float32)
        else:
            logging.error(f"Error al obtener embedding para track_id {track_id}: {response.status_code}")
            return torch.zeros(EMBEDDING_DIM, dtype=torch.float32)
    except Exception as e:
        logging.error(f"Excepción al conectar con el servidor de embeddings: {e}")
        return torch.zeros(EMBEDDING_DIM, dtype=torch.float32)


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Descarga un archivo de un bucket de Google Cloud Storage."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        
        # Crear directorios si no existen
        os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)
        
        blob.download_to_filename(destination_file_name)
        logging.info(f"Archivo descargado exitosamente de gs://{bucket_name}/{source_blob_name} a {destination_file_name}")
        return True
    except Exception as e:
        logging.error(f"Error al descargar archivo de GCS: {e}")
        return False

def main():
    logging.info(f"Iniciando Cliente Federado para el usuario: {USER_ID}")

    # Descargar datos del usuario desde GCS
    bucket_name = "recommender-system-datasets-tesis-experiment"
    # Se asume que en el bucket el archivo tiene el mismo nombre que el destino local deseado
    # O user_{USER_ID}_processed.csv, que parece ser el formato estándar aquí.
    # El prompt dice: "descargar ... en el archivo <user_id>_processed.csv".
    # Asumimos que el source blob sigue el patrón user_histories/user_<id>_processed.csv
    # dado que SELECTED_USER_COMPLETE_PATH termina en user_{USER_ID}_processed.csv
    filename = f"user_{USER_ID}_processed.csv"
    source_blob_name = f"user_histories/{filename}"
    
    if not download_blob(bucket_name, source_blob_name, SELECTED_USER_COMPLETE_PATH):
        logging.warning("No se pudo descargar el archivo de GCS. Verificando si existe localmente...")

    if not os.path.exists(SELECTED_USER_COMPLETE_PATH):
        logging.error(f"Archivo de datos no encontrado: {SELECTED_USER_COMPLETE_PATH}")
        return

    client_data = Client(
        path=SELECTED_USER_COMPLETE_PATH,
        recompensa_func=calcular_recompensa_normalizada,
        get_embedding_func=ejemplo_get_embedding,
        batch_size=32,
        split_ratios=(0.7, 0.15, 0.15)
    )

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Dispositivo detectado: {DEVICE}")

    actor = ContextAwareActor(embedding_dim=EMBEDDING_DIM).to(DEVICE)
    critic = ContextAwareCritic(action_dim=64).to(DEVICE)

    recommender = Recommender(client=client_data)
    trainer = RecommenderTrainer(
        actor=actor,
        critic=critic,
        client=client_data,
        recommender=recommender,
        device=DEVICE
    )

    # Inicializar el cliente Flower
    flower_client = FlowerClient(trainer)

    # Iniciar la conexión con el servidor federado
    logging.info(f"Conectando al servidor federado en {SERVER_ADDRESS}...")
    fl.client.start_numpy_client(
        server_address=SERVER_ADDRESS,
        client=flower_client
    )

if __name__ == "__main__":
    main()
