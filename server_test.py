import flwr as fl
from typing import Dict, List, Tuple, Optional
import numpy as np
import logging
from flwr.server.strategy import FedAvg
from flwr.common import Parameters, Scalar, FitIns
import torch
import json
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DDPGFedAvg(FedAvg):
    def __init__(self, save_path: str = "federated_metrics.json", *args, **kwargs):
        self.save_path = save_path
        self.metrics_history = []
        super().__init__(*args, **kwargs)
    
    def aggregate_fit(self, server_round, results, failures):
        """Agregar resultados del entrenamiento de clientes"""
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_parameters is not None:
            # Calcular métricas promedio de los clientes
            val_rewards = []
            train_losses = []
            actor_losses = []
            
            for client, fit_res in results:
                metrics = fit_res.metrics
                if metrics:
                    val_rewards.append(metrics.get("val_reward", 0.0))
                    train_losses.append(metrics.get("train_loss", 1.0))
                    actor_losses.append(metrics.get("actor_loss", 0.0))
            
            # Estadísticas de parámetros (para verificar si el modelo cambia)
            if aggregated_parameters:
                ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)
                norms = [np.linalg.norm(arr) for arr in ndarrays]
                avg_norm = np.mean(norms)
                logging.info(f"[Ronda {server_round}] Norma promedio de pesos: {avg_norm:.4f}")

            if val_rewards:
                avg_val_reward = np.mean(val_rewards)
                avg_train_loss = np.mean(train_losses)
                avg_actor_loss = np.mean(actor_losses)
                
                logging.info(f"📊 [Ronda {server_round}] RESUMEN:")
                logging.info(f"   • Recompensa Val: {avg_val_reward:.4f}")
                logging.info(f"   • Critic Loss: {avg_train_loss:.4f}")
                logging.info(f"   • Actor Loss: {avg_actor_loss:.4f}")
                
                # Actualizar métricas agregadas
                if aggregated_metrics is None:
                    aggregated_metrics = {}
                aggregated_metrics["avg_val_reward"] = float(avg_val_reward)
                aggregated_metrics["avg_train_loss"] = float(avg_train_loss)
                aggregated_metrics["avg_actor_loss"] = float(avg_actor_loss)

                # Guardar en el historial para el JSON
                self.metrics_history.append({
                    "round": server_round,
                    "avg_val_reward": float(avg_val_reward),
                    "avg_train_loss": float(avg_train_loss),
                    "avg_actor_loss": float(avg_actor_loss),
                    "weight_norm": float(avg_norm) if 'avg_norm' in locals() else 0.0
                })
                self._save_metrics()
        
        return aggregated_parameters, aggregated_metrics

    def _save_metrics(self):
        """Guarda el historial de métricas en un archivo JSON"""
        try:
            with open(self.save_path, "w") as f:
                json.dump(self.metrics_history, f, indent=4)
        except Exception as e:
            logging.error(f"Error al guardar métricas: {e}")
    
    def configure_fit(self, server_round, parameters, client_manager):
        """Configurar entrenamiento para cada cliente - Versión actualizada"""
        # Configurar parámetros para esta ronda
        config = {
            "server_round": server_round,
            "local_epochs": 5,
            "epsilon_start": max(0.9 * (0.95 ** (server_round - 1)), 0.1),
            "epsilon_end": 0.1,
            "epsilon_decay": 0.995,
        }
        
        # Obtener clientes para esta ronda
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients
        )
        
        # Crear instrucciones de entrenamiento para cada cliente
        fit_instructions = []
        for client in clients:
            fit_ins = FitIns(parameters, config)
            fit_instructions.append((client, fit_ins))
        
        return fit_instructions
    
    def configure_evaluate(self, server_round, parameters, client_manager):
        """Configurar evaluación para cada cliente"""
        config = {
            "server_round": server_round
        }
        
        # Obtener clientes para evaluación
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients
        )
        
        # Crear instrucciones de evaluación
        evaluate_instructions = []
        for client in clients:
            evaluate_ins = fl.common.EvaluateIns(parameters, config)
            evaluate_instructions.append((client, evaluate_ins))
        
        return evaluate_instructions

def get_initial_parameters() -> Parameters:
    """Inicializar parámetros del modelo global"""
    # Crear modelos dummy para obtener la estructura
    # IMPORTANTE: Importa aquí para evitar dependencias circulares
    try:
        from libs.model import ContextAwareActor, ContextAwareCritic
        
        actor = ContextAwareActor(embedding_dim=64)
        critic = ContextAwareCritic(action_dim=64)
        
        # Combinar parámetros
        params = []
        for model in [actor, critic]:
            params.extend([val.cpu().numpy() for _, val in model.state_dict().items()])
        
        return fl.common.ndarrays_to_parameters(params)
    except ImportError:
        # Si no se pueden importar los modelos, crear parámetros dummy
        logging.warning("No se pudieron importar los modelos, usando parámetros dummy")
        # Crear parámetros dummy basados en dimensiones esperadas
        import numpy as np
        
        # Estimación de tamaño de parámetros para actor y crítico
        # Ajusta estos tamaños según tu arquitectura real
        dummy_params = [
            np.random.randn(64, 64).astype(np.float32),  # Capa 1 actor
            np.random.randn(64).astype(np.float32),      # Bias capa 1 actor
            np.random.randn(64, 64).astype(np.float32),  # Capa 2 actor
            np.random.randn(64).astype(np.float32),      # Bias capa 2 actor
            np.random.randn(64, 64).astype(np.float32),  # Capa 1 crítico
            np.random.randn(64).astype(np.float32),      # Bias capa 1 crítico
        ]
        
        return fl.common.ndarrays_to_parameters(dummy_params)

def evaluate_metrics_aggregation_fn(results):
    """Función para agregar métricas de evaluación"""
    metrics_list = [m for _, m in results if m]
    if not metrics_list:
        return {}
    
    aggregated = {
        "avg_val_reward": float(np.mean([m.get("val_reward", 0.0) for m in metrics_list])),
        "avg_precision@5": float(np.mean([m.get("precision@5", 0.0) for m in metrics_list])),
        "avg_ndcg@5": float(np.mean([m.get("ndcg@5", 0.0) for m in metrics_list])),
    }
    
    logging.info(f"📈 [EVALUACIÓN] Recompensa: {aggregated['avg_val_reward']:.4f}, NDCG@5: {aggregated['avg_ndcg@5']:.4f}")
    return aggregated

def fit_metrics_aggregation_fn(results):
    """Función para agregar métricas de entrenamiento (fit)"""
    metrics_list = [m for _, m in results if m]
    if not metrics_list:
        return {}
    
    aggregated = {
        "avg_train_loss": float(np.mean([m.get("train_loss", 0.0) for m in metrics_list])),
        "avg_actor_loss": float(np.mean([m.get("actor_loss", 0.0) for m in metrics_list])),
        "avg_val_reward": float(np.mean([m.get("val_reward", 0.0) for m in metrics_list])),
    }
    return aggregated

def main():
    """Función principal del servidor - Versión actualizada"""
    # Definir estrategia
    strategy = DDPGFedAvg(
        fraction_fit=0.5,  # Fracción de clientes para entrenamiento
        fraction_evaluate=0.5,  # Fracción de clientes para evaluación
        min_fit_clients=2,  # Mínimo de clientes para entrenamiento
        min_evaluate_clients=2,  # Mínimo de clientes para evaluación
        min_available_clients=2,  # Mínimo de clientes disponibles
        initial_parameters=get_initial_parameters(),
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
    )
    
    # Configurar servidor
    server_config = fl.server.ServerConfig(num_rounds=10)
    
    # Usar el método recomendado para versiones nuevas
    # Opción 1: Usar el método nuevo recomendado
    try:
        # Para Flower 1.5+ con la nueva API
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=server_config,
            strategy=strategy,
            grpc_max_message_length=1024*1024*1024  # 1GB para modelos grandes
        )
    except TypeError as e:
        # Fallback para versiones ligeramente diferentes
        logging.warning(f"Intentando método alternativo: {e}")
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=server_config,
            strategy=strategy
        )

if __name__ == "__main__":
    main()

    