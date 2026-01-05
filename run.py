#!/usr/bin/env python3
"""
Script: ejecutar_federated.py
Uso: python ejecutar_federated.py <veces>
Script Python para ejecutar múltiples veces el proceso federado.
"""

import sys
import json
import time
import logging
import requests
import subprocess
from typing import Optional, Dict, Any
import argparse
from datetime import datetime

# Configuración
API_URL = "http://10.10.0.2:8081/get_user"
PYTHON_SCRIPT = "main_federated.py"

# Configurar logging
def setup_logging(log_to_file: bool = True) -> logging.Logger:
    """Configura el sistema de logging"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Formato del log
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler para consola
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler para archivo (opcional)
    if log_to_file:
        file_handler = logging.FileHandler('ejecutar_federated.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

class FederatedExecutor:
    """Clase para ejecutar el proceso federado"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.session = requests.Session()
        # Configurar timeout y reintentos
        self.session.mount('http://', requests.adapters.HTTPAdapter(
            max_retries=3,
            pool_connections=10,
            pool_maxsize=10
        ))
    
    def get_user_id(self) -> Optional[str]:
        """Obtiene user_id de la API"""
        try:
            self.logger.info(f"Haciendo petición a: {API_URL}")
            
            response = self.session.get(
                API_URL,
                timeout=10,  # 10 segundos de timeout
                headers={'Accept': 'application/json'}
            )
            response.raise_for_status()  # Lanza excepción para códigos 4xx/5xx
            
            # Intentar parsear JSON
            try:
                data = response.json()
                user_id = data.get('user_id')
                
                if user_id:
                    self.logger.debug(f"Respuesta JSON: {json.dumps(data, indent=2)}")
                    return str(user_id)
                else:
                    self.logger.warning(f"Campo 'user_id' no encontrado en respuesta: {data}")
                    
            except json.JSONDecodeError:
                # Si no es JSON válido, intentar extraer de otra forma
                self.logger.warning("Respuesta no es JSON válido, intentando extracción manual")
                content = response.text
                return self._extract_user_id_from_text(content)
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error en petición HTTP: {e}")
        except Exception as e:
            self.logger.error(f"Error inesperado: {e}")
        
        return None
    
    def _extract_user_id_from_text(self, text: str) -> Optional[str]:
        """Extrae user_id de texto no JSON"""
        # Método 1: Buscar patrón JSON-like
        import re
        
        # Patrón para "user_id":"valor"
        pattern = r'"user_id"\s*:\s*"([^"]+)"'
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        
        # Patrón para 'user_id':'valor'
        pattern = r"'user_id'\s*:\s*'([^']+)'"
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        
        # Patrón para user_id=valor
        pattern = r'user_id=([^\s&]+)'
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        
        self.logger.warning(f"No se pudo extraer user_id del texto: {text[:200]}...")
        return None
    
    def execute_python_script(self, user_id: str) -> bool:
        """Ejecuta el script Python con el user_id"""
        try:
            self.logger.info(f"Ejecutando {PYTHON_SCRIPT} con user_id: {user_id}")
            
            # Construir comando
            cmd = [sys.executable, PYTHON_SCRIPT, "--user-id", user_id]
            
            # Ejecutar con subprocess
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutos de timeout máximo
            )
            
            # Logear salida
            if result.stdout:
                self.logger.info(f"Salida de {PYTHON_SCRIPT}:\n{result.stdout}")
            if result.stderr:
                self.logger.warning(f"Salida de error de {PYTHON_SCRIPT}:\n{result.stderr}")
            
            self.logger.info(f"Script ejecutado exitosamente con código: {result.returncode}")
            return True
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"Script {PYTHON_SCRIPT} excedió el tiempo límite")
            return False
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Script falló con código {e.returncode}: {e.stderr}")
            return False
        except FileNotFoundError:
            self.logger.error(f"Script no encontrado: {PYTHON_SCRIPT}")
            return False
        except Exception as e:
            self.logger.error(f"Error ejecutando script: {e}")
            return False
    
    def run_iterations(self, num_iterations: int, delay: float = 1.0) -> Dict[str, Any]:
        """Ejecuta múltiples iteraciones"""
        stats = {
            'total_iterations': num_iterations,
            'successful_api_calls': 0,
            'failed_api_calls': 0,
            'successful_scripts': 0,
            'failed_scripts': 0,
            'start_time': datetime.now(),
            'iterations': []
        }
        
        self.logger.info(f"Iniciando {num_iterations} iteraciones para API: {API_URL}")
        
        for i in range(1, num_iterations + 1):
            iteration_info = {'iteration': i, 'success': False}
            
            try:
                self.logger.info(f"--- Iteración {i}/{num_iterations} ---")
                
                # Obtener user_id
                user_id = self.get_user_id()
                
                if user_id:
                    stats['successful_api_calls'] += 1
                    self.logger.info(f"User ID obtenido: {user_id}")
                    iteration_info['user_id'] = user_id
                    
                    # Ejecutar script Python
                    if self.execute_python_script(user_id):
                        stats['successful_scripts'] += 1
                        iteration_info['success'] = True
                        iteration_info['script_status'] = 'success'
                    else:
                        stats['failed_scripts'] += 1
                        iteration_info['script_status'] = 'failed'
                else:
                    stats['failed_api_calls'] += 1
                    self.logger.warning("No se pudo obtener user_id, saltando iteración")
                    iteration_info['script_status'] = 'skipped'
                
                # Esperar entre iteraciones (si no es la última)
                if i < num_iterations and delay > 0:
                    self.logger.debug(f"Esperando {delay} segundos...")
                    time.sleep(delay)
                    
            except KeyboardInterrupt:
                self.logger.info("Ejecución interrumpida por el usuario")
                stats['interrupted'] = True
                break
            except Exception as e:
                self.logger.error(f"Error en iteración {i}: {e}")
                stats['failed_scripts'] += 1
                iteration_info['error'] = str(e)
            
            stats['iterations'].append(iteration_info)
        
        stats['end_time'] = datetime.now()
        stats['duration'] = (stats['end_time'] - stats['start_time']).total_seconds()
        
        return stats
    
    def print_statistics(self, stats: Dict[str, Any]):
        """Imprime estadísticas de la ejecución"""
        self.logger.info("\n" + "="*50)
        self.logger.info("ESTADÍSTICAS DE EJECUCIÓN")
        self.logger.info("="*50)
        
        self.logger.info(f"Total de iteraciones programadas: {stats['total_iterations']}")
        self.logger.info(f"Llamadas API exitosas: {stats['successful_api_calls']}")
        self.logger.info(f"Llamadas API fallidas: {stats['failed_api_calls']}")
        self.logger.info(f"Scripts ejecutados exitosamente: {stats['successful_scripts']}")
        self.logger.info(f"Scripts fallidos: {stats['failed_scripts']}")
        
        if stats['total_iterations'] > 0:
            success_rate = (stats['successful_scripts'] / stats['total_iterations']) * 100
            self.logger.info(f"Tasa de éxito: {success_rate:.2f}%")
        
        self.logger.info(f"Duración total: {stats['duration']:.2f} segundos")
        
        if stats.get('interrupted'):
            self.logger.warning("Ejecución interrumpida por el usuario")

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description='Ejecutar múltiples veces el proceso federado',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Ejemplos:
  %(prog)s 5                      # Ejecutar 5 veces
  %(prog)s 10 --delay 2           # Ejecutar 10 veces con 2 segundos de espera
  %(prog)s 100 --no-log-file      # Ejecutar 100 veces sin log a archivo
  %(prog)s 50 --debug             # Modo debug con más información
        '''
    )
    
    parser.add_argument(
        'num_iterations',
        type=int,
        default=1,
        help='Número de veces a ejecutar el proceso'
    )
    
    parser.add_argument(
        '--delay',
        type=float,
        default=1.0,
        help='Segundos de espera entre iteraciones (default: 1.0)'
    )
    
    parser.add_argument(
        '--no-log-file',
        action='store_true',
        help='No guardar log en archivo'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Habilitar modo debug (más información)'
    )
    
    parser.add_argument(
        '--output-stats',
        type=str,
        help='Archivo para guardar estadísticas en JSON'
    )
    
    args = parser.parse_args()
    
    # Validar argumentos
    if args.num_iterations <= 0:
        print("ERROR: El número de iteraciones debe ser mayor que 0")
        sys.exit(1)
    
    # Configurar logging
    logger = setup_logging(log_to_file=not args.no_log_file)
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Modo debug activado")
    
    # Crear ejecutor
    executor = FederatedExecutor(logger)
    
    try:
        # Ejecutar iteraciones
        stats = executor.run_iterations(args.num_iterations, args.delay)
        
        # Mostrar estadísticas
        executor.print_statistics(stats)
        
        # Guardar estadísticas si se solicita
        if args.output_stats:
            try:
                # Convertir datetime a string para JSON
                serializable_stats = stats.copy()
                serializable_stats['start_time'] = serializable_stats['start_time'].isoformat()
                serializable_stats['end_time'] = serializable_stats['end_time'].isoformat()
                
                with open(args.output_stats, 'w') as f:
                    json.dump(serializable_stats, f, indent=2)
                logger.info(f"Estadísticas guardadas en: {args.output_stats}")
            except Exception as e:
                logger.error(f"Error guardando estadísticas: {e}")
        
        # Retornar código de salida apropiado
        if stats.get('interrupted'):
            sys.exit(130)  # SIGINT
        elif stats['successful_scripts'] == 0 and stats['total_iterations'] > 0:
            sys.exit(1)  # Fallo total
        else:
            sys.exit(0)  # Éxito
            
    except KeyboardInterrupt:
        logger.info("\nEjecución cancelada por el usuario")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Error crítico: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()