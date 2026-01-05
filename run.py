#!/usr/bin/env python3
"""
Script simplificado para obtener user_id de la API y ejecutar main_federated.py
Uso: python run_federated.py
"""

import sys
import requests
import subprocess
import time
from typing import Optional

# Configuración
API_URL = "http://10.10.0.2:8081/get_user"
PYTHON_SCRIPT = "main_federated.py"

def get_user_id() -> Optional[str]:
    """Obtiene user_id de la API"""
    try:
        print(f"Obteniendo user_id de: {API_URL}")
        
        response = requests.get(
            API_URL,
            timeout=10,
            headers={'Accept': 'application/json'}
        )
        response.raise_for_status()
        
        # Intentar parsear como JSON
        try:
            data = response.json()
            user_id = data.get('user_id')
            if user_id:
                print(f"✓ User ID obtenido: {user_id}")
                return str(user_id)
            else:
                print(f"✗ Campo 'user_id' no encontrado en respuesta")
                print(f"Respuesta completa: {data}")
                
        except ValueError:  # Si no es JSON válido
            # Buscar user_id en el texto de respuesta
            text = response.text
            if "user_id" in text.lower():
                # Extraer usando métodos simples
                lines = text.split('\n')
                for line in lines:
                    if "user_id" in line.lower():
                        parts = line.split(':')
                        if len(parts) > 1:
                            user_id = parts[1].strip().strip('"\'')
                            print(f"✓ User ID extraído: {user_id}")
                            return user_id
            
            print(f"✗ No se pudo extraer user_id")
            print(f"Respuesta: {text[:200]}...")
                
    except requests.exceptions.RequestException as e:
        print(f"✗ Error en la petición HTTP: {e}")
    except Exception as e:
        print(f"✗ Error inesperado: {e}")
    
    return None

def execute_federated_script(user_id: str) -> bool:
    """Ejecuta el script federado con el user_id"""
    try:
        print(f"Ejecutando: {PYTHON_SCRIPT} --user-id {user_id}")
        
        # Cambiar al directorio del script si es necesario
        import os
        script_dir = os.path.dirname(os.path.abspath(PYTHON_SCRIPT))
        if script_dir:
            os.chdir(script_dir)
        
        # Ejecutar el script permitiendo que los logs fluyan en tiempo real
        result = subprocess.run(
            [sys.executable, "-u", PYTHON_SCRIPT, "--user-id", user_id],
            text=True
        )
        
        if result.returncode == 0:
            print(f"✓ Script ejecutado exitosamente")
            return True
        else:
            print(f"✗ Script falló con código: {result.returncode}")
            return False
            
    except FileNotFoundError:
        print(f"✗ Archivo no encontrado: {PYTHON_SCRIPT}")
        return False
    except Exception as e:
        print(f"✗ Error ejecutando script: {e}")
        return False

def main_simple():
    """Versión simple: una sola ejecución"""
    print("=" * 50)
    print("EJECUTOR FEDERADO SIMPLIFICADO")
    print("=" * 50)
    
    user_id = get_user_id()
    
    if user_id:
        success = execute_federated_script(user_id)
        if success:
            print("\n✅ Proceso completado exitosamente")
            sys.exit(0)
        else:
            print("\n❌ El script federado falló")
            sys.exit(1)
    else:
        print("\n❌ No se pudo obtener user_id")
        sys.exit(1)

def main_multiple(times: int = 1, delay: float = 1.0):
    """Versión múltiple: ejecutar varias veces"""
    print("=" * 50)
    print(f"EJECUTOR FEDERADO - {times} EJECUCIONES")
    print("=" * 50)
    
    successes = 0
    failures = 0
    
    for i in range(times):
        print(f"\n--- Ejecución {i+1}/{times} ---")
        
        user_id = get_user_id()
        
        if user_id:
            if execute_federated_script(user_id):
                successes += 1
            else:
                failures += 1
        else:
            failures += 1
        
        # Esperar entre ejecuciones si no es la última
        if i < times - 1 and delay > 0:
            print(f"Esperando {delay} segundos...")
            time.sleep(delay)
    
    # Mostrar resumen
    print("\n" + "=" * 50)
    print("RESUMEN FINAL")
    print("=" * 50)
    print(f"Total ejecuciones: {times}")
    print(f"Exitosas: {successes}")
    print(f"Fallidas: {failures}")
    
    if successes > 0:
        print(f"\n✅ {successes} ejecuciones completadas exitosamente")
    if failures > 0:
        print(f"❌ {failures} ejecuciones fallaron")

if __name__ == "__main__":
    # Verificar si se pasó argumento para múltiples ejecuciones
    if len(sys.argv) > 1:
        try:
            times = int(sys.argv[1])
            delay = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
            main_multiple(times, delay)
        except ValueError:
            print("Uso: python run_federated.py [veces] [delay_opcional]")
            print("Ejemplo: python run_federated.py 5 2.0")
            sys.exit(1)
    else:
        main_simple()