import argparse
import subprocess
import sys
import time
import signal
import os

def main():
    parser = argparse.ArgumentParser(description="Ejecutar múltiples clientes federados simultáneamente.")
    parser.add_argument("-n", "--num_clients", type=int, default=1, help="Número de clientes a ejecutar")
    parser.add_argument("--script", type=str, default="main_federated.py", help="Nombre del script del cliente (por defecto: main_federated.py)")
    
    args = parser.parse_args()
    
    # Obtener la ruta absoluta del script a ejecutar
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, args.script)
    
    if not os.path.exists(script_path):
        print(f"Error: No se encontró el script en {script_path}")
        return

    processes = []

    def cleanup_processes(signum, frame):
        print("\n\n🛑 Deteniendo todos los clientes...")
        for p in processes:
            if p.poll() is None:  # Si el proceso sigue corriendo
                p.terminate()
                try:
                    p.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    p.kill()
        print("✅ Todos los clientes detenidos.")
        sys.exit(0)

    # Registrar el manejador de señales para Ctrl+C
    signal.signal(signal.SIGINT, cleanup_processes)

    print(f"🚀 Iniciando {args.num_clients} instancias de cliente...")
    print(f"📜 Script: {script_path}")
    print("Prepara Ctrl+C para detener la ejecución.\n")

    try:
        for i in range(args.num_clients):
            print(f"🔸 Lanzando cliente {i+1}/{args.num_clients}...")
            # Usar subprocess.Popen para lanzar el proceso en "paralelo" (como en otra terminal)
            # sys.executable asegura que usamos el mismo intérprete de Python
            proc = subprocess.Popen([sys.executable, script_path])
            processes.append(proc)
            
            # Pequeña pausa para no saturar el servidor de registro de golpe
            time.sleep(1)

        print(f"\n✅ {len(processes)} clientes corriendo en segundo plano.")
        print("Monitoreando procesos... (Ctrl+C para salir)")
        
        # Mantener el script principal vivo mientras los hijos corren
        while any(p.poll() is None for p in processes):
            time.sleep(1)
            
        print("Todos los procesos han terminado por su cuenta.")

    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        cleanup_processes(None, None)

if __name__ == "__main__":
    main()
