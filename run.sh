#!/bin/bash
# crear_servicio.sh - Crea un servicio systemd para main.py

SCRIPT_NAME="run.py"
SERVICE_NAME="cliente_federado"
USER=$(whoami)
WORK_DIR="$PWD"
PYTHON_PATH="/opt/conda/envs/models_env/bin/python"

# Verificar que main.py existe
if [ ! -f "$SCRIPT_NAME" ]; then
    echo "Error: $SCRIPT_NAME no encontrado en el directorio actual"
    exit 1
fi

# Crear servicio systemd
sudo tee /etc/systemd/system/$SERVICE_NAME.service > /dev/null <<EOF
[Unit]
Description=Servicio Python: $SCRIPT_NAME
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$WORK_DIR
Environment=PYTHONUNBUFFERED=1
ExecStart=$PYTHON_PATH -u $WORK_DIR/$SCRIPT_NAME 1
Restart=always
RestartSec=10
StandardOutput=append:$WORK_DIR/$SERVICE_NAME.log
StandardError=append:$WORK_DIR/$SERVICE_NAME.error.log

[Install]
WantedBy=multi-user.target
EOF

# Recargar systemd y habilitar servicio
sudo systemctl daemon-reload
sudo systemctl enable $SERVICE_NAME
sudo systemctl start $SERVICE_NAME

echo "✅ Servicio creado: $SERVICE_NAME"
echo "📁 Directorio: $WORK_DIR"
echo ""
echo "📋 COMANDOS ÚTILES:"
echo "   Ver estado:    sudo systemctl status $SERVICE_NAME"
echo "   Ver logs:      sudo journalctl -u $SERVICE_NAME -f"
echo "   Detener:       sudo systemctl stop $SERVICE_NAME"
echo "   Iniciar:       sudo systemctl start $SERVICE_NAME"
echo "   Reiniciar:     sudo systemctl restart $SERVICE_NAME"
