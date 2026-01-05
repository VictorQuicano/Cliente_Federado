#!/bin/bash

# Script: ejecutar_federated.sh
# Uso: ./ejecutar_federated.sh <veces>

# Verificar que se pasen los argumentos necesarios
if [ $# -ne 1 ]; then
    echo "Uso: $0 <número_de_ejecuciones>"
    echo "Ejemplo: $0 5"
    exit 1
fi

NUM_VECES=$1
API_URL="http://34.151.202.191:8082/get_user"

echo "Ejecutando $NUM_VECES veces para la API: $API_URL"

for ((i=1; i<=NUM_VECES; i++))
do
    echo "--- Iteración $i ---"
    
    # Hacer la petición GET a la API
    RESPONSE=$(curl -s -X GET "$API_URL")
    
    # Verificar si curl fue exitoso
    if [ $? -ne 0 ]; then
        echo "Error al hacer la petición a la API"
        continue
    fi
    
    # Mostrar la respuesta para debugging
    echo "Respuesta cruda: $RESPONSE"
    
    # Extraer el user_id de la respuesta - método más robusto
    # Opción 1: Usar grep con expresión más flexible
    USER_ID=$(echo "$RESPONSE" | grep -o '"user_id":"[^"]*"' | cut -d'"' -f4)
    
    # Si no funciona, probar con jq si está disponible
    if [ -z "$USER_ID" ]; then
        if command -v jq &> /dev/null; then
            USER_ID=$(echo "$RESPONSE" | jq -r '.user_id')
        fi
    fi
    
    # Si todavía no funciona, usar sed
    if [ -z "$USER_ID" ]; then
        USER_ID=$(echo "$RESPONSE" | sed -n 's/.*"user_id":"\([^"]*\)".*/\1/p')
    fi
    
    # Verificar si se obtuvo un user_id
    if [ -z "$USER_ID" ]; then
        echo "No se pudo extraer user_id de la respuesta: $RESPONSE"
        continue
    fi
    
    echo "User ID obtenido: $USER_ID"
    
    # Ejecutar el comando Python con el user_id
    python main_federated.py --user-id "$USER_ID"
    
    # Esperar un segundo entre ejecuciones (opcional)
    sleep 1
done

echo "Proceso completado"