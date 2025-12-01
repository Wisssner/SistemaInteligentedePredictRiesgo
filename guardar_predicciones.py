import requests
from datetime import datetime

# Tu URL base de Realtime Database
FIREBASE_RTD_URL = "https://si-grupo2-default-rtdb.firebaseio.com/predicciones.json"

def guardar_prediccion_firestore(inputs, outputs):
    data = {
        "timestamp": datetime.utcnow().isoformat(),
        "inputs": inputs,
        "outputs": outputs
    }

    try:
        response = requests.post(FIREBASE_RTD_URL, json=data)
        if response.status_code == 200:
            print("✔️ Predicción guardada en Realtime Database")
            return True
        else:
            print("❌ Error al guardar:", response.text)
            return False
    except Exception as e:
        print("❌ Excepción:", e)
        return False
