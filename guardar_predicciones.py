import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

# Inicializar Firebase con tu archivo JSON de la cuenta de servicio
cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred)

# Cliente Firestore
db = firestore.client()

def guardar_prediccion_firestore(inputs, outputs):
    doc_ref = db.collection("predicciones").document()  # crea un nuevo documento con ID automático
    doc_ref.set({
        "timestamp": datetime.utcnow(),
        "inputs": inputs,
        "outputs": outputs
    })
