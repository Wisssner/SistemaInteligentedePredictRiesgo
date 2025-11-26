from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Librerías de Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score, roc_auc_score,
                             roc_curve, precision_recall_curve, average_precision_score)

# Modelos de Machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               AdaBoostClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# SMOTE para balanceo de clases
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Para guardar modelos
import joblib
import pickle
from datetime import datetime

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Crear carpetas usando BASE
BASE = Path("C:/2025/CURSOS UNMSM/CICLO 8/SI/SI - PROYECTO FINAL/SistemaInteligentedePredicci-ndeRiego")
visual_dir = BASE / "visualizaciones1"
models_dir = BASE / "modelos_guardados"

visual_dir.mkdir(exist_ok=True)
models_dir.mkdir(exist_ok=True)

print("="*80)
print("FASE 1: COMPRENSIÓN DEL NEGOCIO (Business Understanding)")
print("="*80)
print("""
OBJETIVO: Predecir el resultado (éxito/fracaso) de cultivos basándose en:
- Tipo de cultivo (crop ID)
- Tipo de suelo (soil_type)
- Etapa de plántula (Seedling Stage)
- MOI (Índice de humedad del suelo)
- Temperatura
- Humedad ambiental

PROBLEMA: Clasificación binaria (result: 0 = fracaso, 1 = éxito)
""")

print("\n" + "="*80)
print("FASE 2: COMPRENSIÓN DE LOS DATOS (Data Understanding)")
print("="*80)
archivo = BASE / "dataSalvadora.xlsx"
df = pd.read_excel(archivo)

print(f"\n📊 Dimensiones del dataset: {df.shape[0]} filas x {df.shape[1]} columnas")
print(f"\n📋 Columnas: {df.columns.tolist()}")
print(f"\n🔍 Tipos de datos:\n{df.dtypes}")
print(f"\n📈 Primeras filas del dataset:")
print(df.head(10))

# Información estadística
print("\n" + "="*80)
print("ESTADÍSTICAS DESCRIPTIVAS")
print("="*80)
print(df.describe())

# Valores únicos en variables categóricas
print("\n📌 Valores únicos en variables categóricas:")
for col in ['crop ID', 'soil_type', 'Seedling Stage']:
    print(f"\n{col}: {df[col].nunique()} valores únicos")
    print(df[col].value_counts())

# Distribución de la variable objetivo
print("\n🎯 Distribución de la variable objetivo (result):")
print(df['result'].value_counts())
print(f"\nProporción:")
print(df['result'].value_counts(normalize=True))

# Valores nulos
print("\n❓ Valores nulos por columna:")
print(df.isnull().sum())

print("\n" + "="*80)
print("FASE 3: PREPARACIÓN DE LOS DATOS (Data Preparation)")
print("="*80)

# Copiar dataset original
df_original = df.copy()

# 3.1 Limpieza de datos
print("\n🧹 Paso 3.1: Limpieza de datos")
print(f"\nOutliers detectados en humidity: {len(df[df['humidity'] > 200])} registros")
print(f"Rango actual de humidity: [{df['humidity'].min()}, {df['humidity'].max()}]")

# Limitar humidity a valores razonables (0-100%)
df['humidity_cleaned'] = df['humidity'].clip(upper=100)
print(f"Rango corregido de humidity: [{df['humidity_cleaned'].min()}, {df['humidity_cleaned'].max()}]")

# 3.2 Codificación de variables categóricas
print("\n🔤 Paso 3.2: Codificación de variables categóricas")

# Label Encoding para variables categóricas
le_crop = LabelEncoder()
le_soil = LabelEncoder()
le_seedling = LabelEncoder()

df['crop_encoded'] = le_crop.fit_transform(df['crop ID'])
df['soil_encoded'] = le_soil.fit_transform(df['soil_type'])
df['seedling_encoded'] = le_seedling.fit_transform(df['Seedling Stage'])

# 3.3 Ingeniería de características
print("\n⚙️ Paso 3.3: Ingeniería de características")

df['temp_humidity_ratio'] = df['temp'] / (df['humidity_cleaned'] + 1)
df['moi_temp_interaction'] = df['MOI'] * df['temp']
df['temp_squared'] = df['temp'] ** 2
df['humidity_squared'] = df['humidity_cleaned'] ** 2
df['moi_squared'] = df['MOI'] ** 2

print("Nuevas características creadas:")
print("- temp_humidity_ratio: Relación temperatura/humedad")
print("- moi_temp_interaction: Interacción MOI x temperatura")
print("- temp_squared, humidity_squared, moi_squared: Términos cuadráticos")

# Seleccionar características finales
feature_cols = ['crop_encoded', 'soil_encoded', 'seedling_encoded', 'MOI', 
                'temp', 'humidity_cleaned', 'temp_humidity_ratio', 
                'moi_temp_interaction', 'temp_squared', 'humidity_squared', 'moi_squared']

X = df[feature_cols]
y = df['result']

print(f"\n✅ Dataset preparado con {X.shape[1]} características")

print("\n" + "="*80)
print("EXPLORACIÓN VISUAL DE DATOS (EDA)")
print("="*80)

# Crear gráficos y guardarlos en visualizaciones/
# 1. Matriz de correlación
plt.figure(figsize=(14, 10))
correlation_matrix = df[['MOI', 'temp', 'humidity_cleaned', 'result', 
                         'temp_humidity_ratio', 'moi_temp_interaction']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Matriz de Correlación', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(visual_dir / "01_correlation_matrix.png", dpi=300, bbox_inches='tight')
plt.close()

# 2. Distribución de la variable objetivo
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
df['result'].value_counts().plot(kind='bar', ax=axes[0], color=['#e74c3c', '#2ecc71'])
axes[0].set_title('Distribución de Result (Conteo)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Result')
axes[0].set_ylabel('Frecuencia')
axes[0].set_xticklabels(['Fracaso (0)', 'Éxito (1)'], rotation=0)

df['result'].value_counts(normalize=True).plot(kind='pie', ax=axes[1], 
                                                autopct='%1.1f%%', 
                                                colors=['#e74c3c', '#2ecc71'])
axes[1].set_title('Proporción de Result', fontsize=14, fontweight='bold')
axes[1].set_ylabel('')
plt.tight_layout()
plt.savefig(visual_dir / "02_target_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

# Guardar otros gráficos de EDA en visualizaciones/

print("\n" + "="*80)
print("FASE 4: MODELADO (Modeling)")
print("="*80)

# División del dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                      random_state=42, stratify=y)

# Escalado de características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Verificar si se necesita SMOTE
class_distribution = y_train.value_counts()
minority_class = class_distribution.min()
majority_class = class_distribution.max()
imbalance_ratio = majority_class / minority_class

use_smote = imbalance_ratio > 1.5

if use_smote:
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
else:
    X_train_resampled = X_train_scaled
    y_train_resampled = y_train

# Entrenamiento de modelos
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
}

# Evaluar y almacenar resultados
results = []
trained_models = {}

# Entrenamiento y evaluación de los modelos
for name, model in models.items():
    model.fit(X_train_resampled, y_train_resampled)
    
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    if y_pred_proba is not None:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
    else:
        roc_auc = np.nan
        avg_precision = np.nan
    
    cv_scores = cross_val_score(model, X_train_resampled, y_train_resampled, 
                                cv=5, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc,
        'Avg_Precision': avg_precision,
        'CV_Mean': cv_mean,
        'CV_Std': cv_std
    })
    
    trained_models[name] = {
        'model': model,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

# Guardar resultados de la comparación de modelos
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('F1-Score', ascending=False)
results_df.to_csv(models_dir / 'model_comparison_results.csv', index=False)

# Guardar el mejor modelo
best_model_name = results_df.iloc[0]['Model']
best_model = trained_models[best_model_name]['model']
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_filename = models_dir / f"best_model_{best_model_name.replace(' ', '_')}_{timestamp}.pkl"
joblib_filename = models_dir / f"best_model_{best_model_name.replace(' ', '_')}_{timestamp}.joblib"

model_package = {
    'model': best_model,
    'scaler': scaler,
    'label_encoders': {
        'crop': le_crop,
        'soil': le_soil,
        'seedling': le_seedling
    },
    'feature_names': feature_cols,
    'model_name': best_model_name,
    'metrics': {
        'accuracy': results_df.iloc[0]['Accuracy'],
        'f1_score': results_df.iloc[0]['F1-Score'],
        'precision': results_df.iloc[0]['Precision'],
        'recall': results_df.iloc[0]['Recall']
    },
    'timestamp': timestamp
}

with open(model_filename, 'wb') as f:
    pickle.dump(model_package, f)
joblib.dump(model_package, joblib_filename)

# Guardar función de predicción
prediction_script = f"""
import pickle
import numpy as np

def predict_crop_result(model_path, crop_id, soil_type, seedling_stage, moi, temp, humidity):
    # Código de predicción aquí...
"""

with open(models_dir / "prediccion_facil.py", 'w') as f:
    f.write(prediction_script)

print("✅ Archivos generados y guardados en las carpetas correspondientes.")
