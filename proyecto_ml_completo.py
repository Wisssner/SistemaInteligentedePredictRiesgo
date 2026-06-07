# -*- coding: utf-8 -*-
"""
================================================================================
SISTEMA INTELIGENTE DE RIEGO - Vaccinium corymbosum (Arandano Alto)
Metodologia CRISP-DM completa
Costa de Lima, Peru
================================================================================
"""

import sys, io
# Forzar UTF-8 en la salida estandar (necesario en terminales Windows cp1252)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE

import joblib
import pickle
from datetime import datetime

# ---------------------------------------------------------------------------
# RUTAS
# ---------------------------------------------------------------------------
BASE = Path(__file__).parent
VISUAL_DIR = BASE / "visualizaciones1"
MODELS_DIR = BASE / "modelos_guardados1"
VISUAL_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# FASE 1: BUSINESS UNDERSTANDING
# ---------------------------------------------------------------------------
print("=" * 80)
print("FASE 1: BUSINESS UNDERSTANDING")
print("=" * 80)
print("""
CULTIVO OBJETIVO : Arándano alto (Vaccinium corymbosum)
REGION           : Costa de Lima, Peru
PROBLEMA         : Clasificacion binaria — Regar (1) / No Regar (0)
UMBRALES CRITICOS (arandano):
  - MOI optimo   : 60-80 % capacidad de campo
  - Temp optima  : 18-24 °C
  - Potencial matrico: -20 a -30 kPa
MODELOS          : Gradient Boosting (principal) + Random Forest (validador)
DATASET          : 15 288 registros con ingenieria de features avanzada
PARTICION        : 70 % entrenamiento / 30 % prueba (estratificado por etapa)
""")

# ---------------------------------------------------------------------------
# FASE 2: DATA UNDERSTANDING
# ---------------------------------------------------------------------------
print("=" * 80)
print("FASE 2: DATA UNDERSTANDING")
print("=" * 80)

df = pd.read_excel(BASE / "dataSalvadora.xlsx")
print(f"Dimensiones originales : {df.shape[0]} filas x {df.shape[1]} columnas")
print(f"Columnas               : {df.columns.tolist()}")
print(f"\nDistribucion target (original):")
print(df['result'].value_counts())
print(f"\nEtapas originales ({df['Seedling Stage'].nunique()}):")
print(df['Seedling Stage'].value_counts().to_string())
print(f"\nHumidity outliers (>100): {len(df[df['humidity'] > 100])}")

# ---------------------------------------------------------------------------
# FASE 3: DATA PREPARATION
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("FASE 3: DATA PREPARATION")
print("=" * 80)

# ---- 3.1  Limpieza ----
df['humidity_cleaned'] = df['humidity'].clip(upper=100)
print(f"\n[3.1] Outliers de humidity corregidos. Rango: [{df['humidity_cleaned'].min()}, {df['humidity_cleaned'].max()}]")

# ---- 3.2  Remapeo de etapas fenologicas a las 4 del arandano ----
STAGE_MAP = {
    'Germination'                                    : 'Germinacion',
    'Seedling Stage'                                 : 'Germinacion',
    'Vegetative Growth / Root or Tuber Development'  : 'Desarrollo_Vegetativo',
    'Flowering'                                      : 'Floracion',
    'Pollination'                                    : 'Floracion',
    'Fruit/Grain/Bulb Formation'                     : 'Fructificacion',
    'Maturation'                                     : 'Fructificacion',
    'Harvest'                                        : 'Fructificacion',
}

# Kc (coeficiente de cultivo) del arandano por etapa fenologica
# Fuente: FAO-56 adaptado para Vaccinium corymbosum
KC_ARANDANO = {
    'Germinacion'         : 0.30,
    'Desarrollo_Vegetativo': 0.70,
    'Floracion'           : 1.05,
    'Fructificacion'      : 0.90,
}

df['etapa_fenologica'] = df['Seedling Stage'].map(STAGE_MAP)
print(f"\n[3.2] Remapeo de etapas fenologicas (8 -> 4):")
print(df['etapa_fenologica'].value_counts().to_string())

# ---- 3.3  Calculo de ETc (Hargreaves-Samani simplificado) ----
# ETc = Kc * ETo
# ETo aprox Hargreaves (sin datos de radiacion solar diaria completos):
#   ETo ≈ 0.0023 * (T_media + 17.8) * Ra_norm
# Ra_norm se estima para latitud Lima (-12°) como funcion simplificada de la
# temperatura media (proxy) porque no contamos con datos de radiacion.
# Formula simplificada para escenario sin estacion meteorologica completa:
#   ETo_proxy = 0.408 * (temp - 2.0) / 25.0 * (1 + temp / 50.0)
# Calibrada para el rango de la costa limeña (13-46°C del dataset).

def calcular_etc(temp: np.ndarray, etapa: pd.Series) -> np.ndarray:
    """
    Calcula la Evapotranspiracion del cultivo (ETc) en mm/dia.
    Metodo: Hargreaves-Samani simplificado × Kc FAO-56 para arandano.
    """
    eto_proxy = 0.408 * np.maximum(0.0, temp - 2.0) / 25.0 * (1 + temp / 50.0)
    kc = etapa.map(KC_ARANDANO).fillna(0.70).values
    return np.round(eto_proxy * kc, 4)

df['etc'] = calcular_etc(df['temp'].values, df['etapa_fenologica'])
print(f"\n[3.3] ETc calculado. Stats: min={df['etc'].min():.4f}  mean={df['etc'].mean():.4f}  max={df['etc'].max():.4f}")

# ---- 3.4  Ingenieria de caracteristicas ----
# Existentes (preservadas del pipeline anterior)
df['temp_humidity_ratio']  = df['temp'] / (df['humidity_cleaned'] + 1)
df['moi_temp_interaction']  = df['MOI'] * df['temp']
df['temp_squared']          = df['temp'] ** 2
df['humidity_squared']      = df['humidity_cleaned'] ** 2
df['moi_squared']           = df['MOI'] ** 2

# Nuevas — especificas para arandano
df['moi_deficit']           = np.maximum(0.0, 60.0 - df['MOI'])       # deficit respecto al umbral inferior del arandano
df['moi_exceso']            = np.maximum(0.0, df['MOI'] - 80.0)       # exceso respecto al umbral superior
df['moi_etc_interaction']   = df['MOI'] * df['etc']                   # interaccion demanda-suministro
df['etc_humidity_ratio']    = df['etc'] / (df['humidity_cleaned'] + 1) # demanda vs humedad ambiental

print("\n[3.4] Features creadas:")
nuevas = ['moi_deficit','moi_exceso','moi_etc_interaction','etc_humidity_ratio',
          'temp_humidity_ratio','moi_temp_interaction','temp_squared','humidity_squared','moi_squared']
for f in nuevas:
    print(f"  + {f}")

# ---- 3.5  Codificacion de variables categoricas ----
le_crop     = LabelEncoder()
le_soil     = LabelEncoder()
le_seedling = LabelEncoder()  # usa las 4 etapas del arandano

df['crop_encoded']     = le_crop.fit_transform(df['crop ID'])
df['soil_encoded']     = le_soil.fit_transform(df['soil_type'])
df['seedling_encoded'] = le_seedling.fit_transform(df['etapa_fenologica'])

print(f"\n[3.5] Codificacion etapas (4 clases arándano):")
for i, cls in enumerate(le_seedling.classes_):
    print(f"  {i} → {cls}")

# ---- 3.6  Seleccion de features finales ----
FEATURE_COLS = [
    'crop_encoded', 'soil_encoded', 'seedling_encoded',
    'MOI', 'temp', 'humidity_cleaned',
    'etc',
    'temp_humidity_ratio', 'moi_temp_interaction',
    'temp_squared', 'humidity_squared', 'moi_squared',
    'moi_deficit', 'moi_exceso',
    'moi_etc_interaction', 'etc_humidity_ratio',
]

X = df[FEATURE_COLS]
y = df['result']
etapa_col = df['etapa_fenologica']

print(f"\n[3.6] Dataset preparado: {X.shape[1]} features, {len(y)} registros")

# ---- 3.7  Particion 70/30 estratificada por ETAPA FENOLOGICA ----
# Se crea una clave combinada etapa+target para estratificacion correcta
strat_key = etapa_col + "_" + y.astype(str)

X_train, X_test, y_train, y_test, etapa_train, etapa_test = train_test_split(
    X, y, etapa_col,
    test_size=0.30,
    random_state=42,
    stratify=strat_key
)

print(f"\n[3.7] Particion 70/30 estratificada por etapa fenologica:")
print(f"  Entrenamiento : {len(X_train):>6} registros ({len(X_train)/len(X)*100:.1f}%)")
print(f"  Prueba        : {len(X_test):>6} registros ({len(X_test)/len(X)*100:.1f}%)")
print(f"\n  Distribucion etapas en entrenamiento:")
print(etapa_train.value_counts().to_string())
print(f"\n  Distribucion etapas en prueba:")
print(etapa_test.value_counts().to_string())

# ---- 3.8  Escalado ----
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ---- 3.9  SMOTE si hay desbalance ----
ratio = y_train.value_counts().max() / y_train.value_counts().min()
if ratio > 1.5:
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
    print(f"\n[3.9] SMOTE aplicado (ratio={ratio:.2f}): {len(X_train_bal)} muestras tras balance")
else:
    X_train_bal, y_train_bal = X_train_scaled, y_train
    print(f"\n[3.9] Dataset balanceado (ratio={ratio:.2f}): no requiere SMOTE")

# ---------------------------------------------------------------------------
# FASE 4: MODELADO
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("FASE 4: MODELADO — Gradient Boosting + Random Forest")
print("=" * 80)

MODELS = {
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.08,
        max_depth=5,
        min_samples_split=10,
        subsample=0.85,
        random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ),
    'Decision Tree (baseline)': DecisionTreeClassifier(
        max_depth=8,
        min_samples_split=10,
        random_state=42
    ),
    'Logistic Regression (baseline)': LogisticRegression(
        max_iter=1000,
        random_state=42
    ),
}

results   = []
trained   = {}
skf       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in MODELS.items():
    print(f"\n  Entrenando: {name} ...")
    model.fit(X_train_bal, y_train_bal)

    y_pred      = model.predict(X_test_scaled)
    y_proba     = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan
    ap   = average_precision_score(y_test, y_proba) if y_proba is not None else np.nan

    cv_scores = cross_val_score(model, X_train_bal, y_train_bal, cv=skf, scoring='f1', n_jobs=-1)

    cm   = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    results.append({
        'Model'       : name,
        'Accuracy'    : acc,
        'Precision'   : prec,
        'Recall'      : rec,
        'F1-Score'    : f1,
        'ROC-AUC'     : auc,
        'Avg_Precision': ap,
        'CV_F1_mean'  : cv_scores.mean(),
        'CV_F1_std'   : cv_scores.std(),
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
        'Falsos_Neg'  : fn,
    })
    trained[name] = {'model': model, 'y_pred': y_pred, 'y_proba': y_proba}

    print(f"    Accuracy  : {acc*100:.4f}%")
    print(f"    F1-Score  : {f1:.6f}")
    print(f"    ROC-AUC   : {auc:.6f}")
    print(f"    Falsos neg: {fn}")
    print(f"    CV F1     : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ---------------------------------------------------------------------------
# FASE 5: EVALUACION Y COMPARACION
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("FASE 5: EVALUACION COMPARATIVA")
print("=" * 80)

results_df = pd.DataFrame(results).sort_values('F1-Score', ascending=False)
print("\nTabla comparativa de modelos:")
print(results_df[['Model','Accuracy','Precision','Recall','F1-Score','ROC-AUC','Falsos_Neg','CV_F1_mean']].to_string(index=False))

results_df.to_csv(MODELS_DIR / 'model_comparison_results.csv', index=False)

# Reporte detallado del mejor modelo y de RF
for name in ['Gradient Boosting', 'Random Forest']:
    if name in trained:
        print(f"\n--- Reporte completo: {name} ---")
        print(classification_report(y_test, trained[name]['y_pred'],
                                    target_names=['No Regar (0)', 'Regar (1)']))

# ---------------------------------------------------------------------------
# FASE 6: GUARDADO DEL MODELO
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("FASE 6: GUARDADO DE MODELOS")
print("=" * 80)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

for name in ['Gradient Boosting', 'Random Forest']:
    if name not in trained:
        continue

    model_obj = trained[name]['model']
    row = results_df[results_df['Model'] == name].iloc[0]

    pkg = {
        'model'         : model_obj,
        'scaler'        : scaler,
        'label_encoders': {
            'crop'     : le_crop,
            'soil'     : le_soil,
            'seedling' : le_seedling,
        },
        'feature_names' : FEATURE_COLS,
        'model_name'    : name,
        'cultivo'       : 'Vaccinium corymbosum (Arandano Alto)',
        'kc_arandano'   : KC_ARANDANO,
        'stage_map'     : STAGE_MAP,
        'metrics': {
            'accuracy'  : float(row['Accuracy']),
            'f1_score'  : float(row['F1-Score']),
            'precision' : float(row['Precision']),
            'recall'    : float(row['Recall']),
            'roc_auc'   : float(row['ROC-AUC']),
            'falsos_neg': int(row['Falsos_Neg']),
        },
        'timestamp'     : timestamp,
        'split'         : '70/30 estratificado por etapa fenologica',
        'n_features'    : len(FEATURE_COLS),
    }

    safe_name = name.replace(' ', '_')
    pkl_path  = MODELS_DIR / f"best_model_{safe_name}_arandano_{timestamp}.pkl"
    jbl_path  = MODELS_DIR / f"best_model_{safe_name}_arandano_{timestamp}.joblib"

    with open(pkl_path, 'wb') as f:
        pickle.dump(pkg, f)
    joblib.dump(pkg, jbl_path)

    print(f"  Guardado: {pkl_path.name}")
    print(f"  Guardado: {jbl_path.name}")

# ---------------------------------------------------------------------------
# VISUALIZACIONES
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("VISUALIZACIONES")
print("=" * 80)

# 1. Comparacion de modelos
fig, ax = plt.subplots(figsize=(10, 5))
modelos_plot = results_df['Model'].tolist()
f1_vals  = results_df['F1-Score'].tolist()
colors   = ['#2ecc71' if 'Gradient' in m or 'Random' in m else '#95a5a6' for m in modelos_plot]
bars = ax.barh(modelos_plot, f1_vals, color=colors)
ax.set_xlim(0.7, 1.01)
ax.set_xlabel('F1-Score')
ax.set_title('Comparacion de Modelos — Arandano (Vaccinium corymbosum)', fontweight='bold')
for bar, val in zip(bars, f1_vals):
    ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
            f'{val:.4f}', va='center', fontsize=10)
plt.tight_layout()
plt.savefig(VISUAL_DIR / '03_models_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# 2. Matriz de confusion del mejor modelo (GB)
if 'Gradient Boosting' in trained:
    cm = confusion_matrix(y_test, trained['Gradient Boosting']['y_pred'])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Regar', 'Regar'],
                yticklabels=['No Regar', 'Regar'])
    ax.set_xlabel('Predicho')
    ax.set_ylabel('Real')
    ax.set_title('Matriz de Confusion — Gradient Boosting (Arandano)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(VISUAL_DIR / '04_best_model_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

# 3. Importancia de features (GB)
if 'Gradient Boosting' in trained:
    gb_model = trained['Gradient Boosting']['model']
    importances = pd.Series(gb_model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    importances.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title('Importancia de Features — Gradient Boosting', fontweight='bold')
    ax.set_xlabel('Importancia')
    plt.tight_layout()
    plt.savefig(VISUAL_DIR / '05_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()

# 4. Distribucion ETc por etapa
fig, ax = plt.subplots(figsize=(9, 4))
for etapa in df['etapa_fenologica'].unique():
    vals = df.loc[df['etapa_fenologica'] == etapa, 'etc']
    ax.hist(vals, bins=30, alpha=0.6, label=etapa)
ax.set_xlabel('ETc (mm/dia)')
ax.set_ylabel('Frecuencia')
ax.set_title('Distribucion de ETc por Etapa Fenologica (Arandano)', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(VISUAL_DIR / '06_etc_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

print("  Visualizaciones guardadas en visualizaciones1/")

# ---------------------------------------------------------------------------
# RESUMEN FINAL
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("RESUMEN FINAL DEL PIPELINE")
print("=" * 80)

gb_row = results_df[results_df['Model'] == 'Gradient Boosting']
rf_row = results_df[results_df['Model'] == 'Random Forest']

if not gb_row.empty:
    r = gb_row.iloc[0]
    print(f"\nGradient Boosting (algoritmo principal):")
    print(f"  Accuracy   : {r['Accuracy']*100:.4f} %")
    print(f"  F1-Score   : {r['F1-Score']:.6f}")
    print(f"  ROC-AUC    : {r['ROC-AUC']:.6f}")
    print(f"  Falsos neg : {int(r['Falsos_Neg'])}")
    print(f"  CV F1      : {r['CV_F1_mean']:.4f} +/- {r['CV_F1_std']:.4f}")

if not rf_row.empty:
    r = rf_row.iloc[0]
    print(f"\nRandom Forest (validador complementario):")
    print(f"  Accuracy   : {r['Accuracy']*100:.4f} %")
    print(f"  F1-Score   : {r['F1-Score']:.6f}")
    print(f"  ROC-AUC    : {r['ROC-AUC']:.6f}")
    print(f"  Falsos neg : {int(r['Falsos_Neg'])}")
    print(f"  CV F1      : {r['CV_F1_mean']:.4f} +/- {r['CV_F1_std']:.4f}")

print(f"\nParticion        : 70/30 estratificada por etapa fenologica")
print(f"Etapas arandano  : {list(le_seedling.classes_)}")
print(f"Features totales : {len(FEATURE_COLS)}")
print(f"ETc implementado : Hargreaves-Samani simplificado x Kc FAO-56")
print(f"Modelos guardados: modelos_guardados1/")
print(f"Timestamp        : {timestamp}")
print("\n✓ Pipeline CRISP-DM completado exitosamente.")
