# -*- coding: utf-8 -*-
"""
================================================================================
proyecto_ml_completo.py  —  Pipeline TinyML Edge AI
Sistema Inteligente de Riego — Parques Urbanos
================================================================================
Entrena un Decision Tree ultra-ligero (max_depth=4) con 3 features del sensor:
  - Soil Moisture        (humedad de suelo, %)
  - Air temperature (C)  (temperatura del aire, °C)
  - Air humidity (%)     (humedad relativa del aire, %)

Exporta el modelo como código C puro en modelo_edge.h para flashear al ESP32.
El ESP32 toma decisiones 100% offline — sin internet, sin latencia.
================================================================================
"""

import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ── 1. CARGA DEL DATASET ─────────────────────────────────────────────────────
DATASET_PATH = r"C:\Users\John\Downloads\archive\TARP.csv"

print("=" * 65)
print("  PIPELINE TINYML -- SISTEMA DE RIEGO EDGE AI (ESP32)")
print("=" * 65)

df = pd.read_csv(DATASET_PATH)
print(f"\n[INFO] Dataset cargado  : {df.shape[0]} filas x {df.shape[1]} columnas")
print(f"[INFO] Columnas         : {list(df.columns)}")

# ── 2. SELECCION ESTRICTA DE FEATURES (solo sensores del nodo IoT) ───────────
FEATURES = ["Soil Moisture", "Air temperature (C)", "Air humidity (%)"]
TARGET   = "Status"

X = df[FEATURES].copy()
y = df[TARGET].map({"ON": 1, "OFF": 0})

missing = y.isnull().sum()
if missing > 0:
    raise ValueError(f"Hay {missing} valores nulos en Status tras el mapeo ON/OFF.")

print(f"\n[INFO] Features usadas  : {FEATURES}")
print(f"[INFO] Muestras ON  (1) : {y.sum()}")
print(f"[INFO] Muestras OFF (0) : {(y == 0).sum()}")

# ── 3. SPLIT ENTRENAMIENTO / PRUEBA ──────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
print(f"\n[INFO] Train : {len(X_train)} muestras")
print(f"[INFO] Test  : {len(X_test)} muestras")

# ── 4. ENTRENAMIENTO — DECISION TREE max_depth=4 ─────────────────────────────
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc    = accuracy_score(y_test, y_pred)

sep = "-" * 55
print(f"\n{sep}")
print("  RESULTADOS -- Decision Tree (max_depth=4)")
print(sep)
print(f"  Accuracy : {acc:.4f}  ({acc * 100:.2f} %)")
print()
print(classification_report(y_test, y_pred, target_names=["OFF (0)", "ON  (1)"]))

cm = confusion_matrix(y_test, y_pred)
print(f"  Matriz de confusion:\n{cm}\n")
print(f"  Nodos del arbol  : {clf.tree_.node_count}  (ultra-ligero para microcontrolador)")


# ── 5. EXPORTAR A C HEADER — modelo_edge.h ───────────────────────────────────
def _sanitize(name: str) -> str:
    """Convierte un nombre de feature a identificador C valido."""
    return (
        name.replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("%", "pct")
            .replace("/", "_")
            .replace("-", "_")
    )


def export_tree_to_c(tree_clf, feature_names: list, output_path: str = "modelo_edge.h"):
    """
    Recorre el arbol de decision y genera codigo C puro (funcion predict inline)
    compatible con Arduino / ESP32 sin ninguna dependencia externa.
    """
    tree_   = tree_clf.tree_
    c_names = [_sanitize(f) for f in feature_names]

    lines = []
    lines += [
        "/*",
        " * modelo_edge.h",
        " * Sistema Inteligente de Riego -- Parques Urbanos",
        " * Generado automaticamente por proyecto_ml_completo.py",
        " *",
        f" * Algoritmo : Decision Tree Classifier (max_depth={tree_clf.max_depth})",
        f" * Accuracy  : {acc * 100:.2f} %   (test 30 %)",
        f" * Nodos     : {tree_.node_count}",
        f" * Features  : Soil Moisture | Air temperature | Air humidity",
        " * Clases    : 0 = OFF (no regar)  |  1 = ON (activar valvula)",
        " *",
        " * -- USO EN ESP32 (Arduino IDE / PlatformIO) ----------------------",
        " * #include \"modelo_edge.h\"",
        " *",
        " * void loop() {",
        " *   float sm   = leerHumedadSuelo();   // 0-100 %",
        " *   float temp = dht.readTemperature();",
        " *   float hum  = dht.readHumidity();",
        " *",
        " *   int decision = predict(sm, temp, hum);",
        " *   digitalWrite(VALVE_PIN, decision);  // 1=abierta  0=cerrada",
        " * }",
        " */",
        "",
        "#ifndef MODELO_EDGE_H",
        "#define MODELO_EDGE_H",
        "",
        "// -- Metadatos del modelo -------------------------------------------",
        f"#define MODEL_N_FEATURES  {len(feature_names)}",
        f"#define MODEL_MAX_DEPTH   {tree_clf.max_depth}",
        f"#define MODEL_N_NODES     {tree_.node_count}",
        f"#define MODEL_ACCURACY    {acc:.6f}f",
        "",
        "// -- Indices de features --------------------------------------------",
    ]
    for i, cname in enumerate(c_names):
        lines.append(f"#define IDX_{cname.upper():<30} {i}")

    lines += [
        "",
        "/**",
        " * @brief Infiere si se debe activar el riego (Edge AI, 100 % offline).",
        f" * @param {c_names[0]:<32} Humedad del suelo (0-100 %)",
        f" * @param {c_names[1]:<32} Temperatura del aire (grados C)",
        f" * @param {c_names[2]:<32} Humedad relativa del aire (0-100 %)",
        " * @return  1 = abrir valvula (ON)  |  0 = cerrar valvula (OFF)",
        " */",
        "inline int predict(" + ", ".join(f"float {n}" for n in c_names) + ") {",
    ]

    def recurse(node: int, depth: int):
        pad = "    " * (depth + 1)
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            cname     = _sanitize(feature_names[tree_.feature[node]])
            threshold = tree_.threshold[node]
            lines.append(f"{pad}if ({cname} <= {threshold:.6f}f) {{")
            recurse(tree_.children_left[node],  depth + 1)
            lines.append(f"{pad}}} else {{")
            recurse(tree_.children_right[node], depth + 1)
            lines.append(f"{pad}}}")
        else:
            values     = tree_.value[node][0]
            pred_cls   = int(np.argmax(values))
            label      = "ON " if pred_cls == 1 else "OFF"
            confidence = int(values[pred_cls] / values.sum() * 100)
            lines.append(f"{pad}return {pred_cls};  // {label}  (confianza {confidence}%)")

    recurse(0, 0)

    lines += [
        "}",
        "",
        "#endif  // MODELO_EDGE_H",
    ]

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    size = os.path.getsize(output_path)
    print(f"[OK] Modelo exportado  --> {output_path}  ({size} bytes)")
    print(f"     Copia modelo_edge.h a la carpeta de tu sketch Arduino/PlatformIO.")


export_tree_to_c(clf, FEATURES, output_path="modelo_edge.h")

# ── Intentar tambien con micromlgen si esta instalado ────────────────────────
try:
    from micromlgen import port
    micromlgen_code = port(clf, classmap={0: "OFF", 1: "ON"})
    with open("modelo_edge_micromlgen.h", "w", encoding="utf-8") as f:
        f.write(micromlgen_code)
    print("[OK] Version micromlgen --> modelo_edge_micromlgen.h")
except ImportError:
    print("[INFO] micromlgen no instalado -- modelo_edge.h (manual) es suficiente.")
    print("       Para instalar: pip install micromlgen")

print("\n" + "=" * 65)
print("  PIPELINE COMPLETADO -- modelo_edge.h listo para el ESP32")
print("=" * 65 + "\n")
