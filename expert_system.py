# -*- coding: utf-8 -*-
"""
================================================================================
expert_system.py  —  Motor de Inferencia Difusa Mamdani
Sistema Inteligente de Riego — Vaccinium corymbosum (Arandano Alto)
Costa de Lima, Peru

Implementacion COMPLETA del ciclo Mamdani:
  1. Fuzzificacion     : funciones trapezoidales y triangulares puras en NumPy
  2. Evaluacion        : 22 reglas SI-ENTONCES con operadores min (AND) / max (OR)
  3. Implicacion       : metodo min (Mamdani clasico)
  4. Agregacion        : union de consecuentes por max
  5. Defuzzificacion   : metodo del centroide (Center of Gravity)

Calibrado especificamente para los umbrales del arandano:
  MOI optimo  : 60-80% capacidad de campo
  Temp optima : 18-24°C  (costa limeña)
  Potencial   : -20 a -30 kPa

Interfaz compatible con la funcion evaluate_expert_system() existente en app.py.
================================================================================
"""

import numpy as np

# ==============================================================================
# 1.  FUNCIONES DE PERTENENCIA
# ==============================================================================

def _trap(x: float, a: float, b: float, c: float, d: float) -> float:
    """
    Funcion de pertenencia trapezoidal.
    Sube de a a b, plana de b a c, baja de c a d.
    """
    x = float(x)
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if a < x < b:
        return (x - a) / (b - a)
    return (d - x) / (d - c)          # c < x < d


def _tri(x: float, a: float, b: float, c: float) -> float:
    """Funcion de pertenencia triangular (caso especial de trapecio con b==c)."""
    return _trap(x, a, b, b, c)


def _trap_vec(xs: np.ndarray, a, b, c, d) -> np.ndarray:
    """Version vectorizada de _trap para el universo de salida."""
    result = np.zeros_like(xs, dtype=float)
    for i, x in enumerate(xs):
        result[i] = _trap(x, a, b, c, d)
    return result


def _tri_vec(xs: np.ndarray, a, b, c) -> np.ndarray:
    """Version vectorizada de _tri."""
    return _trap_vec(xs, a, b, b, c)


# ==============================================================================
# 2.  CONJUNTOS DIFUSOS — ENTRADAS
# ==============================================================================

# ---- MOI  [0, 100] (humedad del suelo) ----
def _moi_sets(moi: float) -> dict:
    """
    Fuzzifica el indice de humedad del suelo (MOI).
    Umbrales calibrados para Vaccinium corymbosum.
    """
    return {
        'muy_seco' : _trap(moi,  0,  0, 15, 30),   # suelo critico, riesgo desecacion
        'seco'     : _tri (moi, 20, 37, 52),        # por debajo del optimo arandano
        'optimo'   : _trap(moi, 52, 60, 78, 86),    # rango ideal 60-80% CC
        'humedo'   : _tri (moi, 78, 88, 96),        # sobre el limite superior
        'saturado' : _trap(moi, 90, 96,100,100),    # asfixia radicular
    }


# ---- Temperatura  [0, 50] °C ----
def _temp_sets(temp: float) -> dict:
    """
    Fuzzifica la temperatura ambiental.
    Rango optimo del arandano en costa limeña: 18-24°C.
    """
    return {
        'fria'   : _trap(temp,  0,  0, 12, 18),
        'fresca' : _tri (temp, 14, 20, 26),
        'optima' : _trap(temp, 18, 20, 24, 28),     # rango optimo arandano
        'calida' : _tri (temp, 25, 32, 40),
        'extrema': _trap(temp, 36, 42, 50, 50),
    }


# ---- Humedad ambiental  [0, 100] % ----
def _hum_sets(hum: float) -> dict:
    """
    Fuzzifica la humedad relativa del aire.
    Influye en la tasa de evapotranspiracion (ETc).
    """
    return {
        'muy_seca' : _trap(hum,  0,  0, 20, 35),
        'seca'     : _tri (hum, 25, 40, 55),
        'moderada' : _tri (hum, 45, 60, 75),
        'alta'     : _tri (hum, 65, 78, 90),
        'muy_alta' : _trap(hum, 85, 92,100,100),
    }


# ---- ETc  [0, 2] mm/dia (proxy) ----
def _etc_sets(etc: float) -> dict:
    """
    Fuzzifica la evapotranspiracion del cultivo (ETc).
    """
    return {
        'baja'   : _trap(etc, 0.0, 0.0, 0.20, 0.50),
        'media'  : _tri (etc, 0.30, 0.65, 1.00),
        'alta'   : _tri (etc, 0.80, 1.15, 1.50),
        'extrema': _trap(etc, 1.30, 1.60, 2.00, 2.00),
    }


# ==============================================================================
# 3.  CONJUNTOS DIFUSOS — SALIDA  [0, 100]
# ==============================================================================

UNIVERSE = np.linspace(0, 100, 1001)

OUTPUT_SETS = {
    'no_regar'     : lambda xs: _trap_vec(xs,  0,  0, 10, 25),
    'riego_minimo' : lambda xs: _tri_vec (xs, 15, 30, 45),
    'regar'        : lambda xs: _tri_vec (xs, 40, 55, 70),
    'riego_intenso': lambda xs: _tri_vec (xs, 60, 75, 90),
    'riego_urgente': lambda xs: _trap_vec(xs, 80, 90,100,100),
}


# ==============================================================================
# 4.  BASE DE REGLAS MAMDANI
# ==============================================================================

def _build_rules(moi_fs: dict, temp_fs: dict, hum_fs: dict,
                 etc_fs: dict) -> list:
    """
    Devuelve lista de (strength, output_label).
    AND = min   |   OR = max   |   Implicacion = min (Mamdani)
    22 reglas calibradas para Vaccinium corymbosum en costa limeña.
    """
    m = moi_fs
    t = temp_fs
    h = hum_fs
    e = etc_fs

    rules = [
        # ===== EMERGENCIAS — riego urgente =====
        # R01: Suelo muy seco + calor extremo → urgente
        (min(m['muy_seco'], t['extrema']),                   'riego_urgente'),
        # R02: Suelo muy seco + ETc extrema → urgente
        (min(m['muy_seco'], e['extrema']),                   'riego_urgente'),
        # R03: Suelo muy seco + aire muy seco + calor → urgente
        (min(m['muy_seco'], h['muy_seca'], t['calida']),     'riego_urgente'),

        # ===== RIEGO INTENSO =====
        # R04: Suelo muy seco (sin otro agravante) → intenso
        (m['muy_seco'],                                      'riego_intenso'),
        # R05: Suelo seco + calor extremo → intenso
        (min(m['seco'], t['extrema']),                       'riego_intenso'),
        # R06: Suelo seco + aire muy seco + ETc alta → intenso
        (min(m['seco'], h['muy_seca'], e['alta']),           'riego_intenso'),
        # R07: Suelo seco + ETc extrema → intenso
        (min(m['seco'], e['extrema']),                       'riego_intenso'),

        # ===== REGAR =====
        # R08: Suelo seco + calor → regar
        (min(m['seco'], t['calida']),                        'regar'),
        # R09: Suelo seco + aire seco → regar
        (min(m['seco'], h['seca']),                          'regar'),
        # R10: Suelo seco + ETc media → regar
        (min(m['seco'], e['media']),                         'regar'),
        # R11: Suelo optimo + calor + aire seco → regar (evitar deficit)
        (min(m['optimo'], t['calida'], h['seca']),           'regar'),
        # R12: Suelo optimo + ETc alta → regar (mantener rango)
        (min(m['optimo'], e['alta']),                        'regar'),
        # R13: Suelo seco + temperatura fresca (arandano sensible) → regar
        (min(m['seco'], t['fresca']),                        'regar'),

        # ===== RIEGO MINIMO =====
        # R14: Suelo optimo + temperatura optima → minimo
        (min(m['optimo'], t['optima']),                      'riego_minimo'),
        # R15: Suelo optimo + aire moderado → minimo
        (min(m['optimo'], h['moderada']),                    'riego_minimo'),
        # R16: Suelo optimo + temperatura fria → minimo (reducir en frio)
        (min(m['optimo'], t['fria']),                        'riego_minimo'),
        # R17: Suelo seco + temperatura fria + aire humedo → minimo
        (min(m['seco'], t['fria'], h['alta']),               'riego_minimo'),

        # ===== NO REGAR =====
        # R18: Suelo humedo → no regar (sobre limite superior arandano)
        (m['humedo'],                                        'no_regar'),
        # R19: Suelo saturado → no regar (asfixia radicular)
        (m['saturado'],                                      'no_regar'),
        # R20: Suelo optimo + aire muy humedo + frio → no regar
        (min(m['optimo'], h['muy_alta'], t['fria']),         'no_regar'),
        # R21: Suelo humedo + aire muy humedo → no regar
        (min(m['humedo'], h['muy_alta']),                    'no_regar'),
        # R22: ETc baja + suelo optimo + aire alto → no regar
        (min(e['baja'], m['optimo'], h['alta']),             'no_regar'),
    ]

    # Eliminar reglas con activacion cero
    return [(s, lbl) for s, lbl in rules if s > 0.0]


# ==============================================================================
# 5.  MOTOR DE INFERENCIA Y DEFUZZIFICACION
# ==============================================================================

def _aggregate(activated_rules: list) -> np.ndarray:
    """
    Agrega los consecuentes difusos cortados (implicacion min).
    Metodo: union por max sobre el universo de salida.
    """
    aggregated = np.zeros(len(UNIVERSE))

    for strength, label in activated_rules:
        if strength <= 0.0:
            continue
        # Conjunto difuso del consecuente sobre el universo
        membership = OUTPUT_SETS[label](UNIVERSE)
        # Cortar al nivel de activacion (implicacion Mamdani = min)
        clipped = np.minimum(membership, strength)
        # Agregar por maximo
        aggregated = np.maximum(aggregated, clipped)

    return aggregated


def _centroid(aggregated: np.ndarray) -> float:
    """
    Defuzzificacion por centroide (Center of Gravity).
    Devuelve valor en [0, 100].
    """
    denom = aggregated.sum()
    if denom < 1e-10:
        return 0.0
    return float(np.dot(UNIVERSE, aggregated) / denom)


# ==============================================================================
# 6.  FUNCION PUBLICA PRINCIPAL
# ==============================================================================

def evaluate_expert_system(temp, hum, cultivo, suelo, etapa, moi,
                            etc: float = None) -> dict:
    """
    Motor de Inferencia Difusa Mamdani para decision de riego.

    Parametros
    ----------
    temp    : float — temperatura ambiental (°C)
    hum     : float — humedad relativa del aire (%)
    cultivo : str   — tipo de cultivo (informativo, no afecta logica difusa)
    suelo   : str   — tipo de suelo (informativo)
    etapa   : str   — etapa fenologica del arandano
    moi     : float — indice de humedad del suelo (0-100)
    etc     : float — evapotranspiracion del cultivo mm/dia (opcional)

    Retorna
    -------
    dict con claves:
        Regar, Porcentaje_Riego, Porcentaje_No_Riego,
        Score_Difuso, Nivel_Prioridad,
        Activaciones (dict de pertenencias por variable)
    """

    # ---- Validacion y limpieza ----
    try:
        temp = max(0.0, min(50.0,  float(temp)))
    except (TypeError, ValueError):
        temp = 25.0

    try:
        hum  = max(0.0, min(100.0, float(hum)))
    except (TypeError, ValueError):
        hum  = 50.0

    try:
        moi  = max(0.0, min(100.0, float(moi)))
    except (TypeError, ValueError):
        moi  = 50.0

    if etc is None:
        # Estimacion ETc si no se proporciona
        eto = 0.408 * max(0.0, temp - 2.0) / 25.0 * (1 + temp / 50.0)
        KC_DEFAULT = {
            'germinacion': 0.30, 'germinación': 0.30,
            'desarrollo_vegetativo': 0.70, 'desarrollo vegetativo': 0.70,
            'floracion': 1.05, 'floración': 1.05, 'flowering': 1.05,
            'fructificacion': 0.90, 'fructificación': 0.90,
        }
        kc = KC_DEFAULT.get(str(etapa).lower().strip(), 0.70)
        etc = max(0.0, eto * kc)
    else:
        try:
            etc = max(0.0, float(etc))
        except (TypeError, ValueError):
            etc = 0.5

    # ---- Fuzzificacion ----
    moi_fs  = _moi_sets(moi)
    temp_fs = _temp_sets(temp)
    hum_fs  = _hum_sets(hum)
    etc_fs  = _etc_sets(etc)

    # ---- Evaluacion de reglas ----
    activated = _build_rules(moi_fs, temp_fs, hum_fs, etc_fs)

    # ---- Agregacion ----
    aggregated = _aggregate(activated)

    # ---- Defuzzificacion por centroide ----
    score = _centroid(aggregated)

    # ---- Interpretacion del resultado ----
    porcentaje_riego    = round(score, 2)
    porcentaje_no_riego = round(100.0 - score, 2)
    decision            = "Requiere Riego" if score > 50.0 else "No Requiere Riego"

    if score >= 80:
        nivel = "URGENTE"
    elif score >= 65:
        nivel = "ALTA"
    elif score >= 50:
        nivel = "MEDIA"
    elif score >= 35:
        nivel = "BAJA"
    else:
        nivel = "MUY BAJA"

    return {
        "Regar"             : decision,
        "Porcentaje_Riego"  : porcentaje_riego,
        "Porcentaje_No_Riego": porcentaje_no_riego,
        "Score_Difuso"      : round(score, 4),
        "Nivel_Prioridad"   : nivel,
        "Motor"             : "Mamdani",
        "Activaciones": {
            "moi" : {k: round(v, 4) for k, v in moi_fs.items()  if v > 0},
            "temp": {k: round(v, 4) for k, v in temp_fs.items() if v > 0},
            "hum" : {k: round(v, 4) for k, v in hum_fs.items()  if v > 0},
            "etc" : {k: round(v, 4) for k, v in etc_fs.items()  if v > 0},
        },
        "Reglas_activadas"  : len(activated),
        "ETc_usado"         : round(etc, 4),
    }


# ==============================================================================
# COMPATIBILIDAD: validate_batch (mantener firma del modulo anterior)
# ==============================================================================

def validate_batch(test_data: list, ml_predictions: list = None) -> object:
    """
    Valida el sistema experto contra multiples casos.
    Retorna DataFrame con resultados comparativos.
    """
    import pandas as pd

    results = []
    for i, data in enumerate(test_data):
        result = evaluate_expert_system(
            temp    = data.get('temperatura', data.get('temp', 25)),
            hum     = data.get('humedad',     data.get('humidity', 50)),
            cultivo = data.get('cultivo',     data.get('crop_id', '')),
            suelo   = data.get('suelo',       data.get('soil_type', '')),
            etapa   = data.get('etapa',       data.get('seedling_stage', '')),
            moi     = data.get('moi',         50),
            etc     = data.get('etc',         None),
        )
        result['caso'] = i + 1

        if ml_predictions and i < len(ml_predictions):
            result['ml_prediction'] = ml_predictions[i]
            result['diferencia']    = abs(result['Porcentaje_Riego'] - ml_predictions[i])

        results.append(result)

    return pd.DataFrame(results)


# ==============================================================================
# PRUEBA RAPIDA (ejecutar directamente)
# ==============================================================================

if __name__ == '__main__':
    import sys, io
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    casos = [
        dict(temp=38, hum=25, moi=15,  etapa='Floracion',
             cultivo='Blueberry', suelo='Sandy Soil', desc='Emergencia: suelo seco + calor'),
        dict(temp=22, hum=65, moi=70,  etapa='Floracion',
             cultivo='Blueberry', suelo='Sandy Soil', desc='Optimo: condiciones ideales'),
        dict(temp=20, hum=80, moi=90,  etapa='Germinacion',
             cultivo='Blueberry', suelo='Sandy Soil', desc='Saturado: riesgo asfixia'),
        dict(temp=30, hum=40, moi=45,  etapa='Fructificacion',
             cultivo='Blueberry', suelo='Sandy Soil', desc='Seco + calor moderado'),
        dict(temp=15, hum=90, moi=65,  etapa='Desarrollo_Vegetativo',
             cultivo='Blueberry', suelo='Sandy Soil', desc='Frio + humedo + optimo'),
    ]

    print("=" * 70)
    print("PRUEBA MOTOR MAMDANI — Vaccinium corymbosum")
    print("=" * 70)
    for c in casos:
        r = evaluate_expert_system(
            temp=c['temp'], hum=c['hum'], moi=c['moi'],
            etapa=c['etapa'], cultivo=c['cultivo'], suelo=c['suelo']
        )
        print(f"\n[{c['desc']}]")
        print(f"  Inputs : MOI={c['moi']}%  Temp={c['temp']}C  Hum={c['hum']}%")
        print(f"  Score  : {r['Score_Difuso']:.2f}/100  ->  {r['Regar']}")
        print(f"  Nivel  : {r['Nivel_Prioridad']}  (ETc={r['ETc_usado']} mm/d, reglas={r['Reglas_activadas']})")
        print(f"  Activ. : MOI={r['Activaciones']['moi']}  Temp={r['Activaciones']['temp']}")
