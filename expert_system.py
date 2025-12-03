import numpy as np
from typing import Dict, List, Optional


# ============================================================
# FUNCIONES DE PERTENENCIA DIFUSA
# ============================================================
def triangular(x, a, b, c):
    """Función de pertenencia triangular. Devuelve grado de pertenencia [0, 1]."""
    x = np.array(x, dtype=float)
    left_denom = b - a if (b - a) != 0 else 1e-6
    right_denom = c - b if (c - b) != 0 else 1e-6
    left = (x - a) / left_denom
    right = (c - x) / right_denom
    res = np.maximum(0, np.minimum(left, right))
    return float(res) if np.isscalar(x) else res


def trapezoidal(x, a, b, c, d):
    """Función de pertenencia trapezoidal. Devuelve grado de pertenencia [0, 1]."""
    x = np.array(x, dtype=float)
    denom_left = b - a if (b - a) != 0 else 1e-6
    denom_right = d - c if (d - c) != 0 else 1e-6
    left = (x - a) / denom_left
    right = (d - x) / denom_right
    res = np.minimum(np.minimum(left, 1.0), right)
    res = np.maximum(res, 0.0)
    return float(res) if np.isscalar(x) else res


def normalize_sigmoid(value, center=50, steepness=0.05):
    """Normalización sigmoidal para transiciones suaves."""
    return 1 / (1 + np.exp(-steepness * (value - center)))


def normalize_gaussian(value, mean, std):
    """Normalización gaussiana para valores óptimos."""
    return np.exp(-0.5 * ((value - mean) / std) ** 2)


# ============================================================
# FUZZIFICACIÓN DE VARIABLES CONTINUAS
# ============================================================
def fuzzify_moi(moi: float) -> Dict[str, float]:
    """Convierte MOI crisp a conjuntos difusos."""
    return {
        'muy_bajo': trapezoidal(moi, 0, 0, 15, 25),
        'bajo': triangular(moi, 15, 30, 45),
        'medio': triangular(moi, 35, 50, 65),
        'alto': triangular(moi, 55, 70, 85),
        'muy_alto': trapezoidal(moi, 75, 90, 100, 100)
    }


def fuzzify_temperature(temp: float) -> Dict[str, float]:
    """Convierte temperatura crisp a conjuntos difusos."""
    return {
        'frio': trapezoidal(temp, 0, 0, 12, 18),
        'templado': triangular(temp, 15, 20, 25),
        'calido': triangular(temp, 22, 28, 34),
        'muy_calido': trapezoidal(temp, 30, 38, 50, 50)
    }


def fuzzify_humidity(hum: float) -> Dict[str, float]:
    """Convierte humedad ambiental crisp a conjuntos difusos."""
    return {
        'muy_seca': trapezoidal(hum, 0, 0, 20, 30),
        'seca': triangular(hum, 20, 35, 50),
        'normal': triangular(hum, 40, 55, 70),
        'humeda': triangular(hum, 60, 75, 90),
        'muy_humeda': trapezoidal(hum, 80, 95, 100, 100)
    }


# ============================================================
# MAPEOS Y CATEGORIZACIÓN
# ============================================================
cultivo_map = {
    "wheat": "baja", "carrot": "alta", "potato": "alta",
    "chilli": "alta", "chili": "alta", "tomato": "alta",
    "pepper": "alta", "corn": "media", "maize": "media",
    "rice": "alta", "cotton": "media", "sugarcane": "alta"
}

suelo_map = {
    "sandy soil": "baja", "sandy": "baja", "chalky soil": "baja",
    "chalky": "baja", "red soil": "media", "red": "media",
    "loam soil": "media", "loam": "media", "alluvial soil": "media",
    "alluvial": "media", "alluvian soil": "media", "alluvian": "media",
    "black soil": "alta", "black": "alta", "clay soil": "alta", "clay": "alta"
}

etapa_map = {
    "harvest": "baja", "harvesting": "baja", "maturation": "media",
    "germination": "media", "seedling stage": "media", "seedling": "media",
    "vegetative growth": "alta",
    "vegetative growth / root or tuber development": "alta",
    "root development": "alta", "tuber development": "alta",
    "flowering": "alta", "pollination": "alta", "fruit formation": "alta",
    "fruit/grain/bulb formation": "alta", "grain formation": "alta",
    "bulb formation": "alta"
}


def get_categoria(valor, mapping):
    """Normaliza y categoriza texto a baja/media/alta."""
    if valor is None:
        return "media"

    valor_norm = str(valor).lower().strip()

    if valor_norm in mapping:
        return mapping[valor_norm]

    for key in mapping.keys():
        if key in valor_norm or valor_norm in key:
            return mapping[key]

    return "media"


# ============================================================
# OPERADORES DIFUSOS
# ============================================================
def fuzzy_AND(a: float, b: float) -> float:
    """AND difuso usando t-norma (mínimo)."""
    return min(a, b)


def fuzzy_OR(a: float, b: float) -> float:
    """OR difuso usando t-conorma (máximo)."""
    return max(a, b)


def fuzzy_NOT(a: float) -> float:
    """NOT difuso (complemento)."""
    return 1.0 - a


# ============================================================
# BASE DE REGLAS DIFUSAS
# ============================================================
class FuzzyRule:
    """Representa una regla difusa del sistema experto."""

    def __init__(self, name: str, condition_func, strength: float):
        self.name = name
        self.condition_func = condition_func
        self.strength = strength

    def evaluate(self, moi_fuzzy, temp_fuzzy, hum_fuzzy,
                 cat_cultivo, cat_suelo, cat_etapa,
                 moi, temp, hum) -> float:
        """Evalúa la regla y retorna activación * fuerza."""
        activation = self.condition_func(
            moi_fuzzy, temp_fuzzy, hum_fuzzy,
            cat_cultivo, cat_suelo, cat_etapa,
            moi, temp, hum
        )
        return activation * self.strength


def create_fuzzy_rules() -> List[FuzzyRule]:
    """Crea la base de conocimiento con reglas difusas."""
    rules = [
        # ===== REGLAS CRÍTICAS =====
        FuzzyRule(
            name="R1_CRITICO_MOI_TEMP",
            condition_func=lambda mf, tf, hf, cc, cs, ce, m, t, h:
                fuzzy_AND(mf['muy_bajo'], tf['muy_calido']),
            strength=20.0
        ),
        FuzzyRule(
            name="R2_CRITICO_MOI",
            condition_func=lambda mf, tf, hf, cc, cs, ce, m, t, h:
                mf['muy_bajo'],
            strength=18.0
        ),
        FuzzyRule(
            name="R3_ESTRES_TERMICO",
            condition_func=lambda mf, tf, hf, cc, cs, ce, m, t, h:
                tf['muy_calido'],
            strength=15.0
        ),
        FuzzyRule(
            name="R4_AIRE_SECO_MOI_BAJO",
            condition_func=lambda mf, tf, hf, cc, cs, ce, m, t, h:
                fuzzy_AND(hf['muy_seca'], mf['bajo']),
            strength=14.0
        ),

        # ===== REGLAS DE ALTA NECESIDAD =====
        FuzzyRule(
            name="R5_DEMANDA_ALTA_CRITICA",
            condition_func=lambda mf, tf, hf, cc, cs, ce, m, t, h:
                mf['medio'] if (cc == "alta" and ce == "alta") else 0.0,
            strength=12.0
        ),
        FuzzyRule(
            name="R6_SUELO_ARENOSO_CALOR",
            condition_func=lambda mf, tf, hf, cc, cs, ce, m, t, h:
                fuzzy_AND(mf['medio'], tf['calido']) if cs == "baja" else 0.0,
            strength=10.0
        ),
        FuzzyRule(
            name="R7_MOI_BAJO_CALOR",
            condition_func=lambda mf, tf, hf, cc, cs, ce, m, t, h:
                fuzzy_AND(mf['bajo'], tf['calido']),
            strength=11.0
        ),
        FuzzyRule(
            name="R8_EVAPORACION_ALTA",
            condition_func=lambda mf, tf, hf, cc, cs, ce, m, t, h:
                fuzzy_AND(hf['seca'], tf['calido']),
            strength=8.0
        ),

        # ===== REGLAS DE REDUCCIÓN =====
        FuzzyRule(
            name="R9_MOI_SATURADO",
            condition_func=lambda mf, tf, hf, cc, cs, ce, m, t, h:
                mf['muy_alto'],
            strength=-22.0
        ),
        FuzzyRule(
            name="R10_CONDICIONES_OPTIMAS",
            condition_func=lambda mf, tf, hf, cc, cs, ce, m, t, h:
                fuzzy_AND(fuzzy_AND(tf['templado'], hf['humeda']), mf['alto']),
            strength=-18.0
        ),
        FuzzyRule(
            name="R11_EXCESO_HUMEDAD",
            condition_func=lambda mf, tf, hf, cc, cs, ce, m, t, h:
                fuzzy_AND(mf['alto'], hf['muy_humeda']),
            strength=-16.0
        ),
        FuzzyRule(
            name="R12_FRIO_RETENCION_ALTA",
            condition_func=lambda mf, tf, hf, cc, cs, ce, m, t, h:
                fuzzy_AND(tf['frio'], mf['alto']) if cs == "alta" else 0.0,
            strength=-14.0
        ),
        FuzzyRule(
            name="R13_COSECHA_MOI_OK",
            condition_func=lambda mf, tf, hf, cc, cs, ce, m, t, h:
                mf['medio'] if ce == "baja" else 0.0,
            strength=-12.0
        ),
        FuzzyRule(
            name="R14_CULTIVO_RESISTENTE",
            condition_func=lambda mf, tf, hf, cc, cs, ce, m, t, h:
                fuzzy_AND(mf['medio'], tf['templado']) if cc == "baja" else 0.0,
            strength=-10.0
        ),

        # ===== REGLAS DE AJUSTE FINO =====
        FuzzyRule(
            name="R15_VENTANA_CRITICA",
            condition_func=lambda mf, tf, hf, cc, cs, ce, m, t, h:
                fuzzy_AND(mf['medio'], tf['calido']),
            strength=7.0
        ),
        FuzzyRule(
            name="R16_SUELO_FRANCO_EQUILIBRADO",
            condition_func=lambda mf, tf, hf, cc, cs, ce, m, t, h:
                fuzzy_AND(mf['medio'], tf['templado']) if cs == "media" else 0.0,
            strength=-6.0
        ),
        FuzzyRule(
            name="R17_ETAPA_SENSIBLE",
            condition_func=lambda mf, tf, hf, cc, cs, ce, m, t, h:
                mf['medio'] if ce == "alta" else 0.0,
            strength=8.0
        ),
        FuzzyRule(
            name="R18_NOCHE_HUMEDA",
            condition_func=lambda mf, tf, hf, cc, cs, ce, m, t, h:
                fuzzy_AND(tf['templado'], hf['muy_humeda']),
            strength=-5.0
        )
    ]

    return rules


# ============================================================
# SISTEMA EXPERTO PRINCIPAL
# ============================================================
def evaluate_expert_system(temp, hum, cultivo, suelo, etapa, moi):
    """Sistema experto difuso completo para decisión de riego."""

    # 1. VALIDACIÓN Y NORMALIZACIÓN
    try:
        temp = float(temp)
        temp = max(0, min(50, temp))
    except Exception:
        temp = 25.0

    try:
        hum = float(hum)
        hum = max(0, min(100, hum))
    except Exception:
        hum = 50.0

    if moi is None:
        moi = 50.0 + (hum - 50) * 0.5 - (temp - 25) * 1.2
        moi = max(0, min(100, moi))
    else:
        try:
            moi = float(moi)
            moi = max(0, min(100, moi))
        except Exception:
            moi = 50.0

    # Categorización
    cat_cultivo = get_categoria(cultivo, cultivo_map)
    cat_suelo = get_categoria(suelo, suelo_map)
    cat_etapa = get_categoria(etapa, etapa_map)

    # 2. FUZZIFICACIÓN
    moi_fuzzy = fuzzify_moi(moi)
    temp_fuzzy = fuzzify_temperature(temp)
    hum_fuzzy = fuzzify_humidity(hum)

    # 3. SCORE BASE
    moi_score = (
        moi_fuzzy['muy_bajo'] * 95.0 +
        moi_fuzzy['bajo'] * 70.0 +
        moi_fuzzy['medio'] * 40.0 +
        moi_fuzzy['alto'] * 20.0 +
        moi_fuzzy['muy_alto'] * 5.0
    )
    moi_contribution = moi_score * 0.45

    temp_score = (
        temp_fuzzy['frio'] * 25.0 +
        temp_fuzzy['templado'] * 30.0 +
        temp_fuzzy['calido'] * 55.0 +
        temp_fuzzy['muy_calido'] * 85.0
    )
    temp_contribution = temp_score * 0.22

    hum_score = (
        hum_fuzzy['muy_seca'] * 85.0 +
        hum_fuzzy['seca'] * 65.0 +
        hum_fuzzy['normal'] * 40.0 +
        hum_fuzzy['humeda'] * 25.0 +
        hum_fuzzy['muy_humeda'] * 15.0
    )
    hum_contribution = hum_score * 0.15

    cultivo_scores = {"baja": 20, "media": 50, "alta": 80}
    suelo_scores = {"baja": 70, "media": 50, "alta": 30}
    etapa_scores = {"baja": 30, "media": 50, "alta": 80}

    cultivo_contribution = cultivo_scores[cat_cultivo] * 0.08
    suelo_contribution = suelo_scores[cat_suelo] * 0.06
    etapa_contribution = etapa_scores[cat_etapa] * 0.04

    score_base = (
        moi_contribution +
        temp_contribution +
        hum_contribution +
        cultivo_contribution +
        suelo_contribution +
        etapa_contribution
    )

    # 4. INFERENCIA DIFUSA
    rules = create_fuzzy_rules()
    ajuste_total = 0.0
    reglas_activas = []

    for rule in rules:
        activation_strength = rule.evaluate(
            moi_fuzzy, temp_fuzzy, hum_fuzzy,
            cat_cultivo, cat_suelo, cat_etapa,
            moi, temp, hum
        )

        if abs(activation_strength) > 0.1:
            ajuste_total += activation_strength
            activation = (
                abs(activation_strength / rule.strength)
                if rule.strength != 0 else 0
            )
            reglas_activas.append({
                'nombre': rule.name,
                'activacion': round(activation, 3),
                'contribucion': round(activation_strength, 2)
            })

    reglas_activas.sort(key=lambda x: abs(x['contribucion']), reverse=True)

    # 5. DEFUZZIFICACIÓN
    score_total = score_base + ajuste_total
    riego = 100 / (1 + np.exp(-0.05 * (score_total - 50)))
    riego = max(0, min(100, riego))

    porcentaje_riego = round(float(riego), 2)
    porcentaje_no_riego = round(100.0 - porcentaje_riego, 2)
    decision = "Requiere Riego" if porcentaje_riego > 50.0 else "No Requiere Riego"

    if porcentaje_riego >= 80:
        nivel = "🔴 URGENTE"
    elif porcentaje_riego >= 65:
        nivel = "🟠 ALTA"
    elif porcentaje_riego >= 50:
        nivel = "🟡 MEDIA"
    elif porcentaje_riego >= 35:
        nivel = "🟢 BAJA"
    else:
        nivel = "⚪ MUY BAJA"

    return {
        "Regar": decision,
        "Porcentaje_Riego": porcentaje_riego,
        "Porcentaje_No_Riego": porcentaje_no_riego,
        "Score_Total": round(score_total, 2),
        "Score_Base": round(score_base, 2),
        "Ajuste_Difuso": round(ajuste_total, 2),
        "Nivel_Prioridad": nivel.split()[1] if ' ' in nivel else nivel,
        "Reglas_Activadas": len(reglas_activas),
        "Top_Reglas": reglas_activas[:5],
        "Memberships": {
            "moi": {k: round(v, 3) for k, v in moi_fuzzy.items() if v > 0.01},
            "temp": {k: round(v, 3) for k, v in temp_fuzzy.items() if v > 0.01},
            "hum": {k: round(v, 3) for k, v in hum_fuzzy.items() if v > 0.01}
        }
    }