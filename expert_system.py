# ============================================================
# expert_system_optimized.py - VERSIÓN OPTIMIZADA PARA ML
# Sistema calibrado con pesos ajustados y reglas refinadas
# ============================================================

import numpy as np

# ------------------------------------------------------------
# FUNCIONES AUXILIARES
# ------------------------------------------------------------
def triangular(x, a, b, c):
    x = np.array(x, dtype=float)
    left_denom = b - a if (b - a) != 0 else 1e-6
    right_denom = c - b if (c - b) != 0 else 1e-6
    left = (x - a) / left_denom
    right = (c - x) / right_denom
    res = np.maximum(0, np.minimum(left, right))
    return res

def trapezoidal(x, a, b, c, d):
    x = np.array(x, dtype=float)
    denom_left = b - a if (b - a) != 0 else 1e-6
    denom_right = d - c if (d - c) != 0 else 1e-6
    left = (x - a) / denom_left
    right = (d - x) / denom_right
    res = np.minimum(np.minimum(left, 1.0), right)
    res = np.maximum(res, 0.0)
    return res

# ------------------------------------------------------------
# MAPEOS MEJORADOS CON MÁS VARIANTES
# ------------------------------------------------------------
cultivo_map = {
    "wheat": "baja", 
    "carrot": "alta", "potato": "alta", "chilli": "alta", 
    "chili": "alta", "tomato": "alta", "pepper": "alta",
    "corn": "media", "maize": "media", "rice": "alta",
    "cotton": "media", "sugarcane": "alta"
}

suelo_map = {
    "sandy soil": "baja", "sandy": "baja", "chalky soil": "baja", "chalky": "baja",
    "red soil": "media", "red": "media", "loam soil": "media", "loam": "media",
    "alluvial soil": "media", "alluvial": "media", "alluvian soil": "media", "alluvian": "media",
    "black soil": "alta", "black": "alta", "clay soil": "alta", "clay": "alta"
}

etapa_map = {
    "harvest": "baja", "harvesting": "baja",
    "maturation": "media", "germination": "media", 
    "seedling stage": "media", "seedling": "media",
    "vegetative growth": "alta", 
    "vegetative growth / root or tuber development": "alta",
    "root development": "alta", "tuber development": "alta",
    "flowering": "alta", "pollination": "alta", 
    "fruit formation": "alta", "fruit/grain/bulb formation": "alta",
    "grain formation": "alta", "bulb formation": "alta"
}

def get_categoria(valor, mapping):
    """Función mejorada de normalización"""
    if valor is None:
        return "media"
    
    valor_norm = str(valor).lower().strip()
    
    # Búsqueda directa
    if valor_norm in mapping:
        return mapping[valor_norm]
    
    # Búsqueda por coincidencia parcial
    for key in mapping.keys():
        if key in valor_norm or valor_norm in key:
            return mapping[key]
    
    # Valor por defecto si no se encuentra
    return "media"

# ------------------------------------------------------------
# FUNCIÓN DE NORMALIZACIÓN NO LINEAL
# ------------------------------------------------------------
def normalize_sigmoid(value, center=50, steepness=0.05):
    """
    Normalización sigmoidal para transiciones suaves
    """
    return 1 / (1 + np.exp(-steepness * (value - center)))

def normalize_gaussian(value, mean, std):
    """
    Normalización gaussiana para valores óptimos
    """
    return np.exp(-0.5 * ((value - mean) / std) ** 2)

# ------------------------------------------------------------
# SISTEMA EXPERTO OPTIMIZADO
# ------------------------------------------------------------
def evaluate_expert_system(temp, hum, cultivo, suelo, etapa, moi):
    """
    Sistema experto optimizado con calibración mejorada
    
    PESOS ACTUALIZADOS:
    - MOI: 45% (factor principal pero no único)
    - Temperatura: 22% (muy importante en zonas cálidas)
    - Humedad ambiental: 15%
    - Cultivo: 8%
    - Suelo: 6%
    - Etapa: 4%
    """
    
    # =========================================================
    # VALIDACIÓN Y NORMALIZACIÓN DE ENTRADAS
    # =========================================================
    try:
        temp = float(temp)
        temp = max(0, min(50, temp))  # Limitar rango
    except:
        temp = 25.0
    
    try:
        hum = float(hum)
        hum = max(0, min(100, hum))
    except:
        hum = 50.0
    
    if moi is None:
        # Cálculo mejorado de MOI si no se proporciona
        moi = 50.0 + (hum - 50) * 0.5 - (temp - 25) * 1.2
        moi = max(0, min(100, moi))
    else:
        try:
            moi = float(moi)
            moi = max(0, min(100, moi))
        except:
            moi = 50.0

    print(f"\n{'='*60}")
    print(f"🔍 SISTEMA EXPERTO - ANÁLISIS DE RIEGO")
    print(f"{'='*60}")
    print(f"📊 ENTRADAS:")
    print(f"   • Temperatura: {temp:.1f}°C")
    print(f"   • Humedad ambiental: {hum:.1f}%")
    print(f"   • MOI (humedad suelo): {moi:.1f}%")
    print(f"   • Cultivo: '{cultivo}'")
    print(f"   • Tipo de suelo: '{suelo}'")
    print(f"   • Etapa fenológica: '{etapa}'")

    # Obtener categorías
    cat_cultivo = get_categoria(cultivo, cultivo_map)
    cat_suelo = get_categoria(suelo, suelo_map)
    cat_etapa = get_categoria(etapa, etapa_map)
    
    print(f"\n📋 CATEGORIZACIÓN:")
    print(f"   • Demanda hídrica cultivo: {cat_cultivo.upper()}")
    print(f"   • Retención de agua suelo: {cat_suelo.upper()}")
    print(f"   • Necesidad hídrica etapa: {cat_etapa.upper()}")

    # =========================================================
    # CÁLCULO DE SCORES MEJORADO
    # =========================================================
    
    print(f"\n💯 CÁLCULO DE SCORES:")
    print(f"{'-'*60}")
    
    # -------------------------
    # FACTOR 1: MOI (PESO: 45%)
    # -------------------------
    # Usar función no lineal para MOI
    if moi < 30:
        # Crítico: necesidad muy alta
        moi_score = 100 - moi * 0.8
    elif moi < 50:
        # Bajo: necesidad alta
        moi_score = 75 - (moi - 30) * 1.5
    elif moi < 70:
        # Medio: necesidad moderada
        moi_score = 45 - (moi - 50) * 1.0
    else:
        # Alto: necesidad baja
        moi_score = 25 - (moi - 70) * 0.8
    
    moi_score = max(0, min(100, moi_score))
    moi_contribution = moi_score * 0.45
    
    print(f"1️⃣  MOI (45%): {moi:.1f}% → Score: {moi_score:.1f} → Contribución: {moi_contribution:.1f}")
    
    # -------------------------
    # FACTOR 2: TEMPERATURA (PESO: 22%)
    # -------------------------
    # Temperatura óptima: 20-25°C
    if temp < 15:
        temp_score = 20 + (15 - temp) * 0.5  # Frío leve aumenta necesidad
    elif temp < 25:
        temp_score = 20 + (temp - 15) * 1.0  # Rango óptimo-cálido
    elif temp < 35:
        temp_score = 30 + (temp - 25) * 2.5  # Calor moderado
    else:
        temp_score = 55 + (temp - 35) * 3.0  # Calor extremo
    
    temp_score = max(0, min(100, temp_score))
    temp_contribution = temp_score * 0.22
    
    print(f"2️⃣  Temperatura (22%): {temp:.1f}°C → Score: {temp_score:.1f} → Contribución: {temp_contribution:.1f}")
    
    # -------------------------
    # FACTOR 3: HUMEDAD AMBIENTAL (PESO: 15%)
    # -------------------------
    # Humedad baja = mayor evapotranspiración
    if hum < 30:
        hum_score = 90 - hum * 0.5
    elif hum < 60:
        hum_score = 70 - (hum - 30) * 1.5
    else:
        hum_score = 25 - (hum - 60) * 0.3
    
    hum_score = max(0, min(100, hum_score))
    hum_contribution = hum_score * 0.15
    
    print(f"3️⃣  Humedad ambiental (15%): {hum:.1f}% → Score: {hum_score:.1f} → Contribución: {hum_contribution:.1f}")
    
    # -------------------------
    # FACTOR 4: DEMANDA DEL CULTIVO (PESO: 8%)
    # -------------------------
    cultivo_scores = {
        "baja": 20,   # Cultivos resistentes (ej: trigo)
        "media": 50,  # Cultivos moderados
        "alta": 80    # Cultivos exigentes (ej: tomate, chile)
    }
    cultivo_score = cultivo_scores[cat_cultivo]
    cultivo_contribution = cultivo_score * 0.08
    
    print(f"4️⃣  Cultivo (8%): {cat_cultivo.upper()} → Score: {cultivo_score:.1f} → Contribución: {cultivo_contribution:.1f}")
    
    # -------------------------
    # FACTOR 5: TIPO DE SUELO (PESO: 6%)
    # -------------------------
    suelo_scores = {
        "baja": 70,   # Arenoso: retiene poco agua
        "media": 50,  # Franco: retención media
        "alta": 30    # Arcilloso: retiene mucho agua
    }
    suelo_score = suelo_scores[cat_suelo]
    suelo_contribution = suelo_score * 0.06
    
    print(f"5️⃣  Suelo (6%): {cat_suelo.upper()} → Score: {suelo_score:.1f} → Contribución: {suelo_contribution:.1f}")
    
    # -------------------------
    # FACTOR 6: ETAPA FENOLÓGICA (PESO: 4%)
    # -------------------------
    etapa_scores = {
        "baja": 30,   # Cosecha: menos agua
        "media": 50,  # Germinación: moderada
        "alta": 80    # Floración/fructificación: crítica
    }
    etapa_score = etapa_scores[cat_etapa]
    etapa_contribution = etapa_score * 0.04
    
    print(f"6️⃣  Etapa (4%): {cat_etapa.upper()} → Score: {etapa_score:.1f} → Contribución: {etapa_contribution:.1f}")
    
    # -------------------------
    # SCORE BASE
    # -------------------------
    score_base = (moi_contribution + temp_contribution + hum_contribution + 
                  cultivo_contribution + suelo_contribution + etapa_contribution)
    
    print(f"\n📊 Score base (suma ponderada): {score_base:.2f}/100")
    
    # =========================================================
    # REGLAS DE AJUSTE CONTEXTUAL
    # =========================================================
    
    print(f"\n🔧 AJUSTES POR REGLAS CONTEXTUALES:")
    print(f"{'-'*60}")
    
    ajuste_total = 0
    reglas_aplicadas = []
    
    # REGLA 1: Condición CRÍTICA (MOI < 25 + Temp > 32)
    if moi < 25 and temp > 32:
        ajuste = 18
        ajuste_total += ajuste
        reglas_aplicadas.append(f"   🚨 R1: Condición crítica (MOI<25 + T>32): +{ajuste:.1f}")
    
    # REGLA 2: MOI muy bajo (<20)
    if moi < 20:
        ajuste = 20
        ajuste_total += ajuste
        reglas_aplicadas.append(f"   💧 R2: MOI crítico (<20): +{ajuste:.1f}")
    
    # REGLA 3: Estrés térmico severo (>38°C)
    if temp > 38:
        ajuste = 15
        ajuste_total += ajuste
        reglas_aplicadas.append(f"   🔥 R3: Estrés térmico severo (>38°C): +{ajuste:.1f}")
    
    # REGLA 4: Aire muy seco + MOI medio-bajo
    if hum < 25 and moi < 55:
        ajuste = 12
        ajuste_total += ajuste
        reglas_aplicadas.append(f"   💨 R4: Aire muy seco + MOI bajo: +{ajuste:.1f}")
    
    # REGLA 5: Cultivo exigente + Etapa crítica + MOI<60
    if cat_cultivo == "alta" and cat_etapa == "alta" and moi < 60:
        ajuste = 10
        ajuste_total += ajuste
        reglas_aplicadas.append(f"   🌱 R5: Alta demanda en fase crítica: +{ajuste:.1f}")
    
    # REGLA 6: Suelo arenoso + Calor + MOI medio
    if cat_suelo == "baja" and temp > 28 and 30 < moi < 65:
        ajuste = 8
        ajuste_total += ajuste
        reglas_aplicadas.append(f"   🏜️  R6: Suelo arenoso + calor: +{ajuste:.1f}")
    
    # REGLA 7: MOI MUY alto (>85) - casi nunca regar
    if moi > 85:
        ajuste = -25
        ajuste_total += ajuste
        reglas_aplicadas.append(f"   💦 R7: MOI muy alto (saturado): {ajuste:.1f}")
    
    # REGLA 8: Condiciones ÓPTIMAS (temp 18-26 + hum>65 + MOI>65)
    if 18 <= temp <= 26 and hum > 65 and moi > 65:
        ajuste = -18
        ajuste_total += ajuste
        reglas_aplicadas.append(f"   ✅ R8: Condiciones óptimas: {ajuste:.1f}")
    
    # REGLA 9: MOI alto + humedad alta
    if moi > 75 and hum > 75:
        ajuste = -15
        ajuste_total += ajuste
        reglas_aplicadas.append(f"   💧 R9: Exceso de humedad: {ajuste:.1f}")
    
    # REGLA 10: Temperatura baja + MOI alto + suelo arcilloso
    if temp < 18 and moi > 60 and cat_suelo == "alta":
        ajuste = -12
        ajuste_total += ajuste
        reglas_aplicadas.append(f"   ❄️  R10: Frío + retención alta: {ajuste:.1f}")
    
    # REGLA 11: Cosecha con humedad adecuada
    if cat_etapa == "baja" and moi > 45:
        ajuste = -10
        ajuste_total += ajuste
        reglas_aplicadas.append(f"   🌾 R11: Cosecha con humedad OK: {ajuste:.1f}")
    
    # REGLA 12: Cultivo resistente + condiciones normales
    if cat_cultivo == "baja" and 40 < moi < 70 and 18 < temp < 30:
        ajuste = -8
        ajuste_total += ajuste
        reglas_aplicadas.append(f"   🌿 R12: Cultivo resistente + normal: {ajuste:.1f}")
    
    # REGLA 13: Estrés combinado moderado
    if 30 < temp < 38 and hum < 40 and 25 < moi < 45:
        ajuste = 10
        ajuste_total += ajuste
        reglas_aplicadas.append(f"   ⚠️  R13: Estrés moderado combinado: +{ajuste:.1f}")
    
    # REGLA 14: Ventana crítica de riego (MOI 35-50 + temp>30)
    if 35 < moi < 50 and temp > 30:
        ajuste = 8
        ajuste_total += ajuste
        reglas_aplicadas.append(f"   ⏰ R14: Ventana crítica de riego: +{ajuste:.1f}")
    
    # REGLA 15: Suelo franco + condiciones moderadas
    if cat_suelo == "media" and 45 < moi < 65 and 22 < temp < 32:
        ajuste = -5
        ajuste_total += ajuste
        reglas_aplicadas.append(f"   🏞️  R15: Suelo franco equilibrado: {ajuste:.1f}")
    
    # Mostrar reglas aplicadas
    if reglas_aplicadas:
        for regla in reglas_aplicadas:
            print(regla)
        print(f"\n   📊 Ajuste total por reglas: {ajuste_total:+.2f}")
    else:
        print("   ℹ️  No se aplicaron reglas contextuales")
    
    # =========================================================
    # CÁLCULO FINAL
    # =========================================================
    
    score_total = score_base + ajuste_total
    
    # Normalizar con función sigmoidal para evitar saturación
    # Mapear [0, 120] → [0, 100] con curva suave
    riego = 100 / (1 + np.exp(-0.05 * (score_total - 50)))
    riego = max(0, min(100, riego))
    
    print(f"\n{'='*60}")
    print(f"📈 RESULTADO FINAL:")
    print(f"{'='*60}")
    print(f"   Score total: {score_total:.2f}")
    print(f"   Necesidad de riego: {riego:.1f}%")
    
    porcentaje_riego = round(float(riego), 2)
    porcentaje_no_riego = round(100.0 - porcentaje_riego, 2)
    decision = "Requiere Riego" if porcentaje_riego > 50.0 else "No Requiere Riego"
    
    # Determinar nivel de prioridad
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
        "Nivel_Prioridad": nivel.split()[1] if ' ' in nivel else nivel
    }


# ------------------------------------------------------------
# FUNCIÓN DE VALIDACIÓN BATCH
# ------------------------------------------------------------
def validate_batch(test_data, ml_predictions=None):
    """
    Valida el sistema experto contra múltiples casos
    
    Args:
        test_data: Lista de diccionarios con las entradas
        ml_predictions: Lista opcional con predicciones del modelo ML
    
    Returns:
        DataFrame con resultados comparativos
    """
    import pandas as pd
    
    results = []
    
    for i, data in enumerate(test_data):
        result = evaluate_expert_system(
            temp=data['temperatura'],
            hum=data['humedad'],
            cultivo=data['cultivo'],
            suelo=data['suelo'],
            etapa=data['etapa'],
            moi=data.get('moi')
        )
        
        result['caso'] = i + 1
        
        if ml_predictions and i < len(ml_predictions):
            result['ml_prediction'] = ml_predictions[i]
            result['diferencia'] = abs(result['Porcentaje_Riego'] - ml_predictions[i])
        
        results.append(result)
    
    return pd.DataFrame(results)
