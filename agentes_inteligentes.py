"""
================================================================================
MÓDULO DE AGENTES INTELIGENTES CON IA GENERATIVA
Sistema de Predicción de Riego - Agentes Basados en Reglas Expertas + Gemini AI
================================================================================
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import os
from dotenv import load_dotenv
load_dotenv()
# Gemini AI Integration
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️ google-generativeai no instalado. Ejecuta: pip install google-generativeai")

# Configuración de Gemini
GEMINI_API_KEY = os.getenv('GOOGLE_AI_API_KEY', 'AIzaSyAi5MJIR_5746NNuqp6-wGDe0H8ZlJS0Pw')
GEMINI_MODEL = os.getenv('GOOGLE_AI_MODEL', 'gemini-2.5-flash')
GEMINI_ENABLED = os.getenv('GOOGLE_AI_ENABLED', 'true').lower() == 'true'

if GEMINI_AVAILABLE and GEMINI_ENABLED and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel(GEMINI_MODEL)
        print(f"✅ Gemini AI activado: {GEMINI_MODEL}")
    except Exception as e:
        GEMINI_ENABLED = False
        print(f"⚠️ Error al configurar Gemini: {e}")
else:
    GEMINI_ENABLED = False


class AgenteRecomendaciones:
    """
    Agente Inteligente de Recomendaciones Agrícolas
    
    Proporciona recomendaciones personalizadas basadas en:
    - Condiciones climáticas actuales
    - Tipo de cultivo y etapa de crecimiento
    - Tipo de suelo
    - Índice de humedad (MOI)
    """
    
    def __init__(self):
        # Base de conocimiento de cultivos
        self.cultivos_db = {
            'Wheat': {
                'temp_optima': (15, 25),
                'humedad_optima': (50, 70),
                'moi_optimo': (40, 60),
                'plagas_comunes': ['Pulgón', 'Roya', 'Mosca de Hesse'],
                'fertilizacion': {
                    'Germination': 'Nitrógeno bajo, Fósforo alto',
                    'Vegetative': 'Nitrógeno alto, Potasio medio',
                    'Flowering': 'Nitrógeno medio, Potasio alto',
                    'Ripening': 'Fertilización mínima'
                }
            },
            'Rice': {
                'temp_optima': (20, 35),
                'humedad_optima': (70, 90),
                'moi_optimo': (70, 90),
                'plagas_comunes': ['Barrenador del tallo', 'Gorgojo del agua'],
                'fertilizacion': {
                    'Germination': 'Nitrógeno medio, Fósforo alto',
                    'Vegetative': 'Nitrógeno alto, Potasio alto',
                    'Flowering': 'Nitrógeno bajo, Potasio medio',
                    'Ripening': 'Sin fertilización'
                }
            },
            'Maize': {
                'temp_optima': (18, 30),
                'humedad_optima': (55, 75),
                'moi_optimo': (45, 65),
                'plagas_comunes': ['Gusano cogollero', 'Barrenador europeo'],
                'fertilizacion': {
                    'Germination': 'Nitrógeno bajo, Fósforo alto',
                    'Vegetative': 'Nitrógeno muy alto, Potasio medio',
                    'Flowering': 'Nitrógeno alto, Potasio alto',
                    'Ripening': 'Potasio medio'
                }
            },
            'Sugarcane': {
                'temp_optima': (25, 35),
                'humedad_optima': (60, 80),
                'moi_optimo': (55, 75),
                'plagas_comunes': ['Barrenador de la caña', 'Pulgón amarillo'],
                'fertilizacion': {
                    'Germination': 'Nitrógeno medio, Fósforo alto',
                    'Vegetative': 'Nitrógeno muy alto, Potasio alto',
                    'Flowering': 'Nitrógeno medio, Potasio muy alto',
                    'Ripening': 'Potasio medio'
                }
            },
            'Cotton': {
                'temp_optima': (21, 30),
                'humedad_optima': (50, 70),
                'moi_optimo': (40, 60),
                'plagas_comunes': ['Gusano rosado', 'Picudo del algodón'],
                'fertilizacion': {
                    'Germination': 'Nitrógeno bajo, Fósforo alto',
                    'Vegetative': 'Nitrógeno alto, Potasio medio',
                    'Flowering': 'Nitrógeno medio, Potasio alto',
                    'Ripening': 'Potasio bajo'
                }
            }
        }
        
        # Base de conocimiento de suelos
        self.suelos_db = {
            'Black Soil': {
                'retencion_agua': 'alta',
                'drenaje': 'medio',
                'nutrientes': 'ricos',
                'recomendacion': 'Excelente para cultivos de larga duración'
            },
            'Clay Soil': {
                'retencion_agua': 'muy alta',
                'drenaje': 'bajo',
                'nutrientes': 'medios',
                'recomendacion': 'Requiere manejo cuidadoso del riego'
            },
            'Red Soil': {
                'retencion_agua': 'media',
                'drenaje': 'alto',
                'nutrientes': 'bajos',
                'recomendacion': 'Necesita fertilización regular'
            },
            'Loamy Soil': {
                'retencion_agua': 'media',
                'drenaje': 'medio',
                'nutrientes': 'altos',
                'recomendacion': 'Ideal para la mayoría de cultivos'
            },
            'Sandy Soil': {
                'retencion_agua': 'baja',
                'drenaje': 'muy alto',
                'nutrientes': 'bajos',
                'recomendacion': 'Requiere riego frecuente y fertilización'
            }
        }
        
        # Configuración de Gemini
        self.use_gemini = GEMINI_ENABLED
    
    def calcular_score_salud(self, crop_id: str, temp: float, humedad: float, 
                            moi: float) -> Tuple[int, str]:
        """
        Calcula un score de salud del cultivo (0-100)
        
        Returns:
            Tuple[int, str]: (score, descripción)
        """
        if crop_id not in self.cultivos_db:
            return 50, "Cultivo no reconocido"
        
        cultivo = self.cultivos_db[crop_id]
        score = 100
        problemas = []
        
        # Evaluar temperatura
        temp_min, temp_max = cultivo['temp_optima']
        if temp < temp_min - 5 or temp > temp_max + 5:
            score -= 30
            problemas.append("temperatura crítica")
        elif temp < temp_min or temp > temp_max:
            score -= 15
            problemas.append("temperatura subóptima")
        
        # Evaluar humedad
        hum_min, hum_max = cultivo['humedad_optima']
        if humedad < hum_min - 15 or humedad > hum_max + 15:
            score -= 25
            problemas.append("humedad crítica")
        elif humedad < hum_min or humedad > hum_max:
            score -= 10
            problemas.append("humedad subóptima")
        
        # Evaluar MOI
        moi_min, moi_max = cultivo['moi_optimo']
        if moi < moi_min - 15 or moi > moi_max + 15:
            score -= 25
            problemas.append("humedad del suelo crítica")
        elif moi < moi_min or moi > moi_max:
            score -= 10
            problemas.append("humedad del suelo subóptima")
        
        score = max(0, min(100, score))
        
        if score >= 85:
            estado = "Excelente - Condiciones óptimas"
        elif score >= 70:
            estado = "Bueno - Condiciones favorables"
        elif score >= 50:
            estado = f"Regular - {', '.join(problemas)}"
        elif score >= 30:
            estado = f"Malo - {', '.join(problemas)}"
        else:
            estado = f"Crítico - {', '.join(problemas)}"
        
        return score, estado
    
    def generar_recomendaciones(self, crop_id: str, soil_type: str, 
                               seedling_stage: str, moi: float, 
                               temp: float, humedad: float) -> Dict:
        """
        Genera recomendaciones personalizadas basadas en las condiciones actuales
        
        Returns:
            Dict con recomendaciones categorizadas
        """
        recomendaciones = {
            'fertilizacion': [],
            'plagas': [],
            'manejo': [],
            'riego': [],
            'prioridades': []
        }
        
        # Obtener información del cultivo
        if crop_id in self.cultivos_db:
            cultivo = self.cultivos_db[crop_id]
            
            # Recomendaciones de fertilización
            if seedling_stage in cultivo['fertilizacion']:
                fertilizacion = cultivo['fertilizacion'][seedling_stage]
                recomendaciones['fertilizacion'].append({
                    'titulo': f'Fertilización para etapa {seedling_stage}',
                    'descripcion': fertilizacion,
                    'prioridad': 'media',
                    'accion': 'Aplicar según calendario de fertilización'
                })
            
            # Alertas de plagas basadas en condiciones
            temp_min, temp_max = cultivo['temp_optima']
            hum_min, hum_max = cultivo['humedad_optima']
            
            if temp > temp_max and humedad > hum_max:
                recomendaciones['plagas'].append({
                    'titulo': 'Riesgo elevado de plagas',
                    'descripcion': f'Condiciones favorables para: {", ".join(cultivo["plagas_comunes"][:2])}',
                    'prioridad': 'alta',
                    'accion': 'Inspeccionar cultivo y aplicar control preventivo'
                })
            elif temp > temp_max or humedad > hum_max:
                recomendaciones['plagas'].append({
                    'titulo': 'Monitoreo de plagas recomendado',
                    'descripcion': 'Condiciones moderadamente favorables para plagas',
                    'prioridad': 'media',
                    'accion': 'Realizar inspección visual regular'
                })
        
        # Recomendaciones de suelo
        if soil_type in self.suelos_db:
            suelo = self.suelos_db[soil_type]
            recomendaciones['manejo'].append({
                'titulo': f'Manejo de suelo {soil_type}',
                'descripcion': suelo['recomendacion'],
                'prioridad': 'baja',
                'accion': f'Drenaje {suelo["drenaje"]}, retención {suelo["retencion_agua"]}'
            })
        
        # Recomendaciones de riego basadas en MOI
        if moi < 30:
            recomendaciones['riego'].append({
                'titulo': 'Riego urgente necesario',
                'descripcion': f'MOI crítico: {moi:.1f}',
                'prioridad': 'alta',
                'accion': 'Regar inmediatamente - Suelo muy seco'
            })
        elif moi < 40:
            recomendaciones['riego'].append({
                'titulo': 'Riego recomendado pronto',
                'descripcion': f'MOI bajo: {moi:.1f}',
                'prioridad': 'media',
                'accion': 'Programar riego en las próximas 24 horas'
            })
        elif moi > 80:
            recomendaciones['riego'].append({
                'titulo': 'Evitar riego',
                'descripcion': f'MOI alto: {moi:.1f}',
                'prioridad': 'media',
                'accion': 'Suelo saturado - Riesgo de pudrición de raíces'
            })
        
        # Recomendaciones de temperatura
        if crop_id in self.cultivos_db:
            temp_min, temp_max = self.cultivos_db[crop_id]['temp_optima']
            if temp < temp_min - 5:
                recomendaciones['manejo'].append({
                    'titulo': 'Protección contra frío',
                    'descripcion': f'Temperatura muy baja: {temp}°C',
                    'prioridad': 'alta',
                    'accion': 'Considerar protección térmica o cobertura'
                })
            elif temp > temp_max + 5:
                recomendaciones['manejo'].append({
                    'titulo': 'Protección contra calor',
                    'descripcion': f'Temperatura muy alta: {temp}°C',
                    'prioridad': 'alta',
                    'accion': 'Aumentar frecuencia de riego, considerar sombreado'
                })
        
        # Generar lista de prioridades
        todas_recomendaciones = []
        for categoria in ['fertilizacion', 'plagas', 'manejo', 'riego']:
            todas_recomendaciones.extend(recomendaciones[categoria])
        
        # Ordenar por prioridad
        prioridad_orden = {'alta': 0, 'media': 1, 'baja': 2}
        todas_recomendaciones.sort(key=lambda x: prioridad_orden[x['prioridad']])
        recomendaciones['prioridades'] = todas_recomendaciones[:5]  # Top 5
        
        # Enriquecer con Gemini AI si está disponible
        if self.use_gemini:
            recomendaciones = self._enriquecer_con_gemini(
                recomendaciones, crop_id, soil_type, seedling_stage, 
                moi, temp, humedad
            )
        
        return recomendaciones
    
    def _enriquecer_con_gemini(self, recomendaciones: Dict, crop_id: str, 
                               soil_type: str, seedling_stage: str,
                               moi: float, temp: float, humedad: float) -> Dict:
        """
        Usa Gemini AI para generar recomendaciones contextuales adicionales
        """
        try:
            prompt = f"""
Eres un experto agrónomo especializado en agricultura de precisión. Analiza las siguientes condiciones:

**Cultivo:** {crop_id}
**Tipo de Suelo:** {soil_type}
**Etapa de Crecimiento:** {seedling_stage}
**Condiciones Actuales:**
- Temperatura: {temp}°C
- Humedad Ambiental: {humedad}%
- MOI (Índice de Humedad del Suelo): {moi}

**Recomendaciones del Sistema Experto:**
{json.dumps(recomendaciones, indent=2, ensure_ascii=False)}

Proporciona:
1. **Análisis Contextual**: Evalúa si hay condiciones anómalas o preocupantes
2. **Recomendación Prioritaria**: Una acción específica y urgente (máximo 2 líneas)
3. **Consejo del Experto**: Un insight valioso basado en tu experiencia (máximo 2 líneas)

Formato de respuesta (JSON):
{{
  "analisis": "texto breve",
  "recomendacion_prioritaria": "texto breve",
  "consejo_experto": "texto breve",
  "nivel_alerta": "bajo|medio|alto"
}}
"""
            
            response = gemini_model.generate_content(prompt)
            
            # Extraer JSON de la respuesta
            response_text = response.text.strip()
            
            # Intentar extraer JSON si está en un bloque de código
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0].strip()
            
            gemini_insights = json.loads(response_text)
            
            # Añadir insights de Gemini a las recomendaciones
            recomendaciones['gemini_insights'] = gemini_insights
            
            # Añadir recomendación prioritaria de Gemini si es relevante
            if gemini_insights.get('nivel_alerta') in ['medio', 'alto']:
                recomendaciones['prioridades'].insert(0, {
                    'titulo': '🤖 Recomendación IA Gemini',
                    'descripcion': gemini_insights.get('recomendacion_prioritaria', ''),
                    'prioridad': 'alta' if gemini_insights.get('nivel_alerta') == 'alto' else 'media',
                    'accion': gemini_insights.get('consejo_experto', '')
                })
            
        except Exception as e:
            print(f"⚠️ Error al usar Gemini: {e}")
            recomendaciones['gemini_insights'] = {
                'error': str(e),
                'analisis': 'No disponible'
            }
        
        return recomendaciones
    
    def evaluate_with_gemini(self, temp, humidity, moi, crop, soil, stage):
        try:
            prompt = f"""
Eres un experto agrónomo. Analiza estas condiciones de cultivo:

**Cultivo:** {crop}
**Tipo de Suelo:** {soil}
**Etapa de Crecimiento:** {stage}
**Condiciones Actuales:**
- Temperatura: {temp}°C
- Humedad Ambiental: {humidity}%
- Humedad del Suelo (MOI): {moi}%

Basándote en estos datos, determina si el cultivo necesita riego.

Formato de respuesta (JSON):
{{
    "Regar": "Requiere Riego o No Requiere Riego",
    "Porcentaje_Riego": número entre 0-100,
    "Porcentaje_No_Riego": número entre 0-100
}}
"""
        
            response = gemini_model.generate_content(prompt)
            response_text = response.text.strip()
        
            # Extraer JSON si viene en un bloque de código
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0].strip()

            return json.loads(response_text)

        except Exception as e:
            print(f"⚠️ Error al usar Gemini: {e}")
            return {
                "error": "Error al llamar a Gemini",
                "detalle": str(e),
            }

class AgenteOptimizacion:
    """
    Agente Inteligente de Optimización de Riego
    
    Genera calendarios de riego optimizados y alertas predictivas basadas en:
    - Predicciones del modelo ML
    - Datos históricos
    - Tendencias climáticas
    - Tipo de cultivo y etapa de crecimiento
    """
    
    def __init__(self):
        # Parámetros de riego por cultivo (litros/m²)
        self.riego_base = {
            'Wheat': {'Germination': 15, 'Vegetative': 25, 'Flowering': 30, 'Ripening': 20},
            'Rice': {'Germination': 30, 'Vegetative': 40, 'Flowering': 45, 'Ripening': 35},
            'Maize': {'Germination': 18, 'Vegetative': 28, 'Flowering': 35, 'Ripening': 22},
            'Sugarcane': {'Germination': 25, 'Vegetative': 35, 'Flowering': 40, 'Ripening': 30},
            'Cotton': {'Germination': 20, 'Vegetative': 30, 'Flowering': 35, 'Ripening': 25}
        }
        
        # Horarios óptimos de riego (evitar horas de calor)
        self.horarios_optimos = ['06:00', '07:00', '18:00', '19:00']
        
        # Configuración de Gemini
        self.use_gemini = GEMINI_ENABLED
    
    def calcular_volumen_riego(self, crop_id: str, seedling_stage: str, 
                               temp: float, humedad: float, moi: float) -> float:
        """
        Calcula el volumen de agua recomendado (litros/m²)
        """
        if crop_id not in self.riego_base:
            base = 25  # Default
        else:
            base = self.riego_base[crop_id].get(seedling_stage, 25)
        
        # Ajustar por temperatura
        if temp > 30:
            base *= 1.3
        elif temp > 25:
            base *= 1.15
        elif temp < 15:
            base *= 0.8
        
        # Ajustar por humedad ambiental
        if humedad < 40:
            base *= 1.2
        elif humedad > 70:
            base *= 0.85
        
        # Ajustar por MOI actual
        if moi < 30:
            base *= 1.4
        elif moi < 40:
            base *= 1.2
        elif moi > 70:
            base *= 0.5
        elif moi > 60:
            base *= 0.7
        
        return round(base, 1)
    
    def generar_calendario_riego(self, crop_id: str, seedling_stage: str,
                                 moi: float, temp: float, humedad: float,
                                 dias: int = 7) -> Dict:
        """
        Genera un calendario de riego optimizado para los próximos días
        
        Returns:
            Dict con calendario y recomendaciones
        """
        calendario = []
        hoy = datetime.now()
        
        # Simular tendencia (en producción, usar datos reales de pronóstico)
        for i in range(dias):
            fecha = hoy + timedelta(days=i)
            
            # Simular variación de condiciones
            temp_dia = temp + np.random.uniform(-3, 3)
            humedad_dia = humedad + np.random.uniform(-10, 10)
            moi_dia = moi - (i * 5) if moi > 40 else moi  # MOI decrece sin riego
            
            # Determinar si se necesita riego
            necesita_riego = moi_dia < 45 or temp_dia > 30
            
            if necesita_riego:
                volumen = self.calcular_volumen_riego(
                    crop_id, seedling_stage, temp_dia, humedad_dia, moi_dia
                )
                horario = np.random.choice(self.horarios_optimos)
                
                calendario.append({
                    'fecha': fecha.strftime('%Y-%m-%d'),
                    'dia_semana': fecha.strftime('%A'),
                    'horario': horario,
                    'volumen': volumen,
                    'moi_estimado': round(moi_dia, 1),
                    'temp_estimada': round(temp_dia, 1),
                    'prioridad': 'alta' if moi_dia < 35 else 'media',
                    'notas': self._generar_notas_riego(moi_dia, temp_dia, humedad_dia)
                })
                
                # Actualizar MOI después del riego
                moi_dia += 25
            else:
                calendario.append({
                    'fecha': fecha.strftime('%Y-%m-%d'),
                    'dia_semana': fecha.strftime('%A'),
                    'horario': 'No requerido',
                    'volumen': 0,
                    'moi_estimado': round(moi_dia, 1),
                    'temp_estimada': round(temp_dia, 1),
                    'prioridad': 'baja',
                    'notas': 'Condiciones óptimas - Sin riego necesario'
                })
        
        # Calcular estadísticas
        total_agua = sum(dia['volumen'] for dia in calendario)
        dias_riego = sum(1 for dia in calendario if dia['volumen'] > 0)
        
        return {
            'calendario': calendario,
            'estadisticas': {
                'total_agua_semana': round(total_agua, 1),
                'dias_riego': dias_riego,
                'promedio_por_riego': round(total_agua / dias_riego, 1) if dias_riego > 0 else 0,
                'eficiencia': 'Alta' if dias_riego <= 4 else 'Media'
            }
        }
    
    def _generar_notas_riego(self, moi: float, temp: float, humedad: float) -> str:
        """Genera notas contextuales para cada sesión de riego"""
        notas = []
        
        if moi < 30:
            notas.append("Suelo muy seco")
        elif moi < 40:
            notas.append("Suelo seco")
        
        if temp > 32:
            notas.append("Temperatura alta - Regar temprano")
        elif temp > 28:
            notas.append("Evitar horas de calor")
        
        if humedad < 40:
            notas.append("Baja humedad ambiental")
        
        return " | ".join(notas) if notas else "Condiciones normales"
    
    def generar_alertas(self, crop_id: str, moi: float, temp: float, 
                       humedad: float, prediccion_riego: int) -> List[Dict]:
        """
        Genera alertas predictivas basadas en condiciones actuales y predicción
        
        Returns:
            Lista de alertas con nivel de urgencia
        """
        alertas = []
        
        # Alerta de MOI crítico
        if moi < 25:
            alertas.append({
                'tipo': 'Crítico',
                'titulo': 'Nivel de humedad del suelo crítico',
                'mensaje': f'MOI actual: {moi:.1f} - Riego urgente requerido',
                'urgencia': 'alta',
                'icono': 'exclamation-triangle-fill',
                'color': 'danger'
            })
        elif moi < 35 and prediccion_riego == 1:
            alertas.append({
                'tipo': 'Advertencia',
                'titulo': 'Riego recomendado',
                'mensaje': f'MOI: {moi:.1f} - El modelo predice necesidad de riego',
                'urgencia': 'media',
                'icono': 'exclamation-circle-fill',
                'color': 'warning'
            })
        
        # Alerta de temperatura extrema
        if temp > 35:
            alertas.append({
                'tipo': 'Advertencia',
                'titulo': 'Temperatura extrema',
                'mensaje': f'{temp}°C - Aumentar frecuencia de riego',
                'urgencia': 'alta',
                'icono': 'thermometer-high',
                'color': 'danger'
            })
        elif temp < 10:
            alertas.append({
                'tipo': 'Advertencia',
                'titulo': 'Temperatura baja',
                'mensaje': f'{temp}°C - Reducir riego, riesgo de heladas',
                'urgencia': 'media',
                'icono': 'thermometer-low',
                'color': 'info'
            })
        
        # Alerta de humedad extrema
        if humedad < 30:
            alertas.append({
                'tipo': 'Información',
                'titulo': 'Humedad ambiental muy baja',
                'mensaje': f'{humedad}% - Mayor evapotranspiración',
                'urgencia': 'media',
                'icono': 'droplet-half',
                'color': 'warning'
            })
        elif humedad > 85:
            alertas.append({
                'tipo': 'Información',
                'titulo': 'Humedad ambiental muy alta',
                'mensaje': f'{humedad}% - Riesgo de enfermedades fúngicas',
                'urgencia': 'baja',
                'icono': 'cloud-rain-fill',
                'color': 'info'
            })
        
        # Alerta de eficiencia
        if moi > 75 and prediccion_riego == 1:
            alertas.append({
                'tipo': 'Optimización',
                'titulo': 'Posible sobre-riego',
                'mensaje': f'MOI alto ({moi:.1f}) - Considerar reducir frecuencia',
                'urgencia': 'baja',
                'icono': 'info-circle-fill',
                'color': 'success'
            })
        
        # Validar con Gemini AI si está disponible
        if self.use_gemini:
            alertas = self._validar_alertas_con_gemini(
                alertas, crop_id, moi, temp, humedad
            )
        
        return alertas
    
    def _validar_alertas_con_gemini(self, alertas: List[Dict], crop_id: str,
                                     moi: float, temp: float, humedad: float) -> List[Dict]:
        """
        Usa Gemini AI para detectar anomalías y validar alertas
        """
        try:
            prompt = f"""
Eres un sistema de detección de anomalías climáticas para agricultura.

**Cultivo:** {crop_id}
**Condiciones Actuales:**
- Temperatura: {temp}°C
- Humedad: {humedad}%
- MOI: {moi}

**Alertas del Sistema:**
{json.dumps(alertas, indent=2, ensure_ascii=False)}

¿Detectas alguna anomalía climática crítica que requiera atención inmediata?
Si es así, proporciona una alerta adicional.

Formato de respuesta (JSON):
{{
  "anomalia_detectada": true/false,
  "alerta_adicional": {{
    "tipo": "texto",
    "titulo": "texto",
    "mensaje": "texto",
    "urgencia": "baja|media|alta",
    "icono": "nombre-icono",
    "color": "danger|warning|info"
  }}
}}
"""
            
            response = gemini_model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Extraer JSON
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0].strip()
            
            gemini_validation = json.loads(response_text)
            
            # Añadir alerta de Gemini si detectó anomalía
            if gemini_validation.get('anomalia_detectada') and gemini_validation.get('alerta_adicional'):
                alerta_gemini = gemini_validation['alerta_adicional']
                alerta_gemini['tipo'] = '🤖 ' + alerta_gemini.get('tipo', 'Alerta IA')
                alertas.insert(0, alerta_gemini)
            
        except Exception as e:
            print(f"⚠️ Error al validar con Gemini: {e}")
        
        return alertas
    
    def calcular_metricas_eficiencia(self, calendario: List[Dict]) -> Dict:
        """
        Calcula métricas de eficiencia del plan de riego
        """
        if not calendario:
            return {}
        
        total_agua = sum(dia['volumen'] for dia in calendario)
        dias_riego = sum(1 for dia in calendario if dia['volumen'] > 0)
        
        return {
            'consumo_total': round(total_agua, 1),
            'consumo_promedio_dia': round(total_agua / len(calendario), 1),
            'dias_con_riego': dias_riego,
            'dias_sin_riego': len(calendario) - dias_riego,
            'eficiencia_porcentaje': round((1 - dias_riego / len(calendario)) * 100, 1),
            'clasificacion': self._clasificar_eficiencia(dias_riego, len(calendario))
        }
    
    def _clasificar_eficiencia(self, dias_riego: int, total_dias: int) -> str:
        """Clasifica la eficiencia del plan de riego"""
        ratio = dias_riego / total_dias
        if ratio <= 0.4:
            return 'Excelente - Uso eficiente del agua'
        elif ratio <= 0.6:
            return 'Bueno - Uso moderado del agua'
        elif ratio <= 0.8:
            return 'Regular - Considerar optimización'
        else:
            return 'Bajo - Revisar estrategia de riego'


# Funciones de utilidad para integración con Flask
def obtener_recomendaciones_completas(crop_id: str, soil_type: str, 
                                     seedling_stage: str, moi: float,
                                     temp: float, humedad: float) -> Dict:
    """
    Función helper para obtener todas las recomendaciones de ambos agentes
    """
    agente_rec = AgenteRecomendaciones()
    agente_opt = AgenteOptimizacion()
    
    # Score de salud
    score, estado = agente_rec.calcular_score_salud(crop_id, temp, humedad, moi)
    
    # Recomendaciones
    recomendaciones = agente_rec.generar_recomendaciones(
        crop_id, soil_type, seedling_stage, moi, temp, humedad
    )
    
    # Calendario de riego
    calendario_data = agente_opt.generar_calendario_riego(
        crop_id, seedling_stage, moi, temp, humedad
    )
    
    # Alertas (asumiendo predicción de riego basada en MOI)
    prediccion_riego = 1 if moi < 45 else 0
    alertas = agente_opt.generar_alertas(crop_id, moi, temp, humedad, prediccion_riego)
    
    return {
        'salud': {
            'score': score,
            'estado': estado
        },
        'recomendaciones': recomendaciones,
        'calendario': calendario_data,
        'alertas': alertas,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

