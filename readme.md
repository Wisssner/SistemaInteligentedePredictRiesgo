# 🚀 Guía de Inicio - Sistema Inteligente de Predicción de Riego

## 📋 Resumen del Sistema

Tu sistema web ha sido transformado en un **Sistema Inteligente** con dos agentes de IA:

### 🤖 Agente 1: Agente de Recomendaciones Agrícolas
- **Función**: Proporciona consejos personalizados sobre cultivos
- **Capacidades**:
  - Calcula score de salud del cultivo (0-100)
  - Recomendaciones de fertilización según etapa de crecimiento
  - Alertas de plagas basadas en condiciones climáticas
  - Consejos de manejo de cultivo y suelo
  - Optimización de recursos

### 🤖 Agente 2: Agente de Optimización de Riego
- **Función**: Genera calendarios de riego optimizados
- **Capacidades**:
  - Calendario semanal con horarios óptimos
  - Cálculo de volumen de agua por sesión
  - Alertas predictivas de condiciones críticas
  - Métricas de eficiencia de uso de agua
  - Proyecciones de 7, 14 o 21 días

---

## 🛠️ Paso a Paso para Iniciar el Proyecto

### Paso 1: Instalar Dependencias

Abre PowerShell en la carpeta del proyecto y ejecuta:

```powershell
pip install -r requirements.txt
```

**Dependencias instaladas**:
- Flask 3.0.0 (servidor web)
- pandas 2.1.3 (procesamiento de datos)
- numpy 1.26.2 (cálculos numéricos)
- scikit-learn 1.3.2 (modelo ML)
- requests 2.31.0 (conexión Firebase)
- openpyxl 3.1.2 (lectura de Excel)
- python-dateutil 2.8.2+ (manejo de fechas)

### Paso 2: Verificar Estructura de Archivos

Asegúrate de tener esta estructura:

```
sistema_web_completo/
├── app.py                          # Aplicación Flask principal
├── agentes_inteligentes.py         # ✨ NUEVO: Módulo de agentes IA
├── proyecto_ml_completo.py         # Script de entrenamiento ML
├── requirements.txt                # Dependencias
├── dataSalvadora.xlsx             # Dataset de entrenamiento
├── dataSalvadorasintarget.xlsx    # Dataset sin target
├── modelos_guardados/
│   └── best_model_Decision_Tree_20251126_003051.pkl
├── templates/
│   ├── base.html
│   ├── index.html                  # ✨ ACTUALIZADO: Con nuevos agentes
│   ├── prediccion_individual.html
│   ├── prediccion_masiva.html
│   ├── dashboard.html
│   ├── analisis_vivo.html
│   ├── agente_recomendaciones.html # ✨ NUEVO: Interfaz Agente 1
│   └── agente_optimizacion.html    # ✨ NUEVO: Interfaz Agente 2
├── static/
│   └── (archivos CSS/JS si existen)
└── uploads/
```

### Paso 3: Iniciar el Servidor

En PowerShell, ejecuta:

```powershell
python app.py
```

Deberías ver:

```
 * Serving Flask app 'app'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
 * Running on http://192.168.x.x:5000
```

### Paso 4: Acceder al Sistema

Abre tu navegador y ve a:

```
http://localhost:5000
```

---

## 🎯 Cómo Usar los Agentes Inteligentes

### Opción 1: Agente de Recomendaciones Agrícolas

1. **Acceder**: Click en "🤖 Agente de Recomendaciones" en la página principal
2. **Configurar**:
   - Selecciona tipo de cultivo (Wheat, Rice, Maize, Sugarcane, Cotton)
   - Selecciona tipo de suelo (Black, Clay, Red, Loamy, Sandy)
   - Selecciona etapa de crecimiento (Germination, Vegetative, Flowering, Ripening)
   - Los valores de temperatura, humedad y MOI se cargan automáticamente de Firebase
3. **Generar**: Click en "Generar Recomendaciones"
4. **Revisar**:
   - **Score de Salud**: Gauge visual (0-100)
   - **Recomendaciones Prioritarias**: Alertas ordenadas por urgencia
   - **Tabs por Categoría**: Fertilización, Plagas, Manejo, Riego

### Opción 2: Agente de Optimización de Riego

1. **Acceder**: Click en "🤖 Agente de Optimización" en la página principal
2. **Configurar**:
   - Selecciona tipo de cultivo y etapa de crecimiento
   - Valores climáticos se cargan automáticamente
   - Selecciona días a proyectar (7, 14 o 21 días)
3. **Generar**: Click en "Generar Calendario de Riego"
4. **Revisar**:
   - **Alertas**: Condiciones críticas en tiempo real
   - **Estadísticas**: Total agua, días de riego, eficiencia
   - **Calendario**: Tabla con horarios, volúmenes y notas
   - **Métricas**: Gráfico de distribución y análisis detallado

---

## 🔧 Funcionalidades Existentes (Mantenidas)

### 1. Predicción Individual
- Predicción de necesidad de riego para un cultivo
- Integración con Firebase para datos en tiempo real
- Lógica difusa para cálculo automático de MOI

### 2. Predicción Masiva
- Carga de archivo Excel con múltiples cultivos
- Predicción batch para todos los registros
- Exportación de resultados

### 3. Dashboard
- Visualización de histórico de sensores Firebase
- Gráficos de temperatura y humedad

### 4. Análisis en Vivo
- Monitoreo en tiempo real
- Predicciones automáticas con datos de Firebase

---

## 📊 Arquitectura del Sistema Inteligente

```
┌─────────────────────────────────────────────────────────────┐
│                    SISTEMA INTELIGENTE                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Firebase   │───▶│  Flask App   │◀───│  Modelo ML   │  │
│  │  (IoT Data)  │    │   (app.py)   │    │ (Decision    │  │
│  └──────────────┘    └──────┬───────┘    │  Tree)       │  │
│                             │             └──────────────┘  │
│                             │                                │
│                    ┌────────▼────────┐                       │
│                    │  Agentes IA     │                       │
│                    │  (agentes_      │                       │
│                    │   inteligentes  │                       │
│                    │   .py)          │                       │
│                    └────────┬────────┘                       │
│                             │                                │
│              ┌──────────────┴──────────────┐                 │
│              │                             │                 │
│      ┌───────▼────────┐          ┌────────▼────────┐        │
│      │ Agente de      │          │ Agente de       │        │
│      │ Recomendaciones│          │ Optimización    │        │
│      │                │          │                 │        │
│      │ • Score Salud  │          │ • Calendario    │        │
│      │ • Fertilización│          │ • Alertas       │        │
│      │ • Plagas       │          │ • Eficiencia    │        │
│      │ • Manejo       │          │ • Proyecciones  │        │
│      └────────────────┘          └─────────────────┘        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧪 Pruebas Rápidas

### Probar Agente de Recomendaciones desde Python:

```python
from agentes_inteligentes import AgenteRecomendaciones

agente = AgenteRecomendaciones()

# Calcular score de salud
score, estado = agente.calcular_score_salud(
    crop_id='Wheat',
    temp=28,
    humedad=65,
    moi=45
)
print(f"Score: {score} - {estado}")

# Generar recomendaciones
recomendaciones = agente.generar_recomendaciones(
    crop_id='Wheat',
    soil_type='Black Soil',
    seedling_stage='Flowering',
    moi=45,
    temp=28,
    humedad=65
)
print(recomendaciones)
```

### Probar Agente de Optimización desde Python:

```python
from agentes_inteligentes import AgenteOptimizacion

agente = AgenteOptimizacion()

# Generar calendario
calendario = agente.generar_calendario_riego(
    crop_id='Wheat',
    seedling_stage='Flowering',
    moi=45,
    temp=28,
    humedad=65,
    dias=7
)
print(calendario)
```

---

## 🎨 Características de la Interfaz

### Diseño Moderno
- ✅ Bootstrap 5 para diseño responsive
- ✅ Bootstrap Icons para iconografía
- ✅ Chart.js para visualizaciones interactivas
- ✅ Código de colores por prioridad (rojo=alta, amarillo=media, verde=baja)
- ✅ Animaciones suaves y transiciones

### Experiencia de Usuario
- ✅ Datos precargados desde Firebase
- ✅ Validación de formularios
- ✅ Feedback visual inmediato
- ✅ Scroll automático a resultados
- ✅ Responsive para móviles y tablets

---

## 🔍 Solución de Problemas

### Error: "ModuleNotFoundError: No module named 'agentes_inteligentes'"
**Solución**: Asegúrate de estar en la carpeta correcta del proyecto.

### Error: "FileNotFoundError: best_model_Decision_Tree..."
**Solución**: Verifica que el archivo del modelo existe en `modelos_guardados/`

### Error: Firebase no responde
**Solución**: Verifica tu conexión a internet. El sistema funciona sin Firebase pero con datos por defecto.

### El servidor no inicia
**Solución**: 
1. Verifica que el puerto 5000 no esté en uso
2. Ejecuta: `netstat -ano | findstr :5000`
3. Si está ocupado, cambia el puerto en `app.py` línea final

---

## 📈 Próximos Pasos Sugeridos

1. **Personalizar Base de Conocimiento**: Edita `agentes_inteligentes.py` para añadir más cultivos o reglas
2. **Integrar Pronóstico del Clima**: Conecta con API de clima para proyecciones más precisas
3. **Añadir Persistencia**: Guarda histórico de recomendaciones en base de datos
4. **Notificaciones**: Implementa alertas por email o SMS
5. **Dashboard de Agentes**: Crea vista consolidada de ambos agentes

---

## 📞 Soporte

Si tienes problemas:
1. Revisa los logs en la consola donde ejecutaste `python app.py`
2. Verifica que todas las dependencias estén instaladas
3. Asegúrate de tener Python 3.8 o superior

---

## ✨ ¡Disfruta tu Sistema Inteligente!

Tu sistema ahora combina:
- 🧠 Machine Learning (Decision Tree)
- 🤖 Inteligencia Artificial (Agentes Basados en Reglas)
- 🌐 IoT (Firebase)
- 📊 Visualización de Datos
- 💧 Optimización de Recursos

**¡Feliz cultivo inteligente!** 🌱
