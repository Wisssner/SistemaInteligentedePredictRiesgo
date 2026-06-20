/*
 * modelo_edge.h
 * Sistema Inteligente de Riego -- Parques Urbanos
 * Generado automaticamente por proyecto_ml_completo.py
 *
 * Algoritmo : Decision Tree Classifier (max_depth=4)
 * Accuracy  : 66.25 %   (test 30 %)
 * Nodos     : 31
 * Features  : Soil Moisture | Air temperature | Air humidity
 * Clases    : 0 = OFF (no regar)  |  1 = ON (activar valvula)
 *
 * -- USO EN ESP32 (Arduino IDE / PlatformIO) ----------------------
 * #include "modelo_edge.h"
 *
 * void loop() {
 *   float sm   = leerHumedadSuelo();   // 0-100 %
 *   float temp = dht.readTemperature();
 *   float hum  = dht.readHumidity();
 *
 *   int decision = predict(sm, temp, hum);
 *   digitalWrite(VALVE_PIN, decision);  // 1=abierta  0=cerrada
 * }
 */

#ifndef MODELO_EDGE_H
#define MODELO_EDGE_H

// -- Metadatos del modelo -------------------------------------------
#define MODEL_N_FEATURES  3
#define MODEL_MAX_DEPTH   4
#define MODEL_N_NODES     31
#define MODEL_ACCURACY    0.662500f

// -- Indices de features --------------------------------------------
#define IDX_SOIL_MOISTURE                  0
#define IDX_AIR_TEMPERATURE_C              1
#define IDX_AIR_HUMIDITY_PCT               2

/**
 * @brief Infiere si se debe activar el riego (Edge AI, 100 % offline).
 * @param Soil_Moisture                    Humedad del suelo (0-100 %)
 * @param Air_temperature_C                Temperatura del aire (grados C)
 * @param Air_humidity_pct                 Humedad relativa del aire (0-100 %)
 * @return  1 = abrir valvula (ON)  |  0 = cerrar valvula (OFF)
 */
inline int predict(float Soil_Moisture, float Air_temperature_C, float Air_humidity_pct) {
    if (Soil_Moisture <= 59.500000f) {
        if (Soil_Moisture <= 20.500000f) {
            if (Air_temperature_C <= 29.695001f) {
                if (Air_temperature_C <= 29.575000f) {
                    return 1;  // ON   (confianza 75%)
                } else {
                    return 0;  // OFF  (confianza 66%)
                }
            } else {
                if (Soil_Moisture <= 18.500000f) {
                    return 1;  // ON   (confianza 79%)
                } else {
                    return 1;  // ON   (confianza 65%)
                }
            }
        } else {
            if (Air_temperature_C <= 11.770000f) {
                if (Soil_Moisture <= 25.000000f) {
                    return 1;  // ON   (confianza 100%)
                } else {
                    return 0;  // OFF  (confianza 100%)
                }
            } else {
                if (Air_temperature_C <= 13.765000f) {
                    return 1;  // ON   (confianza 78%)
                } else {
                    return 1;  // ON   (confianza 59%)
                }
            }
        }
    } else {
        if (Soil_Moisture <= 69.500000f) {
            if (Soil_Moisture <= 61.500000f) {
                if (Air_humidity_pct <= 2.495000f) {
                    return 1;  // ON   (confianza 80%)
                } else {
                    return 0;  // OFF  (confianza 65%)
                }
            } else {
                if (Air_temperature_C <= 20.705000f) {
                    return 0;  // OFF  (confianza 66%)
                } else {
                    return 0;  // OFF  (confianza 62%)
                }
            }
        } else {
            if (Air_humidity_pct <= 73.240002f) {
                if (Air_humidity_pct <= 73.055000f) {
                    return 0;  // OFF  (confianza 68%)
                } else {
                    return 1;  // ON   (confianza 85%)
                }
            } else {
                if (Air_humidity_pct <= 87.584999f) {
                    return 0;  // OFF  (confianza 75%)
                } else {
                    return 0;  // OFF  (confianza 70%)
                }
            }
        }
    }
}

#endif  // MODELO_EDGE_H