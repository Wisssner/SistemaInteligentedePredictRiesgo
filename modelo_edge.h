/*
 * modelo_edge.h
 * Sistema Inteligente de Riego -- Parques Urbanos
 * Generado automaticamente por proyecto_ml_completo.py
 *
 * Algoritmo : Decision Tree Classifier (max_depth=4)
 * Accuracy  : 87.19 %   (test 30 %)
 * Nodos     : 17
 * Features  : Soil Moisture | Temperature | Time
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
#define MODEL_N_NODES     17
#define MODEL_ACCURACY    0.871933f

// -- Indices de features --------------------------------------------
#define IDX_SOIL_MOISTURE                  0
#define IDX_TEMPERATURE                    1
#define IDX_TIME                           2

/**
 * @brief Infiere si se debe activar el riego (Edge AI, 100 % offline).
 * @param Soil_Moisture                    Humedad del suelo (0-100 %)
 * @param Temperature                      Temperatura ambiente (grados C)
 * @param Time                             Hora del dia / ciclo temporal
 * @return  1 = abrir valvula (ON)  |  0 = cerrar valvula (OFF)
 */
inline int predict(float Soil_Moisture, float Temperature, float Time) {
    if (Time <= 90.500000f) {
        if (Soil_Moisture <= 59.500000f) {
            if (Temperature <= 10.500000f) {
                if (Soil_Moisture <= 20.500000f) {
                    return 1;  // ON   (confianza 78%)
                } else {
                    return 0;  // OFF  (confianza 64%)
                }
            } else {
                if (Temperature <= 25.500000f) {
                    return 1;  // ON   (confianza 77%)
                } else {
                    return 1;  // ON   (confianza 98%)
                }
            }
        } else {
            if (Time <= 39.500000f) {
                if (Time <= 9.500000f) {
                    return 0;  // OFF  (confianza 74%)
                } else {
                    return 0;  // OFF  (confianza 100%)
                }
            } else {
                if (Temperature <= 22.500000f) {
                    return 0;  // OFF  (confianza 65%)
                } else {
                    return 1;  // ON   (confianza 93%)
                }
            }
        }
    } else {
        return 0;  // OFF  (confianza 100%)
    }
}

#endif  // MODELO_EDGE_H