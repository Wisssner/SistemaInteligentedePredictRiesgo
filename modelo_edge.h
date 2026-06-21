/*
 * modelo_edge.h
 * Sistema Inteligente de Riego -- Parques Urbanos
 * Generado automaticamente por proyecto_ml_completo.py
 *
 * Algoritmo : Decision Tree Classifier (max_depth=8)  [TinyML pragmatico]
 * Accuracy  : 91.08 %   (test 30 %)
 * Nodos     : 147
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
#define MODEL_MAX_DEPTH   8
#define MODEL_N_NODES     147
#define MODEL_ACCURACY    0.910833f

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
                    if (Time <= 9.500000f) {
                        return 0;  // OFF  (confianza 100%)
                    } else {
                        if (Time <= 39.500000f) {
                            if (Temperature <= 1.500000f) {
                                if (Temperature <= 0.500000f) {
                                    return 0;  // OFF  (confianza 100%)
                                } else {
                                    return 0;  // OFF  (confianza 75%)
                                }
                            } else {
                                if (Temperature <= 7.500000f) {
                                    return 1;  // ON   (confianza 100%)
                                } else {
                                    return 1;  // ON   (confianza 52%)
                                }
                            }
                        } else {
                            return 1;  // ON   (confianza 100%)
                        }
                    }
                } else {
                    if (Time <= 39.500000f) {
                        if (Time <= 9.500000f) {
                            return 0;  // OFF  (confianza 100%)
                        } else {
                            if (Temperature <= 1.500000f) {
                                if (Temperature <= 0.500000f) {
                                    return 0;  // OFF  (confianza 100%)
                                } else {
                                    return 0;  // OFF  (confianza 80%)
                                }
                            } else {
                                if (Temperature <= 7.500000f) {
                                    return 1;  // ON   (confianza 93%)
                                } else {
                                    return 1;  // ON   (confianza 61%)
                                }
                            }
                        }
                    } else {
                        if (Temperature <= 2.500000f) {
                            if (Temperature <= 0.500000f) {
                                return 0;  // OFF  (confianza 100%)
                            } else {
                                if (Time <= 45.500000f) {
                                    return 0;  // OFF  (confianza 85%)
                                } else {
                                    return 0;  // OFF  (confianza 93%)
                                }
                            }
                        } else {
                            if (Temperature <= 6.500000f) {
                                if (Time <= 88.500000f) {
                                    return 0;  // OFF  (confianza 79%)
                                } else {
                                    return 0;  // OFF  (confianza 58%)
                                }
                            } else {
                                if (Soil_Moisture <= 34.500000f) {
                                    return 0;  // OFF  (confianza 71%)
                                } else {
                                    return 0;  // OFF  (confianza 62%)
                                }
                            }
                        }
                    }
                }
            } else {
                if (Temperature <= 25.500000f) {
                    if (Soil_Moisture <= 20.500000f) {
                        if (Time <= 39.500000f) {
                            if (Temperature <= 23.500000f) {
                                if (Temperature <= 17.500000f) {
                                    return 1;  // ON   (confianza 85%)
                                } else {
                                    return 1;  // ON   (confianza 100%)
                                }
                            } else {
                                if (Time <= 9.500000f) {
                                    return 1;  // ON   (confianza 100%)
                                } else {
                                    return 0;  // OFF  (confianza 70%)
                                }
                            }
                        } else {
                            return 1;  // ON   (confianza 100%)
                        }
                    } else {
                        if (Time <= 39.500000f) {
                            if (Temperature <= 23.500000f) {
                                if (Temperature <= 17.500000f) {
                                    return 1;  // ON   (confianza 90%)
                                } else {
                                    return 1;  // ON   (confianza 100%)
                                }
                            } else {
                                if (Soil_Moisture <= 49.500000f) {
                                    return 1;  // ON   (confianza 55%)
                                } else {
                                    return 1;  // ON   (confianza 100%)
                                }
                            }
                        } else {
                            if (Temperature <= 19.500000f) {
                                if (Temperature <= 16.500000f) {
                                    return 0;  // OFF  (confianza 59%)
                                } else {
                                    return 1;  // ON   (confianza 51%)
                                }
                            } else {
                                if (Temperature <= 24.500000f) {
                                    return 1;  // ON   (confianza 62%)
                                } else {
                                    return 1;  // ON   (confianza 74%)
                                }
                            }
                        }
                    }
                } else {
                    if (Temperature <= 29.500000f) {
                        if (Time <= 39.500000f) {
                            return 1;  // ON   (confianza 100%)
                        } else {
                            if (Soil_Moisture <= 20.500000f) {
                                return 1;  // ON   (confianza 100%)
                            } else {
                                if (Temperature <= 27.500000f) {
                                    return 1;  // ON   (confianza 79%)
                                } else {
                                    return 1;  // ON   (confianza 89%)
                                }
                            }
                        }
                    } else {
                        if (Soil_Moisture <= 49.500000f) {
                            if (Temperature <= 30.500000f) {
                                if (Time <= 45.500000f) {
                                    return 1;  // ON   (confianza 100%)
                                } else {
                                    return 1;  // ON   (confianza 95%)
                                }
                            } else {
                                return 1;  // ON   (confianza 100%)
                            }
                        } else {
                            if (Temperature <= 35.500000f) {
                                if (Time <= 38.500000f) {
                                    return 1;  // ON   (confianza 79%)
                                } else {
                                    return 1;  // ON   (confianza 99%)
                                }
                            } else {
                                return 1;  // ON   (confianza 100%)
                            }
                        }
                    }
                }
            }
        } else {
            if (Time <= 39.500000f) {
                if (Time <= 9.500000f) {
                    if (Soil_Moisture <= 69.500000f) {
                        if (Temperature <= 10.500000f) {
                            return 0;  // OFF  (confianza 100%)
                        } else {
                            return 1;  // ON   (confianza 100%)
                        }
                    } else {
                        return 0;  // OFF  (confianza 100%)
                    }
                } else {
                    return 0;  // OFF  (confianza 100%)
                }
            } else {
                if (Temperature <= 22.500000f) {
                    if (Temperature <= 6.500000f) {
                        if (Temperature <= 2.500000f) {
                            if (Temperature <= 0.500000f) {
                                return 0;  // OFF  (confianza 100%)
                            } else {
                                if (Time <= 50.500000f) {
                                    return 0;  // OFF  (confianza 88%)
                                } else {
                                    return 0;  // OFF  (confianza 94%)
                                }
                            }
                        } else {
                            if (Temperature <= 5.500000f) {
                                if (Soil_Moisture <= 81.500000f) {
                                    return 0;  // OFF  (confianza 85%)
                                } else {
                                    return 0;  // OFF  (confianza 77%)
                                }
                            } else {
                                if (Soil_Moisture <= 88.500000f) {
                                    return 0;  // OFF  (confianza 76%)
                                } else {
                                    return 0;  // OFF  (confianza 50%)
                                }
                            }
                        }
                    } else {
                        if (Temperature <= 13.500000f) {
                            if (Temperature <= 11.500000f) {
                                if (Time <= 88.500000f) {
                                    return 0;  // OFF  (confianza 67%)
                                } else {
                                    return 0;  // OFF  (confianza 78%)
                                }
                            } else {
                                if (Time <= 75.500000f) {
                                    return 0;  // OFF  (confianza 64%)
                                } else {
                                    return 0;  // OFF  (confianza 54%)
                                }
                            }
                        } else {
                            if (Temperature <= 20.500000f) {
                                if (Time <= 59.500000f) {
                                    return 1;  // ON   (confianza 52%)
                                } else {
                                    return 0;  // OFF  (confianza 54%)
                                }
                            } else {
                                if (Time <= 43.500000f) {
                                    return 0;  // OFF  (confianza 56%)
                                } else {
                                    return 1;  // ON   (confianza 60%)
                                }
                            }
                        }
                    }
                } else {
                    if (Temperature <= 28.500000f) {
                        if (Temperature <= 24.500000f) {
                            if (Time <= 89.500000f) {
                                if (Time <= 75.500000f) {
                                    return 1;  // ON   (confianza 71%)
                                } else {
                                    return 1;  // ON   (confianza 62%)
                                }
                            } else {
                                if (Temperature <= 23.500000f) {
                                    return 0;  // OFF  (confianza 88%)
                                } else {
                                    return 1;  // ON   (confianza 75%)
                                }
                            }
                        } else {
                            if (Temperature <= 26.500000f) {
                                if (Soil_Moisture <= 60.500000f) {
                                    return 1;  // ON   (confianza 57%)
                                } else {
                                    return 1;  // ON   (confianza 79%)
                                }
                            } else {
                                if (Time <= 43.500000f) {
                                    return 1;  // ON   (confianza 100%)
                                } else {
                                    return 1;  // ON   (confianza 83%)
                                }
                            }
                        }
                    } else {
                        if (Temperature <= 30.500000f) {
                            if (Time <= 63.500000f) {
                                if (Soil_Moisture <= 72.500000f) {
                                    return 1;  // ON   (confianza 98%)
                                } else {
                                    return 1;  // ON   (confianza 93%)
                                }
                            } else {
                                if (Soil_Moisture <= 76.500000f) {
                                    return 1;  // ON   (confianza 92%)
                                } else {
                                    return 1;  // ON   (confianza 86%)
                                }
                            }
                        } else {
                            return 1;  // ON   (confianza 100%)
                        }
                    }
                }
            }
        }
    } else {
        return 0;  // OFF  (confianza 100%)
    }
}

#endif  // MODELO_EDGE_H