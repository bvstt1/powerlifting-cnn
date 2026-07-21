# Guía para modelo binario de clasificación de Deadlifts con keypoints de esqueleto

## Contexto

- Dataset actual: 4 archivos CSV con levantamientos etiquetados (`front`, `lateral`, `left`, `right`)
- Columnas: video_id, movimiento, camara, archivo, resultado, label, codigos, criterios, etiquetador, timestamp
- Label: `1` = válido, `0` = inválido / nulo
- Resultado: `valido` / `invalido`
- Se usan keypoints de esqueleto extraídos de los videos como entrada al modelo

## Estado actual del dataset

| Archivo | Total | Válidos (label=1) | Inválidos (label=0) |
|---|---|---|---|
| `etiquetado_dl_front.csv` | 259 | 192 | 67 |
| `etiquetado_dl_lateral.csv` | 518 | 354 | 164 |
| `etiquetado_dl_left.csv` | 259 | 177 | 82 |
| `etiquetado_dl_right.csv` | 259 | 177 | 82 |

**Nota:** Los 82 inválidos en `left` y `right` son exactamente los mismos `video_id`.

## Recomendación de tamaño de dataset para modelo binario

### Objetivo ideal: ~1000–1500 levantamientos totales

| Clase | Cantidad recomendada | % del total |
|---|---|---|
| Válidos | 700–1000 | ~65–70% |
| Inválidos | 300–500 | ~30–35% |

### Mínimos absolutos para deep learning

| Clase | Cantidad mínima |
|---|---|
| Válidos | 500 |
| Inválidos | 200 |
| **Total** | **~700** |

Por debajo de estos números, se recomienda empezar con un baseline clásico (Random Forest, XGBoost con features estadísticas) antes de invertir en deep learning.

## Evaluación por vista

| Vista | Válidos | Estado | Inválidos | Estado |
|---|---|---|---|---|
| **Frontal** | 192 | ✅ Aceptable | 67 | ❌ Muy bajo |
| **Lateral** | 354 | ✅ Bien | 164 | ⚠️ Bajo |
| **Left** | 177 | ⚠️ Justo | 82 | ❌ Muy bajo |
| **Right** | 177 | ⚠️ Justo | 82 | ❌ Muy bajo |

## Estrategias recomendadas

### 1. Consolidar vistas en un solo modelo
En lugar de un modelo por cámara, fusionar todas las vistas en un solo modelo usando la cámara como feature.

- **Inválidos totales:** 67 + 164 + 82 = **313** (todavía único)
- **Válidos totales:** 192 + 354 + 177 + 177 = **900**
- **Total:** ~1213 levantamientos ✅
- Esto se acerca bastante a la recomendación ideal.

### 2. Aumentar la clase inválido
- Re-etiquetar videos específicamente buscando malas ejecuciones
- Pedir a etiquetadores que prioricen movimientos incorrectos
- Si es posible, buscar más footage de deadlifts con errores comunes (espalda redondeada, cadera baja, etc.)

### 3. Data augmentation para keypoints de esqueleto
Multiplicar artificialmente los inválidos (y válidos) con:

- Rotaciones 3D (±15° en cada eje)
- Escalado (0.9–1.1)
- Ruido gaussiano en coordenadas de keypoints
- Jittering temporal (desplazar frames ±2–3)
- Cropping temporal (ventanas solapadas)
- Elastic transformations

Un factor de **x3–x5** es razonable.

### 4. Balanceo de clases
- **Weighted loss / Focal loss** para que la minoría (inválidos) tenga más peso
- **Oversampling** de la clase inválida en cada batch
- No se recomienda undersampling de válidos (se pierde variabilidad)

## Arquitectura de modelo sugerida

Con este tamaño de datos, se recomienda un modelo **pequeño**:

| Componente | Opción recomendada |
|---|---|
| Backbone | LSTM bidireccional (1-2 capas, 64-128 unidades) o GCN pequeña |
| Head | MLP de 1-2 capas (64 → 32 → 1) |
| Regularización | Dropout (0.3–0.5), BatchNorm, Weight Decay |
| Optimizador | Adam con learning rate 1e-4 a 3e-4 |
| Early stopping | Paciencia de 10–15 épocas |
| Evaluación | F1-score, Precision-Recall (no solo accuracy) |

### Por qué no transformers grandes
Con ~1000 secuencias, un transformer tiene demasiados parámetros y overfittea. Solo recomendable si tienes 5000+ muestras o usas pre-training.

## Flujo de trabajo recomendado

```
1. Extraer keypoints → 2. Data augmentation → 3. Baseline clásico (RF/XGBoost
   con features estáticas) → 4. Si separable → 5. Red neuronal (LSTM/GCN)
   → 6. Cross-validation estratificada → 7. Evaluación en test set
```

### Baseline clásico (paso 3)
Antes de saltar a deep learning, probar con:

- Features por frame: ángulos (cadera, rodilla, torso), velocidades de keypoints, simetría
- Agregación temporal: media, std, min, max, rango por secuencia
- Modelo: Random Forest (100–300 árboles) o XGBoost
- Si el F1-score con esto ya es bajo (>0.7), probablemente deep learning no mejore significativamente

## Resumen

| Estrategia | Inválidos | Válidos | Total | Veredicto |
|---|---|---|---|---|
| Separado por vista (solo lateral) | 164 | 354 | 518 | ⚠️ Justo |
| Separado por vista (solo frontal) | 67 | 192 | 259 | ❌ Muy justo |
| **Consolidar todas las vistas** | **313** | **900** | **~1213** | **✅ Recomendado** |
| Consolidar + augmentation x3 | ~940 | ~2700 | ~3640 | ✅ Excelente |
| **Ideal (con más etiquetado)** | **500+** | **1000+** | **1500+** | **✅ Meta final** |
