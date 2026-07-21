# Especificación: Construcción de la CNN para Deadlift Front (MediaPipe Keypoints)

## Objetivo

Implementar el pipeline completo para entrenar una CNN binaria que
clasifique un levantamiento de **Deadlift (vista frontal)** como
**Válido (0)** o **Nulo (1)** utilizando exclusivamente keypoints
extraídos previamente con MediaPipe.

## Contexto

-   Los videos ya fueron procesados.
-   Los keypoints ya fueron extraídos.
-   El dataset ya está etiquetado.
-   Esta etapa comienza desde los archivos de keypoints y termina con un
    modelo entrenado y evaluado.
-   No modificar el etiquetado.

## Pipeline requerido

1.  Cargar los archivos de keypoints.
2.  Validar integridad:
    -   eliminar videos corruptos.
    -   detectar NaN.
    -   detectar frames sin pose.
    -   interpolar pequeños huecos.
3.  Seleccionar únicamente los keypoints relevantes para Deadlift Front:
    -   nariz
    -   hombros
    -   codos
    -   muñecas
    -   caderas
    -   rodillas
    -   tobillos
    -   talones
    -   foot index
4.  Normalización espacial:
    -   centrar todos los keypoints respecto al punto medio entre ambas
        caderas.
    -   normalizar la escala utilizando la altura corporal (o una medida
        equivalente constante).
5.  Mantener para cada keypoint:
    -   x
    -   y
    -   visibility/confidence
6.  Normalización temporal:
    -   todos los videos deben tener exactamente la misma cantidad de
        frames.
    -   utilizar interpolación temporal, no padding con ceros.
7.  División del dataset:
    -   Train
    -   Validation
    -   Test
    -   evitar fuga de datos.
8.  Ajustar StandardScaler únicamente con Train y aplicarlo a Validation
    y Test.
9.  Data augmentation SOLO en Train:
    -   pequeña traslación
    -   pequeño escalado
    -   pequeño ruido gaussiano
    -   pequeñas rotaciones
    -   pequeñas variaciones temporales
10. Construir una CNN que reciba tensores (frames × keypoints ×
    características):
    -   Conv2D(32)
    -   BatchNorm
    -   ReLU
    -   Conv2D(64)
    -   BatchNorm
    -   ReLU
    -   MaxPooling
    -   Dropout
    -   Conv2D(128)
    -   GlobalAveragePooling
    -   Dense(64)
    -   Dropout
    -   Dense(1, sigmoid)
11. Configuración:
    -   Binary Crossentropy
    -   Adam (lr=0.001)
    -   EarlyStopping
    -   ReduceLROnPlateau
    -   ModelCheckpoint
12. Métricas:
    -   Accuracy
    -   Precision
    -   Recall
    -   F1-score
    -   ROC-AUC
    -   Matriz de confusión
13. Guardar:
    -   mejor modelo
    -   historial de entrenamiento
    -   curvas de loss y accuracy

## Importante

No utilizar imágenes originales. El entrenamiento debe usar únicamente
los keypoints ya extraídos.

## Entregables esperados

-   Código modular y bien documentado.
-   Explicación de cada etapa.
-   Arquitectura implementada.
-   Resumen del preprocesamiento.
-   Resultados de entrenamiento y validación.
-   Gráficos de entrenamiento.
-   Matriz de confusión.
-   Recomendaciones de mejora justificadas.
