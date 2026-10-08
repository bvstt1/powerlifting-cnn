# Reporte BP_FRONT PoseCNN

## 1. Dataset utilizado

- Modalidad: Pose temporal, exclusivamente vista `front`.
- Keypoints RAW: `C:\Users\basti\MediapipePythonProjects\keypoints\bp\front`.
- Extension: `.npy`.
- Videos originales: 663.
- CSV vigente: `C:\Users\basti\MediapipePythonProjects\etiquetado\bp\bp_front.csv` (663 filas).
- Pose files: 663, rango `bp_001` a `bp_663`.
- Convencion: `bp_NNN.npy`.
- No se uso `group_id`, pairing ni informacion left/right.

## 2. Modelo Pose y mapa de keypoints

- Modelo: `bp_front_skeleton_v8.pt`.
- SHA-256: `ff6b67374b50e9e98cab732ecbcc26ad80e5a41daac5f64a22c7c6fcffdd09dd`.
- Extractor: `C:\Users\basti\MediapipePythonProjects\extract_keypoints\bp\extract_keypoints_front_bp.py`.
- Configuracion embebida: `kpt_shape=[6, 3]`.
- Shape real uniforme: `(T, 6, 3)`.
- Features: canal 0 `x` normalizada por ancho de imagen, canal 1 `y` normalizada por alto, canal 2 `confidence` YOLO.

| index | keypoint_name | descripcion |
|---:|---|---|
| 0 | `left_elbow` | Codo izquierdo del levantador |
| 1 | `left_shoulder` | Hombro izquierdo del levantador |
| 2 | `right_shoulder` | Hombro derecho del levantador |
| 3 | `right_elbow` | Codo derecho del levantador |
| 4 | `right_wrist` | Muneca derecha del levantador |
| 5 | `left_wrist` | Muneca izquierda del levantador |

El orden proviene de las conexiones del visualizador YOLO vigente. El comentario historico de `(0,5)` repite por error "muneca derecha"; la cadena bilateral y el hecho de que el indice 4 ya es la muneca derecha determinan que el indice 5 es la muneca izquierda. No se aplicaron flips horizontales.

## 3. Auditoria de integridad

- Layout estructural: las 663 secuencias tienen `(T,6,3)`; `T` es variable y se resume abajo.
- Dtypes: `{'float32': 663}`.
- NaN totales: 0.
- Inf totales: 0.
- Frames con missing: 0.
- Frames completamente missing: 0.
- Confianza minima global: 0.967082.
- Confianza media ponderada: 0.997348.
- Estados: `{'POSE_QUALITY_OK': 662, 'POSE_QUALITY_EXCLUDED': 1}`.
- Archivo por muestra: `C:\Users\basti\MediapipePythonProjects\training\data\bp\front\bp_front_pose_audit.csv`.

### Distribucion de T_original

| N | min | P1 | P5 | P25 | median | P75 | P95 | P99 | max | mean | std |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 663 | 21 | 31.00 | 39.00 | 62.00 | 79.00 | 100.00 | 169.90 | 235.90 | 302 | 87.54 | 42.26 |

## 4. Quality rules y preprocessing

- `POSE_QUALITY_EXCLUDED`: shape distinto de `(T,6,3)`, menos de 10 frames, Inf, confianza fuera de `[0,1]`, mediana de confianza menor que 0.5, ratio missing x/y mayor que 1.00%, gap mayor que 3 frames, escala interhombros degenerada o duplicado exacto no canonico.
- `POSE_QUALITY_WARNING`: missing escaso e interpolable de hasta 3 frames o mediana de confianza entre 0.5 y 0.9.
- `POSE_QUALITY_OK`: estructura valida, finita y sin warnings.
- La tolerancia de 3 frames equivale aproximadamente a 100 ms a 30 FPS. No se copio el umbral historico de 30%; el dataset BP real tiene cero missing y permite una regla mucho mas conservadora.
- Pose utilizables: 662.
- Pose excluidos: 1.
- Duplicados exactos detectados: `[{'canonical_id': 'bp_351', 'duplicate_ids': ['bp_352'], 'all_ids': ['bp_351', 'bp_352'], 'pose_content_sha256': 'ef0f4dfa8dcd39e468837501beae135e9be0fe6c863c89c475b7bceaec1f71fe', 'decoded_video_identical': True, 'decoded_video_sha256': {'bp_351': '3c4fa13f160d1945fc748ba16ed29dd6f30baadeeb24e38a6869d190a2649336', 'bp_352': '3c4fa13f160d1945fc748ba16ed29dd6f30baadeeb24e38a6869d190a2649336'}, 'decoded_frames': {'bp_351': 110, 'bp_352': 110}}]`. Se conserva el ID menor como canonico; el duplicado se excluye sin borrar ni modificar RAW y sin consultar labels.
- Manifest: `C:\Users\basti\MediapipePythonProjects\training\data\bp\front\bp_front_pose_preprocessing_manifest.csv`.
- RAW originales no fueron modificados; copia derivada trazable en `C:\Users\basti\MediapipePythonProjects\training\data\bp\front\raw_sequences.npz`.

## 5. Labels y clases

- Convencion confirmada: `0 = invalido`, `1 = valido`.
- Evidencia: todas las filas `resultado=valido` corresponden a 1, todas las `resultado=invalido` a 0, sin contradicciones; el loader historico solo acepta `{0,1}` y AGENTS.md documenta la misma convencion.
- Total: 663.
- Validos: 400 (60.33%).
- Invalidos: 263 (39.67%).
- Labels faltantes: 0.
- Labels fuera de `{0,1}`: `[]`.
- Clase positiva de todas las metricas: `1 = valido`.

## 6. Correspondencia y cohortes

- VIDEO + CSV + POSE: 663.
- Video sin CSV: `[]`.
- Video sin Pose: `[]`.
- CSV sin video: `[]`.
- CSV sin Pose: `[]`.
- Pose sin video: `[]`.
- Pose sin CSV: `[]`.
- IDs duplicados en videos/Pose/CSV: `[]` / `[]` / `[]`.
- Cohorte A Pose-only: 662.
- Cohorte B Common Pose+Seg: 661.
- Manifest segmentacion: 663 registros, 662 validos y `QUALITY_ERROR=['bp_349']`.
- `bp_349` conserva Pose valida y permanece en Pose-only; solo se excluye de common.
- `bp_352` se excluye de ambas cohortes porque su Pose y sus 110 frames decodificados son identicos a `bp_351`; `bp_351` queda como muestra canonica.

## 7. Normalizacion

### Espacial

Para cada secuencia se calcula un centro fijo:

`center = median_t((left_shoulder_t + right_shoulder_t) / 2)`

y una escala robusta fija:

`scale = median_t(||left_shoulder_t - right_shoulder_t||_2)`

Luego: `xy_normalized_t = (xy_t - center) / scale`. La confianza no cambia. Usar estadisticos fijos por secuencia reduce posicion y distancia a camara sin borrar el movimiento temporal de hombros, la simetria ni la geometria relativa de codos y munecas.

### Temporal

Se mantiene `T=170` para comparabilidad con Deadlift. Cada canal se remuestrea linealmente sobre tiempo normalizado `[0,1]`. Solo llegan a esta etapa secuencias validas y sin gaps largos; no se inventan datos para muestras excluidas.

## 8. Split monovista

- Estrategia: `StratifiedShuffleSplit` por muestra, seed 42.
- Proporcion objetivo: 70/15/15.
- No se uso `GroupShuffleSplit`, `group_id` ni logica multivista.
- Split Pose-only: `C:\Users\basti\MediapipePythonProjects\training\data\bp\front\bp_front_pose_split.csv`.
- El mismo archivo guarda por separado `common_split`, split estratificado congelado para la futura comparacion Pose vs Segmentation vs Fusion sobre la cohorte common.

| Split | N | Invalidos (0) | Validos (1) |
|---|---:|---:|---:|
| Train | 462 | 182 | 280 |
| Validation | 100 | 40 | 60 |
| Test | 100 | 40 | 60 |

## 9. Scaler y augmentation

- `StandardScaler` ajustado exclusivamente con train original preprocesado, antes de augmentation.
- Validation y test solo reciben `transform`.
- Scaler: `C:\Users\basti\MediapipePythonProjects\training\models\bp\front\posecnn\bp_front_pose_scaler.pkl`.
- Augmentation solo train: tres copias por original mas el original.
- Transformaciones independientes con probabilidad 0.5: shift x/y `+-0.05`, escala `0.9-1.1`, ruido gaussiano `sigma=0.01`, rotacion `+-10 grados`, shift temporal `+-3` frames.
- No hay horizontal flip, oversampling, undersampling ni class weights. El augmentation conserva exactamente la proporcion de clases del train.

## 10. PoseCNN e hiperparametros

- Input PyTorch: `(N,3,170,6)` desde almacenamiento `(N,170,6,3)`.
- Conv2D: 32, 64, pooling temporal, Dropout2D 0.40, Conv2D 128.
- Global average pooling.
- MLP: 128, 64, 1 con dropout 0.50 y salida Sigmoid.
- Adam, lr 0.001, weight decay 0.0001, BCELoss.
- Batch 16, maximo 200 epochs, early stopping patience 15 por validation loss.
- Scheduler `ReduceLROnPlateau`, factor 0.5, patience 5.
- Threshold fijo 0.5; no fue optimizado con test.
- Mejor epoch: 68; epoch final: 83.
- Historial: `C:\Users\basti\MediapipePythonProjects\training\results\bp\front\posecnn\bp_front_pose_training_history.csv` (83 epochs).
- Checkpoint: `C:\Users\basti\MediapipePythonProjects\training\models\bp\front\posecnn\bp_front_pose_best_model.pth`.

## 11. Metricas finales

| Split | Accuracy | Precision | Recall | F1 | ROC-AUC | Specificity | NPV |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validation | 0.9300 | 0.9077 | 0.9833 | 0.9440 | 0.9750 | 0.8500 | 0.9714 |
| Test | 0.8400 | 0.8548 | 0.8833 | 0.8689 | 0.9354 | 0.7750 | 0.8158 |

Test se evaluo una sola vez para la revision final deduplicada, despues de seleccionar el checkpoint por validation loss. Clase positiva: valido (1). Una ejecucion preliminar fue invalidada al descubrir el duplicado exacto `bp_351/bp_352`; sus artefactos se reemplazaron, no se usaron para ajustar hiperparametros y no forman parte de estos resultados.

### Confusion Matrix Test

| TN | FP | FN | TP |
|---:|---:|---:|---:|
| 31 | 9 | 7 | 53 |

## 12. 5-Fold Stratified Cross-Validation

- Scope: development (`train + validation`) del split principal; el holdout test final queda excluido.
- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- Cada fold externo empieza desde secuencias preprocesadas originales. Dentro del outer-train se reserva un 15% estratificado exclusivamente para scheduler, early stopping y seleccion de checkpoint; el scaler se ajusta solo con el inner-train original y augmentation se aplica solo al inner-train. El outer-validation se evalua una vez despues de fijar el checkpoint.
- No hay GroupKFold ni datos augmentados pregenerados compartidos.

| Fold | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| 1 | 0.8496 | 0.9474 | 0.7941 | 0.8640 | 0.9533 |
| 2 | 0.9204 | 0.9403 | 0.9265 | 0.9333 | 0.9572 |
| 3 | 0.8393 | 0.8049 | 0.9706 | 0.8800 | 0.9171 |
| 4 | 0.8214 | 0.8158 | 0.9118 | 0.8611 | 0.9218 |
| 5 | 0.9018 | 0.9254 | 0.9118 | 0.9185 | 0.9693 |
| Mean | 0.8665 | 0.8867 | 0.9029 | 0.8914 | 0.9437 |
| Std | 0.0380 | 0.0629 | 0.0585 | 0.0293 | 0.0206 |

## 13. Reproducibilidad

- Configuracion: `C:\Users\basti\MediapipePythonProjects\training\models\bp\front\posecnn\bp_front_pose_config.json`.
- Seed: 42.
- Python: 3.10.11.
- PyTorch: 2.13.0.dev20260521+cu130.
- CUDA runtime: 13.0.
- Dispositivo: cuda (NVIDIA GeForce RTX 5060 Laptop GPU).
- NumPy: 2.2.6.
- SciPy: 1.15.3.
- scikit-learn: 1.7.2.

## 14. Limitaciones y comparacion metodologica

- El esquema Pose contiene hombros, codos y munecas, pero no barra, pecho, cabeza ni gluteos. BP-02 es directamente observable; BP-01, BP-03 y BP-08 son parcial o debilmente observables. El modelo puede aprender proxies y no sustituye aun la futura fusion con segmentacion.
- El orden anatomico se reconstruye del skeleton operativo; el `data.yaml` original del entrenamiento YOLO no esta conservado.
- No se conserva el split usado para entrenar `bp_front_skeleton_v8.pt`; por ello no puede certificarse la independencia end-to-end del extractor Pose respecto a los videos del holdout.
- `T=170` sobremuestrea la mayoria de secuencias y elimina duracion absoluta, aunque mantiene comparabilidad con las PoseCNN anteriores.
- El split monovista cumple el requisito experimental actual, pero no controla identidad de atleta o sesion porque esa metadata no existe en el CSV.
- Como en Deadlift, el scaler se ajusta solo con train original y augmentation se restringe a train. A diferencia del pipeline DL lateral, no existe mapping fisico, grouping ni pairing; BP_FRONT usa estratificacion directa por muestra.
- La arquitectura conserva la familia Conv 32/64/128 + global pooling; agrega la MLP solicitada 128/64/1 sin redisenar radicalmente la baseline.

No se entreno Segmentacion CNN ni Pose+Seg Fusion y no se modifico ningun artefacto de segmentacion.
