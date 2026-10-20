# BP front: qué usar y qué es investigación

## 1. Datos generados y ruta al clasificador

```text
dataset/bp/front/ + models/bp/bp_front_seg_v2.pt
    └─ extract_segmentation_front_bp.py
       ├─ raw/               máscaras/detecciones fuente por vídeo
       └─ features/          features iniciales del extractor
raw/ + keypoints/bp/yolo/front/
    └─ extract_segmentation/bp/front_temporal_v1/extract_all.py
       └─ temporal_features_v1/sequences/   (T, 38), entrada de SegCNN front
```

`temporal_features_v1/` incluye
[`bp_front_features_schema.json`](temporal_features_v1/bp_front_features_schema.json),
[`bp_front_features_common_cohort.csv`](temporal_features_v1/bp_front_features_common_cohort.csv),
manifiestos y reportes. `training/bp/experiments/bp_front_01_segcnn_v1/` y los flujos BP front
posteriores leen esa cohorte y sus secuencias. **Las features temporales se
derivan del `raw/` fuente, no de `tracking_v2/` ni de `tracking_v3/`.** El
directorio histórico `temporal_features_v1/code/` no es la ubicación vigente
del código: los scripts están en `extract_segmentation/bp/front_temporal_v1/`.

## 2. Evidencia y auditoría

- `manifests/`: configuración congelada, inventario de los 663 vídeos,
  calidad, errores y auditoría de barra. El manifiesto final distingue
  `source_raw_path` de `segmentation_path` (derivada V2).
- `previews/` y `tracking_v2/{audits,manifests,previews}/`: inspección y
  comprobaciones; no son nuevos vídeos etiquetados.
- `tracking_v2/{raw,features}/`: **reconstrucción derivada** de tracks V2.2
  desde el RAW fuente. Se preserva porque permite comparar decisiones de
  identidad y continuidad; no sustituye retroactivamente al RAW original.

## 3. Investigación posterior, separada del entrenamiento vigente

- `tracking_v3/test_14/` y `tracking_v3/validation_76/`: prototipo y
  evaluación de un algoritmo posterior de seguimiento de barra. No hay
  cohorte global V3 utilizada por la SegCNN front. En Validation76 se
  documentaron dos fusiones erróneas; no equivale a un tracking aprobado.
- [`pose_guided/`](pose_guided/README.md) reúne las cuatro etapas en orden:
  `01_feasibility/v1/`, `02_specification/`, `03_prototype_v31/` y
  `04_revision_v31/`. La última es una revisión experimental, **no** una
  validación independiente ni una fuente nueva del entrenamiento CNN.

Las cuatro etapas pose-guided se trasladaron como unidades completas,
conservando snapshots, imágenes y manifiestos históricos. Sus rutas antiguas
siguen apareciendo dentro de esa evidencia, como explica su índice. El
flujo de extracción y tracking V2/V3 está en `extract_segmentation/README.md`.
