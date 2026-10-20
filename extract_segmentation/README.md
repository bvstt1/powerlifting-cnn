# Extracción de segmentación BP: mapa de publicación

Las versiones nombran **capas distintas**: checkpoint YOLO, esquema de RAW,
algoritmo de tracking y esquema de features temporales. Un `v3` de tracking
no sustituye al checkpoint de segmentación `v2`.

## Regla de compatibilidad de datos

- El sufijo `vN` de un **directorio de pipeline de datos** (`front_temporal_v1`,
  `side_temporal_v1`, `temporal_features_v1`) identifica un contrato legible:
  estructura de archivos, nombres/orden de canales, unidades, disponibilidad,
  procedencia y significado de cada valor. Front y lateral tienen contratos
  distintos aunque ambos sean `v1`.
- Los scripts dentro de ese directorio se llaman por su función (`config.py`,
  `schema.py`, `extract_all.py`, `validate.py`); no reciben un segundo `v1`.
  Corregir documentación o refactorizar sin cambiar el contrato no crea `v2`.
  Si cambia la interpretación o el formato de los datos, crear un **nuevo**
  contrato/directorio y preservar el anterior; nunca sobrescribir una cohorte
  congelada con otra semántica bajo el mismo `vN`.
- Checkpoints YOLO (`bp_front_seg_v2.pt`) identifican **pesos del detector**;
  `tracking_v2.2`/`tracking_v3` identifican **algoritmos**. Ninguno declara
  compatibilidad de features por su número. Los nombres `*_segcnn_v1` en
  `training/` identifican ejecuciones históricas: no se renombran para
  hacerlos coincidir con la versión de YOLO. El consumidor debe comprobar
  explícitamente el esquema de su entrada.

| Consumidor | RAW aceptado | Features aceptadas | Procedencia del detector |
|---|---|---|---|
| Front `front_temporal_v1` → `bp_front_01_segcnn_v1` | `bp_front_segmentation_raw_v2` | `temporal_features_v1`, 38 canales en orden del esquema | `bp_front_seg_v2.pt`, pose front v8 |
| Left/right `side_temporal_v1` → `bp_side_segcnn_v1` | `bp_side_segmentation_raw_v1` | `bp_side_temporal_features_v1`, 62 canales en orden del esquema | `bp_side_seg_v1.pt`, pose lateral v2 |

Los lectores RAW rechazan versiones incompatibles. Los consumidores SegCNN
comprueban la versión del esquema de features y de las secuencias además del
orden y shape de canales. Las cohortes históricas siguen en las mismas rutas;
esta regla no cambia sus bytes ni convierte los experimentos V3/V3.1 en fuente
de entrenamiento.

## Linaje del tracking frontal de barra

Para evitar el salto aparente de V1 a V3, llamamos **Tracking V1
(denominación retrospectiva)** al tracking inicial que se ejecuta **durante**
`bp/extract_segmentation_front_bp.py`, usando `track_detections()` y
`select_main_tracks()` de `bp/segmentation_utils.py`. Agrupa detecciones por
clase y frame mediante asignación basada en IoU de máscara/caja, desplazamiento
del centro y cambio de área; selecciona un track principal por clase (barra,
banco, persona). Sus parámetros históricos predeterminados son `max_gap=2`,
`stationary_bench_max_gap=60` y `max_cost=0.85`. La salida está en
`segmentation/bp/front/{raw,features}/` y su configuración está registrada
en el campo `tracking` de la metadata RAW. **No** existe un archivo o
directorio `tracking_v1/` ni un campo histórico con el valor «V1»: este nombre
solo permite describir con precisión esa primera etapa, sin modificar sus
artefactos ni reinterpretar sus hashes.

| Etapa | Implementación | Entrada → salida | Alcance |
|---|---|---|---|
| Tracking V1 (retrospectivo) | `bp/extract_segmentation_front_bp.py` + `bp/segmentation_utils.py` | Detecciones YOLO front → `segmentation/bp/front/{raw,features}/` | Asociación y selección inicial; preserva todas las detecciones en RAW. |
| Tracking V2.2 | `bp/rebuild_tracking_front_bp_from_raw.py` | RAW del tracking inicial → `segmentation/bp/front/tracking_v2/` | Reasociación derivada, con auditoría; no vuelve a ejecutar YOLO. |
| Tracking V3 (prototipo) | `bp/bar_tracking_v3_utils.py` + `bp/rebuild_bar_tracking_v3_from_raw.py` | RAW/V2 → `segmentation/bp/front/tracking_v3/{test_14,validation_76}/` | Comparación experimental de continuidad de barra; no es un dataset global de features para CNN. |

«Tracking V1» **no** es `bp_front_seg_v1.pt` (checkpoint de YOLO) ni
`bp_front_segmentation_raw_v2` (formato de almacenamiento). Son versiones de
otras capas.

## Flujo que alimenta los clasificadores actuales

| Vista / etapa | Código fuente | Entrada → salida | Estado |
|---|---|---|---|
| Front, inferencia YOLO | `bp/extract_segmentation_front_bp.py` y `bp/segmentation_utils.py` | `models/bp/bp_front_seg_v2.pt` + `dataset/bp/front/` → `segmentation/bp/front/{raw,features,manifests}/` | Extractor base; RAW con máscaras y tracks auditables. |
| Front, tracking derivado | `bp/tracking_front_v2/` (exploración, reconstrucción y auditoría); `bp/finalize_segmentation_front_bp.py` (cierre del dataset) | RAW front → `segmentation/bp/front/tracking_v2/` | Derivación/auditoría V2.2. No sustituir el RAW fuente. |
| Front, features para SegCNN | `bp/front_temporal_v1/{config,raw_reader,geometry,features,schema,extract_all,pilot,validate,test_geometry}.py` | RAW front + pose YOLO front (`bp_front_skeleton_v8.pt`) → `segmentation/bp/front/temporal_features_v1/` | **Cohorte que consumen los entrenamientos BP front** (`training/bp/experiments/bp_front_01_segcnn_v1/` y fusión). Usa RAW fuente, no `tracking_v3`. |
| Left/right, inferencia YOLO | `bp/extract_segmentation_side_bp.py` | `models/bp/bp_side_seg_v1.pt` + vídeos por vista → `segmentation/bp/{left,right}/raw/` | Mismo checkpoint, salidas independientes por vista; sin tracking lateral en RAW. |
| Left/right, features para SegCNN | `bp/side_temporal_v1/{run,raw_reader,geometry,features,schema,report}.py` | RAW lateral + pose YOLO lateral → `segmentation/bp/{left,right}/temporal_features_v1/` | Cohortes que consume `training/bp_side_segcnn_v1/`. |

`models/bp/bp_front_seg_v1.pt` queda como checkpoint histórico: los
extractores front y visores actuales apuntan a `bp_front_seg_v2.pt`. Los
checkpoints de pose y objeto son modelos de tareas distintas.

El código `front_temporal_v1/` vivía dentro de la carpeta de resultados
`segmentation/bp/front/temporal_features_v1/code/`. Ahora está junto a los
otros extractores; **la ruta de los datos generados no cambió**. Desde la raíz
del proyecto se puede usar:

```powershell
python extract_segmentation/bp/front_temporal_v1/test_geometry.py
python extract_segmentation/bp/front_temporal_v1/pilot.py
python extract_segmentation/bp/front_temporal_v1/extract_all.py --workers 6
python extract_segmentation/bp/front_temporal_v1/validate.py
python -m extract_segmentation.bp.side_temporal_v1.run --view left --phase verify
```

El extractor front admite `--videos` o `--all`, y el lateral requiere
`--view` y fases `audit`, `pilot`, `all`, `verify`. Los scripts de features
front usan paths de salida históricos y algunos verificadores requieren
los manifiestos y splits de la cohorte ya creada; no son pipelines desde cero
para un dataset nuevo.

## Experimentos de tracking: preservar y distinguir

El código V2/V3 está agrupado físicamente en `bp/tracking_front_v2/` y
`bp/tracking_front_v3/`, con puntos de acceso y guías por fase. Los manifiestos
históricos conservan los hashes/rutas originales; los cambios de ruta e
imports del código actual están documentados en esas guías, no se atribuyen
retroactivamente a las métricas previas.

| Código / salida | Propósito | Decisión de publicación |
|---|---|---|
| `bp/tracking_front_v3/` contiene `bar_tracking_v3_utils.py`, `rebuild_bar_tracking_v3_from_raw.py`, `test_bar_tracking_v3_prototype.py`, `validate_bar_tracking_v3_76.py`, `verify_bar_tracking_v3_validation76.py` → `segmentation/bp/front/tracking_v3/{test_14,validation_76}/` | Comparación experimental de identidad de barra V3 con RAW existente. | Conservar y rotular como experimento; no presentarlo como salida de la SegCNN vigente. |
| `segmentation/bp/front/pose_guided_feasibility/`, `pose_guided_tracking_v31_spec/`, `pose_guided_tracking_v31_prototype/`, `pose_guided_tracking_v31_revision_final/` | Estudios y prototipos V3.1 asistidos por pose. | Conservar su trazabilidad. La revisión final se declara **prometedora, pendiente de validación independiente**; «final» nombra la revisión, no un tracker aprobado. |
| `bp/tracking_front_v2/scout_tracking_cases_front_bp.py` | Selección/inspección de casos de tracking. | Utilidad de análisis, no extractor alternativo de producción. |

No se movieron los experimentos V3.1: varios contienen snapshots y rutas
congeladas. Tampoco se movieron los modelos `.pt`, los vídeos ni los arrays
generados. Antes de publicar nuevas ejecuciones, documentar por separado
checkpoint de segmentación, esquema RAW, tracking, esquema temporal y cohorte
del clasificador, junto con sus hashes cuando correspondan.
