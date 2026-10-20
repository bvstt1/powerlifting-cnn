# Investigación pose-guided de la barra (BP front)

Esta rama reúne las cuatro etapas del estudio que utiliza muñecas YOLO Pose
para evaluar la identidad de la barra detectada por segmentación. No alimenta
las features temporales de la SegCNN entrenada ni constituye un tracker
confirmado. Cada etapa conserva juntos su código, resultados, manifiestos y
evidencias visuales.

| Etapa | Directorio | Estado |
|---|---|---|
| 1. Factibilidad | [`01_feasibility/`](01_feasibility/v1/) | Evidencia exploratoria de sincronía y geometría en Validation76. |
| 2. Especificación | [`02_specification/`](02_specification/) | Definición y protocolo congelados antes del prototipo V3.1. |
| 3. Prototipo | [`03_prototype_v31/`](03_prototype_v31/) | Ensayo shadow de desarrollo; requiere revisión por abstenciones excesivas. |
| 4. Revisión | [`04_revision_v31/`](04_revision_v31/) | Prometedor para validación independiente; **no aprobado** para CNN. |

Flujo de fuentes: `front/raw/` + `tracking_v2/` + selecciones `tracking_v3/`
+ pose `keypoints/bp/front/` → análisis pose-guided. Los RAW y selecciones
fuente permanecen en sus carpetas originales.

## ¿De dónde salen los CSV?

Son **tablas de auditoría de distintas granularidades**, no datasets
independientes: un mismo vídeo puede aparecer una vez, por frame, por track,
por transición entre tracks o por variante experimental. No se suman sus
filas para contar vídeos ni se usan como etiquetas de válido/nulo de la CNN.

| Etapa y productor | CSV | Qué representa |
|---|---|---|
| `01_feasibility/v1/01_audit_sync.py` | `bp_front_pose_guided_sync_audit.csv` | Una fila por vídeo de Validation76: rutas, frames, FPS, pose, RAW/V2/V3 y comprobaciones de alineación. |
| `01_feasibility/v1/02_anatomical_metrics.py` | `bp_front_pose_guided_handoff_metrics.csv`, `bp_front_pose_guided_track_identity_metrics.csv` | Medidas de muñecas/máscaras por handoff y por track; incorporan las referencias V3 para comparación descriptiva. |
| `01_feasibility/v1/03_verify_results.py` | `bp_front_pose_guided_feasibility_summary.csv` | Resumen de comprobaciones y magnitudes del estudio exploratorio. |
| `02_specification/` | `BP_FRONT_Pose_Guided_Tracking_V31_Parameters.csv` | Tabla de parámetros y criterios de la especificación; no es una salida de inferencia ni la genera el prototipo. |
| `03_prototype_v31/00_reserve_and_snapshot.py` | `bp_front_v31_reservation_inventory.csv` | Inventario de MP4 y estado de reserva/exclusión antes del ensayo; **no** son predicciones de los vídeos reservados. |
| `03_prototype_v31/01_run_shadow.py` | `bp_front_v31_sync_audit.csv`, `bp_front_v31_frame_identity.csv`, `bp_front_v31_track_consistency.csv`, `bp_front_v31_track_summary.csv` | Sincronía y decisiones por frame/track del prototipo en desarrollo. |
| `03_prototype_v31/01_run_shadow.py` | `bp_front_v31_handoff_decisions.csv`, `bp_front_v31_selected_trajectories.csv`, `bp_front_v31_comparison_v2_v3_v31.csv`, `bp_front_v31_ablation_results.csv` | Decisiones por handoff, trayectorias y comparaciones/ablaciones entre métodos. |
| `04_revision_v31/01_diagnose.py` | `diagnosis_omitted_candidates.csv`, `diagnosis_39_correct_boundary_abstentions.csv`, `diagnosis_preview_index.csv` | Diagnóstico **previo** de candidatas omitidas, bordes con abstención e imágenes a revisar. |
| `04_revision_v31/02_run_revision.py` | `bp_front_v31_revision_frame_identity.csv`, `bp_front_v31_revision_selected_trajectories.csv`, `bp_front_v31_revision_handoff_comparison.csv`, `bp_front_v31_revision_per_video_comparison.csv` | Resultados recalculados de la revisión, comparables con el prototipo original. |
| `04_revision_v31/02_run_revision.py` | `bp_front_v31_revision_dominance_evidence.csv`, `bp_front_v31_revision_context_evidence.csv`, `bp_front_v31_revision_preview_index.csv` | Evidencia de decisiones anatómicas y catálogo de imágenes de inspección. |
| Revisión visual registrada | `bp_front_v31_revision_visual_review.csv` | Anotaciones de **frames inspeccionados**; `04_verify_and_report.py` las lee, pero no las crea a partir de predicciones ni equivalen a verdad para todos los frames. |

Los prefijos largos dejan trazabilidad de cada etapa; no indican que se hayan
entrenado diez modelos. En particular, `03_verify_results.py` y
`04_verify_and_report.py` leen CSV anteriores para comprobarlos y crear
informes/manifiestos, no para duplicar las muestras del dataset.

## Procedencia histórica

Las carpetas se trasladaron como unidades completas, incluidas imágenes,
CSV y snapshots. Los manifiestos originales guardan **rutas de antes del
traslado** y hashes de la ejecución histórica. Se conservan sin modificarlos:
no se deben interpretar esas rutas como instrucciones vigentes, ni atribuir
los resultados anteriores al código reorganizado. Para verificar exactamente
la ejecución histórica hace falta el commit previo a esta reorganización.
Los tests sintéticos pueden ejecutarse desde las rutas actuales; los scripts
de preflight que exigen el snapshot histórico completo no pasan a ser una
nueva evaluación confirmatoria por trasladar directorios.
