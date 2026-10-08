# Reporte BP_FRONT Bar Tracking Audit V1

Fecha: `2026-10-05T23:49:11.044861+00:00`

## 1. Alcance e inmutabilidad

- Dataset: `BP_FRONT_SEG_DATASET_V1`; tracking: `bp_front_bench_tracking_v2.2`.
- Videos auditables: **662**; `bp_349` permanece `QUALITY_ERROR` y no se intento recuperar.
- La auditoria solo leyo RAW, tracking V2 y FEATURES; no ejecuto YOLO ni recalculo features.
- Los 662 audits confirman `source_raw_unchanged=1`.
- No se utilizaron labels, resultados ni criterios de jueceo.

## 2. Correcciones de metadata y QA

- `bp_349` conserva `status=ERROR`, `quality_status=QUALITY_ERROR` y ahora usa `quality_reasons=degenerate_polygon`.
- La seleccion QA ya no marca metricas perfectas como bajas y conserva todos los empates del cutoff.
- Casos falsos `low_person_detection_ratio/main_track_coverage` restantes: **0**.
- QUALITY_WARNING sigue siendo metadata descriptiva; no implica exclusion.

## 3. Conteos reales de BAR

| Condicion | Videos |
|---|---:|
| detection_ratio = 1 | 645 |
| detection_ratio < 1 | 17 |
| main_track_coverage = 1 | 569 |
| main_track_coverage < 1 | 93 |
| main_track_coverage < 0.95 | 85 |
| main_track_coverage < 0.80 | 51 |
| main_track_coverage < 0.60 | 5 |

## 4. Distribuciones

| Metrica | N | Media | Mediana | Std | Min | P1 | P5 | P25 | P50 | P75 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bar_detection_ratio | 662 | 0.998948 | 1.000000 | 0.008330 | 0.906250 | 0.952244 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| bar_main_track_coverage | 662 | 0.968818 | 1.000000 | 0.089585 | 0.451613 | 0.608948 | 0.727693 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| bar_confidence_mean_por_video | 662 | 0.832891 | 0.838477 | 0.037048 | 0.640686 | 0.708664 | 0.767502 | 0.813907 | 0.838477 | 0.856840 | 0.883213 | 0.901155 | 0.910220 |
| bar_number_of_tracks | 662 | 1.561934 | 1.000000 | 1.070732 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 2.000000 | 4.000000 | 6.000000 | 8.000000 |
| bar_longest_missing_gap | 662 | 0.072508 | 0.000000 | 0.587534 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 2.000000 | 9.000000 |

## 5. Diagnostico de continuidad

- `continuous_single_track`: **461** videos.
- `continuous_main_with_residual_multiple_instances`: **107** videos.
- `fragmented_track_ids_with_competing_instances`: **49** videos.
- `fragmented_track_ids`: **29** videos.
- `raw_detection_gaps_only`: **8** videos.
- `fragmented_track_ids_rejected_handoff`: **7** videos.
- `accepted_bar_handoff`: **1** videos.

Videos con deteccion completa pero principal incompleto: **76**.
En estos casos la diferencia no es ausencia YOLO: existen detecciones BAR bajo otros track_id; la auditoria distingue competencia simultanea, fragmentacion y candidatos de handoff rechazados.

Desglose de esos casos:

- `fragmented_track_ids_with_competing_instances`: **46** videos.
- `fragmented_track_ids`: **24** videos.
- `fragmented_track_ids_rejected_handoff`: **6** videos.

Las causas observables mas comunes son cambio de track_id al variar posicion/forma de la barra, competencia entre detecciones simultaneas y handoffs geometricamente plausibles que no superan el criterio V2 estricto. La multiplicidad aislada no implica fallo si el principal conserva cobertura completa.

Categorias descriptivas:

- `A_complete_detection_complete_track`: **569** videos.
- `B_complete_detection_fragmented_track`: **73** videos.
- `C_incomplete_detection_reasonable_track`: **6** videos.
- `D_incomplete_detection_fragmented_track`: **3** videos.
- `E_extreme_visual_review`: **11** videos.

## 6. Casos estructurales

- Videos con multiples instancias BAR: **169**.
- Frames con multiples instancias BAR: **1167**.
- Videos con mas de un track BAR: **198**.
- Handoffs BAR aceptados: **1**.
- Videos con candidatos de handoff rechazados: **7**.

## 7. Revision visual focalizada

Se seleccionaron **14** videos sin labels.
### `bp_001`

- Seleccion: `perfect_single_track_control`.
- Frames: `50|35`.
- Clasificacion visual: `E_MAIN_TRACK_REPRESENTS_RELEVANT_TRAJECTORY`.
- Hallazgo: Control correcto: una sola identidad BAR estable y mascara coherente durante el movimiento.
- Riesgo para features temporales: `LOW`.

### `bp_004`

- Seleccion: `complete_detection_low_coverage|low_main_coverage`.
- Frames: `0|61|20|21|31`.
- Clasificacion visual: `B_TRACK_ID_CHANGE+C_PARTIAL_DUPLICATE+F_SEVERE`.
- Hallazgo: La barra esta segmentada durante toda la secuencia, pero se reparte entre tracks 0, 3 y 4; el principal solo conserva frames 20-47.
- Riesgo para features temporales: `HIGH`.

### `bp_019`

- Seleccion: `rejected_handoff_candidate`.
- Frames: `49|61|30|18|31`.
- Clasificacion visual: `B_TRACK_ID_CHANGE+F_SEVERE`.
- Hallazgo: La barra fisica continua bajo track 4 al final; el handoff fue rechazado y FEATURES pierde 13 frames con deteccion disponible.
- Riesgo para features temporales: `HIGH`.

### `bp_067`

- Seleccion: `rejected_handoff_candidate`.
- Frames: `68|89|66|14|45`.
- Clasificacion visual: `B_TRACK_ID_CHANGE+F_SEVERE`.
- Hallazgo: Transicion clara de track 0 a 3 sobre la misma barra fisica; el principal omite los 22 frames finales aunque la mascara sigue siendo util.
- Riesgo para features temporales: `HIGH`.

### `bp_139`

- Seleccion: `incomplete_detection|long_detection_gap`.
- Frames: `87|93|43|75|48`.
- Clasificacion visual: `A_REAL_RAW_LOSS+B_LATE_TRACK_FRAGMENT`.
- Hallazgo: La barra sale parcialmente del encuadre al final y hay nueve frames RAW ausentes; reaparece una deteccion residual bajo otro track.
- Riesgo para features temporales: `MEDIUM`.

### `bp_198`

- Seleccion: `low_main_coverage`.
- Frames: `49|27|54|14|20`.
- Clasificacion visual: `A_REAL_RAW_LOSS+B_TRACK_ID_CHANGE+F_SEVERE`.
- Hallazgo: Hay dos frames de perdida RAW, pero la mayor diferencia proviene de tracks 5 y 6 que continúan la barra mientras el principal termina en frame 26.
- Riesgo para features temporales: `HIGH`.

### `bp_241`

- Seleccion: `long_detection_gap`.
- Frames: `122|11|121|65`.
- Clasificacion visual: `A_REAL_RAW_LOSS+E_MAIN_MATCHES_RAW`.
- Hallazgo: El principal coincide exactamente con la disponibilidad RAW; los nueve frames faltantes aparecen al salir la barra del encuadre.
- Riesgo para features temporales: `MEDIUM`.

### `bp_301`

- Seleccion: `low_main_coverage`.
- Frames: `34|39|60|26|30`.
- Clasificacion visual: `A_SHORT_RAW_GAP+B_TRACK_ID_CHANGE+F_SEVERE`.
- Hallazgo: Tras un gap corto por desenfoque, track 4 recupera correctamente la barra pero el principal permanece en track 0 y omite 22 frames detectados.
- Riesgo para features temporales: `HIGH`.

### `bp_467`

- Seleccion: `incomplete_detection`.
- Frames: `44|50|63|8|32`.
- Clasificacion visual: `A_REAL_RAW_LOSS+B_TRACK_ID_CHANGE+F_SEVERE`.
- Hallazgo: Un gap visible separa tracks 0 y 4; la barra vuelve a estar correctamente segmentada, pero esos 14 frames no entran al principal.
- Riesgo para features temporales: `HIGH`.

### `bp_469`

- Seleccion: `incomplete_detection`.
- Frames: `57|60|67|22|34`.
- Clasificacion visual: `A_REAL_RAW_LOSS+B_LATE_TRACK_FRAGMENT`.
- Hallazgo: La mayor parte de la trayectoria permanece en track 0; tras gaps finales aparecen tracks cortos 4 y 5 sobre la barra real.
- Riesgo para features temporales: `MEDIUM`.

### `bp_518`

- Seleccion: `multiple_bar_instances`.
- Frames: `58|146|80`.
- Clasificacion visual: `C_PARTIAL_REPRESENTATIONS+D_RESIDUAL_FALSE_BAR+E_MAIN_CORRECT`.
- Hallazgo: El principal track 0 cubre toda la barra; las instancias laterales son fragmentos o detecciones residuales y no desplazan la trayectoria principal.
- Riesgo para features temporales: `LOW`.

### `bp_520`

- Seleccion: `accepted_handoff|multiple_bar_instances|perfect_main_multiple_instances_control`.
- Frames: `0|114|115|142|78`.
- Clasificacion visual: `B_AUDITED_HANDOFF+C_PARTIAL_REPRESENTATIONS+E_MAIN_CORRECT`.
- Hallazgo: El handoff original 0 a 5 corresponde a la misma barra y el track virtual 7 mantiene cobertura completa; los laterales residuales no alteran el principal.
- Riesgo para features temporales: `LOW`.

### `bp_541`

- Seleccion: `complete_detection_low_coverage|low_main_coverage`.
- Frames: `0|86|78|102|95`.
- Clasificacion visual: `B_TRACK_ID_CHANGE+F_SEVERE`.
- Hallazgo: Tracks 0 y 5 representan consecutivamente la misma barra, pero el principal final solo toma track 5 y pierde 87 frames iniciales disponibles.
- Riesgo para features temporales: `HIGH`.

### `bp_544`

- Seleccion: `complete_detection_low_coverage|low_main_coverage`.
- Frames: `0|95|21|101`.
- Clasificacion visual: `B_TRACK_ID_CHANGE+D_RESIDUAL_FALSE_BAR+F_SEVERE`.
- Hallazgo: La barra cambia de track 0 a 5 con continuidad visual; detecciones laterales residuales coexisten y el principal omite 96 frames disponibles.
- Riesgo para features temporales: `HIGH`.

## 8. Conclusion tecnica

La extraccion BP_FRONT_SEG_DATASET_V1 puede permanecer congelada, pero existe evidencia suficiente para justificar Tracking V3 antes de usar la continuidad BAR como feature temporal definitiva.

No se creo Tracking V3 ni se modificaron parametros, asociaciones, mascaras o features.
