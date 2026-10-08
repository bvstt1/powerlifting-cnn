# Reporte BP_FRONT Bar TrackingV3 Prototype

Fecha: `2026-10-07T01:06:22.688505+00:00`

## A. Algoritmo V3

- Version: `bp_front_bar_tracking_v3_prototype_1`.
- Entrada: detecciones y mascaras existentes de Tracking V2; YOLO=False.
- Tracklets BAR V2 como nodos y handoffs compatibles como aristas de un DAG.
- Busqueda global determinista de la trayectoria que contiene la identidad principal V2.
- Fast path: si V2 ya cubre todos los frames BAR RAW, se conserva exactamente.
- No se interpolan gaps RAW; permanecen con indice -1.

## B. Diferencias con V2

V2 extiende greedy solo hacia delante, exige 2-5 frames solapados y geometria casi identica. V3 puede recuperar prefijos y sufijos, atravesar gaps RAW cortos sin inventar detecciones y compara caminos globales penalizando cada handoff.

## C. Features de asociacion

Centroide observado y predicho, gap temporal, orientacion, eje mayor, area, bbox, confidence, IoU/containment de mascaras y trayectoria robusta inicial/final.

## D. Score

Cada arista guarda `temporal`, `centroid_distance`, `predicted_motion`, `orientation`, `length`, `area`, `bbox_geometry`, `confidence` y `mask_transition`. El score total es su suma ponderada. El camino suma recompensa por deteccion existente y resta penalizacion de handoff, gaps e incompatibilidad.

## E. Thresholds

- `max_missing_gap_frames=8`.
- `max_overlap_frames=3`.
- `motion_window=5`.
- `max_track_center_x_delta_over_width=0.18`.
- `min_track_area_ratio=0.25`.
- `max_track_area_ratio=4.0`.
- `min_track_length_ratio=0.35`.
- `max_track_length_ratio=2.8`.
- `max_track_orientation_delta_deg=40.0`.
- `max_edge_x_delta_over_width=0.1`.
- `max_edge_y_delta_over_height=0.35`.
- `max_predicted_error_over_diagonal=0.15`.
- `minimum_edge_score=0.58`.
- `minimum_overlap_identity_evidence=0.35`.
- `handoff_penalty=1.5`.
- `gap_penalty_per_frame=0.08`.
- `minimum_coverage_gain=3`.
- `minimum_score_improvement=0.75`.
- `minimum_runner_up_margin=0.25`.

## F. Comparacion de los 14 videos

| Video | V2 coverage | V3 coverage | Handoffs | Path V3 | Resultado | Evaluacion visual |
|---|---:|---:|---:|---|---|---|
| bp_001 | 1.000000 | 1.000000 | 0 | 0 | UNCHANGED | Fast path conserva exactamente el track simple y estable de V2 |
| bp_004 | 0.451613 | 1.000000 | 2 | 0->3->4 | IMPROVED | Los handoffs 0->3 y 3->4 siguen la misma barra; V3 recupera los 62 frames sin saltos fisicos |
| bp_019 | 0.790323 | 1.000000 | 1 | 0->4 | IMPROVED | El handoff 0->4 recupera el sufijo de la misma barra y no selecciona la deteccion residual simultanea |
| bp_067 | 0.755556 | 1.000000 | 1 | 0->3 | IMPROVED | La transicion 0->3 es continua en posicion, forma y movimiento; se recuperan los 22 frames finales |
| bp_139 | 0.896907 | 0.896907 | 0 | 0 | UNCHANGED | V3 rechaza correctamente anexar el singleton terminal y conserva los gaps RAW reales |
| bp_198 | 0.490909 | 0.963636 | 2 | 0->5->6 | IMPROVED | Los handoffs 0->5 y 5->6 mantienen la misma barra; los dos frames RAW ausentes siguen vacios |
| bp_241 | 0.931298 | 0.931298 | 0 | 0 | UNCHANGED | Fast path conserva V2 y no intenta rellenar la salida real de la barra fuera del encuadre |
| bp_301 | 0.573770 | 0.934426 | 1 | 0->4 | IMPROVED | V3 mantiene vacio el gap por desenfoque y enlaza correctamente la reaparicion track 4 |
| bp_467 | 0.687500 | 0.906250 | 1 | 0->4 | IMPROVED | El handoff 0->4 recupera la misma barra despues del gap RAW de seis frames sin interpolacion |
| bp_469 | 0.838235 | 0.838235 | 0 | 0 | UNCHANGED | V3 conserva V2 porque las rutas 0->5 y 0->4->5 son ambiguas; no fuerza los fragmentos terminales |
| bp_518 | 1.000000 | 1.000000 | 0 | 0 | UNCHANGED | Fast path mantiene track 0 completo y no fusiona fragmentos laterales ni detecciones residuales |
| bp_520 | 1.000000 | 1.000000 | 0 | 7 | UNCHANGED | Fast path conserva exactamente el track virtual 7 y el handoff V2 auditado 0->5 |
| bp_541 | 0.544503 | 1.000000 | 1 | 0->5 | IMPROVED | El prefijo track 0 y el principal track 5 son la misma barra; V3 recupera los 191 frames |
| bp_544 | 0.524752 | 1.000000 | 1 | 0->5 | IMPROVED | El handoff 0->5 conserva identidad fisica y cobertura completa sin incorporar residuos laterales |

## G. Handoffs aceptados

- `bp_004` 0->3 en frame 20: score=0.8907441930984471, gap=0, pred_error=0.011774272083188025.
- `bp_004` 3->4 en frame 48: score=0.8372470013937338, gap=0, pred_error=0.036799230244193425.
- `bp_019` 0->4 en frame 48: score=0.9294563933233632, gap=0, pred_error=0.006569647081057785.
- `bp_067` 0->3 en frame 66: score=0.9252700987951523, gap=0, pred_error=0.004868679128225335.
- `bp_198` 0->5 en frame 27: score=0.8732315432236845, gap=0, pred_error=0.008574735958118421.
- `bp_198` 5->6 en frame 42: score=0.7989690874673947, gap=0, pred_error=0.045852108213094205.
- `bp_301` 0->4 en frame 39: score=0.6187462295993318, gap=2, pred_error=0.06170081060892022.
- `bp_467` 0->4 en frame 50: score=0.6850578952200996, gap=6, pred_error=0.1090488949542273.
- `bp_541` 0->5 en frame 88: score=0.808574326254801, gap=0, pred_error=0.0470986775359276.
- `bp_544` 0->5 en frame 96: score=0.7105286214873706, gap=0, pred_error=0.073024248823445.

## H. Handoffs rechazados relevantes

- `bp_198` 7->8: score=0.5442756914682728; `score_below_threshold`.
- `bp_198` 5->8: score=0.48854456398164275; `vertical_jump_too_large|predicted_motion_incompatible|score_below_threshold`.

## I. Casos mejorados

`bp_004`, `bp_019`, `bp_067`, `bp_198`, `bp_301`, `bp_467`, `bp_541`, `bp_544`

## J. Casos no mejorados

`bp_001`, `bp_139`, `bp_241`, `bp_469`, `bp_518`, `bp_520`

## K. Regresiones

Ninguna.

## L. Conclusion

V3 aprobado para ampliar prueba a los 76 casos de deteccion completa y track incompleto.

No se modificaron RAW, mascaras, detecciones, Tracking V2 ni FEATURES V2. No se entreno ninguna red.
