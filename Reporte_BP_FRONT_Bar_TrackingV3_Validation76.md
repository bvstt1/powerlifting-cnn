# Reporte BP_FRONT Bar TrackingV3 Validation76

Fecha: `2026-10-09T08:05:28.536747+00:00`

## Objetivo

Cerrar formalmente Validation76 del algoritmo congelado mediante reejecucion determinista, revision visual explicita 76/76 y evidencia 1:1 para los 79 handoffs aceptados.

## 1. Configuracion congelada

- Algoritmo: `bp_front_bar_tracking_v3_prototype_1`.
- Hash utils: `bf103391f9171c80739e52345375f464ea7ac822d22c71d4804c0c810350317d`.
- Hash runner: `8b1814e32a1fa8d631427901bf2a6701090b24e63a8f5a81ab2485585f3cfb10`.
- No se cambiaron parametros, thresholds ni pesos respecto al prototipo de 14 casos.
- YOLO=False; labels=False; interpolacion=False.

### Parametros

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

## 2. Correccion de auditoria

- Se elimino el auto-review; ninguna fila sale de `PENDING` sin registro explicito.
- `algorithm_status` permanece separado de `visual_review_status`.
- Los resultados visuales no reescriben coverage, path, score ni decision V3.

## 3. Conjunto

- Seleccion automatica: `bar_detection_ratio == 1.0` y `bar_main_track_coverage < 1.0`.
- N evaluados: **76**.
- IDs escritos manualmente: **0**.

## 4. Resultados tecnicos

| Estado | N |
|---|---:|
| FULLY_RECOVERED | 72 |
| PARTIALLY_RECOVERED | 1 |
| UNCHANGED_CONSERVATIVE | 3 |
| REGRESSED | 0 |

## 5. Coverage

| Version | Mean | Median | Min | P5 | P25 | P75 | P95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| V2 | 0.762817 | 0.767779 | 0.451613 | 0.604866 | 0.688088 | 0.851749 | 0.924556 |
| V3 | 0.994055 | 1.000000 | 0.631579 | 0.996875 | 1.000000 | 1.000000 | 1.000000 |

- Incremento medio: **0.231238**.
- Frames recuperables omitidos por V2: **1680**.
- Frames recuperados por V3: **1641**.
- Recovery rate global: **0.976786**.

## 6. Handoffs

- Handoffs aceptados en path global: **79**.
- Candidatos compatibles/rechazados relevantes no seleccionados: **17**.
- Handoffs sospechosos: **47**.
- Handoffs revisados individualmente: **79/79**.
- Evidencias unicas de handoff: **79**.
- Falsos merges confirmados: **2**.
- Edge score: {'mean': 0.8541755390450865, 'median': 0.8535022133067873, 'min': 0.6926170973316641, 'p5': 0.7536484615717848, 'p25': 0.8344685175349632, 'p75': 0.8831784604104136, 'p95': 0.9320246159233629}.
- Runner-up margin: {'mean': 27.035096160009868, 'median': 19.869616591046892, 'min': 2.022525409283446, 'p5': 8.800314084717606, 'p25': 13.209036655431305, 'p75': 29.38639967981173, 'p95': 72.76788408472152}.
- Gap frames: {'mean': 0.05063291139240506, 'median': 0.0, 'min': 0.0, 'p5': 0.0, 'p25': 0.0, 'p75': 0.0, 'p95': 0.0}.
- Handoffs/video: {'mean': 1.0394736842105263, 'median': 1.0, 'min': 0.0, 'p5': 1.0, 'p25': 1.0, 'p75': 1.0, 'p95': 2.0}.

## 7. Reproducibilidad prototype_14

- Casos originales contenidos en validation_76: **5**.
- Reproducidos exactamente: `bp_004`, `bp_019`, `bp_067`, `bp_541`, `bp_544`.
- Inconsistencias: **0**.

## 8. Casos sospechosos

- `bp_004` 0->3 frame 20, score=0.8907441930984471, review=VISUALLY_CONFIRMED: `temporal_overlap`.
- `bp_009` 0->3 frame 58, score=0.8833884472998628, review=VISUALLY_CONFIRMED: `temporal_overlap`.
- `bp_009` 3->4 frame 65, score=0.854108109223184, review=VISUALLY_CONFIRMED: `competitive_handoff_candidates`.
- `bp_019` 0->4 frame 48, score=0.9294563933233632, review=VISUALLY_CONFIRMED: `temporal_overlap`.
- `bp_067` 0->3 frame 66, score=0.9252700987951523, review=VISUALLY_CONFIRMED: `temporal_overlap`.
- `bp_078` 0->3 frame 55, score=0.8627409390951402, review=VISUALLY_CONFIRMED: `temporal_overlap`.
- `bp_081` 0->3 frame 68, score=0.8958610634033476, review=VISUALLY_CONFIRMED: `temporal_overlap`.
- `bp_102` 0->3 frame 17, score=0.8814584385825949, review=VISUALLY_CONFIRMED: `temporal_overlap`.
- `bp_103` 0->3 frame 47, score=0.8816953998109053, review=VISUALLY_CONFIRMED: `temporal_overlap`.
- `bp_104` 0->3 frame 61, score=0.8664664887300209, review=VISUALLY_CONFIRMED: `temporal_overlap`.
- `bp_113` 0->3 frame 44, score=0.8714519126876061, review=VISUALLY_CONFIRMED: `temporal_overlap`.
- `bp_126` 0->5 frame 62, score=0.8235858700545742, review=VISUALLY_CONFIRMED: `competitive_handoff_candidates`.
- `bp_147` 0->3 frame 21, score=0.8895156717723929, review=VISUALLY_CONFIRMED: `temporal_overlap`.
- `bp_182` 0->4 frame 52, score=0.829795318779187, review=VISUALLY_CONFIRMED: `temporal_overlap|competitive_handoff_candidates|three_or_more_handoffs`.
- `bp_182` 4->7 frame 64, score=0.6968405399653668, review=VISUALLY_CONFIRMED: `temporal_overlap|large_length_change|competitive_handoff_candidates|three_or_more_handoffs`.
- `bp_182` 7->8 frame 75, score=0.8186502567373908, review=VISUALLY_CONFIRMED: `three_or_more_handoffs`.
- `bp_186` 0->5 frame 34, score=0.8347245460748609, review=VISUALLY_CONFIRMED: `temporal_overlap|competitive_handoff_candidates`.
- `bp_191` 0->4 frame 54, score=0.8857045610012394, review=VISUALLY_CONFIRMED: `temporal_overlap`.
- `bp_192` 0->4 frame 74, score=0.8411295683676316, review=VISUALLY_CONFIRMED: `temporal_overlap`.
- `bp_199` 0->4 frame 31, score=0.8995963991319302, review=VISUALLY_CONFIRMED: `temporal_overlap`.
- `bp_200` 0->4 frame 22, score=0.9260799059614366, review=VISUALLY_CONFIRMED: `temporal_overlap`.
- `bp_201` 0->3 frame 19, score=0.9423501784495452, review=VISUALLY_CONFIRMED: `temporal_overlap`.
- `bp_222` 0->4 frame 24, score=0.8886949419967969, review=VISUALLY_CONFIRMED: `temporal_overlap`.
- `bp_223` 0->5 frame 23, score=0.9338209175589953, review=VISUALLY_CONFIRMED: `temporal_overlap|competitive_handoff_candidates`.
- `bp_224` 0->4 frame 21, score=0.8969204904211917, review=VISUALLY_CONFIRMED: `temporal_overlap`.
- `bp_226` 0->4 frame 24, score=0.8811504253233442, review=VISUALLY_CONFIRMED: `temporal_overlap`.
- `bp_227` 0->5 frame 14, score=0.9318250268527372, review=VISUALLY_CONFIRMED: `temporal_overlap|competitive_handoff_candidates`.
- `bp_232` 0->4 frame 24, score=0.9132978109953627, review=VISUALLY_CONFIRMED: `temporal_overlap`.
- `bp_233` 0->4 frame 18, score=0.9389893959199703, review=VISUALLY_CONFIRMED: `temporal_overlap`.
- `bp_249` 0->4 frame 9, score=0.8824138658499582, review=VISUALLY_CONFIRMED: `temporal_overlap`.
- `bp_277` 0->4 frame 65, score=0.8504032168265175, review=VISUALLY_CONFIRMED: `temporal_overlap`.
- `bp_282` 0->5 frame 43, score=0.8490292927382735, review=VISUALLY_CONFIRMED: `temporal_overlap`.
- `bp_313` 0->4 frame 130, score=0.8849947095595398, review=VISUALLY_CONFIRMED: `temporal_overlap`.
- `bp_460` 0->4 frame 60, score=0.892966688055769, review=VISUALLY_CONFIRMED: `temporal_overlap`.
- `bp_478` 0->5 frame 11, score=0.8517488878061268, review=VISUALLY_CONFIRMED: `temporal_overlap`.
- `bp_517` 0->6 frame 62, score=0.9448493538166793, review=VISUALLY_CONFIRMED: `temporal_overlap`.
- `bp_528` 0->5 frame 109, score=0.8337779002350173, review=VISUALLY_CONFIRMED: `temporal_overlap`.
- `bp_538` 0->4 frame 51, score=0.6955849959606162, review=VISUALLY_CONFIRMED: `temporal_overlap|large_length_change`.
- `bp_540` 0->4 frame 55, score=0.6926170973316641, review=VISUALLY_CONFIRMED: `temporal_overlap|large_length_change`.
- `bp_541` 0->5 frame 88, score=0.808574326254801, review=VISUALLY_CONFIRMED: `temporal_overlap|competitive_handoff_candidates`.
- `bp_543` 0->4 frame 43, score=0.880420482658999, review=VISUALLY_CONFIRMED: `temporal_overlap`.
- `bp_544` 0->5 frame 96, score=0.7105286214873706, review=VISUALLY_CONFIRMED: `temporal_overlap|large_length_change`.
- `bp_552` 0->4 frame 60, score=0.8545579977103649, review=VISUALLY_CONFIRMED: `temporal_overlap`.
- `bp_557` 0->4 frame 50, score=0.8829684735209643, review=VISUALLY_CONFIRMED: `temporal_overlap`.
- `bp_571` 0->7 frame 129, score=0.9185057616195147, review=VISUALLY_CONFIRMED: `temporal_overlap`.
- `bp_585` 0->5 frame 69, score=0.8604541987727529, review=VISUALLY_CONFIRMED: `temporal_overlap`.
- `bp_590` 0->5 frame 22, score=0.8795676634662702, review=VISUALLY_CONFIRMED: `temporal_overlap`.

## 9. Auditoria visual

- Videos revisados explicitamente: **76/76**.
- Videos PENDING: **0**.
- Handoffs PENDING: **0**.
- Videos con falso merge: **1**.
- Handoffs con falso merge: **2**.
- RAW gaps preservados: **76/76** (el filtro contiene deteccion RAW completa).
- Los overviews son auxiliares y no cuentan como evidencia 1:1.

- `bp_004` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_004/00_0_to_3/bp_004_0_to_3_contact_sheet.jpg|segmentation/bp/front/tracking_v3/validation_76/previews/bp_004/01_3_to_4/bp_004_3_to_4_contact_sheet.jpg`.
- `bp_005` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_005/00_0_to_3/bp_005_0_to_3_contact_sheet.jpg`.
- `bp_009` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_009/00_0_to_3/bp_009_0_to_3_contact_sheet.jpg|segmentation/bp/front/tracking_v3/validation_76/previews/bp_009/01_3_to_4/bp_009_3_to_4_contact_sheet.jpg`.
- `bp_019` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_019/00_0_to_4/bp_019_0_to_4_contact_sheet.jpg`.
- `bp_050` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_050/00_0_to_3/bp_050_0_to_3_contact_sheet.jpg`.
- `bp_052` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_052/00_0_to_3/bp_052_0_to_3_contact_sheet.jpg`.
- `bp_056` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_056/00_0_to_3/bp_056_0_to_3_contact_sheet.jpg`.
- `bp_058` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_058/00_0_to_3/bp_058_0_to_3_contact_sheet.jpg`.
- `bp_059` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_059/00_0_to_3/bp_059_0_to_3_contact_sheet.jpg`.
- `bp_067` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_067/00_0_to_3/bp_067_0_to_3_contact_sheet.jpg`.
- `bp_078` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_078/00_0_to_3/bp_078_0_to_3_contact_sheet.jpg`.
- `bp_079` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_079/00_0_to_3/bp_079_0_to_3_contact_sheet.jpg`.
- `bp_081` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_081/00_0_to_3/bp_081_0_to_3_contact_sheet.jpg`.
- `bp_085` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_085/00_0_to_3/bp_085_0_to_3_contact_sheet.jpg`.
- `bp_086` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_086/00_0_to_3/bp_086_0_to_3_contact_sheet.jpg`.
- `bp_094` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_094/00_0_to_3/bp_094_0_to_3_contact_sheet.jpg`.
- `bp_096` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_096/00_0_to_3/bp_096_0_to_3_contact_sheet.jpg`.
- `bp_099` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_099/00_0_to_3/bp_099_0_to_3_contact_sheet.jpg`.
- `bp_102` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_102/00_0_to_3/bp_102_0_to_3_contact_sheet.jpg|segmentation/bp/front/tracking_v3/validation_76/previews/bp_102/01_3_to_4/bp_102_3_to_4_contact_sheet.jpg`.
- `bp_103` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_103/00_0_to_3/bp_103_0_to_3_contact_sheet.jpg`.
- `bp_104` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_104/00_0_to_3/bp_104_0_to_3_contact_sheet.jpg`.
- `bp_108` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_108/00_0_to_5/bp_108_0_to_5_contact_sheet.jpg`.
- `bp_113` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_113/00_0_to_3/bp_113_0_to_3_contact_sheet.jpg`.
- `bp_123` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_123/00_0_to_4/bp_123_0_to_4_contact_sheet.jpg`.
- `bp_126` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_126/00_0_to_5/bp_126_0_to_5_contact_sheet.jpg`.
- `bp_129` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_129/00_0_to_3/bp_129_0_to_3_contact_sheet.jpg`.
- `bp_134` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_134/00_0_to_3/bp_134_0_to_3_contact_sheet.jpg`.
- `bp_147` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_147/00_0_to_3/bp_147_0_to_3_contact_sheet.jpg`.
- `bp_165` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_165/00_0_to_3/bp_165_0_to_3_contact_sheet.jpg`.
- `bp_171` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_171/00_0_to_3/bp_171_0_to_3_contact_sheet.jpg`.
- `bp_179` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_179/00_0_to_3/bp_179_0_to_3_contact_sheet.jpg`.
- `bp_182` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Los tres handoffs 0->4->7->8 fueron inspeccionados individualmente y mantienen la misma barra fisica; 4->7 cambia cobertura, no identidad. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_182/00_0_to_4/bp_182_0_to_4_contact_sheet.jpg|segmentation/bp/front/tracking_v3/validation_76/previews/bp_182/01_4_to_7/bp_182_4_to_7_contact_sheet.jpg|segmentation/bp/front/tracking_v3/validation_76/previews/bp_182/02_7_to_8/bp_182_7_to_8_contact_sheet.jpg`.
- `bp_183` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_183/00_0_to_4/bp_183_0_to_4_contact_sheet.jpg`.
- `bp_186` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_186/00_0_to_5/bp_186_0_to_5_contact_sheet.jpg`.
- `bp_191` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_191/00_0_to_4/bp_191_0_to_4_contact_sheet.jpg`.
- `bp_192` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_192/00_0_to_4/bp_192_0_to_4_contact_sheet.jpg`.
- `bp_194` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_194/00_0_to_4/bp_194_0_to_4_contact_sheet.jpg`.
- `bp_199` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_199/00_0_to_4/bp_199_0_to_4_contact_sheet.jpg`.
- `bp_200` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_200/00_0_to_4/bp_200_0_to_4_contact_sheet.jpg`.
- `bp_201` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_201/00_0_to_3/bp_201_0_to_3_contact_sheet.jpg`.
- `bp_222` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_222/00_0_to_4/bp_222_0_to_4_contact_sheet.jpg`.
- `bp_223` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_223/00_0_to_5/bp_223_0_to_5_contact_sheet.jpg`.
- `bp_224` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_224/00_0_to_4/bp_224_0_to_4_contact_sheet.jpg`.
- `bp_226` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_226/00_0_to_4/bp_226_0_to_4_contact_sheet.jpg`.
- `bp_227` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_227/00_0_to_5/bp_227_0_to_5_contact_sheet.jpg`.
- `bp_232` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_232/00_0_to_4/bp_232_0_to_4_contact_sheet.jpg`.
- `bp_233` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_233/00_0_to_4/bp_233_0_to_4_contact_sheet.jpg`.
- `bp_238` [PARTIALLY_RECOVERED / VISUALLY_CONFIRMED_PARTIAL]: PARTIALLY_RECOVERED confirmado. El handoff 0->7 es correcto; cinco frames de la misma barra quedan sin asociar porque el track 6 supera el solape maximo congelado. Se registra subrecuperacion sin forzar merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_238/00_0_to_7/bp_238_0_to_7_contact_sheet.jpg`.
- `bp_249` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_249/00_0_to_4/bp_249_0_to_4_contact_sheet.jpg`.
- `bp_266` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_266/00_0_to_4/bp_266_0_to_4_contact_sheet.jpg`.
- `bp_277` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_277/00_0_to_4/bp_277_0_to_4_contact_sheet.jpg`.
- `bp_282` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_282/00_0_to_5/bp_282_0_to_5_contact_sheet.jpg`.
- `bp_297` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_297/00_0_to_4/bp_297_0_to_4_contact_sheet.jpg`.
- `bp_300` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_300/00_0_to_4/bp_300_0_to_4_contact_sheet.jpg`.
- `bp_311` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_311/00_0_to_4/bp_311_0_to_4_contact_sheet.jpg`.
- `bp_313` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_313/00_0_to_4/bp_313_0_to_4_contact_sheet.jpg`.
- `bp_342` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_342/00_0_to_4/bp_342_0_to_4_contact_sheet.jpg`.
- `bp_358` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_358/00_0_to_4/bp_358_0_to_4_contact_sheet.jpg`.
- `bp_460` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_460/00_0_to_4/bp_460_0_to_4_contact_sheet.jpg`.
- `bp_478` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_478/00_0_to_5/bp_478_0_to_5_contact_sheet.jpg`.
- `bp_517` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_517/00_0_to_6/bp_517_0_to_6_contact_sheet.jpg`.
- `bp_528` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_528/00_0_to_5/bp_528_0_to_5_contact_sheet.jpg`.
- `bp_536` [UNCHANGED_CONSERVATIVE / VISUALLY_CONFIRMED_CONSERVATIVE]: Conservador correcto: la propuesta no aporta ganancia minima ni mejora de score; se mantiene V2 con cinco huecos internos. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_536/00_0_to_0/bp_536_0_to_0_contact_sheet.jpg`.
- `bp_538` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Handoff de score bajo revisado: cambio de longitud por segmentacion parcial/completa de la misma barra; sin falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_538/00_0_to_4/bp_538_0_to_4_contact_sheet.jpg`.
- `bp_540` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Handoff de score bajo revisado: mascara corta y mascara completa permanecen colineales sobre la misma barra; sin falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_540/00_0_to_4/bp_540_0_to_4_contact_sheet.jpg`.
- `bp_541` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_541/00_0_to_5/bp_541_0_to_5_contact_sheet.jpg`.
- `bp_543` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_543/00_0_to_4/bp_543_0_to_4_contact_sheet.jpg`.
- `bp_544` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Handoff revisado: cambio grande de longitud por extension de mascara sobre la misma barra; sin falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_544/00_0_to_5/bp_544_0_to_5_contact_sheet.jpg`.
- `bp_552` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_552/00_0_to_4/bp_552_0_to_4_contact_sheet.jpg`.
- `bp_557` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_557/00_0_to_4/bp_557_0_to_4_contact_sheet.jpg`.
- `bp_571` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_571/00_0_to_7/bp_571_0_to_7_contact_sheet.jpg`.
- `bp_585` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_585/00_0_to_5/bp_585_0_to_5_contact_sheet.jpg`.
- `bp_590` [FULLY_RECOVERED / VISUALLY_CONFIRMED]: Todos los contact sheets de sus handoffs aceptados fueron inspeccionados individualmente; la identidad de barra es consistente y no se observa falso merge. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_590/00_0_to_5/bp_590_0_to_5_contact_sheet.jpg`.
- `bp_661` [UNCHANGED_CONSERVATIVE / VISUALLY_CONFIRMED_CONSERVATIVE]: Conservador correcto: solo un frame recuperable y la propuesta corresponde a la estructura inferior, no a la barra sostenida. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_661/00_0_to_0/bp_661_0_to_0_contact_sheet.jpg`.
- `bp_662` [UNCHANGED_CONSERVATIVE / VISUALLY_CONFIRMED_CONSERVATIVE]: Conservador correcto: 0->4 y 4->5 muestran saltos entre barra superior e inferior; la evidencia contradice identidad comun. El track base V2 conservado presenta identidad visual cuestionable, pero V3 no publica nuevos merges. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_662/00_0_to_4/bp_662_0_to_4_contact_sheet.jpg|segmentation/bp/front/tracking_v3/validation_76/previews/bp_662/01_4_to_5/bp_662_4_to_5_contact_sheet.jpg`.
- `bp_663` [FULLY_RECOVERED / VISUAL_PROBLEM]: VISUAL_PROBLEM: los dos handoffs publicados 0->4 y 4->5 cambian entre la barra superior sostenida y una estructura inferior. Dos falsos merges confirmados. Evidencia: `segmentation/bp/front/tracking_v3/validation_76/previews/bp_663/00_0_to_4/bp_663_0_to_4_contact_sheet.jpg|segmentation/bp/front/tracking_v3/validation_76/previews/bp_663/01_4_to_5/bp_663_4_to_5_contact_sheet.jpg`.

## 10. Casos prioritarios

- `bp_182`: Los tres handoffs 0->4->7->8 fueron inspeccionados individualmente y mantienen la misma barra fisica; 4->7 cambia cobertura, no identidad.
- `bp_238`: PARTIALLY_RECOVERED confirmado. El handoff 0->7 es correcto; cinco frames de la misma barra quedan sin asociar porque el track 6 supera el solape maximo congelado. Se registra subrecuperacion sin forzar merge.
- `bp_536`: Conservador correcto: la propuesta no aporta ganancia minima ni mejora de score; se mantiene V2 con cinco huecos internos.
- `bp_538`: Handoff de score bajo revisado: cambio de longitud por segmentacion parcial/completa de la misma barra; sin falso merge.
- `bp_540`: Handoff de score bajo revisado: mascara corta y mascara completa permanecen colineales sobre la misma barra; sin falso merge.
- `bp_544`: Handoff revisado: cambio grande de longitud por extension de mascara sobre la misma barra; sin falso merge.
- `bp_661`: Conservador correcto: solo un frame recuperable y la propuesta corresponde a la estructura inferior, no a la barra sostenida.
- `bp_662`: Conservador correcto: 0->4 y 4->5 muestran saltos entre barra superior e inferior; la evidencia contradice identidad comun. El track base V2 conservado presenta identidad visual cuestionable, pero V3 no publica nuevos merges.
- `bp_663`: VISUAL_PROBLEM: los dos handoffs publicados 0->4 y 4->5 cambian entre la barra superior sostenida y una estructura inferior. Dos falsos merges confirmados.

## 11. Integridad y verificador

- Verificador read-only: **NOT_OK**.
- Checks aprobados: **27/28**.
- Integridad PRE/POST: **OK**.
- Prototype suite: **Tracking V3 prototype tests: OK (14/14)**.
- Artefactos RAW/V2/features/audits comparados PRE/POST: **304**; discrepancias: **0**.
- Hash utils PRE: `bf103391f9171c80739e52345375f464ea7ac822d22c71d4804c0c810350317d`; POST: `bf103391f9171c80739e52345375f464ea7ac822d22c71d4804c0c810350317d`.
- Hash runner PRE: `8b1814e32a1fa8d631427901bf2a6701090b24e63a8f5a81ab2485585f3cfb10`; POST: `8b1814e32a1fa8d631427901bf2a6701090b24e63a8f5a81ab2485585f3cfb10`.
- Hash auditor PRE: `95ccd60de9e7a11f02357ad02724a9ae087eb28f6fda62fcf7a2ffab77c444bd`; POST: `5f400b82f023b85af201fe256aa25245be974160f23986617284cad1e7f2782f` (cambio permitido y limitado a auditoria).

## 12. Conclusion

**VALIDATION76_NOT_APPROVED: se confirmaron 2 falsos merges visuales en 1 video(s).**

No se modificaron RAW, mascaras, detecciones, Tracking V2 ni FEATURES V2.
No se procesaron los 662 videos y no se generaron FEATURES V3 globales.
