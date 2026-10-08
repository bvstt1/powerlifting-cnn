# Reporte BP_FRONT Bar TrackingV3 Validation76

Fecha: `2026-10-07T15:18:54.175039+00:00`

## 1. Configuracion congelada

- Algoritmo: `bp_front_bar_tracking_v3_prototype_1`.
- Hash utils: `bf103391f9171c80739e52345375f464ea7ac822d22c71d4804c0c810350317d`.
- Hash runner: `8b1814e32a1fa8d631427901bf2a6701090b24e63a8f5a81ab2485585f3cfb10`.
- No se cambiaron parametros, thresholds ni pesos respecto al prototipo de 14 casos.
- YOLO=False; labels=False; interpolacion=False.

## 2. Conjunto

- Seleccion automatica: `bar_detection_ratio == 1.0` y `bar_main_track_coverage < 1.0`.
- N evaluados: **76**.
- IDs escritos manualmente: **0**.

## 3. Resultados

| Estado | N |
|---|---:|
| FULLY_RECOVERED | 72 |
| PARTIALLY_RECOVERED | 1 |
| UNCHANGED_CONSERVATIVE | 3 |
| REGRESSED | 0 |

## 4. Coverage

| Version | Mean | Median | Min | P5 | P25 | P75 | P95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| V2 | 0.762817 | 0.767779 | 0.451613 | 0.604866 | 0.688088 | 0.851749 | 0.924556 |
| V3 | 0.994055 | 1.000000 | 0.631579 | 0.996875 | 1.000000 | 1.000000 | 1.000000 |

- Incremento medio: **0.231238**.
- Frames recuperables omitidos por V2: **1680**.
- Frames recuperados por V3: **1641**.
- Recovery rate global: **0.976786**.

## 5. Handoffs

- Handoffs aceptados en path global: **79**.
- Candidatos compatibles/rechazados relevantes no seleccionados: **17**.
- Handoffs sospechosos: **47**.
- Edge score: {'mean': 0.8541755390450865, 'median': 0.8535022133067873, 'min': 0.6926170973316641, 'p5': 0.7536484615717848, 'p25': 0.8344685175349632, 'p75': 0.8831784604104136, 'p95': 0.9320246159233629}.
- Runner-up margin: {'mean': 27.035096160009868, 'median': 19.869616591046892, 'min': 2.022525409283446, 'p5': 8.800314084717606, 'p25': 13.209036655431305, 'p75': 29.38639967981173, 'p95': 72.76788408472152}.
- Gap frames: {'mean': 0.05063291139240506, 'median': 0.0, 'min': 0.0, 'p5': 0.0, 'p25': 0.0, 'p75': 0.0, 'p95': 0.0}.
- Handoffs/video: {'mean': 1.0394736842105263, 'median': 1.0, 'min': 0.0, 'p5': 1.0, 'p25': 1.0, 'p75': 1.0, 'p95': 2.0}.

## 6. Reproducibilidad prototype_14

- Casos originales contenidos en validation_76: **5**.
- Reproducidos exactamente: `bp_004`, `bp_019`, `bp_067`, `bp_541`, `bp_544`.
- Inconsistencias: **0**.

## 7. Casos sospechosos

- `bp_004` 0->3 frame 20, score=0.8907441930984471: `temporal_overlap`.
- `bp_009` 0->3 frame 58, score=0.8833884472998628: `temporal_overlap`.
- `bp_009` 3->4 frame 65, score=0.854108109223184: `competitive_handoff_candidates`.
- `bp_019` 0->4 frame 48, score=0.9294563933233632: `temporal_overlap`.
- `bp_067` 0->3 frame 66, score=0.9252700987951523: `temporal_overlap`.
- `bp_078` 0->3 frame 55, score=0.8627409390951402: `temporal_overlap`.
- `bp_081` 0->3 frame 68, score=0.8958610634033476: `temporal_overlap`.
- `bp_102` 0->3 frame 17, score=0.8814584385825949: `temporal_overlap`.
- `bp_103` 0->3 frame 47, score=0.8816953998109053: `temporal_overlap`.
- `bp_104` 0->3 frame 61, score=0.8664664887300209: `temporal_overlap`.
- `bp_113` 0->3 frame 44, score=0.8714519126876061: `temporal_overlap`.
- `bp_126` 0->5 frame 62, score=0.8235858700545742: `competitive_handoff_candidates`.
- `bp_147` 0->3 frame 21, score=0.8895156717723929: `temporal_overlap`.
- `bp_182` 0->4 frame 52, score=0.829795318779187: `temporal_overlap|competitive_handoff_candidates|three_or_more_handoffs`.
- `bp_182` 4->7 frame 64, score=0.6968405399653668: `temporal_overlap|large_length_change|competitive_handoff_candidates|three_or_more_handoffs`.
- `bp_182` 7->8 frame 75, score=0.8186502567373908: `three_or_more_handoffs`.
- `bp_186` 0->5 frame 34, score=0.8347245460748609: `temporal_overlap|competitive_handoff_candidates`.
- `bp_191` 0->4 frame 54, score=0.8857045610012394: `temporal_overlap`.
- `bp_192` 0->4 frame 74, score=0.8411295683676316: `temporal_overlap`.
- `bp_199` 0->4 frame 31, score=0.8995963991319302: `temporal_overlap`.
- `bp_200` 0->4 frame 22, score=0.9260799059614366: `temporal_overlap`.
- `bp_201` 0->3 frame 19, score=0.9423501784495452: `temporal_overlap`.
- `bp_222` 0->4 frame 24, score=0.8886949419967969: `temporal_overlap`.
- `bp_223` 0->5 frame 23, score=0.9338209175589953: `temporal_overlap|competitive_handoff_candidates`.
- `bp_224` 0->4 frame 21, score=0.8969204904211917: `temporal_overlap`.
- `bp_226` 0->4 frame 24, score=0.8811504253233442: `temporal_overlap`.
- `bp_227` 0->5 frame 14, score=0.9318250268527372: `temporal_overlap|competitive_handoff_candidates`.
- `bp_232` 0->4 frame 24, score=0.9132978109953627: `temporal_overlap`.
- `bp_233` 0->4 frame 18, score=0.9389893959199703: `temporal_overlap`.
- `bp_249` 0->4 frame 9, score=0.8824138658499582: `temporal_overlap`.
- `bp_277` 0->4 frame 65, score=0.8504032168265175: `temporal_overlap`.
- `bp_282` 0->5 frame 43, score=0.8490292927382735: `temporal_overlap`.
- `bp_313` 0->4 frame 130, score=0.8849947095595398: `temporal_overlap`.
- `bp_460` 0->4 frame 60, score=0.892966688055769: `temporal_overlap`.
- `bp_478` 0->5 frame 11, score=0.8517488878061268: `temporal_overlap`.
- `bp_517` 0->6 frame 62, score=0.9448493538166793: `temporal_overlap`.
- `bp_528` 0->5 frame 109, score=0.8337779002350173: `temporal_overlap`.
- `bp_538` 0->4 frame 51, score=0.6955849959606162: `temporal_overlap|large_length_change`.
- `bp_540` 0->4 frame 55, score=0.6926170973316641: `temporal_overlap|large_length_change`.
- `bp_541` 0->5 frame 88, score=0.808574326254801: `temporal_overlap|competitive_handoff_candidates`.
- `bp_543` 0->4 frame 43, score=0.880420482658999: `temporal_overlap`.
- `bp_544` 0->5 frame 96, score=0.7105286214873706: `temporal_overlap|large_length_change`.
- `bp_552` 0->4 frame 60, score=0.8545579977103649: `temporal_overlap`.
- `bp_557` 0->4 frame 50, score=0.8829684735209643: `temporal_overlap`.
- `bp_571` 0->7 frame 129, score=0.9185057616195147: `temporal_overlap`.
- `bp_585` 0->5 frame 69, score=0.8604541987727529: `temporal_overlap`.
- `bp_590` 0->5 frame 22, score=0.8795676634662702: `temporal_overlap`.

## 8. Auditoria visual

- Filas seleccionadas: **76**.
- Revisadas: **0**.
- Falsos merges confirmados: **0**.
- Overviews: `C:\Users\basti\MediapipePythonProjects\segmentation\bp\front\tracking_v3\validation_76\previews\_overviews\handoffs_overview_01.jpg`, `C:\Users\basti\MediapipePythonProjects\segmentation\bp\front\tracking_v3\validation_76\previews\_overviews\handoffs_overview_02.jpg`, `C:\Users\basti\MediapipePythonProjects\segmentation\bp\front\tracking_v3\validation_76\previews\_overviews\handoffs_overview_03.jpg`, `C:\Users\basti\MediapipePythonProjects\segmentation\bp\front\tracking_v3\validation_76\previews\_overviews\handoffs_overview_04.jpg`, `C:\Users\basti\MediapipePythonProjects\segmentation\bp\front\tracking_v3\validation_76\previews\_overviews\handoffs_overview_05.jpg`, `C:\Users\basti\MediapipePythonProjects\segmentation\bp\front\tracking_v3\validation_76\previews\_overviews\handoffs_overview_06.jpg`, `C:\Users\basti\MediapipePythonProjects\segmentation\bp\front\tracking_v3\validation_76\previews\_overviews\handoffs_overview_07.jpg`, `C:\Users\basti\MediapipePythonProjects\segmentation\bp\front\tracking_v3\validation_76\previews\_overviews\handoffs_overview_08.jpg`, `C:\Users\basti\MediapipePythonProjects\segmentation\bp\front\tracking_v3\validation_76\previews\_overviews\handoffs_overview_09.jpg`, `C:\Users\basti\MediapipePythonProjects\segmentation\bp\front\tracking_v3\validation_76\previews\_overviews\handoffs_overview_10.jpg`, `C:\Users\basti\MediapipePythonProjects\segmentation\bp\front\tracking_v3\validation_76\previews\_overviews\handoffs_overview_11.jpg`.

- `bp_004` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_005` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_009` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_019` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_050` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_052` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_056` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_058` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_059` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_067` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_078` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_079` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_081` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_085` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_086` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_094` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_096` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_099` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_102` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_103` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_104` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_108` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_113` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_123` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_126` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_129` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_134` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_147` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_165` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_171` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_179` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_182` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_183` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_186` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_191` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_192` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_194` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_199` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_200` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_201` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_222` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_223` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_224` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_226` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_227` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_232` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_233` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_238` [PARTIALLY_RECOVERED]: Pendiente de revision.
- `bp_249` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_266` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_277` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_282` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_297` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_300` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_311` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_313` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_342` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_358` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_460` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_478` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_517` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_528` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_536` [UNCHANGED_CONSERVATIVE]: Pendiente de revision.
- `bp_538` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_540` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_541` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_543` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_544` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_552` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_557` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_571` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_585` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_590` [FULLY_RECOVERED]: Pendiente de revision.
- `bp_661` [UNCHANGED_CONSERVATIVE]: Pendiente de revision.
- `bp_662` [UNCHANGED_CONSERVATIVE]: Pendiente de revision.
- `bp_663` [FULLY_RECOVERED]: Pendiente de revision.

## 9. Conclusion

VALIDACION PENDIENTE: faltan revisiones visuales.

No se modificaron RAW, mascaras, detecciones, Tracking V2 ni FEATURES V2.
No se procesaron los 662 videos y no se generaron FEATURES V3 globales.
