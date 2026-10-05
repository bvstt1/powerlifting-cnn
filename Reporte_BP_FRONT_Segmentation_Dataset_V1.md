# Reporte BP_FRONT Segmentation Dataset V1

Fecha de cierre: `2026-10-05T22:09:15.098892+00:00`

## 1. Configuracion exacta

- Pipeline: `BP_FRONT_SEG_DATASET_V1`.
- Extractor: `bp_front_segmentation_extractor_v1`.
- Tracking: `bp_front_bench_tracking_v2.2`.
- Modelo: `C:\Users\basti\MediapipePythonProjects\models\bp\bp_front_seg_v2.pt`.
- SHA-256 modelo: `22514988978ac6bb0c9b86e14b37043a58912ca0e30be3da20571c9172020f38`.
- Clases: `0=bar`, `1=bench`, `2=person`.
- `imgsz=960`, `conf=0.25`, `iou=0.7`, `retina_masks=True`.
- `proximity_candidate=0.01 x diagonal`, solo descriptivo.
- Configuracion congelada: `segmentation/bp/front/manifests/bp_front_dataset_v1_metadata.json`.

## 2. Entorno

- Git: `09fc8057b4d17f3b107ff436373378848a3fbd10` (worktree dirty al congelar: `True`).
- Python: `3.10.11`.
- Ultralytics: `8.4.137`.
- PyTorch: `2.13.0.dev20260521+cu130`; CUDA: `13.0`.
- OpenCV: `4.13.0`; GPU: `NVIDIA GeForce RTX 5060 Laptop GPU`.

## 3. Integridad y alcance

- Videos registrados: **663/663**.
- COMPLETE: **176**; WARNING: **486**; ERROR: **1**.
- Frames procesados con T original: **57913**.
- Instancias/mascaras: **212930**.
- Round-trip correcto: **212930/212930**; fallidos: **0**.
- QUALITY_OK: **176**; QUALITY_WARNING: **486**; QUALITY_ERROR: **1**.

### Detection ratio por video

| Grupo | N | Media | Mediana | Std | Min | P1 | P5 | P25 | P50 | P75 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bar | 662 | 0.998948 | 1.000000 | 0.008330 | 0.906250 | 0.952244 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| bench | 662 | 0.999359 | 1.000000 | 0.009682 | 0.826667 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| person | 662 | 1.000000 | 1.000000 | 0.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

### Main track coverage por video

| Grupo | N | Media | Mediana | Std | Min | P1 | P5 | P25 | P50 | P75 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bar | 662 | 0.968818 | 1.000000 | 0.089585 | 0.451613 | 0.608948 | 0.727693 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| bench | 662 | 0.999359 | 1.000000 | 0.009682 | 0.826667 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| person | 662 | 1.000000 | 1.000000 | 0.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

### Confidence global por instancia

| Grupo | N | Media | Mediana | Std | Min | P1 | P5 | P25 | P50 | P75 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bar | 59090 | 0.829976 | 0.846552 | 0.085433 | 0.250450 | 0.334034 | 0.716691 | 0.824030 | 0.846552 | 0.867135 | 0.896220 | 0.912019 | 0.926748 |
| bench | 95927 | 0.909468 | 0.930245 | 0.076674 | 0.250194 | 0.569257 | 0.809010 | 0.878026 | 0.930245 | 0.961963 | 0.968659 | 0.970654 | 0.972343 |
| person | 57913 | 0.963747 | 0.965765 | 0.008354 | 0.738604 | 0.923244 | 0.953645 | 0.962932 | 0.965765 | 0.967500 | 0.969599 | 0.970934 | 0.972798 |

## 4. Videos extremos

### 10 peores bar por detection_ratio

- `bp_467`: detection_ratio=0.906250
- `bp_139`: detection_ratio=0.907216
- `bp_469`: detection_ratio=0.911765
- `bp_137`: detection_ratio=0.930556
- `bp_241`: detection_ratio=0.931298
- `bp_468`: detection_ratio=0.934211
- `bp_301`: detection_ratio=0.934426
- `bp_198`: detection_ratio=0.963636
- `bp_474`: detection_ratio=0.970588
- `bp_236`: detection_ratio=0.978261

### 10 peores bar por main_track_coverage

- `bp_004`: main_track_coverage=0.451613
- `bp_198`: main_track_coverage=0.490909
- `bp_544`: main_track_coverage=0.524752
- `bp_541`: main_track_coverage=0.544503
- `bp_301`: main_track_coverage=0.573770
- `bp_182`: main_track_coverage=0.604651
- `bp_194`: main_track_coverage=0.604938
- `bp_540`: main_track_coverage=0.611511
- `bp_663`: main_track_coverage=0.625000
- `bp_662`: main_track_coverage=0.631579

### 10 peores bar por confidence media

- `bp_520`: confidence_mean=0.640686
- `bp_518`: confidence_mean=0.660797
- `bp_522`: confidence_mean=0.665279
- `bp_519`: confidence_mean=0.686798
- `bp_018`: confidence_mean=0.701311
- `bp_378`: confidence_mean=0.707440
- `bp_517`: confidence_mean=0.707719
- `bp_478`: confidence_mean=0.709268
- `bp_019`: confidence_mean=0.724242
- `bp_479`: confidence_mean=0.730805

### 10 peores bench por detection_ratio

- `bp_399`: detection_ratio=0.826667
- `bp_398`: detection_ratio=0.855422
- `bp_397`: detection_ratio=0.893333
- `bp_001`: detection_ratio=1.000000
- `bp_002`: detection_ratio=1.000000
- `bp_003`: detection_ratio=1.000000
- `bp_004`: detection_ratio=1.000000
- `bp_005`: detection_ratio=1.000000
- `bp_006`: detection_ratio=1.000000
- `bp_007`: detection_ratio=1.000000

### 10 peores bench por main_track_coverage

- `bp_399`: main_track_coverage=0.826667
- `bp_398`: main_track_coverage=0.855422
- `bp_397`: main_track_coverage=0.893333
- `bp_001`: main_track_coverage=1.000000
- `bp_002`: main_track_coverage=1.000000
- `bp_003`: main_track_coverage=1.000000
- `bp_004`: main_track_coverage=1.000000
- `bp_005`: main_track_coverage=1.000000
- `bp_006`: main_track_coverage=1.000000
- `bp_007`: main_track_coverage=1.000000

### 10 peores bench por confidence media

- `bp_397`: confidence_mean=0.573739
- `bp_399`: confidence_mean=0.589433
- `bp_105`: confidence_mean=0.597945
- `bp_398`: confidence_mean=0.609672
- `bp_106`: confidence_mean=0.614024
- `bp_107`: confidence_mean=0.628028
- `bp_110`: confidence_mean=0.636127
- `bp_111`: confidence_mean=0.647583
- `bp_109`: confidence_mean=0.703454
- `bp_108`: confidence_mean=0.736557

### 10 peores person por detection_ratio

- `bp_001`: detection_ratio=1.000000
- `bp_002`: detection_ratio=1.000000
- `bp_003`: detection_ratio=1.000000
- `bp_004`: detection_ratio=1.000000
- `bp_005`: detection_ratio=1.000000
- `bp_006`: detection_ratio=1.000000
- `bp_007`: detection_ratio=1.000000
- `bp_008`: detection_ratio=1.000000
- `bp_009`: detection_ratio=1.000000
- `bp_010`: detection_ratio=1.000000

### 10 peores person por main_track_coverage

- `bp_001`: main_track_coverage=1.000000
- `bp_002`: main_track_coverage=1.000000
- `bp_003`: main_track_coverage=1.000000
- `bp_004`: main_track_coverage=1.000000
- `bp_005`: main_track_coverage=1.000000
- `bp_006`: main_track_coverage=1.000000
- `bp_007`: main_track_coverage=1.000000
- `bp_008`: main_track_coverage=1.000000
- `bp_009`: main_track_coverage=1.000000
- `bp_010`: main_track_coverage=1.000000

### 10 peores person por confidence media

- `bp_460`: confidence_mean=0.912842
- `bp_486`: confidence_mean=0.915301
- `bp_487`: confidence_mean=0.916018
- `bp_488`: confidence_mean=0.916701
- `bp_485`: confidence_mean=0.916917
- `bp_461`: confidence_mean=0.919514
- `bp_454`: confidence_mean=0.922832
- `bp_462`: confidence_mean=0.923790
- `bp_457`: confidence_mean=0.924331
- `bp_455`: confidence_mean=0.924605

## 5. Tracking, stitching y multiples instancias

- bar: 169 videos con multiples instancias; 1167 frames multiples; 4 maximo de instancias.
- bench: 427 videos con multiples instancias; 37946 frames multiples; 3 maximo de instancias.
- person: 0 videos con multiples instancias; 0 frames multiples; 1 maximo de instancias.
- Stitches bench aceptados: 0.
- Frames fallback de identidad bench: 101.
- Handoffs bar aceptados: 1.

## 6. Interacciones globales

`bar_person` representa barra frente a persona completa. No se infiere contacto barra-pecho.

### Variables continuas y proximity_candidate

| Grupo | N | Media | Mediana | Std | Min | P1 | P5 | P25 | P50 | P75 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bar_person / minimum_mask_distance_px | 56034 | 1.526220 | 1.000000 | 6.649016 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 2.000000 | 3.605551 | 5.000000 | 210.000000 |
| bar_person / minimum_mask_distance_normalized | 56034 | 0.000693 | 0.000454 | 0.003018 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000454 | 0.000908 | 0.001637 | 0.002270 | 0.095329 |
| bar_person / intersection_area_px | 56034 | 8.719795 | 0.000000 | 58.392560 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 3.000000 | 40.000000 | 138.000000 | 4181.000000 |
| bar_person / intersection_area_normalized | 56034 | 0.000004 | 0.000000 | 0.000028 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000001 | 0.000019 | 0.000067 | 0.002016 |
| bar_person / iou | 56034 | 0.000036 | 0.000000 | 0.000270 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000013 | 0.000161 | 0.000527 | 0.020098 |
| bar_person / a_contained_in_b_ratio | 56034 | 0.000405 | 0.000000 | 0.002373 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000159 | 0.001812 | 0.006113 | 0.143185 |
| bar_person / b_contained_in_a_ratio | 56034 | 0.000040 | 0.000000 | 0.000307 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000014 | 0.000178 | 0.000576 | 0.022846 |
| bar_person / proximity_candidate | 56034 | 0.996716 | 1.000000 | 0.057210 | 0.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| person_bench / minimum_mask_distance_px | 57880 | 0.833009 | 1.000000 | 0.739413 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 1.000000 | 2.000000 | 2.828427 | 16.000000 |
| person_bench / minimum_mask_distance_normalized | 57880 | 0.000378 | 0.000454 | 0.000336 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000454 | 0.000454 | 0.000908 | 0.001284 | 0.007263 |
| person_bench / intersection_area_px | 57880 | 34.254302 | 0.000000 | 174.986211 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 2.000000 | 124.050000 | 886.210000 | 5570.000000 |
| person_bench / intersection_area_normalized | 57880 | 0.000017 | 0.000000 | 0.000084 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000001 | 0.000060 | 0.000427 | 0.002686 |
| person_bench / iou | 57880 | 0.000157 | 0.000000 | 0.000839 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000010 | 0.000615 | 0.003222 | 0.028321 |
| person_bench / a_contained_in_b_ratio | 57880 | 0.000196 | 0.000000 | 0.001050 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000012 | 0.000750 | 0.004054 | 0.033978 |
| person_bench / b_contained_in_a_ratio | 57880 | 0.000812 | 0.000000 | 0.004135 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000049 | 0.003585 | 0.019498 | 0.145381 |
| person_bench / proximity_candidate | 57880 | 1.000000 | 1.000000 | 0.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| bar_bench / minimum_mask_distance_px | 56001 | 387.059178 | 378.639679 | 114.187782 | 192.023438 | 207.241409 | 227.000000 | 297.042084 | 378.639679 | 457.132355 | 592.041382 | 721.000000 | 867.000000 |
| bar_bench / minimum_mask_distance_normalized | 56001 | 0.175704 | 0.171882 | 0.051835 | 0.087168 | 0.094076 | 0.103046 | 0.134841 | 0.171882 | 0.207513 | 0.268755 | 0.327295 | 0.393571 |
| bar_bench / intersection_area_px | 56001 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| bar_bench / intersection_area_normalized | 56001 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| bar_bench / iou | 56001 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| bar_bench / a_contained_in_b_ratio | 56001 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| bar_bench / b_contained_in_a_ratio | 56001 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| bar_bench / proximity_candidate | 56001 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## 7. Almacenamiento

- RAW fuente: **623946188 bytes (0.581 GiB)**.
- FEATURES fuente: **63016842 bytes (0.059 GiB)**.
- RAW final V2: **627100204 bytes (0.584 GiB)**.
- FEATURES final V2: **63495423 bytes (0.059 GiB)**.
- Manifests/audits en directorio manifests: **1290427 bytes**.
- Total final V2 + manifests: **691886054 bytes (0.644 GiB)**.
- Bytes/frame final V2: **11924.71**.
- Bytes/instancia RAW final: **2945.10**.
- Ratio packbits/full masks: **0.012184**.
- Diferencia frente a estimacion Smoke V1 (~0.628 GiB, RAW+FEATURES fuente): **+0.012 GiB**.

## 8. Previews QA

Se seleccionaron automaticamente 33 videos sin usar labels.
- `bp_001`: low_person_detection_ratio, low_person_main_track_coverage.
- `bp_002`: low_person_detection_ratio, low_person_main_track_coverage.
- `bp_003`: low_person_detection_ratio, low_person_main_track_coverage.
- `bp_004`: low_bar_main_track_coverage.
- `bp_008`: extreme_distance_bar_person.
- `bp_105`: high_stitching, low_bench_confidence_mean.
- `bp_106`: high_stitching.
- `bp_108`: high_stitching.
- `bp_109`: high_stitching.
- `bp_110`: high_stitching.
- `bp_139`: long_gap_bar, low_bar_detection_ratio.
- `bp_198`: low_bar_main_track_coverage.
- `bp_235`: extreme_distance_person_bench.
- `bp_236`: extreme_distance_person_bench.
- `bp_237`: extreme_distance_bar_bench.
- `bp_238`: extreme_distance_bar_bench, extreme_distance_person_bench.
- `bp_241`: extreme_distance_bar_bench, long_gap_bar.
- `bp_358`: multiple_bench.
- `bp_364`: multiple_bench.
- `bp_397`: long_gap_bench, low_bench_confidence_mean, low_bench_detection_ratio, low_bench_main_track_coverage.
- `bp_398`: long_gap_bench, low_bench_detection_ratio, low_bench_main_track_coverage.
- `bp_399`: long_gap_bench, low_bench_confidence_mean, low_bench_detection_ratio, low_bench_main_track_coverage.
- `bp_460`: low_person_confidence_mean.
- `bp_467`: long_gap_bar, low_bar_detection_ratio.
- `bp_469`: low_bar_detection_ratio.
- `bp_486`: low_person_confidence_mean.
- `bp_487`: low_person_confidence_mean.
- `bp_518`: low_bar_confidence_mean, multiple_bar.
- `bp_520`: low_bar_confidence_mean, multiple_bar.
- `bp_522`: low_bar_confidence_mean, multiple_bar.
- `bp_535`: extreme_distance_bar_person.
- `bp_537`: extreme_distance_bar_person, multiple_bench.
- `bp_544`: low_bar_main_track_coverage.

## 9. Errores

- `bp_349`: `ValueError`, poligono YOLO degenerado de 2 puntos en frame 121 (base cero), clase `person`, instancia local 0 / global 470, confidence 0.990939. La validacion rechazo la pareja completa y la escritura atomica no publico RAW ni FEATURES parciales.

## 10. Confirmaciones metodologicas

- No se utilizaron labels, resultados ni codigos de falta.
- No se entreno ninguna CNN ni otra red.
- No hubo interpolacion ni normalizacion temporal a 170 frames.
- No hubo split, augmentation ni StandardScaler.
- T original se preservo y se valido para cada video completo.
- Todas las instancias RAW se conservaron; el track principal sigue siendo auditable.
- `proximity_candidate` no se interpreta como contacto fisico ni ground truth.
- El pipeline se detiene aqui; no se inicia ninguna etapa de entrenamiento o fusion.
