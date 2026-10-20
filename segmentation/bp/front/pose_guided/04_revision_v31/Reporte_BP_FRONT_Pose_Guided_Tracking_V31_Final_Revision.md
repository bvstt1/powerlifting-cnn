# BP_FRONT — Última revisión controlada Pose-Guided Tracking V3.1

**Decisión:** `REVISION_PROMISING_READY_FOR_INDEPENDENT_VALIDATION`. Prototipo **offline/sombra**, no tracker aprobado, sin datos para CNN ni prueba confirmatoria. Especificación congelada `735befd58856d5d4156bf0ba4b169b2151aeec73e169d46553d67860e8c61855` inalterada. Configuración de revisión única `experimental_revision_config.json`; diff textual `revision_code_diff.patch`. No se cambiaron umbrales V3 ni V3.1 originales. Los nuevos márgenes de dominancia (0.10× separación de muñecas por muñeca en distancia **y** extensión; máscara ganadora ≤0.15× escala; 3 frames consecutivos) son **experimentales** y provienen de diagnóstico de desarrollo, no de calibración independiente.

## 1. Causa exacta diagnosticada ANTES de modificar
`v31_identity.py` clasifica dos máscaras como `SUPPORTED_HELD_BAR` basándose en `q` firmado y eje; en el segundo pase (líneas 67–71 del original), si hay más de una, **revierte ambas a `INDETERMINATE` sin comparar distancias ni extensión real**. En `bp_518` los **67** frames omitidos respecto a V3 se deben precisamente a `multiple supported BAR candidates without calibrated main-bar margin` (65 con dos BAR, 2 con tres); en `bp_520`, los **88** omitidos tienen el mismo motivo (88 con dos BAR). No eran pérdidas causadas por handoffs (0) ni división interna (0). En [518 f0](diagnostic_previews/bp_518/frame_00000.jpg), la candidata de V3 mide ~52/69 px a muñecas frente a ~1051/299 px de la rival; en [520 f0](diagnostic_previews/bp_520/frame_00000.jpg), ~53/71 frente a ~218/975 px. La rival mantiene `q`/eje aparentes porque es **fragmento lateral de rack/barra**, pero su extensión hasta ambas manos es ~200–1000 px. Esto no equivale a afirmar que todo frame omitido esté visualmente anotado como correcto.

`diagnosis_omitted_candidates.csv` detalla **todas** las candidatas de cada frame omitido (confidence BAR/ambas muñecas, máscara, eje, extensión, q, anchura, soporte temporal, razón exacta). `diagnosis_39_correct_boundary_abstentions.csv` documenta individualmente los 39 bordes correctos previos sin continuidad: 38 edges Pose compatibles y uno insuficiente; predominan entrada/salida de competidores (20 con 1→2 BAR, 10 con 2→1), pero también hay 4 con BAR única a cada lado. Los datos no prueban por sí solos identidad física.

## 2. Corrección localizada y decisión anatómica
`v31_revision_identity.py` recupera **únicamente el estado previo al segundo pase** de candidatas anuladas por competencia; permite una ganadora si ya tenía soporte bilateral por `q`, eje y movimiento temporal **y** domina a CADA competidora en distancia a máscara **y** extensión a ambas muñecas por un margen de escala, persiste ≥3 frames consecutivos en el mismo V2/extractor, con ambas muñecas fiables. Ni V2 main, ni V3, ni confidence mayor son entrada del ranking. Si una competidora está cerca de una mano, la máscara es parcial, el intervalo es corto, falta Pose o los márgenes son similares: ambas continúan indeterminadas. No se condena al perdedor como objeto falso solo por perder la comparación.

`v31_revision_handoff.py` permite corroborar **un extremo** de edge V3 ya seleccionado y `POSE_COMPATIBLE` solo si su propia anatomía individual (`q`/eje) es plausible, hay ≥3 observaciones consecutivas compatibles del **mismo subtracklet**, no hay competidor ambiguo en el frame, la relación q concuerda y no hay split híbrido. No convierte automáticamente `POSE_COMPATIBLE` en frame seguro. Rechazos V3 siguen rechazados; V3.1 original (y sus scores recalculados tras split) quedan intactos. `02_run_revision.py` hace replay de métricas BAR/pose del CSV original, cuyas fuentes RAW/MP4 se verificaron por hash. No reejecuta YOLO, no usa reviews visuales como input, y conserva índices BAR originales. La velocidad solo es válida en frames consecutivos del mismo subtracklet o a través de un handoff V3 seleccionado y Pose-compatible; gaps y cambios sin enlace devuelven `NaN` + máscara de disponibilidad=0.

## 3. Evidencia visual acotada
Se generaron 53 previews diagnósticas antes del cambio en inicio, medio, final, rachas de abstención, competencia y aparición/desaparición; se revisaron explícitamente frames individuales de 518/520 y controles/cambios muestreados en `bp_front_v31_revision_visual_review.csv`. [518 f0/f30/f69](diagnostic_previews/bp_518/), [520 f0/f78/f80](diagnostic_previews/bp_520/) y [comparativas revisadas](previews/) muestran barra seleccionada sobre ambas manos y fragmento lateral corto. Se inspeccionaron también `bp_517` f0/f9, `bp_585` f71, `bp_311` f150, `bp_102` f17 y controles. **Solo los frames anotados** tienen revisión visual explícita; ninguno de los otros frames recuperados se convierte automáticamente en ground truth. Máscaras parciales de `bp_544`/`bp_238` siguen ambiguas cuando las geometrías compiten.

## 4. Resultados por video (V3 / V3.1 original / revisión, frames)
| Video | V3 | Original confiables | Revisión confiables | Revisión indeterminados | Splits V2 |
|---|---:|---:|---:|---:|---:|
| bp_518 | 161 | 94 | 160 | 1 | 0 |
| bp_520 | 157 | 69 | 156 | 1 | 0 |
| bp_663 | 64 | 46 | 46 | 15 | 2 |
| bp_662 | 48 | 67 | 67 | 9 | 2 |
| bp_182 | 86 | 80 | 80 | 6 | 0 |
| bp_238 | 99 | 99 | 99 | 5 | 0 |
| bp_538 | 139 | 138 | 138 | 1 | 0 |
| bp_540 | 139 | 138 | 138 | 1 | 0 |
| bp_544 | 202 | 201 | 201 | 1 | 0 |

`bp_518`: **94→160/161** (+66), queda uno no confiable. `bp_520`: **69→156/157** (+87), queda uno no confiable. En conjunto +153/155 omitidos; **son casos Prototype14 únicos, NO Validation76**. La cifra de 157 frames V3 atribuida a `bp_518` en el encargo no coincide con sus artefactos: V3 allí selecciona **161**; 157 corresponde a `bp_520`. Su incremento no puede atribuirse a Validation76. Ninguna selección recuperada eligió instancia distinta de V3; esto facilita comparación, **no demuestra que V3 tuviera siempre identidad correcta**.

## 5. Development76, 77 correctos y 2 falsos merges
| Medida | V3 | V3.1 original | Revisión |
|---|---:|---:|---:|
| Frames seleccionados / confiables (76 videos) | 7102 | 6940 | 6989 |
| Déficit de frames confiables respecto V3 | — | 162 | 113 |
| Handoffs visualmente correctos con **ambos extremos confiables** | — | 38/77 | 42/77 |
| Handoffs correctos Pose incompatibles | — | 0/77 | 0/77 |
| Falsos merges visualmente confirmados `bp_663` Pose incompatibles | 0/2 detectados por V3 | 2/2 | 2/2 |

El verificador Pose conserva la clasificación 76 compatibles + 1 indeterminado de los 77 correctos; la **continuidad efectivamente publicada** mejora solo **4/77 bordes**, de 38 a 42, y **35/77 siguen con abstención**. `bp_238` conserva gaps e indeterminación, `bp_182` los 3 edges compatibles; `bp_538/540/544` no sufren rechazos por recorte/longitud. `bp_662` conserva splits @21/@62 y propuestas V3 rechazadas; `bp_663` conserva splits @17/@49, veta 0→4 @11 y 4→5 @51 y no publica la única BAR inferior falsa. No se recuperó ningún frame previamente marcado `OBSERVED_INCOMPATIBLE` en esos videos.

## 6. Resultados agregados y regresiones
| Video con recuperación | Original | Revisado | Recuperados |
|---|---:|---:|---:|
| bp_102 | 62 | 63 | 1 |
| bp_198 | 52 | 53 | 1 |
| bp_222 | 73 | 74 | 1 |
| bp_224 | 66 | 67 | 1 |
| bp_226 | 80 | 81 | 1 |
| bp_277 | 77 | 81 | 4 |
| bp_311 | 151 | 158 | 7 |
| bp_478 | 38 | 42 | 4 |
| bp_517 | 73 | 91 | 18 |
| bp_518 | 94 | 160 | 66 |
| bp_520 | 69 | 156 | 87 |
| bp_571 | 140 | 143 | 3 |
| bp_585 | 66 | 75 | 9 |

La unión 85 development pasa de 7592 a 7795 frames confiables (+203) sobre 7925 seleccionados V3; Validation76 (76 videos, **sin** 518/520) pasa de 6940 a 6989 (+49, déficit aún 113 contra V3). No se retiró ningún frame que V3.1 original publicaba; no se publicó una instancia que ese mismo frame careciera de BAR RAW. Los 17 edges V3 no seleccionados siguen sin referencia visual negativa por candidato. No hay etiquetas visuales exhaustivas por frame, por lo que **cobertura recuperada no es precisión de identidad**. Persisten abstenciones de bordes correctos y la limitación previa: ruta por frames/subtracklets, no optimización global completamente independiente de ancla V2.

## 7. Pruebas, integridad y límites
Las 11 pruebas sintéticas originales + 9 pruebas de revisión pasan: ambigüedad real, fragmento lateral distante, rival próxima a una sola mano, parcial, gap y persistencia, contexto correcto/falso/insuficiente y veto ante track híbrido. `bp_front_v31_revision_verification.json` comprueba índices BAR por frame, `NaN`/máscaras de disponibilidad, velocidades, handoffs revisados y datos visuales sólo tras ejecutar el algoritmo. SHA-256 PRE/POST: 122 ficheros del prototipo original, 2046 fuentes protegidas (RAW, modelos, V2/V3, checkpoints/splits, labels y especificación) y 663 MP4: **cero cambios de bytes**. **Excepción de ruta detectada durante el trabajo:** `extract_keypoints/bp/extract_keypoints_front_bp.py` fue renombrado externamente a `extract_keypoints/bp/individual/extract_keypoints_front_bp.py`; SHA-256 idéntico, ruta NO intacta y manifest de especificación aún apunta a la ruta antigua. No se revirtió trabajo ajeno; verificar la nueva ruta documental antes de reproducir en otro entorno. Versiones y hashes de código/config/resultados en `bp_front_v31_revision_manifest.json`.

**Límites:** margen nuevo ajustado conceptualmente tras inspeccionar 518/520 en desarrollo, sin referencia visual global. No tratarlo como sensibilidad/especificidad confirmatoria. Persisten 35/77 handoffs correctos con pérdida de continuidad y algún frame `VISUALLY_INDETERMINATE` podría aparecer en una anotación exhaustiva futura. La procedencia Pose histórica carece de timestamps/hash MP4 original; solo secuencia y muestras visuales respaldan sincronía. El pool de 559 reservados no acredita independencia de evento físico; **no se abrió ni se procesó**.

## 8. Decisión de parada
`REVISION_PROMISING_READY_FOR_INDEPENDENT_VALIDATION` **solo como candidata a preparación de validación independiente, NO tracking final**: recupera contrastes anatómicos artificialmente ambiguos y mejora modestamente bordes sin perder los dos vetos confirmados ni aceptar un nuevo falso merge observado en las muestras revisadas. Congelar hash del código/config presente; antes de ejecutar confirmación, resolver identidad física/sesión y generar anotaciones visuales ciegas por frame/segmento/edge. Si no puede acreditarse la independencia o aparecen identidades erróneas, **detener optimización del tracking** y continuar tesis con señales RAW + disponibilidad explícita. No continuar automáticamente con V3.2/V4 ni entrenar CNN con esta salida.
