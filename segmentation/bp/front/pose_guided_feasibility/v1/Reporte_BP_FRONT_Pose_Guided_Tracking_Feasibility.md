# BP_FRONT: viabilidad Pose-Guided Bar Tracking (exploratorio)

## 1. Objetivo e hipótesis
Investigar si las muñecas YOLO Pose RAW aportan evidencia **adicional** para distinguir la barra sostenida de estructuras paralelas detectadas como `bar`. Estado: **PROMISING, exclusivamente exploratorio**; no se implementó ni aprobó V3.1.

## 2. Estado V3 y referencias
Validation76: 72 `FULLY_RECOVERED`, 1 `PARTIALLY_RECOVERED`, 3 `UNCHANGED_CONSERVATIVE`, 0 `REGRESSED`; 1641/1680 frames recuperados. De 79 handoffs aceptados, 77 visualmente correctos y 2 falsos merges en `bp_663` (ambos `suspicious=0`); 17 propuestas no seleccionadas, **sin etiqueta visual de verdad**. Los estados visuales fueron usados solo en la evaluación posterior; nunca entran en las métricas ni en la decisión Pose.

## 3–5. Fuentes, auditoría RAW y sincronización
- Videos: `dataset/bp/front/bp_XXX.mp4` (hash idéntico al `video_sha256` de cada RAW).
- Pose RAW: `keypoints/bp/front/bp_XXX.npy`, `(T,6,3)` float, `x,y` normalizados y confidence por punto; índices `[0:left_elbow,1:left_shoulder,2:right_shoulder,3:right_elbow,4:right_wrist,5:left_wrist]`. Extractor: `extract_keypoints/bp/extract_keypoints_front_bp.py`, YOLO Pose `models/bp/bp_front_skeleton_v8.pt`; frame decodificado secuencialmente; `conf>=0.5` y NaN en coordenadas no fiables. **No** se utilizaron secuencias T=170 para PoseCNN.
- Segmentación: `segmentation/bp/front/raw/bp_XXX_segmentation.npz`, `frame_index`, `timestamp_sec`, `frame_image_shape_hw`, `frame_instance_offsets`, clases, confidence, bboxes píxel, `mask_bits`/`mask_byte_offsets`/`instance_mask_origin_xy`/`instance_mask_shape_hw`; máscaras recuperadas con `np.unpackbits(bitorder='little')`. Modelo de procedencia en metadata: `models/bp/bp_front_seg_v2.pt`.
- Tracks: `segmentation/bp/front/tracking_v2/raw/bp_XXX_segmentation.npz` (`instance_track_id` de V2, mismos índices/máscaras de instancia RAW); `segmentation/bp/front/tracking_v2/features/`; V3: `segmentation/bp/front/tracking_v3/validation_76/selections/` (índices seleccionados y track IDs), `audits/` (propuestas), `manifests/*handoffs.csv` y `*visual_review.csv`. Los IDs no son automáticamente identidades físicas.
- Comprobación automática en `bp_front_pose_guided_sync_audit.csv`: 76 videos, 7141 frames, 7141 con ambas muñecas válidas (100.00%); FPS=30, 1920×1080 en estos 76, base cero; índices consecutivos en RAW/V2/V3, shapes por frame, timestamps `i/fps`, hash del MP4, recuento exacto de frames Pose y geometría normalizada `[0,1]`; transformación Pose a píxeles `(x·W,y·H)`. Los JPG originales son BGR al leer OpenCV y se escriben conservando colores; no hay recorte/letterbox persistido en los artefactos. Superposición visual comprobada en los nueve videos prioritarios, en frames separados: manos y máscaras corresponden al video y al mismo instante observable.
- **Límite de procedencia:** Pose `.npy` no incluye timestamp, hash del video ni frame IDs internos: igualdad de T + código extractor secuencial + coherencia de movimiento y superposición visual proporcionan evidencia de sincronía, no una certificación criptográfica del procesamiento Pose original en TODOS los frames. No se corrigieron offsets ni se inventaron frames. Gaps RAW siguen vacíos. Verificación independiente de provenance Pose sigue pendiente.

## 6. Definiciones geométricas y procedimiento exploratorio
Para máscara binaria `M_t`, muñeca `w_t=(xW,yH)` y centro `c_t`:
- `d_mask(w,M)=min_{p∈M}||w-p||₂` (transformada euclídea, coordenada de muñeca redondeada al píxel; error de discretización ≤0.71 px). Referencia física `S=||w_L-w_R||₂` solo si ambas muñecas son fiables; bilateral = `max(d_L,d_R)/S`; conservar valores separados y confidence. La distancia sola falla si una máscara está parcialmente recortada o si dos barras flanquean la mano.
- PCA sobre los píxeles de la máscara: autovector principal `u`; `d_eje=|(w-c)×u|`, extensión = exceso de proyección `(w-c)·u` fuera del intervalo proyectado de píxeles de M. Colinealidad no determina identidad; máscara corta puede aumentar muchísimo `d_mask` por extensión sin cambiar `d_eje`.
- Desplazamiento **firmado** `q(w,M)=c_y-w_y` (y crece hacia abajo). En el borde de un handoff se comparan medianas de los **dos últimos frames reales** de origen y **dos primeros frames reales** de destino; contexto adicional: hasta seis frames reales por lado. `Δq_L, Δq_R` detecta salto inferior/superior aunque `d_mask` absoluto apenas cambie. En ventanas se computa mediana de desplazamiento euclídeo barra frente a promedio de muñecas, solo en pares consecutivos con ambas fiables; `PAUSE_OR_INSUFFICIENT` no es fallo de identidad.
- Para clasificar evidencia descriptiva: mínimo 3 observaciones bilaterales en cada ventana de seis y dos por lado del borde, `conf>=0.5` ya exigido por extractor. `DIFFERENT_OBJECT` si los dos `|Δq|>max(45 px, 0.08·min(S_pre,S_post))`, mismo signo y anchuras de las máscaras comparables (ratio >0.65). `SAME_BAR` si ambos `|Δq|<max(30 px,0.045·min(S))` y distancias al eje de ambas muñecas, antes/después, <0.15·min(S). En otros casos `INSUFFICIENT_EVIDENCE`. Estos **márgenes son exploratorios**, definidos después de inspeccionar el caso y controles; NO son parámetros de V3 ni una regla lista para desplegar. Medianas de seis y dos frames se publican juntas para exponer cambios de identidad DENTRO de un track.

## 7–8. Casos críticos (píxeles; distancias, ejes y confidence de ventanas de seis; Δq en borde de dos)
| Caso | score V3 | máscara L/R pre→post | eje L/R pre→post | Δq L / Δq R | conf L/R post | movimiento post | V3 seleccionó | Pose |
|---|---:|---:|---:|---:|---:|---|---:|---|
| bp_663 0→4 @11 | 0.8143 | 55.1/64.2→71.8/60.5 | 33.4/37.7→67.8/58.1 | 118.0 / 118.5 | 0.9944/0.9945 | MEASURABLE | 1 | POSE_SUPPORTS_DIFFERENT_OBJECT |
| bp_663 4→5 @51 | 0.8171 | 61.3/67.8→54.1/65.5 | 36.4/36.3→33.4/37.0 | -97.6 / -97.0 | 0.9957/0.9957 | MEASURABLE | 1 | POSE_SUPPORTS_DIFFERENT_OBJECT |
| bp_662 0→4 @19 | 0.8172 | 56.1/60.9→64.5/62.2 | 33.6/33.0→38.7/35.8 | 96.1 / 100.3 | 0.9945/0.9946 | MEASURABLE | 0 | POSE_SUPPORTS_DIFFERENT_OBJECT |
| bp_662 4→5 @66 | 0.8064 | 66.6/64.0→54.1/62.1 | 58.2/42.1→34.7/37.3 | -109.0 / -109.6 | 0.9957/0.9957 | MEASURABLE | 0 | POSE_SUPPORTS_DIFFERENT_OBJECT |

En `bp_663`, los dos saltos firmados bilaterales (0→4 inferior; 4→5 superior) superan ~97 px pese a que la **distancia sin signo puede ser muy parecida**. Confidence en ambos bordes ~0.994–0.996 (confidence alta no garantiza localización perfecta). [Antes del primer handoff](previews/bp_663/0_4_frame_00010.jpg), [después](previews/bp_663/0_4_frame_00011.jpg); [antes del segundo](previews/bp_663/4_5_frame_00050.jpg), [después](previews/bp_663/4_5_frame_00051.jpg). Estructura inferior está ~120 px por debajo de la barra superior; la barra superior está en tracks 0 y 5 y **también** en parte del track 4: el track 4 mezcla instancias a lo largo del tiempo. RAW solo aporta una máscara `bar` en cada frame de esos bordes; no se inventaron máscaras simultáneas de la barra omitida. Ventanas globales de seis en el segundo borde son engañosas porque el mismo track 4 alterna entre objetos en frames 45–50; por ello se publican ambas escalas temporales.

En `bp_662`, V3 rechazó 0→4 y 4→5 (scores ~0.817 y ~0.806), mientras V2 ancló track 4. Las [previews 0→4](previews/bp_662/0_4_frame_00019.jpg) y [4→5](previews/bp_662/4_5_frame_00066.jpg) evidencian barra superior sostenida y barra inferior; `0/5` se alinean con la superior. Track 4 contiene también muchos frames de la superior: la discrepancia es una **contaminación temporal del track V2 principal**, no prueba de que todos sus frames sean barra inferior. La selección inicial y el handoff son problemas separados. Los dos rechazados carecen de ground truth de handoff en el CSV (`NOT_APPLICABLE`); esta observación de previews no los convierte en negativos confirmados de toda la población.

## 9. Identidad de track y controles exigidos
Conteo descriptivo para bp_662/663 de signo `c_y-promedio(y_muñecas)` sobre **todos** los frames BAR disponibles (no equivale a verdad visual frame a frame):
| video | track V2 | principal V2 | frames | barra arriba | barra abajo |
|---|---:|---|---:|---:|---:|
| bp_662 | 0 | False | 19 | 19 | 0 |
| bp_662 | 4 | True | 48 | 41 | 7 |
| bp_662 | 5 | False | 11 | 11 | 0 |
| bp_663 | 0 | False | 11 | 11 | 0 |
| bp_663 | 4 | True | 40 | 32 | 8 |
| bp_663 | 5 | False | 13 | 13 | 0 |

| Control correcto | score V3 | Δq L / R (px) | Pose | revisión original |
|---|---:|---:|---|---|
| bp_182 0→4 @52 | 0.8298 | 3.4 / 3.3 | POSE_SUPPORTS_SAME_BAR | segmentation/bp/front/tracking_v3/validation_76/previews/bp_182/00_0_to_4/bp_182_0_to_4_contact_sheet.jpg |
| bp_182 4→7 @64 | 0.6968 | -5.2 / -4.4 | POSE_SUPPORTS_SAME_BAR | segmentation/bp/front/tracking_v3/validation_76/previews/bp_182/01_4_to_7/bp_182_4_to_7_contact_sheet.jpg |
| bp_182 7→8 @75 | 0.8187 | 7.2 / -13.2 | POSE_SUPPORTS_SAME_BAR | segmentation/bp/front/tracking_v3/validation_76/previews/bp_182/02_7_to_8/bp_182_7_to_8_contact_sheet.jpg |
| bp_238 0→7 @101 | 0.8532 | -4.4 / 0.3 | POSE_INSUFFICIENT_EVIDENCE | segmentation/bp/front/tracking_v3/validation_76/previews/bp_238/00_0_to_7/bp_238_0_to_7_contact_sheet.jpg |
| bp_538 0→4 @51 | 0.6956 | -3.3 / -2.0 | POSE_SUPPORTS_SAME_BAR | segmentation/bp/front/tracking_v3/validation_76/previews/bp_538/00_0_to_4/bp_538_0_to_4_contact_sheet.jpg |
| bp_540 0→4 @55 | 0.6926 | 3.2 / 3.0 | POSE_SUPPORTS_SAME_BAR | segmentation/bp/front/tracking_v3/validation_76/previews/bp_540/00_0_to_4/bp_540_0_to_4_contact_sheet.jpg |
| bp_544 0→5 @96 | 0.7105 | 1.2 / 0.7 | POSE_SUPPORTS_SAME_BAR | segmentation/bp/front/tracking_v3/validation_76/previews/bp_544/00_0_to_5/bp_544_0_to_5_contact_sheet.jpg |

`bp_182`: 3 correctos compatibles, incluidos grandes cambios de longitud. `bp_238` correcto permanece **indeterminado** (2 frames origen y 3 destino en la ventana local; cuatro frames RAW puenteados, sin reconstrucción). `bp_538/540/544`: 3 correctos compatibles aunque `d_mask` cambia ~275–335 px por máscara parcial/completa; eje y `q` permanecen estables. [Previews controles](previews/) con video, máscaras, track, selección V3 y ambas muñecas. `bp_102/103`: previews `identity_frame_*` muestran dos estructuras paralelas y V3 sin cambio de identidad confirmado: controles cualitativos, no handoffs negativos nuevos.

## 10–14. Discriminación, 79 aceptados y 17 no seleccionados
| Referencia original (solo evaluación) | Pose SAME | Pose DIFFERENT | Pose INSUFFICIENT | N |
|---|---:|---:|---:|---:|
| Aceptados visualmente correctos | 76 | 0 | 1 | 77 |
| Aceptados falsos merges | 0 | 2 | 0 | 2 |
| No seleccionados, **sin verdad visual por candidato** | 0 | 3 | 14 | 17 |

Los 17 no seleccionados NO se cuentan como negativos: por propuesta figuran así:
| Propuesta | score V3 | Pose | referencia original |
|---|---:|---|---|
| bp_009 0→4 @65 | 0.7633 | POSE_INSUFFICIENT_EVIDENCE | NOT_APPLICABLE |
| bp_126 0→6 @62 | 0.7933 | POSE_INSUFFICIENT_EVIDENCE | NOT_APPLICABLE |
| bp_182 0→5 @57 | 0.7566 | POSE_INSUFFICIENT_EVIDENCE | NOT_APPLICABLE |
| bp_182 5→6 @63 | 0.7703 | POSE_INSUFFICIENT_EVIDENCE | NOT_APPLICABLE |
| bp_182 5→7 @64 | 0.6111 | POSE_INSUFFICIENT_EVIDENCE | NOT_APPLICABLE |
| bp_182 6→7 @64 | 0.8326 | POSE_INSUFFICIENT_EVIDENCE | NOT_APPLICABLE |
| bp_186 3→5 @34 | 0.7489 | POSE_INSUFFICIENT_EVIDENCE | NOT_APPLICABLE |
| bp_223 4→5 @20 | 0.8806 | POSE_INSUFFICIENT_EVIDENCE | NOT_APPLICABLE |
| bp_223 6→7 @69 | 0.9194 | POSE_INSUFFICIENT_EVIDENCE | NOT_APPLICABLE |
| bp_224 5→6 @56 | 0.8848 | POSE_INSUFFICIENT_EVIDENCE | NOT_APPLICABLE |
| bp_226 5→6 @68 | 0.8746 | POSE_INSUFFICIENT_EVIDENCE | NOT_APPLICABLE |
| bp_227 4→5 @11 | 0.8787 | POSE_INSUFFICIENT_EVIDENCE | NOT_APPLICABLE |
| bp_227 6→7 @53 | 0.9266 | POSE_INSUFFICIENT_EVIDENCE | NOT_APPLICABLE |
| bp_238 5→6 @93 | 0.5650 | POSE_SUPPORTS_DIFFERENT_OBJECT | NOT_APPLICABLE |
| bp_541 4→5 @87 | 0.7237 | POSE_INSUFFICIENT_EVIDENCE | NOT_APPLICABLE |
| bp_662 0→4 @19 | 0.8172 | POSE_SUPPORTS_DIFFERENT_OBJECT | NOT_APPLICABLE |
| bp_662 4→5 @66 | 0.8064 | POSE_SUPPORTS_DIFFERENT_OBJECT | NOT_APPLICABLE |

De los 77 correctos, **0 aparecen incompatibles**, 76 compatibles y 1 indeterminado; ello NO implica que se recuperaría el mismo número al alterar un tracker real. Tres de 17 no seleccionados muestran evidencia de objeto diferente (dos en bp_662, uno en bp_238), catorce son indeterminados. No hay TP/FP definibles para los 17 sin revisión específica. Para los 79 aceptados, matriz 77/2 anterior usa exactamente la revisión visual original, sin crear etiquetas nuevas. Scores V3 ~0.81 en bp_663 y ~0.69–0.71 en máscaras parciales correctas: Pose firmado aporta contraste que ni score alto ni sola proximidad garantizan. La aparente separabilidad está **sesgada por exploración del mismo conjunto**.

## 15–17. Limitaciones, trade-offs y propuesta solo conceptual
Una muñeca poco fiable: no sustituir por cero ni imponer bilateralidad; registrar evidencia unilateral y estado indeterminado para la decisión actual. Ambas ausentes: indeterminado. Las 76 secuencias aquí tienen cobertura bilateral de frames excepcionalmente alta; no extrapolar a iluminación/oclusiones nuevas. Pausas: un desplazamiento cercano a cero no demuestra que la barra está equivocada. Tracklets híbridos muestran que persistencia de track ID no garantiza identidad interna. Cuando falta máscara, no estimar su ubicación. Los proxies PCA y distancias a eje son sensibles a oclusión / barras paralelas; el signo vertical puede fallar en cámaras invertidas o perspectiva distinta. El gran cambio de longitud de un mismo objeto necesita máscara parcial explícita: distancia a máscara sola produciría falsos rechazos.

**Concepto eventual Pose-Guided Tracking V3.1 (NO implementado):** un verificador externo entre propuestas V3 y publicación; entradas: Pose RAW por frame + confidence, máscaras BAR V2 RAW + índices V3 y metadata fps/resolución/hash de video. Preflight de secuencia y overlays, sin remuestreo; comparar candidatos de barra inicial por compatibilidad bilateral *y persistencia* antes de anclar (componente de selección independiente), y verificar cada borde sobre frames reales con distancias a máscara, PCA, signo vertical, movimiento medible y cambios intra-track (componente handoff). Ante datos ausentes o ambiguos mantener salida V3 sin promover asociaciones Pose nuevas y generar marca para revisión. Parámetros eventuales: mínimos de confidence/cobertura/ventana, márgenes de salto normalizados, tolerancia a recorte y calibración de orientación por cámara. Ningún número de este experimento debe desplegarse sin contrastación separada y ablation con keypoints de baja confianza, pausas, máscaras parciales, estructuras paralelas y videos nuevos no usados para ajustar la propuesta.

## 18. Integridad y 19. Conclusión
`bp_front_pose_guided_experiment_manifest.json` contiene SHA-256 **antes y después** de 705 fuentes: MP4, Pose RAW, segmentación RAW, tracks/features V2, selección/audits V3, modelos BP, CSV de etiquetas y código fuente. Coinciden todos; `03_verify_results.py` falla ante cualquier cambio, ausencia de preview o alteración del número/referencia de handoffs. Solo se ejecutaron los tres scripts offline de este directorio; sin YOLO, training, cambios de V3 ni generación global de features. El manifiesto no es una prueba de procesos externos al experimento.

**Conclusión:** las muñecas son prometedoras como evidencia *complementaria* mediante desplazamiento bilateral vertical firmado, no mediante distancia simple. Los dos falsos merges conocidos quedan identificados y ninguno de los 77 correctos se marca distinto en esta exploración; uno queda indeterminado. `bp_662` revela un track principal V2 parcialmente contaminado y la necesidad de evaluar separadamente selección de identidad. **No aprobar V3.1** aún: Validation76 ya se usó para inspeccionar/escoger esta combinación de características. Próximo experimento: congelar especificación, establecer procedencia verificable de Pose y evaluar sobre videos independientes (sin tocar ahora Validation17_RAW_GAPS). Si se utiliza Pose para seleccionar/tracking, llamar a esa rama *Pose-Guided Segmentation*, no *Segmentation-only*. *Pose + Segmentation* designa un clasificador multimodal posterior; resultados válido/nulo no intervinieron aquí.
