# Informe ejecutivo: diseño BP_FRONT Pose-Guided Tracking V3.1

**Decisión:** `SPECIFICATION_FROZEN_READY_FOR_PROTOTYPE` — autoriza **solo diseñar posteriormente un prototipo en sombra y con salidas aisladas**, con la especificación y hashes de este directorio. **No** autoriza modificar V3/V2, publicar una trayectoria V3.1 para la CNN ni afirmar validación. No se ha implementado ni probado V3.1 en esta tarea. Los márgenes Pose quedan `EXPLORATORY/PENDING_CALIBRATION`, no congelados como umbrales productivos.

## Decisiones principales

1. **Tres niveles separados:** identidad de cada máscara/frame, contaminación dentro de un mismo `v2_track_id` y verificación de asociaciones entre subtracklets. Tracks V2 de `bp_662`/`bp_663` incluyen tramos mixtos: ni elegir todo el track principal ni eliminarlo en bloque preserva identidad.
2. **Ruta sin ancla infalible:** preflight de sincronización → geometría/soporte anatómico por *todas* las máscaras BAR → detección de cambios dentro de track → semillas de barra con soporte temporal → propuestas geométricas V3 congeladas (scores recalculados en copia solo cuando cambia segmentación lógica) → veto/compatibilidad/abstención Pose de handoffs → ruta global coherente y salida por frame con calidad. El baseline V3 original sigue disponible, incluidas sus rutas `LOCKED_V2_COMPLETE_RAW_COVERAGE` y `KEEP_V2`; ninguna de ellas es ground truth de barra sostenida. No usar una selección V2 como referencia para «demostrar» su propia corrección.
3. **BAR falsa única:** `OBSERVED_INCOMPATIBLE`, índice confiable `-1`; BAR única dudosa: `OBSERVED_UNVERIFIED`, índice confiable `-1`; sin BAR: `NO_BAR_DETECTED`. Nunca inventar máscara para cubrir un frame, ni pasar salto inferior falso como descenso real a CNN. El índice legacy V3 se preserva aparte para comparación y revisión.
4. **Tres decisiones por handoff:** `POSE_COMPATIBLE`, `POSE_INCOMPATIBLE`, `POSE_INSUFFICIENT_EVIDENCE`. Un edge rechazado por V3 no se convierte en aceptado porque Pose «parece cercano». Compatible no basta si el track está híbrido. Insuficiente no equivale a aceptar ni a rechazar: marcar riesgo, abstenerse para salida confiable en bordes peligrosos y solicitar revisión; en ausencia de riesgo conservar legacy como no confirmado, sin promoverlo a identidad segura.
5. **Precisión antes que cobertura ficticia:** los 1641/1680 frames recuperados por V3 no son objetivo rígido; la cobertura de identidad segura y los falsos merges importan más. Frames inválidos llevan máscara de disponibilidad y velocidades NaN, no ceros ni interpolación automática. El futuro entrenamiento de *PoseCNN + Segmentación* es otro experimento: la rama preparada con muñecas se documenta *Pose-Guided Segmentation*.

## Fuente y límites de la evidencia

`../pose_guided_feasibility/v1/`: 76 videos, 7141 frames con dos muñecas finitas, 77 handoffs aceptados correctos (76 Pose compatibles, 1 indeterminado en `bp_238`), 2 falsos merges de `bp_663` ambos identificados por salto vertical firmado bilateral (+118/+118.5 y −97.6/−97.0 px); 17 propuestas V3 no seleccionadas no tienen automáticamente referencia visual negativa. En `bp_662`/`bp_663` track 4 hay 41/7 y 32/8 frames arriba/abajo por **clasificación geométrica**, no por referencia visual frame a frame. Máscara parcial/completa en `bp_538/540/544` enseña por qué proximidad absoluta no basta. Es evidencia de *viabilidad exploratoria*, no validación de sensibilidad/especificidad de V3.1.

Se auditó el flujo real: `extract_keypoints/bp/extract_keypoints_front_bp.py` usa `bp_front_skeleton_v8.pt` y almacena solo `(T,6,3)` normalizado; `extract_segmentation/bp/extract_segmentation_front_bp.py` sí almacena hash de video/modelo y timestamps nominales. `extract_segmentation/bp/bar_tracking_v3_utils.py` selecciona rutas que pasan por el track principal V2 y puede bloquear V2 completo. Prototype14 está en `segmentation/bp/front/tracking_v3/test_14/`; Validation76 en `.../validation_76/`. Son **ambos** desarrollo/regresión. `BP_FRONT_Pose_Guided_Tracking_V31_Provenance_Audit.md` delimita sincronía sustentada por secuencia/overlays y falta de certificación criptográfica retrospectiva Pose.

## Riesgos y gates antes de cualquier implementación/confirmación

| Riesgo pendiente | Regla de diseño congelada | Evidencia requerida después |
|---|---|---|
| `.npy` Pose sin timestamps/hash MP4 ni modelo persistidos | `SYNC_BLOCKED` si falla alineación; no declarar trazabilidad histórica inexistente | Auditoría visual/PTS/frames más extensa; futuro formato con hash de video y frames. Reextracción solo con autorización. |
| Valores de salto, ventanas y confidence derivados de Validation76 | Tratarlos como **referencia exploratoria**, sin desplegar como regla | Calibración en desarrollo distinto, versionar tabla *efectiva* y hash de código antes de abrir confirmación. |
| Casos raros de manos ausentes, pausa y cámara diferente | Abstención, no extrapolación de calidad 100% observada | Robustez y anotaciones visuales ciegas, incluidas máscaras parciales/estructura paralela. |
| Ancla V2 híbrida y fast path | Analizar cada frame y dividir lógicamente en copias auditables | Pruebas de invariantes, trazabilidad de subtracklets; recomputación geométrica sin cambiar V3. |
| Duplicados físicos / `group_id_status=unverified_individual` | Partición por levantamiento/sesión y exclusión conservadora de incertidumbre | Inventario y adjudicación de grupos **antes** de seleccionar test independiente. |
| Confirmación con pocos falsos merges | No declarar éxito si N positivo insuficiente | Tamaño/precisión estadística preinscritos y referencia visual por frame/edge. |

## Estado de congelación

**Se congela ahora:** contratos de entrada y preflight; medidas matemáticas; tres validadores y su orden; estados/conflictos; política de BAR única/gap/abstención; no circularidad; esquema de trazabilidad propuesto; distinción desarrollo/regresión/confirmación; métricas y condiciones de bloqueo. `BP_FRONT_Pose_Guided_Tracking_V31_Parameters.csv` identifica para cada número si proviene del extractor, es solo exploratorio o sigue pendiente. El hash SHA-256 final de la especificación y las dependencias se registran en el manifest de este directorio.

**No se congela como valor válido para producción:** límites numéricos obtenidos de Validation76, identidad visual de todos los frames del track 4, tolerancia de pérdidas en un nuevo test y calibración para cámara distinta. Se congelará una versión **adicional de parámetros efectivos** antes de confirmar; si se modifica el diseño, emitir nueva versión/hash y repetir separación de conjuntos. Una especificación congelada **no** significa algoritmo entrenado, implementado ni validado.

**Recomendación:** autorizar en otra tarea únicamente un prototipo offline en sombra, aislado de tracking V3, RAW y features actuales, siempre que documente provenance y abstenciones. **No autorizar** publicación de V3.1 ni su uso para generar secuencias definitivas de CNN hasta superar el protocolo visual independiente de `BP_FRONT_Pose_Guided_Tracking_V31_Validation_Protocol.md` con parámetros y código previamente congelados. No ejecutar ahora Validation17_RAW_GAPS ni inferencia YOLO.

## Entregables en este directorio

`BP_FRONT_Pose_Guided_Tracking_V31_Specification.md` · `BP_FRONT_Pose_Guided_Tracking_V31_Parameters.csv` · `BP_FRONT_Pose_Guided_Tracking_V31_Validation_Protocol.md` · `BP_FRONT_Pose_Guided_Tracking_V31_Provenance_Audit.md` · `BP_FRONT_Pose_Guided_Tracking_V31_Spec_Manifest.json` · este reporte.
