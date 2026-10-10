# Protocolo preinscrito para validar BP_FRONT Pose-Guided Tracking V3.1

**Estado:** protocolo de diseño congelado; **ningún video confirmatorio fue seleccionado, anotado ni procesado en esta tarea**. No implica existencia, entrenamiento ni validación de V3.1. Versionar con el hash de la especificación y, *después* de calibrar en desarrollo, publicar manifest separado de parámetros efectivos **antes** de abrir etiquetas confirmatorias.

## 1. Registro de poblaciones y exclusiones

1. `segmentation/bp/front/tracking_v3/test_14` (Prototype14), `tracking_v3/validation_76` y `pose_guided_feasibility/v1` son **desarrollo/regresión únicamente**: se usaron para hallar señal y márgenes. No se mezclan con confirmación, aunque un video aparezca en ambos primeros conjuntos. Congelar lista de `video_id`, hash MP4 y eventos/grupos físicos; su unión se excluye de confirmación.
2. Inventariar *sin ejecutar tracking* un pool nuevo con videos BP_FRONT no inspeccionados para diseño; separar desarrollo adicional si hace falta calibrar parámetros y confirmación reservada antes de revisar cualquier resultado V3.1 de ésta. No procesar hoy los 662 videos; su existencia no justifica afirmar que son independientes. Registrar fechas/sesiones/cámaras/atleta y posibles duplicados; usar hash de archivo + inspección de near-duplicates y metadatos/visual de levantamientos repetidos. `group_id_status=unverified_individual` en metadatos RAW: **no demuestra independencia física**. Si no se puede resolver duplicación de evento, excluir o agrupar conservadoramente de la evaluación primaria. Dividir *por evento físico y sesión*, no por frame, ni por vista aleatoria, ni por track.
3. Subconjunto 17 videos con RAW gaps: prueba posterior **específica** de robustez sin inferencia YOLO ni relleno; verificar sus IDs/solapamientos con desarrollo y confirmación. Si ya fueron inspeccionados para diseño o hay solapamiento, **no** son confirmación independiente. No ejecutarlo ahora.
4. Registrar muestreo y prevalencia (no prometer sensibilidad de falso merge con solo 2 casos): casos V3 correctos, fragmentaciones, falsos merges, anclas híbridas, barras paralelas, anchuras parcial/completa, pausas y movimiento rápido, muñecas parcialmente ausentes, cambios de iluminación/cámara/resolución, detecciones BAR ausentes y BAR falsa única. Muestreo de errores puede enriquecer falsos merges, pero reportar separado de rendimiento en una muestra consecutiva representativa. Dimensionar N y precisión del intervalo de confianza **con prevalencia estimada en desarrollo y criterio de coste acordado antes de abrir confirmación**. Si no hay suficientes positivos visualmente demostrados, emitir `INCONCLUSIVE`, nunca «0 errores implica éxito».

## 2. Referencia visual independiente

- Unidad de anotación por video: frame `(video_id, frame_index)`, máscara RAW `global_instance_index`, identidad física persistente de **barra sostenida**, `NO_BAR_VISIBLE`, `HELD_BAR_VISIBLE_NOT_DETECTED`, `AMBIGUOUS_VISUAL`, etiqueta de máscara (`HELD_BAR`, `OTHER_OBJECT`, `UNCERTAIN`); intervalos de identidad del mismo `v2_track_id` y transiciones `SAME_PHYSICAL_OBJECT`, `DIFFERENT_OBJECT`, `INDETERMINATE`. Anotar si máscara parcialmente visible, punto de solape, ausencia de BAR RAW y nivel de certeza. *BAR visible sin máscara* difiere de barra verdaderamente fuera de encuadre.
- Dos revisores independientes y ciegos a salida V3.1, scores Pose y validez/nulidad, usando MP4 con acceso a máscaras RAW/IDs neutrales; anotar frames de cambio y bordes además de intervalos regulares y controles negativos. Adjudicar discrepancias por tercer revisor sin conocer la salida evaluada; guardar votos, evidencia y procedencia. `AMBIGUOUS_VISUAL` se excluye de denominadores de verdad correspondientes y se informa su cantidad. Mapeo de IDs físicos entre frames no puede inferirse del ID V2. Capturar explícitamente switches internos del track y potenciales falsos merges **no marcados suspicious por V3**.
- La referencia original `validation_76/*handoffs.csv` etiqueta 77 correctos/2 incorrectos aceptados; los 17 no seleccionados figuran `NOT_APPLICABLE`, **nunca negativos automáticos**. En una prueba de regresión visual nueva, anotar individualmente un rechazo para poder contarlo. No trasplantar etiquetas de validez/nulidad de levantamiento.
- Publicar junto con evaluación: lista y hashes de MP4/RAW/Pose/reference, mapping físico y discordancias/resolución, timestamps de adjudicación y hash de configuración. Prohibir que la referencia entre en algoritmo, selección inicial, umbrales o selección de la ruta.

## 3. Fases que no deben mezclarse

| Fase | Conjunto | Decisiones autorizadas | Prohibiciones |
|---|---|---|---|
| Desarrollo y calibración | Prototype14/Validation76 y, si es necesario, **nuevo desarrollo separado** | Diseñar estados, analizar fallos, fijar márgenes/calidad/ventanas, prueba de unidad con sintéticos y negativos visuales realmente anotados | Llamarlo confirmación; decidir con labels de clasificación. |
| Congelación pre-confirmatoria | Antes de abrir nuevo test | Fijar código V3.1, hash de esta especificación, tabla de parámetros efectiva con valores y unidades, versión de modelo, protocolo, tests e IDs físicos de las particiones | Modificar esos valores al ver resultados de test. |
| Confirmación | Videos/eventos nunca usados en diseño o calibración | Ejecutar una vez, comparar baselines con verdad visual ciega; reportar todos los casos | Ajustar thresholds, excluir errores post hoc, redefinir qué es falso merge. |
| Regresión | Prototype14 + Validation76 | Detectar rotura de casos conocidos, coherencia y reproducción del baseline V3 | Presentar éxitos como generalización. |
| Robustez | 17 RAW gaps después y estratos reservados si son independientes | Caracterizar gaps, oclusión, máscaras únicas y confianza Pose | Reetiquetar RAW o tratarlos automáticamente como nueva confirmación. |

Los splits Train/Validation/Test del clasificador final permanecen inmutables. Si existe solapamiento de evento físico entre datos usados para elegir tracking y futuros tests del clasificador, informarlo y planear evaluación de clasificación independiente; no usar rendimiento de CNN para elegir versión del tracker.

## 4. Baselines, ablations y pruebas necesarias

- Reejecutar **en copia aislada** la misma V3 congelada sin cambiar `bar_tracking_v3_utils.py` ni artefactos de Validation76; compararla en *las mismas detecciones RAW* con V3.1. Reportar también V2 main. No imputar BAR inexistente ni añadir YOLO para V3.1. Controlar comportamiento `LOCKED_V2_COMPLETE_RAW_COVERAGE`, `KEEP_V2` y `PUBLISH_GLOBAL_PATH`.
- Ablation por componentes: a) V3 sin Pose; b) identidad por frame/selección inicial sin handoff; c) detección intratrack sin selección inicial; d) verificación de handoff sin división; e) sistema completo. Comparar con máscaras parcial/completa y estructuras paralelas; no atribuir beneficio de selección inicial a verificación de handoff. Un tracker «simulado» con ground truth inyectado NO es evaluador justo.
- Casos de contrato: `bp_663` 0→4 y 4→5 (veto), `bp_662` ancla híbrida y propuestas rechazadas (sin ground truth por propuesta), `bp_182` tres handoffs correctos, `bp_238` gap correcto indeterminado, `bp_538/540/544` longitud parcial/completa, `bp_102/103` estructuras paralelas, Prototype14 `bp_198/301/467` con gaps sin inventar, `bp_469` ambigüedad, `bp_520` ID virtual. Resultados esperados de estas pruebas **son regresión**, no predicción confirmatoria.
- Robustez no entrenada: remover una/ambas muñecas, degradar confidence, drop/crop artificial únicamente de **copia para pruebas unitarias**, inversión/rotación sin transformación declarada -> abstención; pausa real frente a BAR estática espuria; pérdida intermitente de detección; correlación errónea por movimiento conjunto de cámara. Siempre informar intervención/sintético separado del desempeño natural.

## 5. Métricas, denominadores y errores

Con referencia visual no ambigua:

| Métrica | Definición y denominador |
|---|---|
| Falso merge | edge seleccionado que une identidades físicas diferentes / edges seleccionados con verdad. Informar número por evento y por video; no promediar frames del mismo evento como casos independientes. |
| Handoff correcto conservado | correctos de V3 que V3.1 conserva como misma identidad / correctos de V3 con verdad. |
| Handoff incorrecto rechazado | incorrectos V3 marcados incompatibles o cuyo edge no se publica / incorrectos V3 con verdad; separar veto de abstención. |
| Rechazo incorrecto | edge correcto V3 vetado / edges correctos V3 con verdad; intervalos de confianza. |
| Indeterminados | abstenciones por fuente de incertidumbre / propuestas o frames elegibles; separar ausencia Pose de falta de segmentación y de ground truth ambiguo. |
| Identidad frame | `TP`: held-bar instance correcta seleccionada; `FP`: selección de otra estructura; `FN`: barra held-bar detectada RAW pero omitida; `UNOBSERVABLE`: held-bar visible sin máscara; siempre reportar por frame **y por evento**. Precision = TP/(TP+FP); recall condicionado a BAR RAW disponible = TP/(TP+FN), reportar cobertura bruta aparte. |
| Cobertura fiable | frames con identidad correcta y `OBSERVED_CONFIDENT` / frames con barra held-bar visible; además cobertura condicionada a existencia de máscara held-bar RAW. No penalizar doblemente falta RAW como error de tracking. |
| Frames incorrectos | seleccionados con identidad distinta / frames con identidad anotada; contar secuencias/longitud de saltos ficticios. |
| Contaminación intratrack | detección de tracks con switches visuales (precision/recall de cambios y latencia de localización); sobresegmentación falsa de track puro. |
| Track principal | identidad física del tramo de arranque y fracción de frames correctos del path; ancla V2 híbrida no puede recibir una etiqueta única sin intervalos. |
| Regresiones V3 | diferencias pareadas por evento en errores de identidad, falsas omisiones, cobertura confiable, abstenciones, handoffs correctos y saltos ficticios. |

Comparación primaria jerarquizada: (1) no incrementar falsos merges/frames incorrectos respecto V3 en referencia no ambigua; (2) conservar handoffs correctos con tolerancia de abstención/rechazo preacordada; (3) maximizar cobertura **de identidad confiable** sujeta a (1) y (2), no necesariamente 1641/1680; (4) explicar faltantes. Establecer márgenes cuantitativos de aceptación y precisión estadística *en desarrollo/criterio de riesgo* antes de abrir confirmación; si no se fijan, conclusión `INCONCLUSIVE`. Intervalos Wilson para proporciones, bootstrap pareado **por evento/grupo físico** para diferencias y tablas de confusión con `INDETERMINATE` explícito; errores por video y estratos aun cuando no haya positivos en un estrato. No declarar robustez por 0 falsos positivos en un N pequeño.

## 6. Regla de parada confirmatoria

Si preflight falla, provenance Pose no permite alinear un video, anotación no es independiente o grupos físicos se solapan sin poder separarse: `BLOCKED/INCONCLUSIVE` para esa unidad, conservar logs y denominadores. Si se observa fallo crítico, congelar resultado y analizarlo **después** del análisis preinscrito; corregir solo bajo nueva versión y **nuevo** test independiente. V3.1 jamás se aprobará basándose exclusivamente en Validation76 ni en los 17 gaps.
