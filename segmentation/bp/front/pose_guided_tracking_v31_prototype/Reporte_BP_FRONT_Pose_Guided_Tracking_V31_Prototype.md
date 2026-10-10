# BP_FRONT: Tracking Pose-Guided V3.1 — prototipo sombra (desarrollo)

**Estado:** `PROTOTYPE_NEEDS_REVISION`. Implementación offline sin YOLO/entrenamiento ni publicación. Especificación `bp_front_pose_guided_v31_spec_1`, SHA-256 `735befd58856d5d4156bf0ba4b169b2151aeec73e169d46553d67860e8c61855`; configuración experimental `experimental_config.json` (9b369e597a5c857bb7dc8fcd169f1ab60e6e3c4a3952d004ad9220bdec33ba80). Los valores 30/45 px y ratios derivan de Validation76: **no generalizan por este resultado**. Prototype14 y Validation76 son desarrollo/regresión, jamás confirmación.

## Reserva y procedencia
Inventario read-only: 663 MP4 con SHA-256, 104 excluidos por uso previo y 559 candidatos `RESERVED_UNVERIFIED_DO_NOT_OPEN`; **0 duplicados exactos por SHA-256**, sin poder descartar near-duplicates ni repeticiones físicas. Esos candidatos **no se procesaron** ni se inspeccionó salida V3.1. La independencia física/sesión no está acreditada (`group_id_status=unverified_individual`), por lo que la confirmación sigue `BLOCKED_PHYSICAL_EVENT_INDEPENDENCE_UNVERIFIED`. No se tocó ningún split del clasificador. Posible intersección con futuros tests CNN requiere cruce de IDs/eventos antes de evaluarlos.

Preflight de los 85 videos de desarrollo: hash de MP4 contra RAW e inventario, T Pose/Seg/V2/V3/video, FPS=30, 1920×1080, índices 0-based, timestamps nominales, bytes de máscara RAW→V2, referencias V3 y coordenadas Pose. `sync_sheets/` contiene 11 láminas de inicio/medio/final para los 85 videos: revisión visual de puntos en manos y máscaras sobre video; cero discrepancias observadas en **esos frames muestreados**. Pose `.npy` carece de timestamp, hash histórico de MP4 y hash de frame: **no** certificación criptográfica ni revisión exhaustiva frame a frame. Cualquier futuro mismatch es `SYNC_BLOCKED`; V3 queda baseline separado.

## Componentes y pruebas
`v31_identity.py`: todas las máscaras BAR RAW, distancia a máscara/eje/extensión, q firmado, escala/confidence y soporte temporal; `v31_consistency.py`: saltos robustos y subtracklets virtuales V2→extractor→índice RAW; `v31_handoff.py`: clasificación Pose y **recomputación geométrica con las funciones V3 originales** sobre subtracklets de límites nuevos, sin alterar V3. `01_run_shadow.py` compara V2/V3 y propone salida confiable. `test_v31_shadow.py`: 11 pruebas sintéticas (incluyen intercambio de muñecas, ausencia, barra inmóvil/en movimiento, parcial, híbrido que vuelve y falsa BAR única), ejecutadas correctamente. La prueba de orientación es de política de bloqueo, no prueba con una cámara invertida real.

## Handoffs con revisión visual original — solo Validation76
| Referencia visual | Compatible | Incompatible | Insuficiente | N |
|---|---:|---:|---:|---:|
| 77 aceptados correctos | 76 | 0 | 1 | 77 |
| 2 falsos merges | 0 | 2 | 0 | 2 |

`bp_663` 0→4 @11 y 4→5 @51: ambos `POSE_INCOMPATIBLE`, sin publicar BAR inferior como confiable en bordes; track 4 se divide lógicamente en **[11–16] / [17–48] / [49–50]**, cortes 17 y 49. No se inventan máscaras de barra superior cuando RAW solo contiene la estructura inferior. En 4→5 la decisión usa **solo dos frames del segmento corto** con corroboración de split previo: es una relajación *experimental* documentada, no evidencia confirmatoria. Frames confiables `bp_663`: 46 de 64 frente a 64 V3; 3 BAR únicas incompatibles y resto sin verificar.

`bp_662` track 4: splits lógicos @21/@62; V2 principal híbrido; las propuestas 0→4 @19 y 4→5 @66 reciben `POSE_INCOMPATIBLE` en la copia (V3 originalmente las rechazó, Pose no las promociona). 67/76 frames confiables; el segmento ambiguo no condena todo track 4. Los conteos previos 41/7 y 32/8 son proxies geométricos, no anotación de verdad visual por frame.

Controles `bp_182`: 3 edges compatibles; `bp_238`: correcto indeterminado (gap real), no frames inventados; `bp_538/540/544`: handoffs compatibles con máscaras parcial/completa y grandes cambios de longitud; `bp_102/103`: overlays y estructuras paralelas. Prototype14 único (9 videos fuera de Validation76) muestra riesgo de abstención elevado en `bp_518` y `bp_520` (ver tabla); gaps en `bp_198/301/467` quedan ausentes. Controles son regresión/desarrollo, no validación independiente.

**Diferencia crucial:** el verificador declara 76/77 handoffs correctos compatibles y ninguno incompatible, **pero solo 38/77 tienen ambos frames del borde publicados como confiables**; 39/77 se abstienen en alguno de los lados por ambigüedad por frame, máscaras competidoras, recorte o evidencia insuficiente. Por tanto, no informar «76 handoffs conservados» como si fueran trayectorias continuas. Son pérdidas reales de recuperación que requieren revisión antes de aprobar cualquier uso downstream.

Las 17 propuestas no seleccionadas por V3 mantienen referencia `NOT_APPLICABLE`: Pose emite 14 insuficientes y 3 incompatibles entre ellas; **ninguna** se contabiliza como negativo confirmado sin anotación visual. No se usa su revisión como feature.

## Cobertura y abstenciones (76 videos de Validation76)
| Magnitud | Frames |
|---|---:|
| Frames MP4 | 7141 |
| BAR RAW en algún índice | 7141 |
| V2 selected | 5461 |
| V3 selected | 7102 (V2 + 1641 frames recuperados) |
| V3.1 `OBSERVED_CONFIDENT` | 6940 |
| V3.1 `OBSERVED_UNVERIFIED` | 198 |
| V3.1 `OBSERVED_INCOMPATIBLE` | 3 |
| `NO_BAR_DETECTED` | 0 |
| Menos frames confiables que selección V3 | 162 |

**No hay ground truth visual exhaustiva por frame**: «confiable» es el estado del prototipo, no accuracy demostrada. El **proxy anatómico** selecciona 3 instancias `INCOMPATIBLE_WITH_HELD_BAR` en V3 y 0 en V3.1; esto NO equivale a contar frames erróneos con referencia visual. Los CSV distinguen índice legacy y nuevo; cuando nuevo=-1, `bar_x_px`, `bar_y_px`, `vx_px_s`, `vy_px_s` son `NaN` con máscaras disponibilidad=0, nunca velocidad cero. Cobertura menor no es automáticamente mala si elimina falsas identidades, pero pérdidas en casos visualmente correctos exigen análisis.

## Ablaciones de desarrollo (frames seleccionados, NO precisión de identidad)
| Variante | Frames | Alcance |
|---|---:|---|
| V3 original | 7102 | baseline sin Pose |
| V3 + identidad por frame | 6905 | sin split ni veto de handoff |
| V3 + consistencia intratrack | 7014 | descarta track completo con split; control de componente, conservador y *no* solución final |
| V3 + verificación de handoff | 7049 | sin split ni selección nueva; corta destino tras edge incompatible |
| V3.1 completo | 6940 | subtracklets + evaluador + salida confiable |

Las ablaciones no son equivalentes en política de abstención; los números **no** identifican por sí solos causalidad ni precisión. No atribuir beneficios de selección inicial al validador de handoffs. La variante TRACK_ONLY pierde cobertura por descartar todo track híbrido y no reproduce división lógica; está etiquetada limitación.

## Regresiones más relevantes y diagnóstico
| Video | V3 frames | V31 confiables | Pérdida |
|---|---:|---:|---:|
| bp_520 | 157 | 69 | 88 |
| bp_518 | 161 | 94 | 67 |
| bp_517 | 92 | 73 | 19 |
| bp_201 | 61 | 43 | 18 |
| bp_663 | 64 | 46 | 18 |
| bp_126 | 69 | 54 | 15 |
| bp_139 | 87 | 73 | 14 |
| bp_179 | 227 | 216 | 11 |

En `bp_520` V2 track virtual 7 cubre 157 frames V3, pero 88 quedan indeterminados por máscaras BAR competidoras anatómicamente plausibles sin margen de selección principal calibrado; no es un error de sincronización. `bp_518` también abstiene muchos frames. El prototipo **no preserva suficiente continuidad de asociaciones correctas**: `PROTOTYPE_NEEDS_REVISION`. No se ajustaron umbrales hasta forzar 77/77 bordes confiables. Se requiere mejor discriminación temporal de múltiples máscaras, menos abstención en mascaras parciales y anotación visual real por frame/track para medir FP/FN de identidad. Los dos falsos merges rechazados **no compensan por sí solos** esas regresiones.

La salida actual es una **propuesta por frame con huecos** y subtracklets trazables, no una optimización global de conectividad independiente de la ancla V2. Cuando una máscara apoyada se readquiere tras un gap o cambia de subtracklet, la identidad de ese frame puede tener soporte anatómico, pero el enlace entre segmentos no queda demostrado por ello; velocidades a través del hueco siguen inválidas. Antes de considerar `PROTOTYPE_READY_FOR_CALIBRATION`, implementar selección/reidentificación global sin ancla circular, más controles para conservar edges correctos y una política explícita de procedencia entre segmentos.

## Integridad y reproducibilidad
Antes de ejecución se hashearon 2046 fuentes (RAW, V2, V3, Prototype14, Validation76, modelos, keypoints, labels, features, PoseCNN/checkpoints/splits BP y especificación) y 663 MP4 de inventario. Después, **0 diferencias** en fuentes y videos; hashes finales de código/config/resultados en `bp_front_v31_prototype_manifest.json`; versiones/OS/seed/tiempo allí. Tests sintéticos: 11/11. Sin YOLO, entrenamiento, Features V3.1 globales ni ejecución de Validation17_RAW_GAPS. Scripts y resultados solo en este directorio. Las revisiones visuales originales se cargaron **solo en este verificador**, después de calcular y persistir los resultados; jamás en los tres validadores.

**Recomendación:** no publicar V3.1 ni generar entradas CNN. Autorizar solo revisión de la selección entre BAR simultáneas y política de preservación de handoffs correctos, con nuevo análisis de desarrollo y sin mirar los candidatos confirmatorios. Después deberán congelarse valores efectivos nuevos, comprobar independencia física y ejecutar evaluación visual independiente con referencias reales por frame.
