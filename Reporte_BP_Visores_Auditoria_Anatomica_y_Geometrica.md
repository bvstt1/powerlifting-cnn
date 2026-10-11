# BP — Auditoría anatómica y geométrica de los visores FRONT, LEFT y RIGHT

## Resumen ejecutivo

**Resultado lateral:** los índices `2` y `3` de `bp_side_skeleton_v2.pt`
corresponden, según la evidencia disponible, a **cadera y hombro**. La franja
vigente del 45–75 % entre ambos atraviesa una zona superior del torso y su
contorno procede de la **máscara de persona**, no de un punto de pecho. Los
overlays muestran que una franja alternativa entre hombro y codo tiende a
desplazarse al brazo o a estrecharse. Se conservaron **índices y fórmula**;
se corrigieron nombres, docstrings y textos para dejar de llamarla «pecho
anatómicamente detectado». La inferencia lateral ahora declara `imgsz=960`;
**el checkpoint ya empleaba 960 como valor por defecto**, por lo que esto
fija explícitamente el valor y **no** explica ninguna mejora del baseline real.

**Resultado frontal:** el selector original puede aceptar una estructura
inferior etiquetada `BAR`, incluso cuando las muñecas sostienen una barra
superior: quedó demostrado con RAW existente en `bp_662/663`. Se añadió una
guardia limitada: con muñecas fiables, se descarta una candidata cuya
**mediana vertical de máscara** esté más de `0,03·alto_frame` **por debajo**
del promedio vertical de las muñecas. Se distingue además «YOLO detectó
BAR» de «BAR asociada al levantador». **No se relajó** la regla horizontal
del 50 %: en `bp_535/537` la barra superior del soporte existe como objeto,
pero los overlays no permiten verificar que sea la barra *sostenida*. Aceptar
automáticamente esos casos aumentaría cobertura sin demostrar precisión.

**Alcance:** auditoría acotada a casos de desarrollo/regresión, sin entrenar
redes, sin cambiar pesos ni tracking y sin producir el dataset completo de
features. No se ha demostrado contacto físico barra-pecho en ninguna vista.

### Política de versionado de resultados

**Los resultados producidos por `display/bp/audit_visores_bp.py` no se
versionan.** La carpeta `display/bp/auditoria_visores_bp/` está ignorada por
Git: sus JPG, CSV, JSON, snapshots y diff se conservan solamente en el
workspace donde se ejecutó la auditoría. Los enlaces a esa carpeta dentro de
este reporte son **rutas locales**; no estarán disponibles al clonar el
repositorio hasta regenerar los resultados con los videos y modelos
correspondientes. Se versionan este informe, el código de los visores y el
script de auditoría, no las salidas del script.

El script permite capturar una nueva referencia local con
`--phase actual-before` **antes** de futuros cambios y comparar luego con
`--phase actual-after`, `--phase actual-compare` y `--phase verify`. Para
repetir controles concretos están `--phase raw-controls`, `--phase mp-right`
y `--phase temporal`. La comparación histórica descrita aquí usó snapshots
`source_before/` capturados durante la auditoría: **una clonación nueva no
puede reconstruir retrospectivamente esos snapshots exactos** a partir del
commit previo porque `yolo_seg_bp_front.py` ya contenía trabajo local no
confirmado antes de comenzar. Los resultados históricos permanecen
documentados numéricamente en este reporte, sin presentar una regeneración
futura como si fuera idéntica.

## 1. Fuentes examinadas y trazabilidad

- Código: [`display/bp/yolo_seg_bp_side.py`](display/bp/yolo_seg_bp_side.py),
  [`display/bp/yolo_seg_bp_front.py`](display/bp/yolo_seg_bp_front.py),
  [`extract_keypoints/bp/extract_keypoints_yolo_bp.py`](extract_keypoints/bp/extract_keypoints_yolo_bp.py),
  [`extract_keypoints/bp/extract_keypoints_mediapipe_bp.py`](extract_keypoints/bp/extract_keypoints_mediapipe_bp.py),
  [`display/bp/visualizar_keypoints_yolo_bp.py`](display/bp/visualizar_keypoints_yolo_bp.py)
  y el esqueleto histórico [`display/bp/fast/yolo_skelton_bp_side.py`](display/bp/fast/yolo_skelton_bp_side.py).
- Pesos leídos, sin modificaciones: `bp_front_seg_v2.pt`,
  `bp_front_skeleton_v8.pt`, `bp_side_seg_v1.pt` y
  `bp_side_skeleton_v2.pt`.
- Anotaciones de contraste: Pose Landmarker de MediaPipe, con landmarks
  anatómicos conocidos, en frames LEFT preextraídos y en inferencias
  **puntuales** RIGHT. MediaPipe no se tomó como ground truth infalible.
- Evidencia congelada frontal: máscaras y cajas de
  `segmentation/bp/front/raw/bp_{535,537,662,663}_segmentation.npz`, sin
  alterarlas, más los registros de revisión visual preexistentes de 662/663.

**Límite de fuentes:** `Documentacion_BP_Visores_Segmentacion_y_CNN.md`,
solicitado como lectura inicial, **no está actualmente en este workspace**
(no se sobrescribió ni se reconstruyó). Se utilizó el contexto aportado en
la conversación y los archivos reales. El checkpoint lateral conserva la
ruta de entrenamiento `/content/bp_side-2/data.yaml`, y el frontal apunta a
un `data.yaml` fuera de este workspace: **ni los `data.yaml` ni las
anotaciones originales del entrenamiento están disponibles aquí**. El
checkpoint confirma `kpt_shape=[8,3]` lateral y `[6,3]` frontal, pero
**no incrusta nombres de articulaciones**. Por ello no se presenta el esquema
como certificación directa contra anotaciones originales inexistentes.

| Checkpoint | `train_args.data` embebido (ruta histórica, no presente) | `train_args.imgsz` | `train_args.fliplr` |
|---|---|---:|---:|
| BP side pose v2 | `/content/bp_side-2/data.yaml` | 960 | 0,0 |
| BP front pose v8 | `C:\Users\PC\PycharmProjects\bp_train\dataset_bp_front_v6\data.yaml` | 960 | 0,0 |
| BP side segmentation v1 | `C:\Users\PC\PycharmProjects\sq_train\datasets\bp_side_seg_v1\data.yaml` | 960 | 0,0 |
| BP front segmentation v2 | `/content/bp_front_seg-2/data.yaml` | 960 | 0,0 |

`kpt_shape`, el path histórico de entrenamiento y las conexiones del
esqueleto son evidencias diferentes: ninguna ruta histórica demuestra por sí
sola la nomenclatura original de los puntos, y no se ha recuperado una
anotación `data.yaml` para certificarla retrospectivamente.

## 2. Registro de inferencia: corrección de procedencia indispensable

Los cuatro checkpoints se entrenaron con `imgsz=960`, y Ultralytics carga
**`imgsz=960` por defecto desde estos checkpoints** cuando se invoca el modelo
sin ese argumento. Se comprobó mediante `modelo.predictor.args` en una
inferencia real. Los visores **antes** de la auditoría ya trabajaban con:

| Modelo | Visor ANTES efectivo | Visor DESPUÉS efectivo |
|---|---:|---:|
| BP front segmentation v2 | 960 | 960 |
| BP front pose v8 | 960 explícito | 960 explícito |
| BP side segmentation v1 | 960 embebido | 960 explícito |
| BP side pose v2 | 960 embebido | 960 explícito |

En los visores se usa la configuración de predicción por defecto
`conf=0,25`, `iou=0,7`, `retina_masks=False`; los parámetros quedan registrados
en [`parametros_actual-before.json`](display/bp/auditoria_visores_bp/parametros_actual-before.json)
y [`parametros_actual-after.json`](display/bp/auditoria_visores_bp/parametros_actual-after.json).
Los RAW frontales preexistentes se extrajeron también con `imgsz=960`, pero
con **`retina_masks=True`**: no deben sustituir silenciosamente las máscaras
de un visor con `retina_masks=False`, aunque en frames muestreados la geometría
de sus cajas sea comparable. El informe del dataset congelado registra modelo,
versión y parámetros en
[`segmentation/bp/front/manifests/bp_front_dataset_v1_metadata.json`](segmentation/bp/front/manifests/bp_front_dataset_v1_metadata.json).

**Rectificación de una exploración preliminar:** se guardaron unas imágenes
`before/`, `after/` y `comparaciones/` con **640 forzados**. Fue un experimento
de sensibilidad a resolución, **no un antes/después fiel al visor original**.
Se conservaron con sus parámetros explícitos y no se usan para atribuir
mejoras al cambio de código. La comparación válida es
[`comparaciones_actual/`](display/bp/auditoria_visores_bp/comparaciones_actual/),
**960→960**. Su tabla numérica por video/frame es
[`comparacion_frames_actual.csv`](display/bp/auditoria_visores_bp/comparacion_frames_actual.csv).

## 3. Anatomía FRONT y lateral: evidencia por fuente

| Índice | FRONT v8 | SIDE v2 | MediaPipe en orden equiparable SIDE |
|---:|---|---|---|
| 0 | Codo izquierdo | Rodilla visible | 25 LEFT / 26 RIGHT |
| 1 | Hombro izquierdo | Tobillo visible | 27 / 28 |
| **2** | Hombro derecho | **Cadera visible** | **23 / 24** |
| **3** | Codo derecho | **Hombro visible** | **11 / 12** |
| 4 | Muñeca derecha | Muñeca/mano visible | 15 / 16 |
| **5** | Muñeca izquierda | **Codo visible** | **13 / 14** |
| 6 | — | Talón visible | 29 / 30 |
| 7 | — | Punta visible | 31 / 32 |

Fuentes de apoyo: los extractores YOLO y MediaPipe independientes, conexiones
de esqueleto `2–0–1–6–7` (pierna) y `2–3–5–4` (torso y brazo), inferencia
del checkpoint de ocho puntos y overlays de frames reales. **No se sustituyó
el esquema personalizado por COCO** ni se aplicó flip de imagen. En las
comparaciones LEFT guardadas, la distancia mediana YOLO↔MediaPipe en
**7 frames** fue aproximadamente **58 px para cadera 2** y **42 px para
hombro 3**, con errores individuales mayores cuando hay oclusión. En RIGHT,
MediaPipe puede asignar el hombro equivocado en algunos frames; en los siete
frames puntuales con visibilidad suficiente, las medianas para el landmark
correspondiente fueron ~**56 px (cadera)** y ~**65 px (hombro)**, pero
**no** garantizan exactitud anatómica en todos los frames. La secuencia
cadera-hombro-codo-muñeca, la geometría de las piernas y las imágenes
permiten distinguir el esquema operativo de la falsa lectura
`2=hombro,3=codo`.

**Ejemplos visuales con todos los índices y nombres:**

- [LEFT `bp_003` f30 antes/original/después](display/bp/auditoria_visores_bp/comparaciones_actual/left_bp_003_f0030.jpg).
- [LEFT `bp_345` f61](display/bp/auditoria_visores_bp/actual-after/left_bp_345_f0061.jpg).
- [RIGHT `bp_003` f30 antes/original/después](display/bp/auditoria_visores_bp/comparaciones_actual/right_bp_003_f0030.jpg).
- [RIGHT `bp_342` f0, con puntos MediaPipe blancos independientes](display/bp/auditoria_visores_bp/mp_right/bp_342_f0000.jpg).
- [RIGHT `bp_003` f30, ejemplo donde MediaPipe confunde un hombro](display/bp/auditoria_visores_bp/mp_right/bp_003_f0030.jpg).
- Tabla de contraste: [`mp_right_comparison.csv`](display/bp/auditoria_visores_bp/mp_right_comparison.csv)
  y distancias LEFT en [`frames_actual-after.csv`](display/bp/auditoria_visores_bp/frames_actual-after.csv).

### Hipótesis A frente a B y por qué visualmente funcionaba

En la formulación vigente, la coordenada horizontal avanza desde **cadera
(`2`) hacia hombro (`3`)**:

```text
x_inicio = x_cadera + 0,45 · (x_hombro - x_cadera)
x_fin    = x_cadera + 0,75 · (x_hombro - x_cadera)
```

Esa franja se sitúa entre el tronco medio y el hombro en las imágenes
inspeccionadas, incluso si el sentido de `x` se invierte en LEFT/RIGHT: se
usan `min`/`max`, no una dirección lateral fija. **Después** se toma el primer
píxel superior de la **máscara `person`** en cada columna, se exige 70 % de
cobertura y se usa una mediana. Por ello el punto final no es sencillamente
la interpolación entre articulaciones. Puede dibujarse sobre el contorno del
torso alto y verse plausible aunque los nombres originales del código sean
incorrectos.

La alternativa de trasladar exactamente el 45–75 % a **hombro (`3`)–codo
(`5`)** se inspeccionó sin integrarla: tiende a desplazarse hacia el brazo
levantado, o produce una franja estrecha/inutilizable. Como evidencia
cuantitativa: RIGHT `bp_003` f30 produjo **ROI original `(959,418)`** frente
a **alternativa `(840,495)`**, un cambio de 77 px en altura que ubica la
alternativa hacia el brazo. LEFT `bp_003` f30 produjo `(984,380)` con la
original, y el overlay muestra que el brazo no es una referencia torácica
mejor pese a una altura coincidente en alguna columna. La ROI no constituye
una segmentación del pecho: puede incluir **brazo, ropa o contorno dorsal** y
su fiabilidad varía entre atletas y perspectivas. `bp_190 left` no dispone
de YOLO Pose y usa respaldo del banco, identificado como estimación.

**Veredicto:** la evidencia favorece **Hipótesis A: nomenclatura errónea,
geometría de franja de torso razonable**, no un error demostrado que
justifique cambiar índices. Se renombraron `CADERA=2`, `HOMBRO=3`, las
funciones y textos a «contorno del torso». Los coeficientes `0,45/0,75`,
keypoints numéricos, reglas del perfil y geometría de barra **no cambiaron**.
Una comprobación de funciones antiguas/nuevas sobre las **mismas** inferencias
960 produjo coordenadas **exactamente iguales en cinco controles** LEFT y
RIGHT; véase [`regresiones_actual.json`](display/bp/auditoria_visores_bp/regresiones_actual.json).

## 4. Revisión de la geometría lateral

1. **Orientación:** `inclinacion_barra` usa la mediana vertical de columnas
   de máscara interiores y ajusta `y=a·x+b`, descartando extremos de discos;
   la condición `|a|≤0,35` y el residuo limitado rechazan máscaras deformadas.
2. **Espejo LEFT/RIGHT:** las coordenadas usan `x` de imagen real. Al reflejar
   una máscara sintética, la pendiente pasó de **+0,24548 a −0,24548**,
   conservando el ancho visible de 122 px. La proyección debe evaluarse en
   la `x` de destino de la misma imagen; el signo vertical no se invierte.
3. **Separación:** `Δ = y_contorno − [y_barra_prox + radio + a·(x_contorno−x_barra)]`.
   `Δ>0` significa barra proyectada por encima del contorno; `Δ<0`, solape
   proyectado. Ni solape ni cercanía prueban contacto físico.
4. **Cercanía:** `|Δ| ≤ max(6, round(0,012·alto_frame))`, aproximadamente
   13 px en 1080. No se modificó esta tolerancia.
5. **Alcance:** con `D=|x_contorno−x_barra|` y longitud observada `L`,
   `D≤2L` usa la proyección descriptiva ordinaria, `2L<D≤4L` dibuja línea
   naranja de baja confianza **sin juicio de contacto**, `D>4L` se abstiene.
6. **Máscaras parciales:** el valor de orientación puede no describir el
   tramo oculto; el aviso de alcance se mantiene. En RIGHT `bp_203` f0,
   **`D/L≈2,64`**, por lo que se mantiene como proyección lejana.

Controles **reales 960→960**: LEFT `bp_003` f30 tuvo `Δ≈−11 px` antes y
después, y RIGHT `bp_003` f30 `Δ≈−13 px` antes y después; no son vistas
emparejadas del mismo intento. RIGHT `bp_342` f18 dio `Δ≈+12 px`, estado
«SOLO CERCA (2D) - NO ES CONTACTO». No se optimizaron umbrales con estas
etiquetas. LEFT `bp_190` muestra que, aun con pose YOLO ausente, el banco
puede aportar una región de respaldo **menos fiable**.

## 5. Filtro frontal: análisis del 50 %, protección 662/663 y 535/537

**Antes:** una candidata BAR se conservaba si al menos 50 % de sus píxeles
caía dentro de la caja de `person` **o** del intervalo horizontal entre
muñecas. Máscaras que coincidían en ≥85 % de la menor se deduplicaban. Si
había dos barras internas distintas, el visor declaraba «medida ambigua».

**Problema demostrado:** en RAW960 existente:

| Video/frame | Detecciones RAW | Relación con muñecas | Resultado anterior | Resultado tras guardia |
|---|---|---|---|---|
| `bp_662` f18 | BAR superior conf≈0,75; inferior conf≈0,30 | Medianas `dy≈−35 px` y `+80 px` | Dos plausibles por cobertura horizontal | Solo superior asociada |
| `bp_662` f19 | Solo BAR inferior conf≈0,66 | `dy≈+70 px` | Inferior asociada | Identidad no verificada |
| `bp_663` f11 | Solo BAR inferior conf≈0,65 | `dy≈+90 px` | Inferior asociada | Identidad no verificada |
| `bp_663` f18 | BAR superior conf≈0,86 | `dy≈−39 px` | Superior asociada | Superior asociada |

`dy = mediana_y_mascara_BAR − promedio_y_muñecas` se interpreta con `y`
creciente hacia abajo. Las vistas RAW muestran claramente la barra superior
agarrada y una estructura inferior que **no lo está**. El filtro mínimo
rechaza candidatas con muñecas fiables cuando `dy > 0,03·alto_frame`
(~32 px para 1080). La condición se combina con el selector vigente, sin
fabricar otra identidad ni modificar el RAW. Ejemplos:
[RAW `bp_663` f11](display/bp/auditoria_visores_bp/raw960/bp_663_f0011.jpg),
[RAW `bp_662` f18](display/bp/auditoria_visores_bp/raw960/bp_662_f0018.jpg),
[comparativa del visor `bp_663` f11 a 960/retina_masks=False](display/bp/auditoria_visores_bp/comparaciones_actual/front_bp_663_f0011.jpg).

**Diferencia de inferencia honesta:** en un mismo frame como `bp_663` f11,
el RAW archivado (`retina_masks=True`) detecta únicamente la estructura
inferior, pero esto no autoriza inventar la barra superior. El visor a 960
con `retina_masks=False` también puede obtener esa candidata inferior en
la comparación real; se rechaza. En otras exploraciones **forzadas a 640**
puede aparecer una barra superior distinta: esos experimentos están
separados y **no** sustituyen los RAW congelados ni prueban la corrección.

**Casos 535 y 537:** el RAW archivado informa detección `BAR` en
**263/263** y **287/287** frames, respectivamente. Bajo la configuración
**real del visor (960/retina_masks=False)** la revisión reproducible obtuvo:

| Video | Frames | YOLO visor detecta BAR | Asociación antes | Asociación después |
|---|---:|---:|---:|---:|
| `bp_535` | 263 | 263 | 233 | **233** |
| `bp_537` | 287 | 287 | 200 | **200** |

Fuente:
[`verificacion_actual_535_537_frames.csv`](display/bp/auditoria_visores_bp/verificacion_actual_535_537_frames.csv)
y [`regresiones_actual.json`](display/bp/auditoria_visores_bp/regresiones_actual.json).
Los 30 y 87 frames sin asociación **no** son BAR ausente del modelo. En
`bp_535` f50 y `bp_537` f60 una barra está realmente visible arriba mientras
las muñecas aparecen ~190/~144 px **más abajo**, con poses afectadas por la
secuencia. El levantamiento nulo según el usuario **no se usó** para decidir
si existe barra. Se probó una alternativa basada en columnas de intersección
absoluta con el intervalo entre muñecas: recuperaría esos 30/87 casos, pero
**no demuestra que el objeto sea la barra sostenida**, y una regla basada
solo en intersección también admitiría la estructura inferior de 662/663.
No se integró por falta de mejora visual demostrada. En cambio, cuando el
modelo emite BAR pero falta persona segmentada o el filtro se abstiene, el
visor ahora indica **«BAR detectada por visor; identidad no verificada»** o
**«Sin persona segmentada; BAR detectada por visor»**. «Por visor» se refiere
a la inferencia corriente; **no** afirma que esa máscara sea el RAW960
archivado. La clase detectada, la identidad sostenida y la validez del intento
son problemas separados.

Las tablas de cada candidata, caja, confianza, fracción sobre persona,
fracción entre muñecas y decisión están en
[`frames_actual-before.csv`](display/bp/auditoria_visores_bp/frames_actual-before.csv),
[`frames_actual-after.csv`](display/bp/auditoria_visores_bp/frames_actual-after.csv)
y [`raw960_frames.csv`](display/bp/auditoria_visores_bp/raw960_frames.csv).

## 6. Zona central FRONT y estados de barra

El visor frontal combina máscara `person` y una franja interior entre los
**hombros 1/2**. Marca ambigua una silueta cuyo contorno superior queda más
de `0,35` veces la separación interhombros por encima de estos. El perfil
puede ser **cabeza, ropa o torso**, no una máscara de pecho. Las muestras
`bp_003`, `bp_342` y `bp_591` siguen identificadas como anatómicamente
ambiguas; no se encontró evidencia suficiente para cambiar su región con
muñecas o banco sin arriesgar seleccionar cabeza/brazos. Ejemplos
[`bp_003` f30](display/bp/auditoria_visores_bp/comparaciones_actual/front_bp_003_f0030.jpg),
[`bp_342` f30](display/bp/auditoria_visores_bp/comparaciones_actual/front_bp_342_f0030.jpg),
[`bp_591` f30](display/bp/auditoria_visores_bp/comparaciones_actual/front_bp_591_f0030.jpg).
Una distancia aproximada de +6 px en `bp_003` f30 es distancia a **silueta
central**, no prueba de contacto con el pecho.

La procedencia frontal sigue separada:

1. BAR detectada por la inferencia actual.
2. BAR observada pero no asociada o varias BAR internas distintas: identidad
   ambigua y sin juicio de contacto.
3. BAR observada cuyo borde central falta: **proyección espacial** desde
   máscara visible, de baja confianza.
4. Ninguna BAR detectada por YOLO durante un hueco interior corto: **línea
   temporal estimada mediante muñecas**, sin crear máscara.
5. Ausencia sin anclas suficientes: no evaluable.

La lógica temporal existente de `bp_467` permanece igual: anclas 43 y 50,
estimación discontinua **44–49** y contacto **no evaluable**. `bp_139` no
extrapola una barra que sale por el borde superior; `bp_267` no necesita
interpolación. Véanse
[`temporal/bp_467_f0046.jpg`](display/bp/auditoria_visores_bp/temporal/bp_467_f0046.jpg)
y [`temporal_controls.json`](display/bp/auditoria_visores_bp/temporal_controls.json).
La estimación utiliza información posterior del video completo: **no es
causal ni ground truth del frame**.

## 7. Regresión, integridad y archivos de evidencia

**Selección auditada:** **61** frames puntuales entre FRONT, LEFT y RIGHT,
incluyendo controles correctos, atletas diferentes, máscaras parciales,
duplicados, salida del encuadre y detección ausente. No se procesaron los
662/663 videos frontalmente para ajustar reglas; `bp_535` y `bp_537` sí se
comprobaron íntegros **solo en la verificación acotada de cobertura del
filtro**, sin etiquetas de validez para calibrar.

**Resultados de regresión:**

- Con fuente y tamaño **idénticos 960**, `13/13` frames muestreados LEFT y
  `14/14` RIGHT conservan una medida de separación antes/después. La ROI
  basada en pose permaneció `10/13` LEFT y `14/14` RIGHT; no se atribuye
  ninguna ganancia espuria al código lateral.
- Cinco frames de pose LEFT/RIGHT evaluados sobre **la misma inferencia 960**
  devuelven una ROI **exactamente idéntica** antes/después del renombrado.
- La pendiente sintética espejada `+0,24548 ↔ −0,24548`, con longitud
  invariante de 122 px, confirma signo y orientación. Los umbrales de
  cercanía y `D/L` no se modificaron.
- La nueva guardia frontal conserva la barra superior observada y se abstiene
  ante la estructura inferior RAW960; para `bp_535/537` no aumenta cobertura
  sin prueba de identidad.
- Ejecución headless completa de los visores en `left/bp_190`,
  `right/bp_003`, `right/bp_203`, `front/bp_467` y `front/bp_663` sin fallos.
- No se realizaron flips, splits, cambios de etiquetas, entrenamiento o
  ajuste de checkpoints.

**Artefactos:** las rutas bajo `auditoria_visores_bp/` son locales e
ignoradas por Git según la política anterior.

| Entregable | Ubicación |
|---|---|
| Script de verificación: lee inputs, escribe solo evidencia nueva | [`display/bp/audit_visores_bp.py`](display/bp/audit_visores_bp.py) |
| Tabla consolidada por video/frame y previews 960→960 | [`comparacion_frames_actual.csv`](display/bp/auditoria_visores_bp/comparacion_frames_actual.csv) |
| Revisión visual explícita por caso | [`revisiones_visuales.csv`](display/bp/auditoria_visores_bp/revisiones_visuales.csv) |
| Previews originales/antes/después | [`comparaciones_actual/`](display/bp/auditoria_visores_bp/comparaciones_actual/) |
| Máscaras existentes RAW960 (control de identidad) | [`raw960/`](display/bp/auditoria_visores_bp/raw960/) |
| Overlay independiente de MP RIGHT | [`mp_right/`](display/bp/auditoria_visores_bp/mp_right/) |
| Previews de interpolación temporal | [`temporal/`](display/bp/auditoria_visores_bp/temporal/) |
| Parámetros efectivos | [`parametros_actual-before.json`](display/bp/auditoria_visores_bp/parametros_actual-before.json), [`parametros_actual-after.json`](display/bp/auditoria_visores_bp/parametros_actual-after.json) |
| Comparación de comportamiento | [`regresiones_actual.json`](display/bp/auditoria_visores_bp/regresiones_actual.json) |
| Diff de visores contra snapshot local antes de esta auditoría | [`code_diff.patch`](display/bp/auditoria_visores_bp/code_diff.patch) |

Los [hashes BEFORE](display/bp/auditoria_visores_bp/input_hashes_actual-before.json)
y [AFTER](display/bp/auditoria_visores_bp/input_hashes_actual-after.json)
coincidieron para los **cuatro pesos**, los **tres CSV BP**, los RAW frontales
seleccionados `535/537/662/663`, el checkpoint PoseCNN frontal y su split.
La auditoría no escribió sobre videos ni `.npy` originales. Esto es una
verificación de **los inputs seleccionados**, no una certificación de que
otros archivos no relacionados del working tree —con cambios preexistentes
del usuario— estén intactos.

## 8. Limitaciones y características candidatas para SegCNN

La información que se podría exportar de forma **reproducible** por frame
incluye keypoints `(x,y,confianza)` y disponibilidad; BAR asociada o no,
posición/anchura/orientación/confianza; separación **barra–silueta 2D**
normalizada, cobertura y `D/L`; referencia/bounding box/estabilidad del
banco; número de candidatas y duplicados; calidad de pose, máscara observada,
proyección espacial, estimación temporal, ausencia o identidad incierta.
La tabla debería conservar `(ruta de video, vista, frame original, versión
de modelos y parámetros de inferencia)` y **máscaras de disponibilidad y
procedencia de cada medida**. `left/bp_NNN` y `right/bp_NNN` no se consideran
un mismo intento por su número. No se ha validado supervisión frame a frame
de contacto; `label=0` del CSV o una distancia pequeña **no** la proporcionan.

Para una futura exportación `(T,F)` faltan: especificación estable de canales
y unidades, auditoría anatómica del contorno en otros atletas, evaluación de
identidad de barra separada de BAR de soporte, política explícita para
`OBSERVADO_NO_VERIFICADO` y versionado de los parámetros. **Se puede comenzar
una extracción descriptiva de desarrollo con procedencia y abstenciones;
no** conviene tratar estas mediciones como contacto anatómico confirmado ni
entrenar aún un modelo de fusión que dependa de dicha afirmación. El baseline
BP frontal PoseCNN existente no recibió nuevas features.

## 9. Cambios integrados y siguiente paso

Archivos de aplicación modificados: `display/bp/yolo_seg_bp_side.py`
(nomenclatura y `imgsz=960` explícito, sin alterar ROI),
`display/bp/yolo_seg_bp_front.py` (guardia inferior y estados que distinguen
BAR inferida por visor de BAR asociada). Archivos **nuevos**: script de
auditoría, previews/tablas/parametrización/diff y este reporte. El documento
original solicitado no estaba disponible y **no** se sobrescribió.

**Recomendación:** la próxima fase, si se autoriza, sería especificar y
auditar un extractor **descriptivo** `(T,F)` basado en la inferencia
explícitamente versionada, con `observado/proyectado/estimado/no_verificado/
ausente`, sin entrenar todavía ni reinterpretar «pecho» como ground truth.
Una evaluación posterior con anotaciones anatómicas/manuales específicas
sería necesaria para sustentar un detector de contacto real.
