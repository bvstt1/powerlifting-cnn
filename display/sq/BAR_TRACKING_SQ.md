# Seguimiento exploratorio de la barra en el visor SQ

## Alcance

El visor YOLO `display/sq/visualizar_keypoints_yolo_sq.py` conserva la reproducción
y el esqueleto; añade CSRT opcional para seguir **una ROI elegida manualmente**
en la barra de una vista, preferentemente el extremo visible, manga o centro de
disco. Ni este código ni los CSV diagnósticos alimentan PoseCNN. No se han
modificado vídeos, etiquetas, `.npy` ni checkpoints.

## Procedencia y licencia

Consultados el README, `tracker.py` y la licencia MIT de
<https://github.com/Marticles/barbell-path-tracker> (autor Marticles,
copyright 2018). Conceptos aprovechados: dibujar la ROI con el ratón, iniciar
el seguimiento y acumular centros para trazar el recorrido. No se copió la
clase original ni sus algoritmos: usaba OpenCV 3, Dlib y APIs antiguas y no
contemplaba pausas, fallos ni saltos temporales. `display/sq/bar_tracker.py`
es implementación nueva basada en `cv2.TrackerCSRT_create()` (o
`cv2.legacy.TrackerCSRT_create()` si esa ruta existe) de OpenCV actual, con
estados, exportación y mapeo de coordenadas. La licencia original se incluye
en `display/sq/third_party/barbell-path-tracker-LICENSE` para dejar constancia
de la procedencia y respetar sus condiciones.

## Ejecutar el visor

Desde la raíz del repositorio:

```powershell
python display/sq/visualizar_keypoints_yolo_sq.py --view left --videos sq_007 sq_225 --bar-track
```

Usa inferencia YOLO en directo y los mismos checkpoints que el visor anterior.
Para reproducción ágil sin inferencia, puede leer las secuencias YOLO ya
extraídas, **solo lectura**, sincronizadas por frame:

```powershell
python display/sq/visualizar_keypoints_yolo_sq.py --view left --videos sq_007 sq_225 --bar-track --keypoints
```

Elige siempre la ROI de la barra en la imagen; el esqueleto no identifica el
extremo de la barra. Al cambiar de vídeo se limpia tracker/recorrido. Un
vídeo frontal o derecho puede abrirse con `--view`, aunque estas pruebas se
centraron exclusivamente en `sq_left`.

### Controles

| Tecla / gesto | Acción |
|---|---|
| `T` | Activa/desactiva tracking; apagar limpia la sesión |
| `S` | Pausa y arma selección; **arrastrar con botón izquierdo** desde una esquina a la opuesta sobre la manga/disco. Al soltar se inicializa CSRT en el frame visible |
| `Espacio` | Reanuda / pausa; dibujar en pausa no actualiza CSRT |
| `N` | Avanza exactamente un frame en pausa |
| `B` | Retrocede un frame; descarta tracker y trayectoria, requiere `S` para reiniciar |
| `,` / `.` | Salta 30 frames atrás / adelante; descarta tracker y trayectoria |
| `R` | Reinicia tracker y trayectoria; luego `S` selecciona de nuevo |
| `[` / `]` | Vídeo anterior / siguiente de `--videos`; limpia estado |
| `M` | Muestra/oculta EMA opcional (naranja), siempre separada de la trayectoria original magenta |
| `E` | Exporta CSV de la sesión actual, con nombre único, a `display/sq/bar_diagnostics/` (o `--export-dir`) |
| `Q` / `Esc` | Cierra |

La caja y trayectoria de barra son **magenta**. El **punto exacto seguido**
se señala con una **diana grande** (aro amarillo, centro blanco, retícula)
y la etiqueta `BARRA (X,Y)` en píxeles originales junto a la caja. No es un
landmark del esqueleto: representa el centro de la ROI que devuelve CSRT.
Durante el arrastre se marca provisionalmente el centro de la caja que vas a
seleccionar; no se registra como una medición hasta soltar e inicializar CSRT.
El panel repite `PUNTO BARRA: X=… Y=… px`. **Antes de pulsar `S` y arrastrar
una ROI, no existe todavía un punto que dibujar**: el panel dice
`SIN PUNTO: pulsa S y arrastra ROI`. El esqueleto es verde y sus puntos son
cian/amarillo. El panel indica frame, tiempo y estado:
`sin inicializar`, `siguiendo`, `seguimiento perdido`. Ante pérdida no se
dibuja caja ni centro de ese frame y el CSV registra `perdido` con campos de
posición/caja vacíos. Si CSRT devuelve una caja sobre cuerpo/rack/spotter sin
señalar error, el visor **no la considera automáticamente correcta**: hay
que detectarlo visualmente y usar `R` + `S` para seleccionar de nuevo.

El gráfico lateral traza `y` de la barra **en píxeles de imagen** frente al
frame/tiempo; `y` **aumenta hacia abajo**. Trazo magenta = mediciones crudas;
el naranja opcional es un EMA únicamente visual. No se reemplaza la curva
original ni se incluyen segmentos inventados entre un fallo y una selección.

### Conversión de coordenadas

La vista se muestra en un lienzo fijo de 1500×800 (`WINDOW_AUTOSIZE`). El
vídeo mantiene relación de aspecto dentro de su región; se calculan sus
offsets y escalas reales. Clicks en panel o bandas negras se rechazan; la ROI
se convierte al tamaño **original** del vídeo antes de inicializar CSRT.
No se hace tracking sobre la imagen redimensionada. Se recomiendan ventanas
sin reescalado por SO/DPI para confirmar con el ratón que la ROI dibujada
corresponde al objetivo.

## Pruebas realizadas en desarrollo (sin usar test)

Los IDs y labels se contrastaron con el split congelado de
`sq_left/baseline_t170_v2`: `sq_007` es **válido/train** y `sq_225` es
**inválido SQ-02/validation**. Se escogieron ROIs visibles a mano sobre el
frame 10, en píxeles originales de 1920×1080. El script de inspección sin GUI
emplea el **mismo módulo BarTracker** que el visor:

```powershell
python display/sq/bar_tracking_demo.py --video sq_007 --frame 10 --inspect-only
python display/sq/bar_tracking_demo.py --video sq_225 --frame 10 --inspect-only
python display/sq/bar_tracking_demo.py --video sq_007 --frame 10 --roi 724 156 32 40 --max-frames 105
python display/sq/bar_tracking_demo.py --video sq_225 --frame 10 --roi 780 145 44 43 --max-frames 170
```

| Caso | ROI frame 10 | Frames seguidos | Fallos CSRT | Lectura del preview |
|---|---|---:|---:|---|
| `sq_007` válido | extremo rojo de la manga | 68 (frames 10–77; fin de vídeo) | 0 | Caja permanece visualmente en el extremo durante descenso hasta frame 49 y ascenso posterior. `y` crudo 176→593→157 px, sin saltos interpolados |
| `sq_225` inválido SQ-02 | región de manga/collar izquierdo | 168 (frames 10–177; fin de vídeo) | 0 | Sigue la barra durante fondo y ascenso; alrededor de frame 36 alcanza `y≈522 px`. La gráfica cruda muestra un **nuevo descenso observado** aprox. frames 111–138 (`y≈237→284 px`), antes del siguiente ascenso. Es indicio visual/temporal para revisar SQ-02, no veredicto reglamentario. La caja puede deslizarse **a lo largo de la manga hacia el rack** aunque CSRT siga devolviendo `ok=True`: hay que vigilar ese posible drift |

Artefactos locales, deliberadamente separados del dataset y excluidos de Git:

- `display/sq/bar_diagnostics/sq_007_bar_tracking.csv`, `sq_007_bar_preview.jpg`, `sq_007_bar_bottom_frame_0049.jpg`, `sq_007_bar_y_raw.png`.
- `display/sq/bar_diagnostics/sq_225_bar_tracking.csv`, `sq_225_bar_preview.jpg`, `sq_225_bar_bottom_frame_0036.jpg`, `sq_225_bar_y_raw.png`.

**Límite de verificación:** los CSV confirman actualización en frames
consecutivos; previews y capturas del fondo se revisaron visualmente. Una
prueba de integración simula eventos de ratón/teclado (T, S, arrastre, N, B, R,
reselección, saltos `,`/`.`, cambio de vídeo `[`/`]`, Q) con `VideoCapture` y `.npy` reales, pero
**no equivale a comprobar el ratón sobre una ventana nativa** ni el escalado
de pantalla del equipo. Los tests sin GUI cubren además conversión con bandas,
pausa sin actualizar, saltos, reinicio, fallo y CSV sin posiciones inventadas.
Confirma en pantalla el arrastre y la caja al abrir el visor con el comando
indicado; usa `R` + `S` si la ROI no corresponde exactamente a la manga.

## OpenCV actual

Se mantuvo OpenCV **4.13.0.92**. El entorno tenía simultáneamente
`opencv-python` y `opencv-contrib-python`; inicialmente `cv2` cargaba la rueda
estándar, sin CSRT. Reinstalar **solo** la rueda contrib de la **misma versión**
(`python -m pip install --force-reinstall --no-deps opencv-contrib-python==4.13.0.92`)
activó `cv2.TrackerCSRT_create()` y `cv2.legacy.TrackerCSRT_create()` sin
bajar versión ni cambiar NumPy o el resto del entorno. El visor produce un
error descriptivo si CSRT no está disponible, en lugar de seguir con otro
algoritmo silenciosamente.

Pruebas:

```powershell
python -m unittest display.sq.test_bar_tracker
```

## Etapa posterior

Decidir si hay calidad/consistencia suficiente para obtener features de
barra en un experimento **separado**. Por ahora el tracking es diagnóstico
visual; no constituye un sexto keypoint de PoseCNN ni una prueba automática
de validez de la sentadilla.
