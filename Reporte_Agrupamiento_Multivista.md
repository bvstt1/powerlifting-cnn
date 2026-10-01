# Reporte de Agrupamiento Multivista para Entrenamiento

**Proyecto:** Juez virtual de powerlifting
**Fecha:** 15 de septiembre de 2026

## 1. Objetivo

Este documento define cómo deben agruparse los videos de distintas vistas antes
de dividir el dataset en entrenamiento, validación y test. El objetivo principal
es impedir que dos grabaciones del mismo levantamiento aparezcan en particiones
diferentes y produzcan métricas artificialmente altas.

La metodología se aplicará a:

- Peso muerto (`dl`).
- Press de banca (`bp`).
- Sentadilla (`sq`).

## 2. Muestra y levantamiento no son lo mismo

Cada archivo de video constituye una **muestra**, pero un mismo levantamiento
puede tener varias muestras si fue grabado simultáneamente desde left, right y
front.

Ejemplo:

```text
Levantamiento 050
├── Vista right: video A
└── Vista left:  video B
```

Aunque A y B sean archivos diferentes, contienen la misma ejecución, el mismo
atleta, la misma carga, el mismo ritmo y el mismo entorno. Por ello deben
considerarse parte de un único grupo experimental.

## 3. Por qué se necesita `group_id`

Un split aleatorio convencional puede producir esta situación:

```text
Train: right del levantamiento 050
Test:  left del levantamiento 050
```

La red podría reconocer características ya vistas durante el entrenamiento:

- Identidad y proporciones del atleta.
- Ropa y equipamiento.
- Duración y ritmo de la repetición.
- Carga utilizada.
- Fondo, iluminación y sesión de grabación.
- Trayectoria general del movimiento.

En ese caso, el test deja de medir correctamente la generalización a un
levantamiento desconocido. Este problema se denomina **fuga de información** o
**data leakage**.

Para evitarlo, cada ejecución física debe recibir un `group_id`. Todas las
vistas conocidas del mismo levantamiento deben permanecer juntas en train, val
o test.

## 4. Las etiquetas pueden ser diferentes

Compartir un `group_id` no obliga a que left y right tengan la misma etiqueta.

Ejemplo válido:

```text
group_id = dl_event_050
right → label 1
left  → label 0
```

Esto puede ocurrir cuando la etiqueta representa lo que es observable desde
cada cámara. El agrupamiento solo controla la partición del dataset; no modifica
ni reconcilia las etiquetas.

Si en el futuro la etiqueta se define como resultado global del levantamiento,
las discrepancias entre vistas deberán revisarse por separado.

## 5. Estrategia universal

Se utilizará el mismo principio para todos los ejercicios:

1. Si dos o más videos pertenecen con certeza al mismo levantamiento, comparten
   `group_id`.
2. Si no existe evidencia suficiente para relacionarlos, cada video recibe un
   grupo individual.
3. Los IDs numéricos similares no demuestran por sí solos que dos videos sean
   pareja.
4. Los videos sin pareja siguen siendo utilizables.

Esto permite usar una única estrategia de split estratificado por grupos para
DL, BP y SQ. Cuando todos los grupos contienen una sola muestra, el resultado es
equivalente en la práctica a un split estratificado convencional.

## 6. Mapeo conocido para DL side

La carpeta consolidada `dataset/dl/side` contiene:

```text
Right: dl_001 – dl_413
Left:  dl_414 – dl_825
```

En la vista left faltaba el video original `dl_310` (corrección de auditoría:
la hipótesis previa del hueco en `dl_401` quedó refutada por conteo de frames
— `left[N]==right[N]` hasta N=309 y `left[N]==right[N+1]` desde N=310; el
evento 401 sí tiene ambas vistas). Después de corregir la numeración, la
relación entre vistas queda así:

```text
Right dl_001 – dl_309 ↔ Side-left dl_414 – dl_722
Right dl_310          ↔ sin pareja left
Right dl_311 – dl_413 ↔ Side-left dl_723 – dl_825
```

Ejemplos:

```text
Right dl_001 ↔ Side-left dl_414 → group_id dl_event_001
Right dl_309 ↔ Side-left dl_722 → group_id dl_event_309
Right dl_310                    → group_id dl_event_310
Right dl_311 ↔ Side-left dl_723 → group_id dl_event_311
Right dl_401 ↔ Side-left dl_813 → group_id dl_event_401
Right dl_413 ↔ Side-left dl_825 → group_id dl_event_413
```

## 7. Tarea pendiente prioritaria para BP y SQ

> **TAREA PRINCIPAL:** identificar qué videos left y right de press de banca y
> sentadilla corresponden al mismo levantamiento físico.

No se deben emparejar videos únicamente porque tengan el mismo número. La
relación debe confirmarse mediante contenido visual o metadata confiable.

### Evidencias recomendadas

- Timestamp o registro de grabación.
- Duración y cantidad de frames.
- Instante de inicio y final del movimiento.
- Identidad, ropa y posición del atleta.
- Discos, carga y configuración de la barra.
- Ritmo de ejecución y trayectoria de la barra.
- Fondo, personas y objetos visibles.
- Archivos generados por la herramienta de grabación sincronizada.

### Restricción importante

No utilizar las etiquetas válido/nulo ni los códigos de falta para decidir si
dos videos son pareja. Hacerlo introduciría información del objetivo durante la
construcción del split.

## 8. Entregable esperado

Se recomienda crear un manifiesto por ejercicio, por ejemplo:

```csv
video_id,source_view,source_video_id,group_id,paired
bp_001,left,bp_001,bp_event_001,1
bp_514,right,bp_014,bp_event_001,1
bp_002,left,bp_002,bp_event_002,0
bp_515,right,bp_015,bp_event_003,0
```

Campos mínimos:

| Campo | Descripción |
|-------|-------------|
| `video_id` | ID utilizado por keypoints y CSV consolidado |
| `source_view` | Vista original: left o right |
| `source_video_id` | Nombre o ID previo a la consolidación |
| `group_id` | Identificador del levantamiento físico |
| `paired` | Indica si tiene otra vista confirmada |

También pueden añadirse atleta, sesión, timestamp y observaciones.

## 9. Procedimiento recomendado

1. Revisar BP y registrar parejas confirmadas.
2. Asignar un grupo individual a cada video BP sin pareja.
3. Repetir el proceso para SQ.
4. Comprobar que ningún video aparezca en más de un grupo.
5. Verificar manualmente una muestra de las parejas registradas.
6. Generar train, val y test usando `group_id`.
7. Guardar el manifiesto junto con los resultados del entrenamiento.

## 10. Criterios de finalización

La tarea estará completa cuando:

- Todos los videos BP y SQ tengan un `group_id`.
- Las parejas estén respaldadas por evidencia visual o metadata.
- Los videos no confirmados tengan grupos individuales.
- No existan IDs duplicados dentro del manifiesto.
- Ningún `group_id` esté repartido entre train, val y test.
- El split conserve una distribución razonable de etiquetas válidas e
  inválidas.

## 11. Recomendación final

No es necesario detener todos los experimentos hasta emparejar BP y SQ. Se
pueden entrenar modelos separados por vista usando grupos individuales. Sin
embargo, el manifiesto es obligatorio antes de entrenar modelos que combinen
left y right o antes de comparar resultados multivista como si cada archivo
fuera una observación independiente.

La regla metodológica que debe conservarse es:

> Ningún levantamiento físico conocido debe aparecer simultáneamente en los
> datos de entrenamiento y en los datos de evaluación.
