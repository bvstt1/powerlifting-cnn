# Segmentación de press de banca: mapa de datos

Esta carpeta contiene **salidas** de extracción y experimentos, no modelos
entrenados. El código que genera RAW y secuencias se encuentra en
[`extract_segmentation/bp/`](../../extract_segmentation/bp/); sus fases y
versiones están explicadas en
[`extract_segmentation/README.md`](../../extract_segmentation/README.md).

| Vista | Datos usados por los clasificadores | Evidencia de procedencia | Derivaciones y experimentos |
|---|---|---|---|
| [front](front/README.md) | `front/raw/` + `front/temporal_features_v1/sequences/` (38 canales) | `front/manifests/`, esquemas, cohorte e informes de features | `front/tracking_v2/` (derivación auditada); `front/tracking_v3/` y `front/pose_guided/` (experimentos) |
| [left](left/README.md) | `left/raw/` + `left/temporal_features_v1/sequences/` (62 canales) | `left/manifests/`, esquema, cohorte e informes | No hay una rama de tracking lateral dentro de esta vista |
| [right](right/README.md) | `right/raw/` + `right/temporal_features_v1/sequences/` (62 canales) | `right/manifests/`, esquema, cohorte e informes | No hay una rama de tracking lateral dentro de esta vista |

El reporte [`Reporte_BP_Laterales_Temporal_Features_Resumen.md`](Reporte_BP_Laterales_Temporal_Features_Resumen.md)
resume ambas vistas laterales. `previews/` contiene imágenes de inspección,
no ejemplos adicionales de entrenamiento. Los CSV de etiquetado vigentes
están en `etiquetado/bp/`, fuera de este directorio.

Los manifiestos incluyen hashes y rutas de los archivos que existían en
la máquina donde se produjeron los datos. **Mover salidas históricas cambia
esas rutas**, incluso cuando sus bytes permanecen iguales. Por ello, estos
índices distinguen su función sin reasignar archivos ni afirmar que los
resultados experimentales pasaron a formar parte de las cohortes entrenadas.
