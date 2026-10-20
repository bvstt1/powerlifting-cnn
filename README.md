# Juez Virtual de Powerlifting

Proyecto de tesis para desarrollar un sistema de análisis de levantamientos de
powerlifting mediante visión por computador y aprendizaje profundo. El trabajo
cubre **sentadilla (SQ)**, **press de banca (BP)** y **peso muerto (DL)**,
utilizando grabaciones desde vistas frontal y laterales, extracción de pose,
etiquetado técnico y clasificación de levantamientos válidos/no válidos.

## Pipeline

1. Detección de pose y objetos con MediaPipe y modelos YOLO.
2. Extracción de secuencias de keypoints (`.npy`, normalmente `(T, N, 3)`).
3. Etiquetado manual y auditoría en CSV.
4. Limpieza, interpolación y normalización de secuencias.
5. Partición de datos agrupada por evento físico, escalado y aumento de train.
6. Entrenamiento y evaluación de PoseCNN y modelos experimentales adicionales.

## Organización

- `extract_keypoints/`, `extract_segmentation/`: extracción de pose, segmentación
  y seguimiento de barra.
- `keypoints/`, `dataset/`: datos derivados y vídeos locales; los vídeos y parte
  de los artefactos grandes no se distribuyen en Git.
- `etiquetado/`: CSV de etiquetas y respaldos históricos.
- `training/`: código, documentación y experimentos de clasificación. Consultar
  [`training/README.md`](training/README.md) para el índice detallado, comandos,
  reglas de reproducibilidad y layout vigente.
- `display/`, `live/`: visualizaciones, herramientas interactivas y pruebas con
  cámaras.
- `models/`, `runs/`: modelos de visión y ejecuciones YOLO.

Los experimentos nuevos de entrenamiento organizan entradas intermedias en
`training/data/<ejercicio>/<vista>/`, checkpoints y scalers en
`training/models/<ejercicio>/<vista>/<modelo>/runs/<run-id>/`, y resultados en
`training/results/<ejercicio>/<vista>/<modelo>/runs/<run-id>/`. No sobrescribir
ejecuciones previas: conservar configuración, particiones y trazabilidad de
entradas/código junto con cada experimento.

Los experimentos BP históricos están archivados en
`training/bp/experiments/`; sus manifiestos y reportes congelados documentan las
ejecuciones originales. El manifiesto de migración está en
[`training/docs/migration_artifact_manifest.json`](training/docs/migration_artifact_manifest.json).

## Empezar

Se requiere Python con las dependencias de visión y ML usadas por el proyecto
(entre ellas NumPy, OpenCV, MediaPipe, Ultralytics, PyTorch, SciPy y
scikit-learn). Aún no hay un `requirements.txt` o `pyproject.toml` canónico;
las versiones y los comandos específicos están documentados en
[`training/README.md`](training/README.md).

Ejecutar los comandos desde la raíz del repositorio. Ejemplos de comprobaciones:

```powershell
python -B -m training.tests.test_event_mapping
python -m training.tests.test_bp_side_posecnn
```

Para un experimento de entrenamiento, consultar primero el
[`README de training`](training/README.md) y el reporte/configuración del
experimento o ejecución correspondiente.
Los datos locales ignorados por Git pueden ser necesarios para ejecutar el
pipeline completo.

## Estado y documentación

Los pipelines de entrenamiento y estudios de BP, SQ y DL están en desarrollo
experimental. Las métricas deben leerse junto a su protocolo, cohorte y
partición: resultados de distintas cohortes o arquitecturas no son directamente
comparables. El inventario y los informes vigentes se mantienen en
`training/docs/` y en los directorios de experimentos.

El índice de comandos y estructura de entrenamiento está en
[`training/README.md`](training/README.md); los resultados y protocolos se
documentan junto a cada ejecución.
