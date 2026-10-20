# Investigación posterior del tracking de barra V3

V3 trabaja sobre RAW/tracking V2.2 ya extraídos; no entrena una CNN ni
reemplaza el extractor frontal vigente. El código fuente está agrupado
físicamente aquí y `run.py` da acceso a sus etapas:

| Etapa | Script real | Propósito |
|---|---|---|
| `prototype` | `rebuild_bar_tracking_v3_from_raw.py` + `bar_tracking_v3_utils.py` | Probar rutas de barra en `tracking_v3/test_14/`. |
| `test` | `test_bar_tracking_v3_prototype.py` | Comparar los 14 casos con los resultados esperados. |
| `validate` | `validate_bar_tracking_v3_76.py` | Evaluación posterior sobre los 76 casos seleccionados. |
| `verify` | `verify_bar_tracking_v3_validation76.py` | Verificar la configuración y los artefactos históricos, y distinguir los hashes del código reubicado. |

```powershell
python extract_segmentation/bp/tracking_front_v3/run.py prototype --help
python extract_segmentation/bp/tracking_front_v3/run.py verify
```

La evaluación original registró las rutas anteriores y los hashes originales
en `segmentation/bp/front/tracking_v3/validation_76/manifests/`. Ambos juegos
de hashes están separados en `code_versions.py`: los históricos se comprueban
contra los manifiestos; los del código reorganizado comprueban este árbol de
trabajo. **No se afirma que sean idénticos byte a byte ni que el código nuevo
produjera las métricas originales.** Para reproducir exactamente la ejecución
original se utiliza el commit original de Git. `verify` es de solo lectura
por defecto; la evaluación histórica documentó dos *false merges*, por lo que
su estado no es `APPROVED`, independientemente de esta reorganización.

Los estudios V3.1 pose-guided están en `segmentation/bp/front/` y continúan
siendo experimentales. Algunos snapshots V3.1 registran las rutas antiguas
de los scripts V3: son evidencia de sus ejecuciones históricas, no rutas de
esta distribución.
