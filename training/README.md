# Entrenamiento

Código y experimentos de clasificación para DL, BP y SQ. Ejecutar los módulos
desde la raíz del repositorio. Los reportes vigentes se encuentran junto a cada
experimento en `bp/experiments/` o en `results/<ejercicio>/<vista>/<modelo>/`.
La evidencia de rutas y hashes de la migración está en
[`docs/migration_artifact_manifest.json`](docs/migration_artifact_manifest.json).

## Estructura

- `common/`: arquitectura y utilidades compartidas (`train_utils.py`: PoseCNN,
  aumentos, scaler, métricas).
- `configs/`: configuración de ejercicios/vistas (`training_config.py`).
- `dl/`: pipeline y baselines de deadlift (`01_*.py`–`12_*.py`,
  `apply_manual_label_audit.py`, `export_discordant_labels.py`).
- `bp/`: press banca (`train_posecnn.py` para `bp_left`/`bp_right`,
  `bp_front_posecnn.py`, `bp_front_pose_fault_analysis.py`).
- `sq/`: sentadilla (`train_posecnn.py` para `sq_front`/`sq_left`/`sq_right`,
  `controlled_front.py` para experimentos A–D sin test,
  `run_analysis.py` para congelar/reconstruir referencias left v2,
  `select_front.py`, `diagnose_front.py`, `complete_report.py`;
  `controlled_left.py` para comparación de scaler SQ left v2 con
  `point_xy_v1`, `diagnose_left_invalid.py` y `report_controlled_left.py`;
  `controlled_right.py` audita vídeo/pose y compara scalers right con test reservado).
  `audit_right_cv_errors.py` compara exclusivamente predicciones externas de
  desarrollo, documenta errores por clase y exporta vídeos anotados para revisión.
- `tests/`: pruebas (`test_bp_side_posecnn.py`, `test_sq_posecnn.py`,
  `test_sq_controlled_front.py`,
  `test_event_mapping.py`).
- `docs/`: manifiesto de migración y trazabilidad de artefactos. Los reportes
  narrativos de BP están co-localizados con los experimentos históricos; los
  resultados de DL y SQ se publican en sus directorios de `results/`.
- `legacy/`: utilidades históricas (`fix_csv.py`).
- `bp/experiments/bp_*_v1/`: 20 experimentos BP migrados desde la raíz de
  `training/`. Cada directorio conserva scripts, manifiestos, checkpoints,
  métricas y reportes. Consultar el índice y la nota de hashes antes de repetir
  verificaciones de ejecuciones antiguas.

  Los quince experimentos monovista llevan numeración cronológica por vista:
  `bp_<vista>_<NN>_<tipo>_v1` con `NN` según el orden documentado de ejecución
  (`01` SegCNN cohorte, `02` PoseCNN cohorte común/compatibilidad, `03` fusión,
  `04` CV, `05` test final). Mapeo de nombres:
  | Antes | Ahora |
  |---|---|
  | `bp_front_segcnn_v1` | `bp_front_01_segcnn_v1` |
  | `bp_front_posecnn_common_v1` | `bp_front_02_posecnn_common_v1` |
  | `bp_front_fusion_v1` | `bp_front_03_fusion_v1` |
  | `bp_front_cross_validation_5fold_v1` | `bp_front_04_cross_validation_5fold_v1` |
  | `bp_front_final_test_v1` | `bp_front_05_final_test_v1` |
  | `bp_left_segcnn_v1` | `bp_left_01_segcnn_v1` |
  | `bp_left_posecnn_compatibility_v1` | `bp_left_02_posecnn_compatibility_v1` |
  | `bp_left_fusion_v1` | `bp_left_03_fusion_v1` |
  | `bp_left_cv_v1` | `bp_left_04_cv_v1` |
  | `bp_left_final_test_v1` | `bp_left_05_final_test_v1` |
  | `bp_right_segcnn_v1` | `bp_right_01_segcnn_v1` |
  | `bp_right_posecnn_common_v1` | `bp_right_02_posecnn_common_v1` |
  | `bp_right_fusion_v1` | `bp_right_03_fusion_v1` |
  | `bp_right_cv_v1` | `bp_right_04_cv_v1` |
  | `bp_right_final_test_v1` | `bp_right_05_final_test_v1` |

  Los paquetes compartidos por vista sin ejecución propia (`bp_side_segcnn_v1`,
  `bp_side_cv_v1`), el core común (`bp_posecnn_common_v1`) y el evaluador
  (`bp_lateral_final_test_v1`) quedan con el nombre original. Los reportes y manifiestos congelados citan rutas
  antiguas y **no se reescriben**: son evidencia de la ejecución previa. El
  mapeo está en `training/legacy/number_bp_experiments.py` y el verificador de
  hashes ya resuelve las rutas nuevas.

  Atribución de arquitectura: todas las métricas que se citan de estos
  experimentos pertenecen a la arquitectura que cada reporte indica, y las
  reportes de cada experimento identifican explícitamente la arquitectura y el
  protocolo de las métricas.
  Solamente PoseCNN tiene checkpoints en `training/models/` (más los
  históricos BP/SQ); los pesos SegCNN y de fusión de BP **no están en
  `models/`**: residen dentro de `bp/experiments/bp_front_01_segcnn_v1/`,
  `bp_left_01_segcnn_v1/`, `bp_right_01_segcnn_v1/` (SegCNN),
  `bp_left_03_fusion_v1/` y `bp_right_03_fusion_v1/` (cabezas), mientras la
  fusión tardía front (`late_sin`) no tiene checkpoint de cabeza, solo pesos
  de sus dos modelos base. Un resultado de test/CV reportado por un experimento
  SegCNN o de fusión **no es un resultado de PoseCNN** aunque comparta cohorte
  con ella.
- `data/`, `models/`, `results/`: artefactos por ejercicio/vista/ejecución.

La migración de los artefactos PoseCNN históricos BP front, DL y SQ está
registrada en `docs/migration_artifact_manifest.json`. SQ y BP lateral ya
seguían `posecnn/runs/`; DL conserva los nombres de V2/V3 y llama
`legacy_default` a los archivos que antes estaban directamente en la vista.
Los paquetes BP de segmentación/fusión/test en `bp/experiments/` son archivo
histórico co-localizado, no destino de nuevas salidas.

## Convención para trabajos nuevos

El código nuevo se coloca en `bp/`, `dl/` o `sq/` y reutiliza `common/` cuando
corresponda. Las entradas intermedias van a `data/<ejercicio>/<vista>/`, los
checkpoints y scalers a `models/<ejercicio>/<vista>/<modelo>/runs/<run-id>/`,
y métricas, predicciones y reportes a `results/<ejercicio>/<vista>/<modelo>/runs/<run-id>/`.
Cada ejecución debe conservar su configuración, split, hashes de entrada y
versión de código. No sobrescribir ejecuciones anteriores ni trasladar un
protocolo congelado: sus rutas y hashes forman parte de la evidencia.

Los directorios BP históricos se movieron a `bp/experiments/`; la convención
anterior aplica a trabajos nuevos. Algunos artefactos antiguos de `data/`, `models/` y `results/`
ya están registrados en Git; `.gitignore` impide incorporar otros por defecto
pero **no elimina los registrados**. Los reportes narrativos dentro de
`bp/experiments/*/` y `results/` sí pueden publicarse. Para reproducir
ejecuciones locales, conservar checkpoints, datos y manifiestos juntos; un
clon sin los artefactos ignorados no puede repetir todas las evaluaciones.

### Migración BP y hashes

Se trasladaron **los 20 directorios completos** `training/bp_*_v1/` a
`training/bp/experiments/bp_*_v1/` con
`python -B -m training.legacy.relocate_bp_experiments --apply`. La migración
comparó el SHA-256 de cada uno de los **949 archivos** antes y después del
movimiento: todos los bytes coincidían. Después se adaptaron 51 scripts en la
migración inicial y se ajustaron los imports locales para ejecutarlos como
módulos (rutas y raíces del proyecto actualizadas).
Los resultados y manifiestos históricos **no se reescribieron**.
Los cinco informes comparativos BP que estaban en la raíz se trasladaron a
`training/docs/bp/`; los scripts que los generan apuntan ahora allí.

Un hash SHA-256 es la huella de los **bytes de un archivo**, no de su nombre
ni de la carpeta donde está. Por ello, un checkpoint movido sin editar conserva
su hash. Un script editado para apuntar a la ruta nueva tiene un hash nuevo.
Además, un manifiesto que contiene el texto de la ruta antigua conserva su
propio hash, pero sigue señalando la ubicación anterior: su verificador
original puede fallar aunque los pesos sean idénticos. Los protocolos
congelados y sus hashes siguen siendo evidencia de la ejecución **previa a la
migración**; no son una verificación válida de una ejecución nueva con el código
reubicado. Para revalidar una ejecución futura hace falta un manifiesto nuevo,
nuevos hashes del código y un protocolo nuevo, sin sustituir los anteriores.

Ejemplo concreto: `bp_front_05_final_test_v1/bp_front_final_test_protocol_frozen.json`
registra el peso `pose_checkpoint` con su ruta antigua
`training/bp_front_posecnn_common_v1/seed_42/best_validation_loss.pt` y hash
`ed189755…d78a35c1`. El peso movido a `training/bp/experiments/` conserva ese
hash. El campo `evaluation_code` registra, en cambio, el hash del evaluador
**antes** de editar sus rutas: el evaluador migrado detecta el cambio y se
detiene antes de abrir Test. Eso demuestra que el protocolo no está siendo
suplantado por una versión diferente del programa.

Para comprobar pesos y scalers del protocolo sin ejecutar Test:
`python -B -m training.bp.experiments.verify_relocation`. No ejecutar el test
final antiguo como si fuese una evaluación nueva: exige el código congelado.

## Comandos (desde la raíz del repo)

```powershell
python -m training.tests.test_bp_side_posecnn
python -B -m training.tests.test_event_mapping
python -m training.dl.run_posecnn --profile dl_left --stage split
python -m training.bp.train_posecnn --view left --stage audit
python -m training.bp.train_posecnn --view right --stage audit
python -m training.bp.train_posecnn --view left --stage all --run-id baseline_t170_v1
python -m training.bp.train_posecnn --view right --stage all --run-id baseline_t170_v1
python -m training.sq.train_posecnn --view front --stage all --run-id baseline_t170_v1
python -m training.sq.train_posecnn --view left --stage all --run-id baseline_t170_v1
python -m training.bp.bp_front_posecnn --stage audit
python -m training.sq.run_analysis --stage freeze
python -m training.sq.run_analysis --stage left
python -m training.sq.controlled_front --variant A --stage prepare
python -m training.sq.controlled_front --variant A --stage cv
# Repetir A con B, C, D. El test no participa en ninguna de estas etapas.
python -m training.sq.select_front
python -m training.sq.controlled_front --variant D --stage fit-final
python -m training.sq.controlled_front --variant D --stage evaluate-test
python -m training.sq.complete_report
python -m training.sq.diagnose_left_invalid
python -m training.sq.controlled_left --stage prepare
python -m training.sq.controlled_left --stage cv
python -m training.sq.controlled_left --stage verify
python -m training.sq.controlled_left --stage select
python -m training.sq.controlled_left --stage fit-final
# Solo si point_xy_v1 quedó seleccionado; test exploratorio:
python -m training.sq.controlled_left --stage evaluate-test
python -m training.sq.report_controlled_left
python -m training.sq.controlled_right --stage audit
python -m training.sq.controlled_right --stage prepare
python -m training.sq.controlled_right --stage cv --variant temporal
python -m training.sq.controlled_right --stage cv --variant point_xy
python -m training.sq.controlled_right --stage select
python -m training.sq.controlled_right --stage fit-final
python -m training.sq.controlled_right --stage evaluate-test
python -m training.sq.controlled_right --stage report
python -m training.sq.controlled_right --stage verify
python -m training.sq.complete_report
python -m training.sq.audit_right_cv_errors --stage export
python -m training.sq.audit_right_cv_errors --stage visuals
python -m training.sq.audit_right_cv_errors --stage report
python -m training.sq.controlled_right --stage report
```

La prueba de mapping DL ejecuta siempre las aserciones de IDs; las dos
comprobaciones de longitud de keypoints se omiten con mensaje explícito si
faltan sus `.npy` locales (por ejemplo, `keypoints/dl/yolo/left/dl_309.npy`).

### DL lateral: comparación de fuente de keypoints (source_cmp_v1)

`training/dl/compare_keypoint_sources.py` reentrena PoseCNN V3 para left/right
con YOLO y MediaPipe sobre **los mismos vídeos por vista** (403 left, 411 right),
el split físico V3 y los folds físicos V3. Usa los arrays normalizados YOLO V3
locales, pues los `.npy` YOLO left pueden estar incompletos, y los `.npy`
MediaPipe laterales completos. Conserva el preprocesamiento V3, el scaler
ajustado exclusivamente en train original, los aumentos de train y la receta
PoseCNN V3; `visibility` MediaPipe y confianza YOLO son canales distintos.
El extractor guarda `[cadera, hombro]` en ambas fuentes, pero la normalización
V3 interpreta el índice 0 como hombro: se conserva esa convención heredada
para aislar el cambio de fuente. Los datos intermedios y manifiestos están en
`data/dl/<vista>/posecnn/source_cmp_v1/`; pesos y predicciones separados por
fuente bajo `models/` y `results/` con run-id `source_cmp_v1_<fuente>`.

```powershell
python -B -m training.dl.compare_keypoint_sources --stage prepare
python -B -m training.dl.compare_keypoint_sources --stage cv
python -B -m training.dl.compare_keypoint_sources --stage final
python -B -m training.dl.compare_keypoint_sources --stage compare
```

El script evita sobrescribir ejecuciones existentes. Reportes locales:
`results/dl/{left,right}/posecnn/source_cmp_v1/report.md` (junto a
`comparison.json`); los manifiestos incluyen hashes de inputs y del código
usado en el entrenamiento. La edición posterior del texto explicativo del
generador de reportes no altera las métricas ni el código de entrenamiento;
para reproducir *byte a byte* el hash de código en los manifiestos se necesita
la versión exacta utilizada originalmente. YOLO final reprodujo el test V3;
la CV YOLO se reentrenó con orden de muestras por split, por lo que sus curvas
no son necesariamente idénticas a la CV V3 congelada.

**Estado de las comparaciones por fuente:** DL left y right sí tienen cuatro
ejecuciones `source_cmp_v1_<fuente>` (CV y modelo final YOLO/MediaPipe). DL front
solo tiene el clasificador histórico MediaPipe; DL side solo YOLO. El módulo
`training/common/compare_keypoint_sources.py` implementa el protocolo para
BP/SQ front/left/right, pero en `training/results/bp/` y `training/results/sq/`
no hay resultados `source_cmp_v1` de MediaPipe. La presencia de archivos
`keypoints/*/mediapipe/` no significa que las PoseCNN de esas seis vistas
ya se hayan entrenado. Las métricas BP/SQ publicadas corresponden a sus
ejecuciones YOLO y no constituyen comparaciones pareadas de fuente.

**Trazabilidad del código DL:** los manifiestos y configuraciones
`source_cmp_v1` registraron SHA-256 `014d31e8…c4232eb` para el script durante
el entrenamiento. El script publicado tiene SHA-256 `8cadba6a…52442d37`:
después de entrenar se corrigió únicamente la frase explicativa del generador
de reportes sobre la cohorte V3. Las salidas numéricas y pesos no se
recalcularon; el hash almacenado documenta el estado de ejecución y no
coincide byte a byte con el archivo publicado. Los reportes por vista
documentan esta diferencia y la corrección textual.

### BP front: alcance de los reportes históricos de CV y Test

`bp/experiments/bp_front_04_cross_validation_5fold_v1/Reporte_BP_FRONT_CrossValidation_5Fold_V1.md`
dejó la columna «Test histórico (fuente ausente)» como N/D porque ese informe
no incorporó un reporte autorizado de Test. **No significa que no existan
resultados finales BP front:** `bp/experiments/bp_front_05_final_test_v1/
bp_front_final_test_metrics.json` y `bp_front_final_test_integrity.json`
registran una única evaluación de 100 vídeos sobre la cohorte común:
PoseCNN Acc/F1/AUC=0.8400/0.8667/0.9260; SegCNN SIN
0.8700/0.8976/0.9210; fusión tardía Pose+SIN
0.8400/0.8730/0.9260. `04` es CV sobre desarrollo y `05` evaluación de
checkpoints finales; no son los mismos modelos por fold ni deben rellenarse
retroactivamente los N/D del informe congelado.

Notas BP lateral: implementación compartida parametrizada por vista; `bp_left`
y `bp_right` tienen datos, scaler, particiones, checkpoints y resultados
propios. La preparación estable vive en
`training/data/bp/<view>/posecnn/`; cada ejecución usa
`training/models/bp/<view>/posecnn/runs/<run-id>/` y
`training/results/bp/<view>/posecnn/runs/<run-id>/`. `T=170` es el baseline
heredado de DL; la mediana BP es 79 frames, así que la mayoría de secuencias
se interpolan hacia arriba (limitación documentada, sin búsqueda de
longitudes en esta ejecución).
