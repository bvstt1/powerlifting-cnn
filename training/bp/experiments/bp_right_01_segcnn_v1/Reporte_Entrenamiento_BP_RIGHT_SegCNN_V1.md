# Entrenamiento BP_RIGHT SegCNN V1 — Validation exclusivamente

**RIGHT_SEGCNN_TRAINED_VALIDATION_COMPLETE.** Ocho ejecuciones totales en el proyecto (cuatro en esta vista); ningún NPZ Test abierto ni evaluado. Clase positiva **1=válido**; FP=levantamiento inválido predicho válido. Threshold fijo 0,5.

## Cohorte y trazabilidad

- Train **431**; Validation **93**; split Test registrado únicamente por CSV de cohortes, sin abrir sus secuencias. `source_manifest.json` guarda SHA-256 de split/labels/cohorte/esquema/config y hashes **solo de secuencias Train/Validation**, así como versión de arquitectura. Identidad física entre vistas no verificada; no se usó ninguna secuencia LEFT en esta vista.
- `1=válido, 0=inválido`: Train {1: 190, 0: 241}, Validation {0: 52, 1: 41}. Sin split fuera de Train/Validation. Histórico PoseCNN: {'historical_split_matches_current_exclusions': {'train': ['bp_354'], 'validation': [], 'test': []}, 'pose_checkpoint_path': 'C:\\Users\\basti\\MediapipePythonProjects\\training\\models\\bp\\right\\posecnn\\runs\\baseline_t170_v1\\best_model.pth', 'pose_checkpoint_exists': True, 'right_historical_train_includes_bp_354': True}.
- El Test lateral ya se evaluó históricamente para PoseCNN, por lo que no es inéditamente ciego para *todo* el proyecto. Sus resultados no se consultaron para entrenar ni seleccionar SegCNN. Si se comparará/fusionará con PoseCNN, RIGHT requiere un nuevo baseline PoseCNN en cohorte común sin `bp_354`; los splits restantes coinciden documentalmente. LEFT comparte las particiones actuales de la PoseCNN histórica.

## Protocolo congelado

- Grid idéntico a BP lateral PoseCNN: `linspace(0,T_original-1,170)`; QUALITY/conteos se toman del frame próximo. Variables continuas se interpolan solo entre frames contiguos observados con misma procedencia. BAR requiere asociación local corroborada; selección no verificada usa frame próximo sin crear velocidad. Relación mantiene estado directo/proyección ordinaria/lejana y no cruza cambios; ausencia/ambigüedad deja NaN y máscara 0. No hay tracking persistente.
- Imputación mediana y media/desviación de **Train observado remuestreado**; mismo transform para Validation. `channels_and_train_preprocessing.json` conserva índices, nombres, constantes detectadas únicamente en Train y estadísticas por canal. Cada valor retiene máscara y codificación categórica de procedencia; se descartan únicamente canales constantes en Train. Son señales parcialmente pose-assisted, incluso la variante sin relación.
- Ablación definida antes de entrenar: elimina **19** features: bar_proximal_thickness_h, bar_proximal_x_w, bar_proximal_y_h, bar_visible_length_relation_w, extrap_distance_w, extrap_ratio_d_l, q_bench_reference_used, q_pose_torso_reliable, q_profile_bench_estimated, q_profile_observed, q_projection_far, q_projection_ordinary, q_relation_direct, q_relation_unevaluable, sep_delta_h_s, sep_signed_h, torso_profile_coverage, torso_profile_x_w, torso_profile_y_h. Retira perfil de torso y extremo proximal porque dependen de la región de torso; conserva BAR, PERSON y BENCH generales, muñeca–BAR y dinámica general. Retira también sus máscaras/procedencia al generar entradas. FULL C=144 (66,849 parámetros); sin relación C=90 (58,209 parámetros). Arquitectura FRONT reproducida: Conv1D 5 (padding 2) ×32/64, MaxPool2, Dropout .20, Conv1D 3 (padding 1) ×128, GAP, MLP 128→64→1 Dropout .40, logits y BCEWithLogitsLoss.
- Dos inicializaciones independientes: seeds 42/1337; AdamW 1e-3, decay 1e-4, batch 32, ReduceLROnPlateau patience 5/factor .5, early stopping patience 15, hasta 200 epochs. Mejor época y semilla por **menor loss Validation**, sin optimizar threshold.

## Resultados por semilla (una predicción por levantamiento)

| Variante | Seed | C | Parámetros | Época | Loss Train / Val | Acc | Prec | Recall/sens válido | F1 | Macro-F1 | Bal Acc | ROC-AUC | AP/PR-AUC | TN/FP/FN/TP |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| full | 42 | 144 | 66,849 | 2 | 0.2601 / 0.3188 | 0.8817 | 0.8947 | 0.8293 | 0.8608 | 0.8790 | 0.8762 | 0.9369 | 0.9144 | 48/4/7/34 |
| full | 1337 | 144 | 66,849 | 6 | 0.2021 / 0.3278 | 0.8817 | 0.8409 | 0.9024 | 0.8706 | 0.8808 | 0.8839 | 0.9315 | 0.9124 | 45/7/4/37 |
| sin_relacion | 42 | 90 | 58,209 | 5 | 0.1776 / 0.3091 | 0.9140 | 0.9024 | 0.9024 | 0.9024 | 0.9128 | 0.9128 | 0.9371 | 0.8804 | 48/4/4/37 |
| sin_relacion | 1337 | 90 | 58,209 | 6 | 0.1929 / 0.3030 | 0.9247 | 0.9250 | 0.9024 | 0.9136 | 0.9235 | 0.9224 | 0.9402 | 0.9427 | 49/3/4/37 |

Especificidad de inválidos = TN/(TN+FP); sensibilidad de válidos = TP/(TP+FN), en `metrics.json` por seed. FP por variante:
- full seed 42: **4 FP**, IDs ['bp_001', 'bp_384', 'bp_441', 'bp_456'].
- full seed 1337: **7 FP**, IDs ['bp_001', 'bp_384', 'bp_387', 'bp_441', 'bp_460', 'bp_578', 'bp_585'].
- sin_relacion seed 42: **4 FP**, IDs ['bp_001', 'bp_384', 'bp_387', 'bp_460'].
- sin_relacion seed 1337: **3 FP**, IDs ['bp_001', 'bp_384', 'bp_387'].

Checkpoints elegidos por variante: FULL **seed 42** (`full\seed_42\best_validation_loss.pt`, SHA 09efb59b197897482218af5407ec0652e05ed0ec35feb2f5f306799551b7f187); sin relación **seed 1337** (`sin_relacion\seed_1337\best_validation_loss.pt`, SHA 28d1ddc1a1408d88bb7617b97d9cb7f86ef7b783fa735fd1d8326ee7ac25458b). Candidato preliminar mínimo loss Validation en esta vista: **sin_relacion seed 1337**, loss 0.3030, F1 0.9136. Es preliminar, no resultado Test.

## Comparación pareada y proyecciones

FULL menos sin relación en la misma semilla (loss menor favorece FULL): [(42, 0.0096, -0.0417), (1337, 0.0248, -0.043)] como `(seed, Δloss, ΔF1)`. Patrón de loss FULL: **deterioro** frente a ablación en las dos semillas; tamaño de entrada y optimización también cambian, por lo que no se atribuye causalidad a las proyecciones. Pareo de checkpoints seleccionados (pueden ser seeds diferentes): **80** ambos correctos, **5** ambos erróneos, **6** solo FULL errado, **2** solo sin relación errado, **8** predicciones discordantes. `validation_paired_seed_42.csv`, `validation_paired_seed_1337.csv`, `validation_paired_selected.csv` y `paired_comparison_validation.json` contienen los casos y probabilidades.

La mayoría de las separaciones originales son proyecciones 2D, no observaciones físicas de contacto. Los subgrupos siguientes son **descriptivos en Validation** (umbrales fijos antes del análisis: ≥50% BAR corroborada, ≥80% proyecciones, <20% velocidad); no modifican ningún modelo:

| Variante | Subgrupo | N videos | Errores | FP | FN |
|---|---|---:|---:|---:|---:|
| full seed 42 | bar_corrob_ge50pct | 51 | 7 | 1 | 6 |
| full seed 42 | bar_corrob_lt50pct | 42 | 4 | 3 | 1 |
| full seed 42 | bar_ambiguous_any | 13 | 4 | 3 | 1 |
| full seed 42 | bar_unverified_any | 42 | 4 | 3 | 1 |
| full seed 42 | projection_ge80pct | 85 | 9 | 2 | 7 |
| full seed 42 | projection_lt80pct | 8 | 2 | 2 | 0 |
| full seed 42 | far_projection_any | 5 | 0 | 0 | 0 |
| full seed 42 | bench_missing_any | 19 | 0 | 0 | 0 |
| full seed 42 | person_missing_any | 0 | 0 | 0 | 0 |
| full seed 42 | velocity_available_lt20pct | 42 | 4 | 3 | 1 |
| sin_relacion seed 1337 | bar_corrob_ge50pct | 51 | 4 | 2 | 2 |
| sin_relacion seed 1337 | bar_corrob_lt50pct | 42 | 3 | 1 | 2 |
| sin_relacion seed 1337 | bar_ambiguous_any | 13 | 2 | 1 | 1 |
| sin_relacion seed 1337 | bar_unverified_any | 42 | 3 | 1 | 2 |
| sin_relacion seed 1337 | projection_ge80pct | 85 | 6 | 2 | 4 |
| sin_relacion seed 1337 | projection_lt80pct | 8 | 1 | 1 | 0 |
| sin_relacion seed 1337 | far_projection_any | 5 | 0 | 0 | 0 |
| sin_relacion seed 1337 | bench_missing_any | 19 | 0 | 0 | 0 |
| sin_relacion seed 1337 | person_missing_any | 0 | 0 | 0 | 0 |
| sin_relacion seed 1337 | velocity_available_lt20pct | 42 | 3 | 1 | 2 |

`validation_source_quality.csv` conserva tasas por video; por seed `validation_quality_subgroups.csv` y `validation_fp_ids.json`. Un N bajo o vacío no autoriza conclusiones sobre ese estrato.

## Generalización y siguiente paso

- full seed 42: train F1 0.8992 vs Validation F1 0.8608; train loss 0.2601 vs Validation loss 0.3188 (brecha +0.0586).
- full seed 1337: train F1 0.9077 vs Validation F1 0.8706; train loss 0.2021 vs Validation loss 0.3278 (brecha +0.1256).
- sin_relacion seed 42: train F1 0.9337 vs Validation F1 0.9024; train loss 0.1776 vs Validation loss 0.3091 (brecha +0.1315).
- sin_relacion seed 1337: train F1 0.9340 vs Validation F1 0.9136; train loss 0.1929 vs Validation loss 0.3030 (brecha +0.1101).

Las curvas por seed, matrices de confusión y probabilidades están en `full/seed_*/` y `sin_relacion/seed_*/`. Para una futura fusión, alinear los videos de la cohorte monovista, usar predicciones fuera de muestra en desarrollo y no usar Test para calibrarla. En RIGHT, reentrenar PoseCNN en la cohorte común sin duplicado antes de afirmar comparabilidad/fusión; no comparar métricas históricas de particiones distintas como equivalentes. No se entrenaron PoseCNN ni fusiones aquí.
