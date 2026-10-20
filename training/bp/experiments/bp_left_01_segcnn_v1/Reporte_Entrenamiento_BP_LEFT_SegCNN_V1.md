# Entrenamiento BP_LEFT SegCNN V1 — Validation exclusivamente

**LEFT_SEGCNN_TRAINED_VALIDATION_COMPLETE.** Ocho ejecuciones totales en el proyecto (cuatro en esta vista); ningún NPZ Test abierto ni evaluado. Clase positiva **1=válido**; FP=levantamiento inválido predicho válido. Threshold fijo 0,5.

## Cohorte y trazabilidad

- Train **422**; Validation **91**; split Test registrado únicamente por CSV de cohortes, sin abrir sus secuencias. `source_manifest.json` guarda SHA-256 de split/labels/cohorte/esquema/config y hashes **solo de secuencias Train/Validation**, así como versión de arquitectura. Identidad física entre vistas no verificada; no se usó ninguna secuencia RIGHT en esta vista.
- `1=válido, 0=inválido`: Train {0: 218, 1: 204}, Validation {1: 44, 0: 47}. Sin split fuera de Train/Validation. Histórico PoseCNN: {'historical_split_matches_current_exclusions': {'train': [], 'validation': [], 'test': []}, 'pose_checkpoint_path': 'C:\\Users\\basti\\MediapipePythonProjects\\training\\models\\bp\\left\\posecnn\\runs\\baseline_t170_v1\\best_model.pth', 'pose_checkpoint_exists': True, 'right_historical_train_includes_bp_354': False}.
- El Test lateral ya se evaluó históricamente para PoseCNN, por lo que no es inéditamente ciego para *todo* el proyecto. Sus resultados no se consultaron para entrenar ni seleccionar SegCNN. Si se comparará/fusionará con PoseCNN, RIGHT requiere un nuevo baseline PoseCNN en cohorte común sin `bp_354`; los splits restantes coinciden documentalmente. LEFT comparte las particiones actuales de la PoseCNN histórica.

## Protocolo congelado

- Grid idéntico a BP lateral PoseCNN: `linspace(0,T_original-1,170)`; QUALITY/conteos se toman del frame próximo. Variables continuas se interpolan solo entre frames contiguos observados con misma procedencia. BAR requiere asociación local corroborada; selección no verificada usa frame próximo sin crear velocidad. Relación mantiene estado directo/proyección ordinaria/lejana y no cruza cambios; ausencia/ambigüedad deja NaN y máscara 0. No hay tracking persistente.
- Imputación mediana y media/desviación de **Train observado remuestreado**; mismo transform para Validation. `channels_and_train_preprocessing.json` conserva índices, nombres, constantes detectadas únicamente en Train y estadísticas por canal. Cada valor retiene máscara y codificación categórica de procedencia; se descartan únicamente canales constantes en Train. Son señales parcialmente pose-assisted, incluso la variante sin relación.
- Ablación definida antes de entrenar: elimina **19** features: bar_proximal_thickness_h, bar_proximal_x_w, bar_proximal_y_h, bar_visible_length_relation_w, extrap_distance_w, extrap_ratio_d_l, q_bench_reference_used, q_pose_torso_reliable, q_profile_bench_estimated, q_profile_observed, q_projection_far, q_projection_ordinary, q_relation_direct, q_relation_unevaluable, sep_delta_h_s, sep_signed_h, torso_profile_coverage, torso_profile_x_w, torso_profile_y_h. Retira perfil de torso y extremo proximal porque dependen de la región de torso; conserva BAR, PERSON y BENCH generales, muñeca–BAR y dinámica general. Retira también sus máscaras/procedencia al generar entradas. FULL C=138 (65,889 parámetros); sin relación C=88 (57,889 parámetros). Arquitectura FRONT reproducida: Conv1D 5 (padding 2) ×32/64, MaxPool2, Dropout .20, Conv1D 3 (padding 1) ×128, GAP, MLP 128→64→1 Dropout .40, logits y BCEWithLogitsLoss.
- Dos inicializaciones independientes: seeds 42/1337; AdamW 1e-3, decay 1e-4, batch 32, ReduceLROnPlateau patience 5/factor .5, early stopping patience 15, hasta 200 epochs. Mejor época y semilla por **menor loss Validation**, sin optimizar threshold.

## Resultados por semilla (una predicción por levantamiento)

| Variante | Seed | C | Parámetros | Época | Loss Train / Val | Acc | Prec | Recall/sens válido | F1 | Macro-F1 | Bal Acc | ROC-AUC | AP/PR-AUC | TN/FP/FN/TP |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| full | 42 | 138 | 65,889 | 50 | 0.0615 / 0.1041 | 0.9451 | 0.9333 | 0.9545 | 0.9438 | 0.9450 | 0.9454 | 0.9937 | 0.9943 | 44/3/2/42 |
| full | 1337 | 138 | 65,889 | 23 | 0.1281 / 0.1301 | 0.9451 | 0.9535 | 0.9318 | 0.9425 | 0.9449 | 0.9446 | 0.9898 | 0.9901 | 45/2/3/41 |
| sin_relacion | 42 | 88 | 57,889 | 48 | 0.0600 / 0.1338 | 0.9451 | 0.9333 | 0.9545 | 0.9438 | 0.9450 | 0.9454 | 0.9923 | 0.9923 | 44/3/2/42 |
| sin_relacion | 1337 | 88 | 57,889 | 28 | 0.1130 / 0.1685 | 0.9451 | 0.9535 | 0.9318 | 0.9425 | 0.9449 | 0.9446 | 0.9802 | 0.9815 | 45/2/3/41 |

Especificidad de inválidos = TN/(TN+FP); sensibilidad de válidos = TP/(TP+FN), en `metrics.json` por seed. FP por variante:
- full seed 42: **3 FP**, IDs ['bp_240', 'bp_241', 'bp_489'].
- full seed 1337: **2 FP**, IDs ['bp_436', 'bp_489'].
- sin_relacion seed 42: **3 FP**, IDs ['bp_240', 'bp_241', 'bp_436'].
- sin_relacion seed 1337: **2 FP**, IDs ['bp_436', 'bp_482'].

Checkpoints elegidos por variante: FULL **seed 42** (`full\seed_42\best_validation_loss.pt`, SHA 1d4ea648af291f7109244bf226b6cadbcbf846d17c13b684db2136a7d00986c1); sin relación **seed 42** (`sin_relacion\seed_42\best_validation_loss.pt`, SHA 190b34a78dd3646df2fd828357dd7377ba1afefd2b8d70677dbb545e02e51ece). Candidato preliminar mínimo loss Validation en esta vista: **full seed 42**, loss 0.1041, F1 0.9438. Es preliminar, no resultado Test.

## Comparación pareada y proyecciones

FULL menos sin relación en la misma semilla (loss menor favorece FULL): [(42, -0.0297, 0.0), (1337, -0.0384, 0.0)] como `(seed, Δloss, ΔF1)`. Patrón de loss FULL: **mejora** frente a ablación en las dos semillas; tamaño de entrada y optimización también cambian, por lo que no se atribuye causalidad a las proyecciones. Pareo de checkpoints seleccionados (pueden ser seeds diferentes): **85** ambos correctos, **4** ambos erróneos, **1** solo FULL errado, **1** solo sin relación errado, **2** predicciones discordantes. `validation_paired_seed_42.csv`, `validation_paired_seed_1337.csv`, `validation_paired_selected.csv` y `paired_comparison_validation.json` contienen los casos y probabilidades.

La mayoría de las separaciones originales son proyecciones 2D, no observaciones físicas de contacto. Los subgrupos siguientes son **descriptivos en Validation** (umbrales fijos antes del análisis: ≥50% BAR corroborada, ≥80% proyecciones, <20% velocidad); no modifican ningún modelo:

| Variante | Subgrupo | N videos | Errores | FP | FN |
|---|---|---:|---:|---:|---:|
| full seed 42 | bar_corrob_ge50pct | 62 | 5 | 3 | 2 |
| full seed 42 | bar_corrob_lt50pct | 29 | 0 | 0 | 0 |
| full seed 42 | bar_ambiguous_any | 37 | 1 | 1 | 0 |
| full seed 42 | bar_unverified_any | 35 | 0 | 0 | 0 |
| full seed 42 | projection_ge80pct | 76 | 5 | 3 | 2 |
| full seed 42 | projection_lt80pct | 15 | 0 | 0 | 0 |
| full seed 42 | far_projection_any | 0 | 0 | 0 | 0 |
| full seed 42 | bench_missing_any | 0 | 0 | 0 | 0 |
| full seed 42 | person_missing_any | 3 | 0 | 0 | 0 |
| full seed 42 | velocity_available_lt20pct | 26 | 0 | 0 | 0 |
| sin_relacion seed 42 | bar_corrob_ge50pct | 62 | 5 | 3 | 2 |
| sin_relacion seed 42 | bar_corrob_lt50pct | 29 | 0 | 0 | 0 |
| sin_relacion seed 42 | bar_ambiguous_any | 37 | 1 | 1 | 0 |
| sin_relacion seed 42 | bar_unverified_any | 35 | 1 | 1 | 0 |
| sin_relacion seed 42 | projection_ge80pct | 76 | 5 | 3 | 2 |
| sin_relacion seed 42 | projection_lt80pct | 15 | 0 | 0 | 0 |
| sin_relacion seed 42 | far_projection_any | 0 | 0 | 0 | 0 |
| sin_relacion seed 42 | bench_missing_any | 0 | 0 | 0 | 0 |
| sin_relacion seed 42 | person_missing_any | 3 | 0 | 0 | 0 |
| sin_relacion seed 42 | velocity_available_lt20pct | 26 | 0 | 0 | 0 |

`validation_source_quality.csv` conserva tasas por video; por seed `validation_quality_subgroups.csv` y `validation_fp_ids.json`. Un N bajo o vacío no autoriza conclusiones sobre ese estrato.

## Generalización y siguiente paso

- full seed 42: train F1 0.9780 vs Validation F1 0.9438; train loss 0.0615 vs Validation loss 0.1041 (brecha +0.0426).
- full seed 1337: train F1 0.9313 vs Validation F1 0.9425; train loss 0.1281 vs Validation loss 0.1301 (brecha +0.0021).
- sin_relacion seed 42: train F1 0.9728 vs Validation F1 0.9438; train loss 0.0600 vs Validation loss 0.1338 (brecha +0.0738).
- sin_relacion seed 1337: train F1 0.9572 vs Validation F1 0.9425; train loss 0.1130 vs Validation loss 0.1685 (brecha +0.0555).

Las curvas por seed, matrices de confusión y probabilidades están en `full/seed_*/` y `sin_relacion/seed_*/`. Para una futura fusión, alinear los videos de la cohorte monovista, usar predicciones fuera de muestra en desarrollo y no usar Test para calibrarla. En RIGHT, reentrenar PoseCNN en la cohorte común sin duplicado antes de afirmar comparabilidad/fusión; no comparar métricas históricas de particiones distintas como equivalentes. No se entrenaron PoseCNN ni fusiones aquí.
