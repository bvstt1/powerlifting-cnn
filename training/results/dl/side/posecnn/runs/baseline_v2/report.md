# Baseline V2: Deadlift Lateral

- Experiment: `baseline_v2`
- Mapping: `physical_event_gap_310`
- Seed: 42
- Best epoch: 10
- Final epoch: 18
- Diagnostico curvas: OVERFITTING
- Divergencia sostenida aproximada: 11

| Split | Accuracy | Precision | Recall | F1 | ROC-AUC | Specificity | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation | 0.7951 | 0.8088 | 0.8209 | 0.8148 | 0.8855 | 0.7636 | 42 | 13 | 12 | 55 |
| Test | 0.7967 | 0.8308 | 0.7941 | 0.8120 | 0.8682 | 0.8000 | 44 | 11 | 14 | 54 |

## Comparacion Historica

| Perfil | F1 anterior | F1 Baseline V2 | AUC anterior | AUC Baseline V2 | Nota |
|---|---:|---:|---:|---:|---|
| dl_side | 0.8219 | 0.8120 | 0.8398 | 0.8682 | PRE-MAPPING-FIX |

La comparacion no aisla cambios de arquitectura: cambiaron el mapping, el split y el orden metodologico del pipeline.
