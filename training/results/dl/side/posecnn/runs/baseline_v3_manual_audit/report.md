# baseline_v3_manual_audit: Deadlift Lateral

- Experiment: `baseline_v3_manual_audit`
- Mapping: `physical_event_gap_310`
- Seed: 42
- Best epoch: 7
- Final epoch: 15
- Diagnostico curvas: POSIBLE OVERFITTING
- Divergencia sostenida aproximada: 8

| Split | Accuracy | Precision | Recall | F1 | ROC-AUC | Specificity | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation | 0.7213 | 0.7391 | 0.7612 | 0.7500 | 0.8133 | 0.6727 | 37 | 18 | 16 | 51 |
| Test | 0.7869 | 0.7595 | 0.8955 | 0.8219 | 0.8179 | 0.6545 | 36 | 19 | 7 | 60 |

## Comparacion Historica

| Perfil | F1 anterior | F1 Baseline V2 | AUC anterior | AUC Baseline V2 | Nota |
|---|---:|---:|---:|---:|---|
| dl_side | 0.8120 | 0.8219 | 0.8682 | 0.8179 | Baseline V2 |

La comparacion no aisla cambios de arquitectura: cambiaron el mapping, el split y el orden metodologico del pipeline.
