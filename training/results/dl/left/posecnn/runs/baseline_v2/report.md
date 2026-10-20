# Baseline V2: Deadlift Left

- Experiment: `baseline_v2`
- Mapping: `physical_event_gap_310`
- Seed: 42
- Best epoch: 16
- Final epoch: 24
- Diagnostico curvas: OVERFITTING
- Divergencia sostenida aproximada: 17 (con oscilaciones transitorias previas)

| Split | Accuracy | Precision | Recall | F1 | ROC-AUC | Specificity | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation | 0.7833 | 0.7941 | 0.8182 | 0.8060 | 0.8833 | 0.7407 | 20 | 7 | 6 | 27 |
| Test | 0.7705 | 0.8519 | 0.6970 | 0.7667 | 0.8820 | 0.8571 | 24 | 4 | 10 | 23 |

## Comparacion Historica

| Perfil | F1 anterior | F1 Baseline V2 | AUC anterior | AUC Baseline V2 | Nota |
|---|---:|---:|---:|---:|---|
| dl_left | 0.8438 | 0.7667 | 0.9134 | 0.8820 | historico |

La comparacion no aisla cambios de arquitectura: cambiaron el mapping, el split y el orden metodologico del pipeline.
