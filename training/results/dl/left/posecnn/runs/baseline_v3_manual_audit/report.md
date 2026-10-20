# baseline_v3_manual_audit: Deadlift Left

- Experiment: `baseline_v3_manual_audit`
- Mapping: `physical_event_gap_310`
- Seed: 42
- Best epoch: 13
- Final epoch: 21
- Diagnostico curvas: NORMAL
- Divergencia sostenida aproximada: 14

| Split | Accuracy | Precision | Recall | F1 | ROC-AUC | Specificity | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation | 0.7833 | 0.8125 | 0.7879 | 0.8000 | 0.8631 | 0.7778 | 21 | 6 | 7 | 26 |
| Test | 0.8167 | 0.8235 | 0.8485 | 0.8358 | 0.8923 | 0.7778 | 21 | 6 | 5 | 28 |

## Comparacion Historica

| Perfil | F1 anterior | F1 Baseline V2 | AUC anterior | AUC Baseline V2 | Nota |
|---|---:|---:|---:|---:|---|
| dl_left | 0.7667 | 0.8358 | 0.8820 | 0.8923 | Baseline V2 |

La comparacion no aisla cambios de arquitectura: cambiaron el mapping, el split y el orden metodologico del pipeline.
