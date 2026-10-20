# baseline_v3_manual_audit: Deadlift Right

- Experiment: `baseline_v3_manual_audit`
- Mapping: `physical_event_gap_310`
- Seed: 42
- Best epoch: 13
- Final epoch: 21
- Diagnostico curvas: NORMAL
- Divergencia sostenida aproximada: 14

| Split | Accuracy | Precision | Recall | F1 | ROC-AUC | Specificity | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation | 0.8226 | 0.8286 | 0.8529 | 0.8406 | 0.8834 | 0.7857 | 22 | 6 | 5 | 29 |
| Test | 0.8226 | 0.8108 | 0.8824 | 0.8451 | 0.8645 | 0.7500 | 21 | 7 | 4 | 30 |

## Comparacion Historica

| Perfil | F1 anterior | F1 Baseline V2 | AUC anterior | AUC Baseline V2 | Nota |
|---|---:|---:|---:|---:|---|
| dl_right | 0.8308 | 0.8451 | 0.8942 | 0.8645 | Baseline V2 |

La comparacion no aisla cambios de arquitectura: cambiaron el mapping, el split y el orden metodologico del pipeline.
