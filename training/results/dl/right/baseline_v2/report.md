# Baseline V2: Deadlift Right

- Experiment: `baseline_v2`
- Mapping: `physical_event_gap_310`
- Seed: 42
- Best epoch: 19
- Final epoch: 27
- Diagnostico curvas: OVERFITTING
- Divergencia sostenida aproximada: 20 (curva de validation ruidosa)

| Split | Accuracy | Precision | Recall | F1 | ROC-AUC | Specificity | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation | 0.8065 | 0.8056 | 0.8529 | 0.8286 | 0.9454 | 0.7500 | 21 | 7 | 5 | 29 |
| Test | 0.8226 | 0.9000 | 0.7714 | 0.8308 | 0.8942 | 0.8889 | 24 | 3 | 8 | 27 |

## Comparacion Historica

| Perfil | F1 anterior | F1 Baseline V2 | AUC anterior | AUC Baseline V2 | Nota |
|---|---:|---:|---:|---:|---|
| dl_right | 0.7576 | 0.8308 | 0.7979 | 0.8942 | historico |

La comparacion no aisla cambios de arquitectura: cambiaron el mapping, el split y el orden metodologico del pipeline.
