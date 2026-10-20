# Baseline V2 - 5-Fold Cross-Validation

CV usa solamente development (train+validation). Test no fue evaluado.

| Perfil | Accuracy CV | F1 CV | AUC CV |
|---|---:|---:|---:|
| dl_left | 0.8251 +/- 0.0517 | 0.8335 +/- 0.0508 | 0.8964 +/- 0.0353 |
| dl_right | 0.7971 +/- 0.0553 | 0.8218 +/- 0.0453 | 0.8639 +/- 0.0611 |
| dl_side | 0.7679 +/- 0.0558 | 0.7815 +/- 0.0602 | 0.8559 +/- 0.0439 |

## Test Baseline V2 guardado vs CV

| Perfil | Test Accuracy/F1/AUC | CV Accuracy/F1/AUC |
|---|---|---|
| dl_left | 0.7705 / 0.7667 / 0.8820 | 0.8251 +/- 0.0517 / 0.8335 +/- 0.0508 / 0.8964 +/- 0.0353 |
| dl_right | 0.8226 / 0.8308 / 0.8942 | 0.7971 +/- 0.0553 / 0.8218 +/- 0.0453 / 0.8639 +/- 0.0611 |
| dl_side | 0.7967 / 0.8120 / 0.8682 | 0.7679 +/- 0.0558 / 0.7815 +/- 0.0602 / 0.8559 +/- 0.0439 |

Los valores historicos se marcan PRE-BASELINE-V2; cambiaron mapping, folds y orden scaler/augmentation.
