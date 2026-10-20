# Baseline V3 - Post Manual Label Audit

## Test V2 vs V3

| Perfil | V2 F1 | V3 F1 | V2 AUC | V3 AUC |
|---|---:|---:|---:|---:|
| dl_left | 0.7667 | 0.8358 | 0.8820 | 0.8923 |
| dl_right | 0.8308 | 0.8451 | 0.8942 | 0.8645 |
| dl_side | 0.8120 | 0.8219 | 0.8682 | 0.8179 |

## CV V2 vs V3

| Perfil | V2 F1 | V3 F1 | V2 AUC | V3 AUC |
|---|---:|---:|---:|---:|
| dl_left | 0.8335 +/- 0.0508 | 0.8445 +/- 0.0395 | 0.8964 +/- 0.0353 | 0.8924 +/- 0.0403 |
| dl_right | 0.8218 +/- 0.0453 | 0.8399 +/- 0.0413 | 0.8639 +/- 0.0611 | 0.8727 +/- 0.0521 |
| dl_side | 0.7815 +/- 0.0602 | 0.8108 +/- 0.0446 | 0.8559 +/- 0.0439 | 0.8757 +/- 0.0476 |

La arquitectura no cambio. Las diferencias reflejan ground truth auditado, descarte del evento 302 y nuevos split/folds.
