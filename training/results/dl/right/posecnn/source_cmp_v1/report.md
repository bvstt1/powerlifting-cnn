# DL right: YOLO frente a MediaPipe (PoseCNN V3)

Ambas fuentes se reentrenaron con la misma cohorte, split por evento físico y folds V3.
El test se evaluó tras CV. Los resultados YOLO finales reproducen V3; los folds CV se reentrenaron con orden de muestras por split, por lo que su trayectoria no es idéntica a la ejecución CV V3 congelada.
El canal visibility de MediaPipe no equivale al confidence de YOLO.
Se conserva la convención de normalización V3 (índice 0 como hombro aunque ambos extractores guardan cadera primero).

Cohorte: {'train': {'n': 287, 'valid': 158}, 'val': {'n': 62, 'valid': 34}, 'test': {'n': 62, 'valid': 34}}; MediaPipe excluidos: 0.

| Partición | Fuente | N | Accuracy | F1 válido | AUC | Especificidad | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| cv_oof | yolo | 349 | 0.8252 | 0.8440 | 0.8721 | 0.7834 | 34 | 27 |
| cv_oof | mediapipe | 349 | 0.7679 | 0.8000 | 0.8484 | 0.6752 | 51 | 30 |

cv_oof: IC95% bootstrap pareado de Δ (MediaPipe−YOLO): {'accuracy': [-0.10028653295128942, -0.011461318051576016], 'f1': [-0.0820579230483803, -0.006006893837614868], 'roc_auc': [-0.06418432956467772, 0.017072550948173934], 'specificity': [-0.18243243243243235, -0.03378188581840267]}; McNemar exacto: {'only_yolo_correct': 41, 'only_mediapipe_correct': 21, 'exact_p': 0.015134080997447081}.

| test | yolo | 62 | 0.8226 | 0.8451 | 0.8645 | 0.7500 | 7 | 4 |
| test | mediapipe | 62 | 0.8065 | 0.8333 | 0.9065 | 0.7143 | 8 | 4 |

test: IC95% bootstrap pareado de Δ (MediaPipe−YOLO): {'accuracy': [-0.12903225806451613, 0.08064516129032262], 'f1': [-0.10855229531610065, 0.07490772087546271], 'roc_auc': [-0.04276513432876863, 0.13143187830687827], 'specificity': [-0.20000000000000007, 0.11764705882352944]}; McNemar exacto: {'only_yolo_correct': 6, 'only_mediapipe_correct': 5, 'exact_p': 1.0}.
