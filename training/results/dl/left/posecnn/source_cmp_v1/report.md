# DL left: YOLO frente a MediaPipe (PoseCNN V3)

Ambas fuentes se reentrenaron con la misma cohorte, split por evento físico y folds V3.
El test se evaluó tras CV. Los resultados YOLO finales reproducen V3; los folds CV se reentrenaron con orden de muestras por split, por lo que su trayectoria no es idéntica a la ejecución CV V3 congelada.
El canal visibility de MediaPipe no equivale al confidence de YOLO.
Se conserva la convención de normalización V3 (índice 0 como hombro aunque ambos extractores guardan cadera primero).

Cohorte: {'train': {'n': 283, 'valid': 156}, 'val': {'n': 60, 'valid': 33}, 'test': {'n': 60, 'valid': 33}}; MediaPipe excluidos: 0.

| Partición | Fuente | N | Accuracy | F1 válido | AUC | Especificidad | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| cv_oof | yolo | 343 | 0.8047 | 0.8194 | 0.8907 | 0.8052 | 30 | 37 |
| cv_oof | mediapipe | 343 | 0.8134 | 0.8367 | 0.8716 | 0.7468 | 39 | 25 |

cv_oof: IC95% bootstrap pareado de Δ (MediaPipe−YOLO): {'accuracy': [-0.03790087463556857, 0.058309037900874605], 'f1': [-0.02446091207511993, 0.06264372299221181], 'roc_auc': [-0.06060701331897557, 0.022958991077556685], 'specificity': [-0.14102564102564108, 0.02517346317711494]}; McNemar exacto: {'only_yolo_correct': 38, 'only_mediapipe_correct': 41, 'exact_p': 0.8221442424521863}.

| test | yolo | 60 | 0.8167 | 0.8358 | 0.8923 | 0.7778 | 6 | 5 |
| test | mediapipe | 60 | 0.7667 | 0.7812 | 0.7980 | 0.7778 | 6 | 8 |

test: IC95% bootstrap pareado de Δ (MediaPipe−YOLO): {'accuracy': [-0.16666666666666674, 0.06666666666666665], 'f1': [-0.17708482142857146, 0.06153985507246377], 'roc_auc': [-0.2141211087483016, 0.0070768118662139536], 'specificity': [-0.15384615384615385, 0.1380541871921176]}; McNemar exacto: {'only_yolo_correct': 8, 'only_mediapipe_correct': 5, 'exact_p': 0.5810546875}.

## Nota de trazabilidad posterior a la ejecución

El manifiesto `training/data/dl/left/posecnn/source_cmp_v1/manifest.json` y
las configuraciones de los modelos guardan el hash SHA-256 del script al
entrenar (`014d31e8…c4232eb`). El script publicado tiene otro hash
(`8cadba6a…52442d37`) porque se corrigió después **solo el texto del
generador de reportes**: antes afirmaba erróneamente que V3 usaba otra
cohorte. La cohorte del YOLO final es la misma V3 (403 muestras left) y sus
métricas Test se reprodujeron; la CV de esta comparación reordenó el
desarrollo y no debe presentarse como réplica exacta de la CV V3 histórica.
Esta nota no implica nuevo entrenamiento ni recálculo de las métricas.
