import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from datetime import datetime


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PATH = SCRIPT_DIR / "data" / "augmented" / "augmented_data.npz"
SPLIT_PATH = SCRIPT_DIR / "data" / "split" / "split_data.npz"
NORMALIZED_PATH = SCRIPT_DIR / "data" / "normalized" / "sequences_170.npz"
MODEL_PATH = SCRIPT_DIR / "models" / "best_model.pth"
HISTORY_PATH = SCRIPT_DIR / "models" / "history.npz"
RESULTS_DIR = SCRIPT_DIR / "results"
OUTPUT_DIR = SCRIPT_DIR / "models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16


class PoseCNN(nn.Module):
    def __init__(self, T=170, K=25, C=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=3, padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, kernel_size=3, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


def get_misclassified():
    data = np.load(INPUT_PATH)
    X_test = torch.from_numpy(data["X_test"]).float().permute(0, 3, 1, 2)
    y_test = data["y_test"]

    try:
        split_data = np.load(SPLIT_PATH)
        ids_test = split_data["ids_test"]
    except Exception:
        ids_test = None

    model = PoseCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    test_ds = torch.utils.data.TensorDataset(X_test, torch.from_numpy(y_test).float())
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE)

    all_preds, all_probs = [], []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(DEVICE)
            outputs = model(X_batch)
            probs = outputs.cpu().numpy().flatten()
            preds = (outputs > 0.5).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_preds.extend(preds)

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    fp_mask = (all_preds == 0) & (y_test == 1)
    fn_mask = (all_preds == 1) & (y_test == 0)

    fps = np.where(fp_mask)[0]
    fns = np.where(fn_mask)[0]

    data_norm = np.load(NORMALIZED_PATH)
    all_ids = data_norm["video_ids"]

    # ids_test may not be available in augmented data; use ordered mapping
    if ids_test is not None:
        test_ids = ids_test
    else:
        test_ids = all_ids[-len(y_test):]

    results = {"fp": [], "fn": []}
    for idx in fps:
        results["fp"].append({
            "index": int(idx),
            "video_id": str(test_ids[idx]) if idx < len(test_ids) else "unknown",
            "real": 1,
            "pred": 0,
            "probability": float(all_probs[idx]),
        })
    for idx in fns:
        results["fn"].append({
            "index": int(idx),
            "video_id": str(test_ids[idx]) if idx < len(test_ids) else "unknown",
            "real": 0,
            "pred": 1,
            "probability": float(all_probs[idx]),
        })

    # Save CSV
    import csv
    csv_path = RESULTS_DIR / "misclassified.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["tipo", "indice", "video_id", "etiqueta_real", "etiqueta_predicha", "probabilidad", "ruta_archivo"])
        for item in results["fp"]:
            ruta = f"keypoints/dl/front/{item['video_id']}.npy" if item["video_id"] != "unknown" else "N/A"
            writer.writerow(["FP", item["index"], item["video_id"], item["real"], item["pred"], f"{item['probability']:.4f}", ruta])
        for item in results["fn"]:
            ruta = f"keypoints/dl/front/{item['video_id']}.npy" if item["video_id"] != "unknown" else "N/A"
            writer.writerow(["FN", item["index"], item["video_id"], item["real"], item["pred"], f"{item['probability']:.4f}", ruta])

    print(f"Errores guardados en: {csv_path}")
    print(f"  Falsos Positivos: {len(results['fp'])}")
    print(f"  Falsos Negativos: {len(results['fn'])}")

    return results


def load_history():
    try:
        data = np.load(HISTORY_PATH, allow_pickle=True)
        history = {k: data[k] for k in data.files}
        return history
    except Exception:
        return None


def analyze_overfitting_from_history(history):
    if history is None:
        return {"best_epoch": "N/A", "diagnosis": "No disponible"}
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    if len(train_loss) == 0:
        return {"best_epoch": "N/A", "diagnosis": "No disponible"}

    train_loss = np.array(train_loss)
    val_loss = np.array(val_loss)
    best_epoch = int(np.argmin(val_loss)) + 1
    last_val = val_loss[-1]
    min_val = val_loss.min()

    analysis = {"best_epoch": best_epoch, "best_val_loss": float(val_loss[best_epoch - 1])}

    if last_val > min_val * 1.08:
        analysis["diagnosis"] = "OVERFITTING"
        for i in range(1, len(val_loss)):
            if val_loss[i] > val_loss[i - 1] and val_loss[i] > min_val * 1.03:
                analysis["divergence_epoch"] = i
                break
        else:
            analysis["divergence_epoch"] = best_epoch
    elif last_val > 0.4 and train_loss[-1] > 0.4:
        analysis["diagnosis"] = "UNDERFITTING"
        analysis["divergence_epoch"] = None
    else:
        analysis["diagnosis"] = "NORMAL"
        analysis["divergence_epoch"] = None

    return analysis


def compute_test_metrics():
    data = np.load(INPUT_PATH)
    X_test = torch.from_numpy(data["X_test"]).float().permute(0, 3, 1, 2)
    y_test = data["y_test"]

    model = PoseCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    test_ds = torch.utils.data.TensorDataset(X_test, torch.from_numpy(y_test).float())
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE)

    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE).unsqueeze(1)
            outputs = model(X_batch)
            probs = outputs.cpu().numpy().flatten()
            preds = (outputs > 0.5).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(y_batch.cpu().numpy().flatten())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else None,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    return metrics


def get_dataset_summary():
    data_norm = np.load(NORMALIZED_PATH)
    total = len(data_norm["labels"])
    validos = int(sum(data_norm["labels"] == 1))
    invalidos = int(sum(data_norm["labels"] == 0))
    return {
        "total": total,
        "validos": validos,
        "invalidos": invalidos,
        "shape": str(data_norm["keypoints"].shape),
    }


def generate_report():
    print("Generando informe automático...")

    dataset = get_dataset_summary()
    metrics = compute_test_metrics()
    history = load_history()
    overfitting = analyze_overfitting_from_history(history)
    errors = get_misclassified()

    total_errors = len(errors["fp"]) + len(errors["fn"])
    error_rate = total_errors / (dataset["validos"] + dataset["invalidos"] - dataset["total"] + 62)  # approximate test size

    cm = np.array(metrics["confusion_matrix"])
    tn, fp_cm, fn_cm, tp = cm.ravel()

    best_val_loss = overfitting.get("best_val_loss", "N/A")
    if isinstance(best_val_loss, float):
        best_val_loss_str = f"{best_val_loss:.4f}"
    else:
        best_val_loss_str = str(best_val_loss)

    report = f"""# Informe de Evaluacion: CNN para Deadlift Frontal

**Fecha de generacion:** {datetime.now().strftime("%Y-%m-%d %H:%M")}

---

## 1. Resumen del Dataset

| Metrica | Valor |
|---|---|
| Total de secuencias | {dataset["total"]} |
| Validos (label=1) | {dataset["validos"]} ({dataset["validos"]/dataset["total"]*100:.1f}%) |
| Invalidos (label=0) | {dataset["invalidos"]} ({dataset["invalidos"]/dataset["total"]*100:.1f}%) |
| Shape de datos | {dataset["shape"]} |

---

## 2. Arquitectura del Modelo

| Capa | Detalle |
|---|---|
| Conv2D + BatchNorm + ReLU | 32 filtros, kernel 3x3, padding same |
| Conv2D + BatchNorm + ReLU | 64 filtros, kernel 3x3, padding same |
| MaxPool2D | kernel (2, 1) |
| Dropout | 25% |
| Conv2D + BatchNorm + ReLU | 128 filtros, kernel 3x3, padding same |
| Global Average Pooling | Adaptativo (1x1) |
| Dense + ReLU + Dropout | 64 unidades, dropout 50% |
| Dense + Sigmoid | 1 unidad (salida binaria) |

### Hiperparametros

| Parametro | Valor |
|---|---|
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Loss | Binary Cross-Entropy (BCE) |
| Batch Size | 16 |
| Early Stopping Patience | 15 epocas |
| ReduceLROnPlateau | factor 0.5, patience 5 |
| Data Augmentation | 3 aumentaciones por secuencia |
| Normalizacion temporal | 170 frames (percentil 90) |

---

## 3. Curvas de Entrenamiento

![Train/Val Loss y Accuracy](train_val_loss.png)

### Interpretacion Automatica

- **Mejor epoca:** {overfitting.get("best_epoch", "N/A")}
- **Mejor validation loss:** {best_val_loss_str}
- **Diagnostico:** {overfitting.get("diagnosis", "No disponible")}

"""

    if overfitting.get("diagnosis") == "OVERFITTING":
        report += f"""- El validation loss aumentó mientras el train loss seguía disminuyendo, indicando sobreajuste.
- El sobreajuste comenzó aproximadamente en la época {overfitting.get("divergence_epoch", "desconocida")}.
- Se recomienda aumentar la regularización (dropout, weight decay) o reducir la capacidad del modelo.

"""
    elif overfitting.get("diagnosis") == "UNDERFITTING":
        report += """- Ambas pérdidas (train y validation) se mantienen altas, indicando subajuste.
- El modelo no está aprendiendo lo suficiente de los datos.
- Se recomienda aumentar la capacidad del modelo o revisar el preprocesamiento.

"""
    else:
        report += """- No hay evidencia de overfitting ni underfitting significativos.
- Las curvas de train y validation loss siguen una tendencia similar, lo que indica que el modelo generaliza correctamente.

"""

    report += f"""---

## 4. Métricas en Test

| Métrica | Valor |
|---|---|
| Accuracy | {metrics["accuracy"]:.4f} |
| Precision | {metrics["precision"]:.4f} |
| Recall | {metrics["recall"]:.4f} |
| F1-Score | {metrics["f1"]:.4f} |
"""

    if metrics["roc_auc"] is not None:
        report += f"| ROC-AUC | {metrics['roc_auc']:.4f} |\n"

    report += f"""
### Matriz de Confusión

| | Pred: Inválido (0) | Pred: Válido (1) |
|---|---|---|
| **Real: Inválido (0)** | {int(tn)} (verdaderos negativos) | {int(fp_cm)} (falsos positivos) |
| **Real: Válido (1)** | {int(fn_cm)} (falsos negativos) | {int(tp)} (verdaderos positivos) |

---

## 5. Validación Cruzada (5-Fold)

![Cross Validation Boxplot](cross_validation_boxplot.png)

<!-- Los resultados de 5-Fold CV se insertan aquí desde la ejecución de 06_cross_validation.py -->

> **Nota:** Los resultados de validación cruzada se generan ejecutando `06_cross_validation.py`. El análisis completo se encuentra en la salida de dicho script.

---

## 6. Análisis de Errores

### Falsos Positivos (FP) — {len(errors['fp'])} casos

Predichos como inválidos cuando eran válidos.

| # | video_id | Probabilidad |
|---|---|---|
"""

    for i, item in enumerate(errors["fp"], 1):
        report += f"| {i} | {item['video_id']} | {item['probability']:.4f} |\n"

    report += f"""
### Falsos Negativos (FN) — {len(errors['fn'])} casos

Predichos como válidos cuando eran inválidos.

| # | video_id | Probabilidad |
|---|---|---|
"""
    for i, item in enumerate(errors["fn"], 1):
        report += f"| {i} | {item['video_id']} | {item['probability']:.4f} |\n"

    report += f"""
El listado completo con rutas de archivo se encuentra en `misclassified.csv`.

---

## 7. Conclusión General

### Estado Actual del Modelo

- El modelo CNN para clasificación binaria de deadlift frontal alcanzó un **F1-score de {metrics['f1']:.4f}** y un **ROC-AUC de {metrics['roc_auc']:.4f}** en el conjunto de test.
- Se detectaron **{total_errors} errores** en el conjunto de test ({len(errors['fp'])} FP, {len(errors['fn'])} FN).
- El análisis de curvas de entrenamiento indica **{overfitting.get('diagnosis', 'estado normal')}**.

### Recomendaciones

"""

    if metrics["f1"] >= 0.85 and metrics["roc_auc"] is not None and metrics["roc_auc"] >= 0.85:
        report += """- El modelo presenta un rendimiento sólido y puede considerarse para su uso en producción con supervisión.
- Se recomienda aumentar el dataset, especialmente la clase minoritaria (inválidos), para mejorar la robustez.
- Probar arquitecturas alternativas (LSTM, Transformer) podría ofrecer mejoras marginales.
"""
    elif metrics["f1"] >= 0.75:
        report += """- El modelo tiene un rendimiento aceptable pero con margen de mejora.
- Se recomienda aumentar el dataset y probar técnicas de balanceo de clases (class weights, focal loss).
- Evaluar si incluir más vistas (lateral) mejora la clasificación.
"""
    else:
        report += """- El modelo actual no alcanza un rendimiento suficiente para producción.
- Se recomienda aumentar significativamente el dataset.
- Considerar un enfoque clásico (Random Forest, XGBoost) como baseline antes de deep learning.
"""

    report += """
---

*Informe generado automáticamente por el pipeline de evaluación.*
"""

    report_path = RESULTS_DIR / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Informe guardado en: {report_path}")


if __name__ == "__main__":
    generate_report()
