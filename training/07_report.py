import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import csv

from train_utils import PoseCNN, DEVICE, evaluate, compute_metrics, analyze_curves


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PATH = SCRIPT_DIR / "data" / "augmented" / "augmented_data.npz"
SPLIT_PATH = SCRIPT_DIR / "data" / "split" / "split_data.npz"
NORMALIZED_PATH = SCRIPT_DIR / "data" / "normalized" / "sequences_170.npz"
MODEL_PATH = SCRIPT_DIR / "models" / "best_model.pth"
HISTORY_PATH = SCRIPT_DIR / "models" / "history.npz"
RESULTS_DIR = SCRIPT_DIR / "results"
OUTPUT_DIR = SCRIPT_DIR / "models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 16


def load_test_data():
    data = np.load(INPUT_PATH)
    X_test = torch.from_numpy(data["X_test"]).float().permute(0, 3, 1, 2)
    y_test = data["y_test"]
    return X_test, y_test


def load_test_ids():
    try:
        split_data = np.load(SPLIT_PATH)
        ids = split_data["ids_test"]
    except Exception:
        data_norm = np.load(NORMALIZED_PATH)
        all_ids = data_norm["video_ids"]
        X_test, y_test = load_test_data()
        ids = all_ids[-len(y_test):]
    return ids


def analyze_predictions():
    X_test, y_test = load_test_data()
    test_ids = load_test_ids()

    model = PoseCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_preds, all_probs = [], []
    with torch.no_grad():
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_test, torch.from_numpy(y_test).float()),
            batch_size=BATCH_SIZE,
        )
        for X_batch, _ in loader:
            X_batch = X_batch.to(DEVICE)
            outputs = model(X_batch)
            probs = outputs.cpu().numpy().flatten()
            preds = (outputs > 0.5).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_preds.extend(preds)

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    rows = []
    for idx in range(len(y_test)):
        real = int(y_test[idx])
        pred = int(all_preds[idx])
        prob = float(all_probs[idx])
        correct = pred == real

        vid = str(test_ids[idx]) if idx < len(test_ids) else "unknown"
        ruta = f"keypoints/dl/front/{vid}.npy" if vid != "unknown" else "N/A"

        tipo_error = ""
        if not correct:
            if pred == 1 and real == 0:
                tipo_error = "FP"
            elif pred == 0 and real == 1:
                tipo_error = "FN"

        rows.append({
            "video_id": vid,
            "indice": idx,
            "etiqueta_real": real,
            "etiqueta_predicha": pred,
            "probabilidad": prob,
            "correcto": correct,
            "ruta_archivo": ruta,
            "tipo_error": tipo_error,
        })

    predictions_path = RESULTS_DIR / "predictions.csv"
    with open(predictions_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "video_id", "indice", "etiqueta_real", "etiqueta_predicha",
            "probabilidad", "correcto", "ruta_archivo", "tipo_error",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Predicciones guardadas en: {predictions_path}")

    errors = [r for r in rows if not r["correcto"]]

    misclassified_path = RESULTS_DIR / "misclassified.csv"
    with open(misclassified_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "video_id", "indice", "etiqueta_real", "etiqueta_predicha",
            "probabilidad", "correcto", "ruta_archivo", "tipo_error",
        ])
        writer.writeheader()
        for r in errors:
            row_out = dict(r)
            row_out["correcto"] = str(r["correcto"])
            writer.writerow(row_out)

    print(f"Errores guardados en: {misclassified_path}")

    fp_count = sum(1 for r in errors if r["tipo_error"] == "FP")
    fn_count = sum(1 for r in errors if r["tipo_error"] == "FN")
    total = len(rows)
    correct = total - len(errors)
    incorrect = len(errors)

    correct_probs = [r["probabilidad"] for r in rows if r["correcto"]]
    error_probs = [r["probabilidad"] for r in errors]
    mean_correct_prob = np.mean(correct_probs) if correct_probs else 0.0
    mean_error_prob = np.mean(error_probs) if error_probs else 0.0

    most_uncertain_correct = min(correct_probs, key=lambda p: abs(p - 0.5)) if correct_probs else None
    most_certain_error = max(error_probs, key=lambda p: max(p, 1 - p)) if error_probs else None

    print()
    print("=" * 34)
    print("ANALISIS DE PREDICCIONES")
    print("=" * 34)
    print()
    print(f"Total muestras:     {total}")
    print(f"Correctas:          {correct}")
    print(f"Incorrectas:        {incorrect}")
    print()
    print(f"Falsos positivos:   {fp_count}")
    print(f"Falsos negativos:   {fn_count}")
    print()

    if most_uncertain_correct is not None:
        idx_uncertain = correct_probs.index(most_uncertain_correct)
        r_uncertain = [r for r in rows if r["correcto"]][idx_uncertain]
        print(f"Prediccion correcta mas insegura:")
        print(f"  video_id: {r_uncertain['video_id']}, probabilidad: {most_uncertain_correct:.4f}")
    print()

    if most_certain_error is not None:
        idx_certain = error_probs.index(most_certain_error)
        r_certain = errors[idx_certain]
        print(f"Prediccion incorrecta mas segura:")
        print(f"  video_id: {r_certain['video_id']}, probabilidad: {most_certain_error:.4f}")
    print()

    print(f"Probabilidad media de aciertos:   {mean_correct_prob:.4f}")
    print(f"Probabilidad media de errores:    {mean_error_prob:.4f}")
    print()
    print("=" * 34)

    result = {
        "fp": [r for r in errors if r["tipo_error"] == "FP"],
        "fn": [r for r in errors if r["tipo_error"] == "FN"],
        "total": total,
        "correct": correct,
        "incorrect": incorrect,
        "fp_count": fp_count,
        "fn_count": fn_count,
        "rows": rows,
    }
    return result


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
    train_loss = list(history.get("train_loss", []))
    val_loss = list(history.get("val_loss", []))
    if len(train_loss) == 0:
        return {"best_epoch": "N/A", "diagnosis": "No disponible"}
    return analyze_curves(history)


def compute_test_metrics():
    X_test, y_test = load_test_data()

    model = PoseCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    test_ds = torch.utils.data.TensorDataset(X_test, torch.from_numpy(y_test).float())
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE)

    criterion = torch.nn.BCELoss()
    _, preds, labels, probs = evaluate(model, test_loader, criterion)
    metrics = compute_metrics(labels, preds, probs)
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
    errors = analyze_predictions()

    total_errors = errors["incorrect"]
    fp_count = errors["fp_count"]
    fn_count = errors["fn_count"]

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

### Falsos Positivos (FP) — {fp_count} casos

Predichos como válidos cuando eran inválidos.

| # | video_id | Probabilidad |
|---|---|---|
"""

    for i, item in enumerate(errors["fp"], 1):
        report += f"| {i} | {item['video_id']} | {item['probabilidad']:.4f} |\n"

    report += f"""
### Falsos Negativos (FN) — {fn_count} casos

Predichos como inválidos cuando eran válidos.

| # | video_id | Probabilidad |
|---|---|---|
"""
    for i, item in enumerate(errors["fn"], 1):
        report += f"| {i} | {item['video_id']} | {item['probabilidad']:.4f} |\n"

    report += f"""
El listado completo de todas las predicciones se encuentra en `predictions.csv`.
Los errores clasificados están en `misclassified.csv`.

---

## 7. Conclusión General

### Estado Actual del Modelo

- El modelo CNN para clasificación binaria de deadlift frontal alcanzó un **F1-score de {metrics['f1']:.4f}** y un **ROC-AUC de {metrics['roc_auc']:.4f}** en el conjunto de test.
- Se detectaron **{total_errors} errores** en el conjunto de test ({fp_count} FP, {fn_count} FN).
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
