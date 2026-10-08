"""Analisis post-hoc por falta y observabilidad de BP_FRONT PoseCNN.

Este script solo lee artefactos congelados. No carga el checkpoint, no ejecuta
inferencia y no modifica modelo, split, preprocessing, labels ni threshold.
"""

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

PREDICTIONS_PATH = (
    SCRIPT_DIR / "results" / "bp" / "front" / "posecnn"
    / "bp_front_pose_test_predictions.csv"
)
SPLIT_PATH = (
    SCRIPT_DIR / "data" / "bp" / "front" / "bp_front_pose_split.csv"
)
LABELS_PATH = PROJECT_ROOT / "etiquetado" / "bp" / "bp_front.csv"
BASE_REPORT_PATH = PROJECT_ROOT / "Reporte_BP_FRONT_PoseCNN.md"
AUDIT_PATH = SCRIPT_DIR / "data" / "bp" / "front" / "bp_front_pose_audit.csv"
PREPROCESSING_MANIFEST_PATH = (
    SCRIPT_DIR / "data" / "bp" / "front"
    / "bp_front_pose_preprocessing_manifest.csv"
)
MODEL_PATH = (
    SCRIPT_DIR / "models" / "bp" / "front" / "posecnn"
    / "bp_front_pose_best_model.pth"
)
SCALER_PATH = (
    SCRIPT_DIR / "models" / "bp" / "front" / "posecnn"
    / "bp_front_pose_scaler.pkl"
)

OUTPUT_DIR = (
    SCRIPT_DIR / "results" / "bp" / "front" / "posecnn" / "analysis"
)
BY_CODE_PATH = OUTPUT_DIR / "bp_front_pose_test_by_fault_code.csv"
INVALID_SAMPLES_PATH = OUTPUT_DIR / "bp_front_pose_test_invalid_samples.csv"
OBSERVABILITY_PATH = OUTPUT_DIR / "bp_front_pose_fault_observability.csv"
OBSERVABILITY_SUMMARY_PATH = (
    OUTPUT_DIR / "bp_front_pose_observability_summary.csv"
)
ANALYSIS_MANIFEST_PATH = OUTPUT_DIR / "bp_front_pose_fault_analysis_manifest.json"
REPORT_PATH = PROJECT_ROOT / "Reporte_BP_FRONT_PoseCNN_Fault_Observability.md"

EXPECTED_CONFUSION_MATRIX = np.asarray([[31, 9], [7, 53]])
EXPECTED_TEST_SIZE = 100
EXPECTED_INVALID = 40
EXPECTED_VALID = 60
EXPECTED_THRESHOLD = 0.5

OBS_DIRECT = "DIRECTAMENTE_OBSERVABLE_POSE"
OBS_PARTIAL = "PARCIALMENTE_OBSERVABLE_POSE"
OBS_WEAK = "DEBILMENTE_OBSERVABLE_POSE"
OBS_NONE = "NO_OBSERVABLE_DIRECTAMENTE_POSE"
OBS_UNKNOWN = "INDETERMINADO"
OBSERVABILITY_ORDER = (OBS_DIRECT, OBS_PARTIAL, OBS_WEAK, OBS_NONE, OBS_UNKNOWN)

# La taxonomia se fija desde el criterio reglamentario y los inputs reales,
# antes de calcular el rendimiento por codigo.
OBSERVABILITY_DEFINITIONS = {
    "BP-01": {
        "pose_observability": OBS_WEAK,
        "available_pose_information": (
            "Trayectoria temporal de ambas munecas y geometria relativa de "
            "munecas, codos y hombros como proxy de la trayectoria de la barra."
        ),
        "missing_information": (
            "Barra, pecho, abdomen, cinturon y evidencia de contacto entre "
            "la barra y la superficie corporal reglamentaria."
        ),
        "justification": (
            "Las munecas pueden correlacionarse con la altura de la barra, "
            "pero Pose no observa la barra ni el objetivo de contacto. Una "
            "clasificacion correcta solo demuestra uso de proxies o "
            "correlaciones del dataset."
        ),
        "potentially_informative_modality": "POSE + SEGMENTATION",
        "modality_rationale": (
            "Pose aporta articulaciones y dinamica; segmentacion podria aportar "
            "barra y contorno corporal. El contacto exacto con pecho/abdomen o "
            "cinturon puede seguir requiriendo semantica adicional."
        ),
    },
    "BP-02": {
        "pose_observability": OBS_DIRECT,
        "available_pose_information": (
            "Posiciones x/y temporales de ambos hombros y ambos codos, que son "
            "las articulaciones directamente implicadas por el criterio."
        ),
        "missing_information": (
            "Superficie anatomica completa de los hombros; los keypoints son "
            "puntos y no contornos corporales."
        ),
        "justification": (
            "La relacion vertical codo-hombro puede medirse directamente con "
            "los keypoints disponibles. La falta de contorno introduce una "
            "aproximacion, pero no elimina la evidencia geometrica principal."
        ),
        "potentially_informative_modality": "POSE",
        "modality_rationale": (
            "Pose ya contiene las articulaciones centrales del criterio. Una "
            "mascara corporal podria refinar la superficie del hombro, pero no "
            "es imprescindible para una primera aproximacion geometrica."
        ),
    },
    "BP-03": {
        "pose_observability": OBS_PARTIAL,
        "available_pose_information": (
            "Trayectoria temporal bilateral de munecas, codos y hombros; las "
            "munecas suelen desplazarse junto con la barra durante el press."
        ),
        "missing_information": (
            "Barra observada directamente y delimitacion explicita de la fase "
            "de press o del comando arbitral."
        ),
        "justification": (
            "Una bajada bilateral de munecas puede ser evidencia parcial de "
            "movimiento descendente, pero Pose no prueba que la barra descendio "
            "ni identifica por si sola la fase reglamentaria exacta."
        ),
        "potentially_informative_modality": "POSE + SEGMENTATION",
        "modality_rationale": (
            "La trayectoria segmentada de la barra complementaria la cinemática "
            "articular. Aun seria necesario identificar correctamente la fase "
            "de press."
        ),
    },
    "BP-08": {
        "pose_observability": OBS_WEAK,
        "available_pose_information": (
            "Posicion temporal de ambos hombros y movimiento relativo de brazos."
        ),
        "missing_information": (
            "Cabeza, gluteos, banco y contacto de cabeza/hombros/gluteos con el "
            "banco."
        ),
        "justification": (
            "Solo los hombros estan representados y ni siquiera se observa la "
            "superficie del banco. El criterio disyuntivo tambien puede violarse "
            "por cabeza o gluteos, completamente ausentes del input."
        ),
        "potentially_informative_modality": "POSE + SEGMENTATION",
        "modality_rationale": (
            "La mascara de persona y banco podria aportar geometria de contacto; "
            "Pose aporta hombros. La segmentacion no garantiza por si sola la "
            "separacion anatomica fiable de cabeza y gluteos."
        ),
    },
    "UNSPECIFIED": {
        "pose_observability": OBS_UNKNOWN,
        "available_pose_information": (
            "Secuencia general de hombros, codos y munecas."
        ),
        "missing_information": (
            "Codigo y criterio de nulidad de la muestra."
        ),
        "justification": (
            "Sin causa etiquetada no es posible relacionar el resultado con la "
            "informacion observable por Pose."
        ),
        "potentially_informative_modality": "INFORMACION NO DISPONIBLE AUN",
        "modality_rationale": (
            "Primero se necesita conocer la causa reglamentaria de nulidad; no "
            "puede asignarse una modalidad adecuada desde una etiqueta binaria."
        ),
    },
}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_codes(value):
    if pd.isna(value) or not str(value).strip():
        return []
    return [part.strip() for part in str(value).split("|") if part.strip()]


def parse_criteria(value):
    if pd.isna(value) or not str(value).strip():
        return []
    return [
        part.strip()
        for part in re.split(r"\s+\|\s+", str(value).strip())
        if part.strip()
    ]


def build_criterion_map(labels):
    criterion_sets = {}
    for row in labels.itertuples(index=False):
        codes = parse_codes(row.codigos)
        criteria = parse_criteria(row.criterios)
        if not codes:
            continue
        if len(codes) != len(criteria):
            raise RuntimeError(
                f"Codigos/criterios no alineados en {row.video_id}: "
                f"{codes} vs {criteria}"
            )
        for code, criterion in zip(codes, criteria):
            criterion_sets.setdefault(code, set()).add(criterion)

    criterion_map = {}
    for code, values in criterion_sets.items():
        if len(values) != 1:
            raise RuntimeError(
                f"Definiciones contradictorias para {code}: {sorted(values)}"
            )
        criterion_map[code] = next(iter(values))
    return criterion_map


def probability_stats(subset):
    if len(subset) == 0:
        return {
            "mean_probability_valid": np.nan,
            "median_probability_valid": np.nan,
            "min_probability_valid": np.nan,
            "max_probability_valid": np.nan,
        }
    probabilities = subset["probability_valid"].astype(float)
    return {
        "mean_probability_valid": float(probabilities.mean()),
        "median_probability_valid": float(probabilities.median()),
        "min_probability_valid": float(probabilities.min()),
        "max_probability_valid": float(probabilities.max()),
    }


def classification_interpretation(categories, predicted_label):
    correct = int(predicted_label) == 0
    if OBS_DIRECT in categories:
        if correct:
            return (
                "Clasificacion correcta con evidencia articular directamente "
                "relevante disponible en Pose."
            )
        return (
            "Error real del clasificador: existia evidencia articular "
            "directamente relevante en Pose; la falta de observabilidad no "
            "explica por si sola el error."
        )
    if OBS_PARTIAL in categories:
        if correct:
            return (
                "Clasificacion correcta con evidencia parcial; puede apoyarse "
                "en cinemática de munecas, pero no prueba observacion directa "
                "de la barra o de la condicion reglamentaria."
            )
        return (
            "Clasificacion incorrecta con evidencia solo parcial; faltan barra "
            "y/o fase reglamentaria, aunque tambien puede existir error de "
            "aprendizaje del clasificador."
        )
    if OBS_WEAK in categories:
        if correct:
            return (
                "Clasificacion correcta pese a evidencia debil; debe "
                "interpretarse como uso de proxies o correlaciones, no como "
                "deteccion directa de la falta."
            )
        return (
            "Clasificacion incorrecta ante evidencia debil; la informacion "
            "reglamentaria ausente es una limitacion plausible, pero no prueba "
            "la causa individual del error."
        )
    if OBS_NONE in categories:
        if correct:
            return (
                "Clasificacion correcta sin evidencia directa; solo puede "
                "atribuirse a proxies o correlaciones del dataset."
            )
        return (
            "Clasificacion incorrecta sin evidencia directa disponible en Pose."
        )
    return (
        "Interpretacion indeterminada porque la muestra no tiene codigo ni "
        "criterio de nulidad."
    )


def markdown_float(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/D"
    return f"{float(value):.4f}"


def markdown_table(dataframe, columns, labels=None):
    labels = labels or columns
    header = "| " + " | ".join(labels) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    rows = []
    for _, row in dataframe.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                value = markdown_float(value)
            value = "" if pd.isna(value) else str(value)
            values.append(value.replace("|", "\\|"))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator] + rows)


def run_analysis():
    required = (
        PREDICTIONS_PATH,
        SPLIT_PATH,
        LABELS_PATH,
        BASE_REPORT_PATH,
        AUDIT_PATH,
        PREPROCESSING_MANIFEST_PATH,
        MODEL_PATH,
        SCALER_PATH,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Faltan artefactos congelados: {missing}")

    input_hashes_before = {str(path): file_sha256(path) for path in required}

    predictions = pd.read_csv(PREDICTIONS_PATH, dtype={"video_id": str})
    split = pd.read_csv(SPLIT_PATH, dtype={"video_id": str})
    labels = pd.read_csv(LABELS_PATH, dtype={"video_id": str})

    assert len(predictions) == EXPECTED_TEST_SIZE
    assert predictions["video_id"].is_unique
    assert set(predictions["true_label"].unique()) == {0, 1}
    assert set(predictions["predicted_label"].unique()) == {0, 1}
    assert np.allclose(predictions["threshold"], EXPECTED_THRESHOLD)
    assert set(predictions["positive_class"]) == {"1=valido"}
    assert (
        predictions["correct"].astype(bool)
        == (predictions["true_label"] == predictions["predicted_label"])
    ).all()

    frozen_test = split.loc[split["pose_only_split"] == "test", ["video_id", "label"]]
    assert len(frozen_test) == EXPECTED_TEST_SIZE
    assert set(frozen_test["video_id"]) == set(predictions["video_id"])

    merged = predictions.merge(
        labels,
        on="video_id",
        how="left",
        validate="one_to_one",
        suffixes=("_prediction", "_csv"),
    )
    assert merged["label"].notna().all()
    assert (merged["true_label"] == merged["label"]).all()

    matrix = confusion_matrix(
        merged["true_label"], merged["predicted_label"], labels=[0, 1]
    )
    assert np.array_equal(matrix, EXPECTED_CONFUSION_MATRIX)
    assert int((merged["true_label"] == 0).sum()) == EXPECTED_INVALID
    assert int((merged["true_label"] == 1).sum()) == EXPECTED_VALID

    criterion_map = build_criterion_map(labels)
    real_codes = sorted(criterion_map)
    if set(real_codes) != set(OBSERVABILITY_DEFINITIONS) - {"UNSPECIFIED"}:
        raise RuntimeError(
            "La taxonomia no cubre exactamente los codigos reales: "
            f"reales={real_codes} taxonomia={sorted(OBSERVABILITY_DEFINITIONS)}"
        )

    labels = labels.copy()
    labels["parsed_codes"] = labels["codigos"].apply(parse_codes)
    merged = merged.copy()
    merged["parsed_codes"] = merged["codigos"].apply(parse_codes)

    full_invalid = labels.loc[labels["label"] == 0]
    test_invalid = merged.loc[merged["true_label"] == 0].copy()
    test_valid = merged.loc[merged["true_label"] == 1].copy()

    observability_rows = []
    for code in real_codes + ["UNSPECIFIED"]:
        definition = OBSERVABILITY_DEFINITIONS[code]
        if code == "UNSPECIFIED":
            full_count = int((full_invalid["parsed_codes"].str.len() == 0).sum())
            test_count = int((test_invalid["parsed_codes"].str.len() == 0).sum())
            criterion = "Codigo y criterio no informados en el CSV"
            source_type = "synthetic_bucket_for_missing_code"
        else:
            full_count = int(full_invalid["parsed_codes"].apply(lambda x: code in x).sum())
            test_count = int(test_invalid["parsed_codes"].apply(lambda x: code in x).sum())
            criterion = criterion_map[code]
            source_type = "real_fault_code"
        observability_rows.append({
            "fault_code": code,
            "source_type": source_type,
            "criterion_description": criterion,
            "pose_observability": definition["pose_observability"],
            "available_pose_information": definition["available_pose_information"],
            "missing_information": definition["missing_information"],
            "justification": definition["justification"],
            "potentially_informative_modality": definition[
                "potentially_informative_modality"
            ],
            "modality_rationale": definition["modality_rationale"],
            "n_full_dataset": full_count,
            "n_test": test_count,
        })
    observability = pd.DataFrame(observability_rows)
    observability_by_code = observability.set_index("fault_code")

    invalid_rows = []
    for row in test_invalid.itertuples(index=False):
        codes = list(row.parsed_codes) or ["UNSPECIFIED"]
        categories = sorted(
            {
                OBSERVABILITY_DEFINITIONS[code]["pose_observability"]
                for code in codes
            },
            key=OBSERVABILITY_ORDER.index,
        )
        notes = "; ".join(
            f"{code}: {OBSERVABILITY_DEFINITIONS[code]['pose_observability']}"
            for code in codes
        )
        invalid_rows.append({
            "video_id": row.video_id,
            "true_label": int(row.true_label),
            "predicted_label": int(row.predicted_label),
            "correct": bool(row.correct),
            "probability_valid": float(row.probability_valid),
            "fault_codes": "|".join(codes),
            "criteria": "" if pd.isna(row.criterios) else str(row.criterios),
            "pose_observability_categories": "|".join(categories),
            "contains_directly_observable_fault": int(OBS_DIRECT in categories),
            "contains_partially_observable_fault": int(OBS_PARTIAL in categories),
            "contains_weakly_observable_fault": int(OBS_WEAK in categories),
            "contains_nonobservable_fault": int(OBS_NONE in categories),
            "contains_indeterminate_fault": int(OBS_UNKNOWN in categories),
            "observability_notes": notes,
            "classification_interpretation": classification_interpretation(
                categories, row.predicted_label
            ),
            "global_confusion_role": "TN" if row.predicted_label == 0 else "FP",
        })
    invalid_samples = pd.DataFrame(invalid_rows).sort_values("video_id")

    by_code_rows = []
    for code in real_codes + ["UNSPECIFIED"]:
        if code == "UNSPECIFIED":
            subset = test_invalid.loc[test_invalid["parsed_codes"].str.len() == 0]
        else:
            subset = test_invalid.loc[
                test_invalid["parsed_codes"].apply(lambda x: code in x)
            ]
        n_test = len(subset)
        n_correct_invalid = int((subset["predicted_label"] == 0).sum())
        n_incorrect_valid = int((subset["predicted_label"] == 1).sum())
        metadata = observability_by_code.loc[code]
        result = {
            "fault_code": code,
            "source_type": metadata["source_type"],
            "criterion_description": metadata["criterion_description"],
            "pose_observability": metadata["pose_observability"],
            "n_full_dataset": int(metadata["n_full_dataset"]),
            "n_test": int(n_test),
            "n_predicted_invalid": n_correct_invalid,
            "n_predicted_valid": n_incorrect_valid,
            "correct_null_predictions": n_correct_invalid,
            "incorrect_valid_predictions": n_incorrect_valid,
            "n_correct_invalid": n_correct_invalid,
            "n_incorrect_valid": n_incorrect_valid,
            "null_detection_rate": (
                n_correct_invalid / n_test if n_test else np.nan
            ),
        }
        result.update(probability_stats(subset))
        by_code_rows.append(result)
    by_code = pd.DataFrame(by_code_rows)

    summary_rows = []
    for category in OBSERVABILITY_ORDER:
        flag_column = {
            OBS_DIRECT: "contains_directly_observable_fault",
            OBS_PARTIAL: "contains_partially_observable_fault",
            OBS_WEAK: "contains_weakly_observable_fault",
            OBS_NONE: "contains_nonobservable_fault",
            OBS_UNKNOWN: "contains_indeterminate_fault",
        }[category]
        subset = invalid_samples.loc[invalid_samples[flag_column] == 1]
        n = len(subset)
        correct_invalid = int((subset["predicted_label"] == 0).sum())
        incorrect_as_valid = int((subset["predicted_label"] == 1).sum())
        result = {
            "observability_category": category,
            "sample_type": "INVALID",
            "membership_mode": "multi_hot_nonexclusive",
            "n": n,
            "correct_invalid": correct_invalid,
            "incorrect_as_valid": incorrect_as_valid,
            "null_detection_rate": correct_invalid / n if n else np.nan,
            "correct_as_valid": np.nan,
            "incorrect_as_invalid": np.nan,
            "valid_classification_rate": np.nan,
        }
        result.update(probability_stats(subset))
        summary_rows.append(result)

    valid_correct = int((test_valid["predicted_label"] == 1).sum())
    valid_incorrect = int((test_valid["predicted_label"] == 0).sum())
    valid_summary = {
        "observability_category": "VALID",
        "sample_type": "VALID",
        "membership_mode": "mutually_exclusive_valid_class",
        "n": len(test_valid),
        "correct_invalid": np.nan,
        "incorrect_as_valid": np.nan,
        "null_detection_rate": np.nan,
        "correct_as_valid": valid_correct,
        "incorrect_as_invalid": valid_incorrect,
        "valid_classification_rate": valid_correct / len(test_valid),
    }
    valid_summary.update(probability_stats(test_valid))
    summary_rows.append(valid_summary)
    observability_summary = pd.DataFrame(summary_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    by_code.to_csv(BY_CODE_PATH, index=False, encoding="utf-8")
    invalid_samples.to_csv(INVALID_SAMPLES_PATH, index=False, encoding="utf-8")
    observability.to_csv(OBSERVABILITY_PATH, index=False, encoding="utf-8")
    observability_summary.to_csv(
        OBSERVABILITY_SUMMARY_PATH, index=False, encoding="utf-8"
    )

    false_positives = invalid_samples.loc[
        invalid_samples["global_confusion_role"] == "FP"
    ].copy()
    true_negatives = invalid_samples.loc[
        invalid_samples["global_confusion_role"] == "TN"
    ].copy()
    assert len(false_positives) == 9
    assert len(true_negatives) == 31

    full_multicode = full_invalid.loc[
        full_invalid["parsed_codes"].str.len() > 1
    ]
    test_multicode_ids = set(
        test_invalid.loc[test_invalid["parsed_codes"].str.len() > 1, "video_id"]
    )
    test_multicode = invalid_samples.loc[
        invalid_samples["video_id"].isin(test_multicode_ids)
    ]

    code_table = observability[[
        "fault_code",
        "criterion_description",
        "pose_observability",
        "available_pose_information",
        "missing_information",
        "justification",
        "potentially_informative_modality",
    ]]
    frequency_table = observability[[
        "fault_code", "source_type", "n_full_dataset", "n_test"
    ]]
    performance_table = by_code[[
        "fault_code",
        "pose_observability",
        "n_test",
        "n_correct_invalid",
        "n_incorrect_valid",
        "null_detection_rate",
        "mean_probability_valid",
        "median_probability_valid",
        "min_probability_valid",
        "max_probability_valid",
    ]]
    category_table = observability_summary[[
        "observability_category",
        "sample_type",
        "membership_mode",
        "n",
        "correct_invalid",
        "incorrect_as_valid",
        "null_detection_rate",
        "mean_probability_valid",
        "median_probability_valid",
        "correct_as_valid",
        "incorrect_as_invalid",
        "valid_classification_rate",
    ]]
    fp_table = false_positives[[
        "video_id",
        "true_label",
        "predicted_label",
        "probability_valid",
        "fault_codes",
        "criteria",
        "pose_observability_categories",
        "classification_interpretation",
    ]]
    tn_table = true_negatives[[
        "video_id",
        "probability_valid",
        "fault_codes",
        "pose_observability_categories",
        "classification_interpretation",
    ]]
    multicode_table = test_multicode[[
        "video_id",
        "fault_codes",
        "pose_observability_categories",
        "predicted_label",
        "probability_valid",
        "correct",
    ]]

    bp01 = by_code.set_index("fault_code").loc["BP-01"]
    bp02 = by_code.set_index("fault_code").loc["BP-02"]
    bp03 = by_code.set_index("fault_code").loc["BP-03"]
    bp08 = by_code.set_index("fault_code").loc["BP-08"]

    report = f"""# BP_FRONT PoseCNN: Fault Code and Pose Observability

## 1. Objetivo y alcance

Este es un analisis post-hoc descriptivo del modelo BP_FRONT PoseCNN ya cerrado. No se reentreno el modelo, no se ejecuto inferencia, no se modificaron probabilidades, threshold, split, preprocessing, labels, scaler ni checkpoint. Todas las conclusiones de rendimiento proceden exclusivamente de `{PREDICTIONS_PATH}` y del holdout Test congelado.

Las metricas globales permanecen sin cambios: Accuracy 0.8400, Precision 0.8548, Recall de la clase positiva valida 0.8833, F1 0.8689 y ROC-AUC 0.9354. La matriz es TN=31, FP=9, FN=7, TP=53. Clase positiva global: `1 = valido`.

El analisis por observabilidad explica resultados; no reemplaza estas metricas globales.

## 2. Informacion disponible en Pose

Input `(T,6,3)` con `x`, `y` y `confidence`:

| Index | Keypoint |
|---:|---|
| 0 | `left_elbow` |
| 1 | `left_shoulder` |
| 2 | `right_shoulder` |
| 3 | `right_elbow` |
| 4 | `right_wrist` |
| 5 | `left_wrist` |

Pose observa geometria y dinamica temporal de hombros, codos y munecas. No observa directamente barra, pecho, abdomen, cinturon, banco, cabeza, gluteos ni una mascara corporal completa. Que el modelo clasifique correctamente una muestra no demuestra que haya detectado directamente su falta reglamentaria.

## 3. Auditoria de codigos y criterios

El CSV original contiene {len(labels)} filas y cuatro codigos reales: `{', '.join(real_codes)}`. Hay {len(full_multicode)} muestras invalidas multicodigo y {int((full_invalid['parsed_codes'].str.len() == 0).sum())} invalidas sin codigo. `UNSPECIFIED` es un bucket descriptivo para estas ultimas, no un codigo reglamentario inventado.

{markdown_table(code_table, list(code_table.columns))}

La taxonomia se definio desde los criterios y los inputs antes de calcular resultados. `BP-02` es directamente observable; `BP-03` parcialmente observable; `BP-01` y `BP-08` debilmente observables. No hay un codigo real clasificado como `{OBS_NONE}` en este CSV.

## 4. Frecuencia en dataset completo y Test

{markdown_table(frequency_table, list(frequency_table.columns))}

Las frecuencias por codigo no suman el total de nulos porque una muestra multicodigo pertenece a cada uno de sus codigos. Las frecuencias del dataset completo son descriptivas; no se usaron predicciones de Train o Validation.

## 5. Null Detection Rate por codigo

Para una muestra realmente invalida, `predicted_label=0` es correcta y `predicted_label=1` es un falso positivo global. Se define:

`Null Detection Rate = invalidas del codigo predichas como invalidas / invalidas Test del codigo`.

No se denomina Recall porque el Recall global usa como clase positiva `1 = valido`.

{markdown_table(performance_table, list(performance_table.columns))}

Resultados principales:

- `BP-02`, directamente observable, obtuvo {bp02['n_correct_invalid']:.0f}/{bp02['n_test']:.0f} y Null Detection Rate {bp02['null_detection_rate']:.2%}.
- `BP-03`, parcialmente observable, obtuvo {bp03['n_correct_invalid']:.0f}/{bp03['n_test']:.0f} y {bp03['null_detection_rate']:.2%}; seis de ocho fueron clasificados como validos.
- `BP-01`, debilmente observable, obtuvo {bp01['n_correct_invalid']:.0f}/{bp01['n_test']:.0f} y {bp01['null_detection_rate']:.2%}. Este resultado alto no prueba observacion de contacto barra-cuerpo; es compatible con proxies de munecas o correlaciones del dataset.
- `BP-08` tiene solo {bp08['n_test']:.0f} caso Test y no permite una conclusion estable; fue clasificado como valido.

## 6. Analisis multi-hot por observabilidad

{markdown_table(category_table, list(category_table.columns))}

Las categorias invalidas son multi-hot y no mutuamente excluyentes. Una muestra con codigos de distinta observabilidad participa en todas las categorias correspondientes. Por eso sus N tampoco tienen que sumar 40. No se forzo una categoria unica por muestra.

## 7. Nueve falsos positivos

Estos son los nueve levantamientos invalidos clasificados como validos:

{markdown_table(fp_table, list(fp_table.columns))}

Seis de los nueve FP contienen `BP-03`, uno `BP-01`, uno `BP-08` y uno no tiene codigo (`UNSPECIFIED`). No hay un FP con `BP-02` en este holdout. La baja observabilidad es una limitacion plausible para BP-01/BP-03/BP-08, pero no se atribuye automaticamente cada error a esa causa; tambien puede existir fallo de aprendizaje o sesgo de muestra. `bp_606` no puede interpretarse causalmente porque carece de codigo.

## 8. Treinta y un true negatives

Estos son los 31 nulos correctamente clasificados:

{markdown_table(tn_table, list(tn_table.columns))}

Los TN incluyen faltas directamente observables y faltas debil o parcialmente observables. Para estas ultimas debe decirse que el modelo clasifico correctamente la muestra mediante informacion indirecta potencial; no que detecto directamente la regla.

## 9. Casos multicodigo

En el Test hay {len(test_multicode)} muestras multicodigo:

{markdown_table(multicode_table, list(multicode_table.columns))}

Cada una participa en todas sus filas per-code y en todos sus flags multi-hot. No se reconciliaron ni priorizaron codigos.

## 10. Levantamientos validos

Los 60 validos reales se mantienen como categoria `VALID`: 53 fueron clasificados correctamente como validos (TP) y 7 como invalidos (FN). Su tasa de clasificacion valida es {valid_correct / len(test_valid):.2%}. No se les asigna observabilidad de falta porque no tienen una causa de nulidad que explicar.

## 11. Proxies e interpretacion

PoseCNN puede usar trayectoria bilateral de munecas como proxy de la barra, angulos y relaciones codo-hombro, amplitud del descenso, sincronizacion bilateral, duracion relativa y patrones generales de ejecucion. Tambien puede explotar correlaciones no causales del dataset. En faltas sin evidencia reglamentaria completa, un acierto solo significa `clasifico correctamente`; no demuestra deteccion directa del codigo.

Los datos apoyan parcialmente la hipotesis de que Pose funciona mejor cuando la evidencia esta representada: BP-02 alcanza 100% y BP-03 solo 25%. Sin embargo, BP-01 tambien alcanza 94.44% pese a ser debilmente observable, por lo que no existe una relacion monotona simple entre observabilidad y rendimiento. Ese resultado refuerza la posibilidad de proxies o correlaciones. BP-08 tiene N=1 y no permite inferencia general.

## 12. Complementariedad esperada con Segmentation

| Codigo | Evidencia potencial principal | Interpretacion |
|---|---|---|
| BP-01 | Pose + Segmentation | Pose aporta munecas/brazos; segmentacion podria aportar barra y contorno corporal. No garantiza por si sola contacto semantico con pecho, abdomen o cinturon. |
| BP-02 | Pose | Hombros y codos contienen la relacion geometrica principal; una mascara podria refinar superficies. |
| BP-03 | Pose + Segmentation | Pose aporta cinemática articular y segmentacion podria aportar trayectoria directa de barra; aun debe identificarse la fase de press. |
| BP-08 | Pose + Segmentation | Pose aporta hombros y una mascara persona-banco podria aportar contacto global; cabeza y gluteos pueden seguir siendo ambiguos. |
| UNSPECIFIED | Informacion no disponible aun | Sin causa de nulidad no puede decidirse que modalidad aporta evidencia. |

No se afirma que Segmentation resuelva todos los criterios. La futura fusion esta metodologicamente justificada porque las modalidades aportan evidencia complementaria, pero debera evaluarse sobre el split common congelado y con sus propias limitaciones.

## 13. Limitaciones

- Los subconjuntos por codigo son pequenos y solapados; BP-08 tiene N=1.
- La observabilidad es una clasificacion tecnica basada en inputs y reglas, no una variable medida experimentalmente.
- No hay atribucion causal de features ni analisis de explicabilidad del modelo.
- Los resultados per-code no son metricas multiclase: PoseCNN sigue siendo binaria.
- Nueve invalidas originales no tienen codigo; una aparece en Test.
- No se analizaron predicciones de Train/Validation para generalizacion.

## 14. Trazabilidad

- Predicciones congeladas: `{PREDICTIONS_PATH}`.
- Split congelado: `{SPLIT_PATH}`.
- Labels: `{LABELS_PATH}`.
- Analisis por codigo: `{BY_CODE_PATH}`.
- Nulos por muestra: `{INVALID_SAMPLES_PATH}`.
- Taxonomia: `{OBSERVABILITY_PATH}`.
- Resumen de observabilidad: `{OBSERVABILITY_SUMMARY_PATH}`.

Este analisis no entreno PoseCNN, Segmentation CNN ni Fusion y no ejecuto inferencia.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")

    input_hashes_after = {str(path): file_sha256(path) for path in required}
    if input_hashes_before != input_hashes_after:
        raise RuntimeError("Un artefacto congelado cambio durante el analisis")

    manifest = {
        "analysis": "BP_FRONT_POSECNN_FAULT_OBSERVABILITY_POSTHOC_V1",
        "inference_executed": False,
        "training_executed": False,
        "threshold_changed": False,
        "frozen_confusion_matrix": matrix.tolist(),
        "test_samples": len(merged),
        "invalid_test_samples": len(test_invalid),
        "valid_test_samples": len(test_valid),
        "real_fault_codes": real_codes,
        "full_invalid_multicode_samples": len(full_multicode),
        "test_invalid_multicode_samples": len(test_multicode),
        "input_sha256_before_and_after": input_hashes_after,
        "outputs": [
            str(BY_CODE_PATH),
            str(INVALID_SAMPLES_PATH),
            str(OBSERVABILITY_PATH),
            str(OBSERVABILITY_SUMMARY_PATH),
            str(REPORT_PATH),
        ],
    }
    ANALYSIS_MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    print("ANALISIS POST-HOC BP_FRONT COMPLETADO")
    print(f"Codigos reales: {real_codes}")
    print(f"Test: {len(merged)} | invalidos={len(test_invalid)} validos={len(test_valid)}")
    print(f"Confusion matrix congelada: {matrix.tolist()}")
    print(f"FP={len(false_positives)} | TN={len(true_negatives)}")
    print(f"Reporte: {REPORT_PATH}")


if __name__ == "__main__":
    run_analysis()
