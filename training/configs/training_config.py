import argparse
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = SCRIPT_DIR.parent


PROFILES = {
    "dl_front": {
        "display_name": "Deadlift Frontal",
        "exercise": "dl",
        "view": "front",
        "keypoints_dir": PROJECT_DIR / "keypoints" / "dl" / "mediapipe" / "front",
        "csv_path": PROJECT_DIR / "etiquetado" / "dl" / "dl_front.csv",
        "data_dir": SCRIPT_DIR / "data" / "dl" / "front" / "posecnn",
        "model_dir": SCRIPT_DIR / "models" / "dl" / "front" / "posecnn" / "runs" / "legacy_default",
        "results_dir": SCRIPT_DIR / "results" / "dl" / "front" / "posecnn" / "runs" / "legacy_default",
        "expected_keypoints": 25,
        "normalization": "dl_front",
        "split_strategy": "stratified",
        "augmentation": ("translation", "scaling", "noise", "rotation", "temporal_shift"),
    },
    "dl_side": {
        "display_name": "Deadlift Lateral",
        "exercise": "dl",
        "view": "side",
        "keypoints_dir": PROJECT_DIR / "keypoints" / "dl" / "yolo" / "side",
        "csv_path": PROJECT_DIR / "etiquetado" / "dl" / "dl_side.csv",
        "data_dir": SCRIPT_DIR / "data" / "dl" / "side" / "posecnn",
        "model_dir": SCRIPT_DIR / "models" / "dl" / "side" / "posecnn" / "runs" / "legacy_default",
        "results_dir": SCRIPT_DIR / "results" / "dl" / "side" / "posecnn" / "runs" / "legacy_default",
        "expected_keypoints": 2,
        "normalization": "dl_side",
        "split_strategy": "grouped",
        "augmentation": ("translation", "scaling", "noise", "temporal_shift"),
    },
    "dl_left": {
        "display_name": "Deadlift Left",
        "exercise": "dl",
        "view": "left",
        "keypoints_dir": PROJECT_DIR / "keypoints" / "dl" / "yolo" / "left",
        "csv_path": PROJECT_DIR / "etiquetado" / "dl" / "dl_left.csv",
        "data_dir": SCRIPT_DIR / "data" / "dl" / "left" / "posecnn",
        "model_dir": SCRIPT_DIR / "models" / "dl" / "left" / "posecnn" / "runs" / "legacy_default",
        "results_dir": SCRIPT_DIR / "results" / "dl" / "left" / "posecnn" / "runs" / "legacy_default",
        "expected_keypoints": 2,
        "normalization": "dl_side",
        "split_strategy": "grouped",
        "shared_event_split": True,
        "augmentation": ("translation", "scaling", "noise", "temporal_shift"),
    },
    "dl_right": {
        "display_name": "Deadlift Right",
        "exercise": "dl",
        "view": "right",
        "keypoints_dir": PROJECT_DIR / "keypoints" / "dl" / "yolo" / "right",
        "csv_path": PROJECT_DIR / "etiquetado" / "dl" / "dl_right.csv",
        "data_dir": SCRIPT_DIR / "data" / "dl" / "right" / "posecnn",
        "model_dir": SCRIPT_DIR / "models" / "dl" / "right" / "posecnn" / "runs" / "legacy_default",
        "results_dir": SCRIPT_DIR / "results" / "dl" / "right" / "posecnn" / "runs" / "legacy_default",
        "expected_keypoints": 2,
        "normalization": "dl_side",
        "split_strategy": "grouped",
        "shared_event_split": True,
        "augmentation": ("translation", "scaling", "noise", "temporal_shift"),
    },
}

# Evento fisico que falta en la vista izquierda (hueco del video original).
# Evidencia (auditoria fase 1, NO autorreferencial):
# - dataset/dl/left tiene dl_001-dl_412 (412); dataset/dl/right dl_001-dl_413.
# - Conteo de frames por video (cv2) y T de keypoints (.npy) coinciden exacto:
#   left[N] == right[N] para N=303-309, y left[N] == right[N+1] para todo
#   N>=310 (frontera nitida 309/310, p. ej. left_310=159=right_311,
#   right_310=210; left_412=163=right_413).
# - Acuerdo de labels CSV: tramo 310-400, L[N] vs R[N+1] 80/91 > L[N] vs
#   R[N] 68/91; tramo >=401, L[N] vs R[N+1] 12/12.
# Conclusion: se elimino el original 310 en left y se renumero: left 310-412
# contienen los levantamientos 311-413. El evento 310 solo existe en right.
DL_LEFT_GAP_EVENT = 310

# Particion comun de eventos fisicos compartida por dl_left y dl_right.
DL_SHARED_DATA_DIR = SCRIPT_DIR / "data" / "dl" / "shared"
DL_SHARED_RESULTS_DIR = SCRIPT_DIR / "results" / "dl" / "shared" / "posecnn" / "runs"
COMMON_EVENT_SPLIT_PATH = DL_SHARED_DATA_DIR / "event_split.npz"
MANUAL_LABEL_AUDIT_PATH = DL_SHARED_DATA_DIR / "manual_label_audit.csv"


def get_config(profile_name):
    if profile_name not in PROFILES:
        available = ", ".join(sorted(PROFILES))
        raise ValueError(f"Perfil desconocido: {profile_name}. Disponibles: {available}")

    config = dict(PROFILES[profile_name])
    config["profile"] = profile_name
    config["target_frames"] = 170
    config["random_seed"] = 42
    config["preprocessed_dir"] = config["data_dir"] / "preprocessed"
    config["normalized_dir"] = config["data_dir"] / "normalized"
    config["split_dir"] = config["data_dir"] / "split"
    config["augmented_dir"] = config["data_dir"] / "augmented"
    config["raw_path"] = config["preprocessed_dir"] / "raw_sequences.npz"
    config["normalized_path"] = (
        config["normalized_dir"]
        / f"sequences_{config['target_frames']}.npz"
    )
    config["split_path"] = config["split_dir"] / "split_data.npz"
    config["augmented_path"] = config["augmented_dir"] / "augmented_data.npz"
    config["scaler_path"] = config["split_dir"] / "scaler.pkl"
    config["model_path"] = config["model_dir"] / "best_model.pth"
    config["history_path"] = config["model_dir"] / "history.npz"
    return config


def parse_config(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="dl_front",
        help="Configuracion del ejercicio y vista",
    )
    args = parser.parse_args()
    return get_config(args.profile)


def _source_video(video_id, view):
    """Descompone un id de video en (numero, vista) del video fuente.

    side M<=413 proviene de right M; side M>=414 proviene de left (M-413).
    Verificado por contenido: T(side M)==T(right M) y T(side M)==T(left M-413)
    para todo M, y labels identicos a los CSV fuente.
    """
    number = int(str(video_id).split("_")[-1])

    if view == "right":
        return number, "right"
    if view == "left":
        return number, "left"
    if view == "side":
        if number <= 413:
            return number, "right"
        return number - 413, "left"

    raise ValueError(f"Vista desconocida: {view}")


def physical_event(video_id, view):
    """UNICA fuente de verdad del mapping a evento fisico.

    Correspondencia auditada contra conteo de frames de los videos fuente:
    - right N -> evento N (001-413).
    - left N<310 -> evento N; left N>=310 -> evento N+1 (el evento fisico
      310 solo existe en right: left 310-412 contienen los levantamientos
      311-413; el evento 401 SI tiene ambas vistas).
    - side M<=413 -> evento M (proviene de right M).
    - side M>=414 -> left (M-413) con la regla left anterior
      (p. ej. side 722->evento 309, side 723->evento 311, side 825->413).

    Devuelve (group_id, vista_fuente). La regla +1 vive solo aqui.
    """
    number, source_view = _source_video(video_id, view)

    if source_view == "left" and number >= DL_LEFT_GAP_EVENT:
        number += 1

    return f"dl_event_{number:03d}", source_view


def dl_physical_event(video_id, view):
    """Compatibilidad: group_id para vistas left/right."""
    if view not in ("left", "right"):
        raise ValueError(f"Vista lateral desconocida: {view}")

    group_id, _ = physical_event(video_id, view)
    return group_id


def build_event_ids_for_view(video_ids, view):
    return np.asarray(
        [dl_physical_event(video_id, view) for video_id in video_ids]
    )


def uses_shared_event_split(config):
    return bool(config.get("shared_event_split", False))


def get_paired_profile(profile_name):
    if profile_name == "dl_left":
        return "dl_right"
    if profile_name == "dl_right":
        return "dl_left"
    return None


def get_event_split_path():
    return COMMON_EVENT_SPLIT_PATH


def build_sample_metadata(video_ids, config):
    video_ids = np.asarray(video_ids).astype(str)

    if config["profile"] in ("dl_left", "dl_right", "dl_side"):
        groups_views = [physical_event(video_id, config["view"]) for video_id in video_ids]
        return (
            np.asarray([group_id for group_id, _ in groups_views]),
            np.asarray([source_view for _, source_view in groups_views]),
        )

    return video_ids.copy(), np.full(len(video_ids), config["view"])
