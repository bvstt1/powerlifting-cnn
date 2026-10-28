"""Carga y validación de la configuración del juez virtual.

Uso:
    from judge.config import load_config
    config = load_config()              # judge/config.yaml
    view = config.view("front")         # modelos de la vista frontal del ejercicio activo

Valida la estructura siempre. Con `check_files=True` (por defecto) comprueba además que
existan los modelos, checkpoints y scalers; el cargador no abre ningún modelo.
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"

VIEWS = ("front", "left", "right")
EXERCISES = ("squat", "bench", "deadlift")
DECISIONS = ("valid", "invalid", "manual_review")
BACKENDS = ("yolo", "mediapipe")
SCALER_TYPES = ("standard", "point_xy")
CAMERA_BACKENDS = ("auto", "dshow", "msmf", "any")
CONFIDENCE_CHANNELS = ("yolo_confidence", "mediapipe_visibility")


class ConfigError(ValueError):
    """Configuración inválida; el mensaje lista todos los problemas encontrados."""


@dataclass(frozen=True)
class CaptureConfig:
    cameras: dict
    width: int
    height: int
    fps: int
    backend: str
    max_skew_ms: float
    process_fps: float = 2.0


@dataclass(frozen=True)
class ControlsConfig:
    start_stop: str
    cancel: str
    quit: str


@dataclass(frozen=True)
class PipelineConfig:
    preview_while_idle: bool
    log_directory: Path


@dataclass(frozen=True)
class SkeletonModel:
    name: str
    backend: str
    model: Path
    points: tuple
    num_points: int
    confidence_channel: str
    connections: tuple = ()
    drawable_points: int = None     # puntos que son posiciones dibujables (por defecto, todos)
    imgsz: int = 960                # tamaño de inferencia YOLO; el mismo con que se extrajeron los datos


@dataclass(frozen=True)
class SegmentationModel:
    name: str
    model: Path
    imgsz: int = 960
    conf: float = 0.25
    iou: float = 0.70


@dataclass(frozen=True)
class Evaluator:
    name: str
    recipe: str
    arch: str
    checkpoint: Path
    scaler: Path
    scaler_type: str
    provenance: str
    subset_indices: tuple = None
    mirror_x: bool = False
    verified: bool = True               # False: no reproduce sus predicciones de entrenamiento
    verification_note: str = None


@dataclass(frozen=True)
class ViewConfig:
    view: str
    skeleton: SkeletonModel
    evaluator: Evaluator
    segmentation: SegmentationModel = None

    def input_points(self):
        """Puntos que recibe la red tras el subconjunto, si lo hay."""
        if self.evaluator.subset_indices is not None:
            return len(self.evaluator.subset_indices)
        return self.skeleton.num_points


@dataclass(frozen=True)
class Config:
    exercise: str
    capture: CaptureConfig
    window_frames: int
    threshold: float
    fusion: dict
    exercises: dict = field(repr=False)
    allow_unverified: bool = False
    controls: ControlsConfig = None
    pipeline: PipelineConfig = None
    source_path: Path = None

    def key_codes(self):
        """{acción: código de tecla} para cv2.waitKey."""
        return {action: key_code(getattr(self.controls, action)) for action in ("start_stop", "cancel", "quit")}

    def view(self, name):
        return self.exercises[self.exercise][name]

    def views(self):
        return self.exercises[self.exercise]

    def decide(self, null_count):
        """Veredicto final dado el número de vistas que dieron NULO."""
        return self.fusion[null_count]


def _path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _require(mapping, keys, where, errors):
    ok = isinstance(mapping, dict)
    if not ok:
        errors.append(f"{where}: se esperaba un mapa")
        return False
    for key in keys:
        if key not in mapping:
            errors.append(f"{where}: falta '{key}'")
            ok = False
    return ok


def _build_capture(raw, errors):
    if not _require(raw, ("cameras", "width", "height", "fps", "backend", "max_skew_ms"), "capture", errors):
        return None
    cameras = raw["cameras"]
    if not _require(cameras, VIEWS, "capture.cameras", errors):
        return None
    indices = [cameras[v] for v in VIEWS]
    if not all(isinstance(i, int) and not isinstance(i, bool) and i >= 0 for i in indices):
        errors.append("capture.cameras: los índices deben ser enteros >= 0")
    elif len(set(indices)) != len(indices):
        errors.append(f"capture.cameras: índices repetidos {indices}; cada vista necesita su propia cámara")
    for key in ("width", "height", "fps"):
        if not isinstance(raw[key], int) or raw[key] <= 0:
            errors.append(f"capture.{key}: debe ser un entero positivo")
    if raw["backend"] not in CAMERA_BACKENDS:
        errors.append(f"capture.backend: '{raw['backend']}' no es uno de {CAMERA_BACKENDS}")
    if not isinstance(raw["max_skew_ms"], (int, float)) or raw["max_skew_ms"] <= 0:
        errors.append("capture.max_skew_ms: debe ser positivo")
    process_fps = raw.get("process_fps", 2.0)
    if not isinstance(process_fps, (int, float)) or isinstance(process_fps, bool) or process_fps <= 0:
        errors.append("capture.process_fps: debe ser positivo")
    return CaptureConfig(dict(cameras), raw["width"], raw["height"], raw["fps"],
                         raw["backend"], float(raw["max_skew_ms"]), float(process_fps))


NAMED_KEYS = {"space": 32, "esc": 27, "enter": 13}


def key_code(name):
    """Código de tecla de cv2.waitKey para un nombre del config ('space', 'esc', 'enter' o un carácter)."""
    name = str(name)
    if name.lower() in NAMED_KEYS:
        return NAMED_KEYS[name.lower()]
    if len(name) == 1:
        return ord(name.lower())
    raise ValueError(f"Tecla desconocida: '{name}'")


def _build_controls(raw, errors):
    if not _require(raw, ("start_stop", "cancel", "quit"), "controls", errors):
        return None
    codes = {}
    for action in ("start_stop", "cancel", "quit"):
        try:
            codes[action] = key_code(raw[action])
        except ValueError as error:
            errors.append(f"controls.{action}: {error}")
    if len(codes) == 3 and len(set(codes.values())) != 3:
        errors.append("controls: cada acción necesita una tecla distinta")
    return ControlsConfig(str(raw["start_stop"]), str(raw["cancel"]), str(raw["quit"]))


def _build_pipeline(raw, errors):
    if not _require(raw, ("preview_while_idle", "log_directory"), "pipeline", errors):
        return None
    if not isinstance(raw["preview_while_idle"], bool):
        errors.append("pipeline.preview_while_idle: debe ser true o false")
    if not raw["log_directory"]:
        errors.append("pipeline.log_directory: no puede estar vacío")
    return PipelineConfig(bool(raw["preview_while_idle"]), _path(raw["log_directory"] or "."))


def _build_fusion(raw, errors):
    mapping = (raw or {}).get("decisions_by_null_count")
    if not isinstance(mapping, dict):
        errors.append("fusion.decisions_by_null_count: se esperaba un mapa")
        return {}
    expected = set(range(len(VIEWS) + 1))
    if set(mapping) != expected:
        errors.append(f"fusion.decisions_by_null_count: debe cubrir exactamente {sorted(expected)} vistas nulas")
    for count, decision in mapping.items():
        if decision not in DECISIONS:
            errors.append(f"fusion: decisión '{decision}' para {count} nulas no es una de {DECISIONS}")
    return dict(mapping)


def _build_skeletons(raw, errors):
    result = {}
    for name, item in (raw or {}).items():
        where = f"skeleton_models.{name}"
        if not _require(item, ("backend", "model", "points", "confidence_channel"), where, errors):
            continue
        if item["backend"] not in BACKENDS:
            errors.append(f"{where}.backend: '{item['backend']}' no es uno de {BACKENDS}")
        if item["confidence_channel"] not in CONFIDENCE_CHANNELS:
            errors.append(f"{where}.confidence_channel: valor inválido '{item['confidence_channel']}'")
        points = tuple(item["points"])
        num_points = int(item.get("num_points", len(points)))
        if item["backend"] == "yolo" and num_points != len(points):
            errors.append(f"{where}: num_points={num_points} no coincide con {len(points)} puntos")
        if len(set(points)) != len(points):
            errors.append(f"{where}: nombres de punto repetidos")
        drawable = int(item.get("drawable_points", num_points))
        if not 0 < drawable <= num_points:
            errors.append(f"{where}.drawable_points: debe estar entre 1 y {num_points}")
        connections = tuple(tuple(pair) for pair in item.get("connections", ()))
        if any(len(pair) != 2 or not all(isinstance(i, int) and 0 <= i < drawable for i in pair)
               for pair in connections):
            errors.append(f"{where}.connections: cada par debe tener 2 índices entre 0 y {drawable - 1}")
        imgsz = item.get("imgsz", 960)
        if not isinstance(imgsz, int) or imgsz <= 0:
            errors.append(f"{where}.imgsz: debe ser un entero positivo")
        result[name] = SkeletonModel(name, item["backend"], _path(item["model"]), points, num_points,
                                     item["confidence_channel"], connections, drawable, imgsz)
    return result


def _build_segmentations(raw, errors):
    result = {}
    for name, item in (raw or {}).items():
        where = f"segmentation_models.{name}"
        if not _require(item, ("model",), where, errors):
            continue
        imgsz, conf, iou = item.get("imgsz", 960), item.get("conf", 0.25), item.get("iou", 0.70)
        if not isinstance(imgsz, int) or imgsz <= 0:
            errors.append(f"{where}.imgsz: debe ser un entero positivo")
        for key, value in (("conf", conf), ("iou", iou)):
            if not isinstance(value, (int, float)) or not 0 < value <= 1:
                errors.append(f"{where}.{key}: debe estar entre 0 y 1")
        result[name] = SegmentationModel(name, _path(item["model"]), imgsz, float(conf), float(iou))
    return result


def _build_evaluators(raw, errors):
    result = {}
    for name, item in (raw or {}).items():
        where = f"evaluators.{name}"
        if not _require(item, ("recipe", "arch", "checkpoint", "scaler", "scaler_type", "provenance"), where, errors):
            continue
        if item["scaler_type"] not in SCALER_TYPES:
            errors.append(f"{where}.scaler_type: '{item['scaler_type']}' no es uno de {SCALER_TYPES}")
        subset = item.get("subset_indices")
        if subset is not None and (len(set(subset)) != len(subset) or any(i < 0 for i in subset)):
            errors.append(f"{where}.subset_indices: índices repetidos o negativos")
        verified = item.get("verified", True)
        if not isinstance(verified, bool):
            errors.append(f"{where}.verified: debe ser true o false")
        elif not verified and not item.get("verification_note"):
            errors.append(f"{where}: una red con verified: false debe explicar el motivo en verification_note")
        result[name] = Evaluator(name, item["recipe"], item["arch"], _path(item["checkpoint"]),
                                 _path(item["scaler"]), item["scaler_type"], item["provenance"],
                                 None if subset is None else tuple(subset), bool(item.get("mirror_x", False)),
                                 verified is True, " ".join(str(item.get("verification_note", "")).split()) or None)
    return result


def _build_exercises(raw, skeletons, segmentations, evaluators, errors):
    result = {}
    if not isinstance(raw, dict):
        errors.append("exercises: se esperaba un mapa")
        return result
    for exercise in EXERCISES:
        if exercise not in raw:
            errors.append(f"exercises: falta '{exercise}'")
            continue
        views = (raw[exercise] or {}).get("views", {})
        built = {}
        for view in VIEWS:
            where = f"exercises.{exercise}.views.{view}"
            if view not in views:
                errors.append(f"{where}: falta la vista")
                continue
            item = views[view]
            if not _require(item, ("skeleton", "evaluator"), where, errors):
                continue
            skeleton, evaluator = skeletons.get(item["skeleton"]), evaluators.get(item["evaluator"])
            if skeleton is None:
                errors.append(f"{where}.skeleton: '{item['skeleton']}' no existe en skeleton_models")
            if evaluator is None:
                errors.append(f"{where}.evaluator: '{item['evaluator']}' no existe en evaluators")
            segmentation = None
            if exercise == "bench":
                segmentation = segmentations.get(item.get("segmentation"))
                if segmentation is None:
                    errors.append(f"{where}: el press de banca exige 'segmentation' válida")
            elif "segmentation" in item:
                errors.append(f"{where}: solo el press de banca usa segmentación")
            if skeleton and evaluator:
                built[view] = ViewConfig(view, skeleton, evaluator, segmentation)
        result[exercise] = built
    unknown = set(raw) - set(EXERCISES)
    if unknown:
        errors.append(f"exercises: ejercicios desconocidos {sorted(unknown)}")
    return result


def _check_lateral_sharing(exercises, errors):
    """Las vistas laterales comparten skeleton y segmentación, pero no la red de evaluación."""
    for exercise, views in exercises.items():
        if len(views) != len(VIEWS):
            continue
        if views["left"].skeleton.name != views["right"].skeleton.name:
            errors.append(f"{exercise}: left y right deben compartir el modelo de skeleton")
        left_seg, right_seg = views["left"].segmentation, views["right"].segmentation
        if left_seg and right_seg and left_seg.name != right_seg.name:
            errors.append(f"{exercise}: left y right deben compartir el modelo de segmentación")
        if views["left"].evaluator.name == views["right"].evaluator.name:
            errors.append(f"{exercise}: cada vista lateral necesita su propia red de evaluación")


def _check_input_shapes(exercises, errors):
    for exercise, views in exercises.items():
        for view, item in views.items():
            subset = item.evaluator.subset_indices
            if subset is not None and max(subset) >= item.skeleton.num_points:
                errors.append(f"{exercise}.{view}: subset_indices excede los {item.skeleton.num_points} puntos")


def missing_files(config):
    """Archivos referenciados por el ejercicio activo que no existen."""
    missing = []
    for view in config.views().values():
        paths = [view.skeleton.model, view.evaluator.checkpoint, view.evaluator.scaler]
        if view.segmentation:
            paths.append(view.segmentation.model)
        missing += [p for p in paths if not p.is_file()]
    return sorted(set(missing))


def load_config(path=None, exercise=None, check_files=True):
    """Carga y valida la configuración. `exercise` reemplaza al del archivo."""
    path = Path(path) if path else DEFAULT_CONFIG
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    errors = []
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: el archivo no contiene un mapa YAML")
    chosen = exercise or raw.get("exercise")
    if chosen not in EXERCISES:
        errors.append(f"exercise: '{chosen}' no es uno de {EXERCISES}")
    capture = _build_capture(raw.get("capture"), errors)
    controls = _build_controls(raw.get("controls"), errors)
    pipeline = _build_pipeline(raw.get("pipeline"), errors)
    fusion = _build_fusion(raw.get("fusion"), errors)
    skeletons = _build_skeletons(raw.get("skeleton_models"), errors)
    segmentations = _build_segmentations(raw.get("segmentation_models"), errors)
    evaluators = _build_evaluators(raw.get("evaluators"), errors)
    exercises = _build_exercises(raw.get("exercises"), skeletons, segmentations, evaluators, errors)
    _check_lateral_sharing(exercises, errors)
    _check_input_shapes(exercises, errors)

    evaluation = raw.get("evaluation") or {}
    window, threshold = evaluation.get("window_frames"), evaluation.get("threshold")
    if not isinstance(window, int) or window <= 0:
        errors.append("evaluation.window_frames: debe ser un entero positivo")
    if not isinstance(threshold, (int, float)) or not 0 < threshold < 1:
        errors.append("evaluation.threshold: debe estar entre 0 y 1")
    allow_unverified = evaluation.get("allow_unverified", False)
    if not isinstance(allow_unverified, bool):
        errors.append("evaluation.allow_unverified: debe ser true o false")
    if errors:
        raise ConfigError(f"{path}:\n  - " + "\n  - ".join(errors))

    config = Config(chosen, capture, window, float(threshold), fusion, exercises, allow_unverified,
                    controls, pipeline, path)
    if check_files:
        absent = missing_files(config)
        if absent:
            listing = "\n  - ".join(str(p.relative_to(ROOT)) for p in absent)
            raise ConfigError(f"Archivos faltantes para '{chosen}':\n  - {listing}")
    return config
