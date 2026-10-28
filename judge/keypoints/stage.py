"""Etapa de keypoints: un extractor por vista según el ejercicio del config.

Las vistas laterales comparten el mismo modelo YOLO, así que se carga una sola vez y sus
dos frames se procesan en una única llamada (más rápido en CPU). MediaPipe tiene estado
temporal por cámara, por lo que cada vista usa su propia instancia.
"""

from collections import defaultdict

from judge.keypoints.drawing import draw_skeleton
from judge.keypoints.mediapipe_pose import MediaPipeFrontDeadliftExtractor
from judge.keypoints.yolo import YoloKeypointExtractor


def create_extractor(skeleton):
    """Extractor del `SkeletonModel` indicado."""
    if skeleton.backend == "yolo":
        return YoloKeypointExtractor(skeleton.model, skeleton.num_points, imgsz=skeleton.imgsz)
    if skeleton.backend == "mediapipe":
        return MediaPipeFrontDeadliftExtractor(skeleton.model)
    raise ValueError(f"Backend de keypoints desconocido: {skeleton.backend}")


class KeypointStage:
    """Extrae y dibuja los keypoints de las 3 vistas del ejercicio activo.

    `extractor_factory` permite inyectar extractores falsos en las pruebas.
    """

    def __init__(self, config, extractor_factory=create_extractor):
        self.config = config
        self._views = config.views()
        self._extractors = {}
        self._groups = defaultdict(list)       # nombre de skeleton compartido -> vistas
        for view, item in self._views.items():
            name = item.skeleton.name
            if item.skeleton.backend == "yolo":
                if name not in self._extractors:
                    self._extractors[name] = extractor_factory(item.skeleton)
                self._groups[name].append(view)
            else:
                self._extractors[(name, view)] = extractor_factory(item.skeleton)
                self._groups[(name, view)].append(view)

    def process(self, images, timestamps=None):
        """images: {vista: imagen BGR}. Devuelve {vista: KeypointResult} para las vistas recibidas."""
        timestamps = timestamps or {}
        results = {}
        for key, views in self._groups.items():
            views = [v for v in views if v in images]
            if not views:
                continue
            extractor = self._extractors[key]
            batch = [images[v] for v in views]
            stamps = [timestamps.get(v) for v in views]
            for view, result in zip(views, extractor.extract_many(batch, stamps)):
                results[view] = result
        return results

    def annotate(self, images, results, min_confidence=0.5, labels=False):
        """{vista: imagen con el esqueleto dibujado}."""
        return {view: draw_skeleton(images[view], results[view], self._views[view].skeleton,
                                    min_confidence, labels)
                for view in results}

    def close(self):
        for extractor in self._extractors.values():
            extractor.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
