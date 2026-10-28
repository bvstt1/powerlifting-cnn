"""Etapa de segmentación: solo press de banca.

El modelo frontal es exclusivo de la vista frontal; las vistas laterales comparten un
mismo modelo, que se carga una vez y procesa los dos frames en una sola llamada.
"""

from collections import defaultdict

from judge.segmentation.drawing import draw_segmentation
from judge.segmentation.yolo_seg import YoloSegmentationExtractor

SUPPORTED_EXERCISE = "bench"


class SegmentationNotAvailable(RuntimeError):
    """El ejercicio activo no usa segmentación."""


def create_extractor(model):
    """Extractor del `SegmentationModel` indicado."""
    return YoloSegmentationExtractor(model.model, imgsz=model.imgsz, conf=model.conf, iou=model.iou)


class SegmentationStage:
    """Segmenta barra, banco y persona en las vistas del press de banca.

    `extractor_factory` permite inyectar extractores falsos en las pruebas.
    """

    def __init__(self, config, extractor_factory=create_extractor):
        if config.exercise != SUPPORTED_EXERCISE:
            raise SegmentationNotAvailable(
                f"La segmentación solo se usa en press de banca; el ejercicio activo es '{config.exercise}'")
        self.config = config
        self._extractors = {}
        self._groups = defaultdict(list)        # nombre del modelo -> vistas que lo comparten
        for view, item in config.views().items():
            model = item.segmentation
            if model.name not in self._extractors:
                self._extractors[model.name] = extractor_factory(model)
            self._groups[model.name].append(view)

    def process(self, images):
        """images: {vista: imagen BGR}. Devuelve {vista: SegmentationResult} para las vistas recibidas."""
        results = {}
        for name, views in self._groups.items():
            views = [v for v in views if v in images]
            if not views:
                continue
            for view, result in zip(views, self._extractors[name].extract_many([images[v] for v in views])):
                results[view] = result
        return results

    def annotate(self, images, results, labels=True):
        """{vista: imagen con las máscaras dibujadas}. `images` puede traer ya el esqueleto."""
        return {view: draw_segmentation(images[view], results[view], labels) for view in results}

    def close(self):
        for extractor in self._extractors.values():
            extractor.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
