"""Pruebas del módulo de segmentación.

python -m unittest judge.tests.test_segmentation
"""

import unittest

import numpy as np

from judge import config as cfg
from judge.config import VIEWS
from judge.segmentation import (CLASS_NAMES, Instance, SegmentationNotAvailable, SegmentationResult,
                                SegmentationStage, draw_segmentation)
from judge.segmentation.yolo_seg import YoloSegmentationExtractor, result_to_instances


class Tensor:
    def __init__(self, array):
        self.array = np.asarray(array)

    def cpu(self):
        return self

    def numpy(self):
        return self.array

    def __len__(self):
        return len(self.array)


def rect_mask(shape, x1, y1, x2, y2):
    mask = np.zeros(shape, dtype=np.float32)
    mask[y1:y2, x1:x2] = 1.0
    return mask


class FakeResult:
    """Resultado de Ultralytics con instancias (class_id, conf, (x1, y1, x2, y2))."""

    def __init__(self, shape, items, names=None, drop_masks=False, mask_shape=None):
        self.names = names or dict(CLASS_NAMES)
        if not items:
            self.boxes, self.masks = None, None
            return
        self.boxes = type("B", (), {})()
        self.boxes.xyxy = Tensor([[x1, y1, x2, y2] for _, _, (x1, y1, x2, y2) in items])
        self.boxes.cls = Tensor([c for c, _, _ in items])
        self.boxes.conf = Tensor([p for _, p, _ in items])
        self.boxes.__class__.__len__ = lambda s: len(s.cls)
        masks = [rect_mask(mask_shape or shape, x1, y1, x2, y2) for _, _, (x1, y1, x2, y2) in items]
        self.masks = None if drop_masks else type("M", (), {})()
        if self.masks is not None:
            self.masks.data = Tensor(np.stack(masks))
            self.masks.xy = [np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
                             for _, _, (x1, y1, x2, y2) in items]


class FakeSegModel:
    def __init__(self, names=None, items=None):
        self.names = names or dict(CLASS_NAMES)
        self.items = items or []
        self.calls = []

    def __call__(self, images, **kwargs):
        self.calls.append({"n": len(images), **kwargs})
        return [FakeResult(image.shape[:2], self.items, self.names) for image in images]


SHAPE = (48, 64)
ITEMS = [(0, 0.9, (10, 20, 40, 24)), (1, 0.8, (5, 25, 60, 40)), (2, 0.7, (20, 5, 35, 30)), (0, 0.4, (0, 0, 4, 4))]


class ResultConversionTest(unittest.TestCase):
    def test_no_detections(self):
        result = result_to_instances(FakeResult(SHAPE, []), SHAPE)
        self.assertEqual(result.instances, ())
        self.assertEqual(result.image_shape, SHAPE)
        self.assertIsNone(result.best("bar"))

    def test_instance_fields_and_classes(self):
        result = result_to_instances(FakeResult(SHAPE, ITEMS), SHAPE)
        self.assertEqual([i.class_name for i in result.instances], ["bar", "bench", "person", "bar"])
        bar = result.best("bar")
        self.assertAlmostEqual(bar.confidence, 0.9, places=5)
        self.assertEqual(bar.bbox_xyxy.tolist(), [10.0, 20.0, 40.0, 24.0])
        self.assertEqual(bar.mask.dtype, bool)
        self.assertEqual(bar.mask.shape, SHAPE)
        self.assertEqual(bar.area, 30 * 4)
        self.assertEqual(bar.polygon.shape, (4, 2))
        self.assertEqual(len(result.by_class("bar")), 2)
        self.assertTrue(result.has("person"))

    def test_best_picks_highest_confidence(self):
        result = result_to_instances(FakeResult(SHAPE, ITEMS), SHAPE)
        self.assertAlmostEqual(result.best("bar").confidence, 0.9, places=5)

    def test_mask_size_must_match_frame(self):
        with self.assertRaises(RuntimeError):
            result_to_instances(FakeResult(SHAPE, ITEMS, mask_shape=(24, 32)), SHAPE)

    def test_boxes_without_masks_is_an_error(self):
        with self.assertRaises(RuntimeError):
            result_to_instances(FakeResult(SHAPE, ITEMS, drop_masks=True), SHAPE)

    def test_unexpected_class_is_an_error(self):
        with self.assertRaises(RuntimeError):
            result_to_instances(FakeResult(SHAPE, [(1, 0.9, (1, 1, 5, 5))], names={0: "bar", 1: "plate", 2: "person"}), SHAPE)

    def test_empty_mask_is_dropped(self):
        result = result_to_instances(FakeResult(SHAPE, [(0, 0.9, (5, 5, 5, 5)), (2, 0.8, (1, 1, 9, 9))]), SHAPE)
        self.assertEqual([i.class_name for i in result.instances], ["person"])


class ExtractorTest(unittest.TestCase):
    def test_rejects_model_with_other_classes(self):
        with self.assertRaises(ValueError):
            YoloSegmentationExtractor("x.pt", model=FakeSegModel(names={0: "bar", 1: "plate", 2: "person"}), device="cpu")

    def test_uses_training_inference_parameters_and_batches(self):
        model = FakeSegModel(items=ITEMS)
        extractor = YoloSegmentationExtractor("x.pt", model=model, device="cpu")
        images = [np.zeros((*SHAPE, 3), dtype=np.uint8)] * 2
        results = extractor.extract_many(images)
        self.assertEqual(len(results), 2)
        self.assertEqual(model.calls, [{"n": 2, "imgsz": 960, "conf": 0.25, "iou": 0.70, "retina_masks": True,
                                        "augment": False, "device": "cpu", "verbose": False}])
        self.assertEqual(extractor.extract_many([]), [])

    def test_single_extract(self):
        extractor = YoloSegmentationExtractor("x.pt", model=FakeSegModel(items=ITEMS), device="cpu")
        self.assertEqual(len(extractor.extract(np.zeros((*SHAPE, 3), dtype=np.uint8)).instances), 4)


def make_instance(class_id, box, shape=SHAPE, confidence=0.9, polygon=True):
    x1, y1, x2, y2 = box
    mask = np.zeros(shape, dtype=bool)
    mask[y1:y2, x1:x2] = True
    poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32) if polygon else None
    return Instance(class_id, CLASS_NAMES[class_id], confidence, np.array(box, dtype=np.float32), mask, poly)


class DrawingTest(unittest.TestCase):
    def setUp(self):
        self.base = np.full((*SHAPE, 3), 100, dtype=np.uint8)

    def test_blends_inside_mask_and_leaves_the_rest_untouched(self):
        result = SegmentationResult((make_instance(1, (10, 10, 30, 30), polygon=False),), SHAPE)
        out = draw_segmentation(self.base, result, labels=False)
        self.assertEqual(out[0, 0].tolist(), [100, 100, 100])               # fuera de la máscara
        self.assertEqual(out[39, 50].tolist(), [100, 100, 100])
        self.assertNotEqual(out[20, 20].tolist(), [100, 100, 100])           # dentro
        # mezcla con alfa 0.35 hacia magenta (255, 0, 255)
        self.assertEqual(out[20, 20].tolist(), [int(100 * 0.65 + 255 * 0.35), int(100 * 0.65), int(100 * 0.65 + 255 * 0.35)])

    def test_original_image_is_not_modified(self):
        result = SegmentationResult((make_instance(0, (5, 5, 20, 15)),), SHAPE)
        draw_segmentation(self.base, result)
        self.assertTrue((self.base == 100).all())

    def test_bar_is_drawn_on_top_of_person_and_bench(self):
        overlap = (10, 10, 30, 30)
        instances = [make_instance(0, overlap, polygon=False), make_instance(2, overlap, polygon=False),
                     make_instance(1, overlap, polygon=False)]
        forward = draw_segmentation(self.base, SegmentationResult(tuple(instances), SHAPE), labels=False)
        backward = draw_segmentation(self.base, SegmentationResult(tuple(reversed(instances)), SHAPE), labels=False)
        self.assertTrue((forward == backward).all())        # el orden de llegada no cambia el dibujo
        # Banco, luego persona y al final la barra (alfa 0.65, amarillo BGR 0,255,255), truncando a uint8.
        pixel = np.array([100, 100, 100], dtype=np.uint8)
        for color, alpha in (((255, 0, 255), 0.35), ((0, 165, 255), 0.30), ((0, 255, 255), 0.65)):
            pixel = (pixel.astype(np.float32) * (1 - alpha) + np.array(color, dtype=np.float32) * alpha).astype(np.uint8)
        self.assertEqual(forward[20, 20].tolist(), pixel.tolist())

    def test_contour_is_drawn_from_polygon(self):
        result = SegmentationResult((make_instance(0, (10, 10, 40, 30)),), SHAPE)
        out = draw_segmentation(self.base, result, labels=False)
        self.assertEqual(out[10, 25].tolist(), [0, 255, 255])                # borde del polígono, amarillo (BGR)

    def test_empty_result_draws_nothing(self):
        out = draw_segmentation(self.base, SegmentationResult((), SHAPE))
        self.assertTrue((out == self.base).all())

    def test_box_outside_image_is_clipped(self):
        instance = make_instance(2, (0, 0, 10, 10))
        instance = Instance(instance.class_id, instance.class_name, 0.9, np.array([-5, -5, 500, 500], dtype=np.float32),
                            instance.mask, None)
        out = draw_segmentation(self.base, SegmentationResult((instance,), SHAPE), labels=False)
        self.assertNotEqual(out[5, 5].tolist(), [100, 100, 100])


class FakeFactoryExtractor:
    def __init__(self, model):
        self.model = model
        self.batches = []
        self.closed = False

    def extract_many(self, images):
        self.batches.append(len(images))
        return [SegmentationResult((make_instance(2, (2, 2, 20, 20), shape=image.shape[:2]),), image.shape[:2])
                for image in images]

    def close(self):
        self.closed = True


class StageTest(unittest.TestCase):
    def stage(self):
        created = []

        def factory(model):
            extractor = FakeFactoryExtractor(model)
            created.append(extractor)
            return extractor

        return SegmentationStage(cfg.load_config(exercise="bench", check_files=False), factory), created

    def images(self):
        return {view: np.zeros((*SHAPE, 3), dtype=np.uint8) for view in VIEWS}

    def test_only_bench_is_supported(self):
        for exercise in ("squat", "deadlift"):
            with self.assertRaises(SegmentationNotAvailable) as context:
                SegmentationStage(cfg.load_config(exercise=exercise, check_files=False), FakeFactoryExtractor)
            self.assertIn(exercise, str(context.exception))

    def test_front_has_its_own_model_and_laterals_share_one(self):
        stage, created = self.stage()
        self.assertEqual(sorted(e.model.name for e in created), ["bp_front", "bp_side"])
        self.assertNotEqual(next(e for e in created if e.model.name == "bp_front").model.model,
                            next(e for e in created if e.model.name == "bp_side").model.model)
        results = stage.process(self.images())
        self.assertEqual(set(results), set(VIEWS))
        self.assertEqual(sorted(sum((e.batches for e in created), [])), [1, 2])

    def test_processes_only_the_views_received(self):
        stage, created = self.stage()
        results = stage.process({"left": self.images()["left"]})
        self.assertEqual(set(results), {"left"})
        self.assertEqual(sorted(sum((e.batches for e in created), [])), [1])

    def test_annotate_draws_over_the_given_images(self):
        stage, _ = self.stage()
        images = self.images()
        results = stage.process(images)
        annotated = stage.annotate(images, results)
        self.assertTrue(all(img.sum() == 0 for img in images.values()))
        self.assertTrue(all(img.sum() > 0 for img in annotated.values()))

    def test_close_closes_every_extractor(self):
        stage, created = self.stage()
        stage.close()
        self.assertTrue(all(e.closed for e in created))


class RealModelsTest(unittest.TestCase):
    """Carga los modelos reales de banca y los ejecuta sobre un frame negro."""

    def test_real_models_have_the_expected_classes_and_run(self):
        config = cfg.load_config(exercise="bench", check_files=False)
        if cfg.missing_files(config):
            self.skipTest("Modelos locales ausentes en esta copia del repositorio")
        with SegmentationStage(config) as stage:
            images = {view: np.zeros((480, 640, 3), dtype=np.uint8) for view in VIEWS}
            results = stage.process(images)
            for view in VIEWS:
                self.assertEqual(results[view].image_shape, (480, 640))
                for instance in results[view].instances:
                    self.assertEqual(instance.mask.shape, (480, 640))
                    self.assertIn(instance.class_name, CLASS_NAMES.values())
            self.assertEqual(set(stage.annotate(images, results)), set(VIEWS))


if __name__ == "__main__":
    unittest.main()
