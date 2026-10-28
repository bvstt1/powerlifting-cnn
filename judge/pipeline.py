"""Pipeline del juez: orquesta captura, keypoints, segmentación, evaluación y fusión.

Flujo de un intento (el juez delimita el levantamiento con el teclado):

    start_lift()  ->  offer(frame_set) ...  ->  finish_lift()  ->  Attempt
                      (por cada conjunto       (evalúa cada vista, fusiona
                       de 3 frames)             y registra el intento)

`offer()` no bloquea: con `threaded=True` el procesamiento (varios segundos por conjunto en
CPU) corre en un hilo y se descartan los conjuntos que llegan mientras está ocupado, siempre
quedándose con el más reciente. Así la ventana y el teclado siguen respondiendo.
"""

import threading
import time

from judge.attempt_log import Attempt


class PipelineError(RuntimeError):
    """Uso inválido del pipeline."""


class JudgePipeline:
    def __init__(self, config, keypoints, evaluator, fusion, segmentation=None, attempt_log=None,
                 threaded=False, min_interval=0.0, clock=time.perf_counter):
        self.config = config
        self.keypoints = keypoints
        self.segmentation = segmentation
        self.evaluator = evaluator
        self.fusion = fusion
        self.attempt_log = attempt_log
        self.recorder = evaluator.new_recorder()
        self._min_interval = min_interval
        self._clock = clock
        self._threaded = threaded
        self._condition = threading.Condition()
        self._pending = None
        self._busy = False
        self._closing = False
        self._last_offer = None
        self.processed = 0
        self.skipped = 0
        self.errors = 0
        self.last_error = None
        self.last_annotated = None          # {vista: imagen con esqueleto (y barra)} del último conjunto procesado
        self.last_decision = None
        self._thread = None
        if threaded:
            self._thread = threading.Thread(target=self._worker, name="judge-processing", daemon=True)
            self._thread.start()

    # ── estado ──────────────────────────────────────────────────────────────
    @property
    def recording(self):
        return self.recorder.recording

    @property
    def state(self):
        return "recording" if self.recording else "idle"

    def frame_counts(self):
        return self.recorder.frame_counts()

    # ── procesamiento de un conjunto de frames ─────────────────────────────
    def offer(self, frame_set, force=False):
        """Entrega un conjunto de frames. Devuelve True si se aceptó para procesarlo.

        Fuera de un levantamiento solo se procesa si `pipeline.preview_while_idle` está activo.
        `force` ignora el tope de ritmo (`min_interval`).
        """
        wanted = self.recording or self.config.pipeline.preview_while_idle
        if not wanted:
            return False
        now = self._clock()
        if not force and self._last_offer is not None and now - self._last_offer < self._min_interval:
            return False
        if not self._threaded:
            self._last_offer = now
            self._process(frame_set)
            return True
        with self._condition:
            if self._busy:
                self.skipped += 1           # ocupado: el conjunto se descarta, llegará uno más reciente
                return False
            self._last_offer = now
            self._pending = frame_set
            self._condition.notify_all()
            return True

    def _process(self, frame_set):
        try:
            images = frame_set.images()
            stamps = {view: frame.timestamp for view, frame in frame_set.frames.items()}
            results = self.keypoints.process(images, stamps)
            self.recorder.add(results)
            annotated = self.keypoints.annotate(images, results)
            if self.segmentation is not None:
                annotated = self.segmentation.annotate(annotated, self.segmentation.process(images))
            self.last_annotated = annotated
            self.processed += 1
        except Exception as error:          # un frame fallido no debe tumbar el levantamiento
            self.errors += 1
            self.last_error = error

    def _worker(self):
        while True:
            with self._condition:
                while self._pending is None and not self._closing:
                    self._condition.wait()
                if self._closing:
                    return
                frame_set, self._pending, self._busy = self._pending, None, True
            try:
                self._process(frame_set)
            finally:
                with self._condition:
                    self._busy = False
                    self._condition.notify_all()

    def drain(self, timeout=None):
        """Espera a que termine el conjunto en proceso y descarta el pendiente."""
        if not self._threaded:
            return True
        deadline = None if timeout is None else self._clock() + timeout
        with self._condition:
            self._pending = None
            while self._busy:
                remaining = None if deadline is None else deadline - self._clock()
                if remaining is not None and remaining <= 0:
                    return False
                self._condition.wait(remaining)
        return True

    # ── ciclo del levantamiento ────────────────────────────────────────────
    def start_lift(self):
        self.drain()
        self.recorder.start()
        self.errors, self.last_error, self.skipped, self.processed = 0, None, 0, 0
        self._last_offer = None
        self.last_decision = None

    def cancel_lift(self):
        self.drain()
        self.recorder.cancel()

    def finish_lift(self, mosaic=None):
        """Termina el levantamiento, evalúa las 3 vistas, fusiona y registra el intento."""
        if not self.recording:
            raise PipelineError("No hay un levantamiento en curso")
        self.drain()
        recording = self.recorder.stop()
        decision = self.fusion.fuse(self.evaluator.evaluate(recording))
        path = None
        if self.attempt_log is not None:
            path = self.attempt_log.save(self.config.exercise, decision, recording, self.errors, mosaic)
        self.last_decision = decision
        return Attempt(self.config.exercise, decision, recording, path, self.errors)

    def close(self):
        if self._thread is not None:
            with self._condition:
                self._closing = True
                self._condition.notify_all()
            self._thread.join(timeout=60)
        self.keypoints.close()
        if self.segmentation is not None:
            self.segmentation.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
