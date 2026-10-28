"""Etapa de evaluación: una red por vista, un veredicto por vista y levantamiento."""

from judge.config import VIEWS
from judge.evaluation.base import INSUFFICIENT, Recording, ViewVerdict
from judge.evaluation.network import ViewEvaluator
from judge.evaluation.recorder import LiftRecorder


def create_view_evaluator(config, view, device="cpu"):
    item = config.view(view)
    return ViewEvaluator(item.evaluator, view, item.skeleton, config.threshold, config.window_frames, device,
                         config.allow_unverified)


class LiftEvaluator:
    """Evalúa un levantamiento completo con la red de cada vista del ejercicio activo.

    `evaluator_factory` permite inyectar redes falsas en las pruebas.
    """

    def __init__(self, config, device="cpu", evaluator_factory=create_view_evaluator):
        self.config = config
        self._evaluators = {view: evaluator_factory(config, view, device) for view in VIEWS}

    def new_recorder(self, max_frames=6000):
        """Grabadora con el número de puntos de cada vista."""
        return LiftRecorder({view: config_view.skeleton.num_points
                             for view, config_view in self.config.views().items()}, max_frames)

    def evaluate(self, recording):
        """Recording (o {vista: secuencia}) -> {vista: ViewVerdict}. Una vista fallida no afecta a las demás."""
        sequences = recording.sequences if isinstance(recording, Recording) else recording
        verdicts = {}
        for view, evaluator in self._evaluators.items():
            sequence = sequences.get(view)
            if sequence is None:
                verdicts[view] = ViewVerdict(view, INSUFFICIENT, None, "sin secuencia para esta vista", 0)
            else:
                verdicts[view] = evaluator.evaluate(sequence)
        return verdicts
