"""Adaptador: FusionInferenceRIGHT('full'|'sin_relacion', 'late'|'embedding')."""
from training.bp.experiments.bp_side_fusion_v1.inference import FusionInference


class FusionInferenceRIGHT(FusionInference):
    def __init__(self,variant,kind,device=None):
        super().__init__("right",variant,kind,device)
