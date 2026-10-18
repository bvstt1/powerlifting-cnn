"""Adaptador: FusionInferenceLEFT('full'|'sin_relacion', 'late'|'embedding')."""
from training.bp_side_fusion_v1.inference import FusionInference


class FusionInferenceLEFT(FusionInference):
    def __init__(self,variant,kind,device=None):
        super().__init__("left",variant,kind,device)
