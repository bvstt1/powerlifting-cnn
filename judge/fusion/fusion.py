"""Fusión de los veredictos de las 3 vistas en la decisión final.

Regla base (configurable en `fusion.decisions_by_null_count` de config.yaml), la de los 3 jueces:
    3 vistas VÁLIDAS             -> VÁLIDO
    2 o más vistas NULAS         -> NULO
    exactamente 1 vista NULA     -> REVISIÓN MANUAL

Una vista "sin datos suficientes" (la secuencia no cumple la calidad con que se entrenó la red)
no vota. Como no se puede afirmar nada con ella:
    - un VÁLIDO exige que las 3 vistas hayan votado VÁLIDO; si falta alguna -> REVISIÓN MANUAL;
    - un NULO se mantiene si ya hay 2 o más vistas NULAS, pues una tercera no lo cambiaría.
"""

from dataclasses import dataclass

from judge.config import VIEWS
from judge.evaluation.base import INSUFFICIENT, INVALID, VALID

DECISION_VALID = "valid"
DECISION_INVALID = "invalid"
DECISION_REVIEW = "manual_review"

DISPLAY = {DECISION_VALID: "VÁLIDO", DECISION_INVALID: "NULO", DECISION_REVIEW: "REVISIÓN MANUAL"}
VIEW_NAMES = {"front": "frontal", "left": "lateral izquierda", "right": "lateral derecha"}
LABEL_NAMES = {VALID: "VÁLIDO", INVALID: "NULO", INSUFFICIENT: "sin datos suficientes"}


@dataclass(frozen=True)
class FinalDecision:
    decision: str                   # valid | invalid | manual_review
    verdicts: dict                  # {vista: ViewVerdict}
    valid_votes: int
    invalid_votes: int
    insufficient_views: tuple       # vistas que no pudieron votar
    explanation: str                # por qué se llegó a esta decisión, para el juez

    @property
    def display(self):
        return DISPLAY[self.decision]

    @property
    def needs_review(self):
        return self.decision == DECISION_REVIEW

    def summary(self):
        """Texto de varias líneas con la decisión, el voto de cada vista y el motivo."""
        lines = [f"DECISIÓN: {self.display}", self.explanation]
        for view in VIEWS:
            verdict = self.verdicts[view]
            detail = LABEL_NAMES[verdict.label]
            if verdict.probability is not None:
                detail += f" (P válido = {verdict.probability:.2f})"
            if verdict.reason:
                detail += f" - {verdict.reason}"
            lines.append(f"  {VIEW_NAMES[view]}: {detail}")
        return "\n".join(lines)


class FusionRule:
    def __init__(self, decisions_by_null_count):
        expected = set(range(len(VIEWS) + 1))
        if set(decisions_by_null_count) != expected:
            raise ValueError(f"La regla debe cubrir exactamente {sorted(expected)} vistas nulas")
        if any(d not in DISPLAY for d in decisions_by_null_count.values()):
            raise ValueError(f"Decisiones válidas: {sorted(DISPLAY)}")
        self._decisions = dict(decisions_by_null_count)

    @classmethod
    def from_config(cls, config):
        return cls(config.fusion)

    def fuse(self, verdicts):
        """{vista: ViewVerdict} -> FinalDecision. Exige las 3 vistas."""
        if set(verdicts) != set(VIEWS):
            raise ValueError(f"Se requieren los veredictos de {VIEWS}, recibidos {sorted(verdicts)}")
        for view, verdict in verdicts.items():
            if verdict.label not in LABEL_NAMES:
                raise ValueError(f"{view}: etiqueta desconocida '{verdict.label}'")

        valid = sum(v.label == VALID for v in verdicts.values())
        invalid = sum(v.label == INVALID for v in verdicts.values())
        insufficient = tuple(view for view in VIEWS if verdicts[view].label == INSUFFICIENT)

        decision = self._decisions[invalid]
        if decision == DECISION_VALID and valid < len(VIEWS):
            decision = DECISION_REVIEW
        return FinalDecision(decision, dict(verdicts), valid, invalid, insufficient,
                             self._explain(decision, valid, invalid, insufficient))

    def _explain(self, decision, valid, invalid, insufficient):
        votes = f"{valid} válida(s), {invalid} nula(s)"
        if insufficient:
            names = ", ".join(VIEW_NAMES[v] for v in insufficient)
            votes += f", {len(insufficient)} sin datos suficientes ({names})"
        if decision == DECISION_VALID:
            return f"Las 3 vistas votaron VÁLIDO ({votes})."
        if decision == DECISION_INVALID:
            return f"Al menos 2 vistas votaron NULO ({votes})."
        if insufficient and invalid < 2:
            return (f"Ninguna decisión es segura porque una vista no pudo votar ({votes}); "
                    "se requiere revisión manual.")
        return f"Exactamente 1 vista votó NULO ({votes}); se requiere revisión manual."
