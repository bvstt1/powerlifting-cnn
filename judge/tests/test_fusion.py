"""Pruebas del módulo de fusión.

python -m unittest judge.tests.test_fusion
"""

import itertools
import unittest

from judge import config as cfg
from judge.config import VIEWS
from judge.evaluation import INSUFFICIENT, INVALID, VALID, ViewVerdict
from judge.fusion import DECISION_INVALID, DECISION_REVIEW, DECISION_VALID, FusionRule

V, N, I = VALID, INVALID, INSUFFICIENT


def verdicts(front, left, right):
    result = {}
    for view, label in zip(VIEWS, (front, left, right)):
        probability = None if label == INSUFFICIENT else (0.9 if label == VALID else 0.1)
        reason = "falta el punto en el primer frame" if label == INSUFFICIENT else None
        result[view] = ViewVerdict(view, label, probability, reason, 100)
    return result


RULE = FusionRule({0: "valid", 1: "manual_review", 2: "invalid", 3: "invalid"})

# (válidas, nulas, sin datos) -> decisión. Con 3 vistas hay exactamente 10 posibilidades.
EXPECTED = {
    (3, 0, 0): DECISION_VALID,
    (2, 1, 0): DECISION_REVIEW,       # 1 nula: revisión manual
    (1, 2, 0): DECISION_INVALID,      # 2 o más nulas: NULO
    (0, 3, 0): DECISION_INVALID,
    (2, 0, 1): DECISION_REVIEW,       # falta una vista que podría ser nula: no se aprueba solo
    (1, 1, 1): DECISION_REVIEW,
    (0, 2, 1): DECISION_INVALID,      # 2 nulas ya deciden aunque la tercera no pueda votar
    (1, 0, 2): DECISION_REVIEW,
    (0, 1, 2): DECISION_REVIEW,
    (0, 0, 3): DECISION_REVIEW,
}


def counts(combo):
    return (combo.count(V), combo.count(N), combo.count(I))

class RuleTest(unittest.TestCase):
    def test_every_combination_of_verdicts(self):
        """Las 27 combinaciones (3 etiquetas x 3 vistas): ninguna queda sin decisión."""
        combinations = list(itertools.product((V, N, I), repeat=3))
        self.assertEqual(len(combinations), 27)
        for combo in combinations:
            self.assertEqual(RULE.fuse(verdicts(*combo)).decision, EXPECTED[counts(combo)], combo)
        self.assertEqual({counts(c) for c in combinations}, set(EXPECTED))

    def test_original_three_judge_rule_when_every_view_voted(self):
        for combo in itertools.product((V, N), repeat=3):
            nulls = combo.count(N)
            expected = {0: DECISION_VALID, 1: DECISION_REVIEW}.get(nulls, DECISION_INVALID)
            self.assertEqual(RULE.fuse(verdicts(*combo)).decision, expected, combo)

    def test_a_valid_never_comes_from_fewer_than_three_valid_votes(self):
        for combo in itertools.product((V, N, I), repeat=3):
            if RULE.fuse(verdicts(*combo)).decision == DECISION_VALID:
                self.assertEqual(combo, (V, V, V))

    def test_two_nulls_stand_even_if_the_third_view_could_not_vote(self):
        for position in range(3):
            combo = [N, N, N]
            combo[position] = I
            self.assertEqual(RULE.fuse(verdicts(*combo)).decision, DECISION_INVALID)

    def test_result_does_not_depend_on_the_order_of_the_views(self):
        for combo in itertools.product((V, N, I), repeat=3):
            decisions = {RULE.fuse(verdicts(*perm)).decision for perm in itertools.permutations(combo)}
            self.assertEqual(len(decisions), 1, combo)

    def test_counts(self):
        decision = RULE.fuse(verdicts(V, N, I))
        self.assertEqual((decision.valid_votes, decision.invalid_votes), (1, 1))
        self.assertEqual(decision.insufficient_views, ("right",))
        self.assertTrue(decision.needs_review)

    def test_rule_comes_from_the_config(self):
        config = cfg.load_config(check_files=False)
        self.assertEqual(FusionRule.from_config(config).fuse(verdicts(V, V, V)).decision, DECISION_VALID)

    def test_custom_mapping_is_honored(self):
        strict = FusionRule({0: "valid", 1: "invalid", 2: "invalid", 3: "invalid"})
        self.assertEqual(strict.fuse(verdicts(V, V, N)).decision, DECISION_INVALID)
        # aun con la regla estricta, un VÁLIDO sigue exigiendo las 3 vistas votando
        lenient = FusionRule({0: "valid", 1: "valid", 2: "valid", 3: "valid"})
        self.assertEqual(lenient.fuse(verdicts(V, V, I)).decision, DECISION_REVIEW)


class InputValidationTest(unittest.TestCase):
    def test_requires_exactly_three_views(self):
        partial = verdicts(V, V, V)
        del partial["right"]
        with self.assertRaises(ValueError):
            RULE.fuse(partial)
        extra = {**verdicts(V, V, V), "top": ViewVerdict("top", VALID, 0.9)}
        with self.assertRaises(ValueError):
            RULE.fuse(extra)

    def test_unknown_label(self):
        bad = verdicts(V, V, V)
        bad["front"] = ViewVerdict("front", "maybe", 0.5)
        with self.assertRaises(ValueError):
            RULE.fuse(bad)

    def test_rule_must_cover_all_null_counts_and_use_known_decisions(self):
        with self.assertRaises(ValueError):
            FusionRule({0: "valid", 1: "manual_review", 2: "invalid"})
        with self.assertRaises(ValueError):
            FusionRule({0: "valid", 1: "maybe", 2: "invalid", 3: "invalid"})


class ExplanationTest(unittest.TestCase):
    def test_summary_for_a_view_that_could_not_vote(self):
        text = RULE.fuse(verdicts(V, I, V)).summary()
        self.assertIn("DECISIÓN: REVISIÓN MANUAL", text)
        self.assertIn("lateral izquierda", text)
        self.assertIn("sin datos suficientes", text)
        self.assertIn("falta el punto en el primer frame", text)
        self.assertIn("P válido = 0.90", text)

    def test_each_outcome_has_its_own_explanation(self):
        valid = RULE.fuse(verdicts(V, V, V)).explanation
        invalid = RULE.fuse(verdicts(N, N, V)).explanation
        one_null = RULE.fuse(verdicts(N, V, V)).explanation
        missing = RULE.fuse(verdicts(V, V, I)).explanation
        self.assertEqual(len({valid, invalid, one_null, missing}), 4)
        self.assertIn("Exactamente 1 vista votó NULO", one_null)
        self.assertIn("no pudo votar", missing)

    def test_display_names(self):
        self.assertEqual(RULE.fuse(verdicts(V, V, V)).display, "VÁLIDO")
        self.assertEqual(RULE.fuse(verdicts(N, N, N)).display, "NULO")
        self.assertEqual(RULE.fuse(verdicts(N, V, V)).display, "REVISIÓN MANUAL")


if __name__ == "__main__":
    unittest.main()

