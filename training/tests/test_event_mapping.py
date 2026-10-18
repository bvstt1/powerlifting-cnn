"""Test del mapping fisico left/right (frontera del hueco en evento 310).

Evidencia independiente: conteo de frames por video (cv2) y T de keypoints
(.npy) coinciden exacto a ambos lados de la frontera:
  left[N] == right[N] para N<=309, left[N] == right[N+1] para N>=310.

Uso: python -m training.tests.test_event_mapping
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from training.configs.training_config import (
    DL_LEFT_GAP_EVENT,
    dl_physical_event,
    physical_event,
)


def keypoint_frames(view, video_id):
    base = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "keypoints", "dl", view, video_id + ".npy",
    )
    return int(np.load(base).shape[0])


def test_gap_constant():
    assert DL_LEFT_GAP_EVENT == 310, (
        f"Hueco esperado en evento 310, configurado={DL_LEFT_GAP_EVENT}"
    )


def test_mapping_boundary():
    assert dl_physical_event("dl_309", "left") == "dl_event_309"
    assert dl_physical_event("dl_310", "left") == "dl_event_311"
    assert dl_physical_event("dl_310", "right") == "dl_event_310"
    assert dl_physical_event("dl_311", "right") == "dl_event_311"


def test_mapping_tail():
    assert dl_physical_event("dl_400", "left") == "dl_event_401"
    assert dl_physical_event("dl_401", "left") == "dl_event_402"
    assert dl_physical_event("dl_412", "left") == "dl_event_413"
    assert dl_physical_event("dl_401", "right") == "dl_event_401"
    assert dl_physical_event("dl_413", "right") == "dl_event_413"


def test_event_310_right_only():
    left_events = {
        dl_physical_event("dl_%03d" % n, "left") for n in range(1, 413)
    }
    assert "dl_event_310" not in left_events, (
        "El evento 310 no debe existir en left"
    )
    assert dl_physical_event("dl_310", "right") == "dl_event_310"


def test_content_boundary():
    # Spot-checks de contenido (frames detectados por video).
    assert keypoint_frames("left", "dl_309") == keypoint_frames("right", "dl_309")
    assert keypoint_frames("left", "dl_310") == keypoint_frames("right", "dl_311")
    assert keypoint_frames("left", "dl_310") != keypoint_frames("right", "dl_310")
    assert keypoint_frames("left", "dl_400") == keypoint_frames("right", "dl_401")
    assert keypoint_frames("left", "dl_412") == keypoint_frames("right", "dl_413")


def test_boundary_308_312_left_right():
    # IDs fuente conocidos de la auditoria (conteo de frames/T literal).
    assert dl_physical_event("dl_308", "left") == "dl_event_308"
    assert dl_physical_event("dl_309", "left") == "dl_event_309"
    assert dl_physical_event("dl_310", "left") == "dl_event_311"
    assert dl_physical_event("dl_311", "left") == "dl_event_312"
    assert dl_physical_event("dl_312", "left") == "dl_event_313"
    assert dl_physical_event("dl_308", "right") == "dl_event_308"
    assert dl_physical_event("dl_309", "right") == "dl_event_309"
    assert dl_physical_event("dl_310", "right") == "dl_event_310"
    assert dl_physical_event("dl_311", "right") == "dl_event_311"
    assert dl_physical_event("dl_312", "right") == "dl_event_312"


def test_boundary_399_402_413_left_right():
    assert dl_physical_event("dl_399", "left") == "dl_event_400"
    assert dl_physical_event("dl_400", "left") == "dl_event_401"
    assert dl_physical_event("dl_401", "left") == "dl_event_402"
    assert dl_physical_event("dl_402", "left") == "dl_event_403"
    assert dl_physical_event("dl_399", "right") == "dl_event_399"
    assert dl_physical_event("dl_400", "right") == "dl_event_400"
    assert dl_physical_event("dl_401", "right") == "dl_event_401"
    assert dl_physical_event("dl_402", "right") == "dl_event_402"
    assert dl_physical_event("dl_413", "right") == "dl_event_413"


def test_boundary_side_right_part():
    for event in (308, 309, 310, 311, 312, 399, 400, 401, 402, 413):
        group_id, source_view = physical_event("dl_%03d" % event, "side")
        assert group_id == "dl_event_%03d" % event
        assert source_view == "right"


def test_boundary_side_left_part():
    # side 414-825 = left 001-412 (verificado por T y labels fuente).
    assert physical_event("dl_414", "side") == ("dl_event_001", "left")
    assert physical_event("dl_722", "side") == ("dl_event_309", "left")
    assert physical_event("dl_723", "side") == ("dl_event_311", "left")
    assert physical_event("dl_813", "side") == ("dl_event_401", "left")
    assert physical_event("dl_814", "side") == ("dl_event_402", "left")
    assert physical_event("dl_824", "side") == ("dl_event_412", "left")
    assert physical_event("dl_825", "side") == ("dl_event_413", "left")


def test_single_source_of_truth():
    # left/right/side comparten la misma funcion y el evento 310 solo
    # existe en right (y en side via right).
    assert physical_event("dl_310", "side") == ("dl_event_310", "right")
    left_events = {
        dl_physical_event("dl_%03d" % n, "left") for n in range(1, 413)
    }
    side_left_events = {
        physical_event("dl_%03d" % m, "side")[0] for m in range(414, 826)
    }
    assert left_events == side_left_events
    assert "dl_event_310" not in left_events


def test_content_side_boundary():
    # Contenido: side 723 = left 310 = right 311 (T=159 auditado).
    assert keypoint_frames("side", "dl_723") == keypoint_frames("left", "dl_310")
    assert keypoint_frames("side", "dl_723") == keypoint_frames("right", "dl_311")
    assert keypoint_frames("side", "dl_722") == keypoint_frames("right", "dl_309")
    assert keypoint_frames("side", "dl_825") == keypoint_frames("right", "dl_413")


if __name__ == "__main__":
    test_gap_constant()
    test_mapping_boundary()
    test_mapping_tail()
    test_event_310_right_only()
    test_content_boundary()
    test_boundary_308_312_left_right()
    test_boundary_399_402_413_left_right()
    test_boundary_side_right_part()
    test_boundary_side_left_part()
    test_single_source_of_truth()
    test_content_side_boundary()
    print("test_event_mapping: TODAS LAS ASSERTIONS PASARON (hueco=310)")
