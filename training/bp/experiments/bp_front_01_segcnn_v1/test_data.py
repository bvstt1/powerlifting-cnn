"""Controles del remuestreo: no cruzar NaN ni saltos de identidad."""
import numpy as np
from .data import remuestrear


def _secuencia():
    t = 5
    x = np.ones((t, 38), np.float32)
    a = np.ones((t, 38), np.int8)
    return {"features": x, "availability": a,
            "prov_bar": np.ones(t, np.int8),
            "prov_sep": np.ones(t, np.int8),
            "bar_track_id": np.array([1, 1, 2, 2, 2]),
            "frame_index": np.arange(t)}


def test_no_atraviesa_cambio_identidad():
    s = _secuencia()
    v, a, p = remuestrear(s)
    tramo = np.flatnonzero((np.linspace(0, 4, 170) > 1) &
                           (np.linspace(0, 4, 170) < 2))
    assert not a[tramo, 0].any()
    assert not a[tramo, 24].any()
    assert a[tramo, 11].all()
    assert np.isnan(v[tramo, 0]).all()


def test_nan_y_proyeccion_no_se_mezclan():
    s = _secuencia()
    s["features"][3, 11] = np.nan
    s["availability"][3, 11] = 0
    s["prov_sep"][3] = 2
    v, a, p = remuestrear(s)
    ts = np.linspace(0, 4, 170)
    tramo = np.flatnonzero((ts > 2) & (ts < 3))
    assert not a[tramo, 11].any()
    assert not a[tramo, 24].any()
    assert np.isnan(v[tramo, 24]).all()


if __name__ == "__main__":
    test_no_atraviesa_cambio_identidad()
    test_nan_y_proyeccion_no_se_mezclan()
    print("Remuestreo: pruebas OK")
