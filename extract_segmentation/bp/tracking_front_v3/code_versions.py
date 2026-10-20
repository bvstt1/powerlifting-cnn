"""Registra hashes SHA-256 históricos y del código V3 reorganizado.

Los primeros corresponden a Validation76 original; los segundos permiten
detectar cambios posteriores en el algoritmo movido, sin equiparar ambos.
"""

HISTORICAL_ALGORITHM_HASHES = {
    "bar_tracking_v3_utils.py": "bf103391f9171c80739e52345375f464ea7ac822d22c71d4804c0c810350317d",
    "rebuild_bar_tracking_v3_from_raw.py": "8b1814e32a1fa8d631427901bf2a6701090b24e63a8f5a81ab2485585f3cfb10",
}

RELOCATED_ALGORITHM_HASHES = {
    "bar_tracking_v3_utils.py": "5e77649830ef8b395e256f22aedc395967fcd0542d578ffaaff0799efb3d4758",
    "rebuild_bar_tracking_v3_from_raw.py": "56150934967e77ff6987c48a6e1e77da2763ab9e479c16436c261ab6b9b08c9a",
}
