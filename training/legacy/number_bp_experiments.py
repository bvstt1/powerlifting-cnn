"""Renombra experimentos BP por vista y su secuencia documentada.

Uso: python -B -m training.legacy.number_bp_experiments [--apply]
No reescribe JSON ni reportes congelados: sus rutas antiguas son evidencia.
"""

import argparse
import hashlib
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "training" / "bp" / "experiments"
RENAME = {
    "bp_front_segcnn_v1": "bp_front_01_segcnn_v1",
    "bp_front_posecnn_common_v1": "bp_front_02_posecnn_common_v1",
    "bp_front_fusion_v1": "bp_front_03_fusion_v1",
    "bp_front_cross_validation_5fold_v1": "bp_front_04_cross_validation_5fold_v1",
    "bp_front_final_test_v1": "bp_front_05_final_test_v1",
    "bp_left_segcnn_v1": "bp_left_01_segcnn_v1",
    "bp_left_posecnn_compatibility_v1": "bp_left_02_posecnn_compatibility_v1",
    "bp_left_fusion_v1": "bp_left_03_fusion_v1",
    "bp_left_cv_v1": "bp_left_04_cv_v1",
    "bp_left_final_test_v1": "bp_left_05_final_test_v1",
    "bp_right_segcnn_v1": "bp_right_01_segcnn_v1",
    "bp_right_posecnn_common_v1": "bp_right_02_posecnn_common_v1",
    "bp_right_fusion_v1": "bp_right_03_fusion_v1",
    "bp_right_cv_v1": "bp_right_04_cv_v1",
    "bp_right_final_test_v1": "bp_right_05_final_test_v1",
}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.digest()


def replace_references(text):
    # Sustituciones simultaneas para no renumerar dos veces nombres ya nuevos.
    names = "|".join(re.escape(name) for name in sorted(RENAME, key=len, reverse=True))
    return re.sub(r"(?<![A-Za-z0-9_])(" + names + r")(?![A-Za-z0-9_])",
                  lambda match: RENAME[match.group(1)], text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if any(not (BASE / old).is_dir() or (BASE / new).exists()
           for old, new in RENAME.items()):
        raise RuntimeError("Origen faltante o destino ocupado; no renombrar parcialmente")
    edits = []
    for path in (ROOT / "training").rglob("*.py"):
        if "__pycache__" in path.parts or path == Path(__file__) or "legacy" in path.parts:
            continue
        original = path.read_text(encoding="utf-8")
        updated = replace_references(original)
        if original != updated:
            compile(updated, str(path), "exec")
            edits.append((path, updated))
    docs = (ROOT / "training" / "docs" / "EXPERIMENTOS.md",
            ROOT / "training" / "docs" / "INVENTARIO_REDES.md")
    for path in docs:
        original = path.read_text(encoding="utf-8")
        updated = replace_references(original)
        if original != updated:
            edits.append((path, updated))
    print(f"Renombres: {len(RENAME)}; referencias actuales a actualizar: {len(edits)}")
    if not args.apply:
        for old, new in RENAME.items():
            print(old, "->", new)
        return
    original_hashes = {}
    for old, new in RENAME.items():
        for path in (BASE / old).rglob("*"):
            if path.is_file():
                original_hashes[(new, path.relative_to(BASE / old))] = sha(path)
        shutil.move(str(BASE / old), str(BASE / new))
    for (new, relative), digest in original_hashes.items():
        if sha(BASE / new / relative) != digest:
            raise RuntimeError(f"Bytes alterados: {new}/{relative}")
    for path, updated in edits:
        rel = path.relative_to(BASE) if path.is_relative_to(BASE) else None
        target = BASE / RENAME[rel.parts[0]] / Path(*rel.parts[1:]) if rel and rel.parts[0] in RENAME else path
        target.write_text(updated, encoding="utf-8")
    print(f"Directorios renombrados: {len(RENAME)}; archivos sin alterar durante traslado: {len(original_hashes)}")


if __name__ == "__main__":
    main()
