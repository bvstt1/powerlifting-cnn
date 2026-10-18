"""Auditoria integral del dataset exportado, sin ajustar reglas con Test."""
import csv
import hashlib
import json
import traceback
from collections import Counter
from pathlib import Path

import numpy as np

from extract_all import (OUT, RAW_DIR, POSE_DIR, ROOT, SEG_MANIFEST,
                         SPLIT_CSV, LABELS_CSV, sha256_archivo)
from features import FEATURE_NAMES, QUALITY_NAMES


def main():
    manifest = list(csv.DictReader(open(OUT / "bp_front_features_manifest.csv",
                                  encoding="utf-8")))
    cohorte = list(csv.DictReader(open(OUT / "bp_front_features_common_cohort.csv",
                                 encoding="utf-8")))
    original = list(csv.DictReader(open(SEG_MANIFEST, encoding="utf-8-sig")))
    split = list(csv.DictReader(open(SPLIT_CSV, encoding="utf-8-sig")))
    assert len(original) == 663 and len(manifest) == 662 and len(cohorte) == 663
    assert len({r["video_id"] for r in cohorte}) == 663
    assert all(r["status"] == "OK" for r in manifest)
    assert len(list((OUT / "sequences").glob("*_temporal.npz"))) == 662
    assert {r["video_id"] for r in manifest} == (
        {r["video_id"] for r in original} - {"bp_349"})
    assert {r["video_id"] for r in cohorte if r["excluido"] == "True"} == {
        "bp_349", "bp_352"}
    assert Counter(r["common_split"] for r in cohorte
                   if r["excluido"] == "False") == {
        "train": 462, "validation": 99, "test": 100}
    assert {r["video_id"] for r in split if r["common_member"] == "True"} == {
        r["video_id"] for r in cohorte if r["excluido"] == "False"}
    totales = Counter()
    disponibles = np.zeros(len(FEATURE_NAMES), dtype=np.int64)
    frames = 0
    valores = 0
    proy_dev = []
    errores = []
    por_split = Counter()
    spart = {r["video_id"]: r["common_split"] for r in split}
    for fila in manifest:
        v = fila["video_id"]
        p = OUT / "sequences" / f"{v}_temporal.npz"
        raw = RAW_DIR / f"{v}_segmentation.npz"
        pose = POSE_DIR / f"{v}.npy"
        try:
            assert sha256_archivo(raw) == fila["raw_sha256"], v
            assert sha256_archivo(pose) == fila["pose_sha256"], v
            assert sha256_archivo(p) == fila["out_sha256"], v
            with np.load(p, allow_pickle=False) as d:
                x = d["features"]
                a = d["availability"]
                t = len(x)
                assert x.shape == a.shape == (int(fila["T"]), len(FEATURE_NAMES))
                assert np.array_equal(a, np.isfinite(x))
                assert tuple(d["feature_names"].tolist()) == tuple(FEATURE_NAMES)
                assert d["quality_flags"].shape == (t, len(QUALITY_NAMES))
                assert tuple(d["quality_names"].tolist()) == tuple(QUALITY_NAMES)
                assert np.array_equal(d["frame_index"], np.arange(t))
                assert int(d["bar_gap_frames"].max()) <= t
                assert np.array_equal(d["quality_flags"][:, 12].astype(bool),
                                      d["est_frames"])
                assert not np.any(d["est_frames"] & (d["bar_n_raw"] > 0))
                assert np.all((d["prov_sep"] != 0) == (a[:, 24] == 1))
                assert np.all((d["prov_bar"] == 1) == (a[:, 0] == 1))
                assert np.all((a[:, 29] == 1) <= (
                    (d["prov_bar"] == 1) & (np.roll(d["prov_bar"], 1) == 1)))
                velo = np.flatnonzero(a[:, 29])
                if len(velo):
                    assert velo.min() > 0
                    assert np.all(d["bar_track_id"][velo] ==
                                  d["bar_track_id"][velo - 1])
                assert np.all(d["prov_sep"] <= 2)
                assert tuple(np.load(pose, mmap_mode="r").shape) == (t, 6, 3)
                assert float(d["fps"][0]) > 0
                assert d["image_hw"].shape == (1, 2)
                disponibles[:] += a.sum(axis=0)
                frames += t
                valores += t * len(FEATURE_NAMES)
                totales["asociada"] += int((d["bar_state"] == "asociada").sum())
                totales["no_verificada"] += int((d["bar_state"] == "observada_no_verificada").sum())
                totales["multiple"] += int((d["bar_state"] == "multiple_ambigua").sum())
                totales["ausente"] += int((d["bar_state"] == "ausente").sum())
                totales["sep_directa"] += int((d["prov_sep"] == 1).sum())
                totales["sep_proyectada"] += int((d["prov_sep"] == 2).sum())
                totales["estimaciones"] += int(d["est_frames"].sum())
                totales["persona"] += int(a[:, 11].sum())
                totales["banco"] += int(a[:, 18].sum())
                totales["perfil"] += int(a[:, 17].sum())
                totales["velocidad"] += int(a[:, 29].sum())
                totales["perfil_ambiguo"] += int(d["quality_flags"][:, 8].sum())
                if spart.get(v) in {"train", "validation"}:
                    ids = np.flatnonzero(d["prov_sep"] == 2)
                    if len(ids):
                        proy_dev.append((v, ids[:3].tolist(), int(len(ids))))
                por_split[spart.get(v, "excluded")] += 1
        except Exception as exc:
            errores.append(f"{v}: {traceback.format_exc().strip().splitlines()[-2:]}")
    assert not errores, errores[:20]
    assert frames == sum(int(r["T"]) for r in manifest)
    assert sum(totales[k] for k in ("asociada", "no_verificada", "multiple", "ausente")) == frames
    observaciones = {
        "videos": len(manifest), "frames": frames, "F": len(FEATURE_NAMES),
        "valores_totales": valores, "valores_ausentes": valores - int(disponibles.sum()),
        "fraccion_ausente": (valores - int(disponibles.sum())) / valores,
        "disponibilidad_por_canal": dict(zip(FEATURE_NAMES,
                                             (disponibles / frames).round(6).tolist())),
        "conteos": dict(totales), "split": dict(por_split),
        "ejemplos_proyeccion_desarrollo": proy_dev[:20],
        "errores": errores,
        "hashes_fundamentales": {},
    }
    fuentes = {
        "modelo_segmentacion": ROOT / "models/bp/bp_front_seg_v2.pt",
        "modelo_pose": ROOT / "models/bp/bp_front_skeleton_v8.pt",
        "csv_labels": LABELS_CSV,
        "split_comun": SPLIT_CSV,
        "manifest_raw": SEG_MANIFEST,
        "config": OUT / "bp_front_features_config.json",
        "schema": OUT / "bp_front_features_schema.json",
    }
    observaciones["hashes_fundamentales"] = {
        k: sha256_archivo(p) for k, p in fuentes.items()}
    (OUT / "bp_front_features_validation.json").write_text(
        json.dumps(observaciones, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: observaciones[k] for k in
                      ("videos", "frames", "F", "valores_ausentes",
                       "fraccion_ausente", "conteos", "split",
                       "ejemplos_proyeccion_desarrollo")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
