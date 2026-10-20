"""Genera las 38 features temporales BP front a partir de RAW y pose.

Procesa la cohorte en segmentation/bp/front/temporal_features_v1/, guarda
secuencias, esquema y manifiestos. No reejecuta YOLO ni entrena la CNN.
"""
import sys, csv, json, hashlib, time, traceback
from pathlib import Path
from multiprocessing import Pool
import numpy as np

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
from features import extraer_video, FEATURE_NAMES, QUALITY_NAMES, F
from config import CONFIG_V1
import schema as SCH

ROOT = CODE.parents[2]
RAW_DIR = ROOT / "segmentation" / "bp" / "front" / "raw"
POSE_DIR = ROOT / "keypoints" / "bp" / "yolo" / "front"
OUT = ROOT / "segmentation" / "bp" / "front" / "temporal_features_v1"
SEQ_DIR = OUT / "sequences"
SEG_MANIFEST = ROOT / "segmentation" / "bp" / "front" / "manifests" / "bp_front_segmentation_manifest.csv"
SPLIT_CSV = ROOT / "training" / "data" / "bp" / "front" / "bp_front_pose_split.csv"
LABELS_CSV = ROOT / "etiquetado" / "bp" / "bp_front.csv"
N_WORKERS = 6


def sha256_archivo(ruta, limite_mb=400):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        while True:
            b = f.read(8 * 1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def procesar_uno(vid):
    t0 = time.time()
    try:
        raw = RAW_DIR / f"{vid}_segmentation.npz"
        pose_p = POSE_DIR / f"{vid}.npy"
        res = extraer_video(raw, pose_p, CONFIG_V1)
        X, A = res["features"], res["availability"]
        out_p = SEQ_DIR / f"{vid}_temporal.npz"
        np.savez_compressed(
            out_p, features=X, availability=A,
            prov_bar=res["prov_bar"], prov_sep=res["prov_sep"],
            quality_flags=res["quality_flags"],
            quality_names=np.asarray(QUALITY_NAMES),
            bar_gap_frames=res["bar_gap_frames"],
            frame_index=res["frame_index"],
            bar_state=np.asarray(res["info"]["bar_state"]),
            bar_track_id=res["info"]["bar_track_id"],
            bar_instance_id=res["info"]["bar_instance_id"],
            bar_n_raw=res["info"]["bar_n_raw"],
            bar_n_kept=res["info"]["bar_n_kept"],
            est_frames=res["estimations"]["frames"],
            est_offset=res["estimations"]["offset"],
            est_ancho_munecas=res["estimations"]["ancho_munecas"],
            est_anclas=res["estimations"]["anclas"],
            feature_names=np.asarray(FEATURE_NAMES),
            config_json=np.asarray([json.dumps(CONFIG_V1, ensure_ascii=False)]),
            raw_path=str(raw), pose_path=str(pose_p),
            fps=np.asarray([res["info"]["fps"]]),
            image_hw=np.asarray([[res["info"]["alto"], res["info"]["ancho"]]]),
        )
        n = len(X)
        fila = {
            "video_id": vid, "T": n, "F": int(F), "status": "OK",
            "bar_asociada": int((res["prov_bar"] == 1).sum()),
            "bar_no_verificada": int((res["prov_bar"] == 4).sum()),
            "bar_ausente": int((res["prov_bar"] == 0).sum()),
            "sep_observada": int((res["prov_sep"] == 1).sum()),
            "sep_proyectada": int((res["prov_sep"] == 2).sum()),
            "sep_ausente": int((res["prov_sep"] == 0).sum()),
            "person_frames": int(A[:, 11].sum()),
            "bench_frames": int(A[:, 18].sum()),
            "profile_frames": int(A[:, 17].sum()),
            "vel_frames": int(A[:, 29].sum()),
            "est_frames": int(res["estimations"]["n_estimated"]),
            "nan_rate": round(float(np.isnan(X).mean()), 5),
            "avail_rate": round(float(A.mean()), 5),
            "bench_stability_iou": (None if res["info"]["bench_stability_iou"] is None
                                    or not np.isfinite(res["info"]["bench_stability_iou"])
                                    else round(float(res["info"]["bench_stability_iou"]), 4)),
            "raw_sha256": sha256_archivo(raw),
            "pose_sha256": sha256_archivo(pose_p),
            "out_sha256": sha256_archivo(out_p),
            "segundos": round(time.time() - t0, 1),
            "error": "",
        }
        return fila
    except Exception as e:
        return {"video_id": vid, "T": "", "F": int(F), "status": "ERROR",
                "bar_asociada": "", "bar_no_verificada": "", "bar_ausente": "",
                "sep_observada": "", "sep_proyectada": "", "sep_ausente": "",
                "person_frames": "", "bench_frames": "", "profile_frames": "",
                "vel_frames": "", "est_frames": "", "nan_rate": "",
                "avail_rate": "", "bench_stability_iou": "",
                "raw_sha256": "", "pose_sha256": "", "out_sha256": "",
                "segundos": round(time.time() - t0, 1),
                "error": f"{type(e).__name__}: {e}"}


def main(solo=None):
    SEQ_DIR.mkdir(parents=True, exist_ok=True)
    SCH.guardar_schema(OUT / "bp_front_features_schema.json")
    (OUT / "bp_front_features_config.json").write_text(
        json.dumps(CONFIG_V1, indent=2, ensure_ascii=False), encoding="utf-8")
    seg_ids = [r["video_id"] for r in
               csv.DictReader(open(SEG_MANIFEST, encoding="utf-8-sig"))]
    # Solo RAW utilizable: existe fichero (excluye bp_349).
    vids = [v for v in seg_ids if (RAW_DIR / f"{v}_segmentation.npz").exists()]
    vids = sorted(vids)
    if solo:
        vids = [v for v in vids if v in set(solo)]
    print(f"videos a procesar: {len(vids)} (workers={N_WORKERS})")
    t0 = time.time()
    with Pool(N_WORKERS) as pool:
        filas = pool.map(procesar_uno, vids)
    for f in filas:
        if f["status"] != "OK":
            print("ERROR", f["video_id"], f["error"])
    with open(OUT / "bp_front_features_manifest.csv", "w",
              newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    ok = sum(1 for f in filas if f["status"] == "OK")
    print(f"OK {ok}/{len(filas)} en {time.time()-t0:.0f}s")
    # quality csv + common cohort
    construir_cohorte(filas)


def construir_cohorte(filas):
    splits = {r["video_id"]: r for r in
              csv.DictReader(open(SPLIT_CSV, encoding="utf-8-sig"))}
    labels = {r["video_id"]: r for r in
              csv.DictReader(open(LABELS_CSV, encoding="utf-8-sig"))}
    seg = {r["video_id"]: r for r in
           csv.DictReader(open(SEG_MANIFEST, encoding="utf-8-sig"))}
    out = []
    for f in filas:
        if f["status"] != "OK":
            continue
        v = f["video_id"]
        sp = splits.get(v, {})
        lb = labels.get(v, {})
        sg = seg.get(v, {})
        out.append({
            "video_id": v, "label": lb.get("label", ""),
            "T": f["T"], "pose_disponible": True,
            "features_disponibles": True,
            "common_split": sp.get("common_split", ""),
            "pose_only_split": sp.get("pose_only_split", ""),
            "quality_status_seg": sg.get("quality_status", ""),
            "nan_rate": f["nan_rate"],
            "video_path": str(ROOT / "dataset" / "bp" / "front" / f"{v}.mp4"),
            "pose_path": str(POSE_DIR / f"{v}.npy"),
            "raw_path": str(RAW_DIR / f"{v}_segmentation.npz"),
            "features_path": str(SEQ_DIR / f"{v}_temporal.npz"),
            "excluido": v == "bp_352",
            "motivo_exclusion": ("duplicado exacto de bp_351; fuera del split"
                                 if v == "bp_352" else ""),
        })
    # bp_349 no tiene RAW; bp_352 tiene features descriptivas, pero fuera de cohorte.
    for v, motivo in (("bp_349", "sin RAW utilizable (QUALITY_ERROR degenerate_polygon)"),):
        sp = splits.get(v, {})
        lb = labels.get(v, {})
        out.append({"video_id": v, "label": lb.get("label", ""),
                    "T": int(np.load(POSE_DIR / f"{v}.npy", mmap_mode="r").shape[0]),
                    "pose_disponible": True,
                    "features_disponibles": False,
                    "common_split": sp.get("common_split", "not_in_common"),
                    "pose_only_split": sp.get("pose_only_split", ""),
                    "quality_status_seg": "",
                    "nan_rate": "", "excluido": True,
                    "video_path": str(ROOT / "dataset" / "bp" / "front" / f"{v}.mp4"),
                    "pose_path": str(POSE_DIR / f"{v}.npy"),
                    "raw_path": "", "features_path": "",
                    "motivo_exclusion": motivo})
    with open(OUT / "bp_front_features_common_cohort.csv", "w",
              newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(sorted(out, key=lambda r: r["video_id"]))
    # quality por video (distribucion de calidad)
    with open(OUT / "bp_front_features_quality.csv", "w",
              newline="", encoding="utf-8") as fh:
        campos = ["video_id", "T", "bar_asociada", "bar_no_verificada",
                  "bar_ausente", "sep_observada", "sep_proyectada",
                  "sep_ausente", "person_frames", "bench_frames",
                  "profile_frames", "vel_frames", "est_frames",
                  "nan_rate", "avail_rate"]
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        for f in filas:
            if f["status"] == "OK":
                w.writerow({k: f[k] for k in campos})
    print(f"cohorte: {len([r for r in out if not r['excluido']])} incluidos, "
          f"{len([r for r in out if r['excluido']])} excluidos")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=N_WORKERS)
    a = ap.parse_args()
    N_WORKERS = a.workers
    main(solo=a.solo)
