"""Diagnóstico descriptivo solo sobre development SQ front: pose, T y vídeo.

python -m training.sq.diagnose_front
"""
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from training.sq import train_posecnn as sq
from training.sq.controlled_front import REFERENCE


def run():
    output = sq.ROOT / "training" / "results" / "sq" / "front" / "posecnn" / "development_diagnostics"
    output.mkdir(parents=True, exist_ok=True)
    data = np.load(REFERENCE["models"] / "provenance" / "split.npz")
    ids = np.concatenate([data["ids_train"], data["ids_validation"]]).astype(str)
    info = []
    for ident in ids:
        raw = np.load(REFERENCE["source"] / f"{ident}.npy", allow_pickle=False)
        xy = raw[..., :2]
        hip = xy[:, [2, 3]].mean(axis=1)
        shoulder = xy[:, [0, 1]].mean(axis=1)
        # Mediana de longitudes de tibias y saltos, sin usar etiquetas.
        tibias = np.stack([np.linalg.norm(xy[:, 4] - xy[:, 9], axis=1),
                           np.linalg.norm(xy[:, 5] - xy[:, 6], axis=1)], axis=1)
        scale = np.nanmedian(tibias)
        jump = np.linalg.norm(np.diff(xy[:, [2, 3, 4, 5, 6, 9]], axis=0), axis=2)
        rel = jump / max(scale, 1e-6)
        halves = max(3, len(raw)//6)
        movement = abs(np.diff(hip[:, 1]))
        total = np.nansum(movement) + 1e-9
        left = np.nanmedian(tibias[:, 0]); right = np.nanmedian(tibias[:, 1])
        info.append({"video_id": ident, "T": len(raw), "tibia_left": float(left),
                     "tibia_right": float(right), "ratio_tibias": float(left / max(right,1e-6)),
                     "largest_jump_in_tibias": float(np.nanmax(rel)), "p99_jump_in_tibias": float(np.nanpercentile(rel,99)),
                     "hip_start_motion_fraction": float(np.nansum(movement[:halves]) / total),
                     "hip_end_motion_fraction": float(np.nansum(movement[-halves:]) / total),
                     "hip_range_in_tibias": float((np.nanmax(hip[:, 1])-np.nanmin(hip[:, 1])) / max(scale,1e-6)),
                     "shoulder_range_in_tibias": float((np.nanmax(shoulder[:, 1])-np.nanmin(shoulder[:, 1])) / max(scale,1e-6))})
    sq.save_csv(output / "development_sequences.csv", info, list(info[0]))
    keys = ("T", "ratio_tibias", "largest_jump_in_tibias", "p99_jump_in_tibias", "hip_start_motion_fraction", "hip_end_motion_fraction", "hip_range_in_tibias")
    result = {k: {"median": float(np.nanmedian([r[k] for r in info])),
                  "p95": float(np.nanpercentile([r[k] for r in info],95)),
                  "max": float(np.nanmax([r[k] for r in info]))} for k in keys}
    for name in ("sq_086", "sq_239", "sq_152", "sq_354"):
        result.setdefault("representative", []).append(name)
    sq.save_json(output / "summary.json", result)

    # Collage auditable de cuatro vídeos de desarrollo de folds diferentes.
    images = []
    for ident in result["representative"]:
        video = sq.ROOT / "dataset" / "sq" / "front" / f"{ident}.mp4"
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            result.setdefault("unavailable_videos", []).append(ident)
            continue
        raw = np.load(REFERENCE["source"] / f"{ident}.npy", allow_pickle=False)
        strip = []
        try:
            for t in (0, len(raw)//4, len(raw)//2, 3*len(raw)//4, len(raw)-1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, t)
                ok, frame = cap.read()
                if not ok:
                    continue
                height, width = frame.shape[:2]
                for i in (2,3,4,5,6,9):
                    x, y = raw[t,i,:2]
                    if np.isfinite([x,y]).all():
                        cv2.circle(frame, (int(x*width), int(y*height)), 8, (0,255,0) if i in (2,3) else (0,0,255), -1)
                        cv2.putText(frame, str(i), (int(x*width)+8, int(y*height)), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1)
                cv2.putText(frame, f"{ident} frame={t} de {len(raw)}", (12,32), cv2.FONT_HERSHEY_SIMPLEX, .8, (0,255,255), 2)
                strip.append(cv2.resize(frame, (320,240)))
        finally:
            cap.release()
        if len(strip) == 5:
            images.append(np.concatenate(strip,axis=1))
    if images:
        cv2.imwrite(str(output / "pose_video_contact_sheet.jpg"), np.concatenate(images,axis=0))
    sq.save_json(output / "summary.json", result)
    print(result)


if __name__ == "__main__":
    run()
