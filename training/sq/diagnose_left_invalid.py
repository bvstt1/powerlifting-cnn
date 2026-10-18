"""Diagnóstico de errores externos de desarrollo SQ left v2, sin usar test.

python -m training.sq.diagnose_left_invalid
"""
import csv
import json

import cv2
import numpy as np

from training.sq import train_posecnn as sq
from training.sq.run_analysis import read_predictions


BASE = sq.paths("left", "baseline_t170_v2")
OUT = sq.ROOT / "training" / "results" / "sq" / "left" / "posecnn" / "development_diagnostics"


def features(row, processed):
    ident = row["video_id"]
    raw = np.load(BASE["source"] / f"{ident}.npy", allow_pickle=False)
    video_path = sq.ROOT / "dataset" / "sq" / "left" / f"{ident}.mp4"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    try:
        frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if abs(frames_video - len(raw)) > 1:
        raise ValueError(f"Frames diferentes en {ident}: video {frames_video}, pose {len(raw)}")
    pre, center, scale = sq.preprocess_left(raw)
    if not np.allclose(pre, processed[ident], atol=2e-5):
        raise ValueError(f"Preprocesamiento no coincide con split congelado: {ident}")
    xy = raw[..., :2]
    valid = np.isfinite(xy).all(axis=-1)
    # Estos ratios se calculan para describir si la falta sigue representada,
    # no para redefinir automáticamente el criterio reglamentario.
    hip_over_knee = (xy[:, 1, 1] - xy[:, 0, 1]) / scale
    hip_over_knee_pre = pre[:, 1, 1] - pre[:, 0, 1]
    jumps = np.linalg.norm(np.diff(xy[..., :2], axis=0), axis=2) / scale
    ankle_motion = np.linalg.norm(xy[:, 2, :2] - np.nanmedian(xy[:, 2, :2], axis=0), axis=1) / scale
    foot_motion = np.linalg.norm(xy[:, 3, :2] - np.nanmedian(xy[:, 3, :2], axis=0), axis=1) / scale
    return {"video_id": ident, "fold": int(row["fold"]),
            "true_label": int(row["true_label"]), "predicted_label": int(row["predicted_label"]),
            "probability_valid": float(row["probability_valid"]), "T_video": frames_video, "T_pose": len(raw),
            "missing_xy_ratio": float((~valid).mean()), "confidence_median": float(np.nanmedian(raw[..., 2])),
            "confidence_min": float(np.nanmin(raw[..., 2])), "max_jump_tibias": float(np.nanmax(jumps)),
            "hip_minus_knee_y_min_tibias": float(np.nanmin(hip_over_knee)),
            "hip_minus_knee_y_max_tibias": float(np.nanmax(hip_over_knee)),
            "hip_minus_knee_y_min_pre": float(np.nanmin(hip_over_knee_pre)),
            "hip_minus_knee_y_max_pre": float(np.nanmax(hip_over_knee_pre)),
            "ankle_displacement_max_tibias": float(np.nanmax(ankle_motion)),
            "toe_displacement_max_tibias": float(np.nanmax(foot_motion)),
            "video_path": str(video_path.relative_to(sq.ROOT))}


def sheet(ids):
    rows = []
    for ident in ids:
        raw = np.load(BASE["source"] / f"{ident}.npy", allow_pickle=False)
        cap = cv2.VideoCapture(str(sq.ROOT / "dataset" / "sq" / "left" / f"{ident}.mp4"))
        strip = []
        try:
            for t in (0, len(raw)//4, len(raw)//2, 3*len(raw)//4, len(raw)-1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, t)
                ok, frame = cap.read()
                if not ok:
                    continue
                height, width = frame.shape[:2]
                for k, color in ((0,(0,0,255)),(1,(0,255,0)),(2,(255,0,0)),(3,(255,255,0)),(4,(0,255,255))):
                    x, y = raw[t,k,:2]
                    if np.isfinite([x,y]).all():
                        pt = (int(x*width), int(y*height))
                        cv2.circle(frame, pt, 8, color, -1)
                        cv2.putText(frame, str(k), (pt[0]+8,pt[1]), cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
                cv2.putText(frame, f"{ident} frame {t}/{len(raw)}", (10,32),cv2.FONT_HERSHEY_SIMPLEX,.75,(0,255,255),2)
                strip.append(cv2.resize(frame,(320,240)))
        finally:
            cap.release()
        if len(strip) == 5:
            rows.append(np.concatenate(strip,axis=1))
    if rows:
        cv2.imwrite(str(OUT / "cv_false_valid_contact_sheet.jpg"),np.concatenate(rows,axis=0))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    split = np.load(BASE["models"] / "provenance" / "split.npz")
    processed = {str(i): x for role in ("train", "validation")
                 for i, x in zip(split[f"ids_{role}"], split[f"X_{role}"])}
    cv = read_predictions(BASE["results"] / "cv_predictions.csv")
    errors = [row for row in cv if row["true_label"] == "0" and row["predicted_label"] == "1"]
    correct = [row for row in cv if row["true_label"] == "0" and row["predicted_label"] == "0"]
    assert len(errors) == 34 and len(correct) == 75
    rows = [features(row, processed) for row in errors + correct]
    sq.save_csv(OUT / "invalid_cv_trajectories.csv", rows, list(rows[0]))
    fields = ("missing_xy_ratio","confidence_median","max_jump_tibias",
              "hip_minus_knee_y_max_tibias","hip_minus_knee_y_max_pre",
              "ankle_displacement_max_tibias","toe_displacement_max_tibias")
    summary = {"false_valid":34,"correct_invalid":75, "video_pose_frame_mismatch":0,"preprocess_mismatch":0,
               "by_group": {}}
    for name, subset in (("false_valid",rows[:34]),("correct_invalid",rows[34:])):
        summary["by_group"][name] = {key: {"median":float(np.median([r[key] for r in subset])),
                                               "p95":float(np.percentile([r[key] for r in subset],95)),
                                               "max":float(np.max([r[key] for r in subset]))} for key in fields}
    selection = ("sq_001","sq_324","sq_247","sq_315","sq_197","sq_208","sq_211","sq_283")
    assert set(selection).issubset(processed)
    sheet(selection)
    summary["contact_sheet_ids"] = list(selection)
    sq.save_json(OUT / "summary.json",summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
