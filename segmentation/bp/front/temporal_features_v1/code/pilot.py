"""Piloto acotado: solo videos common-train de desarrollo autorizado."""
import sys, csv, time
from pathlib import Path
from collections import Counter
import numpy as np
import cv2

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import raw_reader as RR
import geometry as G
from features import extraer_video, FEATURE_NAMES
from config_v1 import CONFIG_V1

ROOT = CODE.parents[4]
RAW_DIR = ROOT / "segmentation" / "bp" / "front" / "raw"
POSE_DIR = ROOT / "keypoints" / "bp" / "yolo" / "front"
VID_DIR = ROOT / "dataset" / "bp" / "front"
OUT_PREV = CODE.parent / "previews" / "pilot"
SPLIT_CSV = ROOT / "training" / "data" / "bp" / "front" / "bp_front_pose_split.csv"

# Solo desarrollo autorizado, todos common-train verificado.
PILOTO = ["bp_001", "bp_139", "bp_182", "bp_186", "bp_351",
          "bp_467", "bp_535", "bp_537", "bp_662", "bp_663"]
FRAMES_POR_VIDEO = 3


def leer_frame(video_path, idx):
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.set(cv2.CAP_PROP_POS_FRAMES, idx):
            return None
        ok, fr = cap.read()
        return fr if ok else None
    finally:
        cap.release()


def dibujar_preview(vid, t, res, frame):
    X = res["features"]
    H, W = res["info"]["alto"], res["info"]["ancho"]
    img = frame.copy()
    # keypoints: 6 puntos px
    pose = np.load(str(POSE_DIR / f"{vid}.npy"))[t]
    with np.load(RAW_DIR / f"{vid}_segmentation.npz", allow_pickle=False) as raw:
        _, grupos = RR.instancias_por_frame(raw)
        instancias = []
        for idx in grupos[t]:
            mascara = RR.decodificar_mascara_crop(raw, idx)
            clase = str(raw["instance_class_name"][idx])
            origen = np.asarray(raw["instance_mask_origin_xy"][idx], dtype=int)
            instancias.append({"idx_global": idx, "clase": clase,
                               "mascara": mascara, "origen": origen,
                               "bbox": raw["instance_bbox_xyxy_px"][idx],
                               "conf": float(raw["instance_confidence"][idx])})
        seleccion = G.seleccionar_barras(instancias, pose, W, H)
        elegidas = {b["idx_global"] for b in seleccion["elegidas"]}
        for ins in instancias:
            if not ins["mascara"].size:
                continue
            x0, y0 = ins["origen"]
            mh, mw = ins["mascara"].shape
            color = ((0, 240, 40) if ins["idx_global"] in elegidas else
                     ((0, 0, 240) if ins["clase"] == "bar" else
                      ((240, 150, 0) if ins["clase"] == "person" else (220, 0, 220))))
            roi = img[y0:y0 + mh, x0:x0 + mw]
            if roi.shape[:2] == ins["mascara"].shape:
                roi[ins["mascara"]] = (
                    roi[ins["mascara"]] * 0.75 + np.asarray(color) * 0.25
                ).astype(np.uint8)
            if ins["clase"] == "bar":
                cv2.putText(img, f"BAR #{ins['idx_global']} {ins['conf']:.2f}",
                            (int(ins["bbox"][0]), max(65, int(ins["bbox"][1]) - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        persona = max((i for i in instancias if i["clase"] == "person"),
                      key=lambda i: i["conf"], default=None)
        banco = max((i for i in instancias if i["clase"] == "bench"),
                    key=lambda i: i["conf"], default=None)
        zona = G.zona_central(pose, persona, banco["bbox"] if banco else None,
                              W, H) if persona else None
        if zona:
            perfil = zona["perfil"][2]
            cv2.polylines(img, [np.asarray(perfil, dtype=np.int32)], False,
                          (255, 255, 0), 3)
            if seleccion["elegidas"]:
                med = G.separacion_barra_silueta(seleccion["elegidas"], perfil, H, W)
                if med:
                    color_med = (0, 170, 255) if med["proyectado"] else (0, 255, 255)
                    cv2.polylines(img, [np.asarray(med["puntos"], dtype=np.int32)],
                                  False, color_med, 2)
    for k in range(6):
        if np.isfinite(pose[k, :2]).all():
            x, y = int(pose[k, 0] * W), int(pose[k, 1] * H)
            col = (0, 255, 0) if pose[k, 2] >= 0.5 else (0, 165, 255)
            cv2.circle(img, (x, y), 8, col, -1)
            cv2.putText(img, str(k), (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2)
    st = res["info"]["bar_state"][t]
    sep = X[t, 24] * H if np.isfinite(X[t, 24]) else float("nan")
    proj = bool(X[t, 37]) if np.isfinite(X[t, 37]) else False
    txt = (f"{vid} f{t} | bar={st} | sep={sep:+.0f}px"
           + (" PROY" if proj else "")
           + f" | amb={X[t,28]:.0f} | avail={res['availability'][t].mean():.2f}")
    cv2.rectangle(img, (0, 0), (W, 60), (0, 0, 0), -1)
    cv2.putText(img, txt, (12, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2)
    # barra: bbox + centro
    if np.isfinite(X[t, 0]):
        cx, cy = int(X[t, 0] * W), int(X[t, 1] * H)
        w, h = int(X[t, 2] * W), int(X[t, 3] * H)
        cv2.rectangle(img, (cx - w // 2, cy - h // 2),
                      (cx + w // 2, cy + h // 2), (0, 255, 255), 3)
        cv2.circle(img, (cx, cy), 7, (0, 255, 255), -1)
    # persona bbox
    if np.isfinite(X[t, 11]):
        pcx, pcy = int(X[t, 11] * W), int(X[t, 12] * H)
        pw, ph = int(X[t, 13] * W), int(X[t, 14] * H)
        cv2.rectangle(img, (pcx - pw // 2, pcy - ph // 2),
                      (pcx + pw // 2, pcy + ph // 2), (255, 200, 0), 2)
    # perfil silueta: punto central
    if np.isfinite(X[t, 17]):
        cv2.circle(img, (int(X[t, 0] * W) if np.isfinite(X[t, 0]) else W // 2,
                         int(X[t, 17] * H)), 7, (255, 200, 0), -1)
    # linea sep vertical
    if np.isfinite(X[t, 0]) and np.isfinite(X[t, 17]):
        x = int(X[t, 0] * W)
        cv2.line(img, (x, int(X[t, 1] * H)), (x, int(X[t, 17] * H)),
                 (255, 0, 255), 2)
    h0, w0 = img.shape[:2]
    sc = min(960 / w0, 540 / h0, 1.0)
    if sc < 1:
        img = cv2.resize(img, (round(w0 * sc), round(h0 * sc)))
    return img


def main():
    OUT_PREV.mkdir(parents=True, exist_ok=True)
    splits = {r["video_id"]: r["common_split"]
              for r in csv.DictReader(
                  open(SPLIT_CSV, encoding="utf-8-sig"))}
    for v in PILOTO:
        assert splits.get(v) == "train", f"{v} no es common-train: {splits.get(v)}"
    filas = []
    for vid in PILOTO:
        t0 = time.time()
        res = extraer_video(RAW_DIR / f"{vid}_segmentation.npz",
                            POSE_DIR / f"{vid}.npy", CONFIG_V1)
        X, A = res["features"], res["availability"]
        st = Counter(res["info"]["bar_state"])
        # frames representativos: primero, medio, y uno no-asociado si existe
        T = X.shape[0]
        cands = [0, T // 2, T - 1]
        for t in range(T):
            if res["info"]["bar_state"][t] in ("observada_no_verificada", "ausente"):
                cands.append(t)
                break
        for t in range(T):
            if res["prov_sep"][t] == 2:
                cands.append(t)
                break
        for t in range(T):
            if res["info"]["bar_state"][t] == "multiple_ambigua":
                cands.append(t)
                break
        for t in sorted(set(c for c in cands if 0 <= c < T)):
            fr = leer_frame(VID_DIR / f"{vid}.mp4", t)
            if fr is None:
                continue
            img = dibujar_preview(vid, t, res, fr)
            cv2.imwrite(str(OUT_PREV / f"{vid}_f{t:04d}.jpg"), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
        filas.append({
            "video_id": vid, "T": T, "F": X.shape[1],
            "bar_asociada": st.get("asociada", 0),
            "bar_no_verificada": st.get("observada_no_verificada", 0),
            "bar_multiple": st.get("multiple_ambigua", 0),
            "bar_ausente": st.get("ausente", 0),
            "sep_observada": int((res["prov_sep"] == 1).sum()),
            "sep_proyectada": int((res["prov_sep"] == 2).sum()),
            "sep_ausente": int((res["prov_sep"] == 0).sum()),
            "person_frames": int(A[:, 11].sum()),
            "bench_frames": int(A[:, 18].sum()),
            "profile_frames": int(A[:, 17].sum()),
            "vel_frames": int(A[:, 29].sum()),
            "est_frames": int(res["estimations"]["n_estimated"]),
            "nan_rate": round(float(np.isnan(X).mean()), 4),
            "avail_rate": round(float(A.mean()), 4),
            "segundos": round(time.time() - t0, 1),
        })
        print(filas[-1])
    with open(CODE.parent / "bp_front_features_pilot_audit.csv", "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    print(f"previews en {OUT_PREV}")


if __name__ == "__main__":
    main()
