"""Constructor temporal (T,F) BP_FRONT. Solo lectura de RAW + pose."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import raw_reader as RR
import geometry as G

# Esquema formal F=38. Unidades y rangos en bp_front_features_schema.json.
FEATURE_NAMES = [
    # BAR (11) POSE-ASSISTED por seleccion (posicion SEG, seleccion POSE)
    "bar_cx_norm", "bar_cy_norm", "bar_w_norm", "bar_h_norm",
    "bar_area_rel", "bar_slope", "bar_bbox_occupancy", "bar_conf_max",
    "bar_frac_inside_person", "bar_frac_between_wrists", "bar_dy_wrists_norm",
    # PERSON (7) SEG-ONLY salvo perfil
    "person_cx_norm", "person_cy_norm", "person_w_norm", "person_h_norm",
    "person_area_rel", "person_conf_max", "silhouette_profile_y_norm",
    # BENCH (6) SEG-ONLY
    "bench_cx_norm", "bench_cy_norm", "bench_w_norm", "bench_h_norm",
    "bench_conf_max", "bench_stability_iou",
    # RELACIONES (5) POSE-ASSISTED
    "sep_signed_norm", "sep_coverage", "sep_extrap_dist_norm",
    "wrist_bar_dist_norm", "profile_ambiguous",
    # DINAMICA (4) POSE-ASSISTED/SEG
    "bar_vx_norm_s", "bar_vy_norm_s", "d_slope_dt", "d_sep_dt",
    # CALIDAD (5)
    "q_bar_associated", "q_bar_unverified", "q_bar_multiple",
    "q_bar_absent", "q_projected",
]
F = len(FEATURE_NAMES)
QUALITY_NAMES = ["bar_observed", "bar_unverified", "bar_multiple",
                 "bar_absent", "person_absent", "bench_absent",
                 "pose_wrists_insufficient", "pose_shoulders_insufficient",
                 "profile_ambiguous", "profile_missing", "sep_missing",
                 "sep_projected", "bar_temporal_estimate", "consecutive_bar_observations"]

# Provenance por frame: 0 ausente, 1 observado, 2 proyectado, 4 no verificado.
PROV_ABSENT, PROV_OBSERVED, PROV_PROJECTED, PROV_UNVERIFIED = 0, 1, 2, 4


def extraer_video(raw_path, pose_path, config):
    d, meta = RR.cargar_raw(raw_path)
    pose = np.load(str(pose_path), allow_pickle=False).astype(np.float32)
    fps = float(meta.get("fps", 30.0)) or 30.0
    H, W = [int(x) for x in d["frame_image_shape_hw"][0]]
    fidx, grupos = RR.instancias_por_frame(d)
    T = len(fidx)
    if pose.shape[0] != T:
        raise ValueError(f"T desincronizado: pose {pose.shape[0]} vs raw {T}")
    if pose.shape[1] != 6 or pose.shape[2] != 3:
        raise ValueError(f"pose shape inesperado {pose.shape}")

    X = np.full((T, F), np.nan, dtype=np.float32)
    avail = np.zeros((T, F), dtype=np.int8)
    prov_bar = np.zeros(T, dtype=np.int8)
    prov_sep = np.zeros(T, dtype=np.int8)
    bar_state = []
    track_hist = []
    instance_hist = []
    bench_boxes = []

    # Primera pasada: por frame.
    cache = {}
    for t in range(T):
        insts = []
        for g in grupos[t]:
            masc = RR.decodificar_mascara_crop(d, g)
            insts.append({
                "idx_global": int(g),
                "clase": str(d["instance_class_name"][g]),
                "conf": float(d["instance_confidence"][g]),
                "bbox": np.asarray(
                    d["instance_bbox_xyxy_px"][g], dtype=float),
                "track_id": int(d["instance_track_id"][g]),
                "mascara": masc,
                "origen": np.asarray(
                    d["instance_mask_origin_xy"][g], dtype=int),
                "area": int(d["instance_mask_area_px"][g]),
            })
        cache[t] = insts
        personas = [i for i in insts if i["clase"] == "person" and i["mascara"].size]
        bancos = [i for i in insts if i["clase"] == "bench" and i["mascara"].size]
        persona = max(personas, key=lambda i: i["conf"]) if personas else None
        banco = max(bancos, key=lambda i: i["conf"]) if bancos else None
        if banco is not None:
            bench_boxes.append(np.asarray(banco["bbox"], dtype=float))

        sel = G.seleccionar_barras(insts, pose[t], W, H)
        bar_state.append(sel["estado"])
        eleg = sel["elegidas"]
        tids = sorted({b["track_id"] for b in eleg})
        track_hist.append(tuple(tids))
        instance_hist.append(tuple(int(b["idx_global"]) for b in eleg))

        # --- BAR ---
        n_raw = sel["n_raw"]
        barras_raw_total = n_raw
        if len(eleg) == 1:
            geo = G.geometria_mascara_union(eleg, W, H)
            X[t, 0] = geo["cx"] / W
            X[t, 1] = geo["cy"] / H
            X[t, 2] = geo["w"] / W
            X[t, 3] = geo["h"] / H
            X[t, 4] = geo["area_rel"]
            X[t, 6] = min(1.0, geo["area_px"] / max(1, (geo["w"] + 1) * (geo["h"] + 1)))
            X[t, 7] = max(float(b["conf"]) for b in eleg)
            aj = G.inclinacion_barra_union(eleg, H)
            if aj is not None:
                X[t, 5] = aj["pendiente"]
            # fracciones de la union
            ys_u = geo["ys"]
            xs_u = geo["xs"]
            if persona is not None:
                bp = np.asarray(persona["bbox"], dtype=float)
                X[t, 8] = float(np.mean(
                    (xs_u >= bp[0]) & (xs_u <= bp[2])
                    & (ys_u >= bp[1]) & (ys_u <= bp[3])))
            mun = sel["munecas"]
            if mun is not None:
                X[t, 9] = float(np.mean(
                    (xs_u >= mun[0, 0]) & (xs_u <= mun[1, 0])))
                X[t, 10] = (float(np.median(ys_u))
                             - float(np.mean(mun[:, 1]))) / H
            for c in (0, 1, 2, 3, 4, 6, 7):
                avail[t, c] = 1
            if np.isfinite(X[t, 5]):
                avail[t, 5] = 1
            if np.isfinite(X[t, 8]):
                avail[t, 8] = 1
            if np.isfinite(X[t, 9]):
                avail[t, 9] = 1
            if np.isfinite(X[t, 10]):
                avail[t, 10] = 1
            prov_bar[t] = PROV_OBSERVED
        else:
            if n_raw > 0:
                prov_bar[t] = PROV_UNVERIFIED
            else:
                prov_bar[t] = PROV_ABSENT

        # --- PERSON ---
        if persona is not None:
            bp = np.asarray(persona["bbox"], dtype=float)
            X[t, 11] = (bp[0] + bp[2]) / 2 / W
            X[t, 12] = (bp[1] + bp[3]) / 2 / H
            X[t, 13] = (bp[2] - bp[0]) / W
            X[t, 14] = (bp[3] - bp[1]) / H
            X[t, 15] = float(persona["mascara"].sum()) / (W * H)
            X[t, 16] = float(persona["conf"])
            avail[t, 11:17] = 1

        # --- BENCH ---
        if banco is not None:
            bb = np.asarray(banco["bbox"], dtype=float)
            X[t, 18] = (bb[0] + bb[2]) / 2 / W
            X[t, 19] = (bb[1] + bb[3]) / 2 / H
            X[t, 20] = (bb[2] - bb[0]) / W
            X[t, 21] = (bb[3] - bb[1]) / H
            X[t, 22] = float(banco["conf"])
            avail[t, 18:23] = 1

        # --- ZONA CENTRAL + SEPARACION ---
        banco_bbox = np.asarray(banco["bbox"], dtype=float) if banco is not None else None
        zona = G.zona_central(pose[t], persona, banco_bbox, W, H) if persona is not None else None
        if zona is not None:
            _, y_c, _ = zona["perfil"]
            X[t, 17] = y_c / H
            avail[t, 17] = 1
            X[t, 28] = 1.0 if zona["ambiguo"] else 0.0
            avail[t, 28] = 1
            if len(eleg) == 1:
                med = G.separacion_barra_silueta(
                    eleg, zona["perfil"][2], H, W)
                if med is not None:
                    X[t, 24] = med["sep_px"] / H
                    X[t, 25] = med["cobertura"]
                    X[t, 26] = med["dist_extra"] / W
                    avail[t, 24:27] = 1
                    prov_sep[t] = (PROV_PROJECTED if med["proyectado"]
                                   else PROV_OBSERVED)
                    X[t, 37] = 1.0 if med["proyectado"] else 0.0
                    avail[t, 37] = 1
        # distancia muneca-barra
        mun = sel["munecas"]
        if mun is not None and len(eleg) == 1:
            geo = G.geometria_mascara_union(eleg, W, H)
            dw = min(float(np.hypot(m[0] - geo["cx"], m[1] - geo["cy"]))
                     for m in mun) / H
            X[t, 27] = dw
            avail[t, 27] = 1

        # --- QUALITY ---
        X[t, 33] = 1.0 if sel["estado"] == "asociada" else 0.0
        X[t, 34] = 1.0 if sel["estado"] == "observada_no_verificada" else 0.0
        X[t, 35] = 1.0 if sel["estado"] == "multiple_ambigua" else 0.0
        X[t, 36] = 1.0 if sel["estado"] == "ausente" else 0.0
        avail[t, 33:37] = 1
        # q_projected permanece NaN si no existe separacion evaluable.

    # --- BENCH estabilidad (video-level, repetida por frame) ---
    stab = np.nan
    if len(bench_boxes) >= 2:
        med = np.median(np.stack(bench_boxes), axis=0)
        a = np.maximum(0, np.minimum(med[2], np.stack(bench_boxes)[:, 2])
                       - np.maximum(med[0], np.stack(bench_boxes)[:, 0]))
        b = np.maximum(0, np.minimum(med[3], np.stack(bench_boxes)[:, 3])
                       - np.maximum(med[1], np.stack(bench_boxes)[:, 1]))
        inter = a * b
        area_m = max(0, med[2] - med[0]) * max(0, med[3] - med[1])
        area_b = (np.stack(bench_boxes)[:, 2] - np.stack(bench_boxes)[:, 0]) * (
            np.stack(bench_boxes)[:, 3] - np.stack(bench_boxes)[:, 1])
        union = area_m + area_b - inter
        iou = np.where(union > 0, inter / union, 0.0)
        stab = float(np.median(iou))
        X[:, 23] = stab
        avail[:, 23] = 1

    # --- DINAMICA (solo identidad continua) ---
    for t in range(1, T):
        if (bar_state[t] == "asociada" and bar_state[t - 1] == "asociada"
                and track_hist[t] == track_hist[t - 1]
                and np.isfinite(X[t, 0]) and np.isfinite(X[t - 1, 0])):
            dt = (int(fidx[t]) - int(fidx[t - 1])) / fps
            if dt > 0:
                X[t, 29] = (X[t, 0] - X[t - 1, 0]) / dt
                X[t, 30] = (X[t, 1] - X[t - 1, 1]) / dt
                avail[t, 29:31] = 1
                if np.isfinite(X[t, 5]) and np.isfinite(X[t - 1, 5]):
                    X[t, 31] = (X[t, 5] - X[t - 1, 5]) / dt
                    avail[t, 31] = 1
                if np.isfinite(X[t, 24]) and np.isfinite(X[t - 1, 24]):
                    # no atravesar cambios observado<->proyectado
                    if prov_sep[t] == prov_sep[t - 1]:
                        X[t, 32] = (X[t, 24] - X[t - 1, 24]) / dt
                        avail[t, 32] = 1

    # --- ESTIMACIONES (separadas, offline, desactivables) ---
    est = estimar_huecos(d, pose, bar_state, W, H, fps, config,
                         cache, instance_hist)
    calidad = np.zeros((T, len(QUALITY_NAMES)), dtype=np.uint8)
    calidad[:, 0] = np.asarray(bar_state) == "asociada"
    calidad[:, 1] = np.asarray(bar_state) == "observada_no_verificada"
    calidad[:, 2] = np.asarray(bar_state) == "multiple_ambigua"
    calidad[:, 3] = np.asarray(bar_state) == "ausente"
    calidad[:, 4] = avail[:, 11] == 0
    calidad[:, 5] = avail[:, 18] == 0
    calidad[:, 6] = [G.munecas_fiables(pose[t], W, H) is None for t in range(T)]
    calidad[:, 7] = [G.hombros_fiables(pose[t], W, H) is None for t in range(T)]
    calidad[:, 8] = X[:, 28] == 1
    calidad[:, 9] = avail[:, 17] == 0
    calidad[:, 10] = avail[:, 24] == 0
    calidad[:, 11] = prov_sep == PROV_PROJECTED
    calidad[:, 12] = est["frames"]
    calidad[:, 13] = avail[:, 29] == 1
    duracion_gap = np.zeros(T, dtype=np.int32)
    for t in range(T):
        if bar_state[t] != "asociada":
            duracion_gap[t] = 1 + (duracion_gap[t - 1] if t else 0)

    info = {"video_meta": meta, "fps": fps, "ancho": W, "alto": H,
            "bar_state": bar_state, "bench_stability_iou": stab,
            "bar_track_id": np.array(
                [t[0] if len(t) == 1 else (-1 if len(t) == 0 else -2)
                 for t in track_hist], dtype=np.int32),
            "bar_n_raw": np.array(
                [sum(1 for i in cache[t] if i["clase"] == "bar")
                 for t in range(T)], dtype=np.int16),
            "bar_n_kept": np.asarray([len(ids) for ids in instance_hist], dtype=np.int16),
            "bar_instance_id": np.asarray([ids[0] if len(ids) == 1 else -1
                                            for ids in instance_hist], dtype=np.int32)}
    return {"features": X, "availability": avail, "prov_bar": prov_bar,
            "prov_sep": prov_sep, "estimations": est, "info": info,
            "frame_index": np.asarray(fidx), "quality_flags": calidad,
            "bar_gap_frames": duracion_gap}


def estimar_huecos(d, pose, bar_state, W, H, fps, config, cache, instance_hist):
    """Huecos interiores cortos sin deteccion BAR cruda. Replica visor."""
    if not config.get("enable_estimations", True):
        return {"frames": np.zeros(len(bar_state), dtype=bool)}
    n_raw = []
    for t in sorted(cache):
        n_raw.append(sum(1 for i in cache[t] if i["clase"] == "bar"))
    n_raw = np.asarray(n_raw)
    T = len(bar_state)
    # anclas: barra asociada + conf>=0.65 + centrada en persona>=0.5 + sin recorte
    anclas = [None] * T
    for t in sorted(cache):
        if bar_state[t] != "asociada":
            continue
        # El ancla debe ser exactamente la instancia seleccionada, no otra BAR.
        cand = [i for i in cache[t] if i["idx_global"] in instance_hist[t]]
        if len(cand) != 1:
            continue
        b = cand[0]
        if b["conf"] < G.CONF_ANCLA_BARRA:
            continue
        pers = [i for i in cache[t] if i["clase"] == "person"]
        if not pers:
            continue
        p = max(pers, key=lambda i: i["conf"])
        bb = np.asarray(b["bbox"], dtype=float)
        pb = np.asarray(p["bbox"], dtype=float)
        if bb[1] < max(8, round(0.01 * H)) or bb[3] > H - max(8, round(0.01 * H)):
            continue
        ix0, iy0 = max(pb[0], bb[0]), max(pb[1], bb[1])
        ix1, iy1 = min(pb[2], bb[2]), min(pb[3], bb[3])
        area_b = max(0, bb[2] - bb[0]) * max(0, bb[3] - bb[1])
        inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
        if not area_b or inter / area_b < 0.5:
            continue
        anclas[t] = {"x": float((bb[0] + bb[2]) / 2),
                     "y": float((bb[1] + bb[3]) / 2),
                     "ancho": float(bb[2] - bb[0])}
    est_mask = np.zeros(T, dtype=bool)
    est_offset = np.full(T, np.nan, dtype=np.float32)
    est_ancho = np.full(T, np.nan, dtype=np.float32)
    est_anclas = np.full((T, 2), -1, dtype=np.int32)
    t = 0
    max_gap = int(config.get("max_estimated_gap", 6))
    while t < T:
        if anclas[t] is not None:
            t += 1
            continue
        s = t
        while t < T and anclas[t] is None:
            t += 1
        ant, post = s - 1, t
        if ant < 0 or post >= T or (post - ant - 1) > max_gap:
            continue
        if np.any(n_raw[s:post] > 0):
            continue
        ext = []
        ok = True
        for f in (ant, post):
            mun = G.munecas_fiables(pose[f], W, H)
            if mun is None:
                ok = False
                break
            a = anclas[f]
            aw = float(mun[1, 0] - mun[0, 0])
            if not (mun[0, 0] <= a["x"] <= mun[1, 0]) or a["ancho"] < 0.2 * aw:
                ok = False
                break
            ext.append((float(a["y"] - np.mean(mun[:, 1])), aw))
        if not ok or len(ext) != 2:
            continue
        (oa, aa), (ob, ab) = ext
        if (abs(oa - ob) > max(12, 0.03 * H)
                or abs(aa - ab) > 0.35 * max(aa, ab)):
            continue
        for f in range(s, post):
            frac = (f - ant) / (post - ant)
            est_mask[f] = True
            est_offset[f] = (1 - frac) * oa + frac * ob
            est_ancho[f] = (1 - frac) * aa + frac * ab
            est_anclas[f] = (ant, post)
    return {"frames": est_mask, "offset": est_offset,
            "ancho_munecas": est_ancho, "anclas": est_anclas,
            "n_estimated": int(est_mask.sum())}
