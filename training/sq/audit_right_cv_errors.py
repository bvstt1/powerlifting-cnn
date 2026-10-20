"""Auditoría comparativa OOF SQ right, sin inferencia, entrenamiento ni test.

python -m training.sq.audit_right_cv_errors --stage export
python -m training.sq.audit_right_cv_errors --stage visuals
python -m training.sq.audit_right_cv_errors --stage report
"""
import argparse
import json
from collections import defaultdict

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from training.sq import controlled_right as exp
from training.sq import train_posecnn as sq
from training.sq.run_analysis import digest, read_predictions


OUT = sq.ROOT / "training" / "results" / "sq" / "right" / "posecnn" / "development_diagnostics"
PRED = {variant: exp.run(variant)["results"] / "cv_predictions.csv" for variant in exp.VARIANTS}
FIELD = ("video_id", "fold", "main_split", "true_label", "baseline_pred", "baseline_probability_valid",
         "point_xy_pred", "point_xy_probability_valid", "error_class", "transition", "codigos", "criterios",
         "video_path", "pose_path")
STATES = ("corrected", "persistent", "new")
CLASSES = ("false_valid", "false_invalid")


def parsed_prediction(path, ids, y, role_by_id, expected_outer):
    rows = read_predictions(path)
    if len(rows) != len(ids) or len({r["video_id"] for r in rows}) != len(ids):
        raise ValueError(f"CV no tiene exactamente una predicción por vídeo: {path}")
    mapped = {}
    for r in rows:
        ident = r["video_id"]
        if ident not in role_by_id or role_by_id[ident] not in ("train", "validation"):
            raise ValueError(f"ID ajeno a desarrollo: {ident}")
        fold, truth, pred, prob = int(r["fold"]), int(r["true_label"]), int(r["predicted_label"]), float(r["probability_valid"])
        if fold not in range(1, 6) or truth not in (0, 1) or pred not in (0, 1) or not 0 <= prob <= 1:
            raise ValueError(f"Predicción inválida: {r}")
        if fold != expected_outer[ident] or truth != int(y[ids.index(ident)]) or pred != int(prob >= 0.5):
            raise ValueError(f"Fold, label o umbral no coinciden con split: {r}")
        mapped[ident] = {"fold": fold, "label": truth, "pred": pred, "prob": prob}
    if set(mapped) != set(ids):
        raise ValueError("CV incompleto o con IDs adicionales")
    return mapped


def classify(label, old_pred, new_pred):
    error_class = "false_valid" if label == 0 else "false_invalid"
    erroneous = 1 if label == 0 else 0
    before, after = old_pred == erroneous, new_pred == erroneous
    if before and after:
        return error_class, "persistent"
    if before:
        return error_class, "corrected"
    if after:
        return error_class, "new"
    return "correct", "correct"


def inputs():
    X, y, ids, _ = exp.development()  # abre exclusivamente train/validation
    ids = ids.astype(str).tolist()
    split = json.loads((exp.DATA / "membership.json").read_text(encoding="utf-8"))
    if split["split_sha256"] != digest(exp.DATA / "split.npz"):
        raise ValueError("Split cambiado")
    role = {ident: group for group, members in split["roles"].items() for ident in members}
    expected = {ids[i]: fold for fold, _, _, outer in exp.folds(ids, y) for i in outer}
    a, b = (parsed_prediction(PRED[variant], ids, y, role, expected) for variant in exp.VARIANTS)
    for ident in ids:
        if a[ident]["fold"] != b[ident]["fold"] or a[ident]["label"] != b[ident]["label"]:
            raise ValueError(f"CV discordante: {ident}")
    return X, ids, y, role, a, b


def verified_rows():
    X, ids, y, role, a, b = inputs()
    labels = {r["video_id"].strip(): r for r in sq.load_labels(exp.run("temporal")["labels"])}
    rows = []
    for ident in ids:
        old, new = a[ident], b[ident]
        if ident not in labels or int(labels[ident]["label"]) != old["label"]:
            raise ValueError(f"Etiqueta CSV distinta: {ident}")
        cls, state = classify(old["label"], old["pred"], new["pred"])
        video, pose = exp.VIDEO / (ident + ".mp4"), exp.run("temporal")["source"] / (ident + ".npy")
        if not video.is_file() or not pose.is_file():
            raise FileNotFoundError(video if not video.is_file() else pose)
        rows.append({"video_id": ident, "fold": old["fold"], "main_split": role[ident], "true_label": old["label"],
                     "baseline_pred": old["pred"], "baseline_probability_valid": old["prob"],
                     "point_xy_pred": new["pred"], "point_xy_probability_valid": new["prob"],
                     "error_class": cls, "transition": state, "codigos": labels[ident]["codigos"],
                     "criterios": labels[ident]["criterios"],
                     "video_path": str(video.relative_to(sq.ROOT)), "pose_path": str(pose.relative_to(sq.ROOT))})
    return X, rows


def validate_matrices(rows):
    for variant, pred in (("temporal", "baseline_pred"), ("point_xy", "point_xy_pred")):
        stored = {int(r["fold"]): json.loads(r["matrix"]) for r in read_predictions(exp.run(variant)["results"] / "cv_metrics.csv")}
        if len(stored) != 5:
            raise ValueError("Faltan métricas CV")
        for fold in range(1, 6):
            group = [r for r in rows if r["fold"] == fold]
            matrix = [[sum(r["true_label"] == truth and r[pred] == guess for r in group) for guess in (0, 1)]
                      for truth in (0, 1)]
            if matrix != stored[fold]:
                raise ValueError(f"Matriz distinta {variant} fold {fold}: {matrix} != {stored[fold]}")


def quality(row, processed):
    ident = row["video_id"]
    raw = np.load(sq.ROOT / row["pose_path"], allow_pickle=False)
    cap = cv2.VideoCapture(str(sq.ROOT / row["video_path"]))
    if not cap.isOpened():
        raise FileNotFoundError(row["video_path"])
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if abs(frames - len(raw)) > 1 or raw.ndim != 3 or raw.shape[1:] != (5, 3):
        raise ValueError(f"Vídeo/pose no coinciden: {ident}")
    pre, center, scale = sq.preprocess_left(raw)
    difference = float(np.max(np.abs(pre - processed)))
    if difference > 2e-5:
        raise ValueError(f"Preprocesamiento distinto del split: {ident}, {difference}")
    xy, conf = raw[..., :2], raw[..., 2]
    missing = ~np.isfinite(xy).all(axis=2) | ((xy == 0).all(axis=2) & (conf == 0))
    finite = np.isfinite(xy).all(axis=2)
    valid_jumps = finite[1:] & finite[:-1]
    jumps = np.linalg.norm(np.diff(xy, axis=0), axis=2) / scale
    jumps[~valid_jumps] = np.nan
    proxy = (xy[:, 1, 1] - xy[:, 0, 1]) / scale
    pre_proxy = pre[:, 1, 1] - pre[:, 0, 1]
    # Hipótesis de saltos geométricos, NO detección certificada de swaps.
    indices = np.argwhere(np.isfinite(jumps) & (jumps > .5))
    vertical_alert = (xy[:, 0, 1] > xy[:, 2, 1]) | (xy[:, 1, 1] > xy[:, 2, 1])
    return {"video_id": ident, "error_class": row["error_class"], "transition": row["transition"],
            "fold": row["fold"], "T_video": frames, "T_pose": len(raw), "preprocess_max_abs_error": difference,
            "missing_xy_ratio": float(missing.mean()), "missing_frames": int(missing.all(axis=1).sum()),
            "longest_missing_gap": max(sq.longest_run(missing[:, i]) for i in range(5)),
            "confidence_median": float(np.nanmedian(conf)), "confidence_min": float(np.nanmin(conf)),
            "max_jump_tibias": float(np.nanmax(jumps)), "jumps_over_half_tibia": int(len(indices)),
            "vertical_order_alert_frames": int(vertical_alert.sum()),
            "hip_knee_min_raw": float(np.nanmin(proxy)), "hip_knee_max_raw": float(np.nanmax(proxy)),
            "hip_knee_min_pre": float(np.min(pre_proxy)), "hip_knee_max_pre": float(np.max(pre_proxy)),
            "hip_knee_range_raw": float(np.nanmax(proxy)-np.nanmin(proxy)),
            "hip_knee_range_pre": float(np.max(pre_proxy)-np.min(pre_proxy)),
            "ankle_displacement_max_tibias": float(np.nanmax(np.linalg.norm(xy[:, 2] - np.nanmedian(xy[:, 2], axis=0),axis=1))/scale),
            "toe_displacement_max_tibias": float(np.nanmax(np.linalg.norm(xy[:, 3] - np.nanmedian(xy[:, 3], axis=0),axis=1))/scale),
            "video_path": row["video_path"], "pose_path": row["pose_path"]}


def snapshot():
    files = [exp.DATA / "split.npz", exp.DATA / "split.csv", exp.run("temporal")["labels"]]
    for variant in exp.VARIANTS:
        p = exp.run(variant)
        files.extend([p["results"] / "cv_predictions.csv", p["results"] / "cv_metrics.csv",
                      p["models"] / "config.json"])
        files.extend(folder / name for folder in (p["models"] / "cv" / f"fold_{i}" for i in range(1, 6))
                     for name in ("scaler.pkl", "best_model.pth", "membership.csv"))
    files.extend([exp.run("point_xy")["models"] / "best_model.pth", exp.run("point_xy")["models"] / "scaler.pkl"])
    return {str(p.relative_to(sq.ROOT)): digest(p) for p in files}


def export():
    X, rows = verified_rows()
    validate_matrices(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = OUT / "source_hashes.json"
    current = snapshot()
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != current:
        raise ValueError("Artefactos de origen alterados desde la auditoría previa")
    if not manifest.exists():
        sq.save_json(manifest, current)
    sq.save_csv(OUT / "cv_comparison_by_video.csv", rows, FIELD)
    groups = defaultdict(list)
    for r in rows:
        if r["error_class"] != "correct":
            groups[r["error_class"], r["transition"]].append(r)
    for cls in CLASSES:
        for state in STATES:
            subset = sorted(groups[cls, state], key=lambda r:r["video_id"])
            sq.save_csv(OUT / f"cv_{cls}_{state}.csv", subset, FIELD)
        combined = sorted([r for r in rows if r["error_class"] == cls],key=lambda r:r["video_id"])
        sq.save_csv(OUT / f"cv_{cls}_before_after.csv", combined, FIELD)

    by_id = {ident: X[i] for i,ident in enumerate([r["video_id"] for r in rows])}
    # Todos los falsos válidos antes/después + controles correctamente clasificados
    # por ambas variantes (no escoger controles según valores del test).
    audited = [r for r in rows if r["error_class"] == "false_valid" or
               (r["true_label"] == 0 and r["error_class"] == "correct")]
    quality_rows = [quality(r,by_id[r["video_id"]]) for r in audited]
    sq.save_csv(OUT / "invalid_cv_trajectories.csv", quality_rows, list(quality_rows[0]))
    selected_quality = {r["video_id"]:r for r in quality_rows}
    # Persistentes de probabilidad más alta, nuevos y casos de faltas temporal/pie.
    priority = [r for r in rows if r["error_class"] == "false_valid" and r["transition"] in ("persistent","new")]
    priority.sort(key=lambda r:(0 if r["transition"] == "persistent" else 1,
                                -r["point_xy_probability_valid"],r["video_id"]))
    review = []
    for rank,r in enumerate(priority,1):
        code = set(r["codigos"].split("|"))
        review.append({"priority":rank, **r,"reason":("persistente alta p" if r["transition"]=="persistent" else "nuevo falso válido") +
                       ("; revisión de rebote/pies en vídeo completo" if code & {"SQ-02","SQ-04"} else ""),
                       "T_video":selected_quality[r["video_id"]]["T_video"],
                       "max_jump_tibias":selected_quality[r["video_id"]]["max_jump_tibias"]})
    sq.save_csv(OUT / "manual_review_priority.csv",review,list(review[0]))
    summary = {"n_development":len(rows),"n_invalid":sum(r["true_label"]==0 for r in rows),
               "n_valid":sum(r["true_label"]==1 for r in rows),"counts":{},"by_fold":{},"codes_nonexclusive":{},
               "audited_invalid":len(audited),"source_hashes":str(manifest.relative_to(sq.ROOT))}
    for fold in ("all",1,2,3,4,5):
        subset = rows if fold == "all" else [r for r in rows if r["fold"]==fold]
        item = {cls:{state:sum(r["error_class"]==cls and r["transition"]==state for r in subset) for state in STATES}
                for cls in CLASSES}
        for cls in CLASSES:
            item[cls]["before"] = item[cls]["corrected"]+item[cls]["persistent"]
            item[cls]["after"] = item[cls]["persistent"]+item[cls]["new"]
        summary["counts" if fold=="all" else "by_fold"][str(fold)] = item
    for code in sorted({c for r in rows if r["true_label"]==0 for c in r["codigos"].split("|") if c}):
        subset = [r for r in rows if r["true_label"]==0 and code in r["codigos"].split("|")]
        summary["codes_nonexclusive"][code] = {"n_videos_with_code":len(subset),
            "before_false_valid":sum(r["baseline_pred"]==1 for r in subset),
            "after_false_valid":sum(r["point_xy_pred"]==1 for r in subset),
            **{state:[r["video_id"] for r in subset if r["error_class"]=="false_valid" and r["transition"]==state]
               for state in STATES}}
    controls = [r for r in audited if r["error_class"] == "correct"]
    summary["controls"] = [r["video_id"] for r in controls[:8]]
    summary["quality_summary"] = {"missing_xy_nonzero":sum(q["missing_xy_ratio"]>0 for q in quality_rows),
        "frame_mismatch":sum(abs(q["T_video"]-q["T_pose"])>1 for q in quality_rows),
        "max_reconstruction_error":max(q["preprocess_max_abs_error"] for q in quality_rows),
        "jumps_over_half_tibia_videos":sum(q["jumps_over_half_tibia"]>0 for q in quality_rows)}
    summary["quality_by_group"] = {}
    for group in ("corrected", "persistent", "new", "correct"):
        subset=[q for q in quality_rows if q["transition"]==group]
        summary["quality_by_group"][group]={"n":len(subset),
            "confidence_median":float(np.median([q["confidence_median"] for q in subset])),
            "max_jump_tibias_median":float(np.median([q["max_jump_tibias"] for q in subset])),
            "hip_knee_max_median":float(np.median([q["hip_knee_max_raw"] for q in subset])),
            "hip_knee_resample_max_difference":float(max(abs(q["hip_knee_max_raw"]-q["hip_knee_max_pre"]) for q in subset)),
            "vertical_order_alert_videos":sum(q["vertical_order_alert_frames"]>0 for q in subset)}
    sq.save_json(OUT / "cv_error_summary.json",summary)
    if snapshot() != current:
        raise RuntimeError("Un artefacto original cambió durante la auditoría")
    print(json.dumps(summary,indent=2,ensure_ascii=False))


def draw_overlay(frame, points, ident, index, total):
    height,width=frame.shape[:2]
    for k,color in enumerate(((0,0,255),(0,255,0),(255,255,0),(255,0,255),(0,255,255))):
        x,y=points[k,:2]
        if np.isfinite([x,y]).all():
            pt=(int(x*width),int(y*height))
            cv2.circle(frame,pt,6,color,-1)
            cv2.putText(frame,str(k),(pt[0]+6,pt[1]-5),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2)
    cv2.putText(frame,f"{ident}  {index+1}/{total}",(10,28),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
    return frame


def visuals():
    # Todas las trayectorias prioritarias se reproducen cuadro por cuadro al exportar;
    # los vídeos resultantes permiten revisar rebotes/pasos a velocidad normal.
    rows = read_predictions(OUT / "manual_review_priority.csv")
    controls = read_predictions(OUT / "cv_comparison_by_video.csv")
    controls = [r for r in controls if r["true_label"] == "0" and r["error_class"] == "correct"][:3]
    selected = rows + controls
    selected = list({r["video_id"]:r for r in selected}.values())
    target = OUT / "visual_evidence"
    target.mkdir(parents=True,exist_ok=True)
    log=[]
    for row in selected:
        ident=row["video_id"]
        raw=np.load(sq.ROOT / row["pose_path"],allow_pickle=False)
        pre,_,scale=sq.preprocess_left(raw)
        cap=cv2.VideoCapture(str(sq.ROOT / row["video_path"]))
        if not cap.isOpened():
            raise FileNotFoundError(row["video_path"])
        fps=cap.get(cv2.CAP_PROP_FPS)
        fps=fps if np.isfinite(fps) and fps>0 else 30
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        mp4=target / f"{ident}_full_overlay.mp4"
        writer=cv2.VideoWriter(str(mp4),cv2.VideoWriter_fourcc(*"mp4v"),fps,(width,height))
        if not writer.isOpened():
            cap.release();raise RuntimeError(f"No se puede exportar {mp4}")
        picks=set(np.linspace(0,len(raw)-1,min(16,len(raw)),dtype=int).tolist())
        thumbnail=[];count=0
        while True:
            ok,frame=cap.read()
            if not ok:break
            if count>=len(raw):
                writer.release();cap.release();raise ValueError(f"Vídeo más largo que pose {ident}")
            frame=draw_overlay(frame,raw[count],ident,count,len(raw))
            writer.write(frame)
            if count in picks:
                thumbnail.append(cv2.resize(frame,(240,320)))
            count+=1
        cap.release();writer.release()
        if abs(count-len(raw))>1 or not mp4.is_file():
            raise ValueError(f"Vídeo completo ilegible {ident}: {count} vs {len(raw)}")
        sheet=np.concatenate([np.concatenate(thumbnail[i:i+4],axis=1) for i in range(0,16,4)],axis=0)
        jpg=target / f"{ident}_timeline.jpg"
        cv2.imwrite(str(jpg),sheet)
        raw_proxy=(raw[:,1,1]-raw[:,0,1])/scale
        post_proxy=pre[:,1,1]-pre[:,0,1]
        fig,axes=plt.subplots(2,1,figsize=(10,5))
        axes[0].plot(np.linspace(0,1,len(raw_proxy)),raw_proxy,label="Original / tibia")
        axes[0].plot(np.linspace(0,1,len(post_proxy)),post_proxy,label="Preprocesado T170",alpha=.75)
        axes[0].set(ylabel="y(cadera)-y(rodilla)",xlabel="Tiempo relativo",title=f"{ident} | fold {row['fold']} | {row['transition']}")
        axes[0].legend()
        for k,name in ((2,"tobillo"),(3,"punta"),(4,"talón")):
            axes[1].plot(np.linspace(0,1,len(raw)),np.linalg.norm(raw[:,k,:2]-np.nanmedian(raw[:,k,:2],axis=0),axis=1)/scale,label=name)
        axes[1].set(ylabel="Desplazamiento / tibia",xlabel="Tiempo relativo")
        axes[1].legend();fig.tight_layout()
        png=target / f"{ident}_trajectories.png";fig.savefig(png);plt.close(fig)
        log.append({"video_id":ident,"source_video":row["video_path"],"frames_decoded":count,"fps":fps,
                    "annotated_full_video":str(mp4.relative_to(sq.ROOT)),"timeline_16_frames":str(jpg.relative_to(sq.ROOT)),
                    "trajectory_plot":str(png.relative_to(sq.ROOT)),"visual_interpretation":"pendiente revisión humana de vídeo completo"})
    sq.save_csv(OUT / "visual_evidence_manifest.csv",log,list(log[0]))
    print("Vídeos completos anotados:",[r["video_id"] for r in log])


def report():
    summary=json.loads((OUT / "cv_error_summary.json").read_text(encoding="utf-8"))
    rows=read_predictions(OUT / "cv_comparison_by_video.csv")
    quality=read_predictions(OUT / "invalid_cv_trajectories.csv")
    evidence=read_predictions(OUT / "visual_evidence_manifest.csv") if (OUT / "visual_evidence_manifest.csv").exists() else []
    def fmt(cls,fold):
        data=summary["counts"]["all"][cls] if fold=="all" else summary["by_fold"][str(fold)][cls]
        return f"{data['before']} | {data['corrected']} | {data['persistent']} | {data['new']} | {data['after']}"
    lines=["# SQ right — auditoría comparativa de errores de CV externo (solo desarrollo)","",
           "`baseline_t170_v2` → `point_xy_v1`; 0=inválido, 1=válido, umbral fijo 0.5. Ningún peso, scaler, partición, etiqueta o predicción original se modifica.",
           f"Una predicción externa por cada uno de **{summary['n_development']}** IDs (inválidos {summary['n_invalid']}, válidos {summary['n_valid']}); etiquetas y folds idénticos entre variantes y contrastados con membresías y split. Cada matriz de `cv_metrics.csv` fue reconstruida a partir de las predicciones, con orden `[[TN,FP],[FN,TP]]`.",
           "", "## Transiciones globales y por fold", "",
           "| Clase | Fold | Antes | Corregidos | Persistentes | Nuevos | Después |","|---|---:|---:|---:|---:|---:|---:|"]
    for cls in CLASSES:
        for fold in ("all",1,2,3,4,5):
            lines.append(f"| {'Falso válido (FP)' if cls=='false_valid' else 'Falso inválido (FN)'} | {fold} | {fmt(cls,fold)} |")
    lines.extend(["","Los CSV `cv_false_valid_<corrected|persistent|new>.csv` y `cv_false_invalid_<...>.csv` contienen **todos** los IDs, fold y p(válido) de ambos modelos. `cv_comparison_by_video.csv` incluye también las predicciones correctas.",""])
    for cls in CLASSES:
        lines.extend([f"### {cls}: IDs con p(válido) antes → después", "", "| Estado | Fold / ID / probabilidades |", "|---|---|"])
        for state in STATES:
            subset=[r for r in rows if r["error_class"]==cls and r["transition"]==state]
            values="; ".join(f"{r['fold']}:{r['video_id']} {float(r['baseline_probability_valid']):.3f}→{float(r['point_xy_probability_valid']):.3f}" for r in subset)
            lines.append(f"| {state} ({len(subset)}) | {values or '—'} |")
        lines.append("")
    lines.extend(["## Códigos de falta (vídeos inválidos; apariciones no exclusivas)","",
                  "| Código | Vídeos | FP antes | FP después | Corregidos | Persistentes | Nuevos |","|---|---:|---:|---:|---|---|---|"])
    for code,data in summary["codes_nonexclusive"].items():
        lines.append(f"| {code} | {data['n_videos_with_code']} | {data['before_false_valid']} | {data['after_false_valid']} | {', '.join(data['corrected']) or '—'} | {', '.join(data['persistent']) or '—'} | {', '.join(data['new']) or '—'} |")
    lines.extend(["","Un vídeo puede pertenecer a múltiples códigos; **no sumar columnas de códigos como vídeos exclusivos**. `criterios` literal y códigos de cada ID están en `cv_comparison_by_video.csv`.",
                  "", "## Integridad y calidad de la señal", "",
                  f"Se reprodujo el preprocesamiento y se comparó con `split.npz` para {summary['audited_invalid']} inválidos de desarrollo; error absoluto máximo {summary['quality_summary']['max_reconstruction_error']:.2g}. Desajustes de longitud vídeo–pose (>1 frame): {summary['quality_summary']['frame_mismatch']}; secuencias con XY faltante: {summary['quality_summary']['missing_xy_nonzero']}; saltos >0.5 tibia en algún punto: {summary['quality_summary']['jumps_over_half_tibia_videos']}. Métricas individuales: `invalid_cv_trajectories.csv` (incluye controles acertados por ambas redes).",
                  "Centro de cadera y escala de tibia constantes por vídeo preservan la diferencia vertical cadera–rodilla en cada frame original; la interpolación a 170 puntos aproxima los extremos y puede suavizar cambios breves. `hip_knee_*_raw/pre` y desplazamientos de pie del CSV documentan la comparación. El proxy no decide automáticamente SQ-01 ni identifica rebote. La confianza YOLO alta no verifica la precisión espacial ni descarta swaps locales.",
                  "", "| Inválidos (OOF) | N | Mediana confianza | Mediana salto máximo (tibias) | Mediana máximo cadera–rodilla (tibias) | Máx. diferencia proxy T170 | Alerta orden vertical (vídeos) |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    for state in (*STATES,"correct"):
        q=summary["quality_by_group"][state]
        lines.append(f"| {state} | {q['n']} | {q['confidence_median']:.6f} | {q['max_jump_tibias_median']:.3f} | {q['hip_knee_max_median']:.3f} | {q['hip_knee_resample_max_difference']:.4f} | {q['vertical_order_alert_videos']} |")
    lines.extend(["","La alerta vertical solo busca casos groseros (rodilla o cadera por debajo del tobillo); 0 alertas **no** certifica ausencia de swaps finos. La mediana de máximos del proxy no equivale a la profundidad reglamentaria medida desde la articulación real.",
                  "", "## Casos prioritarios y evidencia visual", "",
                  "`manual_review_priority.csv` ordena persistentes por p(válido) Point XY y luego nuevos, incluye descripción literal y ruta real de vídeo/pose. Se exportaron vídeos **completos** anotados para los 16 errores FP restantes y tres controles, más gráficos de toda la secuencia y mosaicos de 16 instantes. Se inspeccionaron los mosaicos y gráficos de varios casos; **no se pudo reproducir visualmente aquí cada MP4 completo**. Para confirmar rebote, paso o swaps de corta duración queda obligatoria su inspección cuadro a cuadro/a velocidad normal.",""])
    for r in evidence:
        row=next(item for item in rows if item["video_id"]==r["video_id"])
        lines.append(f"- `{r['video_id']}` ({row['transition']}, fold {row['fold']}, p {float(row['baseline_probability_valid']):.3f}→{float(row['point_xy_probability_valid']):.3f}, códigos `{row['codigos'] or 'sin código'}`): vídeo completo `{r['annotated_full_video']}`, secuencia muestreada `{r['timeline_16_frames']}`, gráfico `{r['trajectory_plot']}`. Interpretación visual específica: **pendiente**.")
    lines.extend(["", "**Ejemplos observados en mosaicos y curvas sincronizadas:**",
                  "- `sq_237` (SQ-01; persistente, p 0.660→0.980): en los 16 frames se ve descenso y ascenso, con dos máximos locales del proxy cadera–rodilla cerca del fondo; no se puede juzgar de forma concluyente la profundidad normativa de la articulación desde estos puntos ni decidir si el label es inconsistente. Solicitar revisión manual del vídeo íntegro.",
                  "- `sq_369` y `sq_288` (SQ-02; persistentes p 0.938→0.910 y 0.632→0.898): las curvas de toda la secuencia muestran una caída y un segundo ascenso del proxy cerca del fondo. Es **compatible** con reversión/rebote temporal; no demuestra por sí solo una falta reglamentaria. Revisar MP4 completos a velocidad normal.",
                  "- `sq_372` (SQ-02; nuevo p 0.369→0.668): la curva presenta dos máximos separados por un descenso notable; puede corresponder a una oscilación o a más de un intento dentro del clip. Requiere aclarar el criterio/segmento temporal manualmente, sin retocar etiqueta por la red.",
                  "- `sq_360` (SQ-04 persistente) y `sq_333` (SQ-04 nuevo): hay desplazamientos de puntos de pie respecto a su mediana temporal, con escalones evidentes para `sq_333`; las imágenes muestran cambios de posición. Movimiento de cámara/cuerpo y proyección 2D impiden certificar pasos solo con este proxy. Vídeo completo exportado para inspección.",
                  "- Controles inválidos acertados por ambas redes `sq_361`, `sq_269`, `sq_285`: vídeos/poses y preprocesamiento también concuerdan; muestran confianza similar y movimientos sin que estas métricas por sí solas expliquen la decisión del clasificador.",
                  "", "### Evidencia comprobada, hipótesis y pendientes", "",
                  "- **Comprobado:** IDs/folds/labels concordantes, vídeo/pose existente, duración y preprocesamiento coincidentes para todos los inválidos analizados; distribuciones de confianza, XY, saltos y proxy medidos por vídeo. Diferencias de error entre scalers provienen de predicciones externas guardadas.",
                  "- **Hipótesis:** estandarizar confianza y geometría por timestep en el scaler temporal puede atenuar la generalización; una falta por rebote o paso podría quedar subrepresentada en cinco puntos o por T170. Estas explicaciones no se demuestran por el resultado del modelo.",
                  "- **Pendiente:** lectura humana completa de los vídeos anotados prioritarios para confirmar rebote, paso, intercambio anatómico puntual o ambigüedad del texto del criterio; revisar especialmente `sq_237` y `sq_372`. Cualquier etiqueta sospechosa se documenta, nunca se cambia por la predicción.",
                  "", "## Test: separado estrictamente", "",
                  "Los cuatro errores test del entrenamiento final (`sq_362`, `sq_297`, `sq_231` falsos válidos y `sq_326` falso inválido) constan en `training/results/sq/right/posecnn/runs/point_xy_v1/report.md`. **No** se incluyeron en los CSV, controles, gráficos ni prioridades de esta auditoría de desarrollo; coincidencias de ID en códigos/errores solo se interpretan según el split real.",
                  "", "## Abrir revisión (PowerShell desde la raíz)", "", "```powershell",
                  "python -m training.sq.audit_right_cv_errors --stage export",
                  "python -m training.sq.audit_right_cv_errors --stage visuals",
                  "python -m training.sq.audit_right_cv_errors --stage report",
                  "Import-Csv 'training/results/sq/right/posecnn/development_diagnostics/manual_review_priority.csv' | Out-GridView",
                  "Invoke-Item 'training/results/sq/right/posecnn/development_diagnostics/visual_evidence'",
                  "```", "", "Los hashes de artefactos de origen están en `source_hashes.json`. Volver a exportar compara sus hashes y rechaza cambios; solamente se escriben resultados bajo `development_diagnostics/` y este reporte."])
    target=OUT / "report.md"
    target.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(target)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",required=True,choices=("export","visuals","report"))
    args=parser.parse_args()
    {"export":export,"visuals":visuals,"report":report}[args.stage]()
