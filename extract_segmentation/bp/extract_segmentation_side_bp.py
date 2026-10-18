"""Auditoría, extracción RAW y verificación independientes de BP left/right.

Uso: python extract_segmentation/bp/extract_segmentation_side_bp.py --view left --phase audit
     python extract_segmentation/bp/extract_segmentation_side_bp.py --view left --phase pilot
     python extract_segmentation/bp/extract_segmentation_side_bp.py --view left --phase all
     python extract_segmentation/bp/extract_segmentation_side_bp.py --view left --phase verify
"""
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
MODELO = RAIZ / "models/bp/bp_side_seg_v1.pt"
VERSION = "bp_side_segmentation_raw_v1"
ESQUEMA = "bp_side_segmentation_raw_v1"
INFERENCIA = {"imgsz": 960, "conf": 0.25, "iou": 0.70,
             "retina_masks": True, "vid_stride": 1, "augment": False,
             "tracking": False, "batch": 4}
CLASES = {0: "bar", 1: "bench", 2: "person"}


def sha(ruta):
    hash_ = hashlib.sha256()
    with open(ruta, "rb") as archivo:
        for bloque in iter(lambda: archivo.read(4 * 1024 * 1024), b""):
            hash_.update(bloque)
    return hash_.hexdigest()


def guardar_json(ruta, objeto):
    temporal = ruta.with_suffix(ruta.suffix + ".tmp")
    temporal.write_text(json.dumps(objeto, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporal, ruta)


def guardar_csv(ruta, filas, campos):
    temporal = ruta.with_suffix(ruta.suffix + ".tmp")
    with temporal.open("w", encoding="utf-8-sig", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=campos, extrasaction="ignore")
        escritor.writeheader()
        escritor.writerows(filas)
    # Indexadores de Windows pueden mantener abierto el CSV unos instantes.
    for intento in range(20):
        try:
            os.replace(temporal, ruta)
            break
        except PermissionError:
            if intento == 19:
                raise
            time.sleep(min(.2 * (intento + 1), 2.0))


def filas_csv(ruta):
    if not ruta.exists():
        return []
    with ruta.open(encoding="utf-8-sig", newline="") as archivo:
        return list(csv.DictReader(archivo))


def rutas(vista):
    base = RAIZ / "segmentation/bp" / vista
    return (RAIZ / "dataset/bp" / vista, RAIZ / "keypoints/bp/yolo" / vista,
            RAIZ / "training/data/bp" / vista / "posecnn/split.csv", base)


def inspeccionar_modelo():
    from ultralytics import YOLO
    import ultralytics
    if not MODELO.is_file():
        raise FileNotFoundError(MODELO)
    modelo = YOLO(str(MODELO))
    clases = {int(k): str(v) for k, v in modelo.names.items()}
    if modelo.task != "segment" or clases != CLASES:
        raise RuntimeError(f"Checkpoint incompatible: task={modelo.task}, classes={clases}")
    return modelo, {"model_path": str(MODELO), "model_sha256": sha(MODELO),
                    "task": modelo.task, "classes": clases,
                    "training_imgsz": modelo.overrides.get("imgsz"),
                    "checkpoint_overrides": {k: str(v) for k, v in modelo.overrides.items()},
                    "model_yaml": str(modelo.model.yaml.get("yaml_file")),
                    "ultralytics_version": ultralytics.__version__}


def auditar(vista, entrada, puntos, split, base):
    (base / "manifests").mkdir(parents=True, exist_ok=True)
    videos = sorted(entrada.glob("*.mp4"))
    npys = sorted(puntos.glob("*.npy"))
    if not videos:
        raise RuntimeError(f"Sin videos en {entrada}")
    tabla_split = {fila["video_id"]: fila["split"] for fila in filas_csv(split)}
    duplicados = sorted(k for k, v in Counter(p.stem.casefold() for p in videos).items() if v > 1)
    duplicados_npy = sorted(k for k, v in Counter(p.stem.casefold() for p in npys).items() if v > 1)
    filas = []
    for numero, video in enumerate(videos, 1):
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        declarados = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ancho, alto = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        contados = 0
        primer_frame = False
        if cap.isOpened():
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                primer_frame = True
                contados += 1
                if (frame.shape[1], frame.shape[0]) != (ancho, alto):
                    raise RuntimeError(f"Resolución variable: {video}")
        cap.release()
        keypoint = puntos / (video.stem + ".npy")
        forma = ""
        if keypoint.exists():
            try:
                forma = str(np.load(keypoint, mmap_mode="r", allow_pickle=False).shape)
            except (ValueError, OSError) as exc:
                forma = "ERROR: " + str(exc)
        estado = "OK" if primer_frame and fps > 0 and math.isfinite(fps) and contados == declarados else "CHECK_VIDEO"
        filas.append({"video_id": video.stem, "view": vista, "video_path": str(video),
                      "keypoints_path": str(keypoint) if keypoint.exists() else "",
                      "keypoints_shape": forma, "keypoints_match_frames":
                      str(forma.startswith(f"({contados}, ")) if forma else "",
                      "split_posecnn": tabla_split.get(video.stem, "unassigned"),
                      "frames_reported": declarados, "frames_decoded": contados,
                      "width": ancho, "height": alto, "fps": fps, "status": estado,
                      "physical_identity": "unverified"})
        if numero % 100 == 0:
            print(vista, "auditoría", numero, "/", len(videos), flush=True)
    guardar_csv(base / "manifests/input_audit.csv", filas, list(filas[0]))
    informe = {"view": vista, "videos": len(videos), "keypoints": len(npys),
               "without_keypoints": sorted({p.stem for p in videos} - {p.stem for p in npys}),
               "orphan_keypoints": sorted({p.stem for p in npys} - {p.stem for p in videos}),
               "duplicate_video_ids": duplicados, "duplicate_keypoint_ids": duplicados_npy,
               "unreadable_or_mismatched": [r["video_id"] for r in filas if r["status"] != "OK"],
               "frame_mismatch_keypoints": [r["video_id"] for r in filas if r["keypoints_match_frames"] == "False"],
               "split_file": str(split) if split.exists() else None,
               "split_counts": dict(Counter(r["split_posecnn"] for r in filas)),
               "physical_identity": "unverified; filenames across views are not pairs",
               "pose_source": "extract_keypoints/bp/extract_keypoints_yolo_bp.py; bp_side_skeleton_v2.pt; imgsz=960; highest confidence person per frame",
               "fps_counts": dict(Counter(str(r["fps"]) for r in filas)),
               "resolution_counts": dict(Counter(f'{r["width"]}x{r["height"]}' for r in filas))}
    guardar_json(base / "manifests/input_audit.json", informe)
    print(json.dumps(informe, indent=2), flush=True)
    return filas


def verificar_archivo(ruta, modelo_hash=None, video_hash=None):
    with np.load(ruta, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    meta = json.loads(str(d["metadata_json"]))
    assert str(d["schema_version"]) == ESQUEMA
    assert meta["model_sha256"] == (modelo_hash or meta["model_sha256"])
    assert meta["video_sha256"] == (video_hash or meta["video_sha256"])
    assert meta["inference"] == INFERENCIA and meta["view"] in ("left", "right")
    assert meta["physical_identity"] == "unverified"
    fps = float(meta["fps"])
    assert math.isfinite(fps) and fps > 0
    t = len(d["frame_index"])
    n = len(d["instance_frame_index"])
    assert t > 0 and t == int(meta["frames_decoded"]) and n == int(meta["instances_total"])
    assert np.array_equal(d["frame_index"], np.arange(t))
    assert np.allclose(d["timestamp_sec"], np.arange(t) / fps, rtol=0, atol=1e-9)
    formas = d["frame_image_shape_hw"]
    assert formas.shape == (t, 2) and np.all(formas > 0)
    fo, mo = d["frame_instance_offsets"], d["mask_byte_offsets"]
    assert fo.shape == (t + 1,) and fo[0] == 0 and fo[-1] == n and np.all(np.diff(fo) >= 0)
    assert d["frame_class_counts"].shape == (t, len(CLASES))
    assert np.all(d["frame_class_counts"] >= 0)
    assert np.array_equal(d["frame_class_counts"].sum(axis=1), np.diff(fo))
    assert mo.shape == (n + 1,) and mo[0] == 0 and mo[-1] == len(d["mask_bits"])
    assert np.all(np.diff(mo) >= 0)
    for clave in ("instance_index_in_frame", "instance_class_id", "instance_class_name",
                  "instance_confidence", "instance_bbox_xyxy_px", "instance_bbox_xyxy_normalized",
                  "instance_mask_origin_xy", "instance_mask_shape_hw", "instance_mask_area_px"):
        assert len(d[clave]) == n, clave
    assert not ("instance_track_id" in d)  # No se realizó tracking.
    assert np.all(np.isfinite(d["instance_confidence"]))
    assert np.all((d["instance_confidence"] >= 0) & (d["instance_confidence"] <= 1))
    for frame in range(t):
        reales = Counter()
        for j in range(int(fo[frame]), int(fo[frame + 1])):
            assert d["instance_frame_index"][j] == frame
            assert d["instance_index_in_frame"][j] == j - fo[frame]
            clase = int(d["instance_class_id"][j])
            assert str(d["instance_class_name"][j]) == CLASES[clase]
            reales[clase] += 1
            h, w = map(int, formas[frame])
            x, y = map(int, d["instance_mask_origin_xy"][j])
            hc, wc = map(int, d["instance_mask_shape_hw"][j])
            assert hc > 0 and wc > 0 and 0 <= x < x + wc <= w and 0 <= y < y + hc <= h
            assert mo[j + 1] - mo[j] == (hc * wc + 7) // 8
            crop = np.unpackbits(d["mask_bits"][mo[j]:mo[j + 1]], bitorder="little", count=hc * wc).reshape(hc, wc)
            assert crop.dtype == np.uint8 and crop.any()
            assert np.any(crop[0]) and np.any(crop[-1]) and np.any(crop[:, 0]) and np.any(crop[:, -1])
            assert int(np.count_nonzero(crop)) == int(d["instance_mask_area_px"][j])
            caja = d["instance_bbox_xyxy_px"][j]
            assert np.all(np.isfinite(caja)) and 0 <= caja[0] < caja[2] <= w and 0 <= caja[1] < caja[3] <= h
            assert np.allclose(d["instance_bbox_xyxy_normalized"][j], caja / [w, h, w, h], atol=1e-6)
        assert np.array_equal(d["frame_class_counts"][frame], [reales[k] for k in CLASES])
    return {"frames": t, "instances": n, "bytes": ruta.stat().st_size, "metadata": meta,
            "class_instances": dict(zip(CLASES.values(), map(int, d["frame_class_counts"].sum(axis=0)))),
            "missing_frames": dict(zip(CLASES.values(), map(int, (d["frame_class_counts"] == 0).sum(axis=0)))),
            "multiple_frames": dict(zip(CLASES.values(), map(int, (d["frame_class_counts"] > 1).sum(axis=0)))),
            "packed_bytes": len(d["mask_bits"]),
            "uncompressed_full_mask_bytes": int(sum(d["frame_image_shape_hw"][d["instance_frame_index"]].prod(axis=1))) }


def preparar_frame(resultado, forma, frame_index, offsets, acumulador):
    h, w = forma
    cajas = resultado.boxes
    n = len(cajas) if cajas is not None else 0
    if n:
        if resultado.masks is None or len(resultado.masks.data) != n:
            raise RuntimeError("YOLO: boxes y máscaras sin correspondencia 1:1")
        mascaras = resultado.masks.data.cpu().numpy() > 0.5
        if mascaras.shape[1:] != (h, w):
            raise RuntimeError(f"Máscara {mascaras.shape} != frame {(h,w)}: verificar retina_masks")
        xyxy = cajas.xyxy.cpu().numpy().astype(np.float32)
        cls = cajas.cls.cpu().numpy().astype(np.int16)
        conf = cajas.conf.cpu().numpy().astype(np.float32)
    else:
        mascaras = np.empty((0, h, w), dtype=bool)
    conteos = Counter()
    for i in range(n):
        ys, xs = np.nonzero(mascaras[i])
        if not xs.size:
            raise RuntimeError(f"Máscara vacía: frame {frame_index}, instancia {i}")
        x, y, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        crop = mascaras[i, y:y2, x:x2]
        packed = np.packbits(crop.reshape(-1), bitorder="little")
        if not np.array_equal(np.unpackbits(packed, bitorder="little", count=crop.size).reshape(crop.shape).astype(bool), crop):
            raise RuntimeError("Roundtrip antes de guardar falló")
        clase = int(cls[i])
        if clase not in CLASES or str(resultado.names[clase]) != CLASES[clase]:
            raise RuntimeError(f"Clase inesperada {clase}")
        caja = xyxy[i]
        if not (0 <= caja[0] < caja[2] <= w and 0 <= caja[1] < caja[3] <= h):
            raise RuntimeError(f"Caja inválida: {caja}")
        for clave, valor in (("instance_frame_index", frame_index), ("instance_index_in_frame", i),
                             ("instance_class_id", clase), ("instance_class_name", CLASES[clase]),
                             ("instance_confidence", float(conf[i])), ("instance_bbox_xyxy_px", caja),
                             ("instance_bbox_xyxy_normalized", caja / [w, h, w, h]),
                             ("instance_mask_origin_xy", (x, y)), ("instance_mask_shape_hw", crop.shape),
                             ("instance_mask_area_px", int(xs.size))):
            acumulador[clave].append(valor)
        acumulador["mask_bits"].append(packed)
        offsets.append(offsets[-1] + len(packed))
        conteos[CLASES[clase]] += 1
    return conteos


def preview(frame, datos, indice, carpeta):
    imagen = frame.copy()
    offsets = datos["frame_instance_offsets"]
    paleta = {"bar": (0, 255, 255), "bench": (255, 130, 0), "person": (0, 220, 0)}
    # El preview se obtiene únicamente del archivo RAW reconstruido.
    conteos = Counter()
    for j in range(int(offsets[indice]), int(offsets[indice + 1])):
        x, y = map(int, datos["instance_mask_origin_xy"][j])
        h, w = map(int, datos["instance_mask_shape_hw"][j])
        a, b = int(datos["mask_byte_offsets"][j]), int(datos["mask_byte_offsets"][j + 1])
        crop = np.unpackbits(datos["mask_bits"][a:b], bitorder="little", count=h * w).reshape(h, w).astype(np.uint8)
        clase = str(datos["instance_class_name"][j]); conteos[clase] += 1
        color = paleta[clase]
        roi = imagen[y:y+h, x:x+w]
        pix = crop.astype(bool)
        roi[pix] = (roi[pix].astype(np.float32) * .65 + np.array(color) * .35).astype(np.uint8)
        contornos, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(roi, contornos, -1, color, 2)
        x1, y1, x2, y2 = map(int, datos["instance_bbox_xyxy_px"][j])
        cv2.rectangle(imagen, (x1, y1), (x2, y2), color, 2)
        cv2.putText(imagen, f'{clase} #{int(datos["instance_index_in_frame"][j])} {float(datos["instance_confidence"][j]):.2f}',
                    (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, .65, color, 2)
    cv2.rectangle(imagen, (0, 0), (min(imagen.shape[1], 940), 44), (0, 0, 0), -1)
    cv2.putText(imagen, f'frame {indice} | detections {sum(conteos.values())} | BAR {conteos["bar"]} PERSON {conteos["person"]} BENCH {conteos["bench"]}',
                (8, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)
    escala = min(1., 1280 / imagen.shape[1])
    if escala < 1:
        imagen = cv2.resize(imagen, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(carpeta / f"frame_{indice:05d}.jpg"), imagen)


def extraer_video(video, vista, base, modelo, modelo_info, split, device, hacer_preview):
    import ultralytics
    from uuid import uuid4
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError("Video ilegible")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    declarados = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    if not math.isfinite(fps) or fps <= 0 or declarados <= 0 or h <= 0 or w <= 0:
        cap.release(); raise RuntimeError("FPS/dimensiones/frames inválidos")
    metadata = {"extraction_id": str(uuid4()), "extractor_version": VERSION, "schema_version": ESQUEMA,
                "video_id": video.stem, "view": vista, "video_path": str(video.resolve()),
                "video_sha256": sha(video), "video_size_bytes": video.stat().st_size,
                "model_path": str(MODELO.resolve()), "model_sha256": modelo_info["model_sha256"],
                "model_task": "segment", "classes": {str(k): v for k, v in CLASES.items()},
                "inference": INFERENCIA, "ultralytics_version": ultralytics.__version__,
                "opencv_version": cv2.__version__, "python_version": platform.python_version(),
                "device": str(device), "fps": fps, "width": w, "height": h,
                "frames_video": declarados, "timestamp_policy": "original zero-based decoded frame index / CAP_PROP_FPS",
                "mask_encoding": "minimal nonzero crop, C-order, np.packbits bitorder=little, threshold >0.5",
                "physical_identity": "unverified", "multiview_pair": None,
                "posecnn_existing_split": split, "split_usage": "metadata only; no split created",
                "tracking": False, "extracted_utc": datetime.now(timezone.utc).isoformat()}
    acum = defaultdict(list); fo = [0]; mo = [0]; formas = []; conteos = Counter()
    frames = 0
    salida = base / "raw" / f"{video.stem}_segmentation.npz"
    temporal = salida.with_suffix(".npz.tmp")
    try:
        while True:
            lote = []
            for _ in range(INFERENCIA["batch"]):
                ok, frame = cap.read()
                if not ok:
                    break
                if frame.shape[:2] != (h, w):
                    raise RuntimeError("Resolución variable")
                lote.append(frame)
            if not lote:
                break
            resultados = modelo(lote, imgsz=INFERENCIA["imgsz"], conf=INFERENCIA["conf"],
                                iou=INFERENCIA["iou"], retina_masks=True, augment=False,
                                device=device, verbose=False)
            if len(resultados) != len(lote):
                raise RuntimeError("Número de resultados != frames")
            for frame, resultado in zip(lote, resultados):
                conteos.update(preparar_frame(resultado, (h, w), frames, mo, acum))
                frames += 1; formas.append((h, w)); fo.append(len(acum["instance_frame_index"]))
        if frames != declarados:
            raise RuntimeError(f"Frames decodificados {frames} != frames declarados {declarados}")
        metadata.update(frames_decoded=frames, instances_total=fo[-1])
        n = fo[-1]
        arrays = {"schema_version": np.asarray(ESQUEMA),
                  "metadata_json": np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
                  "frame_index": np.arange(frames, dtype=np.int32),
                  "timestamp_sec": np.arange(frames, dtype=np.float64) / fps,
                  "frame_image_shape_hw": np.asarray(formas, dtype=np.int32).reshape(frames, 2),
                  "frame_instance_offsets": np.asarray(fo, dtype=np.int64),
                  "frame_class_counts": np.asarray([[0, 0, 0] for _ in range(frames)], dtype=np.int32),
                  "mask_byte_offsets": np.asarray(mo, dtype=np.int64),
                  "mask_bits": np.concatenate(acum["mask_bits"]) if n else np.empty(0, dtype=np.uint8)}
        dtypes = {"instance_frame_index": np.int32, "instance_index_in_frame": np.int16,
                  "instance_class_id": np.int16, "instance_class_name": "<U32",
                  "instance_confidence": np.float32, "instance_bbox_xyxy_px": np.float32,
                  "instance_bbox_xyxy_normalized": np.float32, "instance_mask_origin_xy": np.int32,
                  "instance_mask_shape_hw": np.int32, "instance_mask_area_px": np.int32}
        for clave, tipo in dtypes.items():
            valor = np.asarray(acum[clave], dtype=tipo)
            if clave in ("instance_bbox_xyxy_px", "instance_bbox_xyxy_normalized"):
                valor = valor.reshape(n, 4)
            if clave in ("instance_mask_origin_xy", "instance_mask_shape_hw"):
                valor = valor.reshape(n, 2)
            arrays[clave] = valor
        for f in range(frames):
            for j in range(fo[f], fo[f + 1]):
                arrays["frame_class_counts"][f, arrays["instance_class_id"][j]] += 1
        with temporal.open("wb") as archivo:
            np.savez_compressed(archivo, **arrays)
        verificado = verificar_archivo(temporal, modelo_info["model_sha256"], metadata["video_sha256"])
        os.replace(temporal, salida)
        if hacer_preview:
            carpeta = base / "previews" / video.stem
            carpeta.mkdir(parents=True, exist_ok=True)
            indices = set(np.linspace(0, frames - 1, min(3, frames), dtype=int))
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for f in range(frames):
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Error leyendo preview")
                if f in indices:
                    preview(frame, arrays, f, carpeta)
        return {"video_id": video.stem, "status": "COMPLETE", "frames": frames,
                "instances": n, "raw_path": str(salida), "raw_sha256": sha(salida),
                "video_sha256": metadata["video_sha256"], "bytes": verificado["bytes"],
                "split_posecnn": split, "bar": conteos["bar"], "bench": conteos["bench"],
                "person": conteos["person"], "error": ""}
    finally:
        cap.release()
        if temporal.exists():
            temporal.unlink()


def principal():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", choices=("left", "right"), required=True)
    parser.add_argument("--phase", choices=("audit", "pilot", "all", "verify"), required=True)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    entrada, puntos, split, base = rutas(args.view)
    if args.phase == "audit":
        auditar(args.view, entrada, puntos, split, base); return
    modelo, info = inspeccionar_modelo()
    manifiestos = base / "manifests"
    auditado = {r["video_id"]: r for r in filas_csv(manifiestos / "input_audit.csv")}
    if not auditado:
        raise RuntimeError("Ejecutar --phase audit antes de extraer")
    if args.phase == "verify":
        resultados = []
        metricas = []
        for video_id in sorted(auditado):
            ruta = base / "raw" / f"{video_id}_segmentation.npz"
            try:
                item = verificar_archivo(ruta, info["model_sha256"], sha(entrada / f"{video_id}.mp4"))
                if item["frames"] != int(auditado[video_id]["frames_decoded"]):
                    raise RuntimeError("Frames RAW distintos de auditoría original")
                resultados.append({"video_id": video_id, "status": "OK", "frames": item["frames"],
                                   "instances": item["instances"], "raw_sha256": sha(ruta), "error": ""})
                metricas.append({"video_id": video_id, "split_posecnn": auditado[video_id]["split_posecnn"],
                                 "frames": item["frames"], "instances": item["instances"],
                                 "raw_bytes": item["bytes"], "packed_mask_bytes": item["packed_bytes"],
                                 "full_mask_binary_bytes": item["uncompressed_full_mask_bytes"],
                                 **{f"{k}_instances": v for k, v in item["class_instances"].items()},
                                 **{f"{k}_missing_frames": v for k, v in item["missing_frames"].items()},
                                 **{f"{k}_multiple_frames": v for k, v in item["multiple_frames"].items()}})
            except Exception as exc:
                resultados.append({"video_id": video_id, "status": "ERROR", "frames": "",
                                   "instances": "", "raw_sha256": "", "error": str(exc)})
        guardar_csv(manifiestos / "integrity.csv", resultados, list(resultados[0]))
        if metricas:
            guardar_csv(manifiestos / "quality_report.csv", metricas, list(metricas[0]))
        resumen = {"view": args.view, "videos_expected": len(auditado),
                   "videos_complete": len(metricas), "videos_error": len(resultados) - len(metricas),
                   "errors": [r for r in resultados if r["status"] == "ERROR"],
                   "frames": sum(r["frames"] for r in metricas),
                   "instances": sum(r["instances"] for r in metricas),
                   "raw_bytes": sum(r["raw_bytes"] for r in metricas),
                   "packed_mask_bytes": sum(r["packed_mask_bytes"] for r in metricas),
                   "uncompressed_full_mask_bytes": sum(r["full_mask_binary_bytes"] for r in metricas),
                   "per_class": {k: {"instances": sum(r[f"{k}_instances"] for r in metricas),
                                     "missing_frames": sum(r[f"{k}_missing_frames"] for r in metricas),
                                     "multiple_frames": sum(r[f"{k}_multiple_frames"] for r in metricas)}
                                 for k in CLASES.values()},
                   "model_sha256": info["model_sha256"], "config": INFERENCIA,
                   "physical_identity": "unverified; left/right IDs must not be paired by number"}
        guardar_json(manifiestos / "integrity_summary.json", resumen)
        print(Counter(r["status"] for r in resultados), flush=True)
        return
    if args.phase == "pilot":
        candidatos = [r for r in auditado.values() if r["status"] == "OK" and r["split_posecnn"] in ("train", "validation")]
        candidatos.sort(key=lambda r: int(r["frames_decoded"]))
        indices = np.linspace(0, len(candidatos) - 1, min(8, len(candidatos)), dtype=int)
        seleccion = [candidatos[int(i)] for i in indices]
        guardar_csv(manifiestos / "pilot_exposure.csv", seleccion, list(seleccion[0]))
    else:
        config = json.loads((manifiestos / "frozen_config.json").read_text(encoding="utf-8"))
        if config["inference"] != INFERENCIA or config["model_sha256"] != info["model_sha256"] or config["extractor_sha256"] != sha(Path(__file__)):
            raise RuntimeError("Configuración/modelo/código difieren del piloto congelado")
        seleccion = [r for r in auditado.values() if r["status"] == "OK"]
        seleccion.sort(key=lambda r: r["video_id"])
    (base / "raw").mkdir(parents=True, exist_ok=True)
    manifest = {r["video_id"]: r for r in filas_csv(manifiestos / "raw_manifest.csv")}
    for numero, registro in enumerate(seleccion, 1):
        video_id = registro["video_id"]
        ruta = base / "raw" / f"{video_id}_segmentation.npz"
        try:
            if ruta.exists():
                existente = verificar_archivo(ruta, info["model_sha256"], sha(entrada / f"{video_id}.mp4"))
                if existente["metadata"]["extractor_version"] != VERSION:
                    raise RuntimeError("Versión incompatible de extractor")
                previo = manifest.get(video_id)
                if previo is None or previo.get("raw_sha256") != sha(ruta):
                    counts = Counter(map(str, np.load(ruta, allow_pickle=False)["instance_class_name"]))
                    previo = {"video_id": video_id, "status": "COMPLETE", "frames": existente["frames"],
                              "instances": existente["instances"], "raw_path": str(ruta),
                              "raw_sha256": sha(ruta), "video_sha256": existente["metadata"]["video_sha256"],
                              "bytes": existente["bytes"], "split_posecnn": registro["split_posecnn"],
                              "bar": counts["bar"], "bench": counts["bench"], "person": counts["person"], "error": ""}
                manifest[video_id] = previo
            else:
                manifest[video_id] = extraer_video(entrada / f"{video_id}.mp4", args.view, base,
                                                    modelo, info, registro["split_posecnn"], args.device,
                                                    args.phase == "pilot")
        except Exception as exc:
            manifest[video_id] = {"video_id": video_id, "status": "ERROR", "error": repr(exc)}
            print("ERROR", video_id, repr(exc), flush=True)
        if numero % 10 == 0 or numero == len(seleccion) or args.phase == "pilot":
            guardar_csv(manifiestos / "raw_manifest.csv", sorted(manifest.values(), key=lambda r: r["video_id"]),
                        ["video_id", "status", "frames", "instances", "raw_path", "raw_sha256", "video_sha256", "bytes",
                         "split_posecnn", "bar", "bench", "person", "error"])
        print(args.view, args.phase, numero, "/", len(seleccion), video_id,
              manifest[video_id]["status"], flush=True)
    if args.phase == "pilot" and all(manifest[r["video_id"]]["status"] == "COMPLETE" for r in seleccion):
        guardar_json(manifiestos / "frozen_config.json", {"inference": INFERENCIA, "model_sha256": info["model_sha256"],
                    "extractor_sha256": sha(Path(__file__)), "model_audit": info,
                    "pilot_videos": [r["video_id"] for r in seleccion],
                    "frozen_utc": datetime.now(timezone.utc).isoformat(),
                    "retina_note": "Visor usa retina_masks=False; RAW conserva pixeles binarios de retina_masks=True. No implica mayor precision del modelo."})


if __name__ == "__main__":
    principal()
