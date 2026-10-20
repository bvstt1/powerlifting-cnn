"""Lee y valida instancias RAW BP lateral V1 sin alterar archivos.

Comprueba el esquema, frames, offsets y máscaras antes de entregar la
geometría original de cada detección a las features temporales.
"""
import json
import numpy as np

CLASES = {0: "bar", 1: "bench", 2: "person"}


def cargar(ruta):
    with np.load(ruta, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    meta = json.loads(str(d["metadata_json"]))
    if str(d["schema_version"]) != "bp_side_segmentation_raw_v1":
        raise ValueError("Esquema RAW incompatible")
    t = len(d["frame_index"])
    if not np.array_equal(d["frame_index"], np.arange(t)) or not np.isfinite(meta["fps"]) or meta["fps"] <= 0:
        raise ValueError("FPS/indices invalidos")
    if not np.allclose(d["timestamp_sec"], np.arange(t)/meta["fps"], atol=1e-9, rtol=0):
        raise ValueError("Timestamps no corresponden a frame_index/fps")
    if d["frame_image_shape_hw"].shape != (t, 2) or np.any(d["frame_image_shape_hw"] <= 0):
        raise ValueError("Resolucion invalida")
    f, m = d["frame_instance_offsets"], d["mask_byte_offsets"]
    n = len(d["instance_class_id"])
    if len(f) != t + 1 or f[0] != 0 or f[-1] != n or np.any(np.diff(f) < 0):
        raise ValueError("Offsets por frame invalidos")
    if len(m) != n + 1 or m[0] != 0 or m[-1] != len(d["mask_bits"]):
        raise ValueError("Offsets mascara invalidos")
    if n and (np.any(d["instance_class_id"] < 0) or np.any(d["instance_class_id"] > 2)):
        raise ValueError("Clase RAW desconocida")
    return d, meta


def mascara(d, j):
    h, w = map(int, d["instance_mask_shape_hw"][j])
    if h <= 0 or w <= 0:
        raise ValueError("Mascara sin area")
    a, b = map(int, d["mask_byte_offsets"][j:j+2])
    if b - a != (h*w + 7)//8:
        raise ValueError("Mascara truncada")
    crop = np.unpackbits(d["mask_bits"][a:b], bitorder="little", count=h*w).reshape(h, w).astype(bool)
    if int(crop.sum()) != int(d["instance_mask_area_px"][j]):
        raise ValueError("Area de mascara inconsistente")
    return crop


def instancia(d, j, frame):
    h, w = map(int, d["frame_image_shape_hw"][frame])
    if int(d["instance_frame_index"][j]) != frame or int(d["instance_index_in_frame"][j]) != j-int(d["frame_instance_offsets"][frame]):
        raise ValueError("Indices de instancias inconsistentes")
    clase = int(d["instance_class_id"][j])
    if str(d["instance_class_name"][j]) != CLASES[clase]:
        raise ValueError("Clase diferente del checkpoint")
    x, y = map(int, d["instance_mask_origin_xy"][j])
    crop = mascara(d, j)
    if not (0 <= x < x+crop.shape[1] <= w and 0 <= y < y+crop.shape[0] <= h):
        raise ValueError("Crop fuera del frame")
    box = np.asarray(d["instance_bbox_xyxy_px"][j], dtype=float)
    if not (np.isfinite(box).all() and 0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h):
        raise ValueError("BBox invalida")
    conf = float(d["instance_confidence"][j])
    if not 0 <= conf <= 1:
        raise ValueError("Confianza fuera de rango")
    return {"id": j, "index_in_frame": j-int(d["frame_instance_offsets"][frame]),
            "class": CLASES[clase], "conf": conf, "box": box,
            "crop": crop, "x": x, "y": y, "area": int(d["instance_mask_area_px"][j])}
