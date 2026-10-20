"""Lee RAW BP front V2 y reconstruye máscaras crop sin mutarlo.

Rechaza esquemas incompatibles y entrega instancias por frame para calcular
las features temporales front.
"""

import json
from pathlib import Path
import numpy as np

SCHEMA_VERSION_ESPERADO = "bp_front_segmentation_raw_v2"
CLASES_ESPERADAS = {0: "bar", 1: "bench", 2: "person"}


def cargar_raw(ruta):
    # np.load(NPZ)[clave] descomprime de nuevo cada acceso. En secuencias largas,
    # materializar una sola vez evita miles de descompresiones del mismo RAW.
    with np.load(str(ruta), allow_pickle=False) as archivo:
        d = {clave: archivo[clave] for clave in archivo.files}
    claves = set(d)
    requeridas = {
        "schema_version", "metadata_json", "frame_index",
        "frame_image_shape_hw", "frame_instance_offsets",
        "instance_frame_index", "instance_class_id",
        "instance_class_name", "instance_confidence",
        "instance_bbox_xyxy_px", "instance_track_id",
        "instance_mask_origin_xy", "instance_mask_shape_hw",
        "instance_mask_area_px", "mask_byte_offsets", "mask_bits",
        "polygon_point_offsets", "polygon_xy_px",
    }
    faltantes = requeridas - claves
    if faltantes:
        raise ValueError(f"Claves NPZ faltantes en {ruta}: {sorted(faltantes)}")
    version = str(d["schema_version"])
    if version != SCHEMA_VERSION_ESPERADO:
        raise ValueError(f"schema_version inesperado {version} en {ruta}")
    meta = json.loads(str(d["metadata_json"]))
    return d, meta


def decodificar_mascara_crop(d, indice):
    """Devuelve mascara bool (hc, wc) del crop minimo no-nulo."""
    off = d["mask_byte_offsets"]
    bits = d["mask_bits"]
    shp = d["instance_mask_shape_hw"]
    h, w = int(shp[indice, 0]), int(shp[indice, 1])
    if h <= 0 or w <= 0:
        return np.zeros((0, 0), dtype=bool)
    packed = bits[off[indice]:off[indice + 1]]
    n = h * w
    plano = np.unpackbits(np.asarray(packed, dtype=np.uint8),
                          bitorder="little")[:n]
    if plano.size < n:
        raise ValueError(f"mascara truncada instancia {indice}")
    return plano.reshape(h, w).astype(bool)


def instancias_por_frame(d):
    """Agrupa indices de instancia por frame. No modifica el NPZ."""
    fidx = np.asarray(d["frame_index"])
    off = np.asarray(d["frame_instance_offsets"])
    n_frames = len(fidx)
    grupos = []
    for t in range(n_frames):
        a, b = int(off[t]), int(off[t + 1])
        grupos.append(list(range(a, b)))
    return fidx, grupos


def poligono_instancia(d, indice):
    poff = d["polygon_point_offsets"]
    pxy = d["polygon_xy_px"]
    a, b = int(poff[indice]), int(poff[indice + 1])
    return np.asarray(pxy[a:b], dtype=np.float32)
