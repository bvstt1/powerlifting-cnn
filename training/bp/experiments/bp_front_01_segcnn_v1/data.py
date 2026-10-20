"""Preparación temporal BP_FRONT: solo lee train/validation del split común.

El grid linspace(0,T-1,170) coincide con el del pipeline PoseCNN original.
No interpola NaN ni cambios de identidad; conserva máscaras y procedencia.
"""
import csv
import json
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[4]
FUENTE = RAIZ / "segmentation/bp/front/temporal_features_v1"
COHORTE = FUENTE / "bp_front_features_common_cohort.csv"
ESQUEMA = FUENTE / "bp_front_features_schema.json"
VERSION_FEATURES = "temporal_features_v1"
PASOS = 170
CALIDAD_BINARIA = {28, 33, 34, 35, 36, 37}
RELACION_EXCLUIDA = {
    "silhouette_profile_y_norm", "sep_signed_norm", "sep_coverage",
    "sep_extrap_dist_norm", "d_sep_dt", "q_projected",
    "profile_ambiguous",
}
# Cuatro indicadores binarios marcados QUALITY en el esquema son pose-assisted
# por depender de seleccion de barra / perfil con keypoints.
QUALITY_POSE_ASSISTED = (
    "q_bar_associated", "q_bar_unverified", "q_bar_multiple", "q_projected"
)
PROCEDENCIA_BAR = ("bar_observada", "bar_no_verificada", "bar_ausente")
PROCEDENCIA_SEP = ("sep_directa", "sep_proyectada")


def leer_cohorte():
    with open(COHORTE, encoding="utf-8-sig", newline="") as archivo:
        filas = list(csv.DictReader(archivo))
    ids = [f["video_id"] for f in filas]
    if len(filas) != 663 or len(ids) != len(set(ids)):
        raise ValueError("Cohorte debe contener 663 IDs únicos")
    if {f["video_id"] for f in filas if f["excluido"] == "True"} != {"bp_349", "bp_352"}:
        raise ValueError("Exclusiones no coinciden con inventario congelado")
    grupos = {s: [f for f in filas if f["common_split"] == s and f["excluido"] == "False"]
              for s in ("train", "validation", "test")}
    if [len(grupos[s]) for s in ("train", "validation", "test")] != [462, 99, 100]:
        raise ValueError("Split común cambiado")
    for f in filas:
        if f["label"] not in {"0", "1"}:
            raise ValueError(f"Etiqueta inválida {f['video_id']}")
    esquema = json.loads(ESQUEMA.read_text(encoding="utf-8"))
    if esquema.get("version") != VERSION_FEATURES or esquema["F"] != 38:
        raise ValueError("Esquema inesperado")
    return grupos, [x["nombre"] for x in esquema["features"]]


def remuestrear(secuencia):
    """Devuelve valores, disponibilidad, y 5 bits de procedencia (170 pasos).

    Un punto fraccional requiere dos frames originales consecutivos observados.
    BAR/dinámica: misma identidad de track + prov_bar=1.
    Separación: igual procedencia (directa o proyectada), misma identidad.
    Calidad binaria y procedencia: vecino más cercano, sin interpolación.
    """
    x = secuencia["features"]
    a = secuencia["availability"].astype(bool)
    bar = secuencia["prov_bar"]
    sep = secuencia["prov_sep"]
    track = secuencia["bar_track_id"]
    t, f = x.shape
    if f != 38 or a.shape != x.shape or t < 2:
        raise ValueError("Shapes originales inesperados")
    if not np.array_equal(a, np.isfinite(x)):
        raise ValueError("Disponibilidad/NaN inconsistente")
    if not np.array_equal(secuencia["frame_index"], np.arange(t)):
        raise ValueError("Índice temporal original no contiguo")
    grid = np.linspace(0, t - 1, PASOS, dtype=np.float64)
    v = np.full((PASOS, f), np.nan, dtype=np.float32)
    mask = np.zeros((PASOS, f), dtype=bool)
    origen = np.zeros((PASOS, 5), dtype=np.float32)
    # La bandera procede del instante original más cercano. Nunca se interpola.
    nearest = np.floor(grid + 0.5).astype(int)
    origen[:, 0] = bar[nearest] == 1
    origen[:, 1] = bar[nearest] == 4
    origen[:, 2] = bar[nearest] == 0
    origen[:, 3] = sep[nearest] == 1
    origen[:, 4] = sep[nearest] == 2
    for z, tiempo in enumerate(grid):
        lo, hi = int(np.floor(tiempo)), int(np.ceil(tiempo))
        if lo == hi:
            v[z] = x[lo]
            mask[z] = a[lo]
            continue
        peso = float(tiempo - lo)
        for j in range(f):
            if j in CALIDAD_BINARIA:
                v[z, j], mask[z, j] = x[nearest[z], j], a[nearest[z], j]
                continue
            if not (a[lo, j] and a[hi, j]):
                continue
            if j < 11 or j in (27, 29, 30, 31):
                if bar[lo] != 1 or bar[hi] != 1 or track[lo] != track[hi]:
                    continue
            if j in (24, 25, 26, 32):
                if (sep[lo] == 0 or sep[lo] != sep[hi]
                        or track[lo] != track[hi]):
                    continue
            v[z, j] = (1 - peso) * x[lo, j] + peso * x[hi, j]
            mask[z, j] = True
    if not np.array_equal(mask, np.isfinite(v)):
        raise ValueError("Resampling creó valores sin máscara")
    return v, mask, origen


def cargar_datos(grupos, nombres):
    """No abre secuencias Test. Devuelve arrays train/validation separados."""
    datos = {}
    for split in ("train", "validation"):
        vs, ms, ps, ys, ids = [], [], [], [], []
        for fila in grupos[split]:
            ruta = Path(fila["features_path"])
            if not ruta.is_file() or ruta.parent.resolve() != (FUENTE / "sequences").resolve():
                raise ValueError(f"Ruta de secuencia inesperada: {ruta}")
            with np.load(ruta, allow_pickle=False) as sec:
                if ("config_json" not in sec.files or
                        json.loads(str(sec["config_json"][0])).get("version") != VERSION_FEATURES):
                    raise ValueError(f"Version de features incompatible: {ruta}")
                if tuple(sec["feature_names"].tolist()) != tuple(nombres):
                    raise ValueError(f"Orden de canales alterado: {ruta}")
                if int(fila["T"]) != len(sec["features"]):
                    raise ValueError(f"T inconsistente: {ruta}")
                v, m, p = remuestrear(sec)
            vs.append(v)
            ms.append(m)
            ps.append(p)
            ys.append(int(fila["label"]))
            ids.append(fila["video_id"])
        datos[split] = (np.stack(vs), np.stack(ms), np.stack(ps),
                        np.asarray(ys, dtype=np.int64), ids)
    return datos


def canales_constantes_train(valores, mascara, nombres):
    const = []
    for j, nombre in enumerate(nombres):
        reales = valores[:, :, j][mascara[:, :, j]]
        if reales.size == 0 or (float(reales.max()) - float(reales.min())) <= 1e-7:
            const.append(nombre)
    return const


def preparar_variante(datos, nombres, variante, constantes):
    seleccionados = [i for i, n in enumerate(nombres)
                    if n not in constantes
                    and (variante == "full" or n not in RELACION_EXCLUIDA)]
    usados = [nombres[i] for i in seleccionados]
    if len(seleccionados) == 0:
        raise ValueError("Sin canales")
    # Estadísticas sobre valores observados de TRAIN remuestreado.
    xtr, atr = datos["train"][:2]
    medianas, medias, desvs = [], [], []
    for i in seleccionados:
        observados = xtr[:, :, i][atr[:, :, i]].astype(np.float64)
        med = float(np.median(observados))
        media = float(np.mean(observados))
        desv = float(np.std(observados))
        medianas.append(med)
        medias.append(media)
        desvs.append(max(desv, 1e-6))
    medianas = np.asarray(medianas, dtype=np.float32)
    medias = np.asarray(medias, dtype=np.float32)
    desvs = np.asarray(desvs, dtype=np.float32)
    resultados = {}
    for split in ("train", "validation"):
        x, m, p, y, ids = datos[split]
        xx, mm = x[:, :, seleccionados], m[:, :, seleccionados]
        imputado = np.where(mm, xx, medianas[None, None, :])
        escalado = (imputado - medias[None, None, :]) / desvs[None, None, :]
        # Quitar sep provenance en ablación para no dejarla como proxy.
        procedencia = p if variante == "full" else p[:, :, :3]
        inp = np.concatenate((escalado, mm.astype(np.float32), procedencia), axis=2)
        if not np.isfinite(inp).all():
            raise ValueError(f"No-finitos al preparar {split}")
        resultados[split] = (np.ascontiguousarray(inp.transpose(0, 2, 1)), y, ids)
    return resultados, {
        "indices": seleccionados, "nombres": usados,
        "nombres_mascara": ["available_" + n for n in usados],
        "nombres_procedencia": list(PROCEDENCIA_BAR + (PROCEDENCIA_SEP if variante == "full" else ())),
        "mediana_train": medianas.tolist(), "media_train": medias.tolist(),
        "std_train": desvs.tolist(), "constantes_train": constantes,
        "canales_total": resultados["train"][0].shape[1],
    }
