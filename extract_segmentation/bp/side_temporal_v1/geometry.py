"""Geometría lateral sobre crops binarios; sin tracking ni inferencia YOLO."""
import numpy as np

CONF_POSE = 0.5
WRIST_MAX_DIAG = 0.06
WRIST_COMPETITOR_DIAG = 0.10
SLOPE_MIN_WIDTH = 30
SLOPE_MAX_ABS = 0.35
SLOPE_MAX_RESIDUAL_H = 0.02


def punto_pose(pose, indice, ancho, alto):
    x, y, c = pose[indice]
    if not np.isfinite((x, y, c)).all() or c < CONF_POSE or (x == 0 and y == 0) or not (0 <= x <= 1 and 0 <= y <= 1):
        return None
    return np.asarray((x*ancho, y*alto), dtype=float)


def pixeles(i):
    ys, xs = np.nonzero(i["crop"])
    return xs + i["x"], ys + i["y"]


def bbox_iou(a, b):
    x1, y1 = np.maximum(a[:2], b[:2]); x2, y2 = np.minimum(a[2:], b[2:])
    inter = max(0., x2-x1)*max(0., y2-y1)
    aa = max(0., a[2]-a[0])*max(0., a[3]-a[1])
    ab = max(0., b[2]-b[0])*max(0., b[3]-b[1])
    return inter/(aa+ab-inter) if aa+ab-inter else 0.


def punto_a_mascara(i, punto):
    # Distancia real al pixel segmentado más cercano, no al centro de bbox.
    xs, ys = pixeles(i)
    return float(np.sqrt(np.min((xs-punto[0])**2 + (ys-punto[1])**2)))


def elegir_barra(barras, persona, muneca, ancho, alto):
    """Devuelve (instancia o None, estado). Nunca fusiona ni asigna track ID."""
    if not barras:
        return None, 0
    diag = np.hypot(ancho, alto)
    candidatas = []
    for b in barras:
        xs, ys = pixeles(b)
        largo = int(xs.max()-xs.min())
        if largo < SLOPE_MIN_WIDTH:
            continue
        dist = punto_a_mascara(b, muneca)/diag if muneca is not None else np.inf
        cerca_mano = dist <= WRIST_MAX_DIAG
        # Respaldo estático: solape con bbox PERSON, sin afirmar identidad.
        cuerpo = persona is not None and bbox_iou(b["box"], persona["box"]) > 0
        # El disco puede ocultar el tramo entre muñeca y única manga visible.
        # Solo candidatura descriptiva: su identidad queda NO verificada.
        distancia_x = max(0., b["box"][0]-muneca[0], muneca[0]-b["box"][2]) if muneca is not None else np.inf
        manga_plausible = (muneca is not None and distancia_x <= .25*ancho and
                           abs((b["box"][1]+b["box"][3])/2-muneca[1]) <= .15*alto and
                           b["box"][3]-b["box"][1] <= .15*alto)
        if cerca_mano or cuerpo or manga_plausible:
            candidatas.append((b, dist, cerca_mano))
    if not candidatas:
        return None, 4
    cerca = [c for c in candidatas if c[2]]
    if len(cerca) == 1 and all(c is cerca[0] or c[1] > WRIST_COMPETITOR_DIAG for c in candidatas):
        return cerca[0][0], 1
    if len(candidatas) == 1:
        return candidatas[0][0], 2  # Máscara única plausible, identidad NO verificada.
    return None, 3


def perfil_persona(persona, x_min, x_max, ancho):
    x_min = max(0, int(x_min)); x_max = min(ancho-1, int(x_max))
    if x_max-x_min < 5:
        return None
    crop, x0, y0 = persona["crop"], persona["x"], persona["y"]
    xs, ys = [], []
    for x in range(x_min, x_max+1):
        xc = x-x0
        if 0 <= xc < crop.shape[1]:
            fila = np.flatnonzero(crop[:, xc])
            if fila.size:
                xs.append(x); ys.append(int(fila[0]+y0))
    cobertura = len(xs)/(x_max-x_min+1)
    if cobertura < .7:
        return None
    return {"x": float(np.median(xs)), "y": float(np.median(ys)),
            "coverage": cobertura, "points": np.column_stack((xs, ys)).astype(np.int32)}


def torso(persona, pose, banco_bbox, barra, ancho, alto, banco_ref=False):
    if persona is None:
        return None, 0
    cadera, hombro = punto_pose(pose, 2, ancho, alto), punto_pose(pose, 3, ancho, alto)
    if cadera is not None and hombro is not None:
        a = cadera[0]+.45*(hombro[0]-cadera[0])
        b = cadera[0]+.75*(hombro[0]-cadera[0])
        p = perfil_persona(persona, min(a,b), max(a,b), ancho)
        if p is not None:
            return p, 1
    if banco_bbox is None or barra is None or banco_bbox[2]-banco_bbox[0] < .1*ancho:
        return None, 0
    centro = (banco_bbox[0]+banco_bbox[2])/2
    bx, _ = pixeles(barra)
    direccion = 1 if np.median(bx) > centro else -1
    w = banco_bbox[2]-banco_bbox[0]
    a, b = centro-direccion*.10*w, centro+direccion*.15*w
    p = perfil_persona(persona, min(a,b), max(a,b), ancho)
    if p is not None:
        return p, 2
    return None, 0


def extremo_proximal(barra, x_torso):
    xs, ys = pixeles(barra)
    x_min, x_max = int(xs.min()), int(xs.max())
    franja = max(4, int((x_max-x_min)*.15))
    cerca = xs <= x_min+franja if abs(x_torso-x_min) < abs(x_torso-x_max) else xs >= x_max-franja
    x, y = float(np.median(xs[cerca])), float(np.median(ys[cerca]))
    return (x, y, float(max(0., np.percentile(ys[cerca], 85)-y)))


def pendiente(barra, alto):
    crop = barra["crop"]
    ancho = crop.shape[1]-1
    if ancho < SLOPE_MIN_WIDTH:
        return None
    centros = []
    for xc in range(int(ancho*.15), ancho-int(ancho*.15)+1):
        filas = np.flatnonzero(crop[:, xc])
        if filas.size:
            centros.append((xc+barra["x"], float(np.median(filas)+barra["y"])))
    if len(centros) < ancho*.5:
        return None
    datos = np.asarray(centros)
    a, b = np.polyfit(datos[:, 0], datos[:, 1], 1)
    residuo = float(np.median(np.abs(datos[:, 1]-(a*datos[:, 0]+b))))
    if not np.isfinite((a, b, residuo)).all() or abs(a) > SLOPE_MAX_ABS or residuo > alto*SLOPE_MAX_RESIDUAL_H:
        return None
    return float(a), float(b), residuo


def relacion(barra, perfil, ajuste, alto):
    """estado: 0 abstener, 1 observacion directa, 2 proyectada, 3 lejana."""
    if barra is None or perfil is None:
        return {"state": 0}
    x = perfil["x"]
    xs, _ = pixeles(barra)
    L = float(xs.max()-xs.min())
    if L <= 0:
        return {"state": 0}
    cols = []
    for xc in range(max(0, int(x)-2-barra["x"]), min(barra["crop"].shape[1], int(x)+3-barra["x"])):
        yy = np.flatnonzero(barra["crop"][:, xc])
        if yy.size:
            cols.append(int(yy.max()+barra["y"]))
    if cols:  # Un borde inferido en píxeles BAR reales, sin extensión espacial.
        return {"state": 1, "sep": float(perfil["y"]-np.median(cols)),
                "D": 0., "L": L, "ratio": 0., "endpoint": extremo_proximal(barra, x)}
    extremo = extremo_proximal(barra, x)
    D = abs(x-extremo[0])
    if ajuste is None or D > 4*L:
        return {"state": 0}
    a, _, _ = ajuste
    y_barra = extremo[1]+extremo[2]+a*(x-extremo[0])
    return {"state": 3 if D > 2*L else 2,
            "sep": float(perfil["y"]-y_barra), "D": D, "L": L,
            "ratio": D/L, "endpoint": extremo, "projected_y": float(y_barra)}


def continuidad(anterior, actual, anterior_estado, actual_estado, ancho, alto):
    """Gating local estricto: ambas barras únicas con muñeca corroborada."""
    if anterior_estado != 1 or actual_estado != 1 or anterior is None or actual is None:
        return False
    if bbox_iou(anterior["box"], actual["box"]) < .5:
        return False
    xs0, ys0 = pixeles(anterior); xs1, ys1 = pixeles(actual)
    if abs(np.median(xs1)-np.median(xs0)) > .025*ancho or abs(np.median(ys1)-np.median(ys0)) > .04*alto:
        return False
    if not 1/1.5 <= actual["area"]/anterior["area"] <= 1.5:
        return False
    return True
