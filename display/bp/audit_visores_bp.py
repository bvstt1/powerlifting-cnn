"""Auditoría acotada y de solo lectura de videos/modelos de BP.

Escribe exclusivamente evidencia NUEVA en display/bp/auditoria_visores_bp/.
No altera videos, pesos, keypoints, etiquetas, RAW ni resultados de CNN.
La comparación fiel al visor es:
    python display/bp/audit_visores_bp.py --phase actual-before
    python display/bp/audit_visores_bp.py --phase actual-after
Requiere que source_before/ contenga los snapshots iniciales; los artefactos
before/after SIN prefijo actual fueron un ensayo FORZADO a 640, no el visor.
"""

import argparse
import csv
import difflib
import hashlib
import inspect
import json
import platform
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import ultralytics
from ultralytics import YOLO

import yolo_seg_bp_front as front
import yolo_seg_bp_side as side


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent / "auditoria_visores_bp"
SCRIPTS = (Path("display/bp/yolo_seg_bp_side.py"),
           Path("display/bp/yolo_seg_bp_front.py"))
MODEL_NAMES = ("bp_side_seg_v1.pt", "bp_side_skeleton_v2.pt",
               "bp_front_seg_v2.pt", "bp_front_skeleton_v8.pt")
CHECK_INPUTS = [ROOT / "models" / "bp" / name for name in MODEL_NAMES]
CHECK_INPUTS += [ROOT / "etiquetado" / "bp" / f"bp_{v}.csv"
                 for v in ("front", "left", "right")]
CHECK_INPUTS += [ROOT / "segmentation" / "bp" / "front" / "raw" /
                 f"bp_{i:03d}_segmentation.npz" for i in (535, 537, 662, 663)]
CHECK_INPUTS += [ROOT / "training" / "models" / "bp" / "front" /
                 "posecnn" / "bp_front_pose_best_model.pth",
                 ROOT / "training" / "data" / "bp" / "front" /
                 "bp_front_pose_split.csv"]

CASES = {
    "left": {
        3: (0, 20, 30, 40), 190: (0, 24, 40), 345: (0, 61, 122),
        611: (0, 30, 40),
    },
    "right": {
        3: (0, 20, 30, 40), 342: (0, 18, 25, 35),
        203: (0, 35, 50), 611: (0, 32, 40),
    },
    "front": {
        1: (0,), 3: (0, 30, 45), 139: (85, 93), 200: (19,),
        267: (0, 25), 342: (0, 30), 467: (43, 44, 50),
        518: (58,), 520: (0,), 522: (25,), 535: (0, 50, 80),
        537: (0, 60, 125), 591: (0, 30),
        662: (10, 18, 19, 23, 60), 663: (10, 11, 18, 51),
    },
}
NAMES_SIDE = ("rodilla", "tobillo", "cadera", "hombro",
              "muneca", "codo", "talon", "punta")
NAMES_FRONT = ("codo L", "hombro L", "hombro R", "codo R",
               "muneca R", "muneca L")
SIDE_CONNECTIONS = ((2, 0), (0, 1), (1, 6), (6, 7),
                    (2, 3), (3, 5), (5, 4))
FRONT_CONNECTIONS = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5))
MP_IDS = {"left": (25, 27, 23, 11, 15, 13, 29, 31),
          "right": (26, 28, 24, 12, 16, 14, 30, 32)}


def sha256(path):
    digesto = hashlib.sha256()
    with path.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            digesto.update(bloque)
    return digesto.hexdigest()


def hashes_entradas():
    return {path.relative_to(ROOT).as_posix(): sha256(path) if path.is_file() else None
            for path in CHECK_INPUTS}


def mascara_instancia(resultado, indice, forma):
    mascara = np.zeros(forma, dtype=np.uint8)
    cv2.fillPoly(mascara, [resultado.masks.xy[indice].astype(np.int32)], 1)
    return mascara


def persona_indice(resultado):
    candidatos = [i for i, caja in enumerate(resultado.boxes)
                  if resultado.names[int(caja.cls.item())] == "person"]
    return max(candidatos, key=lambda i: float(resultado.boxes.conf[i])) if candidatos else None


def puntos_pose(resultado):
    if resultado.keypoints is None or len(resultado.keypoints) == 0:
        return None, None
    i = int(resultado.boxes.conf.argmax())
    xy = resultado.keypoints.xy[i].cpu().numpy()
    conf = (resultado.keypoints.conf[i].cpu().numpy()
            if resultado.keypoints.conf is not None else np.ones(len(xy)))
    return xy, conf


def dibujar_puntos(imagen, xy, conf, nombres, conexiones, vista):
    if xy is None:
        return
    for a, b in conexiones:
        if min(conf[a], conf[b]) >= 0.5 and np.any(xy[a]) and np.any(xy[b]):
            cv2.line(imagen, tuple(xy[a].astype(int)),
                     tuple(xy[b].astype(int)), (130, 255, 130), 2)
    for i, ((x, y), confianza) in enumerate(zip(xy, conf)):
        if not confianza >= 0.5 or not (x or y):
            continue
        color = ((70, 240, 60) if i == 2 else
                 (250, 120, 20) if i == 3 else
                 (0, 0, 255) if i == 5 else
                 (255, 0, 255) if i == 4 else (0, 225, 225))
        cv2.circle(imagen, (round(x), round(y)), 8, color, -1)
        cv2.putText(imagen, f"{i} {nombres[i]} {confianza:.2f}",
                    (round(x) + 10, round(y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def dibujar_panel(imagen, lineas):
    alto, ancho = imagen.shape[:2]
    alto_panel = min(alto, 18 + len(lineas) * 26)
    capa = imagen.copy()
    cv2.rectangle(capa, (0, alto - alto_panel), (ancho, alto), (0, 0, 0), -1)
    imagen[alto-alto_panel:] = cv2.addWeighted(
        imagen[alto-alto_panel:], 0.20, capa[alto-alto_panel:], 0.80, 0)
    for i, linea in enumerate(lineas):
        cv2.putText(imagen, linea, (12, alto - alto_panel + 25 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2,
                    cv2.LINE_AA)


def perfil_alternativo(mascara_persona, xy, conf):
    if mascara_persona is None or xy is None or min(conf[3], conf[5]) < 0.5:
        return None
    x_a, x_b = xy[[3, 5], 0]
    x_min = max(0, int(min(x_a + .45 * (x_b - x_a),
                           x_a + .75 * (x_b - x_a))))
    x_max = min(mascara_persona.shape[1] - 1,
                int(max(x_a + .45 * (x_b - x_a),
                        x_a + .75 * (x_b - x_a))))
    funcion = getattr(side, "perfil_silueta_torso", None) or side.perfil_pecho
    return funcion(mascara_persona, x_min, x_max)


def revisar_lateral(vista, frame, seg, pose, audit_mp):
    alto, ancho = frame.shape[:2]
    imagen = seg.plot(img=frame.copy())
    persona = side.mascara_principal(seg, "person", (alto, ancho))
    barra = side.mascara_principal(seg, "bar", (alto, ancho))
    bench = side.mejor_banco(seg)
    caja_banco = seg.boxes.xyxy[bench].cpu().numpy() if bench is not None else None
    xy, conf = puntos_pose(pose)
    funcion = getattr(side, "zona_torso_superior", None) or side.zona_pecho
    zona = funcion(pose, persona) if persona is not None else None
    fuente = "POSE" if zona is not None else "BANCO"
    if zona is None and persona is not None and barra is not None:
        funcion = getattr(side, "zona_torso_sin_pose", None) or side.zona_pecho_sin_pose
        zona = funcion(caja_banco, persona, barra)

    registro = {"zona_fuente": fuente if zona else "NO_EVALUABLE"}
    if zona is not None:
        x, y, perfil = zona
        registro.update(x_contorno=x, y_contorno=y,
                        rango_x=[perfil[0][0], perfil[-1][0]])
        banda = imagen.copy()
        cv2.rectangle(banda, (perfil[0][0], 0),
                      (perfil[-1][0], alto - 1), (0, 180, 255), -1)
        imagen[:] = cv2.addWeighted(imagen, .90, banda, .10, 0)
        cv2.polylines(imagen, [np.asarray(perfil, dtype=np.int32)],
                      False, (0, 255, 255), 4)
        cv2.circle(imagen, (x, y), 9, (255, 0, 0), -1)
        if barra is not None:
            extremo = side.borde_barra_cercano(barra, x)
            ajuste = side.inclinacion_barra(barra)
            if extremo and ajuste:
                xb, yb, radio = extremo
                pendiente, longitud = ajuste
                distancia = abs(x - xb)
                gap = y - (yb + radio + pendiente * (x - xb))
                registro.update(pendiente=pendiente, d_sobre_l=distancia / longitud,
                                separacion_px=gap, y_barra_prox=yb + radio)
        alterna = perfil_alternativo(persona, xy, conf)
        if alterna:
            cv2.polylines(imagen, [np.asarray(alterna[2], dtype=np.int32)],
                          False, (255, 0, 255), 2)
            cv2.circle(imagen, alterna[:2], 7, (255, 0, 255), -1)
            registro.update(x_alternativo=alterna[0], y_alternativo=alterna[1],
                            distancia_roi_px=abs(y - alterna[1]))
    estado = side.dibujar_proximidad(imagen, seg, pose, caja_banco)
    registro["estado"] = estado
    dibujar_puntos(imagen, xy, conf, NAMES_SIDE, SIDE_CONNECTIONS, vista)

    # Superposición comparativa independiente; MediaPipe puede fallar.
    if audit_mp is not None:
        for i in (2, 3, 4, 5):
            x, y, vis = audit_mp[i]
            if np.isfinite([x, y, vis]).all() and vis >= .5:
                cv2.circle(imagen, (round(x * ancho), round(y * alto)),
                           9, (255, 255, 255), 2)
                cv2.putText(imagen, f"MP {i}",
                            (round(x * ancho) + 8, round(y * alto) + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255),
                            2, cv2.LINE_AA)
        if xy is not None:
            registro["mp_vs_yolo_px"] = {
                str(i): float(np.linalg.norm(xy[i] - audit_mp[i, :2] *
                                             np.array([ancho, alto])))
                if np.isfinite(audit_mp[i, :2]).all() else None
                for i in (2, 3, 4, 5)
            }
    dibujar_panel(imagen, [f"SIDE {vista} | indice 2=CADERA 3=HOMBRO 5=CODO 4=MUNECA",
                           f"ROI actual (amarillo) vs hombro-codo (magenta) | {fuente}",
                           f"delta={registro.get('separacion_px', 'ND')} px  "
                           f"D/L={registro.get('d_sobre_l', 'ND')}  |  MP=blanco"])
    registro["keypoints_xy_px"] = xy.round(2).tolist() if xy is not None else None
    registro["keypoints_conf"] = conf.round(3).tolist() if conf is not None else None
    return imagen, registro


def barras_candidatas(seg, pose):
    persona = persona_indice(seg)
    pbox = (seg.boxes.xyxy[persona].cpu().numpy() if persona is not None else None)
    munecas = front.intervalo_munecas(pose)
    lista = []
    for i, caja in enumerate(seg.boxes):
        if seg.names[int(caja.cls.item())] != "bar":
            continue
        bbox = caja.xyxy[0].cpu().numpy()
        ys, xs = np.nonzero(mascara_instancia(seg, i, seg.orig_shape))
        if not xs.size:
            continue
        dentro = (float(np.mean((xs >= pbox[0]) & (xs <= pbox[2]) &
                                  (ys >= pbox[1]) & (ys <= pbox[3])))
                  if pbox is not None else 0.0)
        con_munecas = (float(np.mean((xs >= munecas[0]) & (xs <= munecas[1])))
                       if munecas is not None else 0.0)
        columnas = (int(np.unique(xs[(xs >= munecas[0]) &
                                     (xs <= munecas[1])]).size)
                    if munecas is not None else 0)
        puntos, _ = puntos_pose(pose)
        distancia_vertical = (float((bbox[1] + bbox[3]) / 2 -
                                   np.mean(puntos[[4, 5], 1]))
                              if puntos is not None else None)
        lista.append({"indice": i, "confianza": float(caja.conf.item()),
                      "bbox": bbox.round(1).tolist(),
                      "fraccion_persona": dentro,
                      "fraccion_munecas": con_munecas,
                      "columnas_munecas": columnas,
                      "ancho_munecas": float(munecas[1] - munecas[0])
                      if munecas is not None else None,
                      "dy_barra_munecas": distancia_vertical})
    return lista


def revisar_frontal(frame, seg, pose):
    candidatos = barras_candidatas(seg, pose)
    mejor = front.mejor_banco(seg)
    caja_banco = seg.boxes.xyxy[mejor].cpu().numpy() if mejor is not None else None
    front.seleccionar_banco_frame(seg, None, False)
    retenidas = front.seleccion_barras(seg, pose)
    imagen = seg.plot(img=frame.copy())
    extras = ({"barras_raw": len(candidatos)}
              if "barras_raw" in inspect.signature(front.dibujar_relacion_frontal).parameters
              else {})
    estado = front.dibujar_relacion_frontal(
        imagen, seg, pose, caja_banco, numero_barras=retenidas, **extras)
    seleccionadas = [caja.xyxy[0].cpu().numpy() for caja in seg.boxes
                    if seg.names[int(caja.cls.item())] == "bar"]
    for candidata in candidatos:
        bbox = np.asarray(candidata["bbox"])
        guardada = any(np.allclose(bbox, b, atol=1) for b in seleccionadas)
        candidata["seleccionada"] = guardada
        cv2.rectangle(imagen, tuple(bbox[:2].astype(int)),
                      tuple(bbox[2:].astype(int)),
                      (0, 220, 0) if guardada else (0, 0, 255), 3)
    xy, conf = puntos_pose(pose)
    dibujar_puntos(imagen, xy, conf, NAMES_FRONT, FRONT_CONNECTIONS, "front")
    dibujar_panel(imagen, [f"FRONT | BAR crudas={len(candidatos)} retenidas={retenidas}",
                           "Rectangulo verde=retenido  rojo=descartado",
                           "Distancia a silueta CENTRAL: NO contacto anatomico"])
    registro = {"estado": estado, "barras_raw": len(candidatos),
                "barras_seleccionadas": retenidas, "candidatos": candidatos,
                "keypoints_xy_px": xy.round(2).tolist() if xy is not None else None,
                "keypoints_conf": conf.round(3).tolist() if conf is not None else None}
    mascara_persona = front.mascara_principal(seg, "person", frame.shape[:2])
    mascara_barra = front.mascara_barras(seg, frame.shape[:2])
    if mascara_persona is not None and mascara_barra is not None:
        zona = front.zona_central(pose, mascara_persona, caja_banco)
        if zona is not None:
            (x, y, perfil), fuente, ambigua = zona
            registro.update(zona_fuente=fuente, x_contorno=x, y_contorno=y,
                            contorno_ambiguo=bool(ambigua))
            medida = front.borde_inferior_barra(mascara_barra, perfil)
            if medida is not None:
                registro.update(separacion_px=medida[0],
                                barra_proyectada=bool(medida[2]))
    return imagen, registro


def mp_guardado(vista, numero, frame):
    ruta = ROOT / "keypoints" / "bp" / "mediapipe" / vista / f"bp_{numero:03d}.npy"
    if not ruta.is_file():
        return None
    puntos = np.load(ruta, allow_pickle=False)
    return puntos[frame] if frame < len(puntos) else None


def cargar_snapshot(nombre):
    ruta = OUTPUT / "source_before" / (nombre + ".txt")
    namespace = {"__name__": "snapshot_" + nombre, "__file__": str(ruta)}
    exec(compile(ruta.read_text(encoding="utf-8"), str(ruta), "exec"), namespace)
    return types.SimpleNamespace(**namespace)


def generar_diff_visores():
    """Diff de aplicación con cero líneas de contexto para evitar espacios falsos."""
    snapshot = OUTPUT / "source_before"
    parches = []
    for ruta in SCRIPTS:
        antes = (snapshot / (ruta.name + ".txt")).read_text(encoding="utf-8").splitlines(True)
        despues = (ROOT / ruta).read_text(encoding="utf-8").splitlines(True)
        parches.extend(difflib.unified_diff(
            antes, despues, fromfile=f"{ruta.as_posix()} (before)",
            tofile=f"{ruta.as_posix()} (after)", n=0))
    destino = OUTPUT / "code_diff.patch"
    destino.write_text("".join(parches), encoding="utf-8")
    print(f"Diff de visores auditados: {destino}")


def procesar(fase):
    global side, front
    actual = fase.startswith("actual-")
    previo = fase.endswith("before")
    destino = OUTPUT / fase
    snapshot = OUTPUT / "source_before"
    if fase == "before" and snapshot.is_dir() and any(snapshot.iterdir()):
        raise FileExistsError("El snapshot BEFORE ya existe: no sobrescribir fuentes iniciales")
    destino.mkdir(parents=True, exist_ok=True)
    if previo and (not snapshot.is_dir() or not any(snapshot.iterdir())):
        snapshot.mkdir(parents=True, exist_ok=True)
        for ruta in SCRIPTS:
            (snapshot / (ruta.name + ".txt")).write_bytes((ROOT / ruta).read_bytes())
    elif not snapshot.is_dir():
        raise RuntimeError("Faltan snapshots BEFORE; primero ejecutar --phase before")
    if actual and previo:
        side = cargar_snapshot("yolo_seg_bp_side.py")
        front = cargar_snapshot("yolo_seg_bp_front.py")

    hashes = hashes_entradas()
    (OUTPUT / f"input_hashes_{fase}.json").write_text(
        json.dumps(hashes, indent=2), encoding="utf-8")
    if not previo:
        fase_previa = "actual-before" if actual else "before"
        iniciales = json.loads((OUTPUT / f"input_hashes_{fase_previa}.json").read_text(encoding="utf-8"))
        if iniciales != hashes:
            raise RuntimeError("Un archivo de entrada protegido cambio desde BEFORE")
        generar_diff_visores()

    modelos_seg = {}
    modelos_pose = {}
    filas = []
    lado_imgsz = 960 if actual else getattr(side, "IMG_SIZE", 640)
    seg_frontal_imgsz = 960 if actual else 640
    parametros = {"fase": fase, "python": platform.python_version(),
                  "opencv": cv2.__version__, "ultralytics": ultralytics.__version__,
                  "segmentacion": {"imgsz_front": seg_frontal_imgsz, "imgsz_side": lado_imgsz,
                                    "conf": .25, "iou": .7,
                                    "retina_masks": False, "vid_stride": 1},
                  "pose": {"imgsz_front": 960, "imgsz_side": lado_imgsz,
                           "conf": .25, "iou": .7},
                  "modelos_sha256": {name: hashes[f"models/bp/{name}"]
                                     for name in MODEL_NAMES},
                  "fuente_original_data_yaml_lateral": "No presente en workspace; checkpoint apunta a /content/bp_side-2/data.yaml",
                  "fuente_original_documentacion": str((ROOT / "Documentacion_BP_Visores_Segmentacion_y_CNN.md").exists()),
                  "raw_front_preexistente": "imgsz960/retina_masks=True: mascara RAW difiere del visor retina_masks=False",
                  "interpretacion": ("BASELINE local a imgsz960: snapshot source_before; no recrea retrospectivamente una fuente perdida"
                                     if actual else "EXPERIMENTO FORZADO imgsz640: NO es el default del visor")}
    for vista, videos in CASES.items():
        ejercicio = "front" if vista == "front" else "side"
        if ejercicio not in modelos_seg:
            modelos_seg[ejercicio] = YOLO(str(ROOT / "models" / "bp" /
                                               ("bp_front_seg_v2.pt" if vista == "front" else "bp_side_seg_v1.pt")))
            modelos_pose[ejercicio] = YOLO(str(ROOT / "models" / "bp" /
                                                ("bp_front_skeleton_v8.pt" if vista == "front" else "bp_side_skeleton_v2.pt")))
        for numero, frames in videos.items():
            video = ROOT / "dataset" / "bp" / vista / f"bp_{numero:03d}.mp4"
            if not video.is_file():
                raise FileNotFoundError(video)
            captura = cv2.VideoCapture(str(video))
            if not captura.isOpened():
                raise RuntimeError(f"Video no se abre: {video}")
            for frame_indice in frames:
                captura.set(cv2.CAP_PROP_POS_FRAMES, frame_indice)
                valido, frame = captura.read()
                if not valido:
                    raise RuntimeError(f"Frame {frame_indice} no decodificable: {video}")
                pred_seg = modelos_seg[ejercicio](
                    frame, imgsz=seg_frontal_imgsz if vista == "front" else lado_imgsz,
                    conf=.25, iou=.7,
                    retina_masks=False, verbose=False, show=False, save=False)[0]
                pred_pose = modelos_pose[ejercicio](
                    frame, imgsz=960 if vista == "front" else lado_imgsz,
                    conf=.25, iou=.7, verbose=False, show=False, save=False)[0]
                if vista == "front":
                    anotada, registro = revisar_frontal(frame, pred_seg, pred_pose)
                else:
                    anotada, registro = revisar_lateral(
                        vista, frame, pred_seg, pred_pose,
                        mp_guardado(vista, numero, frame_indice))
                clave = f"{vista}_bp_{numero:03d}_f{frame_indice:04d}"
                ruta = destino / (clave + ".jpg")
                if not cv2.imwrite(str(ruta), anotada):
                    raise RuntimeError(f"No se pudo escribir {ruta}")
                if previo:
                    original = OUTPUT / "original"
                    original.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(original / (clave + ".jpg")), frame):
                        raise RuntimeError(f"No se pudo escribir {clave} original")
                else:
                    fase_previa = "actual-before" if actual else "before"
                    antes = cv2.imread(str(OUTPUT / fase_previa / (clave + ".jpg")))
                    if antes is None:
                        raise RuntimeError(f"Preview BEFORE ausente: {clave}")
                    origen = cv2.imread(str(OUTPUT / "original" / (clave + ".jpg")))
                    escenas = [origen, antes, anotada]
                    escenas = [cv2.resize(img, (960, 540)) for img in escenas]
                    panel = np.hstack(escenas)
                    for i, texto in enumerate(("ORIGINAL", "ANTES", "DESPUES")):
                        cv2.rectangle(panel, (i * 960, 0), (i * 960 + 230, 42), (0, 0, 0), -1)
                        cv2.putText(panel, texto, (i * 960 + 8, 28),
                                    cv2.FONT_HERSHEY_SIMPLEX, .75, (255, 255, 255), 2)
                    comparativas = OUTPUT / ("comparaciones_actual" if actual else "comparaciones")
                    comparativas.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(comparativas / (clave + ".jpg")), panel):
                        raise RuntimeError(f"No se pudo escribir comparativa {clave}")
                filas.append({"video_id": f"bp_{numero:03d}", "vista": vista,
                              "frame": frame_indice, "preview": str(ruta.relative_to(ROOT)),
                              "raw_seg_model": str(pred_seg.names),
                              **registro})
            captura.release()

    with (OUTPUT / f"frames_{fase}.csv").open("w", encoding="utf-8", newline="") as archivo:
        writer = csv.DictWriter(archivo, fieldnames=("video_id", "vista", "frame", "preview",
                                                     "estado", "zona_fuente", "x_contorno", "y_contorno",
                                                     "pendiente", "d_sobre_l", "separacion_px",
                                                     "x_alternativo", "y_alternativo", "distancia_roi_px",
                                                     "barras_raw", "barras_seleccionadas", "candidatos",
                                                     "contorno_ambiguo", "barra_proyectada",
                                                     "keypoints_xy_px", "keypoints_conf", "mp_vs_yolo_px",
                                                     "rango_x", "y_barra_prox", "raw_seg_model"),
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                          for k, v in fila.items()} for fila in filas)
    (OUTPUT / f"parametros_{fase}.json").write_text(json.dumps(parametros, indent=2, ensure_ascii=False),
                                                     encoding="utf-8")
    print(f"{fase.upper()}: {len(filas)} frames acotados auditados en {destino}")


def auditar_raw960():
    """Controles RAW congelados, sin inferencia ni cambio de fuente de datos."""
    sys.path.insert(0, str(ROOT / "extract_segmentation" / "bp"))
    from segmentation_utils import reconstruct_mask

    destino = OUTPUT / "raw960"
    destino.mkdir(parents=True, exist_ok=True)
    filas = []
    casos = {535: (50,), 537: (60, 125), 662: (18, 19),
             663: (10, 11, 18, 51)}
    for numero, indices in casos.items():
        video_id = f"bp_{numero:03d}"
        puntos = np.load(ROOT / "keypoints" / "bp" / "yolo" / "front" /
                         (video_id + ".npy"), allow_pickle=False)
        ruta_raw = (ROOT / "segmentation" / "bp" / "front" / "raw" /
                    (video_id + "_segmentation.npz"))
        cap = cv2.VideoCapture(str(ROOT / "dataset" / "bp" / "front" /
                                   (video_id + ".mp4")))
        with np.load(ruta_raw, allow_pickle=False) as raw:
            if len(puntos) != len(raw["frame_index"]):
                raise ValueError(f"T Pose y RAW incompatibles: {video_id}")
            for n in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, n)
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(f"Frame RAW no decodificable: {video_id} {n}")
                alto, ancho = frame.shape[:2]
                offs = raw["frame_instance_offsets"]
                instancias = range(int(offs[n]), int(offs[n + 1]))
                personas = [i for i in instancias
                            if int(raw["instance_class_id"][i]) == 2]
                mejor = (max(personas, key=lambda i: float(raw["instance_confidence"][i]))
                         if personas else None)
                caja_persona = (raw["instance_bbox_xyxy_px"][mejor]
                                 if mejor is not None else None)
                mu = puntos[n, [5, 4]]
                validas = (np.isfinite(mu).all() and np.min(mu[:, 2]) >= .5
                           and not np.any(np.all(mu[:, :2] == 0, axis=1)))
                px_mu = np.sort(mu[:, 0] * ancho) if validas else None
                altura_mu = np.mean(mu[:, 1] * alto) if validas else None
                lienzo = frame.copy()
                barras = [i for i in instancias
                          if int(raw["instance_class_id"][i]) == 0]
                for i in barras:
                    mascara = reconstruct_mask(raw, i)
                    ys, xs = np.nonzero(mascara)
                    if not xs.size:
                        continue
                    fr_persona = (float(np.mean((xs >= caja_persona[0]) &
                                               (xs <= caja_persona[2]) &
                                               (ys >= caja_persona[1]) &
                                               (ys <= caja_persona[3])))
                                  if caja_persona is not None else 0)
                    fr_munecas = (float(np.mean((xs >= px_mu[0]) & (xs <= px_mu[1])))
                                  if px_mu is not None else 0)
                    antes = fr_persona >= .5 or fr_munecas >= .5
                    dy = float(np.median(ys) - altura_mu) if validas else None
                    despues = antes and (dy is None or dy <= .03 * alto)
                    bbox = raw["instance_bbox_xyxy_px"][i].astype(int)
                    color = ((20, 210, 20) if despues else
                             (0, 0, 255) if antes else (0, 165, 255))
                    capa = lienzo.copy()
                    capa[mascara] = color
                    lienzo[:] = cv2.addWeighted(lienzo, .70, capa, .30, 0)
                    cv2.rectangle(lienzo, tuple(bbox[:2]), tuple(bbox[2:]), color, 3)
                    cv2.putText(lienzo, f"RAW i={i} conf={raw['instance_confidence'][i]:.2f} "
                                f"dy={dy:+.0f}px" if dy is not None else
                                f"RAW i={i} sin munecas", (max(5, bbox[0]), max(30, bbox[1] - 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, .67, color, 2)
                    filas.append({"video_id": video_id, "frame": n, "instance_index": i,
                                 "confidence": float(raw["instance_confidence"][i]),
                                 "bbox": json.dumps(bbox.tolist()),
                                 "fraccion_persona": fr_persona,
                                 "fraccion_munecas": fr_munecas,
                                 "dy_barra_munecas_px": dy,
                                 "filtro_anterior": int(antes), "guardia_propuesta": int(despues),
                                 "fuente": str(ruta_raw.relative_to(ROOT))})
                xy = puntos[n, :, :2] * np.array([ancho, alto])
                dibujar_puntos(lienzo, xy, puntos[n, :, 2], NAMES_FRONT,
                               FRONT_CONNECTIONS, "front")
                dibujar_panel(lienzo, [f"RAW 960 {video_id} frame {n} | mascara EXISTENTE",
                                       "Verde=asociada  Rojo=descartada por debajo de manos",
                                       "Naranja=descarte previo / identidad sin verificar"])
                ruta = destino / f"{video_id}_f{n:04d}.jpg"
                if not cv2.imwrite(str(ruta), lienzo):
                    raise OSError(f"No se pudo guardar {ruta}")
        cap.release()
    with (OUTPUT / "raw960_frames.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=filas[0].keys())
        writer.writeheader()
        writer.writerows(filas)
    print(f"RAW960: {len(filas)} instancias en {len(casos)} videos acotados")


def auditar_mediapipe_derecha():
    """Segundo estimador anatómico, independiente de las etiquetas YOLO."""
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    destino = OUTPUT / "mp_right"
    destino.mkdir(parents=True, exist_ok=True)
    opciones = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(ROOT / "models" /
                                                     "common" / "pose_landmarker_heavy.task")),
        running_mode=vision.RunningMode.IMAGE, num_poses=1,
    )
    modelo = YOLO(str(ROOT / "models" / "bp" / "bp_side_skeleton_v2.pt"))
    casos = {3: (0, 30), 342: (0, 18), 611: (0,), 203: (0, 35)}
    filas = []
    with vision.PoseLandmarker.create_from_options(opciones) as detector:
        for numero, frames in casos.items():
            cap = cv2.VideoCapture(str(ROOT / "dataset" / "bp" / "right" /
                                       f"bp_{numero:03d}.mp4"))
            for n in frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, n)
                valido, frame = cap.read()
                if not valido:
                    raise ValueError(f"Frame no decodificable: right bp_{numero:03d} {n}")
                h, w = frame.shape[:2]
                img_mp = mp.Image(image_format=mp.ImageFormat.SRGB,
                                  data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                res = detector.detect(img_mp)
                res_yolo = modelo(frame, imgsz=960, conf=.25, iou=.7,
                                  verbose=False, show=False, save=False)[0]
                xy, conf = puntos_pose(res_yolo)
                if xy is None:
                    continue
                panel = frame.copy()
                dibujar_puntos(panel, xy, conf, NAMES_SIDE, SIDE_CONNECTIONS, "right")
                mp_landmarks = res.pose_landmarks[0] if res.pose_landmarks else None
                for nombre, indice_yolo, candidatos in (
                    ("cadera", 2, (23, 24)), ("hombro", 3, (11, 12)),
                    ("codo", 5, (13, 14)), ("muneca", 4, (15, 16)),
                ):
                    for idx_mp in candidatos:
                        if mp_landmarks is None:
                            continue
                        kp = mp_landmarks[idx_mp]
                        punto = (round(kp.x * w), round(kp.y * h))
                        if kp.visibility >= .5:
                            cv2.circle(panel, punto, 8, (255, 255, 255), 2)
                            cv2.putText(panel, f"MP {idx_mp}", (punto[0]+8, punto[1]+12),
                                        cv2.FONT_HERSHEY_SIMPLEX, .46, (255, 255, 255), 1)
                        filas.append({"video_id": f"bp_{numero:03d}", "frame": n,
                                     "keypoint_yolo": indice_yolo, "hipotesis": nombre,
                                     "keypoint_mp": idx_mp, "visibilidad_mp": kp.visibility,
                                     "distancia_px": float(np.linalg.norm(
                                         xy[indice_yolo] - np.asarray(punto))),
                                     "pose_yolo_conf": float(conf[indice_yolo])})
                dibujar_panel(panel, [f"RIGHT bp_{numero:03d} frame {n}: YOLO ind/nombre/color",
                                      "MediaPipe blanco = triangulos VISIBLES, no ground truth",
                                      "La semantica MediaPipe puede fallar por oclusion o por lado"])
                ruta = destino / f"bp_{numero:03d}_f{n:04d}.jpg"
                if not cv2.imwrite(str(ruta), panel):
                    raise OSError(f"No se pudo escribir {ruta}")
            cap.release()
    with (OUTPUT / "mp_right_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=filas[0].keys())
        writer.writeheader()
        writer.writerows(filas)
    print(f"MediaPipe RIGHT: {len(casos)} videos, {len(filas)} comparaciones por landmark")


def verificar_regresiones():
    """Contrasta selector congelado anterior y vigente sin tocar los inputs."""
    base = OUTPUT / "source_before" / "yolo_seg_bp_front.py.txt"
    if not base.is_file():
        raise FileNotFoundError("Falta snapshot inicial para comparación")
    original = {"__name__": "audit_front_original", "__file__": str(base)}
    exec(compile(base.read_text(encoding="utf-8"), str(base), "exec"), original)

    seg = YOLO(str(ROOT / "models" / "bp" / "bp_front_seg_v2.pt"))
    pose = YOLO(str(ROOT / "models" / "bp" / "bp_front_skeleton_v8.pt"))
    filas = []
    for vid in (535, 537):
        cap = cv2.VideoCapture(str(ROOT / "dataset" / "bp" / "front" /
                                   f"bp_{vid:03d}.mp4"))
        for n in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            disponible, frame = cap.read()
            if not disponible:
                raise RuntimeError(f"Frame no decodificable: {vid} {n}")
            raw = seg(frame, imgsz=960, conf=.25, iou=.7,
                      retina_masks=False, verbose=False)[0]
            res_pose = pose(frame, imgsz=960, conf=.25, iou=.7,
                            verbose=False)[0]
            crudas = sum(raw.names[int(caja.cls.item())] == "bar"
                          for caja in raw.boxes)
            previo = original["seleccion_barras"](raw.cpu(), res_pose)
            posterior = front.seleccion_barras(raw, res_pose)
            filas.append((vid, n, crudas, previo, posterior))
        cap.release()
    output = OUTPUT / "verificacion_actual_535_537_frames.csv"
    with output.open("w", newline="", encoding="utf-8") as archivo:
        writer = csv.writer(archivo)
        writer.writerow(("video_id", "frame", "barras_detector", "antes", "despues"))
        writer.writerows((f"bp_{vid:03d}", n, raw, pre, post)
                         for vid, n, raw, pre, post in filas)
    resumen = {}
    for vid in (535, 537):
        subset = [(raw, pre, post) for id_, _, raw, pre, post in filas if id_ == vid]
        resumen[f"bp_{vid:03d}"] = {
            "frames": len(subset), "frames_con_BAR_detector": sum(raw > 0 for raw, _, _ in subset),
            "frames_asociados_antes": sum(pre > 0 for _, pre, _ in subset),
            "frames_asociados_despues": sum(post > 0 for _, _, post in subset),
            "cambios_de_asociacion": sum(pre != post for _, pre, post in subset),
        }

    # Proyección espejo: invertir x cambia el signo de la pendiente, no
    # la separación vertical si tanto barra como región se reflejan.
    mask = np.zeros((180, 260), dtype=np.uint8)
    cv2.line(mask, (40, 55), (150, 82), 1, 12)
    pendiente_a, longitud_a = side.inclinacion_barra(mask)
    pendiente_b, longitud_b = side.inclinacion_barra(np.fliplr(mask))
    assert pendiente_a > 0 and pendiente_b < 0
    assert np.isclose(pendiente_a, -pendiente_b, atol=.01)
    assert longitud_a == longitud_b
    resumen["espejo_pendiente"] = {"original": pendiente_a,
                                  "reflejada": pendiente_b,
                                  "ancho_px": longitud_a}

    anterior_side_file = OUTPUT / "source_before" / "yolo_seg_bp_side.py.txt"
    original_side = {"__name__": "audit_side_original",
                     "__file__": str(anterior_side_file)}
    exec(compile(anterior_side_file.read_text(encoding="utf-8"),
                 str(anterior_side_file), "exec"), original_side)
    seg_side = YOLO(str(ROOT / "models" / "bp" / "bp_side_seg_v1.pt"))
    pose_side = YOLO(str(ROOT / "models" / "bp" / "bp_side_skeleton_v2.pt"))
    igualdad = []
    for vista, video, n in (("left", 3, 30), ("left", 345, 61),
                             ("right", 3, 30), ("right", 342, 0),
                             ("right", 611, 32)):
        cap = cv2.VideoCapture(str(ROOT / "dataset" / "bp" / vista /
                                   f"bp_{video:03d}.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"No se pudo abrir control {vista} bp_{video:03d} f{n}")
        resultado = seg_side(frame, imgsz=960, conf=.25, iou=.7,
                             verbose=False)[0]
        pose = pose_side(frame, imgsz=960, conf=.25, iou=.7,
                         verbose=False)[0]
        mask = side.mascara_principal(resultado, "person", frame.shape[:2])
        previo = original_side["zona_pecho"](pose, mask)
        actual = side.zona_torso_superior(pose, mask)
        assert previo is not None and actual is not None
        assert previo == actual, (vista, video, n, previo[:2], actual[:2])
        igualdad.append((vista, f"bp_{video:03d}", n, actual[:2]))
    resumen["ROI_misma_inferencia_960_igual_pre_post"] = igualdad

    raw_rows = list(csv.DictReader((OUTPUT / "raw960_frames.csv").open(encoding="utf-8")))
    for vid, n, expected in (("bp_662", 18, [1, 0]),
                             ("bp_662", 19, [0]),
                             ("bp_663", 11, [0]),
                             ("bp_663", 18, [1])):
        real = [int(row["guardia_propuesta"]) for row in raw_rows
                if row["video_id"] == vid and int(row["frame"]) == n]
        assert real == expected, (vid, n, real, expected)
    resumen["raw960_662_663"] = "Control histórico: superior conservada; estructuras inferiores descartadas"

    antes = list(csv.DictReader((OUTPUT / "frames_actual-before.csv").open(encoding="utf-8")))
    despues = list(csv.DictReader((OUTPUT / "frames_actual-after.csv").open(encoding="utf-8")))
    side_metrics = {}
    for vista in ("left", "right"):
        x = [r for r in antes if r["vista"] == vista]
        y = [r for r in despues if r["vista"] == vista]
        side_metrics[vista] = {"frames": len(x),
                               "separacion_disponible_antes": sum(bool(r["separacion_px"]) for r in x),
                               "separacion_disponible_despues": sum(bool(r["separacion_px"]) for r in y),
                               "pose_ROI_antes": sum(r["zona_fuente"] == "POSE" for r in x),
                               "pose_ROI_despues": sum(r["zona_fuente"] == "POSE" for r in y)}
    resumen["vistas_laterales_muestra"] = side_metrics
    (OUTPUT / "regresiones_actual.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resumen, indent=2, ensure_ascii=False))


def comparar_frames(actual=False):
    """Tabla unificada BEFORE/AFTER con origen y limitaciones por frame."""
    def cargar(fase):
        with (OUTPUT / f"frames_{fase}.csv").open(encoding="utf-8", newline="") as f:
            return {(r["vista"], r["video_id"], int(r["frame"])): r
                    for r in csv.DictReader(f)}

    antes, despues = cargar("actual-before" if actual else "before"), cargar("actual-after" if actual else "after")
    if antes.keys() != despues.keys():
        raise ValueError("Las muestras BEFORE y AFTER no coinciden")
    ruta = OUTPUT / ("comparacion_frames_actual.csv" if actual else "comparacion_frames.csv")
    with ruta.open("w", encoding="utf-8", newline="") as f:
        campos = ("vista", "video_id", "frame", "fuente_roi_antes", "fuente_roi_despues",
                  "delta_antes_px", "delta_despues_px", "delta_cambio_px",
                  "n_bar_raw_antes", "n_bar_raw_despues",
                  "n_bar_filtrada_antes", "n_bar_filtrada_despues",
                  "estado_antes", "estado_despues", "preview_comparativo")
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        for vista, video_id, frame in sorted(antes):
            a, b = antes[(vista, video_id, frame)], despues[(vista, video_id, frame)]
            da, db = a.get("separacion_px", ""), b.get("separacion_px", "")
            clave = f"{vista}_{video_id}_f{frame:04d}.jpg"
            writer.writerow({
                "vista": vista, "video_id": video_id, "frame": frame,
                "fuente_roi_antes": a.get("zona_fuente", ""),
                "fuente_roi_despues": b.get("zona_fuente", ""),
                "delta_antes_px": da, "delta_despues_px": db,
                "delta_cambio_px": float(db) - float(da) if da and db else "",
                "n_bar_raw_antes": a.get("barras_raw", ""),
                "n_bar_raw_despues": b.get("barras_raw", ""),
                "n_bar_filtrada_antes": a.get("barras_seleccionadas", ""),
                "n_bar_filtrada_despues": b.get("barras_seleccionadas", ""),
                "estado_antes": a["estado"], "estado_despues": b["estado"],
                "preview_comparativo": (Path("display/bp/auditoria_visores_bp") /
                                        ("comparaciones_actual" if actual else "comparaciones") / clave).as_posix(),
            })
    print(f"Comparación frame/video: {len(antes)} filas -> {ruta}")


def auditar_temporal():
    """Previews de barras estimadas SOLO con las reglas del visor frontal."""
    destino = OUTPUT / "temporal"
    destino.mkdir(parents=True, exist_ok=True)
    modelo_seg = YOLO(str(ROOT / "models" / "bp" / "bp_front_seg_v2.pt"))
    modelo_pose = YOLO(str(ROOT / "models" / "bp" / "bp_front_skeleton_v8.pt"))
    casos = {467: (43, 44, 46, 49, 50), 139: (85, 87, 93, 96),
             267: (0, 25, 58)}
    registros = {}
    for numero, indices in casos.items():
        video = ROOT / "dataset" / "bp" / "front" / f"bp_{numero:03d}.mp4"
        captura = cv2.VideoCapture(str(video))
        anclas, bar_cruda = [], []
        banco, fijo = front.analizar_banco_video(
            captura, modelo_seg, anclas, bar_cruda)
        estimaciones = front.preparar_estimaciones(
            captura, modelo_pose, anclas, bar_cruda)
        registros[f"bp_{numero:03d}"] = {"frames_estimados": sorted(estimaciones)}
        for n in indices:
            captura.set(cv2.CAP_PROP_POS_FRAMES, n)
            ok, frame = captura.read()
            if not ok:
                raise RuntimeError(f"No se pudo leer {video} frame {n}")
            seg = modelo_seg(frame, verbose=False, show=False, save=False)[0]
            pose = modelo_pose(frame, imgsz=960, verbose=False,
                               show=False, save=False)[0]
            raw_bar = sum(seg.names[int(b.cls.item())] == "bar" for b in seg.boxes)
            box = front.seleccionar_banco_frame(seg, banco, fijo)
            count = front.seleccion_barras(seg, pose)
            anotada = seg.plot(img=frame.copy())
            estado = front.dibujar_relacion_frontal(
                anotada, seg, pose, box, numero_barras=count,
                estimacion_barra=estimaciones.get(n) if count == 0 else None,
                barras_raw=raw_bar)
            original = cv2.resize(frame, (960, 540))
            anotada = cv2.resize(anotada, (960, 540))
            panel = np.hstack((original, anotada))
            dibujar_panel(panel, [f"bp_{numero:03d} frame {n} | {estado}",
                                  "Naranja discontinuo = estimacion temporal, NO mascara observada"])
            archivo = destino / f"bp_{numero:03d}_f{n:04d}.jpg"
            if not cv2.imwrite(str(archivo), panel):
                raise OSError(f"No se pudo escribir {archivo}")
        captura.release()
    (OUTPUT / "temporal_controls.json").write_text(
        json.dumps(registros, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(registros, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("before", "after", "actual-before", "actual-after", "raw-controls", "mp-right", "verify", "compare", "actual-compare", "temporal", "diff"), required=True)
    args = parser.parse_args()
    if args.phase == "raw-controls":
        auditar_raw960()
    elif args.phase == "mp-right":
        auditar_mediapipe_derecha()
    elif args.phase == "verify":
        verificar_regresiones()
    elif args.phase == "compare":
        comparar_frames()
    elif args.phase == "actual-compare":
        comparar_frames(actual=True)
    elif args.phase == "temporal":
        auditar_temporal()
    elif args.phase == "diff":
        generar_diff_visores()
    else:
        procesar(args.phase)


if __name__ == "__main__":
    main()
