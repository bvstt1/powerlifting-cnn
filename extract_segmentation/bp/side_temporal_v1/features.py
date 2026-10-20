"""Calcula las 62 features temporales BP lateral por vídeo y vista.

Combina geometría RAW con pose, disponibilidad y procedencia por canal;
no utiliza etiquetas, no entrena y no inventa tracking en el RAW lateral.
"""
import numpy as np
import cv2

from . import geometry as G
from . import raw_reader as R
from .schema import F, INDEX, NAMES, PROV, QUALITY


def _centro_mascara(instancia):
    m = cv2.moments(instancia["crop"].astype(np.uint8), binaryImage=True)
    return (instancia["x"] + m["m10"]/m["m00"],
            instancia["y"] + m["m01"]/m["m00"])


def _xywh(box, ancho, alto):
    return ((box[0]+box[2])/(2*ancho), (box[1]+box[3])/(2*alto),
            (box[2]-box[0])/ancho, (box[3]-box[1])/alto)


def _banco_referencia(d, ancho):
    """Referencia offline descriptiva; nunca rellena bbox/conf observados."""
    boxes = []
    for f in range(len(d["frame_index"])):
        a, b = map(int, d["frame_instance_offsets"][f:f+2])
        ids = [j for j in range(a,b) if int(d["instance_class_id"][j]) == 1]
        if ids:
            j = max(ids, key=lambda k: float(d["instance_confidence"][k]))
            boxes.append(np.asarray(d["instance_bbox_xyxy_px"][j], dtype=float))
    if len(boxes) < 2:
        return None, np.nan
    medio = np.median(boxes, axis=0)
    estabilidad = float(np.median([G.bbox_iou(b, medio) for b in boxes]))
    if len(boxes) < 5 or estabilidad < .65 or medio[2]-medio[0] < .1*ancho:
        return None, estabilidad
    return medio, estabilidad


def extraer(d, meta, pose):
    t = len(d["frame_index"])
    if pose.shape != (t, 8, 3) or not np.issubdtype(pose.dtype, np.number) or np.isinf(pose).any():
        raise ValueError(f"Pose YOLO incompatible: {pose.shape}, {pose.dtype}")
    if t != int(meta["frames_decoded"]) or meta["view"] not in ("left", "right"):
        raise ValueError("RAW: metadatos T/vista inconsistentes")
    if not np.all(d["frame_image_shape_hw"] == d["frame_image_shape_hw"][0]):
        raise ValueError("Resolución variable no soportada")
    alto, ancho = map(int, d["frame_image_shape_hw"][0]); fps = float(meta["fps"])
    X = np.full((t, F), np.nan, dtype=np.float32)
    proveniencia = np.zeros((t, F), dtype=np.uint8)
    estado_barra = np.zeros(t, dtype=np.uint8)
    estado_perfil = np.zeros(t, dtype=np.uint8)
    estado_relacion = np.zeros(t, dtype=np.uint8)
    ids = np.full((t, 3), -1, dtype=np.int32)  # BAR/PERSON/BENCH global RAW local al archivo
    raw_counts = np.asarray(d["frame_class_counts"], dtype=np.int16)
    ref_banco, estabilidad = _banco_referencia(d, ancho)
    perfiles_piloto = {}
    anterior = None; estado_anterior = 0
    gap = 0; ambiguos = 0

    for f in range(t):
        def poner(nombre, valor, prov):
            if np.isfinite(valor):
                X[f, INDEX[nombre]] = valor
                proveniencia[f, INDEX[nombre]] = prov

        a, b = map(int, d["frame_instance_offsets"][f:f+2])
        inst = [R.instancia(d, j, f) for j in range(a,b)]
        barras = [i for i in inst if i["class"] == "bar"]
        personas = [i for i in inst if i["class"] == "person"]
        bancos = [i for i in inst if i["class"] == "bench"]
        persona = max(personas, key=lambda i: i["conf"], default=None)
        banco = max(bancos, key=lambda i: i["conf"], default=None)
        if persona is not None:
            ids[f,1] = persona["id"]
            box = persona["box"]
            cx, cy = _centro_mascara(persona)
            for n, v in zip(("person_bbox_cx_w","person_bbox_cy_h","person_bbox_w_w","person_bbox_h_h"), _xywh(box, ancho, alto)):
                poner(n, v, PROV["seg"])
            for n,v in (("person_mask_cx_w",cx/ancho),("person_mask_cy_h",cy/alto),
                        ("person_area_frame",persona["area"]/(ancho*alto)),("person_conf",persona["conf"])):
                poner(n,v,PROV["seg"])
        if banco is not None:
            ids[f,2] = banco["id"]
            box = banco["box"]
            cx, cy = _centro_mascara(banco)
            for n,v in zip(("bench_bbox_cx_w","bench_bbox_cy_h","bench_bbox_w_w","bench_bbox_h_h"), _xywh(box, ancho, alto)):
                poner(n,v,PROV["seg"])
            for n,v in (("bench_mask_cx_w",cx/ancho),("bench_mask_cy_h",cy/alto),("bench_conf",banco["conf"])):
                poner(n,v,PROV["seg"])
        if np.isfinite(estabilidad):
            poner("bench_stability_iou", estabilidad, PROV["bench_reference"])
        usar_referencia = banco is None and ref_banco is not None
        box_banco = banco["box"] if banco is not None else ref_banco

        muneca = G.punto_pose(pose[f], 4, ancho, alto)
        cadera = G.punto_pose(pose[f], 2, ancho, alto)
        hombro = G.punto_pose(pose[f], 3, ancho, alto)
        barra, estado = G.elegir_barra(barras, persona, muneca, ancho, alto)
        estado_barra[f] = estado
        poner("bar_candidates_raw", len(barras), PROV["seg"])
        ajuste = None; perfil = None; rel = {"state": 0}
        if barra is not None:
            ids[f,0] = barra["id"]
            xs, ys = G.pixeles(barra)
            prov_bar = PROV["pose"] if estado == 1 else PROV["seg"]
            geo = (("bar_cx_w",np.median(xs)/ancho), ("bar_cy_h",np.median(ys)/alto),
                   ("bar_width_w",barra["crop"].shape[1]/ancho), ("bar_height_h",barra["crop"].shape[0]/alto),
                   ("bar_area_frame",barra["area"]/(ancho*alto)),
                   ("bar_visible_length_w",(xs.max()-xs.min())/ancho),
                   ("bar_conf",barra["conf"]), ("bar_occupancy",barra["area"]/barra["crop"].size))
            for n,v in geo:
                poner(n,v,prov_bar)
            ajuste = G.pendiente(barra, alto)
            if ajuste is not None:
                poner("bar_slope", ajuste[0], prov_bar)
            if muneca is not None:
                poner("wrist_bar_distance_diag", G.punto_a_mascara(barra,muneca)/np.hypot(ancho,alto), PROV["pose"])

        # El perfil de PERSON guiado por pose no depende de BAR. Solo el
        # respaldo orientado por BENCH requiere conocer una barra candidata.
        if persona is not None:
            perfil, origen_perfil = G.torso(persona, pose[f], box_banco, barra, ancho, alto, usar_referencia)
            estado_perfil[f] = origen_perfil
            if perfil is not None:
                pp = PROV["pose"] if origen_perfil == 1 else PROV["bench_estimate"]
                for n,v in (("torso_profile_x_w",perfil["x"]/ancho),
                            ("torso_profile_y_h",perfil["y"]/alto),
                            ("torso_profile_coverage",perfil["coverage"])):
                    poner(n,v,pp)
                if barra is not None:
                    extremo = G.extremo_proximal(barra, perfil["x"])
                    for n,v in (("bar_proximal_x_w",extremo[0]/ancho),
                                ("bar_proximal_y_h",extremo[1]/alto),
                                ("bar_proximal_thickness_h",extremo[2]/alto)):
                        poner(n,v,pp)
                    rel = G.relacion(barra, perfil, ajuste, alto)
                    estado_relacion[f] = rel["state"]
                    if rel["state"]:
                        pr = (PROV["pose"] if origen_perfil==1 else PROV["bench_estimate"]) if rel["state"]==1 else (
                            PROV["projection"] if rel["state"]==2 else PROV["far_projection"])
                        for n,v in (("sep_signed_h",rel["sep"]/alto), ("extrap_distance_w",rel["D"]/ancho),
                                    ("bar_visible_length_relation_w",rel["L"]/ancho),
                                    ("extrap_ratio_d_l",rel["ratio"])):
                            poner(n,v,pr)
                    if f in (0,t//2,t-1):
                        perfiles_piloto[f] = {"points": perfil["points"], "endpoint": extremo,
                                              "slope": ajuste, "relation": rel}

        gap = 0 if barra is not None else gap+1
        ambiguos = ambiguos+1 if estado==3 else 0
        poner("bar_gap_frames",gap,PROV["quality"])
        poner("bar_ambiguity_run_frames",ambiguos,PROV["quality"])
        continuo = f>0 and G.continuidad(anterior, barra, estado_anterior, estado, ancho, alto)
        if continuo:
            for n, clave, escala in (("bar_dx_w","bar_cx_w",1.),("bar_dy_h","bar_cy_h",1.),
                                      ("bar_vx_w_s","bar_cx_w",fps),("bar_vy_h_s","bar_cy_h",fps),
                                      ("bar_slope_delta_s","bar_slope",fps)):
                p = INDEX[clave]
                if np.isfinite(X[f,p]) and np.isfinite(X[f-1,p]):
                    poner(n,(float(X[f,p])-float(X[f-1,p]))*escala,PROV["local_association"])
            if (estado_relacion[f] == estado_relacion[f-1] != 0 and
                    estado_perfil[f] == estado_perfil[f-1] and
                    np.isfinite(X[f,INDEX["sep_signed_h"]]) and np.isfinite(X[f-1,INDEX["sep_signed_h"]])):
                poner("sep_delta_h_s",(float(X[f,INDEX["sep_signed_h"]])-float(X[f-1,INDEX["sep_signed_h"]]))*fps,
                      PROV["local_association"])
        estados = {
            "q_bar_detected": bool(barras), "q_bar_selected": barra is not None,
            "q_bar_identity_unverified": estado in (2,4), "q_bar_identity_ambiguous": estado==3,
            "q_bar_absent": estado==0, "q_person_absent": persona is None,
            "q_bench_absent": banco is None, "q_pose_wrist_reliable": muneca is not None,
            "q_pose_torso_reliable": cadera is not None and hombro is not None,
            "q_profile_observed": estado_perfil[f]==1,
            "q_profile_bench_estimated": estado_perfil[f]==2,
            "q_relation_direct": estado_relacion[f]==1,
            "q_projection_ordinary": estado_relacion[f]==2,
            "q_projection_far": estado_relacion[f]==3,
            "q_relation_unevaluable": estado_relacion[f]==0,
            "q_bench_reference_used": usar_referencia and estado_perfil[f]==2,
            "q_bar_local_association": continuo,
        }
        for n in QUALITY:
            poner(n,float(estados[n]),PROV["quality"])
        anterior, estado_anterior = barra, estado

    if np.isinf(X).any() or not np.array_equal(np.isfinite(X), proveniencia!=0):
        raise ValueError("Inf o procedencia/availability inconsistentes")
    return {"features": X, "availability": np.isfinite(X).astype(np.uint8),
            "provenance": proveniencia, "bar_state": estado_barra,
            "profile_state": estado_perfil, "relation_state": estado_relacion,
            "raw_instance_ids": ids, "raw_class_counts": raw_counts,
            "preview_geometry": perfiles_piloto, "bench_stability": estabilidad,
            "bench_reference_box": ref_banco,
            "fps": fps, "image_hw": np.asarray([alto,ancho],dtype=np.int32),
            "frame_index": d["frame_index"].copy(), "timestamp_sec": d["timestamp_sec"].copy()}
