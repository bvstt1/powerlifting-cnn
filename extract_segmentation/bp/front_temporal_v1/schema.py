"""Describe el contrato de 38 features temporales BP front V1.

Fija nombres, orden, unidades, fórmulas y procedencia de los canales;
permite exportar el esquema JSON usado por el clasificador.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import json
from pathlib import Path as _P
from features import FEATURE_NAMES

GRUPO = {}
for n in FEATURE_NAMES[:11]:
    GRUPO[n] = "BAR"
for n in FEATURE_NAMES[11:18]:
    GRUPO[n] = "PERSON"
for n in FEATURE_NAMES[18:24]:
    GRUPO[n] = "BENCH"
for n in FEATURE_NAMES[24:29]:
    GRUPO[n] = "RELACION"
for n in FEATURE_NAMES[29:33]:
    GRUPO[n] = "DINAMICA"
for n in FEATURE_NAMES[33:]:
    GRUPO[n] = "CALIDAD"

POSE_ASSISTED = {
    "bar_cx_norm", "bar_cy_norm", "bar_w_norm", "bar_h_norm",
    "bar_area_rel", "bar_slope", "bar_bbox_occupancy", "bar_conf_max",
    "bar_frac_inside_person",
    "bar_frac_between_wrists", "bar_dy_wrists_norm",
    "silhouette_profile_y_norm", "sep_signed_norm", "sep_coverage",
    "sep_extrap_dist_norm", "wrist_bar_dist_norm", "profile_ambiguous",
    "bar_vx_norm_s", "bar_vy_norm_s", "d_slope_dt", "d_sep_dt",
}

# Una seleccion de BAR asistida por Pose hace asistidas tambien sus medidas de
# mascara. PERSON/BENCH conservan origen exclusivo de segmentacion.
FORMULAS = {
    "bar_cx_norm": "mediana(x pixeles mascara BAR seleccionada)/W",
    "bar_cy_norm": "mediana(y pixeles mascara BAR seleccionada)/H",
    "bar_w_norm": "(max(x_BAR)-min(x_BAR))/W",
    "bar_h_norm": "(max(y_BAR)-min(y_BAR))/H",
    "bar_area_rel": "numero pixeles mascara BAR/(W*H)",
    "bar_slope": "pendiente de polyfit de medianas y por columna interior (15%-85% del ancho BAR); si |pendiente|>0.35 o residuo>0.02H: NaN",
    "bar_bbox_occupancy": "pixeles mascara BAR/((max(x)-min(x)+1)*(max(y)-min(y)+1))",
    "bar_conf_max": "confianza de instancia BAR seleccionada",
    "bar_frac_inside_person": "fraccion pixeles BAR dentro de bbox PERSON; no usar mascara PERSON para esta fraccion",
    "bar_frac_between_wrists": "fraccion pixeles BAR con x entre las dos munecas fiables",
    "bar_dy_wrists_norm": "(mediana(y_BAR)-promedio(y_munecas))/H; positivo si BAR debajo",
    "person_cx_norm": "(bbox_person_x1+bbox_person_x2)/(2W)",
    "person_cy_norm": "(bbox_person_y1+bbox_person_y2)/(2H)",
    "person_w_norm": "(bbox_person_x2-bbox_person_x1)/W",
    "person_h_norm": "(bbox_person_y2-bbox_person_y1)/H",
    "person_area_rel": "pixeles mascara PERSON/(W*H)",
    "person_conf_max": "confianza de instancia PERSON de mayor confianza",
    "silhouette_profile_y_norm": "mediana de y superior de mascara PERSON por columna central /H; franja hombros 25%-75% o banco 40% central si no hay hombros",
    "bench_cx_norm": "(bbox_bench_x1+bbox_bench_x2)/(2W)",
    "bench_cy_norm": "(bbox_bench_y1+bbox_bench_y2)/(2H)",
    "bench_w_norm": "(bbox_bench_x2-bbox_bench_x1)/W",
    "bench_h_norm": "(bbox_bench_y2-bbox_bench_y1)/H",
    "bench_conf_max": "confianza de instancia BENCH de mayor confianza",
    "bench_stability_iou": "mediana_video IoU(bbox_bench_frame,bbox_mediana_video), repetida por frame; fuente temporal offline",
    "sep_signed_norm": "mediana(y_perfil_silueta - y_borde_inferior_BAR)/H; + barra arriba; si proyeccion, eje extendido marcado en prov_sep=2",
    "sep_coverage": "numero columnas centrales con BAR observada/numero columnas centrales de PERSON",
    "sep_extrap_dist_norm": "min(|x_obj-min(x_BAR)|,|x_obj-max(x_BAR)|)/W en proyeccion; 0 si directa",
    "wrist_bar_dist_norm": "minimo de distancia euclidiana de cada muneca al centro BAR / H",
    "profile_ambiguous": "1 si min(y_hombros)-y_perfil>0.35*ancho_hombros o si perfil se define por banco; 0 otro caso",
    "bar_vx_norm_s": "(bar_cx_norm[t]-bar_cx_norm[t-1])/((frame_index[t]-frame_index[t-1])/fps), solo BAR asociada mismo track",
    "bar_vy_norm_s": "(bar_cy_norm[t]-bar_cy_norm[t-1])/dt, solo BAR asociada mismo track",
    "d_slope_dt": "(bar_slope[t]-bar_slope[t-1])/dt, mismo track",
    "d_sep_dt": "(sep_signed_norm[t]-sep_signed_norm[t-1])/dt, mismo track y misma procedencia",
    "q_bar_associated": "1 si una BAR geometrica seleccionada sin competidora; 0 otro caso (no valida agarre fisico)",
    "q_bar_unverified": "1 si YOLO detecto BAR pero ninguna satisface reglas; 0 otro caso",
    "q_bar_multiple": "1 si varias candidatas BAR no duplicadas plausibles; 0 otro caso",
    "q_bar_absent": "1 si ninguna BAR detectada en el frame; 0 otro caso",
    "q_projected": "1 si separacion por proyeccion espacial, 0 si directa, NaN si no medible",
}

SCHEMA = []
for i, n in enumerate(FEATURE_NAMES):
    if n.startswith("bar_cx") or n.startswith("bar_w"):
        unidad, rango = "fraccion de W", "[0,1] o NaN"
    elif n.startswith("bar_cy") or n.startswith("bar_h") or n.startswith("bar_area"):
        unidad, rango = "fraccion de H (area: fraccion de frame)", "[0,1] o NaN"
    elif n == "bar_slope":
        unidad, rango = "px/px", "[-0.35,0.35] o NaN"
    elif n == "bar_bbox_occupancy":
        unidad, rango = "fraccion de bbox ocupada por mascara", "[0,1] o NaN"
    elif n == "bar_conf_max":
        unidad, rango = "confianza YOLO", "[0.25,1] o NaN"
    elif n.startswith("bar_frac"):
        unidad, rango = "fraccion", "[0,1] o NaN"
    elif n == "bar_dy_wrists_norm":
        unidad, rango = "fraccion de H, + barra debajo manos", "[-1,1] o NaN"
    elif n.startswith("person_") and "conf" not in n and "area" not in n:
        unidad, rango = "fraccion de W/H", "[0,1] o NaN"
    elif n == "person_area_rel":
        unidad, rango = "fraccion de frame", "[0,1] o NaN"
    elif n.endswith("_conf_max"):
        unidad, rango = "confianza YOLO", "[0.25,1] o NaN"
    elif n == "silhouette_profile_y_norm":
        unidad, rango = "fraccion de H", "[0,1] o NaN"
    elif n.startswith("bench_") and "conf" not in n and "stab" not in n:
        unidad, rango = "fraccion de W/H", "[0,1] o NaN"
    elif n == "bench_stability_iou":
        unidad, rango = "IoU mediana video", "[0,1] o NaN"
    elif n == "sep_signed_norm":
        unidad, rango = "fraccion de H firmada (+ separada)", "[-1,1] o NaN"
    elif n == "sep_coverage":
        unidad, rango = "fraccion columnas", "[0,1] o NaN"
    elif n == "sep_extrap_dist_norm":
        unidad, rango = "fraccion de W (0 si observado)", "[0,inf) o NaN"
    elif n == "wrist_bar_dist_norm":
        unidad, rango = "fraccion de H", "[0,inf) o NaN"
    elif n in ("profile_ambiguous", "q_bar_associated", "q_bar_unverified",
               "q_bar_multiple", "q_bar_absent", "q_projected"):
        unidad, rango = "binario", "{0,1}"
    elif n in ("bar_vx_norm_s", "bar_vy_norm_s"):
        unidad, rango = "fraccion de W|H por segundo", "real o NaN"
    elif n in ("d_slope_dt", "d_sep_dt"):
        unidad, rango = "unidad/s", "real o NaN"
    else:
        unidad, rango = "varios", "ver reporte"
    SCHEMA.append({
        "indice": i, "nombre": n, "grupo": GRUPO[n],
        "rama": ("POSE-ASSISTED" if n in POSE_ASSISTED
                 else ("QUALITY" if GRUPO[n] == "CALIDAD" else "SEG-ONLY")),
        "unidad": unidad, "rango": rango,
        "formula": FORMULAS[n],
        "fuente": ("RAW mask/bbox/conf/track + pose YOLO original" if n in POSE_ASSISTED
                   else ("RAW mask/bbox/conf + metadatos" if GRUPO[n] != "CALIDAD"
                         else "flags calculados con RAW y pose original")),
        "dependencia_pose": bool(n in POSE_ASSISTED),
        "procedencia": ("prov_bar/quality_flags" if i < 11 else
                        "prov_sep/quality_flags" if 24 <= i <= 28 else
                        "quality_flags/availability"),
        "normalizacion_geometrica": "W,H originales; FPS original en dinamica; sin scaler estadistico",
        "disponibilidad": "availability=1 si medible, 0 si NaN",
        "faltante": "NaN + availability 0; sin imputacion en extractor",
    })


def guardar_schema(ruta):
    _P(ruta).write_text(json.dumps({
        "version": "temporal_features_v1", "F": len(FEATURE_NAMES),
        "features": SCHEMA,
        "provenance": {"0": "ausente", "1": "observado",
                       "2": "proyectado", "4": "no_verificado"},
        "regla": "asociada significa seleccion geometrica provisional; no prueba que el atleta sostenga la barra",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
