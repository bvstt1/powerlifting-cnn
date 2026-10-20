"""Define el contrato de 62 canales BP lateral temporal V1.

Fija orden, grupos, fórmulas y códigos de procedencia; los valores ausentes
se representan como NaN con disponibilidad explícita.
"""
from collections import OrderedDict

GROUPS = OrderedDict([
    ("BAR", ["bar_cx_w", "bar_cy_h", "bar_width_w", "bar_height_h",
             "bar_area_frame", "bar_slope", "bar_visible_length_w",
             "bar_conf", "bar_occupancy", "bar_candidates_raw",
             "bar_proximal_x_w", "bar_proximal_y_h", "bar_proximal_thickness_h"]),
    ("PERSON", ["person_bbox_cx_w", "person_bbox_cy_h", "person_bbox_w_w",
                "person_bbox_h_h", "person_mask_cx_w", "person_mask_cy_h",
                "person_area_frame", "person_conf", "torso_profile_x_w",
                "torso_profile_y_h", "torso_profile_coverage"]),
    ("BENCH", ["bench_bbox_cx_w", "bench_bbox_cy_h", "bench_bbox_w_w",
               "bench_bbox_h_h", "bench_mask_cx_w", "bench_mask_cy_h",
               "bench_conf", "bench_stability_iou"]),
    ("RELATION", ["sep_signed_h", "extrap_distance_w", "bar_visible_length_relation_w",
                  "extrap_ratio_d_l", "wrist_bar_distance_diag"]),
    ("TEMPORAL", ["bar_dx_w", "bar_dy_h", "bar_vx_w_s", "bar_vy_h_s",
                  "bar_slope_delta_s", "sep_delta_h_s", "bar_gap_frames",
                  "bar_ambiguity_run_frames"]),
    ("QUALITY", ["q_bar_detected", "q_bar_selected", "q_bar_identity_unverified",
                 "q_bar_identity_ambiguous", "q_bar_absent", "q_person_absent",
                 "q_bench_absent", "q_pose_wrist_reliable", "q_pose_torso_reliable",
                 "q_profile_observed", "q_profile_bench_estimated", "q_relation_direct",
                 "q_projection_ordinary", "q_projection_far", "q_relation_unevaluable",
                 "q_bench_reference_used", "q_bar_local_association"]),
])
NAMES = [nombre for nombres in GROUPS.values() for nombre in nombres]
INDEX = {nombre: i for i, nombre in enumerate(NAMES)}
F = len(NAMES)
QUALITY = GROUPS["QUALITY"]

# Provenance por canal: 0 falta, 1 SEG-ONLY, 2 POSE-ASSISTED,
# 3 perfil estimado a partir de banco, 4 proyección ordinaria,
# 5 proyección lejana, 6 referencia temporal del banco,
# 7 asociación local provisional, 8 estado de calidad (dep. indicada en esquema).
PROV = {"missing": 0, "seg": 1, "pose": 2, "bench_estimate": 3,
        "projection": 4, "far_projection": 5, "bench_reference": 6,
        "local_association": 7, "quality": 8}

FORMULAS = {
    "bar_cx_w": "mediana x mascara seleccionada / W", "bar_cy_h": "mediana y mascara seleccionada / H",
    "bar_width_w": "(max x-min x+1)/W", "bar_height_h": "(max y-min y+1)/H",
    "bar_area_frame": "area mascara/(W*H)",
    "bar_slope": "polyfit y=a*x+b de medianas de columnas interiores 15%-85%; ancho>=30, cobertura>=0.5, |a|<=0.35, residuo<=0.02H",
    "bar_visible_length_w": "(max x-min x)/W", "bar_conf": "confianza YOLO instancia BAR",
    "bar_occupancy": "area mascara/(ancho crop * alto crop)",
    "bar_candidates_raw": "numero instancias BAR del frame, incluso sin seleccion",
    "bar_proximal_x_w": "mediana x de franja extrema 15% mas cercana a x_torso / W",
    "bar_proximal_y_h": "mediana y de esa franja / H",
    "bar_proximal_thickness_h": "max(0, percentil85 y - mediana y) / H; no es radio fisico",
    "person_bbox_cx_w": "(x1+x2)/(2W)", "person_bbox_cy_h": "(y1+y2)/(2H)",
    "person_bbox_w_w": "(x2-x1)/W", "person_bbox_h_h": "(y2-y1)/H",
    "person_mask_cx_w": "media x pixeles PERSON/W", "person_mask_cy_h": "media y pixeles PERSON/H",
    "person_area_frame": "area PERSON/(W*H)", "person_conf": "confianza instancia PERSON",
    "torso_profile_x_w": "mediana x del borde superior visible de PERSON/W en franja 45%-75% de cadera->hombro; respaldo franja banco",
    "torso_profile_y_h": "mediana y del borde superior de silueta PERSON/H (no pecho anatomico)",
    "torso_profile_coverage": "columnas con PERSON / columnas de la franja de torso",
    "bench_bbox_cx_w": "(x1+x2)/(2W)", "bench_bbox_cy_h": "(y1+y2)/(2H)",
    "bench_bbox_w_w": "(x2-x1)/W", "bench_bbox_h_h": "(y2-y1)/H",
    "bench_mask_cx_w": "media x pixeles BENCH/W", "bench_mask_cy_h": "media y pixeles BENCH/H",
    "bench_conf": "confianza BENCH observada; NaN si ausente incluso con referencia",
    "bench_stability_iou": "mediana temporal IoU(bbox BENCH observada, bbox mediana del video); offline descriptiva",
    "sep_signed_h": "(y_contorno - y_barra_proyectada)/H; positivo: barra proyectada arriba; no indica contacto",
    "extrap_distance_w": "D/W, 0 si borde BAR observado en x_torso",
    "bar_visible_length_relation_w": "L/W, L=max x-min x de mascara BAR seleccionada",
    "extrap_ratio_d_l": "D/L; 0 si observacion directa; abstencion D>4L",
    "wrist_bar_distance_diag": "distancia euclidea de muneca fiable al pixel de BAR mas cercano / diagonal",
    "bar_dx_w": "(cx[t]-cx[t-1])/W, solo asociacion local respaldada",
    "bar_dy_h": "(cy[t]-cy[t-1])/H, solo asociacion local respaldada",
    "bar_vx_w_s": "bar_dx_w / (1/fps)", "bar_vy_h_s": "bar_dy_h / (1/fps)",
    "bar_slope_delta_s": "(slope[t]-slope[t-1])*fps, si ambos ajustes validos",
    "sep_delta_h_s": "(sep[t]-sep[t-1])*fps, solo misma procedencia y mismo tipo de perfil",
    "bar_gap_frames": "longitud acumulada de frames consecutivos sin BAR seleccionada, 0 si seleccionada",
    "bar_ambiguity_run_frames": "longitud consecutiva en estado ambiguo, 0 en otros",
}
FORMULAS.update({n: "1 si " + n[2:].replace("_", " ") + "; 0 en otro caso (estado observado/calculado)" for n in QUALITY})
POSE_DEP = set(GROUPS["RELATION"] + ["torso_profile_x_w", "torso_profile_y_h", "torso_profile_coverage",
                                      "bar_proximal_x_w", "bar_proximal_y_h", "bar_proximal_thickness_h",
                                      "wrist_bar_distance_diag"] + GROUPS["TEMPORAL"][:6] +
               ["q_bar_selected", "q_bar_identity_unverified", "q_bar_identity_ambiguous",
                "q_pose_wrist_reliable", "q_pose_torso_reliable", "q_profile_observed",
                "q_profile_bench_estimated", "q_relation_direct", "q_projection_ordinary",
                "q_projection_far", "q_relation_unevaluable", "q_bar_local_association"])
# Toda geometria BAR seleccionada puede depender de la muñeca aunque sus pixeles son segmentacion.
POSE_DEP.update(GROUPS["BAR"])
POSE_DEP.discard("bar_candidates_raw")
POSE_DEP.update(["bar_gap_frames", "bar_ambiguity_run_frames"])


def esquema():
    return {"version": "bp_side_temporal_features_v1", "F": F,
            "feature_names": NAMES, "provenance_codes": PROV,
            "states": {"bar": {"0": "absent", "1": "selected_wrist_corrob",
                                "2": "selected_identity_unverified", "3": "ambiguous", "4": "unverified_no_selection"},
                       "profile": {"0": "none", "1": "pose_profile", "2": "bench_estimate"},
                       "relation": {"0": "unevaluable", "1": "observed_direct", "2": "ordinary_projection", "3": "far_projection"}},
            "features": [{"index": INDEX[n], "name": n, "group": g,
                          "branch": "QUALITY" if g == "QUALITY" else ("POSE-ASSISTED" if n in POSE_DEP else "SEG-ONLY"),
                          "depends_on_pose": n in POSE_DEP,
                          "formula": FORMULAS[n], "missing": "NaN + availability=0; sin imputacion"}
                         for g, nombres in GROUPS.items() for n in nombres]}
