"""Define los parámetros fijos de las features temporales BP front V1.

Incluye el esquema RAW aceptado, el modelo pose y las reglas geométricas;
este módulo no ejecuta extracción ni escribe datos.
"""

CONFIG_V1 = {
    "version": "temporal_features_v1",
    "raw_schema": "bp_front_segmentation_raw_v2",
    "raw_inference": {"imgsz": 960, "retina_masks": True,
                      "conf": 0.25, "iou": 0.7},
    "pose_model": "bp_front_skeleton_v8.pt",
    "pose_coords": "x,y normalizados (xyn) + confianza; conversion a px con W,H del frame",
    "conf_pose_min": 0.5,
    "wrist_sep_min_px": 30.0,
    "shoulder_w_min_px": 30.0,
    "guard_dy_frac_h": 0.03,
    "cov_persona": 0.5,
    "cov_munecas": 0.5,
    "dedup_inter_min": 0.85,
    "slope_ancho_min": 30,
    "slope_cobertura_min": 0.5,
    "slope_max_abs": 0.35,
    "slope_residuo_frac_h": 0.02,
    "perfil_ancho_min": 5,
    "perfil_cobertura_min": 0.7,
    "perfil_ambig_frac_w": 0.35,
    "sep_cobertura_obs_min": 0.6,
    "sep_alcance_max_d_l": 4.0,
    "conf_ancla_barra": 0.65,
    "enable_estimations": True,
    "max_estimated_gap": 6,
    "dynamic_identity": "mismo track_id unico en t-1,t + bar asociada; d_sep solo si igual procedencia",
    "no_yolo_rerun": True,
    "no_label_use": True,
}
