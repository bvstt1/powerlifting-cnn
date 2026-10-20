# BP right: datos y evidencia

`raw/` contiene máscaras de barra, banco y persona de los 630 vídeos de
esta vista, producidas por `extract_segmentation/bp/extract_segmentation_side_bp.py`
con `bp_side_seg_v1.pt`. Este RAW lateral conserva las instancias originales
**sin tracking temporal**; el resumen `manifests/integrity_summary.json`
registra 630/630 vídeos completos en la ejecución histórica.

`temporal_features_v1/sequences/` contiene las features (T, 62) calculadas
desde RAW y pose YOLO right por `extract_segmentation/bp/side_temporal_v1/`.
Su esquema, cohorte y manifiesto están en `temporal_features_v1/`. Esta es
la fuente lateral para la SegCNN right, no `front/tracking_v3/`.

`manifests/` reúne auditoría de entradas, configuración, hashes, métricas
de calidad y errores; `previews/` y los CSV/reportes de
`temporal_features_v1/` documentan disponibilidad y revisiones visuales.
Son evidencia de las muestras, no datos de entrada adicionales del modelo.
Los IDs left y right no acreditan intentos físicos emparejados.
