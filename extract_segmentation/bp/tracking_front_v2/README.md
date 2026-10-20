# Análisis y reconstrucción del tracking frontal V2.2

Orden de trabajo: `scout` (exploración de frames con YOLO) → `rebuild`
(reconstruir desde RAW existente, sin YOLO) → `finalize` (cierre e integridad
del dataset V2.2) → `audit` (continuidad BAR y revisión visual). Las fases
se ejecutan individualmente; `run.py` no las encadena ni cambia parámetros.

| Etapa | Script real | Resultado |
|---|---|---|
| `scout` | `scout_tracking_cases_front_bp.py` | Casos difíciles en `segmentation/bp/front/manifests/`. |
| `rebuild` | `rebuild_tracking_front_bp_from_raw.py` | `segmentation/bp/front/tracking_v2/`. |
| `finalize` | `../finalize_segmentation_front_bp.py` | Validación del conjunto V2.2 e informe. |
| `audit` | `audit_bar_tracking_front_bp.py` | Métricas de continuidad y evidencias para investigar V3. |

Ejemplo desde la raíz (consultar opciones de cada etapa con `--help`):

```powershell
python extract_segmentation/bp/tracking_front_v2/run.py rebuild --videos bp_001
```

`rebuild_tracking_front_bp_from_raw.py` está ahora en esta carpeta. El
manifiesto original del dataset registró **su antigua ruta y hash**; Git
conserva ese código histórico. `finalize_segmentation_front_bp.py` sigue en
`extract_segmentation/bp/` porque cierra el dataset completo, no solo el
tracking. Su verificación de la configuración ya congelada puede rechazar el
código reorganizado: usar el commit original para reproducir aquel cierre, o
una salida nueva para una ejecución nueva. No se reescribieron RAW, features,
manifiestos ni resultados al mover los scripts.
