# Historial de etiquetas

Los CSV vigentes para entrenar están en `etiquetado/dl/` y `etiquetado/sq/`.
Los archivos de esta carpeta son respaldos históricos, **no** etiquetas vigentes.

| Respaldo | Motivo y archivos vigentes afectados | Fecha conocida |
|---|---|---|
| `dl/dl_left_pre_manual_audit.csv`, `dl/dl_right_pre_manual_audit.csv`, `dl/dl_side_pre_manual_audit.csv` | Estado anterior a la revisión manual de 41 eventos DL; los vigentes son `dl_left.csv`, `dl_right.csv` y `dl_side.csv`. Incluye correcciones de etiquetas y descarte explícito del evento físico 302. | 2026-10-01: primer registro de la auditoría en Git; la fecha exacta de cada revisión manual no consta. |
| `sq/sq_front.csv.bak` | Estado previo a retirar `sq_148` y `sq_313` de `sq_front.csv`: tenían etiqueta, pero no vídeo frontal fuente. 446 → 444 filas. | Fecha exacta de creación del respaldo no registrada; el CSV corregido consta en Git al 2026-10-17. |
| `sq/sq_left.csv.bak` | Estado previo a retirar `sq_085` de `sq_left.csv`: tenía etiqueta, pero no vídeo izquierdo fuente. 372 → 371 filas. | Fecha exacta de creación del respaldo no registrada; el CSV corregido consta en Git al 2026-10-17. |

La explicación reproducible de DL está en
[`training/data/dl/shared/manual_label_audit.csv`](../../training/data/dl/shared/manual_label_audit.csv):
identifica eventos físicos, IDs de ambas vistas, etiquetas anteriores y nuevas,
descartes y notas de revisión. `training/dl/apply_manual_label_audit.py`
documenta cómo se generaron los respaldos y cómo se aplicaron las revisiones.
La auditoría de calidad correspondiente está en
[`training/data/dl/shared/manual_quality_audit.csv`](../../training/data/dl/shared/manual_quality_audit.csv).

Los números de vídeo de distintas vistas no implican el mismo intento físico.
