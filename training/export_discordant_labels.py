"""Exporta eventos con label discordante entre vistas (pareo por evento fisico).

El pareo usa la correspondencia auditada (hueco en evento 310):
  evento e<310: left dl_e <-> right dl_e
  evento e>310: left dl_{e-1} <-> right dl_e
  evento 310: solo right (sin pareja, se excluye).

NO modifica etiquetas. Solo exporta para revision posterior.

Salida: training/data/dl/discordant_view_labels.csv
Uso: python training/export_discordant_labels.py [--profile ignorado]
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training_config import DL_LEFT_GAP_EVENT, get_config

OUTPUT_COLUMNS = [
    "group_id",
    "left_video_id",
    "right_video_id",
    "left_label",
    "right_label",
    "left_resultado",
    "right_resultado",
    "left_codigos",
    "right_codigos",
]


def left_video_for_event(event):
    if event == DL_LEFT_GAP_EVENT:
        return None
    if event < DL_LEFT_GAP_EVENT:
        return "dl_%03d" % event
    return "dl_%03d" % (event - 1)


def main():
    left_config = get_config("dl_left")
    right_config = get_config("dl_right")
    left_df = pd.read_csv(left_config["csv_path"], dtype=str)
    right_df = pd.read_csv(right_config["csv_path"], dtype=str)
    left_rows = dict(zip(left_df["video_id"], left_df.to_dict("records")))
    right_rows = dict(zip(right_df["video_id"], right_df.to_dict("records")))
    audit_path = left_config["data_dir"].parent / "manual_label_audit.csv"
    excluded_events = set()

    if audit_path.exists():
        audit = pd.read_csv(audit_path)
        excluded_events = set(
            audit.loc[
                audit["discarded"].astype(int) == 1,
                "physical_event",
            ].astype(str)
        )

    discordant = []

    for event in range(1, 414):
        group_id = "dl_event_%03d" % event

        if group_id in excluded_events:
            continue

        right_video = "dl_%03d" % event
        left_video = left_video_for_event(event)

        if left_video is None or left_video not in left_rows:
            continue
        if right_video not in right_rows:
            continue

        left_row = left_rows[left_video]
        right_row = right_rows[right_video]

        if str(left_row["label"]) == str(right_row["label"]):
            continue

        discordant.append({
            "group_id": group_id,
            "left_video_id": left_video,
            "right_video_id": right_video,
            "left_label": left_row["label"],
            "right_label": right_row["label"],
            "left_resultado": left_row.get("resultado", ""),
            "right_resultado": right_row.get("resultado", ""),
            "left_codigos": left_row.get("codigos", ""),
            "right_codigos": right_row.get("codigos", ""),
        })

    output_path = (
        left_config["data_dir"].parent / "discordant_view_labels.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(discordant, columns=OUTPUT_COLUMNS).to_csv(
        output_path, index=False
    )

    print(f"Eventos discordantes: {len(discordant)}")
    print(f"Guardado en: {output_path}")


if __name__ == "__main__":
    main()
