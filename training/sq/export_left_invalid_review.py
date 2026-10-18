"""Exporta falsos válidos de sq_left v2, sin entrenar ni tocar etiquetas.

Uso: python -m training.sq.export_left_invalid_review
"""
import csv
from collections import Counter

from training.sq import train_posecnn as sq
from training.sq.run_analysis import read_predictions


def main():
    p = sq.paths("left", "baseline_t170_v2")
    labels = {row["video_id"]: row for row in sq.load_labels(p["labels"])}
    roles = {}
    with (p["data"] / "split.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            roles[row["video_id"]] = row["split"]
    sources = (("cv_outer", p["results"] / "cv_predictions.csv"),
               ("validation_main", p["results"] / "validation_predictions_recovered.csv"),
               ("test_main", p["results"] / "test_predictions.csv"))
    rows = []
    for source, path in sources:
        for prediction in read_predictions(path):
            if int(prediction["true_label"]) != 0 or int(prediction["predicted_label"]) != 1:
                continue
            ident = prediction["video_id"]
            video = sq.ROOT / "dataset" / "sq" / "left" / f"{ident}.mp4"
            pose = p["source"] / f"{ident}.npy"
            label = labels[ident]
            assert int(label["label"]) == 0
            if source == "cv_outer":
                assert roles[ident] in ("train", "validation")
            else:
                assert roles[ident] == ("test" if source == "test_main" else "validation")
            rows.append({"video_id": ident, "source": source,
                         "fold": prediction.get("fold", ""), "main_split": roles[ident],
                         "probability_valid": float(prediction["probability_valid"]),
                         "codigos": label["codigos"], "criterios": label["criterios"],
                         "video_path": str(video.relative_to(sq.ROOT)),
                         "video_exists": int(video.is_file()),
                         "keypoints_path": str(pose.relative_to(sq.ROOT)),
                         "keypoints_exists": int(pose.is_file())})
    rows.sort(key=lambda r: (r["source"], -r["probability_valid"], r["video_id"]))
    target = p["results"] / "invalid_as_valid_manual_review.csv"
    sq.save_csv(target, rows, list(rows[0]))
    counts = Counter(row["source"] for row in rows)
    print("Exportados:", dict(counts), "| videos no disponibles:",
          [r["video_id"] for r in rows if not r["video_exists"]], "|", target)
    for source in ("cv_outer", "validation_main", "test_main"):
        print("\n" + source)
        for row in rows:
            if row["source"] == source:
                print(row["fold"], row["video_id"], f"p={row['probability_valid']:.4f}",
                      row["codigos"] or "(sin código)", row["video_path"])


if __name__ == "__main__":
    main()
