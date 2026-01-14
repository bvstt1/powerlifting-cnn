import csv
import json

labels = {}

with open("labels.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        attempt_id = row["attempt_id"]

        labels[attempt_id] = {
            "exercise": row["exercise"],
            "judges": [
                {
                    "valid": int(row["judge_1_valid"]),
                    "reason": row["judge_1_reason"] or None
                },
                {
                    "valid": int(row["judge_2_valid"]),
                    "reason": row["judge_2_reason"] or None
                },
                {
                    "valid": int(row["judge_3_valid"]),
                    "reason": row["judge_3_reason"] or None
                }
            ],
            "final": {
                "valid": int(row["final_valid"]),
                "reason": row["final_reason"] or None
            }
        }

with open("labels.json", "w", encoding="utf-8") as f:
    json.dump(labels, f, indent=2, ensure_ascii=False)
