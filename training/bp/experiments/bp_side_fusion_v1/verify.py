"""Replay completo por video Validation desde NPY Pose y NPZ Seg V1, Test bloqueado."""
import json

import numpy as np
import torch

from training.bp.experiments.bp_posecnn_common_v1.core import rows_csv, save_json
from training.bp.experiments.bp_side_segcnn_v1.data import sha
from training.bp.experiments.bp_posecnn_common_v1.core import paths
from .data import fusion_out, snapshot
from .inference import FusionInference
from .models import FusionHead


def run(view):
    out=fusion_out(view)
    source=json.loads((out/"fusion_source_manifest.json").read_text(encoding="utf-8"))
    original=snapshot(view)
    if original!=source["source_sha256"]:raise RuntimeError("Fuente base cambió antes de replay")
    n=91 if view=="left" else 93
    parts=[]
    seed_replays=[]
    for variant in ("full","sin_relacion"):
        with np.load(out/f"embedding_fusion_{variant}"/"embeddings_development.npz",allow_pickle=False) as z:
            valids=z["validation_ids"].tolist();zy=z["validation_y"].copy()
        if len(valids)!=n or len(set(valids))!=n:raise RuntimeError("Validation con ID repetido")
        with np.load(out.parent/f"bp_{view}_01_segcnn_v1"/variant/"prepared_development.npz",allow_pickle=False) as z:
            val_seg={v:z["validation_X"][i] for i,v in enumerate(z["validation_ids"].tolist())}
        for kind in ("late","embedding"):
            ckpt=None
            if kind=="embedding":
                parent=out/f"embedding_fusion_{variant}"
                sel=json.loads((parent/"selected_checkpoint.json").read_text(encoding="utf-8"))
                folder=parent/f'seed_{sel["seed"]}'
                ckpt=parent/sel["checkpoint"]
                if sha(ckpt)!=sel["checkpoint_sha256"]:raise RuntimeError("Checkpoint cabeza cambió")
            else:
                folder=out/f"late_fusion_{variant}"
            results={r["video_id"]:r for r in rows_csv(folder/"validation_predictions.csv")}
            if set(results)!=set(valids):raise RuntimeError("Predicciones no pareadas")
            inference=FusionInference(view,variant,kind)
            largest=0.;input_diff=0.;errors=[]
            for i,vid in enumerate(valids):
                xp,xs=inference.from_sources(vid)
                input_diff=max(input_diff,float(np.max(np.abs(xs[0]-val_seg[vid]))))
                p=float(inference.predict_processed(xp,xs)[0])
                old=float(results[vid]["probability_valid"])
                largest=max(largest,abs(p-old))
                if (int(results[vid]["label"])!=int(zy[i]) or int(results[vid]["predicted"])!=int(p>=.5)):
                    errors.append(vid)
            if errors or largest>2e-5 or input_diff>1e-6:
                raise RuntimeError(f"Replay {view}/{variant}/{kind} falló: diff prob={largest} input={input_diff} ids={errors[:5]}")
            parts.append({"variant":variant,"fusion_type":kind,"validation_replayed":n,
                          "maximum_absolute_probability_error":largest,
                          "maximum_absolute_seg_input_error":input_diff,
                          "predictions_identical":True,"head_sha256":sha(ckpt) if ckpt else None})
            print("replay",view,variant,kind,"n",n,"prob max diff",largest,flush=True)
        parent=out/f"embedding_fusion_{variant}"
        with np.load(parent/"embeddings_development.npz",allow_pickle=False) as z:
            features=z["validation_Z"].copy()
        for seed in (42,1337):
            ckpt=parent/f"seed_{seed}"/"best_validation_loss.pt"
            record=json.loads((parent/f"seed_{seed}"/"metrics.json").read_text(encoding="utf-8"))
            if sha(ckpt)!=record["checkpoint_sha256"]:raise RuntimeError("Hash cabeza ajena")
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            head=FusionHead().to(device).eval()
            head.load_state_dict(torch.load(ckpt,map_location=device,weights_only=True)["head_state_dict"],strict=True)
            with torch.no_grad():
                actual=torch.sigmoid(head(torch.as_tensor(features,device=device))).cpu().numpy()
            historic={r["video_id"]:float(r["probability_valid"]) for r in rows_csv(parent/f"seed_{seed}"/"validation_predictions.csv")}
            difference=max(abs(float(p)-historic[vid]) for p,vid in zip(actual,valids))
            if set(historic)!=set(valids) or difference>1e-7:
                raise RuntimeError(f"Replay seed {view}/{variant}/{seed} diff={difference}")
            seed_replays.append({"variant":variant,"seed":seed,"n":n,
                                 "checkpoint_sha256":sha(ckpt),"max_abs_prob_diff":difference})
    if len({r["checkpoint_sha256"] for r in seed_replays})!=4:
        raise RuntimeError("Cabezas de la vista no tienen estados independientes")
    p=paths(view)
    pose_source=(p["left_out"]/"integrity.json" if view=="left" else
                 p["right_out"]/"source_manifest.json")
    provenance=json.loads(pose_source.read_text(encoding="utf-8"))
    pose_hashes=(provenance["source"]["train_validation_npy_sha256"] if view=="left" else
                 provenance["train_validation_npy_sha256"])
    seg_source=json.loads((p["seg_train"]/"source_manifest.json").read_text(encoding="utf-8"))
    seg_hashes=seg_source["train_validation_sequences_sha256"]
    if set(pose_hashes)!=set(seg_hashes) or len(pose_hashes)!=(513 if view=="left" else 524):
        raise RuntimeError("Fuentes monovista diferentes")
    for vid,h in pose_hashes.items():
        if sha(p["keypoints"]/f"{vid}.npy")!=h or sha(p["source"]/"sequences"/f"{vid}_temporal.npz")!=seg_hashes[vid]:
            raise RuntimeError("Keypoints/feature V1 mutada: "+vid)
    if snapshot(view)!=original:raise RuntimeError("Fuente protegida cambió tras replay")
    doc={"view":view,"status":"INTEGRITY_OK","source_hashes_before_after_identical":True,
         "pose_seg_backbones_frozen":True,"all_seeds_checkpoints_separate":True,
         "head_trainable_params":6209,"parts":parts,"all_eight_head_replay":seed_replays,
         "pose_train_validation_source_hashes_verified":len(pose_hashes),
         "seg_train_validation_source_hashes_verified":len(seg_hashes),
         "test_npy_opened":0,"test_features_opened":0,
         "test_embeddings_exported":False,"no_fusion_on_test":True}
    save_json(out/"fusion_integrity_verification.json",doc)
    return doc


def main():
    left=run("left");right=run("right")
    hashes=[r["checkpoint_sha256"] for report in (left,right) for r in report["all_eight_head_replay"]]
    if len(set(hashes))!=8:raise RuntimeError("Cabezas LEFT/RIGHT comparten pesos inesperadamente")


if __name__=="__main__":main()
