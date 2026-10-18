"""python -m training.bp_left_posecnn_compatibility_v1.audit (Validation exclusivamente)."""
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from training.bp_side_segcnn_v1.train import metrics
from training.bp.train_posecnn import POINTS
from training.common.train_utils import PoseCNN
from training.bp_posecnn_common_v1.core import (ROOT, load_development, paths, predict,
                                                rows_csv, save_csv, save_json, sha, verify_embedding)


def main():
    output=ROOT/"training/bp_left_posecnn_compatibility_v1"
    output.mkdir(parents=True,exist_ok=True)
    p=paths("left"); loaded,provenance,rows=load_development("left")
    train_X,train_y,train_ids=loaded["train"]
    val_X,val_y,val_ids=loaded["validation"]
    save_csv(output/"train_validation_source_audit.csv",rows)
    config=json.loads(p["historical_config"].read_text(encoding="utf-8"))
    problems=[]
    if config["view"]!="left" or config["schema"]!=[list(x) for x in enumerate(POINTS)] or config["input_shape"]!=[170,8,3] or config["channels"]!=["xyn_x","xyn_y","yolo_confidence"]:
        problems.append("Configuración original de modelo/keypoints incompatible")
    if config["seed"]!=42 or config["threshold"]!=.5 or config["training"]!={"batch":16,"epochs":200,"early_patience":8,"adam_lr":.001,"weight_decay":.0001,"lr_factor":.5,"lr_patience":4}:
        problems.append("Protocolo histórico cambió")
    if provenance["historical_minus_current"]!={"train":[],"validation":[]}:
        problems.append("Membresía histórica distinta")
    if len(train_ids)!=422 or len(val_ids)!=91:
        problems.append("Conteos erróneos")
    with open(p["historical_scaler"],"rb") as file:
        scaler=pickle.load(file)
    ref=StandardScaler().fit(train_X.reshape(len(train_X),-1))
    diff={k:float(np.max(np.abs(getattr(scaler,k)-getattr(ref,k)))) for k in ("mean_","var_","scale_")}
    scaler_ok=(isinstance(scaler,StandardScaler) and scaler.n_features_in_==170*8*3 and
               scaler.n_samples_seen_==len(train_X) and all(d<2e-5 for d in diff.values()))
    save_json(output/"scaler_audit.json",{"scaler_sha256":sha(p["historical_scaler"]),
              "n_features":int(scaler.n_features_in_),"n_samples_seen":int(scaler.n_samples_seen_),
              "expected_train_original":len(train_X),"max_abs_fit_difference":diff,
              "fit_original_train_only_verified":scaler_ok})
    if not scaler_ok: problems.append("Scaler histórico no coincide con Train original")
    state=torch.load(p["historical_ckpt"],map_location="cpu",weights_only=True)
    model=PoseCNN(T=170,K=8,C=3)
    try:model.load_state_dict(state,strict=True)
    except RuntimeError as e:problems.append("Pesos incompatibles: "+str(e))
    param=sum(q.numel() for q in model.parameters() if q.requires_grad)
    history=rows_csv(p["historical_history"])
    best=min(history,key=lambda r:float(r["val_loss"]))
    save_json(output/"checkpoint_audit.json",{"model_code":"training/common/train_utils.py:PoseCNN",
              "checkpoint_sha256":sha(p["historical_ckpt"]),"history_sha256":sha(p["historical_history"]),
              "parameters":param,"state_keys":list(state),"expected_state_keys":list(model.state_dict()),
              "best_epoch_by_history":int(best["epoch"]),"recorded_validation_loss":float(best["val_loss"]),
              "architecture":"Conv2D same 32/64/128; BatchNorm/ReLU; MaxPool(2,1), Dropout2d(.4), GAP 128, Dense(128,64), Dropout(.5), sigmoid",
              "input_shape":["N",3,170,8],"hidden_embedding_dimension":64})
    if problems:
        save_json(output/"integrity.json",{"status":"LEFT_POSECNN_RETRAIN_REQUIRED","issues":problems,
                   "source":provenance,"test_npy_opened":0,"checkpoint_sha256":sha(p["historical_ckpt"])})
        print("LEFT_POSECNN_RETRAIN_REQUIRED",problems,flush=True);return
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=model.to(device).eval()
    val_scaled=scaler.transform(val_X.reshape(len(val_X),-1)).reshape(val_X.shape).astype(np.float32)
    probs=predict(model,val_scaled,device)
    with torch.no_grad():
        crit=torch.nn.BCELoss()
        weighted=0.
        for a in range(0,len(val_y),16):
            xb=torch.as_tensor(val_scaled[a:a+16],device=device).permute(0,3,1,2).contiguous()
            yb=torch.as_tensor(val_y[a:a+16],dtype=torch.float32,device=device).unsqueeze(1)
            weighted+=float(crit(model(xb),yb))*len(yb)
        loss=weighted/len(val_y)
    tolerance=2e-4
    val_loss_ok=abs(loss-float(best["val_loss"]))<=tolerance
    if not val_loss_ok: problems.append("Validation loss no coincide con best_epoch histórica")
    score=metrics(val_y,probs)
    vals=[{"video_id":v,"label":int(y),"probability_valid":float(prob),"predicted":int(prob>=.5),
           "false_positive":int(y==0 and prob>=.5),"error":int((prob>=.5)!=y)}
          for v,y,prob in zip(val_ids,val_y,probs)]
    save_csv(output/"validation_predictions.csv",vals)
    save_json(output/"validation_metrics.json",{"validation_loss":loss,"historical_best_epoch":int(best["epoch"]),
              "historical_validation_loss":float(best["val_loss"]),"loss_reproduced":val_loss_ok,
              "numerical_tolerance":tolerance,"metrics":score,"T":170,"K":8,"C":3,
              "validation_video_order":val_ids,"historical_validation_probability_file":"not_available",
              "test_npy_opened":0})
    embedding=verify_embedding(model,val_scaled,device)
    save_json(output/"embeddings_audit.json",embedding)
    status="LEFT_POSECNN_HISTORICAL_COMPATIBLE" if not problems else "LEFT_POSECNN_RETRAIN_REQUIRED"
    save_json(output/"integrity.json",{"status":status,"issues":problems,"source":provenance,
              "scaler_reconstructed_from_original_train":scaler_ok,"weights_match_architecture":True,
              "validation_loss_reproduced":val_loss_ok,"embedding_exact_forward":embedding["exact_probability_match"],
              "test_npy_opened":0,"checkpoint_sha256":sha(p["historical_ckpt"]),
              "scaler_sha256":sha(p["historical_scaler"]),"valid_prediction_sha256":sha(output/"validation_predictions.csv")})
    print(status,"Validation loss",loss,"recorded",best["val_loss"],"F1",score["f1"],
          "CM",score["confusion_matrix_0_1"],"scaler_diff",diff,flush=True)


if __name__=="__main__":main()
