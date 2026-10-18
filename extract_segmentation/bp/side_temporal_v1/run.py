"""python -m extract_segmentation.bp.side_temporal_v1.run --view left --phase audit|pilot|all|verify"""
import argparse
import csv
import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from . import geometry as G
from . import raw_reader as R
from .features import extraer
from .schema import F, INDEX, NAMES, PROV, esquema

ROOT = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parent
FILES = ("run.py", "schema.py", "raw_reader.py", "geometry.py", "features.py")
EXPECTED = {"left": (611, 51576, 166140), "right": (630, 54061, 164706)}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def csv_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, names):
    temp = path.with_suffix(path.suffix+".tmp")
    with open(temp, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=names, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    for attempt in range(20):
        try:
            os.replace(temp,path); return
        except PermissionError:
            if attempt==19: raise
            time.sleep(.2*(attempt+1))


def write_json(path, obj):
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temp,path)


def paths(view):
    base = ROOT/"segmentation/bp"/view
    out = base/"temporal_features_v1"
    return {"base": base, "out": out, "raw": base/"raw", "pose": ROOT/"keypoints/bp/yolo"/view,
            "video": ROOT/"dataset/bp"/view,
            "split": ROOT/"training/data/bp"/view/"posecnn/split.csv",
            "labels": ROOT/"etiquetado/bp"/f"bp_{view}.csv"}


def source_config(view):
    p=paths(view)
    frozen = json.loads((p["base"]/"manifests/frozen_config.json").read_text(encoding="utf-8"))
    summary = json.loads((p["base"]/"manifests/integrity_summary.json").read_text(encoding="utf-8"))
    if summary["videos_error"] or summary["videos_complete"] != EXPECTED[view][0] or (summary["frames"],summary["instances"])!=EXPECTED[view][1:]:
        raise RuntimeError("Integridad RAW previa no coincide")
    if frozen["model_sha256"] != summary["model_sha256"] or frozen["inference"]["retina_masks"] is not True:
        raise RuntimeError("Configuracion RAW inconsistente")
    return frozen,summary


def audit(view):
    p=paths(view); out=p["out"]; out.mkdir(parents=True,exist_ok=True)
    frozen,summary=source_config(view)
    manifest=csv_rows(p["base"]/"manifests/raw_manifest.csv")
    inputs=csv_rows(p["base"]/"manifests/input_audit.csv")
    integrity=csv_rows(p["base"]/"manifests/integrity.csv")
    splits=csv_rows(p["split"]); labels=csv_rows(p["labels"])
    for rows,name in ((manifest,"RAW"),(inputs,"videos"),(integrity,"integrity"),(splits,"split"),(labels,"labels")):
        if len(rows)!=len({r["video_id"] for r in rows}):
            raise RuntimeError("IDs duplicados en "+name)
    if not (len(manifest)==len(inputs)==len(integrity)==len(labels)==EXPECTED[view][0]):
        raise RuntimeError("Conteos RAW/video/label incompatibles")
    sm={r["video_id"]:r["split"] for r in splits}
    if any(s not in {"train","validation","test"} for s in sm.values()):
        raise RuntimeError("Split PoseCNN desconocido")
    if set(sm)-{r["video_id"] for r in manifest}:
        raise RuntimeError("Split sin RAW")
    if any(r["status"]!="OK" for r in integrity):
        raise RuntimeError("Integridad RAW previa incompleta")
    if len(list(p["raw"].glob("*_segmentation.npz")))!=len(manifest):
        raise RuntimeError("NPZ RAW faltantes/sobrantes")
    hashes=defaultdict(list); rows=[]; problemas=[]
    byinput={r["video_id"]:r for r in inputs}
    byintegrity={r["video_id"]:r for r in integrity}
    for k,r in enumerate(sorted(manifest,key=lambda r:r["video_id"]),1):
        vid=r["video_id"]; raw=p["raw"]/f"{vid}_segmentation.npz"; pp=p["pose"]/f"{vid}.npy"
        d,meta=R.cargar(raw)
        pts=np.load(pp,allow_pickle=False,mmap_mode="r")
        t=len(d["frame_index"])
        if (meta["view"]!=view or meta["video_id"]!=vid or pts.shape!=(t,8,3) or
            t!=int(byinput[vid]["frames_decoded"]) or t!=int(byintegrity[vid]["frames"]) or
            meta["video_sha256"]!=r["video_sha256"] or sha(raw)!=r["raw_sha256"] or
            meta["model_sha256"]!=frozen["model_sha256"] or
            not np.all(d["frame_image_shape_hw"]==[1080,1920]) or float(meta["fps"])!=30.):
            raise RuntimeError(f"Fuente incompatible: {view}/{vid}")
        # Muestra inicial y cada clase para reconstruccion, área y bbox.
        for j in sorted(set([0,len(d["instance_class_id"])//2,len(d["instance_class_id"])-1])):
            if j>=0:
                frame=int(d["instance_frame_index"][j]); R.instancia(d,j,frame)
        hashes[meta["video_sha256"]].append((vid,sm.get(vid,"unassigned")))
        pose=np.asarray(pts)
        bad=~np.isfinite(pose[:,:,2]) | (pose[:,:,2]<.5) | ~np.isfinite(pose[:,:,:2]).all(axis=2) | (pose[:,:,:2]==0).all(axis=2)
        filas={"video_id":vid,"view":view,"T":t,"fps":meta["fps"],"width":meta["width"],"height":meta["height"],
               "split":sm.get(vid,"unassigned"), "label_present":vid in {x["video_id"] for x in labels},
               "pose_shape":str(pts.shape),"missing_wrist_frames":int(bad[:,4].sum()),
               "missing_torso_pose_frames":int((bad[:,2]|bad[:,3]).sum()),
               "missing_bar_frames":int((d["frame_class_counts"][:,0]==0).sum()),
               "multiple_bar_frames":int((d["frame_class_counts"][:,0]>1).sum()),
               "missing_person_frames":int((d["frame_class_counts"][:,2]==0).sum()),
               "missing_bench_frames":int((d["frame_class_counts"][:,1]==0).sum()),
               "raw_sha256":r["raw_sha256"],"pose_sha256":sha(pp),"video_sha256":meta["video_sha256"],
               "sync_status":"frame_count_and_index_match_historical_sync_unverified"}
        rows.append(filas)
        if k%100==0: print(view,"audit",k,"/",len(manifest),flush=True)
    duplicates=[members for members in hashes.values() if len(members)>1]
    leaks=[x for x in duplicates if len({sp for _,sp in x if sp!="unassigned"})>1]
    write_csv(out/"source_audit.csv",rows,list(rows[0]))
    result={"view":view,"videos":len(rows),"frames":sum(x["T"] for x in rows),"split":dict(Counter(x["split"] for x in rows)),
            "missing_wrist_frames":sum(x["missing_wrist_frames"] for x in rows),
            "missing_torso_pose_frames":sum(x["missing_torso_pose_frames"] for x in rows),
            "raw_missing":{name:sum(x[f"missing_{name}_frames"] for x in rows) for name in ("bar","person","bench")},
            "raw_multiple_bar":sum(x["multiple_bar_frames"] for x in rows),"exact_video_duplicates":duplicates,
            "cross_split_exact_duplicates":leaks,"sync_limitation":"Frame count, frame_index and fps agree; historical alignment cannot be independently certified",
            "physical_identity":"unverified between left/right","checkpoint_sha256":frozen["model_sha256"],
            "source_integrity_summary_sha256":sha(p["base"]/"manifests/integrity_summary.json")}
    write_json(out/"source_audit.json",result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("exact_video_duplicates",)},indent=2),flush=True)
    if leaks:
        raise RuntimeError("Duplicados exactos cruzan particiones: bloquear entrenamiento")
    return rows


def seleccionar_piloto(rows):
    train=[r for r in rows if r["split"]=="train"]
    elegidos=[]
    claves=("multiple_bar_frames","missing_bar_frames","missing_person_frames",
            "missing_bench_frames","missing_wrist_frames","missing_torso_pose_frames")
    for clave in claves:
        candidatos=sorted(train,key=lambda r:(-int(r[clave]),r["video_id"]))
        for r in candidatos:
            if int(r[clave])>0 and r["video_id"] not in elegidos:
                elegidos.append(r["video_id"]); break
    for r in (min(train,key=lambda r:int(r["T"])),max(train,key=lambda r:int(r["T"]))):
        if r["video_id"] not in elegidos: elegidos.append(r["video_id"])
    for r in sorted(train,key=lambda r:r["video_id"]):
        if len(elegidos)>=10: break
        if r["video_id"] not in elegidos: elegidos.append(r["video_id"])
    return elegidos


def codigo_sha():
    return {file:sha(CODE/file) for file in FILES}


def save_sequence(path, result, meta, vid, view, hashes, config):
    arrays={k:v for k,v in result.items() if k not in ("preview_geometry","bench_reference_box","bench_stability")}
    arrays.update({"schema_version":np.asarray("bp_side_temporal_features_v1"),"video_id":np.asarray(vid),
                   "view":np.asarray(view),"feature_names":np.asarray(NAMES),
                   "quality_names":np.asarray(esquema()["feature_names"][-17:]),
                   "metadata_json":np.asarray(json.dumps({"source_video_sha256":meta["video_sha256"],
                       "source_model_sha256":meta["model_sha256"],"raw_sha256":hashes["raw_sha256"],
                       "pose_sha256":hashes["pose_sha256"],"source_view":view,
                       "physical_identity":"unverified","historical_sync":"frame-count and fps agree; historical alignment unverified",
                       "bench_stability_iou":None if not np.isfinite(result["bench_stability"]) else float(result["bench_stability"]),
                       "bench_reference_bbox_px":None if result["bench_reference_box"] is None else result["bench_reference_box"].tolist(),
                       "config":config},ensure_ascii=False,sort_keys=True))})
    temp=path.with_suffix(".npz.tmp")
    with open(temp,"wb") as f: np.savez_compressed(f,**arrays)
    os.replace(temp,path)


def verify_sequence(path, vid, view, record, frozen):
    with np.load(path,allow_pickle=False) as z:
        d={k:z[k] for k in z.files}
    meta=json.loads(str(d["metadata_json"])); t=int(record["T"])
    if (str(d["video_id"])!=vid or str(d["view"])!=view or
            str(d["schema_version"])!="bp_side_temporal_features_v1" or
            not np.array_equal(d["feature_names"],NAMES) or
            d["features"].shape!=(t,F) or d["availability"].shape!=(t,F) or d["provenance"].shape!=(t,F) or
            not np.array_equal(d["availability"],np.isfinite(d["features"])) or
            not np.array_equal(d["availability"],d["provenance"]!=0) or np.isinf(d["features"]).any() or
            not np.array_equal(d["frame_index"],np.arange(t)) or
            not np.allclose(d["timestamp_sec"],np.arange(t)/float(record["fps"]),rtol=0,atol=1e-9) or
            d["raw_instance_ids"].shape!=(t,3) or d["raw_class_counts"].shape!=(t,3) or
            not np.all((d["raw_instance_ids"]>=-1)) or
            meta["raw_sha256"]!=record["raw_sha256"] or meta["pose_sha256"]!=record["pose_sha256"] or
            meta["source_video_sha256"]!=record["video_sha256"] or
            meta["source_model_sha256"]!=frozen["raw_model_sha256"] or meta["config"]!=frozen):
        raise ValueError(f"Secuencia inconsistente: {view}/{vid}")
    a=d["availability"]; b=d["bar_state"]; r=d["relation_state"]; prof=d["profile_state"]; ids=d["raw_instance_ids"]
    if (not np.isin(b,[0,1,2,3,4]).all() or not np.isin(r,[0,1,2,3]).all() or
        np.any((b==0)&(d["raw_class_counts"][:,0]!=0)) or
        np.any((b!=0)&(d["raw_class_counts"][:,0]==0)) or
        np.any((b==3)&(ids[:,0]!=-1)) or
        np.any((b==0)&(ids[:,0]!=-1)) or
        np.any((r!=0)&(ids[:,0]<0)) or
        np.any((r!=0)&(ids[:,1]<0)) or
        np.any((r==0)&a[:,INDEX["sep_signed_h"]].astype(bool)) or
        np.any((r!=0)&~a[:,INDEX["sep_signed_h"]].astype(bool)) or
        np.any((r==3)&(d["features"][:,INDEX["extrap_ratio_d_l"]]<=2)) or
        np.any((r==2)&(d["features"][:,INDEX["extrap_ratio_d_l"]]>2)) or
        np.any((r!=0)&(d["features"][:,INDEX["extrap_ratio_d_l"]]>4)) or
        np.any((prof==2)&(ids[:,2]<0)&(d["features"][:,INDEX["q_bench_reference_used"]]!=1))):
        raise ValueError(f"Estados/geom inconsistentes: {vid}")
    dyn=a[:,INDEX["bar_vy_h_s"]].astype(bool)
    if np.any(dyn & (~np.r_[False, (b[1:]==1)&(b[:-1]==1)])) or np.any(dyn & (d["features"][:,INDEX["q_bar_local_association"]]!=1)):
        raise ValueError("Velocidad sin asociacion provisional respaldada")
    if not np.array_equal(d["raw_class_counts"],record["raw_class_counts"]):
        raise ValueError("Conteos RAW no coinciden")
    return {"T":t,"missing":int((~a.astype(bool)).sum()),"available":a.sum(axis=0).astype(int),
            "bar":dict(Counter(map(int,b))),"profile":dict(Counter(map(int,prof))),
            "relation":dict(Counter(map(int,r))),"velocity":int(dyn.sum()),
            "bench_reference":int(d["features"][:,INDEX["q_bench_reference_used"]].sum()),
            "bytes":path.stat().st_size}


def preview(view, vid, p, result, d, pose, frames):
    carpeta=p["out"]/"previews"/vid; carpeta.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(str(p["video"]/f"{vid}.mp4"))
    if not cap.isOpened(): raise RuntimeError("Video preview ilegible")
    colors={"bar":(0,255,255),"person":(0,220,0),"bench":(255,120,0)}
    try:
        for f in sorted(frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES,f)
            ok,img=cap.read()
            if not ok: raise RuntimeError("Frame preview ilegible")
            overlay=img.copy()
            a,b=map(int,d["frame_instance_offsets"][f:f+2])
            for j in range(a,b):
                i=R.instancia(d,j,f)
                roi=overlay[i["y"]:i["y"]+i["crop"].shape[0],i["x"]:i["x"]+i["crop"].shape[1]]
                mask=i["crop"]
                roi[mask]=(roi[mask]*.55+np.array(colors[i["class"]])*.45).astype(np.uint8)
                cont,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(roi,cont,-1,colors[i["class"]],2)
                box=i["box"].astype(int)
                cv2.rectangle(overlay,tuple(box[:2]),tuple(box[2:]),colors[i["class"]],2)
                cv2.putText(overlay,f'{i["class"]} #{i["index_in_frame"]} {i["conf"]:.2f}',
                            (box[0],max(25,box[1]-6)),cv2.FONT_HERSHEY_SIMPLEX,.6,colors[i["class"]],2)
            for k in range(8):
                point=G.punto_pose(pose[f],k,img.shape[1],img.shape[0])
                if point is not None:
                    pt=tuple(point.astype(int)); cv2.circle(overlay,pt,5,(255,0,255),-1)
                    cv2.putText(overlay,str(k),pt,cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            info=result["preview_geometry"].get(f)
            if info is None and result["raw_instance_ids"][f,0]>=0 and result["raw_instance_ids"][f,1]>=0:
                barra=R.instancia(d,int(result["raw_instance_ids"][f,0]),f)
                persona=R.instancia(d,int(result["raw_instance_ids"][f,1]),f)
                bench=result["raw_instance_ids"][f,2]
                bbox=(R.instancia(d,int(bench),f)["box"] if bench>=0 else result["bench_reference_box"])
                perfil,_=G.torso(persona,pose[f],bbox,barra,img.shape[1],img.shape[0],bench<0)
                if perfil is not None:
                    fit=G.pendiente(barra,img.shape[0]); extremo=G.extremo_proximal(barra,perfil["x"])
                    info={"points":perfil["points"],"endpoint":extremo,"slope":fit,
                          "relation":G.relacion(barra,perfil,fit,img.shape[0])}
            if info is not None:
                cv2.polylines(overlay,[info["points"]],False,(255,255,0),3)
                x,y,_=info["endpoint"]; cv2.circle(overlay,(round(x),round(y)),8,(0,255,255),-1)
                if info["slope"] is not None:
                    a0,b0,_=info["slope"]
                    bx=d["instance_bbox_xyxy_px"][result["raw_instance_ids"][f,0]]
                    cv2.line(overlay,(int(bx[0]),int(a0*bx[0]+b0)),(int(bx[2]),int(a0*bx[2]+b0)),(0,0,255),2)
                r=info["relation"]
                if r["state"] in (2,3):
                    xt=round(info["points"][:,0].mean()); yp=round(r["projected_y"])
                    cv2.line(overlay,(round(x),round(y)),(xt,yp),(0,0,255) if r["state"]==3 else (255,0,255),2)
                    cv2.putText(overlay,f'sep_2D={r["sep"]:+.1f}px D/L={r["ratio"]:.2f}',(30,130),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
            cv2.rectangle(overlay,(0,0),(1250,85),(0,0,0),-1)
            cv2.putText(overlay,f'{view} {vid} f={f} BAR={int(result["bar_state"][f])} torso={int(result["profile_state"][f])} rel={int(result["relation_state"][f])}',
                        (15,32),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
            cv2.putText(overlay,'RAW masks; geometry 2D != anatomical contact',
                        (15,65),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
            img=cv2.resize(img,(960,540)); overlay=cv2.resize(overlay,(960,540))
            cv2.imwrite(str(carpeta/f"frame_{f:05d}.jpg"),np.concatenate((img,overlay),axis=1))
    finally:
        cap.release()


def process(view, phase, rebuild=False):
    p=paths(view); out=p["out"]; (out/"sequences").mkdir(parents=True,exist_ok=True)
    source_config(view)
    rows=csv_rows(out/"source_audit.csv"); audit_summary=json.loads((out/"source_audit.json").read_text(encoding="utf-8"))
    if len(rows)!=EXPECTED[view][0] or audit_summary["cross_split_exact_duplicates"]:
        raise RuntimeError("Auditoria inicial incompleta o fuga real")
    if phase=="pilot":
        chosen=seleccionar_piloto(rows)
        write_csv(out/"pilot_exposure.csv",[r for r in rows if r["video_id"] in chosen],list(rows[0]))
        frozen={"version":"bp_side_temporal_features_v1", "view":view,
                "code_sha256":codigo_sha(),"schema":esquema(),"raw_model_sha256":audit_summary["checkpoint_sha256"],
                "raw_inference_sha256":sha(p["base"]/"manifests/frozen_config.json"),
                "selection":{"min_width_px":30,"wrist_max_diag":G.WRIST_MAX_DIAG,"wrist_competitor_diag":G.WRIST_COMPETITOR_DIAG,
                             "pose_conf_min":G.CONF_POSE,"no_mask_union":True,
                             "single_sleeve_fallback_dx_w":.25,"single_sleeve_fallback_dy_h":.15,
                             "single_sleeve_fallback_max_h_h":.15,"fallback_identity":"unverified_no_dynamics"},
                "profile":{"hip_shoulder_fraction":[.45,.75],"min_coverage":.7,"bench_min_width_fraction":.1,
                           "bench_reference_min_frames":5,"bench_stability_iou_min":.65},
                "projection":{"short_max_d_l":2.,"far_max_d_l":4.,"slope_max_abs":.35,"residual_max_h":.02},
                "dynamic":{"bbox_iou_min":.5,"max_dx_w":.025,"max_dy_h":.04,"max_area_ratio":1.5},
                "missing":"NaN+availability; no scaler/interpolation", "pilot_ids":chosen}
    else:
        frozen=json.loads((out/"frozen_config.json").read_text(encoding="utf-8"))
        if (frozen["code_sha256"]!=codigo_sha() or frozen["schema"]!=esquema() or
                frozen["raw_inference_sha256"]!=sha(p["base"]/"manifests/frozen_config.json")):
            raise RuntimeError("Código/esquema/config RAW cambiaron tras congelar pilotos")
        chosen=[r["video_id"] for r in rows]
    records={r["video_id"]:r for r in rows}
    state={r["video_id"]:r for r in csv_rows(out/"features_manifest.csv")} if (out/"features_manifest.csv").exists() else {}
    for number,vid in enumerate(chosen,1):
        record=records[vid]; raw=p["raw"]/f"{vid}_segmentation.npz"; pp=p["pose"]/f"{vid}.npy"
        target=out/"sequences"/f"{vid}_temporal.npz"
        try:
            d,meta=R.cargar(raw)
            pose=np.load(pp,allow_pickle=False)
            if sha(raw)!=record["raw_sha256"] or sha(pp)!=record["pose_sha256"] or meta["video_sha256"]!=record["video_sha256"]:
                raise ValueError("Fuente original cambió")
            if rebuild and target.exists():
                with np.load(target,allow_pickle=False) as previous:
                    old=json.loads(str(previous["metadata_json"]))
                    if (str(previous["schema_version"])!="bp_side_temporal_features_v1" or
                        str(previous["video_id"])!=vid or str(previous["view"])!=view or
                        old["raw_sha256"]!=record["raw_sha256"] or old["pose_sha256"]!=record["pose_sha256"]):
                        raise ValueError("No se reemplaza una secuencia ajena/inconsistente")
            if target.exists() and phase!="pilot" and not rebuild:
                summary=verify_sequence(target,vid,view,{**record,"raw_class_counts":d["frame_class_counts"]},frozen)
            else:
                result=extraer(d,meta,pose)
                if phase=="pilot" and number<=2:
                    repetido=extraer(d,meta,pose)
                    for clave in ("features","availability","provenance","bar_state","relation_state","raw_instance_ids"):
                        if not np.array_equal(result[clave],repetido[clave],equal_nan=True):
                            raise RuntimeError("Extraccion no determinista: "+clave)
                save_sequence(target,result,meta,vid,view,record,frozen)
                summary=verify_sequence(target,vid,view,{**record,"raw_class_counts":d["frame_class_counts"]},frozen)
                if phase=="pilot":
                    frames={0,len(result["frame_index"])//2,len(result["frame_index"])-1}
                    for valor in (3,2,0):
                        coincidencias=np.flatnonzero(result["bar_state"]==valor)
                        if len(coincidencias): frames.add(int(coincidencias[len(coincidencias)//2]))
                    for valor in (2,3):
                        coincidencias=np.flatnonzero(result["relation_state"]==valor)
                        if len(coincidencias): frames.add(int(coincidencias[len(coincidencias)//2]))
                    preview(view,vid,p,result,d,pose,frames)
            state[vid]={"video_id":vid,"status":"OK","T":summary["T"],"F":F,
                        "split":record["split"],"raw_sha256":record["raw_sha256"],"pose_sha256":record["pose_sha256"],
                        "video_sha256":record["video_sha256"],"out_sha256":sha(target),"missing_values":summary["missing"],
                        "bar_selected":summary["bar"].get(1,0)+summary["bar"].get(2,0),
                        "bar_corrob":summary["bar"].get(1,0),"bar_unverified":summary["bar"].get(2,0)+summary["bar"].get(4,0),
                        "bar_ambiguous":summary["bar"].get(3,0),"bar_absent":summary["bar"].get(0,0),
                        "profile":summary["profile"].get(1,0)+summary["profile"].get(2,0),
                        "profile_bench":summary["profile"].get(2,0),"direct":summary["relation"].get(1,0),
                        "ordinary":summary["relation"].get(2,0),"far":summary["relation"].get(3,0),
                        "unevaluable":summary["relation"].get(0,0),"velocity":summary["velocity"],
                        "bench_reference":summary["bench_reference"],"bytes":summary["bytes"],"error":""}
        except Exception as exc:
            state[vid]={"video_id":vid,"status":"ERROR","error":f"{type(exc).__name__}: {exc}"}
            print(view,vid,state[vid]["error"],flush=True)
        if number%10==0 or number==len(chosen):
            write_csv(out/"features_manifest.csv",sorted(state.values(),key=lambda x:x["video_id"]),
                      ["video_id","status","T","F","split","raw_sha256","pose_sha256","video_sha256","out_sha256",
                       "missing_values","bar_selected","bar_corrob","bar_unverified","bar_ambiguous","bar_absent",
                       "profile","profile_bench","direct","ordinary","far","unevaluable","velocity","bench_reference","bytes","error"])
        if number%50==0 or number==len(chosen): print(view,phase,number,"/",len(chosen),flush=True)
    if phase=="pilot":
        failures=[vid for vid in chosen if state[vid]["status"]!="OK"]
        audit_pilot={"view":view,"pilot_ids":chosen,"failures":failures,
                     "metrics":{vid:state[vid] for vid in chosen},"no_test_exposure":all(records[vid]["split"]=="train" for vid in chosen)}
        write_json(out/"pilot_audit.json",audit_pilot)
        if failures: raise RuntimeError("Piloto crítico fallido, no se congela configuración")
        write_json(out/"frozen_config.json",frozen)
    elif any(state[v]["status"]!="OK" for v in chosen):
        raise RuntimeError("Extracción incompleta: revisar manifest")


def verify(view):
    p=paths(view);out=p["out"]
    frozen=json.loads((out/"frozen_config.json").read_text(encoding="utf-8"))
    rows=csv_rows(out/"source_audit.csv")
    manifest={r["video_id"]:r for r in csv_rows(out/"features_manifest.csv")}
    splits={r["video_id"]:r["split"] for r in csv_rows(p["split"])}
    labels={r["video_id"] for r in csv_rows(p["labels"])}
    if len(rows)!=EXPECTED[view][0] or len(manifest)!=len(rows):
        raise RuntimeError("Incompleto: no se puede verificar cohortes")
    totals=Counter(); availability=np.zeros(F,dtype=np.int64); cohort=[]; errors=[]
    duplicate_groups=json.loads((out/"source_audit.json").read_text(encoding="utf-8"))["exact_video_duplicates"]
    duplicate_of={v:min(x[0] for x in group) for group in duplicate_groups for v,sp in group
                  if v!=min(x[0] for x in group) and sp in {"train","validation","test"}}
    for n,record in enumerate(rows,1):
        vid=record["video_id"]; raw=p["raw"]/f"{vid}_segmentation.npz"; pp=p["pose"]/f"{vid}.npy"
        file=out/"sequences"/f"{vid}_temporal.npz"
        try:
            d,meta=R.cargar(raw)
            if manifest[vid]["status"]!="OK" or sha(raw)!=record["raw_sha256"] or sha(pp)!=record["pose_sha256"] or sha(file)!=manifest[vid]["out_sha256"]:
                raise ValueError("SHA/manifest incompatible")
            q=verify_sequence(file,vid,view,{**record,"raw_class_counts":d["frame_class_counts"]},frozen)
            if tuple(np.load(pp,mmap_mode="r",allow_pickle=False).shape)!=(q["T"],8,3):
                raise ValueError("Keypoints desincronizados")
            totals["videos"]+=1; totals["frames"]+=q["T"];totals["missing_values"]+=q["missing"]
            availability+=q["available"]
            for k,v in q["bar"].items(): totals[f"bar_{k}"]+=v
            for k,v in q["profile"].items(): totals[f"profile_{k}"]+=v
            for k,v in q["relation"].items(): totals[f"relation_{k}"]+=v
            totals["velocity"]+=q["velocity"];totals["bench_reference"]+=q["bench_reference"]
            for name,idx in (("bar",0),("bench",1),("person",2)):
                totals[f"raw_{name}_missing"]+=int((d["frame_class_counts"][:,idx]==0).sum())
            split=splits.get(vid,"unassigned")
            reason=("exact_duplicate_same_split_of_"+duplicate_of[vid] if vid in duplicate_of else
                    ("unassigned_or_missing_label" if split not in {"train","validation","test"} or vid not in labels else ""))
            cohort.append({"video_id":vid,"view":view,"split":split,"label_present":vid in labels,
                           "pose_available":True,"raw_available":True,"features_available":True,
                           "supervised_member":reason=="","reason":reason,
                           "T":q["T"],"nan_rate":q["missing"]/(q["T"]*F),
                           "source_identity":"unverified_multiview","raw_sha256":record["raw_sha256"],
                           "pose_sha256":record["pose_sha256"],"features_sha256":sha(file)})
        except Exception as exc:
            errors.append({"video_id":vid,"error":str(exc)})
        if n%100==0:print(view,"verify",n,"/",len(rows),flush=True)
    if errors:
        write_json(out/"integrity.json",{"view":view,"status":"ERROR","errors":errors})
        raise RuntimeError(f"{len(errors)} fallos de integridad: {errors[:3]}")
    write_csv(out/f"bp_{view}_features_common_cohort.csv",cohort,list(cohort[0]))
    write_csv(out/"quality.csv",list(manifest.values()),list(manifest[next(iter(manifest))]))
    summary={"view":view,"F":F,"videos":totals["videos"],"frames":totals["frames"],
             "shapes":"(T,%d) con T original por video"%F,
             "missing_values":totals["missing_values"],"missing_percent":100*totals["missing_values"]/(totals["frames"]*F),
             "availability_by_feature":{name:float(round(a/totals["frames"],6)) for name,a in zip(NAMES,availability)},
             "bar_selected":totals["bar_1"]+totals["bar_2"],"bar_wrist_corrob":totals["bar_1"],
             "bar_selected_unverified":totals["bar_2"],"bar_unverified_unselected":totals["bar_4"],
             "bar_ambiguous":totals["bar_3"],"bar_absent":totals["bar_0"],
             "raw_missing":{"bar":totals["raw_bar_missing"],"bench":totals["raw_bench_missing"],"person":totals["raw_person_missing"]},
             "profile_pose":totals["profile_1"],"profile_bench_estimate":totals["profile_2"],
             "relation_direct_observed":totals["relation_1"],
             "relation_ordinary_projected":totals["relation_2"],"relation_far_projected":totals["relation_3"],
             "relation_unevaluable":totals["relation_0"],"velocity":totals["velocity"],
             "bench_reference_used":totals["bench_reference"],
             "splits":dict(Counter(x["split"] for x in cohort)),
             "supervised_cohort":dict(Counter(x["split"] for x in cohort if x["supervised_member"])),
             "duplicates_removed_from_supervised":duplicate_of,
             "unassigned_ids":[x["video_id"] for x in cohort if x["split"]=="unassigned"],
             "errors":errors,"pilot_ids":json.loads((out/"pilot_audit.json").read_text(encoding="utf-8"))["pilot_ids"],
             "source_audit_sha256":sha(out/"source_audit.csv"),"frozen_config_sha256":sha(out/"frozen_config.json"),
             "split_sha256":sha(p["split"]),"label_membership_source_sha256":sha(p["labels"]),
             "historical_sync":"frame counts/FPS/index matched; independent historical sync not certified",
             "physical_identity":"unverified between views"}
    write_json(out/"integrity.json",summary)
    return summary


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--view",required=True,choices=("left","right"))
    ap.add_argument("--phase",required=True,choices=("audit","pilot","all","verify"))
    ap.add_argument("--rebuild",action="store_true",help="Regenera exclusivamente secuencias derivadas tras repetir pilotos y congelar nuevo hash")
    args=ap.parse_args()
    if args.rebuild and args.phase!="all": raise ValueError("--rebuild solo corresponde a --phase all")
    if args.phase=="audit":audit(args.view)
    elif args.phase in ("pilot","all"):process(args.view,args.phase,args.rebuild)
    else:print(json.dumps(verify(args.view),indent=2,ensure_ascii=False),flush=True)


if __name__=="__main__":main()
