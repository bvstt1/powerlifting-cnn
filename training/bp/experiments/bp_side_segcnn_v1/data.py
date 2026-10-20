"""Preparación V1; NPZ de Test nunca se abren, estadísticas solo de Train."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
T_TARGET = 170
VERSION_FEATURES = "bp_side_temporal_features_v1"
VIEWS = {"left": (422, 91, 91, 7), "right": (431, 93, 93, 12)}
# Fijado por definición experimental, antes de inspeccionar métricas.
RELATION_EXCLUDED = frozenset({
    "bar_proximal_x_w", "bar_proximal_y_h", "bar_proximal_thickness_h",
    "torso_profile_x_w", "torso_profile_y_h", "torso_profile_coverage",
    "sep_signed_h", "extrap_distance_w", "bar_visible_length_relation_w",
    "extrap_ratio_d_l", "sep_delta_h_s", "q_pose_torso_reliable",
    "q_profile_observed", "q_profile_bench_estimated", "q_relation_direct",
    "q_projection_ordinary", "q_projection_far", "q_relation_unevaluable",
    "q_bench_reference_used",
})
PROVENANCE_CHANNELS = ("pose", "bench_estimate_or_reference", "projection_ordinary",
                       "projection_far", "local_association")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def csv_rows(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def paths(view):
    if view not in VIEWS: raise ValueError(view)
    src = ROOT / "segmentation" / "bp" / view / "temporal_features_v1"
    return {"source":src,"cohort":src/f"bp_{view}_features_common_cohort.csv",
            "schema":src/f"bp_{view}_features_schema.json", "frozen":src/"frozen_config.json",
            "integrity":src/"integrity.json", "split":ROOT/"training/data/bp"/view/"posecnn/split.csv",
            "labels":ROOT/"etiquetado/bp"/f"bp_{view}.csv",
            "historical":ROOT/"training/models/bp"/view/"posecnn/runs/baseline_t170_v1",
            "out":ROOT/"training/bp/experiments"/f"bp_{view}_01_segcnn_v1"}


def cohort(view):
    p=paths(view)
    rows=csv_rows(p["cohort"]); schema=json.loads(p["schema"].read_text(encoding="utf-8"))
    audit=json.loads(p["integrity"].read_text(encoding="utf-8"))
    if (len(rows)!=audit["videos"] or len({r["video_id"] for r in rows})!=len(rows)
            or schema.get("version")!=VERSION_FEATURES or schema["F"]!=62):
        raise RuntimeError("Cohorte/esquema de fuente inválidos")
    names=schema["feature_names"]
    if len(names)!=62 or len(set(names))!=62 or not RELATION_EXCLUDED.issubset(names):
        raise RuntimeError("Nombres de canales incompletos")
    splits={r["video_id"]:r for r in csv_rows(p["split"])}
    labels={r["video_id"]:r for r in csv_rows(p["labels"])}
    historical_count=sum(VIEWS[view][:3])+(1 if view=="right" else 0)
    if len(labels)!=len(rows) or len(splits)!=historical_count:
        raise RuntimeError("CSV labels/split cambiado")
    groups={role:[] for role in ("train","validation")}
    counts=Counter(); excluded=[]
    for r in rows:
        vid=r["video_id"]; role=r["split"]
        counts[role]+=1
        if role not in ("train","validation","test","unassigned") or r["label_present"]!="True":
            raise RuntimeError("Split/label inválido")
        label=labels[vid]["label"]
        if label not in {"0","1"} or labels[vid]["resultado"].strip().lower()!=({"0":"invalido","1":"valido"}[label]):
            raise RuntimeError("Etiqueta contradictoria: "+vid)
        if role=="unassigned":
            if r["supervised_member"]!="False" or vid in splits: raise RuntimeError("Sin split incorporado")
            excluded.append((vid,r["reason"]));continue
        if vid not in splits or splits[vid]["split"]!=role or splits[vid]["label"]!=label:
            raise RuntimeError("Split/label histórico incompatible: "+vid)
        if r["supervised_member"]=="False":
            if view!="right" or vid!="bp_354" or r["reason"]!="exact_duplicate_same_split_of_bp_027" or role!="train":
                raise RuntimeError("Exclusión no documentada: "+vid)
            excluded.append((vid,r["reason"]));continue
        if r["supervised_member"]!="True" or r["features_available"]!="True" or r["pose_available"]!="True" or r["raw_available"]!="True":
            raise RuntimeError("Membresía incompleta: "+vid)
        if role in groups:groups[role].append(r)
    if tuple(len(groups[role]) for role in groups)!=(VIEWS[view][0],VIEWS[view][1]) or counts["test"]!=VIEWS[view][2] or counts["unassigned"]!=VIEWS[view][3]:
        raise RuntimeError("Conteos de cohortes alterados")
    if set(r["video_id"] for r in groups["train"]) & set(r["video_id"] for r in groups["validation"]):
        raise RuntimeError("Solapamiento Train/Validation")
    # Se leen SOLO filas del split Test; ningún NPZ Test se toca.
    source_hashes={name:sha(p[name]) for name in ("cohort","schema","frozen","integrity","split","labels")}
    historical_ids={role:{v for v,r in splits.items() if r["split"]==role} for role in ("train","validation","test")}
    compat={"historical_split_matches_current_exclusions":
            {role:sorted(historical_ids[role]-{r["video_id"] for r in groups.get(role,[])}) if role!="test" else []
             for role in historical_ids},
            "pose_checkpoint_path":str(p["historical"]/"best_model.pth"),
            "pose_checkpoint_exists":(p["historical"]/"best_model.pth").exists(),
            "right_historical_train_includes_bp_354":view=="right" and "bp_354" in historical_ids["train"]}
    return groups, names, p, {"source_sha256":source_hashes,"excluded":excluded,"counts":dict(counts),
                              "pose_compatibility":compat,"test_npz_opened":0}


def resample(z, names):
    """Grid PoseCNN; nunca interpola ausencias/estados ni inventa tracking."""
    x=z["features"]; a=z["availability"].astype(bool); prov=z["provenance"]
    bar=z["bar_state"]; profile=z["profile_state"]; relation=z["relation_state"]
    counts=z["raw_class_counts"]
    t,f=x.shape
    if (f!=len(names) or t<2 or a.shape!=x.shape or prov.shape!=x.shape or
        not np.array_equal(z["frame_index"],np.arange(t)) or
        not np.array_equal(z["feature_names"],names) or
        not np.array_equal(a,np.isfinite(x)) or not np.array_equal(a,prov!=0) or
        counts.shape!=(t,3) or z["raw_instance_ids"].shape!=(t,3)):
        raise ValueError("Contrato temporal original inválido")
    grid=np.linspace(0,t-1,T_TARGET,dtype=np.float64)
    near=np.floor(grid+.5).astype(int)
    y=np.full((T_TARGET,f),np.nan,np.float32)
    origin=np.zeros((T_TARGET,f),np.uint8)
    quality={i for i,n in enumerate(names) if n.startswith("q_") or n in ("bar_candidates_raw","bar_gap_frames","bar_ambiguity_run_frames")}
    relational={i for i,n in enumerate(names) if n in ("sep_signed_h","extrap_distance_w",
                "bar_visible_length_relation_w","extrap_ratio_d_l","sep_delta_h_s")}
    barfeat={i for i,n in enumerate(names) if (n.startswith("bar_") or n=="wrist_bar_distance_diag")
             and i not in quality}
    temporal={i for i,n in enumerate(names) if n in ("bar_dx_w","bar_dy_h","bar_vx_w_s",
              "bar_vy_h_s","bar_slope_delta_s","sep_delta_h_s")}
    profilefeat={i for i,n in enumerate(names) if n.startswith("torso_profile_")}
    for k,pos in enumerate(grid):
        lo,hi=int(np.floor(pos)),int(np.ceil(pos));nr=near[k]
        if lo==hi:
            y[k]=x[lo];origin[k]=prov[lo];continue
        w=pos-lo
        for j in range(f):
            if j in quality:
                y[k,j]=x[nr,j];origin[k,j]=prov[nr,j];continue
            if not (a[lo,j] and a[hi,j]) or prov[lo,j]!=prov[hi,j]:continue
            if j in profilefeat and (profile[lo]==0 or profile[lo]!=profile[hi]):continue
            if j in relational and (relation[lo]==0 or relation[lo]!=relation[hi] or profile[lo]!=profile[hi]):continue
            if j in barfeat or j in relational:
                if bar[lo]!=bar[hi] or bar[lo] not in (1,2):continue
                if bar[lo]==2: # Identidad incierta: hold del frame original próximo, sin dinámica.
                    if j in temporal:continue
                    y[k,j]=x[nr,j];origin[k,j]=prov[nr,j];continue
                if x[hi,names.index("q_bar_local_association")]!=1:continue
            if j in temporal:
                if bar[lo]!=1 or bar[hi]!=1 or x[lo,names.index("q_bar_local_association")]!=1 or x[hi,names.index("q_bar_local_association")]!=1:continue
            if j in relational and relation[lo] in (2,3):
                # Ambos extremos ya son proyecciones del mismo tipo. Nunca se
                # transforma una proyección en observación directa.
                pass
            y[k,j]=(1-w)*x[lo,j]+w*x[hi,j];origin[k,j]=prov[lo,j]
    if not np.array_equal(np.isfinite(y),origin!=0) or np.isinf(y).any():
        raise ValueError("Remuestreo inventó disponibilidad")
    return y,np.isfinite(y).astype(np.uint8),origin


def load_development(groups,names,p):
    data={}; hashes={}
    for role in ("train","validation"):
        xx=[];mm=[];pp=[];yy=[];ids=[]
        for r in groups[role]:
            vid=r["video_id"]; seq=p["source"]/"sequences"/f"{vid}_temporal.npz"
            if not seq.is_file() or sha(seq)!=r["features_sha256"]:raise RuntimeError("Hash secuencia alterado: "+vid)
            with np.load(seq,allow_pickle=False) as z:
                if (str(z["schema_version"])!=VERSION_FEATURES or
                        z["features"].shape!=(int(r["T"]),62) or
                        str(z["view"])!=p["source"].parent.name or str(z["video_id"])!=vid):
                    raise ValueError("Vista/T alterados: "+vid)
                v,m,q=resample(z,names)
            xx.append(v);mm.append(m);pp.append(q);ids.append(vid);yy.append(int(r["label"] if "label" in r else -1))
            hashes[vid]=r["features_sha256"]
        # El CSV de cohorte no transporta labels: sólo los miembros Train/Val.
        original={row["video_id"]:int(row["label"]) for row in csv_rows(p["labels"]) if row["video_id"] in set(ids)}
        yy=np.asarray([original[vid] for vid in ids],dtype=np.int64)
        if set(yy)!={0,1}:raise RuntimeError("Clase faltante "+role)
        data[role]=(np.stack(xx),np.stack(mm),np.stack(pp),yy,ids)
    return data,hashes


def prepare(data,names,variant):
    if variant not in ("full","sin_relacion"):raise ValueError(variant)
    xtr,atr,ptr=data["train"][:3]
    candidates=[i for i,n in enumerate(names) if variant=="full" or n not in RELATION_EXCLUDED]
    feat_constant=[];selected=[];statistics={}
    for i in candidates:
        obs=xtr[:,:,i][atr[:,:,i].astype(bool)].astype(np.float64)
        if len(obs)==0 or obs.max()-obs.min()<=1e-7:
            feat_constant.append(names[i]);continue
        selected.append(i)
        statistics[names[i]]={"median":float(np.median(obs)),"mean":float(obs.mean()),
                              "std":max(float(obs.std()),1e-6),"observed_train_values":len(obs)}
    if not selected:raise ValueError("Todas las features constantes en Train")
    sources=(2,(3,6),4,5,7)
    raw_names=["value:"+names[i] for i in selected]+["available:"+names[i] for i in selected]
    raw_names += ["origin:"+tag+":"+names[i] for tag in PROVENANCE_CHANNELS for i in selected]
    masks=[];arrays={}
    for role in ("train","validation"):
        x,a,pr,y,ids=data[role]
        z=x[:,:,selected]; valid=a[:,:,selected].astype(bool);orig=pr[:,:,selected]
        med=np.array([statistics[names[i]]["median"] for i in selected],np.float32)
        mean=np.array([statistics[names[i]]["mean"] for i in selected],np.float32)
        std=np.array([statistics[names[i]]["std"] for i in selected],np.float32)
        numeric=(np.where(valid,z,med[None,None,:])-mean[None,None,:])/std[None,None,:]
        bits=[np.isin(orig,s).astype(np.float32) for s in sources]
        inp=np.concatenate((numeric,valid.astype(np.float32),*bits),axis=2).astype(np.float32)
        if not np.isfinite(inp).all():raise RuntimeError("No finitos en "+role)
        arrays[role]=(inp,y,ids)
    # También se detectan constantes de máscaras/procedencia solo en Train.
    train_in=arrays["train"][0]
    keep=np.flatnonzero(np.ptp(train_in,axis=(0,1))>1e-7)
    constants=[n for j,n in enumerate(raw_names) if j not in set(keep.tolist())]
    if len(keep)==0:raise RuntimeError("Entrada sin canales")
    arrays={role:(np.ascontiguousarray(v[:,:,keep].transpose(0,2,1)),y,ids)
            for role,(v,y,ids) in arrays.items()}
    contract={"variant":variant,"original_F":len(names),"excluded_relation_predefined":sorted(RELATION_EXCLUDED),
              "excluded_constant_features_train":feat_constant,"excluded_constant_input_channels_train":constants,
              "selected_feature_names":[names[i] for i in selected],"selected_feature_indices":selected,
              "input_channel_names":[raw_names[j] for j in keep],"input_channel_indices_before_constant_filter":keep.tolist(),
              "C":len(keep),"train_statistics":statistics,
              "provenance_encoding":"per feature one-hot pose(2),bench(3|6),projection(4),far(5),local(7); SEG(1) and QUALITY(8) implicit in availability",
              "missing_policy":"Train median imputation; Train mean/std on observed values; availability + provenance retained",
              "test_npz_opened":0}
    return arrays,contract
