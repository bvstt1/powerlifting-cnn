"""Contrato experimental: lectura exclusiva de RAW y geometría sin inferencia."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SEG = ROOT / 'segmentation/bp/front'
SPEC_SHA = '735befd58856d5d4156bf0ba4b169b2151aeec73e169d46553d67860e8c61855'


def sha(path):
    d = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            d.update(chunk)
    return d.hexdigest()


def json_safe(value):
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (list, tuple)):
        return [json_safe(x) for x in value]
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def load(video, expected_hash):
    """Regresa (pose, raw, v2, selection, features, audit, preflight); caller cierra NPZ."""
    vpath = ROOT / 'dataset/bp/front' / (video+'.mp4')
    ppath = ROOT / 'keypoints/bp/front' / (video+'.npy')
    raw = np.load(SEG/'raw'/(video+'_segmentation.npz'), allow_pickle=False)
    v2 = np.load(SEG/'tracking_v2/raw'/(video+'_segmentation.npz'), allow_pickle=False)
    features = np.load(SEG/'tracking_v2/features'/(video+'_features.npz'), allow_pickle=False)
    branch = 'validation_76' if (SEG/'tracking_v3/validation_76/selections'/(video+'_bar_tracking_v3_selection.npz')).exists() else 'test_14'
    base = SEG/'tracking_v3'/branch
    selection = np.load(base/'selections'/(video+'_bar_tracking_v3_selection.npz'), allow_pickle=False)
    audit = json.loads((base/'audits'/(video+'_bar_tracking_v3_audit.json')).read_text(encoding='utf-8'))
    pose = np.load(ppath, allow_pickle=False)
    cap = cv2.VideoCapture(str(vpath))
    n, fps, w, h = int(cap.get(7)), float(cap.get(5)), int(cap.get(3)), int(cap.get(4))
    cap.release()
    meta = json.loads(str(raw['metadata_json']))
    review_path=HERE/'bp_front_v31_visual_sync_review.json'
    review=json.loads(review_path.read_text(encoding='utf-8')) if review_path.exists() else {}
    sheet_index=json.loads((HERE/'sync_sheets/index.json').read_text(encoding='utf-8')) if (HERE/'sync_sheets/index.json').exists() else {}
    listed=any(video in group for group in sheet_index.values())
    sampled_visual=(listed and review.get('status')=='SEQUENCE_AND_VISUAL_SUPPORTED_FOR_SAMPLED_FRAMES_ONLY'
                    and len(review.get('reviewed_sheets',[]))==len(sheet_index) and not review.get('sampled_alignment_failures'))
    checks = {
        'video_sha256': sha(vpath) == meta.get('video_sha256') == expected_hash,
        'frame_counts': n == len(pose) == len(raw['frame_index']) == len(v2['frame_index']) == len(selection['frame_index']),
        'pose_shape': pose.shape == (n, 6, 3),
        'fps': abs(fps-float(meta['fps'])) < 1e-4 and fps > 0,
        'resolution': np.all(raw['frame_image_shape_hw'] == (h,w)) and np.array_equal(raw['frame_image_shape_hw'],v2['frame_image_shape_hw']),
        'indices': np.array_equal(raw['frame_index'],np.arange(n)) and np.array_equal(v2['frame_index'],raw['frame_index']) and np.array_equal(selection['frame_index'],raw['frame_index']),
        'timestamps': np.allclose(raw['timestamp_sec'],np.arange(n)/fps,atol=1e-5),
        'mask_bits': np.array_equal(raw['mask_bits'],v2['mask_bits']) and np.array_equal(raw['mask_byte_offsets'],v2['mask_byte_offsets']),
        'instance_frames': np.array_equal(raw['instance_frame_index'],v2['instance_frame_index']),
        'pose_normalized': np.all((~np.isfinite(pose[:,:,:2]))|((pose[:,:,:2]>=0)&(pose[:,:,:2]<=1))),
        'confidence': np.all((~np.isfinite(pose[:,:,2]))|((pose[:,:,2]>=0)&(pose[:,:,2]<=1))),
        'spec_sha256': sha(ROOT/'segmentation/bp/front/pose_guided_tracking_v31_spec/BP_FRONT_Pose_Guided_Tracking_V31_Specification.md') == SPEC_SHA,
        'sampled_visual_sync_review':sampled_visual,
    }
    if n:
        indices = selection['selected_instance_indices']
        checks['selection_indices'] = all(int(v2['instance_frame_index'][j]) == t and v2['instance_class_name'][j]=='bar' and int(v2['instance_track_id'][j]) == int(selection['selected_v2_track_ids'][t]) for t,j in enumerate(indices) if 0 <= j < len(v2['instance_frame_index'])) and np.all((indices < len(v2['instance_frame_index'])) & (indices >= -1))
    status = 'SEQUENCE_AND_VISUAL_SUPPORTED' if all(checks.values()) else 'SYNC_BLOCKED'
    preflight = dict(video_id=video, status=status, pose_traceability='NOT_CRYPTOGRAPHICALLY_TRACEABLE',
                     checks=checks, video_sha256=expected_hash, pose_sha256=sha(ppath), raw_sha256=sha(SEG/'raw'/(video+'_segmentation.npz')),
                     fps=fps, width=w, height=h, frames=n, v3_branch=branch,
                     visual_evidence='sync_sheets/index.json: first/middle/last visually reviewed for development; no cryptographic frame-level Pose provenance',
                     orientation='ORIGINAL_1920x1080_NO_FLIP_VERIFIED_FROM_DATA_AND_SAMPLED_OVERLAYS' if all(checks.values()) else 'UNVERIFIED')
    return pose, raw, v2, selection, features, audit, preflight


def crop(raw, index):
    a,b = map(int,raw['mask_byte_offsets'][index:index+2]); x,y=map(int,raw['instance_mask_origin_xy'][index]); h,w=map(int,raw['instance_mask_shape_hw'][index])
    if h<=0 or w<=0:
        raise ValueError('empty BAR mask')
    bits=np.unpackbits(raw['mask_bits'][a:b],bitorder='little',count=h*w).reshape(h,w).astype(np.uint8)
    return x,y,bits


def full_mask(raw,index):
    x,y,part=crop(raw,index); t=int(raw['instance_frame_index'][index]); h,w=map(int,raw['frame_image_shape_hw'][t]); mask=np.zeros((h,w),dtype=bool)
    mask[y:y+part.shape[0],x:x+part.shape[1]]=part.astype(bool)
    return mask


def geometry(raw,index):
    x0,y0,mask=crop(raw,index)
    yy,xx=np.nonzero(mask); xy=np.column_stack((xx+x0,yy+y0)).astype(float)
    if len(xy)!=int(raw['instance_mask_area_px'][index]):
        raise ValueError('mask area mismatch at instance %d'%index)
    center=xy.mean(axis=0)
    cov=np.cov(xy,rowvar=False,bias=True) if len(xy)>1 else np.eye(2)
    values,vectors=np.linalg.eigh(cov); axis=vectors[:,-1]; lo,hi=np.min((xy-center)@axis),np.max((xy-center)@axis)
    angle=np.degrees(np.arctan2(axis[1],axis[0])); angle=((angle+90)%180)-90
    return dict(center=center,axis=axis,extent=(lo,hi),area=len(xy),major_axis=4*np.sqrt(max(0,values[-1])),minor_axis=4*np.sqrt(max(0,values[0])),angle=angle,
                width=float(hi-lo),crop=(x0,y0,mask))


def wrist_measure(raw,pose,index,geom,frame,conf_min,min_span):
    h,w=map(int,raw['frame_image_shape_hw'][frame]); x0,y0,part=geom['crop']; center=geom['center']; axis=geom['axis']
    wrists=[]
    for k in (5,4):
        x,y,conf=map(float,pose[frame,k]); valid=np.isfinite(x) and np.isfinite(y) and np.isfinite(conf) and conf>=conf_min and 0<=x<=1 and 0<=y<=1
        pos=np.array([x*w,y*h]) if valid else None
        wrists.append(dict(valid=bool(valid),confidence=conf if np.isfinite(conf) else None,point=pos))
    span=float(np.linalg.norm(wrists[0]['point']-wrists[1]['point'])) if all(z['valid'] for z in wrists) else None
    scale_valid=span is not None and span>min_span
    # ROI exacta sobre píxeles de máscara y muñecas (redondeadas); sin alterar máscara RAW.
    valid_points=[np.rint(z['point']).astype(int) for z in wrists if z['valid']]
    if valid_points:
        xs=[x0,x0+part.shape[1]-1]+[int(p[0]) for p in valid_points]; ys=[y0,y0+part.shape[0]-1]+[int(p[1]) for p in valid_points]
        xa=max(0,min(xs)-1); xb=min(w,max(xs)+2); ya=max(0,min(ys)-1); yb=min(h,max(ys)+2)
        roi=np.zeros((yb-ya,xb-xa),np.uint8); roi[y0-ya:y0-ya+part.shape[0],x0-xa:x0-xa+part.shape[1]]=part
        dm=cv2.distanceTransform(1-roi,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    for z in wrists:
        if not z['valid']:
            z.update(mask_distance=None,axis_distance=None,extension=None,signed_y=None)
            continue
        p=z['point']; projection=float((p-center)@axis)
        ix,iy=np.clip(np.rint(p).astype(int),[0,0],[w-1,h-1]); lo,hi=geom['extent']
        delta=p-center
        z.update(mask_distance=float(dm[iy-ya,ix-xa]), axis_distance=float(abs(axis[0]*delta[1]-axis[1]*delta[0])),
                 extension=float(max(lo-projection,0,projection-hi)), signed_y=float(center[1]-p[1]))
    return wrists,span if scale_valid else None


def write_csv(path,rows,fields):
    import csv
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(json_safe(v),ensure_ascii=False) if isinstance(v,(dict,list,tuple)) else 'NaN' if isinstance(v,(float,np.floating)) and not np.isfinite(v) else json_safe(v) for k,v in row.items() if k in fields})
