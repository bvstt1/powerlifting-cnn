"""Muestras RGB/pose/RAW para revisión humana offline del alineamiento."""
import json
from pathlib import Path

import cv2
import numpy as np

from v31_common import HERE,ROOT,SEG,full_mask


def main():
    manifest=json.loads((HERE/'bp_front_v31_source_snapshot_before.json').read_text(encoding='utf-8'))
    ids=sorted(set(manifest['validation76_ids'])|set(manifest['prototype14_ids']))
    cards=[]
    for video in ids:
        pose=np.load(ROOT/'keypoints/bp/front'/(video+'.npy'),allow_pickle=False)
        with np.load(SEG/'tracking_v2/raw'/(video+'_segmentation.npz'),allow_pickle=False) as raw:
            cap=cv2.VideoCapture(str(ROOT/'dataset/bp/front'/(video+'.mp4')))
            sample=[]
            for t in sorted(set([0,len(pose)//2,len(pose)-1])):
                cap.set(cv2.CAP_PROP_POS_FRAMES,t);ok,img=cap.read()
                if not ok or img.shape[:2]!=tuple(raw['frame_image_shape_hw'][t]):raise ValueError(video+' frame '+str(t))
                h,w=img.shape[:2];img=cv2.resize(img,(480,270))
                start,end=map(int,raw['frame_instance_offsets'][t:t+2])
                bars=[i for i in range(start,end) if raw['instance_class_name'][i]=='bar']
                for j,idx in enumerate(bars):
                    mask=cv2.resize(full_mask(raw,idx).astype(np.uint8),(480,270),interpolation=cv2.INTER_NEAREST).astype(bool)
                    color=np.array([(0,180,255),(255,80,0),(0,255,30)][j%3])
                    img[mask]=(img[mask]*.5+color*.5).astype(np.uint8)
                for k,col in [(5,(255,0,255)),(4,(0,0,255))]:
                    x,y,c=pose[t,k]
                    if np.isfinite(x) and np.isfinite(y):cv2.circle(img,(int(x*480),int(y*270)),5,col,-1)
                cv2.putText(img,'%s frame=%d BAR=%d'%(video,t,len(bars)),(5,20),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2)
                sample.append(img)
            cap.release()
            cards.append((video,cv2.vconcat(sample)))
    root=HERE/'sync_sheets';root.mkdir(exist_ok=True)
    for start in range(0,len(cards),8):
        group=cards[start:start+8]
        if len(group)%2:group.append(('',np.zeros_like(group[0][1])))
        paired=[cv2.hconcat([group[i][1],group[i+1][1]]) for i in range(0,len(group),2)]
        path=root/('sheet_%02d.jpg'%(start//8+1))
        if not cv2.imwrite(str(path),cv2.vconcat(paired)):raise IOError(path)
    (root/'index.json').write_text(json.dumps({str(i//8+1):[v for v,_ in cards[i:i+8]] for i in range(0,len(cards),8)},indent=2),encoding='utf-8')
    print('sheets',len(list(root.glob('*.jpg'))),'videos',len(cards))


if __name__=='__main__':main()
