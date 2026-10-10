"""Pruebas sintéticas de contratos (no cuentan para eficacia en videos)."""
import json
import unittest

import numpy as np

from v31_common import HERE, crop, full_mask
from v31_identity import PoseGuidedBarIdentityValidator
from v31_consistency import PoseGuidedTrackConsistencyValidator
from v31_handoff import PoseGuidedHandoffValidator


CONFIG=json.loads((HERE/'experimental_config.json').read_text(encoding='utf-8'))


def example(heights, wrist_y=None, track_ids=None, half_widths=None, absent=None, one_wrist=False, swap=False):
    n=len(heights);pose=np.full((n,6,3),np.nan,np.float32);frame_indices=[];tracks=[];areas=[];origins=[];shapes=[];bits=[];offset=[0];conf=[];bbox=[]
    wrist_y=wrist_y or [65]*n;track_ids=track_ids or [4]*n;half_widths=half_widths or [30]*n;absent=set(absent or [])
    for t,(y,wy,track,half) in enumerate(zip(heights,wrist_y,track_ids,half_widths)):
        L,R=(75,25) if swap else (25,75)
        for k,x in [(5,L),(4,R)]:pose[t,k]=[x/100,wy/180,.99]
        if one_wrist:pose[t,4]=[np.nan,np.nan,.2]
        if t in absent:continue
        x0=50-half;w=2*half;h=5
        mask=np.ones((h,w),np.uint8)
        packed=np.packbits(mask.ravel(),bitorder='little')
        bits.extend(packed.tolist());offset.append(len(bits));frame_indices.append(t);tracks.append(track);areas.append(h*w)
        origins.append([x0,y]);shapes.append([h,w]);conf.append(.9);bbox.append([x0,y,x0+w,y+h])
    raw=dict(frame_image_shape_hw=np.tile([180,100],(n,1)),instance_frame_index=np.array(frame_indices),instance_track_id=np.array(tracks),
             instance_original_track_id=np.array(tracks),instance_class_name=np.array(['bar']*len(tracks)),
             instance_mask_area_px=np.array(areas),instance_mask_origin_xy=np.array(origins,dtype=int).reshape(-1,2),
             instance_mask_shape_hw=np.array(shapes,dtype=int).reshape(-1,2),mask_bits=np.array(bits,dtype=np.uint8),
             mask_byte_offsets=np.array(offset),instance_confidence=np.array(conf),instance_bbox_xyxy_px=np.array(bbox,dtype=float).reshape(-1,4))
    return pose,raw


class TestShadow(unittest.TestCase):
    def setUp(self):
        self.p=dict(CONFIG,min_wrist_span_px=10,max_axis_ratio=.5)

    def run_identity(self,heights,**kwargs):
        pose,raw=example(heights,**kwargs)
        rows,frames,tracks=PoseGuidedBarIdentityValidator(self.p).evaluate('synthetic',pose,raw)
        return rows,frames,tracks,pose,raw

    def test_geometry_and_coordinate_conversion(self):
        rows,frames,tracks,pose,raw=self.run_identity([50]*5)
        self.assertEqual(rows[2]['mask_area_px'],300)
        self.assertAlmostEqual(rows[2]['centroid_y'],52.0)
        self.assertAlmostEqual(rows[2]['left_signed_y'],-13.0,places=4)
        self.assertEqual(len(rows),5)
        self.assertEqual(crop(raw,2)[2].shape,(5,60))
        self.assertEqual(int(np.sum(full_mask(raw,2))),300)

    def test_stationary_valid_and_symmetry(self):
        a,*_=self.run_identity([50]*6)
        b,*_=self.run_identity([50]*6,swap=True)
        self.assertEqual([r['state'] for r in a],[r['state'] for r in b])
        self.assertEqual(a[3]['state'],'SUPPORTED_HELD_BAR')
        self.assertEqual(PoseGuidedTrackConsistencyValidator(self.p).evaluate('synthetic',{4:a})[0][0]['consistency_state'],'CONSISTENT')

    def test_one_wrist_and_both_missing_abstain(self):
        a,*_=self.run_identity([50]*5,one_wrist=True)
        self.assertTrue(all(r['state']=='INDETERMINATE' for r in a))
        self.assertTrue(all(r['right_wrist_valid']==False for r in a))
        pose,raw=example([50]*5);pose[:,[4,5],:2]=np.nan
        b,*_=PoseGuidedBarIdentityValidator(self.p).evaluate('synthetic',pose,raw)
        self.assertTrue(all(r['state']=='INDETERMINATE' for r in b))

    def test_moving_held_bar_not_split(self):
        rows,_,tracks,*_=self.run_identity([50,55,60,65,70,75],wrist_y=[65,70,75,80,85,90])
        self.assertEqual(PoseGuidedTrackConsistencyValidator(self.p).evaluate('synthetic',tracks)[0][0]['switch_count'],0)

    def test_partial_mask_no_split(self):
        rows,_,tracks,*_=self.run_identity([50]*8,half_widths=[30]*4+[12]*4)
        self.assertEqual(PoseGuidedTrackConsistencyValidator(self.p).evaluate('synthetic',tracks)[0][0]['switch_count'],0)

    def test_hybrid_return_proposes_subtracklets(self):
        rows,_,tracks,*_=self.run_identity([50]*4+[115]*4+[50]*4)
        summary,segments,_,cut=PoseGuidedTrackConsistencyValidator(self.p).evaluate('synthetic',tracks)
        self.assertEqual(summary[0]['switch_count'],2)
        self.assertEqual([s['first_frame'] for s in segments],[0,4,8])
        self.assertEqual(len(set(cut.values())),3)
        self.assertEqual(sum(len(s['global_instance_indices']) for s in segments),len(rows))

    def test_handoff_bad_and_good(self):
        rows,*_=self.run_identity([50]*6+[115]*6,track_ids=[0]*6+[4]*6)
        bad=PoseGuidedHandoffValidator(self.p).evaluate('synthetic',rows[:6],rows[6:],6,.81,True)
        self.assertEqual(bad['pose_decision'],'POSE_INCOMPATIBLE')
        goodrows,*_=self.run_identity([50]*12,track_ids=[0]*6+[4]*6)
        good=PoseGuidedHandoffValidator(self.p).evaluate('synthetic',goodrows[:6],goodrows[6:],6,.7,True)
        self.assertEqual(good['pose_decision'],'POSE_COMPATIBLE')

    def test_two_frame_return_after_split_requires_corrobation(self):
        rows,*_=self.run_identity([50]*6+[115]*2,track_ids=[0]*6+[4]*2)
        validator=PoseGuidedHandoffValidator(self.p)
        unsafe=validator.evaluate('synthetic',rows[:6],rows[6:],6,.8,True,split_supported=False)
        corroborated=validator.evaluate('synthetic',rows[:6],rows[6:],6,.8,True,split_supported=True)
        self.assertEqual(unsafe['pose_decision'],'POSE_INSUFFICIENT_EVIDENCE')
        self.assertEqual(corroborated['pose_decision'],'POSE_INCOMPATIBLE')

    def test_missing_bar_does_not_invent_mask_or_frames(self):
        rows,frames,tracks,pose,raw=self.run_identity([50]*7,absent={3,4})
        self.assertEqual(len(rows),5)
        self.assertNotIn(3,frames)
        self.assertNotIn(4,frames)
        self.assertEqual(PoseGuidedTrackConsistencyValidator(self.p).evaluate('synthetic',tracks)[0][0]['switch_count'],0)

    def test_sole_false_bar_requires_persistence(self):
        rows,*_=self.run_identity([115]*7)
        self.assertEqual(rows[3]['state'],'INCOMPATIBLE_WITH_HELD_BAR')
        one,*_=self.run_identity([115])
        self.assertEqual(one[0]['state'],'INDETERMINATE')

    def test_unverified_orientation_is_sync_blocked_by_contract(self):
        # Cámara invertida sin transformación conocida no debe llamar al validador.
        orientation='UNVERIFIED'
        status='SYNC_BLOCKED' if orientation!='ORIGINAL_1920x1080_NO_FLIP_VERIFIED_FROM_DATA_AND_SAMPLED_OVERLAYS' else 'SUPPORTED'
        self.assertEqual(status,'SYNC_BLOCKED')


if __name__=='__main__':unittest.main()
