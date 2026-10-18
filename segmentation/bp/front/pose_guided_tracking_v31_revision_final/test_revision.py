"""Pruebas sintéticas independientes de videos; no sirven como referencia visual."""
import json
import unittest
from pathlib import Path

from v31_revision_identity import PoseGuidedCompetitorResolver
from v31_revision_handoff import PoseGuidedHandoffContext

HERE=Path(__file__).resolve().parent
REV=json.loads((HERE/'experimental_revision_config.json').read_text(encoding='utf-8'))
BASE=json.loads((HERE.parent/'pose_guided_tracking_v31_prototype/experimental_config.json').read_text(encoding='utf-8'))


def bar(t,track,index,dl=55,dr=65,el=40,er=45,q=-45,state='INDETERMINATE',reason='multiple supported BAR candidates without calibrated main-bar margin',sid=None):
    return dict(video_id='synthetic',frame_index=str(t),global_instance_index=str(index),v2_track_id=str(track),extractor_track_id=str(track),
        subtracklet_id=sid or '%d:0'%track,state=state,reason=reason,evidence_quality='BOTH_RELIABLE',
        wrist_span_px='780',left_wrist_mask_distance=str(dl),right_wrist_mask_distance=str(dr),
        left_wrist_extension=str(el),right_wrist_extension=str(er),left_signed_y=str(q),right_signed_y=str(q-10),
        left_wrist_axis_distance='40',right_wrist_axis_distance='48')


class TestRevision(unittest.TestCase):
    def resolve(self,n,**competitor):
        frames={t:[bar(t,0,t),bar(t,1,t+1000,**competitor)] for t in range(n)}
        report=PoseGuidedCompetitorResolver(REV).resolve(frames)
        return frames,report

    def test_bilateral_clear_margin_over_three_frames(self):
        frames,report=self.resolve(5,dl=400,dr=800,el=390,er=790)
        self.assertEqual(len(report),5)
        self.assertTrue(all(r[0]['state']=='SUPPORTED_HELD_BAR' and r[1]['state']=='INDETERMINATE' for r in frames.values()))

    def test_two_real_ambiguities_remain(self):
        frames,report=self.resolve(5,dl=65,dr=75,el=45,er=50)
        self.assertEqual(report,{})
        self.assertTrue(all(r[0]['state']=='INDETERMINATE' for r in frames.values()))

    def test_distant_competitor_axis_and_q_not_enough(self):
        frames,report=self.resolve(4,dl=900,dr=1200,el=890,er=1190,q=-43)
        self.assertEqual(len(report),4)
        self.assertTrue(all(r[0]['state']=='SUPPORTED_HELD_BAR' for r in frames.values()))

    def test_competitor_close_to_one_wrist_cannot_be_dominated(self):
        frames,report=self.resolve(4,dl=60,dr=1100,el=45,er=1090)
        self.assertEqual(report,{})

    def test_partial_mask_winner_abstains(self):
        frames={t:[bar(t,0,t,dl=400,dr=55,el=390,er=40),bar(t,1,t+100,dl=800,dr=500,el=790,er=490)] for t in range(4)}
        self.assertEqual(PoseGuidedCompetitorResolver(REV).resolve(frames),{})

    def test_gap_requires_new_consecutive_support(self):
        frames={t:[bar(t,0,t),bar(t,1,t+100,dl=400,dr=800,el=390,er=790)] for t in [0,1,4,5,6]}
        self.assertEqual(set(PoseGuidedCompetitorResolver(REV).resolve(frames)),{4,5,6})

    def test_handoff_context_requires_individual_endpoint(self):
        side0=[bar(i,0,i,state='SUPPORTED_HELD_BAR',reason='bilateral temporal evidence') for i in range(4)]
        side1=[bar(i,1,i+10,state='SUPPORTED_HELD_BAR',reason='bilateral temporal evidence') for i in range(4,8)]
        side1[0]['state']='INDETERMINATE';side1[0]['reason']='incomplete bilateral/temporal evidence'
        allrows={r['frame_index']:[r] for r in side0+side1}
        edge=dict(video_id='synthetic',source_track='0',target_track='1',source_subtracklet='0:0',target_subtracklet='1:0',
                  transition_frame='4',edge_id='0:0>1:0@4',selected_by_v3='True',pose_decision='POSE_COMPATIBLE',v3_geometry_accepted='True')
        trajectory={3:dict(selected_global_instance_index='3',v3_baseline_instance_index='3'),
                    4:dict(selected_global_instance_index='-1',v3_baseline_instance_index='14')}
        promoted,notes=PoseGuidedHandoffContext(BASE,REV).corroborate('synthetic',[edge],None,{'0':side0,'1':side1},
                                                                       {int(k):v for k,v in allrows.items()},{'0':0,'1':0},trajectory)
        self.assertEqual(set(promoted),{4})
        self.assertEqual(notes[0]['eligible'],True)

    def test_handoff_false_or_insufficient_never_promoted(self):
        rows=[bar(i,0,i,state='SUPPORTED_HELD_BAR') for i in range(5)]
        target=bar(5,1,9,state='INDETERMINATE')
        data={i:[r] for i,r in enumerate(rows)};data[5]=[target]
        old={5:dict(selected_global_instance_index='-1',v3_baseline_instance_index='9')}
        for pose in ('POSE_INCOMPATIBLE','POSE_INSUFFICIENT_EVIDENCE'):
            e=dict(selected_by_v3='True',pose_decision=pose,v3_geometry_accepted='True',transition_frame='5',source_subtracklet='0:0',target_subtracklet='1:0',edge_id='x')
            x,_=PoseGuidedHandoffContext(BASE,REV).corroborate('synthetic',[e],None,{'0':rows,'1':[target]},data,{'0':0,'1':0},old)
            self.assertEqual(x,{})

    def test_hybrid_return_blocks_contextual_promotion(self):
        prior=[bar(i,0,i,state='SUPPORTED_HELD_BAR') for i in range(6)]
        endpoint=bar(3,0,3,state='INDETERMINATE')
        prior[3]=endpoint
        data={i:[r] for i,r in enumerate(prior)}
        e=dict(selected_by_v3='True',pose_decision='POSE_COMPATIBLE',v3_geometry_accepted='True',transition_frame='3',
               source_subtracklet='0:0',target_subtracklet='0:0',edge_id='x')
        x,_=PoseGuidedHandoffContext(BASE,REV).corroborate('synthetic',[e],None,{'0':prior},data,{'0':2},
                  {3:dict(selected_global_instance_index='-1',v3_baseline_instance_index='3')})
        self.assertEqual(x,{})


if __name__=='__main__':unittest.main()
