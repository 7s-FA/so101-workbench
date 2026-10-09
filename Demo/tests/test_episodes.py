import copy
import json
import tempfile
import unittest
from pathlib import Path
from arm_sim.common import load_models
from arm_sim.episodes import Recorder, Replay
from arm_sim.workcell import config, slot_centers, vertices

class EpisodesTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'episode.jsonl'; self.models=load_models()
        self.objects={'part':[0.,0.,.7,1.,0.,0.,0.]}
        self.commands={k:m['home'] for k,m in self.models.items()}
    def record(self,synthetic=False,complete=True,tracking=True):
        r=Recorder(self.path,self.models,'scene',1/120,self.objects,synthetic)
        for i in range(3):r.append(self.commands,self.commands,{k:tracking for k in self.models},self.objects)
        r.close(complete)
    def replay(self,**kwargs):return Replay(self.path,self.models,'scene',1/120,self.objects,**kwargs)
    def test_round_trip_and_final_hold(self):
        self.record(); r=self.replay()
        self.assertEqual(r.frame(0)['commands'],self.commands)
        self.assertEqual(r.frame(100),r.frame(2))
    def test_refuses_overwrite(self):
        self.record()
        with self.assertRaises(FileExistsError):self.record()
    def test_incomplete_rejected(self):
        self.record(complete=False)
        with self.assertRaises(ValueError):self.replay()
    def test_scene_change_rejected(self):
        self.record()
        with self.assertRaises(ValueError):Replay(self.path,self.models,'different',1/120,self.objects)
    def test_synthetic_label_required(self):
        self.record(synthetic=True)
        with self.assertRaises(ValueError):self.replay()
        self.replay(allow_synthetic=True)
    def test_idle_is_not_demonstration(self):
        self.record(tracking=False)
        with self.assertRaises(ValueError):self.replay()
    def test_corrupt_commands_and_frame_order_rejected(self):
        self.record(); original=[json.loads(l) for l in self.path.read_text().splitlines()]
        for field,value in [('commands',{'so101':[float('nan')]*6,'omx':self.commands['omx']}),('index',8),('commands',{'so101':[100.]*6,'omx':self.commands['omx']})]:
            records=copy.deepcopy(original);records[1][field]=value
            self.path.write_text('\n'.join(json.dumps(v) for v in records))
            with self.assertRaises(ValueError):self.replay()

class ProductTest(unittest.TestCase):
    def test_three_four_slot_trays(self):
        trays=config()['trays']
        self.assertEqual({t['part'] for t in trays},{'housing','ball','cap'})
        self.assertEqual(len(trays),3)
        for t in trays:self.assertEqual(len(set(slot_centers(t))),4)
    def test_mm_conversion_and_cap_flip(self):
        for kind,size in [('housing',(.05,.05,.023)),('ball',(.03,.03,.03)),('cap',(.05,.05,.0077))]:
            p=vertices(kind)
            for axis,expected in enumerate(size):
                self.assertAlmostEqual(max(v[axis] for v in p)-min(v[axis] for v in p),expected,delta=0.00002)
            self.assertAlmostEqual(min(v[2] for v in p),0,places=5)

if __name__=='__main__':unittest.main()
