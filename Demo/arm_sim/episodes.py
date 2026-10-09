"""Synchronized, validated simulation-only motion capture. No hardware publisher."""
import json
import math
from pathlib import Path

VERSION=1

def vector(values, length, label):
    if not isinstance(values,list) or len(values)!=length or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in values):
        raise ValueError(f'Invalid {label}')

class Recorder:
    def __init__(self,path,models,scene_signature,dt,initial_objects,synthetic=False):
        self.path=Path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.file=self.path.open('x',encoding='utf-8')
        self.count=0
        self.write({'type':'header','version':VERSION,'scene_signature':scene_signature,'dt':dt,
                    'models':models,'initial_objects':initial_objects,'units':'radians, metres, seconds',
                    'source':'synthetic_test' if synthetic else 'leader_ros',
                    'scope':'simulated joint targets and observations; no camera dataset or learned policy'})
    def write(self,value):
        self.file.write(json.dumps(value,separators=(',',':'),allow_nan=False)+'\n')
    def append(self,commands,measured,tracking,objects):
        self.write({'type':'frame','index':self.count,'commands':commands,'measured':measured,'tracking':tracking,'objects':objects})
        self.count+=1
        if self.count%120==0:self.file.flush()
    def close(self,complete):
        if not self.file.closed:
            self.write({'type':'footer','frames':self.count,'complete':complete})
            self.file.close()

class Replay:
    def __init__(self,path,models,scene_signature,dt,initial_objects,allow_synthetic=False):
        # Validate the whole episode before simulation can consume a single command.
        with Path(path).open(encoding='utf-8') as stream:
            records=[json.loads(line) for line in stream if line.strip()]
        if len(records)<3:raise ValueError('Episode has no frames or is incomplete')
        h,self.frames,footer=records[0],records[1:-1],records[-1]
        if h.get('type')!='header' or h.get('version')!=VERSION:raise ValueError('Unsupported episode format')
        if h.get('scene_signature')!=scene_signature or h.get('models')!=models or h.get('initial_objects')!=initial_objects:
            raise ValueError('Scene, assets, robot configuration or initial arrangement changed; record a new episode')
        if h.get('dt')!=dt:raise ValueError('Physics timestep mismatch')
        if h.get('source') not in ('leader_ros','synthetic_test'):raise ValueError('Unknown recording source')
        if h['source']=='synthetic_test' and not allow_synthetic:raise ValueError('Synthetic fixture; use --allow-synthetic-replay only for testing')
        if footer!={'type':'footer','frames':len(self.frames),'complete':True}:raise ValueError('Incomplete episode')
        seen_tracking=False
        for i,f in enumerate(self.frames):
            if f.get('type')!='frame' or f.get('index')!=i:raise ValueError('Missing/out-of-order frame')
            for field in ('commands','measured','tracking'):
                if set(f.get(field,{}))!=set(models):raise ValueError('Missing robot data')
            for kind,m in models.items():
                values=f['commands'][kind]
                vector(values,len(m['joints']),'command')
                vector(f['measured'][kind],len(m['joints']),'measurement')
                if not isinstance(f['tracking'][kind],bool):raise ValueError('Invalid tracking flag')
                seen_tracking |= f['tracking'][kind]
                for v,(lo,hi) in zip(values,m['limits']):
                    if not lo<=v<=hi:raise ValueError('Joint limit exceeded in episode')
                if m['mimic'] and abs(values[-1]+values[-2])>1e-5:raise ValueError('Invalid paired gripper command')
            if set(f.get('objects',{}))!=set(initial_objects):raise ValueError('Missing object poses')
            for pose in f['objects'].values():vector(pose,7,'object pose')
        if not seen_tracking:raise ValueError('No leader commands were captured; idle recording cannot demonstrate a process')
        self.header=h
    def frame(self,index):
        return self.frames[min(index,len(self.frames)-1)]
