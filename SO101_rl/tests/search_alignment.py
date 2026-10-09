"""Finite physical trials of small CAD registration offsets; no taught edits."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import prepare,atomic_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--offset',type=float,nargs=3,action='append',required=True)
    parser.add_argument('--pick-only',action='store_true')
    parser.add_argument('--cap-only',action='store_true',help='Diagnostic home and cap subsequence; cannot establish assembly success')
    parser.add_argument('--yaw',type=float,default=0.)
    parser.add_argument('--receiver-offset',type=float,nargs=3)
    parser.add_argument('--tilt',type=float,nargs=2,default=[0.,0.])
    parser.add_argument('--nest-offsets',type=Path)
    parser.add_argument('--pad-stiffness',type=float,default=0.)
    parser.add_argument('--finger-stiffness',type=float,default=0.)
    parser.add_argument('--tpu-stiffness',type=float,default=0.)
    parser.add_argument('--tpu-damping',type=float,default=5.)
    parser.add_argument('--capture',action='store_true')
    args=parser.parse_args();output=args.output.resolve()
    if len(args.offset)>20:raise ValueError('한 배치는 20회 이하입니다.')
    import json
    nests=json.loads(args.nest_offsets.read_text()) if args.nest_offsets else None
    ref,scene,_=prepare(output,cad_offset=args.offset[0],cad_yaw=args.yaw,receiver_offset=args.receiver_offset,cad_tilt=args.tilt,nest_offsets=nests)
    if args.tpu_stiffness:
        scene['assembly']['tpu_contact']=dict(stiffness_n_m=args.tpu_stiffness,damping_ns_m=args.tpu_damping,measured=False)
    if args.pad_stiffness or args.finger_stiffness:
        scene['pad_contact'].update(compliant_contact_stiffness=args.pad_stiffness,compliant_contact_damping=5.)
        if args.finger_stiffness:scene['pad_contact'].update(finger_contact_stiffness=args.finger_stiffness,finger_contact_damping=5.)
        atomic_json(output/'scene.json',scene)
        manifest=__import__('json').loads((output/'source_manifest.json').read_text())
        manifest['applied']['pad_contact']=scene['pad_contact']
        manifest['applied']['assembly']=scene['assembly']
        manifest['limitations'].append('Diagnostic contact compliance: provisional spring and damping constants; not measured')
        atomic_json(output/'source_manifest.json',manifest)
    if args.cap_only:
        import numpy as np
        from core import JointPath
        ref.steps=[ref.steps[0]]+ref.steps[22:]
        targets=[s['ticks'] for s in ref.steps]
        ref.joint_path=JointPath(targets[0],targets,rate=ref.rate)
        ref.ends=np.cumsum([s.duration for s in ref.joint_path.segments]);ref.duration=float(ref.ends[-1])
        for kind,part in scene['parts'].items():
            if kind=='cap':part['grasp_step']-=21;part['release_step']-=21
            else:part['grasp_step']=100;part['release_step']=101
        atomic_json(output/'diagnostic_subsequence.json',dict(component_test_only=True,steps=ref.steps))
    from isaaclab.app import AppLauncher
    launcher=AppLauncher(headless=True,device='cuda:0',enable_cameras=args.capture)
    import numpy as np
    import torch
    torch.set_num_threads(4)
    from isaac_env import BEnvironment
    results=[]
    try:
        env=BEnvironment(ref,scene,output)
        for trial,offset in enumerate(args.offset):
            if trial:env.set_cad_offset(offset,args.yaw,args.tilt)
            else:env.reset(seed=42,options={'randomized':False})
            states=[];last=-1;release_trace=[]
            for count in range(3500):
                _,_,terminated,truncated,info=env.step(np.zeros(6))
                if 19<=env.step_index<=22:release_trace.append(env.diagnostic_state())
                if env.step_index!=last:
                    states.append(env.diagnostic_state());last=env.step_index
                    atomic_json(output/'progress.json',dict(trial=trial,result=info,state=states[-1]))
                    print('STEP',trial,last+1,'grip',env.playback.ticks(env.q())['gripper'],'hold',env.playback.hold_tick,
                          'lifted',info['lifted'],'assembly',info['assembly_stage_seen'],flush=True)
                if terminated or truncated or (args.pick_only and env.step_index>=6) or (args.cap_only and env.step_index>=6):break
            else:raise RuntimeError('에피소드 검증 제한 시간 초과')
            result=dict(offset_mm=offset,yaw_deg=args.yaw,tilt_deg=args.tilt,control_steps=count+1,partial_pick_check=args.pick_only or args.cap_only,**info)
            atomic_json(output/f'trial_{trial:02d}.json',dict(result=result,states=states,release_trace=release_trace,final=env.diagnostic_state(),scene=env.cfg))
            results.append(result);atomic_json(output/'results.json',results)
            if args.capture:
                from run import capture
                capture(env,output/f'trial_{trial:02d}_assembly.png',view='assembly')
            print('TRIAL',trial,offset,'success',info['is_success'],'step',info['step_index']+1,
                  'lift',info['peak_lift_mm'],'lifted',info['lifted'],'failure',info['failure_reason'],flush=True)
        atomic_json(output/'complete.json',dict(trials=len(results)))
    except BaseException as exc:
        import traceback
        traceback.print_exc();atomic_json(output/'error.json',dict(error=str(exc)));raise
    finally:launcher.app.close(wait_for_replicator=False,skip_cleanup=True)


if __name__=='__main__':main()
