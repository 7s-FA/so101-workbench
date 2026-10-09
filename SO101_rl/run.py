#!/usr/bin/env python3
"""Finite, headless B-only simulation/learning. Never connects to real devices."""
import argparse
import os
from datetime import datetime
from pathlib import Path
import time
from core import ROOT, EPISODE, prepare, atomic_json, read_json

def mark_complete():
    if os.environ.get('SO101_RL_STATUS'):
        Path(os.environ['SO101_RL_STATUS']).write_text('complete')

def evaluate(env, policy, seeds, randomized, *, record=False):
    results=[]
    for seed in seeds:
        observation,_=env.reset(seed=seed,options={'randomized':randomized})
        total=0.;count=0;checkpoints=[];last_index=0
        while True:
            action=env.action_space.sample()*0 if policy is None else policy.predict(observation,deterministic=True)[0]
            observation,reward,terminated,truncated,info=env.step(action)
            total+=reward;count+=1
            # Sample near the end of a taught segment, before the next begins.
            index=info['step_index']
            if record and (env.ref.ends[index]-env.time<=env.control_dt or terminated or truncated) and (index!=last_index or terminated or truncated):
                snapshot=env.diagnostic_state();checkpoints.append(snapshot);last_index=index
                if index in (4,5,8,16,17,20,24,25,28,31):
                    capture(env,env.output/f'step_{index:02d}.png',view='grasp' if index in (4,5,16,17,24,25) else 'wide')
            if terminated or truncated:break
        row=dict(seed=seed,randomized=randomized,reward=total,control_steps=count,**info)
        results.append(row)
        if record:atomic_json(env.output/f'checkpoints_{seed}.json',checkpoints)
        print('B_EVAL', 'baseline' if policy is None else 'policy', row,flush=True)
    return results

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['prepare','smoke','replay','train','evaluate','view'],default='prepare')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs'/datetime.now().strftime('b_%Y%m%d_%H%M%S'))
    parser.add_argument('--episode',type=Path,default=EPISODE)
    parser.add_argument('--jig-pose',type=float,nargs=3,metavar=('X_MM','Y_MM','YAW_DEG'),
                        help='시뮬레이션 운반 지그 위치; 같은 좌표로 OpenCV 경로 보정')
    parser.add_argument('--cad-offset',type=float,nargs=3,metavar=('DX_MM','DY_MM','DZ_MM'),help='원본 경로를 유지하는 시뮬 CAD 정합 보정')
    parser.add_argument('--cad-yaw',type=float,help='시뮬 CAD 정합 방향 보정(도)')
    parser.add_argument('--receiver-offset',type=float,nargs=3,help='시뮬 리니어 팔레트 배치 보정(mm)')
    parser.add_argument('--cad-tilt',type=float,nargs=2,metavar=('ROLL_DEG','PITCH_DEG'),help='시뮬 운반 지그 CAD 기울기 보정')
    parser.add_argument('--rate',type=float,choices=[250.,300.,350.],default=350.)
    parser.add_argument('--steps',type=int,default=2048)
    parser.add_argument('--rollout',type=int,default=256)
    parser.add_argument('--eval-episodes',type=int,default=1)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--capture',action='store_true')
    parser.add_argument('--viewer-smoke',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--viewer-capture',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--viewer-jig-render',action='store_true',help=argparse.SUPPRESS)
    # Parse prepare without launching Kit, so provenance/geometry remains cheap.
    known,_=parser.parse_known_args()
    if known.mode=='prepare':
        args=parser.parse_args()
        ref,scene,report=prepare(args.output,args.episode,args.rate,jig_pose=args.jig_pose,cad_offset=args.cad_offset,cad_yaw=args.cad_yaw,receiver_offset=args.receiver_offset,cad_tilt=args.cad_tilt)
        print(f'B_PREPARED {args.output} | {len(ref.steps)} steps | {ref.duration:.2f}s | {len(scene["static"])} static shapes')
        print('B_MASS_G',{k:round(v['mass_kg']*1000,3) for k,v in scene['parts'].items()})
        mark_complete()
        return
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(headless=known.mode!='view',device='cuda:0')
    args=parser.parse_args()
    if not args.headless and args.mode!='view':parser.error('학습·자동 평가는 headless 전용입니다.')
    if args.mode=='view' and args.headless:parser.error('view는 화면 표시 모드입니다.')
    if args.steps<1 or args.rollout<2 or not 1<=args.eval_episodes<=10:parser.error('학습/평가 횟수 범위를 확인하세요.')
    if args.mode=='evaluate' and args.checkpoint is None:parser.error('--checkpoint가 필요합니다.')
    if args.mode!='evaluate' and args.checkpoint:parser.error('체크포인트는 evaluate에서만 사용합니다.')
    ref,scene,manifest=prepare(args.output,args.episode,args.rate,jig_pose=args.jig_pose,cad_offset=args.cad_offset,cad_yaw=args.cad_yaw,receiver_offset=args.receiver_offset,cad_tilt=args.cad_tilt)
    args.enable_cameras=args.capture or args.mode=='view'
    args.kit_args=(args.kit_args or '')+' --/log/outputStreamLevel=error'
    if args.mode=='view':
        args.kit_args+=' --/app/font/customFontPath=/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf --/app/font/customRegionFiles=['+str(ROOT/'assets/korean_ui.txt')+'] --/app/font/overresolutionSize=66'
    launcher=AppLauncher(args)
    env=None;started=time.monotonic()
    try:
        import numpy as np
        import torch
        from isaac_env import BEnvironment
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.monitor import Monitor
        torch.set_num_threads(4)
        env=BEnvironment(ref,scene,args.output)
        env.reset(seed=args.seed,options={'randomized':False})
        if args.mode=='view':
            from viewer import run_viewer
            run_viewer(launcher.app,env,smoke=args.viewer_smoke,capture_only=args.viewer_capture,jig_render=args.viewer_jig_render)
            mark_complete()
            return
        if args.capture:
            capture(env,args.output/'scene.png')
        seeds=list(range(args.seed+10000,args.seed+10000+args.eval_episodes))
        result={'scope':'B assembly on linear pallet, rigid TPU and provisional magnets','hardware_commands':False,
                'engine':'Isaac Sim / PhysX','episode':ref.document['name'],
                'completed_training_steps':0,'real_deployment_ready':False,'lighting':env.lighting}
        if args.mode=='smoke':
            for _ in range(min(args.steps,256)):
                obs,reward,terminated,truncated,info=env.step(np.zeros(6))
                if terminated or truncated:break
            result.update(smoke=info,smoke_steps=_+1)
        else:
            result['baseline_nominal']=evaluate(env,None,seeds,False,record=args.mode=='replay' and args.capture)
            result['baseline_randomized']=evaluate(env,None,seeds,True)
            atomic_json(args.output/'result.json',result)
            if args.mode=='train':
                class Progress(BaseCallback):
                    def _on_step(self):
                        if self.num_timesteps%256==0:
                            print('B_TRAIN',self.num_timesteps,flush=True)
                        return True
                model=PPO('MlpPolicy',Monitor(env,str(args.output/'train_monitor.csv')),
                          n_steps=args.rollout,batch_size=min(64,args.rollout),n_epochs=5,
                          learning_rate=3e-4,gamma=.995,gae_lambda=.95,ent_coef=.001,
                          policy_kwargs={'net_arch':dict(pi=[64,64],vf=[64,64])},
                          seed=args.seed,device='cpu',verbose=0)
                model.learn(total_timesteps=args.steps,callback=Progress())
                model.save(args.output/'policy')
                atomic_json(args.output/'policy_contract.json',{'sources':manifest['sources'],
                    'rate_ticks_s':args.rate,'observation':'q6, qvel6, nominal6, previous_action6, sin/cos phase',
                    'action':'bounded residual radians [2,2,2,2,2,3] deg; position targets',
                    'hardware_deployment':False})
                result['completed_training_steps']=model.num_timesteps
            elif args.mode=='evaluate':
                contract=read_json(args.checkpoint.parent/'policy_contract.json')
                if contract['sources']!=manifest['sources'] or contract['rate_ticks_s']!=args.rate:
                    raise ValueError('체크포인트의 코드·보정·물성·속도와 현재 원본이 다릅니다.')
                model=PPO.load(args.checkpoint,device='cpu')
            if args.mode in ('train','evaluate'):
                result['policy_nominal']=evaluate(env,model,seeds,False)
                result['policy_randomized']=evaluate(env,model,seeds,True)
        result['elapsed_seconds']=time.monotonic()-started
        atomic_json(args.output/'result.json',result)
        # Keep an inspectable USD with the same real contacts, not a video-only scene.
        env.sim.stage.Export(str(args.output/'scene.usda'))
        print('B_DONE',args.output,flush=True)
        mark_complete()
    except BaseException as exc:
        import traceback
        traceback.print_exc()
        atomic_json(args.output/'error.json',{'type':type(exc).__name__,'error':str(exc),
                                             'elapsed_seconds':time.monotonic()-started})
        raise
    finally:
        if env is not None:env.close()
        launcher.app.close(wait_for_replicator=False,skip_cleanup=True)

def capture(env,path,view='wide'):
    import numpy as np
    from scipy.spatial.transform import Rotation
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension('isaacsim.sensors.camera')
    from isaacsim.sensors.camera import Camera
    from PIL import Image
    eye=np.array([1.0,-1.15,.95] if view=='wide' else [.57,-.60,.38])
    target=np.array([.17,.05,.12] if view=='wide' else [.22,-.14,.055])
    if view=='assembly':
        target=np.array(env.cfg['assembly']['target_poses_mm']['housing'])[:3,3]/1000+[0,0,.018]
        eye=target+[.20,-.25,.14]
    back=eye-target;back/=np.linalg.norm(back)
    right=np.cross([0,0,1],back);right/=np.linalg.norm(right)
    rotation=np.column_stack((right,np.cross(back,right),back))
    cameras=getattr(env,'capture_cameras',{})
    camera=cameras.get(view)
    if camera is None:
        camera=Camera('/World/Camera_'+view,position=eye,resolution=(1100,850))
        camera.initialize();cameras[view]=camera;env.capture_cameras=cameras
    camera.set_world_pose(eye,Rotation.from_matrix(rotation).as_quat()[[3,0,1,2]],camera_axes='usd')
    from pxr import UsdGeom
    UsdGeom.Camera(camera.prim).CreateFocalLengthAttr(24.)
    camera.set_clipping_range(.005,10.)
    for _ in range(30):env.render()
    rgba=camera.get_rgba()
    if rgba is None or not rgba.size:raise RuntimeError('Isaac Sim 이미지가 비어 있습니다.')
    Image.fromarray(rgba[:,:,:3].astype(np.uint8)).save(path)

if __name__=='__main__':
    main()
