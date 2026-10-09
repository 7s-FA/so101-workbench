"""Component check only: a released cap settles magnetically above nested TPU."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import ROOT,Reference,read_json,atomic_json


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--scene',type=Path,default=ROOT/'outputs/assembly_contract/scene.json')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/assembly_component_check')
    parser.add_argument('--tpu-damping',type=float)
    args=parser.parse_args()
    from isaaclab.app import AppLauncher
    launcher=AppLauncher(headless=True,device='cuda:0',enable_cameras=False)
    import torch
    torch.set_num_threads(4)
    from scipy.spatial.transform import Rotation
    from isaac_env import BEnvironment,root_pose
    from assembly import assembly_state
    output=args.output;output.mkdir(parents=True,exist_ok=True)
    try:
        scene=read_json(args.scene)
        if args.tpu_damping is not None:scene['assembly']['tpu_contact']['damping_ns_m']=args.tpu_damping
        atomic_json(output/'scene.json',scene)
        env=BEnvironment(Reference(),scene,output)
        env.reset(seed=42,options={'randomized':False})
        for name,body in env.parts.items():
            tf=np.array(scene['assembly']['target_poses_mm'][name])
            if name=='cap':tf[:3,3]+=[.5,0,1.]
            body.write_root_state_to_sim(torch.tensor(root_pose(tf)[None],dtype=torch.float32,device=body.device));body.reset()
        env._magnet_near=True
        for _ in range(round(3/env.dt)):env._physics_step(env.home)
        poses={};speeds={}
        for name,body in env.parts.items():
            st=body.data.root_state_w[0].cpu().numpy();tf=np.eye(4);tf[:3,3]=st[:3]*1000
            tf[:3,:3]=Rotation.from_quat(st[[4,5,6,3]]).as_matrix();poses[name]=tf;speeds[name]=float(np.linalg.norm(st[7:10]))
        result=assembly_state(poses,speeds,scene['assembly'])
        result.update(magnets=env.last_magnets,speeds_m_s=speeds,component_test_only=True,episode_executed=False)
        atomic_json(output/'result.json',result);print(result,flush=True)
        if not all(result[k] for k in ['housing_on_pallet','tpu_in_housing','cap_aligned']) or env.last_magnets['active_pairs']!=4:
            raise RuntimeError('자석·TPU 틈새 유지 물리 검사 실패')
        atomic_json(output/'complete.json',{'passed':True})
    except BaseException as exc:
        import traceback
        traceback.print_exc();atomic_json(output/'error.json',{'error':str(exc)});raise
    finally:launcher.app.close(wait_for_replicator=False,skip_cleanup=True)


if __name__=='__main__':main()
