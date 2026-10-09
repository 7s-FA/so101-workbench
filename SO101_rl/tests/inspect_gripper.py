"""Finite headless comparison of Isaac link poses against teaching preview."""
import sys
import argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import ROOT, Reference, read_json, atomic_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'outputs/gripper_inspection')
    parser.add_argument('--full',action='store_true')
    parser.add_argument('--grip-effort',type=float)
    parser.add_argument('--grip-stiffness',type=float)
    parser.add_argument('--recorded-arm-load',action='store_true')
    parser.add_argument('--pause-capture',action='store_true')
    parser.add_argument('--jig-reset',action='store_true')
    args=parser.parse_args()
    from isaaclab.app import AppLauncher
    launcher=AppLauncher(headless=True,device='cuda:0',enable_cameras=True)
    import numpy as np
    from isaac_env import BEnvironment
    from run import capture
    ref=Reference();source=ROOT/'outputs/b_teaching_load_records'
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    (output/'error.json').unlink(missing_ok=True)
    records=[];trace=[];seen=set()
    try:
        scene=read_json(source/'scene.json')
        if args.grip_effort is not None:scene['actuators']['gripper_effort_nm']=args.grip_effort
        if args.grip_stiffness is not None:scene['actuators']['gripper_stiffness']=args.grip_stiffness
        if args.recorded_arm_load:scene['load_model']['arm']=read_json(ROOT/'config/arm_load_model.json')
        env=BEnvironment(ref,scene,output)
        env.reset(seed=42,options={'randomized':False})
        if args.jig_reset:
            saved=ref.jig_pose
            env.apply_jig_pose([saved[0]+2,saved[1]+1,saved[2]+.5])
            env.apply_jig_pose(saved);env.sim.play()
        for _ in range(5000 if args.full else 400):
            _,_,terminated,truncated,info=env.step(np.zeros(6))
            trace.append(dict(elapsed=env.playback.elapsed,index=env.step_index,
                requested=env.playback.session.last_goals['gripper'],actual=env.playback.ticks(env.q())['gripper'],
                raw=info['gripper_load_raw_estimate'],hold=env.playback.hold_tick))
            if env.step_index not in seen and (env.playback.waiting or env.playback.fraction>.9):
                seen.add(env.step_index);frames={}
                for name in ('gripper_link','moving_jaw_so101_v1_link'):
                    index=env.robot.body_names.index(name)
                    pos=env.robot.data.body_link_pos_w[0,index].cpu().numpy()
                    quat=env.robot.data.body_link_quat_w[0,index].cpu().numpy()
                    frames[name]=dict(position_m=pos.tolist(),quaternion_wxyz=quat.tolist())
                records.append(dict(**env.diagnostic_state(),isaac_frames=frames))
                if env.step_index in (2,3,4):capture(env,output/f'step_{env.step_index:02d}.png',view='grasp')
                if args.pause_capture and env.step_index==3:
                    env.sim.pause();capture(env,output/'paused.png',view='grasp')
                    env.sim.play();capture(env,output/'resumed.png',view='grasp')
                    break
                print('GRIP_INSPECT',env.step_index,trace[-1],flush=True)
            if terminated or truncated or (not args.full and env.step_index>=6):break
        else:raise RuntimeError('집게 검증 에피소드가 제한 시간 안에 끝나지 않았습니다.')
        atomic_json(output/'trace.json',trace);atomic_json(output/'states.json',records)
        atomic_json(output/'final_state.json',dict(**env.diagnostic_state(),
            velocity_ticks=env.playback.velocity_ticks(env.q(),env.robot.data.joint_vel[0,env.ids].cpu().numpy()),
            execution_report=env.playback.session.report,actuators=scene['actuators']))
        atomic_json(output/'result.json',info)
    except BaseException as exc:
        import traceback
        traceback.print_exc()
        atomic_json(output/'error.json',dict(error=str(exc)))
    finally:launcher.app.close(wait_for_replicator=False,skip_cleanup=True)


if __name__=='__main__':main()
