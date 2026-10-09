#!/usr/bin/env python3
"""Commit the workshop's official LeRobot dataset locally, with no Hub operations.

New episodes are appended to an isolated generation; an atomic symlink switch
publishes only after finalize and reader validation. No existing data is edited.
"""
import argparse
import fcntl
import json
import math
import os
import shutil
import sys
import uuid
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_DATASETS_OFFLINE']='1'
import numpy as np
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from studio.store import Store, atomic_json
from studio.reference import ReferenceCodec, NAMES
from studio.robot import identity, angles_to_pose, NAMES as ROBOT_NAMES
from studio.dataset_retention import prune_dataset_versions

REPO_ID = 'local/so101_studio'
KEYS = {'gripper':'wrist','table':'top'}

def features(shape):
    return {'observation.state':{'dtype':'float32','shape':(6,),'names':[n+'.pos' for n in NAMES]},
            'action':{'dtype':'float32','shape':(6,),'names':[n+'.pos' for n in NAMES]},
            # Video has a regular FPS clock; acquisition timing remains available verbatim.
            'source.timestamp':{'dtype':'float64','shape':(1,),'names':['seconds']},
            **{f'observation.images.{dst}':{'dtype':'video','shape':tuple(shape[src]),
                'names':['height','width','channels']} for src,dst in KEYS.items()}}

def normalized(q, calibration, codec):
    if codec.cfg.get('robot_type')=='so101':
        pose=angles_to_pose(q)
        return np.asarray([pose[n+'.pos'] for n in ROBOT_NAMES],dtype=np.float32)
    ticks = [round(z+a*s) for z,a,s in zip(calibration['zero_ticks'],q,calibration['ticks_per_rad'])]
    return np.asarray(codec.normalize(ticks),dtype=np.float32)

def export_task(store, task_id):
    data_root=store.root.resolve()
    folder=store.task_folder(task_id);folder.mkdir(parents=True,exist_ok=True)
    task_name=folder.name
    repo_id=None
    with store.source_snapshot():
        episodes=[e for e in store.list() if (e.get('task_id') or None)==task_id]
    current=folder/'lerobot';versions=data_root/'.lerobot_versions';versions.mkdir(exist_ok=True)
    if not episodes:
        if current.is_symlink():current.unlink()
        elif current.exists():raise ValueError('Expected a generated dataset symlink')
        return 0
    types={identity(e) for e in episodes}
    if len(types)!=1:raise ValueError('서로 다른 로봇의 기록은 작업 폴더를 나눠 주세요.')
    robot=types.pop();repo_id='local/'+robot+'_'+(task_id or 'unclassified')
    previous=json.loads((current/'studio_manifest.json').read_text()) if current.exists() else {'episodes':[]}
    by_id={e['id']:e for e in episodes}
    rebuild=any(e['studio_id'] not in by_id for e in previous['episodes'])
    rebuild = rebuild or previous.get('task_name') != task_name
    if rebuild:previous={'episodes':[]}
    done={e['studio_id'] for e in previous['episodes']}
    pending=[e for e in episodes if e['id'] not in done]
    labels_changed=any(e.get('result')!=by_id[e['studio_id']]['result'] or
                       e.get('included')!=by_id[e['studio_id']]['included'] for e in previous['episodes'])
    if not pending and not labels_changed:return len(episodes)
    vcodec=previous.get('vcodec','h264') if current.exists() and not rebuild else 'libsvtav1'
    stage=versions/uuid.uuid4().hex
    dataset=None
    try:
        if current.exists() and not rebuild:
            # reflink is fast where supported; never hardlink writable metadata/video files.
            import subprocess
            subprocess.run(['cp','--reflink=auto','-a',str(current.resolve()),str(stage)],check=True)
            dataset=LeRobotDataset(repo_id,root=stage,video_backend='pyav',vcodec=vcodec)
            dataset.start_image_writer(num_threads=2)
        mapping=[dict(e,result=by_id[e['studio_id']]['result'],included=by_id[e['studio_id']]['included'])
                 for e in previous['episodes']]
        for meta in pending:
            with store.export_episode(meta['id']) as (meta,rows,open_frame):
                if not meta.get('calibration'):raise ValueError(f"Episode {meta['id']} has no recorded follower calibration; original kept")
                codec=ReferenceCodec(meta.get('lerobot_calibration'))
                if dataset is None:
                    shape={name:rows[0]['cameras'][name]['shape'] for name in KEYS}
                    dataset=LeRobotDataset.create(repo_id,int(meta.get('requested_fps',30)),features(shape),
                        root=stage,robot_type=robot+'_follower',use_videos=True,image_writer_threads=2,
                        video_backend='pyav',vcodec=vcodec,encoder_threads=2)
                if int(meta.get('requested_fps',30))!=dataset.fps:raise ValueError('Recording FPS differs from dataset')
                index=dataset.num_episodes
                for row in rows:
                    frame={'observation.state':normalized(row['state'],meta['calibration'],codec),
                           'action':normalized(row['action'],meta['calibration'],codec),
                           'source.timestamp':np.array([row['t']],dtype=np.float64),
                           'task':task_name}
                    for name,dst in KEYS.items():
                        with open_frame(name,row['cameras'][name]['file']) as source, Image.open(source) as image:
                            frame['observation.images.'+dst]=np.asarray(image.convert('RGB'))
                    dataset.add_frame(frame)
                dataset.save_episode(parallel_encoding=False)
                mapping.append({'studio_id':meta['id'],'episode_index':index,'synthetic':bool(meta.get('synthetic')),
                                'result':meta['result'],'included':meta['included'],'source_fingerprint':meta.get('fingerprint')})
        dataset.finalize();dataset.stop_image_writer();dataset=None
        reader=LeRobotDataset(repo_id,root=stage,video_backend='pyav')
        if reader.num_episodes!=len(mapping):raise ValueError('Dataset episode count verification failed')
        # Check the newest episode's first decoded state and both camera streams.
        item=reader[reader.meta.episodes[-1]['dataset_from_index']]
        for key in ('observation.state','action','observation.images.wrist','observation.images.top'):
            if key not in item:raise ValueError('Missing feature '+key)
        atomic_json(stage/'studio_manifest.json',{'format':'LeRobotDataset','repo_id':repo_id,'task_id':task_id,'task_name':task_name,'episodes':mapping,
            'robot_type':robot,'vcodec':vcodec,'units':'LeRobot degrees arm, RANGE_0_100 gripper' if robot=='so101' else 'LeRobot RANGE_M100_100 arm, RANGE_0_100 gripper',
            'clock':'timestamp=frame_index/fps; source.timestamp and raw originals preserve actual capture timing',
            'uploaded':False})
        with store.source_snapshot():
            current_labels={e['id']:(e['result'],e['included']) for e in store.list() if (e.get('task_id') or None)==task_id}
            exported_labels={e['studio_id']:(e['result'],e['included']) for e in mapping}
            if current_labels!=exported_labels:
                # A delete/save happened during encoding. Never publish deleted records.
                shutil.rmtree(stage)
                return None
            link=folder/('.lerobot-link-'+uuid.uuid4().hex)
            link.symlink_to(os.path.relpath(stage, folder),target_is_directory=True)
            link.replace(current)
        return len(mapping)
    except BaseException:
        if dataset:
            try:dataset.stop_image_writer()
            except Exception:pass
        # This generation was never published. Preserve previous dataset and all raw episodes.
        if stage.exists():shutil.rmtree(stage)
        raise


def write_dataset(data_root):
    store=Store(Path(data_root).resolve())
    with (store.root/'.lerobot-write.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        from studio.learning import Tasks
        # Include empty folders to remove exports after the last episode moves out.
        ids=[None]+[t['id'] for t in Tasks(store).list()]
        counts=[export_task(store, task_id) for task_id in ids]
        stale=any(n is None for n in counts)
        total=sum(n for n in counts if n is not None)
        legacy=store.root/'lerobot'
        if legacy.is_symlink():legacy.unlink()
    # Only reclaim old generations after a validated export was published. The
    # collector reacquires the writer lock and skips if another export started.
    cleanup = prune_dataset_versions(store.root) if not stale else None
    print(json.dumps({'episodes':total,'root':str(store.root/'tasks'),'stale':stale,'cleanup':cleanup}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data-root',type=Path,required=True)
    write_dataset(parser.parse_args().data_root)
