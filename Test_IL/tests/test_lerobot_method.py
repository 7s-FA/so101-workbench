import copy,io,json,math,queue,time
from types import SimpleNamespace as NS
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from studio.cameras import learning_image
from studio.depth import encode
from studio.engine import Engine,validate_replay
from studio.config import load,fingerprint
from studio.hardware import Hardware,mapping
from studio.policy import PolicyWorker
from studio.robot import pose_to_angles
from studio.server import create_app
from test_studio import FakeHardware,FakeCamera


def test_raw_color_and_depth_are_lossless_and_replay_has_correct_mime(tmp_path):
    app=create_app(load(),tmp_path,launch_sim=False,test_mode=True);e=app.state.engine
    raw=np.random.default_rng(42).integers(0,256,(32,32,3),dtype=np.uint8)
    frames={k:dict(bgr=raw.copy(),jpeg=b'preview deliberately not a valid learning image',shape=list(raw.shape),acquired=1.,sequence=1) for k in ('gripper','table')}
    depth=np.arange(1024,dtype=np.uint16).reshape(32,32)*32
    frames['table']['depth']=encode(depth,.0005)
    try:
        e.store.begin(dict(synthetic=True,depth_cameras=['table']))
        for t in (0.,1/30):e.store.append(dict(t=t,state=[0]*6,action=[0]*6,command=[0]*6),frames)
        e.store.save('Success');e.select(1);e.mode='preview'
        row=e.samples[0];p=e.store.path(1)
        assert row['cameras']['gripper']['file'].endswith('.png')
        assert np.array_equal(cv2.imread(str(p/'gripper'/row['cameras']['gripper']['file'])),raw)
        assert np.array_equal(cv2.imread(str(p/'table_depth'/row['cameras']['table']['depth']['file']),-1),depth)
        with TestClient(app) as c:
            r=c.get('/api/image/gripper')
            assert r.headers['content-type']=='image/png'
    finally:e.close()


def test_recorded_action_is_requested_target_and_command_is_sent_target(tmp_path,monkeypatch):
    e=Engine(load(),tmp_path);hw=FakeHardware(e.measured);e.hardware=hw
    requested=pose_to_angles([150,-20,30,40,50,35]);bounded=e.limit_goal(requested)
    hw.requested_action=requested.copy();hw.target=lambda:(bounded.copy(),[2000]*6)
    e.task_id=e.learning.tasks.save('SO101 test')['id']
    e.cameras={k:FakeCamera() for k in ('gripper','table')};e.mode='teleop';e.ping()
    monkeypatch.setattr(e.dataset_writer,'request',lambda:None)
    try:
        e.operation('record',{})
        for i in range(2):e.update(e.record_start+(i+1)/30,1/30)
        e.operation('save',{'result':'Success'})
        meta,rows=e.store.episode(1)
        assert meta['learning_method']=='lerobot_native_rgbd_v1'
        assert rows[0]['action']==requested and rows[0]['command']!=requested
        validate_replay(meta,rows,fingerprint(),mapping())
    finally:e.close()


def test_hardware_exposes_raw_leader_target_before_follower_clamp():
    raw=pose_to_angles([150,0,0,0,0,30]);m=mapping();bounded=m.clamp_target(raw)
    h=Hardware.__new__(Hardware);h.follower_mapping=m
    h.follower=NS(request=lambda op:dict(leader_target=bounded,leader_angles=raw,leader_ticks=[2000]*6))
    target,ticks=h.target()
    assert target==bounded and h.requested_action==raw and target!=raw


def test_policy_queue_retains_raw_pixels_without_blocking_on_encoding():
    w=PolicyWorker.__new__(PolicyWorker);w.ready=True;w.busy=False;w.requests=queue.Queue(maxsize=1)
    raw=np.zeros((8,8,3),np.uint8);raw[:,:,2]=213
    frames={'gripper':dict(bgr=raw,jpeg=b'not for learning',acquired=1.)}
    assert w.submit([0]*6,frames,1.)
    payload=w.requests.get_nowait()
    assert 'cameras' not in payload and payload['frames']['gripper']['bgr'] is raw
    ext,blob=learning_image(payload['frames']['gripper'])
    assert ext=='.png' and np.array_equal(cv2.imdecode(np.frombuffer(blob,np.uint8),1),raw)
