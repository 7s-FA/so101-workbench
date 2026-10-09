"""RGBD transport/storage/viewer tests; every camera and motor is synthetic."""
import sys,threading,time,json
from types import SimpleNamespace as NS
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from studio.cameras import RealSenseCamera,CameraUnavailable
from studio.depth import encode
from studio.store import Store
from studio.config import load
from studio.server import create_app
from studio.engine import Engine
from test_studio import sample,frames,FakeCamera


def fake_sdk(monkeypatch, missing_depth=False):
    calls=[];camera=RealSenseCamera.__new__(RealSenseCamera)
    camera.device='realsense:TEST435';camera.fps=30;camera.stop_event=threading.Event()
    camera.reconnects=0;camera.failures=0;camera.frame=None;camera.actual={};camera.log=lambda *a,**kw:None
    raw=np.arange(480*640,dtype=np.uint16).reshape(480,640)
    intr=NS(width=640,height=480,fx=610.,fy=611.,ppx=320.,ppy=240.,model='brown_conrady',coeffs=[0.]*5)
    stream=NS(as_video_stream_profile=lambda:NS(get_intrinsics=lambda:intr),get_extrinsics_to=lambda other:NS(rotation=[1,0,0,0,1,0,0,0,1],translation=[.01,0,0]))
    def frame(image):return NS(get_data=lambda:image,get_timestamp=lambda:123.,get_frame_number=lambda:7,get_frame_timestamp_domain=lambda:'hardware_clock')
    frameset=NS(get_color_frame=lambda:frame(np.zeros((480,640,3),np.uint8)),get_depth_frame=lambda:None if missing_depth else frame(raw))
    class Config:
        def enable_device(self,value):calls.append(('serial',value))
        def enable_stream(self,*args):calls.append(('stream',args))
    class Pipeline:
        def start(self,cfg):return NS(get_stream=lambda _:stream,get_device=lambda:NS(get_info=lambda _:'Intel RealSense D435',first_depth_sensor=lambda:NS(get_depth_scale=lambda:.0005)))
        def wait_for_frames(self,**kw):camera.stop_event.set();return frameset
        def stop(self):calls.append('stopped')
    def align(target):
        calls.append(('align',target));return NS(process=lambda frames:frames)
    monkeypatch.setitem(sys.modules,'pyrealsense2',NS(pipeline=Pipeline,config=Config,align=align,stream=NS(color='color',depth='depth'),format=NS(bgr8='bgr8',z16='z16'),camera_info=NS(name='name')))
    return camera,calls,raw


def test_d435_aligned_depth_preserves_16_bits_scale_and_sensor_time(monkeypatch):
    camera,calls,raw=fake_sdk(monkeypatch);camera.run()
    assert ('stream',('color',640,480,'bgr8',30)) in calls
    assert ('stream',('depth',640,480,'z16',30)) in calls and ('align','color') in calls
    assert calls[0]==('serial','TEST435') and calls[-1]=='stopped'
    depth=camera.frame['depth'];decoded=cv2.imdecode(np.frombuffer(depth['png'],np.uint8),cv2.IMREAD_UNCHANGED)
    assert decoded.dtype==np.uint16 and np.array_equal(decoded,raw)
    assert depth['scale_m_per_unit']==.0005 and depth['sensor_timestamp_ms']==123.
    assert depth['frame_number']==7 and depth['aligned_to']=='color'
    assert camera.actual['depth_training'] is False and camera.actual['depth_recording'] is True
    assert camera.actual['intrinsics']['color']['fx']==610.


def test_missing_depth_never_publishes_rgb_only_as_new_frame(monkeypatch):
    camera,calls,_=fake_sdk(monkeypatch,True);camera.run()
    assert camera.frame is None and camera.failures==1 and '깊이' in camera.error
    assert calls[-1]=='stopped'


def rgbd(t=100.):
    value=frames(t)
    value['table']['depth']=encode(np.array([[0,1,256],[1000,32768,65535]],np.uint16),.001)
    value['table']['depth'].update(sensor_timestamp_ms=456.,frame_number=9,timestamp_domain='hardware_clock')
    return value


def test_depth_save_reopen_seek_and_old_rgb_episode(tmp_path):
    app=create_app(load(),tmp_path,launch_sim=False,test_mode=True);e=app.state.engine
    try:
        e.store.begin(dict(synthetic=True,depth_cameras=['table']))
        for t in (0.,1.):e.store.append(sample(t),rgbd(t))
        meta=e.store.save('Success');m,rows=Store(tmp_path).episode(meta['id'])
        depth=rows[0]['cameras']['table']['depth']
        assert 'png' not in depth and 'preview_jpeg' not in depth
        raw=cv2.imread(str(e.store.path(1)/'table_depth'/depth['file']),cv2.IMREAD_UNCHANGED)
        assert raw.dtype==np.uint16 and raw[1,2]==65535 and raw[0,0]==0
        assert depth['scale_m_per_unit']==.001 and m['depth_cameras']==['table']
        e.select(1);e.mode='preview';e.publish()
        client=TestClient(app)
        response=client.get('/api/image/table-depth');assert response.status_code==200 and response.headers['content-type']=='image/jpeg'
        assert e.public['depth_view']['available']
        e.store.begin(dict(synthetic=True))
        for t in (0.,1.):e.store.append(sample(t),frames(t))
        e.store.save('Success');e.select(2);e.publish()
        assert client.get('/api/image/table-depth').status_code==204
        assert not e.public['depth_view']['available']
    finally:e.close()


def test_required_depth_absence_rejects_record_and_keeps_existing_capture(tmp_path):
    store=Store(tmp_path);store.begin(dict(depth_cameras=['table']))
    with pytest.raises(ValueError,match='깊이'):store.append(sample(0),frames())
    assert store.count==0;store.discard()
    e=Engine(load(),tmp_path)
    e.cameras={'gripper':FakeCamera(),'table':FakeCamera()};e.cameras['table'].device='realsense:TEST435'
    try:
        with pytest.raises(CameraUnavailable,match='깊이'):e.frames()
    finally:e.close()


def test_depth_display_never_changes_raw_values():
    depth=encode(np.array([[0,100,500,1500,30000,65535]],np.uint16),.001)
    decoded=cv2.imdecode(np.frombuffer(depth['png'],np.uint8),cv2.IMREAD_UNCHANGED)
    assert decoded.tolist()==[[0,100,500,1500,30000,65535]]
    for scale in (0,float('nan')):
        with pytest.raises(ValueError):encode(decoded,scale)
