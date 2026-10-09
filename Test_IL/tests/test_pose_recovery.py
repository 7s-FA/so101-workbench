"""Viewer recovery, with a deterministic clock and no GPU, server or motor IO."""
import io,json,runpy,urllib.error,sys
from types import SimpleNamespace as NS
import numpy as np
import pytest
from studio.config import ROOT


def runner(tmp_path,monkeypatch,fetch,seconds=0):
    g=runpy.run_path(str(ROOT/'scripts/pose_view.py'))['main'].__globals__
    clock=[0.];handlers={};statuses=[]
    class Event:
        stopped=False
        def set(self):self.stopped=True
        def is_set(self):return self.stopped
        def wait(self,delay):clock[0]+=delay;return self.stopped
    class Renderer:
        frames=0
        def render(self,*args):
            if self.frames:return None
            self.frames+=1;return np.zeros((8,8,3),dtype=np.uint8)
        def close(self):pass
    g['ROOT']=tmp_path;g['PoseRenderer']=Renderer
    g['time']=NS(monotonic=lambda:clock[0],perf_counter=lambda:clock[0],time=lambda:clock[0])
    g['threading']=NS(Event=Event)
    g['signal']=NS(SIGINT=2,SIGTERM=15,signal=lambda n,f:handlers.update({n:f}))
    g['urllib']=NS(request=NS(urlopen=lambda *a,**kw:fetch(clock,handlers,statuses)))
    original=g['atomic_json']
    def write(path,value):statuses.append(value);original(path,value)
    g['atomic_json']=write
    monkeypatch.setattr(sys,'argv',['pose_view','--server','http://offline','--seconds',str(seconds)])
    return g['main'],clock,statuses


def response():return io.BytesIO(json.dumps(dict(display_angles=[0]*6,display_robot='so101',display_source='test')).encode())


@pytest.mark.parametrize('cold_start',[False,True])
def test_reconnect_restores_same_pose_without_stale_image_status(tmp_path,monkeypatch,cold_start):
    count=[0]
    def fetch(*args):
        count[0]+=1
        if count[0]==(1 if cold_start else 2):raise urllib.error.URLError('transient')
        return response()
    main,clock,states=runner(tmp_path,monkeypatch,fetch,seconds=1)
    main()
    reconnect=next(i for i,s in enumerate(states) if s['state']=='reconnecting')
    assert any(s['state']=='ready' for s in states[reconnect+1:])
    assert states[-1]['state']=='stopped' and states[-1]['frames']==1
    assert (tmp_path/'outputs/sim.jpg').is_file()
    assert not any(s['state']=='error' for s in states)


def test_outage_retries_are_bounded_and_report_failure(tmp_path,monkeypatch):
    calls=[]
    def fetch(clock,*args):calls.append(clock[0]);raise urllib.error.URLError('offline')
    main,clock,states=runner(tmp_path,monkeypatch,fetch)
    with pytest.raises(RuntimeError,match='15초'):main()
    assert 15<=clock[0]<=17 and 2<len(calls)<15
    assert states[-1]['state']=='error'


def test_shutdown_during_failed_read_is_clean(tmp_path,monkeypatch):
    def fetch(clock,handlers,states):handlers[15]();raise urllib.error.URLError('server closing')
    main,clock,states=runner(tmp_path,monkeypatch,fetch)
    main()
    assert states[-1]['state']=='stopped'
    assert not any(s['state'] in ('error','reconnecting') for s in states)
