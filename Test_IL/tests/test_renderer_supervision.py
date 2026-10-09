import json
import time
from types import SimpleNamespace as NS

from fastapi.testclient import TestClient
import pytest

from studio import server
from studio.config import IL_PYTHON,load


@pytest.mark.parametrize('launch_error',[False,True])
def test_light_renderer_failure_is_visible_but_never_commands_robot(tmp_path,monkeypatch,launch_error):
    (tmp_path/'outputs').mkdir();(tmp_path/'web').mkdir();monkeypatch.setattr(server,'ROOT',tmp_path)
    calls=[];commands=[]
    engine=NS(start=lambda:None,close=lambda:None,call=lambda *a:calls.append(a),public={})
    monkeypatch.setattr(server,'Engine',lambda *a,**kw:engine)
    def spawn(command,**kwargs):
        commands.append(command)
        if launch_error:raise OSError('test renderer unavailable')
        return NS(wait=lambda *a:1,poll=lambda:1)
    monkeypatch.setattr(server.subprocess,'Popen',spawn)
    with TestClient(server.create_app(load(),tmp_path/'data')) as client:
        deadline=time.monotonic()+3
        while time.monotonic()<deadline:
            status=client.get('/api/status').json()['sim']
            if status['state']=='error':break
            time.sleep(.02)
        assert status['state']=='error' and status['renderer']=='mujoco'
        assert client.get('/api/image/sim').status_code==204
    assert calls==[] and commands[0][0]==str(IL_PYTHON)
    assert commands[0][1].endswith('/pose_view.py')
