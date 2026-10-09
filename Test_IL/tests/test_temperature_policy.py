import pytest
from studio.legacy_omx import Hardware
from studio.config import load
from studio.engine import Engine
from teach_io.temperature import TemperatureGuard,TemperatureFault
from test_follow_response import packed_bus


def test_only_il_explicitly_enables_sustained_policy(monkeypatch):
    import studio.legacy_omx as module
    calls=[]
    monkeypatch.setattr(module,'Follower',lambda *args,**kwargs:calls.append(kwargs))
    hardware=Hardware.__new__(Hardware);hardware.port='fixture';hardware.open_follower()
    assert calls[0]['sustained_temperature'] is True
    assert calls[0]['reference_control'] is True


def test_packed_live_feedback_retains_temperatures_and_motor_faults():
    bus=packed_bus();read=bus.packet.readTxRx;now=[0.]
    def hot(*args):
        data,comm,error=read(*args);data[22]=65
        return data,comm,error
    bus.packet.readTxRx=hot;guard=TemperatureGuard(lambda:now[0])
    for n in range(50):
        now[0]=n/10
        rows=bus.sample(packed=True,temperature_guard=guard,active=True)
        assert all(r['temperature']==65 for r in rows)
    now[0]=5
    with pytest.raises(TemperatureFault):bus.sample(packed=True,temperature_guard=guard,active=True)
    for failure in ({'error':128},{'short':True}):
        with pytest.raises(RuntimeError):packed_bus(**failure).sample(packed=True,temperature_guard=TemperatureGuard(),active=True)


def test_engine_exposes_pending_temperature_without_stopping_healthy_motion(tmp_path):
    engine=Engine(load(),tmp_path)
    guard=TemperatureGuard(lambda:2.)
    guard.observe(12,'packed',66,active=True)
    engine.latest={'temperature':guard.status()}
    try:
        engine.publish()
        assert engine.public['temperature']['pending'][0]['id']==12
        assert engine.public['error'] is None
        assert engine.public['mode']=='idle'
    finally:engine.close()
