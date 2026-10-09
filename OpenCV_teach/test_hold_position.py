"""Replay departure and startup side effects without opening a motor port."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import hold_position as hold
import teach_left_arm as teaching

class Clock:
    def __init__(self): self.now=0.
    def __call__(self): return self.now
    def sleep(self,seconds): self.now+=seconds

class HoldTests(unittest.TestCase):
    def test_incident_departure_stops_on_first_bad_read_without_four_second_wait(self):
        wanted={'wrist_roll.pos':-.791208791}
        hold.check_hold(wanted,{'wrist_roll.pos':-.96})
        with self.assertRaisesRegex(RuntimeError,'자세 유지 이탈'):
            hold.check_hold(wanted,{'wrist_roll.pos':-113.58})
        with self.assertRaises(RuntimeError):hold.check_hold(wanted,{'wrist_roll.pos':float('nan')})

    def test_hold_tolerates_encoder_noise_and_grip_contact_residual(self):
        hold.check_hold({'wrist_roll.pos':0.,'gripper.pos':0.},
                        {'wrist_roll.pos':.09,'gripper.pos':4.})

    def bus(self,clock,departure=False):
        def read(reg,name,**kwargs):
            if reg=='Present_Position':return 2047-9 if departure and clock.now>=.1 else 2047
            return {'Goal_Position':2047,'Torque_Enable':1,'Status':0}[reg]
        return SimpleNamespace(read=Mock(side_effect=read))

    def test_settle_observes_previous_motors_and_rejects_delayed_departure(self):
        clock=Clock();bus=self.bus(clock,True)
        with self.assertRaisesRegex(RuntimeError,'위치 이탈'):
            hold.settle(bus,{'wrist_roll':2047,'gripper':2047},clock=clock,sleep=clock.sleep)
        self.assertLess(clock.now,.12)

    def test_stable_settle_checks_fixed_goals_and_never_writes(self):
        clock=Clock();bus=self.bus(clock)
        self.assertEqual(hold.settle(bus,{'wrist_roll':2047},clock=clock,sleep=clock.sleep),{'wrist_roll':[2047,2047]})
        self.assertGreaterEqual(clock.now,hold.SETTLE_SECONDS)
        bus.read=Mock(side_effect=lambda reg,*a,**kw: 100 if reg=='Goal_Position' else 2047)
        with self.assertRaises(RuntimeError):hold.settle(bus,{'wrist_roll':2047},clock=clock,sleep=clock.sleep)

    def test_cancel_is_checked_without_another_read(self):
        clock=Clock();bus=self.bus(clock)
        with self.assertRaisesRegex(RuntimeError,'cancelled'):
            hold.settle(bus,{'wrist_roll':2047},SimpleNamespace(is_set=lambda:True),clock=clock,sleep=clock.sleep)
        bus.read.assert_not_called()

    def test_preload_must_not_enable_torque_and_never_writes_calibration(self):
        registers={'Torque_Enable':0}
        bus=SimpleNamespace(write=Mock(side_effect=lambda r,n,v,**kw:registers.update({r:v})),
                            read=lambda r,n,**kw:registers[r])
        hold.configure_preload(bus,'wrist_roll')
        self.assertEqual(set(registers),{'Torque_Enable','Acceleration','Goal_Time','Goal_Velocity'})
        registers['Torque_Enable']=1
        with self.assertRaises(RuntimeError):hold.configure_preload(bus,'wrist_roll')

    def test_all_motors_receive_off_attempt_even_when_one_write_fails(self):
        hardware=teaching.Hardware.__new__(teaching.Hardware)
        bus=SimpleNamespace(is_connected=True,read=Mock(return_value=0),
                            write=Mock(side_effect=[IOError('wrist fault'),None,None,None,None,None]),
                            port_handler=SimpleNamespace(closePort=Mock()))
        hardware.leader=None;hardware.follower=SimpleNamespace(bus=bus,id='test')
        hardware.motion_attempted=True;hardware.fault_capture=None
        errors=hardware.close()
        self.assertTrue(errors);self.assertEqual(bus.write.call_count,6)
        self.assertEqual(bus.read.call_count,6);bus.port_handler.closePort.assert_called_once()


class ActivationPathTests(unittest.TestCase):
    def hardware(self, auto_enable):
        entry=SimpleNamespace(id=5,range_min=140,range_max=4004)
        regs={'Torque_Enable':0,'Goal_Position':400,'Present_Position':2047,
              'Operating_Mode':0,'Present_Temperature':30,'Status':0,'Lock':0}
        writes=[]
        def write(reg,name,value,**kwargs):
            writes.append((reg,value));regs[reg]=value
            if reg=='Goal_Position' and auto_enable:regs['Torque_Enable']=1
        def enable(name):write('Torque_Enable',name,1);write('Lock',name,1)
        bus=SimpleNamespace(write=write,read=lambda r,n,**kw:regs[r],enable_torque=enable,
                            _normalize=lambda values:{5:0.},is_connected=True,
                            port_handler=SimpleNamespace(closePort=lambda:None))
        h=teaching.Hardware.__new__(teaching.Hardware)
        h.follower=SimpleNamespace(id='test',bus=bus,calibration={'wrist_roll':entry});h.leader=None
        h.current_raw=lambda name:regs['Present_Position'];h.health=lambda **kw:None
        h.motion_attempted=True;h.fault_capture=None
        return h,regs,writes

    def test_goal_side_effect_aborts_and_existing_cleanup_turns_motor_off(self):
        from unittest.mock import patch
        h,regs,writes=self.hardware(True)
        with patch.object(teaching,'JOINTS',('wrist_roll',)):
            try:
                with self.assertRaisesRegex(RuntimeError,'목표 초기화가 토크'):
                    h.prime_and_enable({'wrist_roll.pos':0.},None)
            finally:h.close()
        self.assertEqual(regs['Torque_Enable'],0)
        self.assertFalse(any(r=='Torque_Enable' and v==1 for r,v in writes))

    def test_explicit_enable_follows_preload_and_settle_preserves_initial_goal(self):
        from unittest.mock import patch
        h,regs,writes=self.hardware(False);clock=Clock();original=hold.settle
        with patch.object(teaching,'JOINTS',('wrist_roll',)),patch.object(hold,'settle',
                side_effect=lambda b,goals,c:original(b,goals,c,clock=clock,sleep=clock.sleep)):
            h.prime_and_enable({'wrist_roll.pos':0.},None)
        self.assertLess(writes.index(('Goal_Position',2047)),writes.index(('Torque_Enable',1)))
        self.assertTrue(h.activation_report[0]['verified'])
        self.assertEqual(regs['Goal_Position'],2047)

if __name__=='__main__':unittest.main()
