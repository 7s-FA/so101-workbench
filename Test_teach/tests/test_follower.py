import copy
import math
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from teach_io.mapping import IDS, MODELS, Mapping, fit_arm_pose, TICKS_PER_RAD
from teach_io.worker import Guard, StartAlignment
from teach_io.bus import Bus


def calibration():
    return dict(kind='omx_follower_urdf',verified=True,ids=IDS,models=MODELS,
        zero_ticks=[2048]*6,ticks_per_rad=[TICKS_PER_RAD,-TICKS_PER_RAD,TICKS_PER_RAD,TICKS_PER_RAD,TICKS_PER_RAD,500],
        limits=[[-2,2]]*5+[[0,1.1]],tick_limits=[[0,4095]]*6)


class MappingTests(unittest.TestCase):
    def test_visual_confirmation_is_separate_and_explicit(self):
        cfg=calibration();cfg['verified']=False
        with self.assertRaises(ValueError):Mapping(cfg,allow_visual_confirmation=True)
        cfg['operator_confirmation']={'method':'user_visual_real_to_sim','accepted_for_low_speed_control':True,'confirmed_at':'2026-09-16'}
        with self.assertRaises(ValueError):Mapping(cfg)
        mapping=Mapping(cfg,allow_visual_confirmation=True)
        self.assertFalse(mapping.cfg['verified'])
        self.assertEqual(len(mapping.to_ticks([0]*6)),6)

    def test_inverse_for_negative_direction(self):
        mapping=Mapping(calibration());q=[.3,-.4,.2,.1,-.5,.75]
        back=mapping.to_angles(mapping.to_ticks(q))
        self.assertLess(max(abs(a-b) for a,b in zip(q,back)),.002)

    def test_leader_or_unverified_rejected(self):
        for key,value in [('kind','omx'),('verified',False),('ids',list(range(1,7)))]:
            cfg=calibration();cfg[key]=value
            with self.assertRaises(ValueError):Mapping(cfg)

    def test_invalid_target_never_clamped(self):
        m=Mapping(calibration())
        for q in ([0]*5,[0]*5+[float('nan')],[0]*5+[1.2],[3]+[0]*5):
            with self.assertRaises(ValueError):m.to_ticks(q)
        cfg=calibration();cfg['zero_ticks'][0]=4090
        with self.assertRaises(ValueError):Mapping(cfg).to_ticks([.2]+[0]*5)

    def test_two_measured_poses_establish_direction(self):
        q1=[0]*5;q2=[.3]*5;t1=[2048]*5
        t2=[2048+TICKS_PER_RAD*.3*s for s in [1,-1,1,-1,1]]
        zero,scale=fit_arm_pose(q1,t1,q2,t2)
        self.assertEqual(zero,t1)
        self.assertEqual([math.copysign(1,s) for s in scale],[1,-1,1,-1,1])
        with self.assertRaises(ValueError):fit_arm_pose(q1,t1,[.01]*5,t2)

    def test_old_simulation_park_exceeds_physical_wrist_limit(self):
        cfg=calibration();cfg['limits']=[[-math.tau,math.tau]]*5+[[0,1.1]]
        with self.assertRaisesRegex(ValueError,'physical range'):
            Mapping(cfg).to_ticks([0,0,0,math.radians(102.4),0,.75])


class GuardTests(unittest.TestCase):
    def test_start_alignment_is_small_slow_and_requires_feedback(self):
        home=[.05]*5+[.5];measured=[0]*5+[.46]
        a=StartAlignment(measured,home,10)
        guard=Guard(Mapping(calibration()));guard.last=measured;guard.last_time=10
        q=measured
        for i in range(1,40):
            now=10+i*.02;previous=q;q=a.advance(home,now)
            self.assertLessEqual(max(abs(x-y) for x,y in zip(q,previous)),.002000001)
            guard.check(q,previous,now)
        self.assertFalse(a.reached(measured))
        self.assertTrue(a.reached(home,stationary=True,effort=0))

    def test_alignment_rejects_far_pose_early_trial_and_stale_input(self):
        with self.assertRaises(ValueError):StartAlignment([0]*6,[math.radians(10.01)]*6,10)
        a=StartAlignment([0]*6,[.05]*6,10)
        with self.assertRaises(ValueError):a.advance([.2]*6,10.02)
        with self.assertRaises(ValueError):a.advance([.05]*6,10.31)

    def test_ten_degree_start_boundary_for_each_joint_and_direction(self):
        for joint in range(6):
            for sign in (-1,1):
                home=[.5]*6
                for degrees,accepted in ((9.99,True),(10,True),(10.01,False)):
                    measured=home.copy();measured[joint]+=sign*math.radians(degrees)
                    with self.subTest(joint=joint,sign=sign,degrees=degrees):
                        if accepted:StartAlignment(measured,home,10)
                        else:
                            with self.assertRaises(ValueError):StartAlignment(measured,home,10)

    def test_ten_degree_alignment_keeps_slow_rate_and_feedback_requirement(self):
        measured=[.1]*5+[.5];home=[v+math.radians(10) for v in measured]
        a=StartAlignment(measured,home,10)
        guard=Guard(Mapping(calibration()));guard.last=measured;guard.last_time=10
        q=measured
        for i in range(1,101):
            previous=q;q=a.advance(home,10+i*.02)
            self.assertLessEqual(max(abs(x-y) for x,y in zip(q,previous)),.002000001)
            guard.check(q,previous,10+i*.02)
            if i<88:self.assertFalse(a.reached(home))
        self.assertFalse(a.reached(measured))
        self.assertTrue(a.reached(home,stationary=True,effort=0))

    def test_start_accepts_stable_logged_residual_but_not_motion_or_high_output(self):
        home=[0,math.radians(-25),math.radians(-55),math.radians(80),0,math.radians(10)]
        a=StartAlignment(home,home,10)
        for degrees in (1.474609375,2.705078125,3.):
            measured=home.copy();measured[3]+=math.radians(degrees)
            self.assertFalse(a.reached(measured))
            self.assertFalse(a.reached(measured,stationary=False,effort=11.3))
            self.assertFalse(a.reached(measured,stationary=True,effort=60))
            self.assertTrue(a.reached(measured,stationary=True,effort=11.3))
        for joint in range(6):
            measured=home.copy();measured[joint]+=math.radians(3.01)
            self.assertFalse(a.reached(measured,stationary=True,effort=0))

    def test_alignment_timeout_reports_remaining_joint_error(self):
        home=[0.]*6;a=StartAlignment(home,home,10)
        measured=home.copy();measured[3]=math.radians(4)
        self.assertFalse(a.reached(measured,stationary=True,effort=0))
        with self.assertRaisesRegex(ValueError,'ID 14 remaining error 4.00 deg'):
            a.advance(home,30.01)

    def test_following_error_stops_arm_but_allows_gripper_contact(self):
        g=Guard(Mapping(calibration()))
        self.assertEqual(len(g.check([0]*6,[0]*5+[.5],10)),6)
        with self.assertRaises(ValueError):g.check([.3]+[0]*5,[0]*6,10.02)

    def test_old_command_and_jump_are_rejected(self):
        for t,q in [(10.31,[0]*6),(10.02,[.1]+[0]*5)]:
            g=Guard(Mapping(calibration()));g.check([0]*6,[0]*6,10)
            with self.assertRaises(ValueError):g.check(q,[0]*6,t)

    def test_slow_fresh_tracking_is_accepted(self):
        g=Guard(Mapping(calibration()))
        for i in range(100):
            q=[i*.002]*5+[.5]
            g.check(q,q,10+i*.02)


class BusTests(unittest.TestCase):
    def test_reference_profile_restores_defaults_and_workshop_elbow_pid(self):
        bus,writes,registers,info=self.fake_bus()
        for motor in IDS:
            for address,value in {80:1000,82:0,84:1000,100:800,102:450,108:10,112:50}.items():
                registers[motor,address]=value
        bus.enable(info,[2048]*6,reference=True)
        for motor in IDS:
            self.assertEqual(registers[motor,108],0)
            self.assertEqual(registers[motor,112],0)
            self.assertEqual(registers[motor,100],800)
            self.assertEqual(registers[motor,84],1500 if motor==13 else 1000)
            self.assertEqual(registers[motor,80],600 if motor==13 else 1000)
        self.assertEqual(registers[16,102],450)
        self.assertEqual(bus.power_profile['name'],'lerobot_reference')
        self.assertTrue(all(x[1]>=64 for x in writes if isinstance(x[0],int)))

    def fake_bus(self, *, pwm_limit=885, current_limit=1750):
        bus=Bus.__new__(Bus);bus.owned=False;writes=[];registers={}
        def read(motor,address,size):
            return registers.get((motor,address),{36:pwm_limit,38:current_limit,44:600}.get(address,0))
        def write(motor,address,size,value):
            writes.append((motor,address,size,value));registers[motor,address]=value
        bus.read=read;bus.write=write
        bus.positions=lambda q:writes.append(('position',q))
        bus.sample=lambda:[{'ticks':2048} for _ in IDS]
        info=[dict(id=i,mode=4 if i<16 else 5,torque=0,watchdog=0,drive=0) for i in IDS]
        return bus,writes,registers,info

    def test_no_eeprom_write(self):
        bus=Bus.__new__(Bus)
        with self.assertRaises(ValueError):bus.write(11,11,1,4)

    def test_enable_matches_pose_before_any_torque(self):
        bus,writes,_,info=self.fake_bus()
        for item in info:item['drive']=4
        bus.enable(info,[2048]*6)
        self.assertEqual(writes[0][0],'position')
        self.assertIn((16,102,2,600),writes)
        self.assertTrue(all(x[1]>=64 for x in writes if isinstance(x[0],int)))
        self.assertEqual(sum(x[1]==64 for x in writes if isinstance(x[0],int)),6)

    def test_explicit_release_clears_and_verifies_all_six_motors(self):
        bus=Bus.__new__(Bus);bus.owned=True;writes=[]
        bus.write=lambda *args:writes.append(args)
        bus.read=lambda *args:0
        bus.release()
        self.assertFalse(bus.owned)
        self.assertEqual(set(writes),{(motor,address,1,0) for motor in IDS for address in (64,98)})

    def test_release_attempts_other_motors_when_one_fails(self):
        bus=Bus.__new__(Bus);bus.owned=True;writes=[]
        def write(*args):
            writes.append(args)
            if args[:2]==(11,64):raise RuntimeError('USB error')
        bus.write=write;bus.read=lambda *args:0
        with self.assertRaisesRegex(RuntimeError,'USB error'):bus.release()
        self.assertIn((16,64,1,0),writes)
        self.assertTrue(bus.owned)

    def test_enable_programs_path_profiles_and_manufacturer_grip_current(self):
        from teach_io.motion_timing import motor_profile
        bus,writes,_,info=self.fake_bus()
        bus.positions=lambda *args:None
        bus.enable(info,[2048]*6)
        for index,motor in enumerate(IDS):
            acceleration,velocity=motor_profile(index)
            self.assertIn((motor,108,4,acceleration),writes)
            self.assertIn((motor,112,4,velocity),writes)
        self.assertIn((16,102,2,600),writes)
        self.assertTrue(all(address>=64 for _,address,_,_ in writes))

    def test_profile_exceeding_stored_limit_is_rejected_before_any_write(self):
        bus=Bus.__new__(Bus);bus.owned=False;writes=[]
        bus.read=lambda *args:1
        bus.write=lambda *args:writes.append(args)
        bus.positions=lambda *args:writes.append(args)
        info=[dict(id=i,mode=4 if i<16 else 5,torque=0,watchdog=0,drive=0) for i in IDS]
        with self.assertRaisesRegex(RuntimeError,'Velocity Limit'):bus.enable(info,[2048]*6)
        self.assertFalse(bus.owned);self.assertEqual(writes,[])


    def test_manufacturer_gains_and_full_stored_pwm_applied_before_torque(self):
        bus,writes,registers,info=self.fake_bus()
        bus.enable(info,[2048]*6)
        first_torque=next(i for i,x in enumerate(writes) if x[:2]==(11,64))
        for motor in IDS:
            for address,value in ((80,1000),(82,0),(84,1000),(100,885)):
                self.assertIn((motor,address,2,value),writes[:first_torque])
        self.assertFalse(any(x[:2] in [(m,102) for m in IDS[:-1]] for x in writes))
        self.assertEqual(bus.power_profile['motors'][-1]['gripper_current_ma'],600)

    def test_lower_hardware_output_caps_are_respected(self):
        bus,writes,_,info=self.fake_bus(pwm_limit=300,current_limit=250)
        bus.enable(info,[2048]*6)
        self.assertIn((16,102,2,250),writes)
        for motor in IDS:self.assertIn((motor,100,2,300),writes)
        self.assertTrue(all(x[1]>=64 for x in writes if isinstance(x[0],int)))

    def test_output_readback_mismatch_never_enables_torque(self):
        bus,writes,_,info=self.fake_bus();read=bus.read
        bus.read=lambda m,a,n:999 if (m,a)==(13,84) else read(m,a,n)
        with self.assertRaisesRegex(RuntimeError,'readback'):bus.enable(info,[2048]*6)
        self.assertFalse(any(x[:2]==(m,64) for x in writes for m in IDS))

    def test_invalid_output_cap_fails_before_any_write(self):
        for pwm,current in ((0,1750),(886,1750),(885,0),(885,1751)):
            bus,writes,_,info=self.fake_bus(pwm_limit=pwm,current_limit=current)
            with self.assertRaisesRegex(RuntimeError,'Limit'):bus.enable(info,[2048]*6)
            self.assertEqual(writes,[])
            self.assertFalse(bus.owned)

    def test_unsupported_mode_refused_before_writes(self):
        bus=Bus.__new__(Bus);bus.owned=False
        info=[dict(id=i,mode=0,torque=0,watchdog=0,drive=0) for i in IDS]
        with self.assertRaises(RuntimeError):bus.enable(info,[2048]*6)
        self.assertFalse(bus.owned)


if __name__=='__main__':unittest.main()
