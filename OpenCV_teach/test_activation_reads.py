"""Replay corrupt startup replies through the installed SDK; no serial port opens."""
from types import SimpleNamespace
from unittest.mock import Mock
import unittest

from lerobot.motors.motors_bus import MotorsBus
import teach_left_arm as teaching


class ActivationReadTests(unittest.TestCase):
    def hardware(self, replies, *, name='wrist_roll', low=0, high=4095):
        # Use the real SDK retry/error handling with a packet transport double.
        packet=SimpleNamespace(read2ByteTxRx=Mock(side_effect=replies),
            getTxRxResult=lambda _: 'Incorrect status packet!',getRxPacketError=lambda _: 'motor alarm')
        bus=SimpleNamespace(packet_handler=packet,port_handler=object(),
            _is_comm_success=lambda code:code==0,_is_error=lambda error:error!=0)
        entry=SimpleNamespace(id=6 if name=='gripper' else 5,range_min=low,range_max=high)
        def read(register,motor,*,normalize,num_retry):
            self.assertEqual(register,'Present_Position');self.assertEqual(motor,name)
            self.assertFalse(normalize)
            return MotorsBus._read(bus,56,2,entry.id,num_retry=num_retry,
                                   raise_on_error=True,err_msg='Startup position read')[0]
        bus.read=read
        hardware=teaching.Hardware.__new__(teaching.Hardware)
        hardware.follower=SimpleNamespace(bus=bus,calibration={name:entry})
        return hardware,packet.read2ByteTxRx

    def test_one_corrupt_reply_recovers_without_substituting_an_old_position(self):
        hardware,read=self.hardware([(0,-3002,0),(2032,0,0)])
        self.assertEqual(hardware.current_raw('wrist_roll'),2032)
        self.assertEqual(read.call_count,2)

    def test_persistent_corruption_is_bounded_and_still_raises(self):
        hardware,read=self.hardware([(0,-3002,0)]*10)
        with self.assertRaises(ConnectionError):hardware.current_raw('wrist_roll')
        self.assertEqual(read.call_count,1+teaching.ACTIVATION_READ_RETRIES)

    def test_valid_packet_with_motor_alarm_is_not_retried_or_accepted(self):
        hardware,read=self.hardware([(2032,0,32),(2032,0,0)])
        with self.assertRaisesRegex(RuntimeError,'motor alarm'):hardware.current_raw('wrist_roll')
        self.assertEqual(read.call_count,1)

    def test_recovered_packet_must_still_pass_raw_and_gripper_calibration_bounds(self):
        for name,low,high,value in [('wrist_roll',0,4095,5000),('gripper',1996,3428,1985),('gripper',1996,3428,3439)]:
            with self.subTest(name=name,value=value):
                hardware,read=self.hardware([(0,-3002,0),(value,0,0)],name=name,low=low,high=high)
                with self.assertRaisesRegex(RuntimeError,'activation blocked'):hardware.current_raw(name)
                self.assertEqual(read.call_count,2)


if __name__=='__main__':unittest.main()
