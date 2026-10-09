import unittest
from unittest.mock import patch
from teach_io.modes import control_profile,release_supported,ROOT


class ModeTests(unittest.TestCase):
    def test_unchecked_candidate_cannot_be_used_for_real_mode(self):
        with self.assertRaises(ValueError):control_profile(ROOT/'config/follower_candidate.json')

    def test_operator_profile_requires_explicit_allowance(self):
        path=ROOT/'config/follower_visual_checked.json'
        with self.assertRaises(ValueError):control_profile(path)
        _,_,mapping=control_profile(path,True)
        self.assertFalse(mapping.cfg['verified'])

    @patch('teach_io.modes.subprocess.run')
    def test_release_requires_support_and_non_test_mode(self,run):
        for supported,allowed in [(False,True),(True,False)]:
            with self.assertRaises(ValueError):release_supported('/dev/fake',supported,allowed)
        run.assert_not_called()

    @patch('teach_io.modes.subprocess.run')
    def test_supported_release_uses_existing_checked_hardware_command(self,run):
        run.return_value.returncode=0;run.return_value.stdout='Torque OFF'
        self.assertEqual(release_supported('/dev/fake',True),'Torque OFF')
        args,kwargs=run.call_args
        self.assertEqual(args[0][-3:],['release','--port','/dev/fake'])
        self.assertEqual(kwargs['input'],'SUPPORTED\n')
