"""Offline playback validation and deterministic clock/transport checks."""

import copy
import math
from pathlib import Path
import tempfile
import unittest

from episode_preview import EpisodePlayer, read_episode
import teach_left_arm as teaching


def document():
    start = dict.fromkeys(teaching.KEYS, 0.)
    goal = dict(start, **{"shoulder_pan.pos": 30., "gripper.pos": 40.})
    return {"schema": 1, "identity": {"demo": False, "pair_name": "offline_robot"},
            "units": teaching.UNITS, "title": "점검", "steps": [
                {"name": "시작", "pose": start}, {"name": "도착", "pose": goal}]}


class EpisodePlayerTests(unittest.TestCase):
    def test_hardware_document_and_immutable_snapshot(self):
        data = document()
        player = EpisodePlayer()
        player.load(data)
        self.assertEqual(player.origin, "실물 파일")
        data["steps"][0]["pose"]["shoulder_pan.pos"] = 123
        self.assertEqual(player.sample(0)[0]["shoulder_pan.pos"], 0)
        with self.assertRaises(ValueError):
            teaching.validate_program(document(), teaching.Demo.identity)

    def test_interpolation_pause_resume_speed_seek_and_completion(self):
        player = EpisodePlayer()
        player.load(document())
        player.play(now=0)
        midpoint, index, phase = player.sample(now=.9)
        self.assertAlmostEqual(midpoint["shoulder_pan.pos"], 15)
        self.assertAlmostEqual(midpoint["gripper.pos"], 20)
        self.assertEqual(index, 1)
        player.pause(now=.9)
        self.assertEqual(player.sample(now=100)[0], midpoint)
        player.set_speed(2., now=100)
        player.play(now=100)
        player.sample(now=101)
        self.assertEqual(player.position, player.duration)
        self.assertFalse(player.running)
        player.play(now=102)
        self.assertEqual(player.position, 0.)
        player.seek(player.step_time(1), now=102)
        self.assertEqual(player.sample(now=102)[0], document()["steps"][1]["pose"])
        player.reset(now=103)
        self.assertFalse(player.running)
        self.assertEqual(player.position, 0.)

    def test_empty_single_step_and_repeated_poses(self):
        player = EpisodePlayer()
        player.play(now=0)
        self.assertFalse(player.running)
        data = document()
        data["steps"] = data["steps"][:1]
        player.load(data)
        self.assertGreater(player.duration, 0)
        player.play(now=0)
        pose, index, phase = player.sample(now=10)
        self.assertEqual(pose, data["steps"][0]["pose"])
        self.assertEqual(phase, "재생 완료")
        data["steps"].append(copy.deepcopy(data["steps"][0]))
        player.load(data)
        player.seek(player.duration)
        self.assertEqual(player.sample()[1], 1)

    def test_invalid_file_does_not_replace_valid_episode(self):
        player = EpisodePlayer()
        player.load(document())
        for change in (lambda d: d.update(schema=99), lambda d: d.update(units={}),
                       lambda d: d.update(steps=[]), lambda d: d.update(identity=None),
                       lambda d: d["steps"][1]["pose"].update({"gripper.pos": 101}),
                       lambda d: d["steps"][1]["pose"].update({"shoulder_pan.pos": math.nan})):
            data = document()
            change(data)
            with self.assertRaises(ValueError):
                player.load(data)
            self.assertEqual(len(player.steps), 2)
            self.assertEqual(player.title, "점검")
        with self.assertRaises(ValueError):
            player.set_speed(10)

    def test_file_size_and_json_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "invalid.json"
            path.write_text("not JSON")
            with self.assertRaises(ValueError):
                read_episode(path)
            with path.open("wb") as file:
                file.truncate(10 * 1024 * 1024 + 1)
            with self.assertRaisesRegex(ValueError, "10MB"):
                read_episode(path)


if __name__ == "__main__":
    unittest.main()
