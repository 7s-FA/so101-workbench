"""Offline episode playback. This module has no robot/session/serial handles."""

from bisect import bisect_right
import json
from pathlib import Path
import time

from teaching_settings import COMMAND_RATE, GRIPPER_COMMAND_RATE, MAX_NAME_LENGTH

MAX_FILE_BYTES = 10 * 1024 * 1024
STEP_HOLD_SECONDS = .4


def read_episode(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_FILE_BYTES + 1)
    if len(raw) > MAX_FILE_BYTES:
        raise ValueError("에피소드 파일은 10MB 이하만 열 수 있습니다.")
    return json.loads(raw.decode("utf-8-sig"))


class EpisodePlayer:
    """An immutable copy of poses, interpolated from the first saved step.

    Playback validates the file schema using its original robot identity.
    This permits visual inspection without hardware or calibration; robot
    execution validates identity separately. Nothing is saved.
    """
    def __init__(self):
        self.steps, self.segments, self.ends = [], [], []
        self.title, self.origin = "에피소드 없음", ""
        self.duration, self.position, self.speed = 0., 0., 1.
        self.running, self.last_time, self.revision = False, None, 0

    def load(self, document, fallback_title="이름 없음"):
        from teach_left_arm import validate_program
        if not isinstance(document, dict) or not isinstance(document.get("identity"), dict):
            raise ValueError("SO-101 에피소드의 로봇 정보가 없습니다.")
        identity = document["identity"]
        if type(identity.get("demo")) is not bool:
            raise ValueError("에피소드의 데모/실물 구분을 확인할 수 없습니다.")
        steps = validate_program(document, identity)
        if not steps:
            raise ValueError("확인할 스텝이 없습니다.")
        segments, total = [], 0.
        for index, step in enumerate(steps):
            if step.get('motion'):
                if not step.get('path'):
                    raise ValueError('상단 카메라 · 집기 설정에서 목표 경로를 먼저 계산하세요.')
                start = steps[index-1]['pose'] if index else step['path'][0]['pose']
                for node in step['path']:
                    target = node['pose']
                    seconds = max(node['duration_s'], max(abs(target[key]-start[key]) /
                                  (GRIPPER_COMMAND_RATE if key=='gripper.pos' else COMMAND_RATE) for key in target))
                    if seconds>0:
                        segments.append((total,total+seconds,index,start,target));total+=seconds
                    if node['hold_s']>0:
                        segments.append((total,total+node['hold_s'],index,target,target));total+=node['hold_s']
                    start=target
                continue
            if step.get("gripper_only") and index:
                step["pose"]={**steps[index-1]["pose"],"gripper.pos":step["pose"]["gripper.pos"]}
            target = step["pose"]
            if index:
                start = steps[index - 1]["pose"]
                seconds = max(abs(target[key] - start[key]) /
                              (GRIPPER_COMMAND_RATE if key == "gripper.pos" else COMMAND_RATE)
                              for key in target)
                if seconds > 0:
                    segments.append((total, total + seconds, index, start, target))
                    total += seconds
            segments.append((total, total + STEP_HOLD_SECONDS, index, target, target))
            total += STEP_HOLD_SECONDS
        # Commit only after the complete document has passed validation.
        self.steps, self.segments = steps, segments
        self.ends = [segment[1] for segment in segments]
        self.duration, self.position, self.running, self.last_time = total, 0., False, None
        title = document.get("title")
        self.title = (title.strip() if isinstance(title, str) and title.strip() else fallback_title)[:MAX_NAME_LENGTH]
        self.origin = "데모 파일" if identity["demo"] else "실물 파일"
        self.revision += 1

    def update(self, now=None):
        now = time.monotonic() if now is None else now
        if self.running and self.last_time is not None:
            self.position = min(self.duration, self.position + max(0., now - self.last_time) * self.speed)
            if self.position >= self.duration:
                self.running = False
        self.last_time = now

    def play(self, now=None):
        if not self.steps:
            return
        if self.position >= self.duration:
            self.position = 0.
        self.running = True
        self.last_time = time.monotonic() if now is None else now

    def pause(self, now=None):
        self.update(now)
        self.running = False

    def seek(self, position, now=None):
        self.position = min(self.duration, max(0., float(position)))
        self.last_time = time.monotonic() if now is None else now

    def reset(self, now=None):
        self.running = False
        self.seek(0., now)

    def set_speed(self, speed, now=None):
        if speed not in (.5, 1., 2.):
            raise ValueError("시뮬레이션 재생 속도는 0.5 / 1 / 2배입니다.")
        self.update(now)
        self.speed = speed

    def step_time(self, index):
        return next(start for start, end, i, source, target in self.segments if i == index and
                    (source is target or self.steps[index].get('motion')))

    def sample(self, now=None):
        self.update(now)
        if not self.segments:
            return None, 0, "대기"
        segment = self.segments[min(bisect_right(self.ends, self.position), len(self.segments) - 1)]
        start, end, index, source, target = segment
        fraction = min(1., max(0., (self.position - start) / (end - start)))
        pose = {key: value + (target[key] - value) * fraction for key, value in source.items()}
        phase = "재생 중" if self.running else "재생 완료" if self.position >= self.duration else "일시정지" if self.position else "재생 준비"
        return pose, index, phase
