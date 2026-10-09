"""An explicitly captured reference for one stationary docking session.

Missing detections do not move the reference. This latch is memory-only: saved
teaching evidence is never restored as a live reference on launch.
"""
from copy import deepcopy
from datetime import datetime
import math
import uuid

from vision_teaching import calibration_signature, reference_key, angle_delta, number
from jig_detection import profile_signature, validate_profile


class JigReference:
    def __init__(self):
        self.value = None
        self.reason = '아직 저장하지 않았습니다.'

    def clear(self, reason='기준을 해제했습니다. 다시 읽어 주세요.'):
        self.value = None
        self.reason = reason

    def capture(self, target, observation, station):
        if target['mode'] != 'rectangle':
            raise ValueError('잡기·놓기에서 위치 기준을 지그를 따라가기로 선택하세요.')
        signature = profile_signature(target)
        if observation.get('profile_signature') != signature:
            raise ValueError('현재 검출값과 지그 설정이 다릅니다. 다시 읽어 주세요.')
        for key in ('x_mm', 'y_mm', 'yaw_deg'):
            number(observation[key])
        self.value = {'id': uuid.uuid4().hex, 'key': reference_key(target),
            'profile': validate_profile(target), 'profile_signature': signature,
            'calibration_signature': calibration_signature(station),
            'captured_at': datetime.now().astimezone().isoformat(),
            'observation': deepcopy(observation),
            'scope': 'operator_confirmed_stationary_docking_session',
            'docking_or_safe_pose_automatically_verified': False}
        self.reason = ''

    def check(self, target, station):
        if self.value is None:
            raise ValueError('먼저 팔을 안전 자세에 두고 도킹을 마친 뒤 지그 기준을 저장하세요. '+self.reason)
        reason = None
        if self.value['calibration_signature'] != calibration_signature(station):
            reason = '카메라·로봇·집게 끝 보정이 바뀌었습니다. 지그 기준을 다시 읽어 주세요.'
        elif target['mode'] != 'rectangle' or self.value['key'] != reference_key(target) or self.value['profile_signature'] != profile_signature(target):
            reason = '지그 종류·크기·높이·방향 설정이 바뀌었습니다. 기준을 다시 읽어 주세요.'
        if reason:
            self.clear(reason)
            raise ValueError(reason)
        return deepcopy(self.value)

    def observations(self, plan, station):
        result = {}
        for target in plan['targets'].values():
            if target['mode'] != 'rectangle':
                continue
            value = self.check(target, station)
            result[value['key']] = {**value['observation'],
                                   'reference_id': value['id'], 'reference_captured_at': value['captured_at']}
        return result

    def observe_current(self, observation, *, position_tolerance=1., angle_tolerance=1.):
        """Only a new, stable measurement can establish movement; occlusion cannot."""
        if self.value is None or observation.get('profile_signature') != self.value['profile_signature']:
            return False
        old = self.value['observation']
        moved = math.hypot(old['x_mm']-observation['x_mm'], old['y_mm']-observation['y_mm']) > position_tolerance
        moved |= abs(angle_delta(old['yaw_deg'], observation['yaw_deg'])) > angle_tolerance
        if moved:
            self.clear('지그 이동이 감지됐습니다. 도킹과 팔 대기 상태를 확인하고 기준을 다시 읽어 주세요.')
        return moved


class JigReferences(JigReference):
    """Independent latches keyed by jig identity; value is the active UI latch."""
    def __init__(self):
        super().__init__();self.values={}

    def clear(self, reason='기준을 해제했습니다. 다시 읽어 주세요.'):
        super().clear(reason);self.values={}

    def capture(self,target,observation,station):
        ref=JigReference();ref.capture(target,observation,station)
        self.values[ref.value['key']]=ref.value;self.value=ref.value;self.reason=''

    def capture_many(self,targets,observations,station):
        staged=JigReferences()
        for target in targets:
            if target['mode']=='rectangle':staged.capture(target,observations[reference_key(target)],station)
        self.values.update(staged.values)
        if staged.value is not None:self.value=staged.value
        self.reason=''

    def check(self,target,station):
        key=reference_key(target)
        value=self.values.get(key)
        if value is None:raise ValueError((self.reason+' ' if self.reason else '')+'먼저 '+target.get('jig_name','지그')+'의 티칭 기준을 읽어 주세요.')
        ref=JigReference();ref.value=value
        try:result=ref.check(target,station)
        except ValueError:
            self.values.pop(key,None)
            if self.value and self.value['key']==key:self.value=None
            self.reason=ref.reason
            raise
        self.value=value
        return result

    def observe_key(self,key,observation,**kwargs):
        ref=JigReference();ref.value=self.values.get(key)
        if ref.observe_current(observation,**kwargs):
            self.clear(ref.reason);return True
        return False

    def observe_current(self,observation,**kwargs):
        return self.observe_key(self.value['key'],observation,**kwargs) if self.value else False
