"""Explicit, independently measured follower calibration (URDF radians)."""
import math

IDS = list(range(11, 17))
MODELS = [1060, 1060, 1060, 1200, 1200, 1200]
JOINTS = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'gripper_joint_1']
TICKS_PER_RAD = 4096 / math.tau
START_POSE_TOLERANCE_RAD = math.radians(10)
OBSERVATION_BOUNDARY_TOLERANCE_RAD = START_POSE_TOLERANCE_RAD
HARD_LIMITS = [[math.radians(a), math.radians(b)] for a,b in
               [(-270,360),(-120,90),(-120,90),(-100,100),(-270,270)]] + [[0,1.1]]


def check_physical_limits(q):
    for name,value,(lo,hi) in zip(JOINTS,vector(q),HARD_LIMITS):
        if not lo <= value <= hi:
            raise ValueError(f'{name}: target exceeds OMX-F physical range')


def vector(values, n=6):
    if len(values) != n or not all(isinstance(x, (float, int)) and math.isfinite(x) for x in values):
        raise ValueError(f'Expected {n} finite joint values')
    return list(values)


class Mapping:
    def __init__(self, cfg, allow_visual_confirmation=False):
        confirmation=cfg.get('operator_confirmation',{})
        visually_checked=(allow_visual_confirmation and
            confirmation.get('method')=='user_visual_real_to_sim' and
            confirmation.get('accepted_for_low_speed_control') is True and
            bool(confirmation.get('confirmed_at')))
        if cfg.get('kind') != 'omx_follower_urdf' or not (cfg.get('verified') is True or visually_checked):
            raise ValueError('A verified follower calibration is required; leader calibration is not accepted')
        if cfg.get('ids') != IDS or cfg.get('models') != MODELS:
            raise ValueError('Expected OMX-F IDs 11–16 / XL430 + XL330')
        self.cfg = cfg
        self.zero = vector(cfg['zero_ticks'])
        self.scale = vector(cfg['ticks_per_rad'])
        for s in self.scale[:5]:
            if abs(abs(s) / TICKS_PER_RAD - 1) > .08:
                raise ValueError('Arm calibration scale must match 4096 ticks/revolution')
        if abs(self.scale[5]) < 100:
            raise ValueError('Gripper calibration span is too small')
        if len(cfg['limits']) != 6 or len(cfg['tick_limits']) != 6:
            raise ValueError('Six joint and encoder limits are required')
        for ranges in (cfg['limits'], cfg['tick_limits']):
            for lo, hi in ranges:
                if not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi:
                    raise ValueError('Invalid calibration limits')

    def to_ticks(self, q):
        q = vector(q)
        check_physical_limits(q)
        ticks = []
        for i, angle in enumerate(q):
            lo, hi = self.cfg['limits'][i]
            if not lo <= angle <= hi:
                raise ValueError(f'{JOINTS[i]} outside calibrated angular range')
            value = round(self.zero[i] + angle * self.scale[i])
            lo, hi = self.cfg['tick_limits'][i]
            if not lo <= value <= hi:
                raise ValueError(f'{JOINTS[i]} outside calibrated encoder range')
            ticks.append(value)
        return ticks

    def to_angles(self, ticks):
        return [(t-z)/s for t,z,s in zip(vector(ticks), self.zero, self.scale)]

    def clamp_target(self, q):
        """Intersect physical, calibrated and encoder ranges for command targets."""
        result=[]
        for i,value in enumerate(vector(q)):
            a,b=sorted((t-self.zero[i])/self.scale[i] for t in self.cfg['tick_limits'][i])
            lo=max(a,HARD_LIMITS[i][0],self.cfg['limits'][i][0])
            hi=min(b,HARD_LIMITS[i][1],self.cfg['limits'][i][1])
            if lo>hi:raise ValueError(f'{JOINTS[i]}: incompatible calibration limits')
            result.append(max(lo,min(hi,value)))
        return result

    def validate_observation(self, q):
        """Accept a 10-degree start observation margin, never as a new target.

        This checks admission of a measured pose for entry geometry auditing;
        it does not widen command limits or the separate entry-settle criterion.
        """
        q=vector(q);bounded=self.clamp_target(q)
        for name,a,b in zip(JOINTS,q,bounded):
            if abs(a-b)>OBSERVATION_BOUNDARY_TOLERANCE_RAD+1e-12:
                raise ValueError(f'{name}: measured {math.degrees(a):.2f} deg is '
                    f'{math.degrees(abs(a-b)):.2f} deg outside command boundary '
                    f'{math.degrees(b):.2f} deg; start observation allowance is 10 deg')
        return q


def fit_arm_pose(q1, ticks1, q2, ticks2):
    """Two measured URDF poses; do not infer joint direction from leader data."""
    for v in (q1, ticks1, q2, ticks2):
        vector(v, 5)
    scale, zero = [], []
    for a, ta, b, tb in zip(q1, ticks1, q2, ticks2):
        if abs(b-a) < .15:
            raise ValueError('Each joint must differ by at least 0.15 rad between measured poses')
        slope = (tb-ta)/(b-a)
        if abs(abs(slope)/TICKS_PER_RAD-1) > .08:
            raise ValueError('Pose measurements do not match encoder travel; check angles/direction/wrap')
        # The encoder scale is known. Measure sign and average offset only.
        slope = math.copysign(TICKS_PER_RAD, slope)
        scale.append(slope)
        zero.append(((ta-a*slope)+(tb-b*slope))/2)
    return zero, scale
