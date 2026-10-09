"""URDF-based offline IK for assisted demonstration, not a collision planner."""
import math
import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from arm_sim.common import ROOT
from arm_sim.factory import arm_base_height


class ArmKinematics:
    def __init__(self, model, pose, table_height):
        self.model = model
        self.so = 'shoulder_pan' in model['joints']
        self.base = np.eye(4)
        self.base[:3, :3] = Rotation.from_euler('z', pose['yaw']).as_matrix()
        self.base[:3, 3] = [*pose['position'], arm_base_height(pose,table_height)]
        urdf = ET.parse(ROOT / model['urdf']).getroot()
        self.fixed_pitch = pose.get('demo_pitch_degrees')
        if self.fixed_pitch is not None:
            if self.so:raise ValueError('Fixed pitch analytic IK is OMX-only')
            joints={j.get('name'):j for j in urdf.findall('joint')}
            def xyz(name):return np.fromstring(joints[name].find('origin').get('xyz'),sep=' ')
            self.shoulder_offset=xyz('joint1')+xyz('joint2')
            first=xyz('joint3');self.l1=float(np.linalg.norm(first));self.alpha=math.atan2(first[2],first[0])
            self.l2=float(xyz('joint4')[0])
            tool=xyz('end_effector_joint');self.tip=float(xyz('joint5')[0]+tool[0]);self.side=float(tool[1])
        parents = {j.find('child').get('link'): j for j in urdf.findall('joint')}
        link = 'gripper_frame_link' if self.so else 'end_effector_link'
        self.chain = []
        while link in parents:
            j = parents[link]
            origin = j.find('origin')
            transform = np.eye(4)
            transform[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
            transform[:3, :3] = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
            axis = None if j.get('type') == 'fixed' else np.fromstring(j.find('axis').get('xyz'), sep=' ')
            index = None if axis is None else model['joints'].index(j.get('name'))
            self.chain.insert(0, (transform, axis, index))
            link = j.find('parent').get('link')
        self.lower = np.array([lim[0] for lim in model['limits'][:5]])
        self.upper = np.array([lim[1] for lim in model['limits'][:5]])
        if self.so:
            # Keep the asymmetric finger on one consistent roll branch. Free
            # roll allowed the jaw to flip under the tabletop during grasping.
            self.lower[4]=-1e-6;self.upper[4]=1e-6
        # Keep OMX's periodic solutions on one practical branch for the demo.
        if not self.so:
            self.lower = np.maximum(self.lower, -math.pi)
            self.upper = np.minimum(self.upper, math.pi)
        self.max_error = 0.

    def transform(self, joints):
        transform = self.base.copy()
        for origin, axis, index in self.chain:
            transform = transform @ origin
            if axis is not None:
                rotation = np.eye(4)
                rotation[:3, :3] = Rotation.from_rotvec(axis * joints[index]).as_matrix()
                transform = transform @ rotation
        return transform

    def position(self, joints):
        return self.transform(joints)[:3, 3]

    def solve_pose(self, point, rotation, previous):
        """Fit a grasp pose, rejecting position/orientation compromises."""
        point=np.asarray(point);rotation=np.asarray(rotation);previous=np.asarray(previous[:5])
        limits=np.asarray(self.model['limits'][:5]);lower=limits[:,0].copy();upper=limits[:,1].copy()
        # Use the actual URDF limits. The previous pose is the first seed, so
        # small Cartesian steps continue its branch across +/-pi when allowed.
        def residual(q):
            tf=self.transform(q)
            return np.r_[100*(tf[:3,3]-point),10*Rotation.from_matrix(rotation.T@tf[:3,:3]).as_rotvec()]
        seeds=[previous,self.model['home'][:5],[0,1,-1,1,0],[1,1,-1,1,0],[-1,1,-1,1,0]]
        best=None
        for seed in seeds:
            sol=least_squares(residual,np.clip(seed,lower+1e-7,upper-1e-7),bounds=(lower,upper),max_nfev=120,ftol=1e-8,xtol=1e-8)
            tf=self.transform(sol.x);pe=float(np.linalg.norm(tf[:3,3]-point));ae=float(Rotation.from_matrix(rotation.T@tf[:3,:3]).magnitude())
            if best is None or np.linalg.norm(residual(sol.x))<best[0]:best=(np.linalg.norm(residual(sol.x)),pe,ae,sol.x)
            if pe<.0001 and ae<.001:
                self.max_error=max(self.max_error,pe);return sol.x
        raise ValueError(f'Grasp pose unreachable: {point.tolist()}, position {best[1]*1000:.2f} mm, orientation {math.degrees(best[2]):.2f} deg')

    def solve(self, point, previous):
        if self.fixed_pitch is not None:
            return self.solve_fixed_pitch(point,previous)
        point = np.asarray(point, dtype=float)
        previous = np.asarray(previous[:5])
        def residual(q):
            t = self.transform(q)
            down = t[:3, 2 if self.so else 0] - [0, 0, -1]
            return np.r_[100 * (t[:3, 3] - point), .025 * down, .001 * (q - previous)]
        best = None
        seeds = [previous, self.model['home'][:5], [0, .8, -1, 1, 0], [1, -.8, 1, 0, 0], [-1, -.8, 1, 0, 0]]
        for seed in seeds:
            sol = least_squares(residual, np.clip(seed, self.lower + 1e-7, self.upper - 1e-7),
                                bounds=(self.lower, self.upper), max_nfev=100, ftol=1e-7, xtol=1e-7)
            error = float(np.linalg.norm(self.position(sol.x) - point))
            if best is None or error < best[0]:
                best = error, sol.x
            if error < .001:
                break
        error, q = best
        if not np.isfinite(q).all() or error > .003:
            raise ValueError(f'Unreachable demo waypoint {point.tolist()}: {error*1000:.1f} mm residual')
        self.max_error = max(self.max_error, error)
        return q

    def solve_fixed_pitch(self, point, previous):
        """Elbow-up OMX branch, fixed wrist pitch and roll, continuous base yaw."""
        p=np.linalg.inv(self.base)@np.r_[point,1.]
        dx,dy,dz=p[:3]-self.shoulder_offset
        radius2=dx*dx+dy*dy-self.side*self.side
        if radius2<=0:raise ValueError('OMX target is inside the shoulder axis')
        radius=math.sqrt(radius2)
        angle=(np.interp(radius,[.12,.36],self.fixed_pitch)
               if isinstance(self.fixed_pitch,list) else self.fixed_pitch)
        pitch=math.radians(float(angle))
        yaw=math.atan2(dy,dx)-math.atan2(self.side,radius)
        yaw=previous[0]+(yaw-previous[0]+math.pi)%(2*math.pi)-math.pi
        x=radius-self.tip*math.cos(pitch);z=dz+self.tip*math.sin(pitch)
        cosine=(x*x+z*z-self.l1*self.l1-self.l2*self.l2)/(2*self.l1*self.l2)
        if abs(cosine)>1.+1e-9:
            raise ValueError(f'OMX fixed-pitch waypoint unreachable: {list(point)}, cosine={cosine:.4f}')
        delta=math.acos(np.clip(cosine,-1,1))
        upper=math.atan2(z,x)+math.atan2(self.l2*math.sin(delta),self.l1+self.l2*math.cos(delta))
        q2=self.alpha-upper;q3=delta-self.alpha;q4=pitch-q2-q3
        q=np.array([yaw,q2,q3,q4,0.])
        limits=np.array(self.model['limits'][:5])
        if (q<limits[:,0]).any() or (q>limits[:,1]).any():raise ValueError('OMX analytic branch exceeds joint limits')
        error=float(np.linalg.norm(self.position(q)-point))
        if error>.0001:raise ValueError(f'OMX analytic FK mismatch: {error}')
        self.max_error=max(self.max_error,error)
        return q
