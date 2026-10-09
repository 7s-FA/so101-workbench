"""B-only training contracts, imported from the current teaching application."""
from bisect import bisect_right
from copy import deepcopy, copy
import hashlib
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
TEACH = PROJECT / 'SO101_teach'
DATA = TEACH / 'data'
TEST = PROJECT / 'Test_teach'
SIM = PROJECT / 'Sim_real'
EPISODE = ROOT / 'config/reference_episodes/a03c1949d9c04ad1ba46e6267fa64c8e.json'
sys.path[:0] = [str(TEACH), str(SIM)]
from so101_teach.domain import JOINTS, Calibration, ModelReference, EpisodeStore, read_json, atomic_json
from so101_teach.geometry import Kinematics, corrected_plan
from so101_teach.joint_path import JointPath

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class Reference:
    """Same 32 ticks, calibrated mapping and current short-ramp interpolation."""
    def __init__(self, episode=EPISODE, rate=350, jig_pose=None):
        self.path = Path(episode)
        self.document = read_json(self.path)
        if self.document['robot_id'] != 'arm2' or self.document['id'] != EPISODE.stem:
            raise ValueError('현재 시험은 로봇팔2의 완제품 B 에피소드만 지원합니다.')
        profiles = read_json(DATA/'robot_profiles.json')
        candidates = [p for p in profiles.values() if p.get('robot_id', 'arm2') == 'arm2'
                      and p['model_reference']['calibration_sha256'] == self.document['calibration_sha256']]
        if len(candidates) != 1:
            raise ValueError('완제품 B와 일치하는 로봇팔2 보정 프로필이 하나여야 합니다.')
        self.profile = deepcopy(candidates[0])
        self.calibration_path = DATA/self.profile['calibration_file']
        self.calibration = Calibration(self.calibration_path)
        self.mapping = ModelReference(self.calibration, self.profile['model_reference'])
        EpisodeStore(DATA/'episodes', self.calibration, 'arm2').validate(self.document)
        self.kin = Kinematics(self.mapping, tcp=self.profile['tcp'])
        self.steps = self.document['steps']
        if len(self.steps) != 32:
            raise ValueError('B 에피소드 스텝이 바뀌었습니다. 제품별 보상 단계 매핑을 갱신하세요.')
        self.jig_references = {}
        for step in self.steps:
            if not step.get('jig_id'):
                continue
            key, value = step['jig_id'], step['jig_reference']
            if key in self.jig_references and self.jig_references[key]['stl_sha256'] != value['stl_sha256']:
                raise ValueError('한 에피소드에 서로 다른 운반 지그 STL이 있습니다.')
            self.jig_references.setdefault(key, deepcopy(value))
        self.taught_jig_references=deepcopy(self.jig_references)
        if len(self.jig_references)!=1:raise ValueError('현재 B 장면은 운반 지그 한 개를 지원합니다.')
        self.jig_id=next(iter(self.jig_references))
        self.rate=rate
        self.limits = np.array([self.mapping.angle_limits(n) for n in JOINTS])
        self.mesh_hashes={key:read_json(DATA/'jigs.json')[key]['mesh']['sha256'] for key in self.jig_references}
        if jig_pose is not None:self.jig_references[self.jig_id]['pose']=self.checked_jig_pose(jig_pose)
        self._compile()

    @staticmethod
    def checked_jig_pose(pose):
        value=np.asarray(pose,dtype=float)
        if value.shape!=(3,) or not np.isfinite(value).all() or np.any(abs(value[:2])>1000) or abs(value[2])>360:
            raise ValueError('지그 위치는 X/Y ±1000mm, 방향 ±360°의 유한한 값이어야 합니다.')
        return value.tolist()

    def _compile(self):
        # Always solve from the original teaching, never from a previously
        # corrected path (otherwise repeated edits accumulate drift).
        plan = corrected_plan(self.kin, self.document['steps'], self.jig_references,
                              lambda key: self.mesh_hashes[key])
        targets = [row['ticks'] for row in plan]
        self.corrections = plan
        self.steps = [{**deepcopy(s), 'ticks': row['ticks']} for s,row in zip(self.document['steps'],plan)]
        self.joint_path = JointPath(targets[0], targets, rate=self.rate)
        self.ends = np.cumsum([s.duration for s in self.joint_path.segments])
        self.duration = float(self.ends[-1])

    def retarget(self, pose):
        result=copy(self)
        result.jig_references=deepcopy(self.taught_jig_references)
        result.jig_references[self.jig_id]['pose']=self.checked_jig_pose(pose)
        result._compile()
        return result

    @property
    def jig_pose(self):return list(self.jig_references[self.jig_id]['pose'])

    def sample(self, seconds):
        index = min(bisect_right(self.ends, seconds), len(self.steps)-1)
        return np.array(self.mapping.angles(self.joint_path.sample(seconds))), index

def merge_robot(destination):
    """Latest teaching geometry/joints, validated Test_teach inertia/contacts."""
    source = TEACH/'assets/so101/so101_modified.urdf'
    physics = TEST/'assets/so101/so101_teach.urdf'
    root, old = ET.parse(source).getroot(), ET.parse(physics).getroot()
    def signature(node):
        return node.tag, sorted(node.attrib.items()), [signature(c) for c in node]
    if [signature(j) for j in root.findall('joint')] != [signature(j) for j in old.findall('joint')]:
        raise ValueError('최신 관절 구조와 이전 물성 모델이 달라 물성을 자동 적용할 수 없습니다.')
    for link in root.findall('link'):
        other = old.find(f"link[@name='{link.get('name')}']")
        if other is None:
            raise ValueError('물성 링크 누락: ' + link.get('name'))
        for visual in link.findall('visual'):
            mesh = visual.find('geometry/mesh')
            if mesh is None:
                continue
            original = (source.parent/mesh.get('filename')).resolve()
            matches = [v for v in other.findall('visual') if v.find('geometry/mesh') is not None
                       and Path(v.find('geometry/mesh').get('filename')).name == original.name]
            match = matches[0].find('geometry/mesh') if len(matches) == 1 else None
            if match is None or sha(original) != sha((physics.parent/match.get('filename')).resolve()):
                raise ValueError('최신 메시와 물성 모델 불일치: ' + str(original))
            a,b = deepcopy(visual),deepcopy(matches[0])
            a.find('geometry/mesh').set('filename','same');b.find('geometry/mesh').set('filename','same')
            if signature(a) != signature(b):
                raise ValueError('메시 배치/스케일과 물성 모델 불일치: '+str(original))
            mesh.set('filename', str(original))
        for tag in ('inertial', 'collision'):
            for node in list(link.findall(tag)):
                link.remove(node)
            for node in other.findall(tag):
                node = deepcopy(node)
                for mesh in node.findall('.//mesh'):
                    mesh.set('filename', str((physics.parent/mesh.get('filename')).resolve()))
                link.append(node)
        for visual in other.findall('visual'):
            if visual.get('name', '').startswith('razer_'):
                visual = deepcopy(visual)
                mesh = visual.find('geometry/mesh')
                mesh.set('filename', str((physics.parent/mesh.get('filename')).resolve()))
                link.append(visual)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root)
    ET.ElementTree(root).write(destination, encoding='utf-8', xml_declaration=True)
    return destination

def bounded_target(reference, action, previous, limits, dt):
    action = np.asarray(action, dtype=float)
    if action.shape != (6,) or not np.isfinite(action).all():
        raise ValueError('정책 행동은 유한한 관절 보정값 6개여야 합니다.')
    requested = np.clip(reference + np.clip(action, -1, 1)*np.radians([2,2,2,2,2,3]), limits[:,0], limits[:,1])
    return previous + np.clip(requested-previous, -.65*dt, .65*dt)

def task_success(lifted, errors_mm, speeds_m_s, released):
    """Rigid placement only; not TPU insertion or snap-fit certification."""
    return bool(all(lifted) and max(errors_mm) <= 8 and max(speeds_m_s) < .02 and released)

def prepare(output, episode=EPISODE, rate=350, jig_pose=None, cad_offset=None,cad_yaw=None,receiver_offset=None,cad_tilt=None,nest_offsets=None):
    from scene_assets import prepare_scene
    ref = Reference(episode, rate, jig_pose=jig_pose)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    robot = merge_robot(output/'assets/so101_rl.urdf')
    scene = prepare_scene(ref, output)
    from alignment import aligned_scene
    alignment_path=ROOT/'config/cad_alignment.json';alignment=read_json(alignment_path)
    scene=aligned_scene(scene,ref,alignment['offset_mm'] if cad_offset is None else cad_offset,
                        alignment.get('yaw_deg',0.) if cad_yaw is None else cad_yaw,
                        alignment.get('tilt_deg',[0.,0.]) if cad_tilt is None else cad_tilt)
    from alignment import align_nests
    scene=align_nests(scene,alignment.get('nest_offsets_mm',{}) if nest_offsets is None else nest_offsets,output)
    from assembly import build_assembly
    assembly_path=ROOT/'config/assembly.json';assembly_settings=read_json(assembly_path)
    if receiver_offset is not None:assembly_settings['receiver_offset_mm']=receiver_offset
    scene['assembly']=build_assembly(scene,ref,assembly_settings)
    for kind,seats in scene['assembly']['magnet_seats'].items():
        mass=sum(np.pi*(s['diameter_mm']/2000)**2*s['depth_mm']/1000*assembly_settings['magnets']['density_kg_m3'] for s in seats)
        scene['parts'][kind]['magnet_mass_kg']=float(mass)
        scene['parts'][kind]['mass_kg']+=float(mass)
        scene['parts'][kind]['magnet_mass_basis']='Hole volume and provisional 7500 kg/m3 density; not weighed'
    for kind,row in scene['parts'].items():row['target_pose_mm']=deepcopy(scene['assembly']['target_poses_mm'][kind])
    scene['target_basis']='CAD pallet seating, TPU nesting, cap supported with a nonzero gap and four magnet pairs'
    physics = read_json(TEST/'config/so101_scene.json')
    scene.update(robot=str(robot), dt=physics['dt'], actuators=deepcopy(physics['actuators']),
                 pad_contact=physics['contact'], rate_ticks_s=rate)
    scene['actuators'].pop('gripper_hold_effort_nm',None)
    response_path=ROOT/'config/servo_response.json'
    response=read_json(response_path)
    scene['actuators'].update(response['actuators'])
    scene['pad_contact'].update(response.get('pad_contact',{}))
    scene['servo_response']=response
    act=scene['actuators']
    load_path=ROOT/'config/gripper_load_model.json';gripper_model=read_json(load_path)
    if gripper_model['calibration_sha256']!=ref.calibration.sha256:
        raise ValueError('집게 부하 추정 기록과 현재 로봇팔2 보정이 다릅니다.')
    arm_load_path=ROOT/'config/arm_load_model.json';arm_model=read_json(arm_load_path)
    if arm_model['calibration_sha256']!=ref.calibration.sha256:
        raise ValueError('팔 부하 추정 기록과 현재 로봇팔2 보정이 다릅니다.')
    scene['load_model']=dict(mode='recorded_stationary_errors_with_arm_effort_fallback',hardware_calibrated=False,
        full_scale_effort_nm=[act['wrist_effort_nm'] if n.startswith('wrist') else
                             act['gripper_effort_nm'] if n=='gripper' else act['arm_effort_nm'] for n in JOINTS],
        raw_full_scale=1023,gripper_contact_load=ref.profile.get('grip_contact_load',80),gripper=gripper_model,arm=arm_model,
        gripper_estimate_scope='Only episode segments with an explicit decrease in requested grip ticks; no synthetic closing load while holding or opening',
        limitation='Recorded error/load approximation. Arm uses it only when stationary and inside recorded range, otherwise effort fraction. Neither is a measured STS raw-load to Nm calibration.',
        grip_hold='Same teaching contact position target; no separate phase-based effort reduction')
    sources = [ref.path, load_path, arm_load_path, response_path, alignment_path, assembly_path, DATA/'robot_profiles.json', DATA/'workcell-preview.json', DATA/'jigs.json',
               ref.calibration_path, ref.calibration_path.with_suffix('.angles.json'),
               TEST/'config/so101_scene.json', TEST/'config/materials.json',
               TEST/'assets/so101/physics_provenance.json', TEST/'assets/so101/so101_teach.urdf',
               TEACH/'assets/so101/so101_modified.urdf', SIM/'config/assembly.json']
    sources += [TEACH/'so101_teach'/name for name in ('domain.py','geometry.py','angle_mapping.py',
                 'joint_path.py','motion.py','gripper_contact.py','load_compensation.py','devices.py',
                 'communication.py','telemetry.py','preview.py','workcell_preview.py','workcell_scene.py','height_reference.py')]
    sources += [Path(p) for p in scene['source_files']]
    sources += list(ROOT.glob('*.py'))
    report = dict(schema=1, robot='so101', robot_id='arm2', engine='Isaac Sim / PhysX',
                  episode=ref.document['name'], step_count=len(ref.steps), duration_s=ref.duration,
                  hardware_commands=False, real_world_validation=False,
                  sources={str(p):sha(p) for p in sources},
                  applied=dict(actuators=scene['actuators'],servo_response=response,cad_offset_mm=scene['carrier']['cad_offset_mm'],cad_yaw_deg=scene['carrier']['cad_yaw_deg'],cad_tilt_deg=scene['carrier']['cad_tilt_deg'],nest_offsets_mm=scene['carrier'].get('nest_offsets_mm',{}),pad_contact=scene['pad_contact'], parts=scene['parts'],
                               load_model=scene['load_model'],assembly=scene['assembly'],episode_executor='SO101_teach MotionSession.on_snapshot'),
                  comparison_only=dict(sim_real_legacy_physics=read_json(SIM/'config/assembly.json')['physics'],
                       previous_simulation_hold_effort_nm=physics['actuators'].get('gripper_hold_effort_nm'),
                       reason='old provisional A masses are not B measurements; real_sim is assisted kinematic playback'),
                  limitations=['B masses estimated from previous material evidence, not weighed',
                               'TPU body deformation is not modeled; optional pad contact compliance is an unmeasured spring approximation',
                               'Four hole-sized magnets use provisional force and density, not measured specifications',
                               'jig observation uses saved teaching pose, not live camera inference',
                               'raw servo load uses recorded error/load approximations and effort fallback; not torque calibrated',
                               'CAD workcell placement has not been measured against robot TCP',
                               'policy has no hardware deployment entry point'])
    atomic_json(output/'source_manifest.json', report)
    atomic_json(output/'scene.json', scene)
    atomic_json(output/'reference_episode.json', ref.document)
    atomic_json(output/'resolved_plan.json', dict(jig_references=ref.jig_references, steps=ref.steps,
                                                corrections=ref.corrections))
    return ref, scene, report
