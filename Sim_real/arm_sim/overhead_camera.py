"""Original interlocking arm base and 32 mm overhead camera stand."""
import math
import numpy as np
from scipy.spatial.transform import Rotation
from arm_sim.common import ROOT

FOLDER = ROOT/'vendor/so_arm100/Optional/Overhead_Cam_Mount_32x32_UVC_Module/stl'
ARM_BASE_RISE=.0072


def arm_base_transform(cfg):
    pose=cfg['arms'][cfg['assembly_camera']['attached_arm']]
    yaw=Rotation.from_euler('z',pose['yaw']).as_matrix()
    tf=np.eye(4);tf[:3,:3]=yaw@Rotation.from_euler('x',90,degrees=True).as_matrix()
    # Centre the original SO101 footprint inside the CAD outline (1.3mm rim).
    # CAD bottom Y=-5mm, highest support ribs Y=2.2mm: 7.2mm total rise.
    tf[:3,3]=[*pose['position'],cfg['table_height']]+yaw@np.array([.0702560637,-.0534304657,.005])
    return tf


def assembly(cfg):
    """Return shared mesh transforms for rendering and geometric checks.

    Bottom's shaft is centred at CAD X=18.7, Z=36.5125 mm, not
    X=12.7, Z=5 mm. Middle/Top share a CAD origin and mate at Y=220/375.
    The 15 mm lens opening is centred on the angled plate; PCB front sits
    behind its 3 mm thickness so the lens passes through the opening.
    """
    plate=arm_base_transform(cfg);rotation=plate[:3,:3]
    # Right connector: the arm base's -Z tabs interleave with Bottom's +Z tabs.
    # Both printed undersides sit on the tabletop; no independent square foot.
    base=plate[:3,3]+rotation@np.array([-.0939209137,-.005,-.20597504272])
    result=[('ArmBase',FOLDER/'arm_base.stl',.001,plate)]
    for name,offset in [('Bottom',[0,0,0]),('Middle',[.0187,.185,.0365125]),('Top',[.0187,.185,.0365125])]:
        tf=np.eye(4);tf[:3,:3]=rotation;tf[:3,3]=base+rotation@offset
        result.append((name,FOLDER/f'cam_mount_{name.lower()}.stl',.001,tf))
    a=math.radians(25);normal=np.array([math.sin(a),-math.cos(a),0.])
    tangent=np.array([math.cos(a),math.sin(a),0.])
    plate_center=.179788*tangent-.31583*normal+np.array([0,0,-.0000005])
    # Module PCB front at local z=1.3 mm; lens extends to z=26.3 mm.
    module_origin=plate_center-(.003+.0013+.0001)*normal
    bx=np.array([0.,0.,1.]);by=np.cross(normal,bx)
    tf=np.eye(4);tf[:3,:3]=rotation@np.column_stack((bx,by,normal))
    tf[:3,3]=base+rotation@(np.array([.0187,.185,.0365125])+module_origin)
    result.append(('Module',ROOT/'vendor/so_arm100/Simulation/SO101/assets/wrist_camera_so101_v1.stl',1.,tf))
    eye=tf[:3,3]+.0263*tf[:3,2]
    return result,eye,tf[:3,2]


def support_boxes(cfg):
    return []
