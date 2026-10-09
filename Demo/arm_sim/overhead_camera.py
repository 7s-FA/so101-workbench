"""Nominal assembly of the upstream 32 mm camera stand, independent of arms."""
import math
import numpy as np
from scipy.spatial.transform import Rotation
from arm_sim.common import ROOT

FOLDER = ROOT/'vendor/so_arm100/Optional/Overhead_Cam_Mount_32x32_UVC_Module/stl'


def assembly(cfg):
    """Return shared mesh transforms for rendering and geometric checks.

    Bottom's shaft is centred at CAD X=18.7, Z=36.5125 mm, not
    X=12.7, Z=5 mm. Middle/Top share a CAD origin and mate at Y=220/375.
    The 15 mm lens opening is centred on the angled plate; PCB front sits
    behind its 3 mm thickness so the lens passes through the opening.
    """
    settings=cfg['assembly_camera'];z=cfg['table_height']+.004
    upright=Rotation.from_euler('x',90,degrees=True).as_matrix()
    rotation=Rotation.from_euler('z',settings['yaw']).as_matrix()@upright
    shaft=np.array([.0187,0.,.0365125])
    base=np.array([*settings['shaft_xy'],z])-rotation@shaft
    result=[]
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
    x,y=cfg['assembly_camera']['shaft_xy'];z=cfg['table_height']
    # Compact independent foot entirely supported by the assembly tabletop.
    return [('Plate',(x,y,z+.002),(.11,.10,.004))]
