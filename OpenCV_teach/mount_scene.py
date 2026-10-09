"""Shared CAD mounting geometry in SO-101 base_link coordinates (metres)."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'calibration/mount_geometry/nominal_pose.json'


def geometry():return json.loads(SOURCE.read_text())

def base_height_mm():return float(geometry()['robot_base_link_in_assembly'][2])

def parts():
    g=geometry();a=np.array(g['assembly_to_forward_left_up']);base=np.array(g['robot_base_link_in_assembly'])
    return [(name,ROOT/'calibration/mount_geometry/source'/(name+'.stl'),
             (a@np.array(offset)-base)/1000.,a)
            for name,offset in g['assembly_translations_original_stl'].items()]

def camera_pose():
    g=geometry();z=np.array(g['nominal_optical_axis_in_robot_axes']);z/=np.linalg.norm(z)
    x=np.cross(z,[0.,0.,1.]);x/=np.linalg.norm(x);y=np.cross(z,x)
    return np.array(g['camera_board_center_in_base_link'])/1000.,np.column_stack([x,y,z])


def add_preview_mount(root):
    import xml.etree.ElementTree as ET
    asset=root.find('asset');world=root.find('worldbody')
    for name,path,pos,rotation in parts():
        mesh='overhead_'+name
        ET.SubElement(asset,'mesh',name=mesh,file=str(path),scale='0.001 0.001 0.001')
        q=Rotation.from_matrix(rotation).as_quat();q=np.r_[q[3],q[:3]]
        ET.SubElement(world,'geom',name=mesh,type='mesh',mesh=mesh,pos=' '.join(map(str,pos)),quat=' '.join(map(str,q)),
                      rgba='0.07 0.085 0.10 1',contype='0',conaffinity='0',group='1',mass='0')
    pos,r=camera_pose();q=Rotation.from_matrix(r).as_quat();q=np.r_[q[3],q[:3]]
    body=ET.SubElement(world,'body',name='overhead_camera_pcb',pos=' '.join(map(str,pos)),quat=' '.join(map(str,q)))
    ET.SubElement(body,'geom',name='overhead_pcb',type='box',size='.016 .016 .0008',rgba='.1 .35 .22 1',contype='0',conaffinity='0',mass='0')
    # Lens envelope is a schematic visual; projection centre stays on the PCB.
    ET.SubElement(body,'geom',name='overhead_lens',type='cylinder',pos='0 0 .010',size='.006 .010',rgba='.02 .025 .03 1',contype='0',conaffinity='0',mass='0')
    ET.SubElement(body,'site',name='overhead_optical_center',size='.0015',rgba='1 .6 .1 1')
    root.find("worldbody/geom[@name='preview_floor']").set('pos',f'0 0 {-base_height_mm()/1000.}')
