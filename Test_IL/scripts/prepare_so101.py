"""Latest teaching SO-101, wrist 32x32 UVC and modified overhead D435 stand."""
import hashlib,json,math,sys
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];TEACH=ROOT.parent/'Test_teach'
sys.path.insert(0,str(ROOT))
from studio.safe_pose import load_safe_pose


def build():
    safe=load_safe_pose()
    start_angles=safe['raw_angles']
    out=ROOT/'assets/so101';out.mkdir(parents=True,exist_ok=True)
    source=TEACH/'assets/so101/so101_teach.urdf';tree=ET.parse(source);root=tree.getroot();root.set('name','so101_il_uvc_overhead_d435')
    assembly_path=ROOT/'assets/camera_mounts/assembly.json'
    if not assembly_path.exists():raise ValueError('Run scripts/build_camera_mounts.py first')
    assembly=json.loads(assembly_path.read_text())
    for item in assembly['attachments']:
        link=root.find(f"link[@name='{item['link']}']");tf=np.array(item['transform_mm'])
        visual=ET.SubElement(link,'visual',name=item['name'])
        ET.SubElement(visual,'origin',xyz=' '.join(map(str,tf[:3,3]/1000)),rpy=' '.join(map(str,Rotation.from_matrix(tf[:3,:3]).as_euler('xyz'))))
        ET.SubElement(ET.SubElement(visual,'geometry'),'mesh',filename=item['file'],scale='.001 .001 .001')
        color='.55 .57 .59 1' if item['color']=='silver' else '.035 .04 .045 1'
        ET.SubElement(ET.SubElement(visual,'material',name=item['name']+'_material'),'color',rgba=color)
    # Empty optical frames are nominal geometry, never substituted for measured calibration.
    def frame(name,parent,point,rotation):
        ET.SubElement(root,'link',name=name)
        joint=ET.SubElement(root,'joint',name=name+'_joint',type='fixed')
        ET.SubElement(joint,'parent',link=parent);ET.SubElement(joint,'child',link=name)
        ET.SubElement(joint,'origin',xyz=' '.join(map(str,np.array(point)/1000)),rpy=' '.join(map(str,Rotation.from_matrix(rotation).as_euler('xyz'))))
    forward=np.array(assembly['optical_axis_base']);right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right)
    # right = -Y for this forward/down camera; down completes the optical frame.
    optical=np.column_stack([right,np.cross(forward,right),forward])
    frame('overhead_rgb_optical_frame','base_link',assembly['new_rgb_center_base_mm'],optical)
    frame('overhead_depth_optical_frame','base_link',assembly['depth_center_base_mm'],optical)
    wrist=np.array(assembly['wrist_camera_transform_mm'])
    frame('gripper_rgb_optical_frame','gripper_link',wrist[:3,3],wrist[:3,:3])
    dest=out/'so101_il.urdf';ET.indent(tree);tree.write(dest,encoding='utf-8',xml_declaration=True)
    files={Path(m.get('filename')) for m in root.findall('.//mesh')}|{source,assembly_path}
    provenance=dict(robot='SO-101',gripper='XLeRobot TPU',cameras={'gripper':'32x32 UVC','table':'RealSense D435'},source=str(source),
        scope='measured joint display; nominal camera frames and unverified printed bracket fit',
        files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)})
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    tc=json.loads((TEACH/'config/so101_scene.json').read_text());table_z=tc['table']['height_mm']/1000
    scene=dict(robot='so101',table=dict(size=[v/1000 for v in tc['table']['size_mm']],center=[v/1000 for v in tc['table']['center_mm']],height=table_z),
        base=[0,0,table_z+assembly['robot_base_above_table_mm']/1000],model_source=str(dest),
        start_angles_deg=[math.degrees(v) for v in start_angles],parts={},jigs_enabled=False,
        start_model_angles_deg=[math.degrees(v) for v in safe['model_angles']],
        start_pose_reference=safe['reference'],
        reference_source=str(TEACH/'config/so101_scene.json'),camera_models={'gripper':'32x32 UVC module','table':'RealSense D435'},gripper_model='xlerobot_tpu',
        camera_assembly=str(assembly_path),camera_extrinsics_status='CAD nominal; not physically calibrated')
    (ROOT/'config/scene.json').write_text(json.dumps(scene,indent=2)+'\n')
    print(dest)
if __name__=='__main__':build()
