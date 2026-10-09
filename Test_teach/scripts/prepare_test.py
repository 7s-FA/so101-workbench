"""Create isolated settings, keeping source Sim_real assets read-only."""
import json, math, re, zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT.parent/'Sim_real'

def mass_estimate(filename,ratio):
    # Use positive XY print extrusion only; exclude purge/custom/retractions.
    with zipfile.ZipFile(filename) as z: lines=z.read('Metadata/plate_1.gcode').decode().splitlines()
    feature='Custom';relative=True;e_last=0.;amount={}
    for line in lines:
        if line.startswith('; FEATURE:'):feature=line.split(':',1)[1].strip()
        code=line.split(';',1)[0].strip()
        if code.startswith('M83'):relative=True
        if code.startswith('M82'):relative=False
        fields={k:float(v) for k,v in re.findall(r'([XYZE])(-?(?:\d+(?:\.\d*)?|\.\d+))',code)}
        if code.startswith('G92') and 'E' in fields:e_last=fields['E'];continue
        if code.split(' ',1)[0] not in ('G0','G1','G2','G3') or 'E' not in fields:continue
        e=fields['E'] if relative else fields['E']-e_last;e_last=fields['E']
        if feature!='Custom' and e>0 and ('X' in fields or 'Y' in fields):amount[feature]=amount.get(feature,0)+e
    total=sum(v*(ratio if k=='Sparse infill' else 1) for k,v in amount.items())
    return total*math.pi*(1.75/2)**2*1.32/1e6,amount

def main():
    materials=json.loads((ROOT/'config/materials.json').read_text())
    for kind,filename,ratio in [('ball','공.gcode.3mf',1),('housing','1단.gcode.3mf',1/3),('cap','2단.gcode.3mf',1/3)]:
        mass,features=mass_estimate(Path('/home/fresh/Downloads')/filename,ratio)
        materials['parts'][kind].update(mass_kg=mass,mass_status='provisional extrusion estimate; not resliced or weighed',
            mass_estimation_method='positive XY extrusion by feature; scale Sparse infill from 15% to 5% for housing/cap, keep shells; not exact reslicing',extrusion_mm_by_feature=features)
    materials['contact'].update(gripper_surface='rubber octagonal pad',pad_span_m=.010,pad_thickness_m=.0005,
        pad_thickness_status='provisional thin pad, user has not measured',static_friction=1.5,dynamic_friction=1.5*.7142857,
        restitution=0.,status='provisional contact parameters; sensitivity testing required, not measured')
    (ROOT/'config/materials.json').write_text(json.dumps(materials,ensure_ascii=False,indent=2)+'\n')
    manifest=json.loads((SOURCE/'config/conveyor_demo.json').read_text())
    tree=ET.parse(SOURCE/'assets/generated/conveyor_demo/omx_no_marker.urdf')
    for node in list(tree.getroot()):
        if node.get('name') in ('world','world_fixed','end_effector_link','end_effector_joint'):tree.getroot().remove(node)
    ET.SubElement(tree.find("joint[@name='gripper_joint_2']"),'mimic',joint='gripper_joint_1',multiplier='-1',offset='0')
    (ROOT/'assets').mkdir(exist_ok=True)
    tree.write(ROOT/'assets/omx_teach.urdf',encoding='utf-8',xml_declaration=True)
    cfg={'table':{'size':[.4,.6],'center':[.14,0],'height':.23},'base':[0,0,.23],
         'model_source':str(ROOT/'assets/omx_teach.urdf'),
         'rest_point':[.205,0,.325],'parts':{},'dt':1/480,'arm_effort_nm':2.,'jaw_effort_nm':.4,
         'actuator_status':'provisional simulated limits; not calibrated to hardware current',
         'grasp_mode':'physics_only_no_attachment','pad_thickness_m':.0005}
    for kind,angle in zip(('housing','ball','cap'),(-65,-40,-15)):
        def xy(a):return [.205*math.cos(math.radians(a)),.205*math.sin(math.radians(a))]
        cfg['parts'][kind]={'source':xy(angle),'destination':xy(-angle),
            'mesh':str(SOURCE/('assets/product/02_ball_d30.stl' if kind=='ball' else 'assets/generated/conveyor_demo/parts/'+manifest['parts'][kind])),
            'jig':str(SOURCE/'assets/generated/conveyor_demo/parts'/manifest['parts'][kind+'_jig']),
            'seating_z':manifest['jig_seating']['contact_height'][kind],
            'offset_xy':manifest['jig_seating']['offset_xy'],
            'grasp_height':{'housing':.012,'ball':.016,'cap':.004}[kind]}
    cfg['parts']['cap'].update(seating_z=.0035,destination_seating_z=.0035,fixture='flat source/destination receiving platforms; isolate transport from centre-post extraction')
    (ROOT/'config/scene.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n')
    print({k:round(v['mass_kg']*1000,3) for k,v in materials['parts'].items()})
if __name__=='__main__':main()
