"""Reproduce link-mass estimates from preserved extrusion evidence, not solid volume."""
from pathlib import Path
import gzip,hashlib,json,math,sys
import xml.etree.ElementTree as ET
import zipfile
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from calibration.link_mass.gcode_mass import extrusion,AUXILIARY
from model_tcp import read_stl


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(name,value):(HERE/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def build():
    pla=json.loads((HERE/'pla_print_analysis.json').read_text())
    tpu=json.loads((HERE/'tpu_print_analysis.json').read_text())
    for filename in ('tpu_slicer_result.json','holder_slicer_result.json'):
        assert json.loads((HERE/'generated'/filename).read_text())['return_code']==0
    with gzip.open(HERE/'generated/holders.gcode.gz','rb') as f:holder=extrusion(f)
    with zipfile.ZipFile(HERE/'generated/holders_project.3mf') as z:
        profile=json.loads(z.read('Metadata/project_settings.config'))
    assert profile['sparse_infill_density']=='15%' and profile['wall_loops']=='4' and profile['layer_height']=='0.1'
    assert profile['sparse_infill_pattern']=='gyroid' and float(profile['filament_density'][0])==1.32
    factor=math.pi*(float(profile['filament_diameter'][0])/2)**2*float(profile['filament_density'][0])/1000
    holder_rows=[]
    for ident,features in holder['extrusion_mm_by_object_feature'].items():
        plastic=sum(value for feature,value in features.items() if feature not in AUXILIARY and feature!='stationary_extrusion')
        holder_rows.append({'id':ident,'name':{'1':'fixed_holder','2':'moving_holder'}.get(ident),
                            'part_mass_g':plastic*factor,'all_deposited_mass_g':sum(features.values())*factor,
                            'extrusion_mm_by_feature':features})
    holder.update(objects=holder_rows,profile=profile,source_gcode='generated/holders.gcode.gz',
                  source_gcode_sha256=sha(HERE/'generated/holders.gcode.gz'),
                  settings_confirmed_by_user=True,method='Source holder print orientation; same user-confirmed PLA profile as original robot G-code; new Bambu 2.8.2.61 slicing.')
    # Keep the relevant print settings; unrelated slicer machine options remain in the input 3MF.
    holder['profile']={k:profile[k] for k in ('filament_density','filament_diameter','sparse_infill_density','sparse_infill_pattern','wall_loops','layer_height','top_shell_layers','bottom_shell_layers')}
    dump('holder_print_analysis.json',holder)
    for data,declared in ((pla,pla['plate_mass_g']),(tpu,tpu['plate_mass_g']),(holder,holder['header']['total filament weight [g]'])):
        measured=sum(row['all_deposited_mass_g'] for row in data['objects'])
        assert abs(measured-declared)<.05,(measured,declared)
    mapping={1:'motor_holder_so101_base_v1.stl',2:'base_so101_v2.stl',3:'base_motor_holder_so101_v1.stl',
             4:'motor_holder_so101_wrist_v1.stl',5:'rotation_pitch_so101_v1.stl',6:'waveshare_mounting_plate_so101_v2.stl',
             9:'upper_arm_so101_v1.stl',10:'under_arm_so101_v1.stl',12:'wrist_roll_pitch_so101_v2.stl'}
    masses={};components=[]
    for row in pla['objects']:
        if not row['name']:continue
        number=int(row['name'].rsplit('_',1)[1]);filename=mapping[number]
        tri=read_stl(ROOT/'assets/so101/assets'/filename)*1000
        mesh_dims=np.ptp(tri.reshape(-1,3),axis=0)
        b=row['bounds_mm'];path_dims=np.array([b[2]-b[0],b[3]-b[1],b[4]])
        difference=abs(np.sort(path_dims)-np.sort(mesh_dims))
        assert difference.max()<.35,(filename,mesh_dims,path_dims)
        masses[filename]=row['part_mass_g']
        components.append({'print_object':row['name'],'object_id':row['id'],'mesh':filename,'mass_g':row['part_mass_g'],
                           'mesh_dimensions_mm':mesh_dims.tolist(),'deposited_path_dimensions_mm':path_dims.tolist(),
                           'max_dimension_difference_mm':float(difference.max()),'method':'original_gcode_object_extrusion'})
    fingers=[row['part_mass_g'] for row in tpu['objects'] if row['name']]
    assert len(fingers)==4 and max(fingers)-min(fingers)<.001
    finger=sum(fingers)/len(fingers)
    for side in ('fixed','moving'):
        row=next(row for row in holder_rows if row['name']==side+'_holder')
        masses['xlerobot_'+side+'_holder.stl']=row['part_mass_g']
        masses['xlerobot_'+side+'_finger.stl']=finger
        components.extend([{'mesh':'xlerobot_'+side+'_holder.stl','mass_g':row['part_mass_g'],'method':'new_slice_same_confirmed_PLA_profile'},
                           {'mesh':'xlerobot_'+side+'_finger.stl','mass_g':finger,'method':'new_slice_original_TPU_3MF_4_identical_fingers'}])
    urdf=ROOT/'assets/so101/so101_modified.urdf';xml=ET.parse(urdf).getroot();links=[]
    labels={'base_link':'베이스','shoulder_link':'어깨','upper_arm_link':'상완','lower_arm_link':'전완',
            'wrist_link':'손목','gripper_link':'고정 집게 측','moving_jaw_so101_v1_link':'가동 집게'}
    used=[]
    for link in xml.findall('link'):
        name=link.get('name')
        if name not in labels:continue
        printed=[];motors=[]
        for visual in link.findall('visual'):
            mesh=visual.find('geometry/mesh')
            if mesh is None:continue
            filename=Path(mesh.get('filename')).name
            if filename.startswith('sts3215'):motors.append(filename)
            else:
                assert filename in masses,filename
                printed.append({'mesh':filename,'mass_g':masses[filename]});used.append(filename)
        subtotal=sum(item['mass_g'] for item in printed);motor_mass=55.*len(motors)
        links.append({'link':name,'label_ko':labels[name],'printed_components':printed,'printed_mass_g':subtotal,
                      'servo_count':len(motors),'servo_nominal_mass_g':motor_mass,'mass_g_estimate':subtotal+motor_mass,
                      'mass_kg_estimate':(subtotal+motor_mass)/1000,
                      'old_urdf_mass_g':float(link.find('inertial/mass').get('value'))*1000})
    assert len(used)==len(set(used))==13
    assert sum(link['servo_count'] for link in links)==6
    total=sum(link['mass_g_estimate'] for link in links)
    result={'schema':1,'status':'calculated_mass_estimate','units':'g',
            'derived_teaching_model':'../Test_teach/assets/so101/so101_teach.urdf',
            'derived_inertia_reference':'../Test_teach/assets/so101/physics_provenance.json',
            'source_urdf':'assets/so101/so101_modified.urdf','source_urdf_sha256':sha(urdf),
            'bambu_version':'02.08.02.61','components':components,'links':links,'total_mass_g_estimate':total,
            'total_printed_mass_g':sum(link['printed_mass_g'] for link in links),'total_servo_mass_g':330.,
            'servo':{'model':'STS3215','nominal_mass_g_each':55.,'manufacturer_tolerance_g_each':1.,
                     'source':'https://files.seeedstudio.com/products/Feetech/108090023_STS3215-C001_Datasheet.pdf','page':4},
            'rubber_pad':json.loads((HERE/'so101_contact.json').read_text()),
            'excluded_unweighed_items':['rubber pads and adhesive','separate screws, nuts and other fasteners','controller PCB','additional wiring outside the nominal servo assembly','table clamp and overhead camera mount'],
            'limitations':['Slicer estimates and manufacturer nominal mass, not weighed link assemblies.',
                           'Current holder print settings were confirmed by the user; original model print orientation used for fresh slicing.',
                           'A per-link mass update does not determine a new centre of mass or inertia tensor.',
                           'Original URDF mass values and motor calibration are untouched.'],
            'source_archive_sha256':{'so101 부품들.gcode.3mf':pla['source_sha256'],
                                     'so101 tpu그립.3mf':tpu['input_project_sha256']},
            'verification':{'plate_vs_extrusion_max_allowed_difference_g':.05,'objects_mapped_by_dimensions':9,'geometry_tolerance_mm':.35}}
    dump('link_mass_estimate.json',result)
    for row in links:print(row['label_ko'],f"{row['printed_mass_g']:.6f} + {row['servo_nominal_mass_g']:.3f} = {row['mass_g_estimate']:.6f} g")
    print('TOTAL',total,'g')
    return result

if __name__=='__main__':build()
