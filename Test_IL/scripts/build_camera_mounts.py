"""Printable camera adapters; millimetres. Original bottom/middle/socket retained.
Run: /path/to/isaac/env_isaaclab/bin/python scripts/build_camera_mounts.py
The private .cad_runtime contains manifold3d; shared simulation packages are untouched.
"""
import json,sys,hashlib,math,copy
from pathlib import Path
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.cad_runtime'),str(ROOT.parent/'Sim_real')]
import numpy as np
import trimesh
import manifold3d
from functools import reduce
from scipy.spatial.transform import Rotation
from arm_sim.conveyor_hardware import d435_vertices
SOURCE=ROOT.parent/'OpenCV_teach/calibration/mount_geometry/source'
OUT=ROOT/'assets/camera_mounts'
PARAMS=dict(top_cut_y_mm=325.,rear_plate_width_mm=64.,rear_plate_height_mm=20.,rear_plate_thickness_mm=5.,
    d435_screw_pitch_mm=45.,d435_clearance_hole_mm=3.4,d435_rear_gap_mm=.5,
    d435_screw_length_mm=8.,d435_max_thread_insertion_mm=3.,
    overhead_forward_shift_mm=25.,overhead_center_body_on_column=True,
    wrist_mount_clearance_shift_mm=1.85,wrist_camera_seating_gap_mm=-.05)


def transform(R=None,t=(0,0,0)):
    m=np.eye(4);m[:3,3]=t
    if R is not None:m[:3,:3]=R
    return m


def box(size,center,R=None):return trimesh.creation.box(size,transform=transform(R,center))
def cyl(radius,length,center,axis=(0,0,1)):
    a=np.array(axis,dtype=float);a/=np.linalg.norm(a)
    m=trimesh.geometry.align_vectors([0,0,1],a);m[:3,3]=center
    return trimesh.creation.cylinder(radius=radius,height=length,sections=64,transform=m)
def native(mesh):
    return manifold3d.Manifold(manifold3d.Mesh64(vert_properties=np.asarray(mesh.vertices,dtype=np.float64),tri_verts=np.asarray(mesh.faces,dtype=np.uint64)))
def from_native(m):
    if m.status()!=manifold3d.Error.NoError:raise ValueError(str(m.status()))
    raw=m.simplify(.0001).to_mesh64()
    return trimesh.Trimesh(vertices=raw.vert_properties[:,:3],faces=raw.tri_verts,process=True)
def union(items):return from_native(reduce(lambda a,b:a+b,[native(m) for m in items]))
def difference(a,items):return from_native(reduce(lambda a,b:a-b,[native(a),*[native(m) for m in items]]))
def intersection(a,b):return from_native(native(a)^native(b))
def bridge(a,b):return trimesh.util.concatenate([a,b]).convex_hull

def export(mesh,name):
    mesh.vertices=np.round(mesh.vertices,5);mesh.process(validate=True)
    mesh.remove_unreferenced_vertices();mesh.fix_normals()
    if not mesh.is_watertight or mesh.volume<=0 or len(mesh.split())!=1:
        mesh.export('/tmp/'+name)
        print('invalid mesh',name,mesh.is_watertight,[(x.volume,x.bounds.tolist()) for x in mesh.split(only_watertight=False)])
        raise ValueError(name+' is not one closed positive-volume solid')
    mesh.export(OUT/name)
    return dict(file=name,watertight=True,connected_solids=1,volume_cm3=mesh.volume/1000,bounds_mm=mesh.bounds.tolist(),triangles=len(mesh.faces))


def build():
    OUT.mkdir(parents=True,exist_ok=True)
    nominal=json.loads((SOURCE.parent/'nominal_pose.json').read_text());A=np.array(nominal['assembly_to_forward_left_up']);base=np.array(nominal['robot_base_link_in_assembly'])
    top_t=np.array(nominal['assembly_translations_original_stl']['cam_mount_top'])
    Ttop=transform(A,A@top_t-base)
    target=np.array(nominal['camera_board_center_in_base_link']);forward=np.array(nominal['nominal_optical_axis_in_robot_axes'])
    left=np.array([0.,1.,0]);left-=forward*(forward@left);left/=np.linalg.norm(left)
    Rcam=np.column_stack([forward,left,np.cross(forward,left)])
    # D435 nominal RGB frame: 10.6 mm forward, 32.5 mm left, 12.5 mm up from bottom-screw frame.
    rgb=np.array([10.6,32.5,12.5])
    # A forward cantilever, with the camera body centered on the original mast.
    rgb_target=target+np.array([PARAMS['overhead_forward_shift_mm'],32.5,0.])
    Tcam=transform(Rcam,rgb_target-Rcam@rgb)
    cam_mesh=trimesh.Trimesh(vertices=d435_vertices()*1000,faces=np.arange(len(d435_vertices())).reshape(-1,3),process=True)
    cam_mesh.export(OUT/'d435_body_mm.stl')
    rear=cam_mesh.bounds[0,0];rear_center=Tcam[:3,3]+Rcam@np.array([rear,0,12.5])
    Rplate=np.column_stack([Rcam[:,1],Rcam[:,2],Rcam[:,0]])
    plate_center=rear_center-forward*(PARAMS['d435_rear_gap_mm']+PARAMS['rear_plate_thickness_mm']/2)
    # Build in robot-base coordinates, then return exactly to the original STL frame.
    original=trimesh.load(SOURCE/'cam_mount_top.stl',force='mesh')
    crop=box([500,1000,500],[0,PARAMS['top_cut_y_mm']-500,0])
    stem=intersection(original,crop);stem.apply_transform(Ttop)
    plate=box([PARAMS['rear_plate_width_mm'],PARAMS['rear_plate_height_mm'],PARAMS['rear_plate_thickness_mm']],plate_center,Rplate)
    foot_center=(Ttop@np.array([0,318,0,1]))[:3]
    foot=box([24,12,32],foot_center,A)
    neck_end=plate_center-forward*6+Rplate[:,1]*8
    column_head=box([24,20,12],[foot_center[0],foot_center[1],577])
    neck=union([bridge(foot,column_head),bridge(column_head,box([16,10,8],neck_end,Rplate))])
    # Back webs brace the cantilever while leaving both screw-head access cylinders open.
    pieces=[stem,plate,neck]
    holes=[];hole_centers=[]
    for side in (-1,1):
        p=plate_center+Rplate[:,0]*(side*PARAMS['d435_screw_pitch_mm']/2);hole_centers.append(p.tolist())
        holes.append(cyl(PARAMS['d435_clearance_hole_mm']/2,80,p,forward))
        holes.append(cyl(3.4,22,p-forward*13.5,forward))
    top_world=difference(union(pieces),holes)
    top=top_world.copy();top.apply_transform(np.linalg.inv(Ttop));top_result=export(top,'overhead_top_d435.stl')
    # Preserve mating/socket region exactly (below the new shoulder/head).
    region=box([300,1000,300],[0,-220,0]) # ends at original Y=280
    old_socket=intersection(original,region)
    new_socket=intersection(top,region)
    socket_delta=[]
    for first,second in ((old_socket,new_socket),(new_socket,old_socket)):
        points=np.vstack([first.vertices,first.triangles_center])
        _,distances,_=trimesh.proximity.closest_point_naive(second,points)
        socket_delta.append(float(distances.max()))
    if max(socket_delta)>.01:raise ValueError('Original top socket changed')
    # The assembly camera remains fixed while the robot sits on the stock 5 mm base support.
    attachments=[]
    def attach(name,file,tf,link='base_link',color='black'):
        attachments.append(dict(name=name,file=str(file),transform_mm=tf.tolist(),link=link,color=color))
    for name in ('arm_base','cam_mount_bottom','cam_mount_middle'):
        offset=np.array(nominal['assembly_translations_original_stl'][name])
        attach(name,SOURCE/(name+'.stl'),transform(A,A@offset-base))
    attach('overhead_top_d435',OUT/'overhead_top_d435.stl',Ttop)
    attach('overhead_d435',OUT/'d435_body_mm.stl',Tcam,color='silver')
    for i,side in enumerate((-1,1)):
        seat=rear_center+Rplate[:,0]*(side*PARAMS['d435_screw_pitch_mm']/2)
        washer=difference(cyl(3.1,.5,seat-forward*.25,forward),[cyl(1.7,2,seat,forward)])
        washer_file=OUT/f'overhead_washer_{i}.stl';washer.export(washer_file)
        attach('overhead_washer_'+str(i),washer_file,np.eye(4),color='silver')
        back=seat-forward*5.5
        screw=union([cyl(1.5,8.1,back+forward*3.95,forward),cyl(2.75,3,back-forward*1.5,forward)])
        screw_file=OUT/f'overhead_screw_{i}.stl';screw.export(screw_file)
        attach('overhead_screw_'+str(i),screw_file,np.eye(4),color='silver')
    # Use the actual XLeRobot wrist mount and camera visual meshes unchanged.
    # Rigid registration of their stock Fixed_Jaw to the SO101 frame: 0.082 mm
    # trimmed surface RMS. Only the assembly placement changes, never mesh scale.
    xsource=OUT/'source_xlerobot'
    fit=np.array([[-.999998893,-.000727135,-.001297987,.04359513],
        [-.001299793,.002488557,.999996059,.04014141],
        [-.000723902,.999996639,-.002489499,1.023277], [0,0,0,1]])
    urdf=ET.parse(xsource/'xlerobot.urdf')
    j=urdf.find("joint[@name='fixed_Right_Arm_Camera']/origin")
    joint=transform(Rotation.from_euler('xyz',list(map(float,j.get('rpy').split()))).as_matrix(),np.array(list(map(float,j.get('xyz').split())))*1000)
    mounting=fit@joint@transform(t=[-15,-26,0])
    mounting[1,3]+=PARAMS['wrist_mount_clearance_shift_mm']
    wrist=trimesh.load(xsource/'XLeRobot_camera1.stl',force='mesh')
    wrist_result=export(wrist.copy(),'wrist_xlerobot_mount.stl')
    module=trimesh.load(xsource/'XLeRobot_camera2.stl',force='mesh')
    module.export(OUT/'wrist_xlerobot_camera.stl')
    # Optical axis from the source's PCB/lens plane; source geometry has ~1 deg roll.
    candidates=(module.area_faces>50)&(module.face_normals[:,0]>.8)
    normal=np.average(module.face_normals[candidates],axis=0,weights=module.area_faces[candidates]);normal/=np.linalg.norm(normal)
    camera_tf=mounting.copy();camera_tf[:3,3]+=mounting[:3,:3]@normal*PARAMS['wrist_camera_seating_gap_mm']
    projection=module.vertices@normal
    lens=module.vertices[projection>projection.max()-.02].mean(axis=0)
    right=np.array([0.,-np.sin(np.deg2rad(1)),np.cos(np.deg2rad(1))]);right-=normal*(right@normal);right/=np.linalg.norm(right)
    Twrist=camera_tf@transform(np.column_stack([right,np.cross(normal,right),normal]),lens)
    attach('wrist_xlerobot_mount',OUT/'wrist_xlerobot_mount.stl',mounting,'gripper_link')
    attach('wrist_32x32_camera',OUT/'wrist_xlerobot_camera.stl',camera_tf,'gripper_link',color='camera')
    # Printable orientation variants, translated only (source orientation retained).
    for mesh,name in ((top,'overhead_top_d435_print.stl'),(wrist,'wrist_xlerobot_mount_print.stl')):
        m=mesh.copy();m.apply_translation(-m.bounds[0]);m.export(OUT/name)
    rgb_world=(Tcam@np.r_[rgb,1])[:3];depth_world=(Tcam@np.r_[[10.6,17.5,12.5],1])[:3]
    report=dict(parameters=PARAMS,attachments=attachments,print_parts=[top_result,wrist_result],
        base_rise_mm=5.,robot_base_above_table_mm=7.4,
        top_interface_max_surface_delta_mm=max(socket_delta),top_interface_volume_delta_mm3=abs(old_socket.volume-new_socket.volume),
        old_rgb_center_base_mm=target.tolist(),new_rgb_center_base_mm=rgb_world.tolist(),depth_center_base_mm=depth_world.tolist(),
        rgb_position_delta_mm=float(np.linalg.norm(rgb_world-target)),optical_axis_base=forward.tolist(),
        optical_angle_delta_deg=float(np.degrees(np.arccos(np.clip(Rcam[:,0]@forward,-1,1)))),tilt_from_down_deg=float(np.degrees(np.arccos(-forward[2]))),
        d435_screw_centers_base_mm=hole_centers,d435_thread_insertion_mm=PARAMS['d435_screw_length_mm']-PARAMS['rear_plate_thickness_mm']-PARAMS['d435_rear_gap_mm'],
        wrist_camera_transform_mm=Twrist.tolist(),wrist_reference='XLeRobot simulation/xlerobot_urdf.zip camera1/camera2, original geometry',wrist_registration_trimmed_rms_mm=.0823,
        physical_fit_verified=False,notes=['D435 optical frames are manufacturer nominal, not measured hand-eye calibration.',
        'Original XLeRobot wrist geometry; rigid assembly placement fitted to the current SO101 holder.',
        'Original bottom, middle and top socket unchanged; top bracing and head redesigned.'],
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [SOURCE/'cam_mount_top.stl',SOURCE.parent/'nominal_pose.json',xsource/'XLeRobot_camera1.stl',xsource/'XLeRobot_camera2.stl',xsource/'xlerobot.urdf']})
    (OUT/'assembly.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['print_parts','top_interface_max_surface_delta_mm','rgb_position_delta_mm','optical_angle_delta_deg','tilt_from_down_deg']},indent=2))
if __name__=='__main__':build()
