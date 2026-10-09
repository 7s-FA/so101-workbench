"""STL geometry for camera-confirmed jig display; no virtual docking pose."""
from copy import deepcopy
from pathlib import Path
import hashlib,math,re,struct
import numpy as np

ROOT=Path(__file__).resolve().parent
ASSETS=ROOT/'assets/jigs/imported'
MAX_BYTES=30*1024*1024
MAX_TRIANGLES=200000


def triangles(path):
    path=Path(path)
    with path.open('rb') as stream:data=stream.read(MAX_BYTES+1)
    if len(data)>MAX_BYTES:raise ValueError('STL 파일은 30MB 이하로 선택하세요.')
    count=struct.unpack_from('<I',data,80)[0] if len(data)>=84 else 0
    if count and len(data)==84+50*count:
        if count>MAX_TRIANGLES:raise ValueError('삼각형이 너무 많습니다. 20만 개 이하 STL을 사용하세요.')
        dtype=np.dtype([('normal','<f4',3),('vertices','<f4',(3,3)),('attr','<u2')])
        vertices=np.frombuffer(data,dtype=dtype,count=count,offset=84)['vertices'].astype(float)
    else:
        try:
            text=data.decode('ascii');rows=re.findall(r'^\s*vertex\s+([^\r\n]+)',text,re.M|re.I)
            if not text.lstrip().lower().startswith('solid') or not rows or len(rows)%3 or len(rows)>MAX_TRIANGLES*3:raise ValueError()
            vertices=np.asarray([[float(x) for x in row.split()] for row in rows],float).reshape(-1,3,3)
        except (UnicodeError,ValueError):raise ValueError('읽을 수 있는 바이너리 또는 ASCII STL이 아닙니다.') from None
    if not len(vertices) or not np.isfinite(vertices).all():raise ValueError('STL 좌표가 비어 있거나 잘못됐습니다.')
    return vertices


def binary_stl(vertices):
    cross=np.cross(vertices[:,1]-vertices[:,0],vertices[:,2]-vertices[:,0]);length=np.linalg.norm(cross,axis=1)
    if not np.any(length>1e-10):raise ValueError('STL에 유효한 면이 없습니다.')
    dtype=np.dtype([('normal','<f4',3),('vertices','<f4',(3,3)),('attr','<u2')]);out=np.zeros(len(vertices),dtype=dtype)
    out['normal']=cross/np.maximum(length[:,None],1e-20);out['vertices']=vertices
    return b'OpenCV teach normalized visual jig'.ljust(80,b'\0')+struct.pack('<I',len(out))+out.tobytes()


def validate_model(value):
    if not isinstance(value,dict):raise ValueError('STL 설정이 올바르지 않습니다.')
    result=deepcopy(value)
    if not re.fullmatch(r'[a-f0-9]{64}\.stl',str(result.get('asset',''))):raise ValueError('등록 STL 식별자가 올바르지 않습니다.')
    dims=result.get('dimensions_mm')
    if not isinstance(dims,list) or len(dims)!=3:raise ValueError('STL 치수를 확인하세요.')
    from vision_teaching import number
    result['dimensions_mm']=[number(x,.001,5000) for x in dims]
    if not isinstance(result.get('source_name'),str) or len(result['source_name'])>255:raise ValueError('STL 파일 이름을 확인하세요.')
    # Retain only the physical support-plane height from older records.
    from mount_scene import base_height_mm
    old=result.pop('pose',{})
    result['bottom_z_mm']=number(result.get('bottom_z_mm',old.get('z_mm',-base_height_mm())),-2000,2000)
    if 'rim_height_mm' in result:result['rim_height_mm']=number(result['rim_height_mm'],.001,result['dimensions_mm'][2]+.001)
    return result


def import_stl(path,unit='mm'):
    factors={'mm':1.,'cm':10.,'m':1000.}
    if unit not in factors:raise ValueError('STL 단위를 선택하세요.')
    vertices=triangles(path)*factors[unit]
    low=vertices.min(axis=(0,1));high=vertices.max(axis=(0,1));dims=high-low
    if np.any(dims<.001) or np.any(dims>5000):raise ValueError('STL 크기가 잘못됐습니다. 단위와 입체 형상을 확인하세요.')
    # Bbox centre is independent of the arbitrary CAD origin. Long side = +X.
    vertices-=np.array([(low[0]+high[0])/2,(low[1]+high[1])/2,low[2]])
    quarter_turn=bool(dims[1]>dims[0])
    if quarter_turn:
        vertices=vertices@np.array([[0,-1,0],[1,0,0],[0,0,1.]]).T
        dims=dims[[1,0,2]]
    raw=binary_stl(vertices);name=hashlib.sha256(raw).hexdigest()+'.stl'
    ASSETS.mkdir(parents=True,exist_ok=True)
    destination=ASSETS/name
    if not destination.exists():
        import os,tempfile
        fd,tmp=tempfile.mkstemp(dir=ASSETS,prefix='.import-')
        try:
            with os.fdopen(fd,'wb') as stream:stream.write(raw)
            os.replace(tmp,destination)
        finally:
            if os.path.exists(tmp):os.unlink(tmp)
    from mount_scene import base_height_mm
    return validate_model({'asset':name,'source_name':Path(path).name,'dimensions_mm':dims.tolist(),
        'unit':unit,'long_axis_quarter_turn':quarter_turn,'origin':'bounding_box_xy_center_z_bottom',
        'bottom_z_mm':-base_height_mm(),'rim_height_mm':outer_rim_height(vertices),'hole_layout':projected_holes(vertices),'hole_layout_version':2})


def resolve_asset(model):
    model=validate_model(model);path=ASSETS/model['asset']
    if not path.is_file():raise ValueError('등록 STL 파일이 없습니다. STL 파일을 다시 선택하세요.')
    if hashlib.sha256(path.read_bytes()).hexdigest()+'.stl'!=model['asset']:raise ValueError('등록 STL 파일이 변경됐습니다. 다시 등록하세요.')
    return path


def outer_rim_height(vertices):
    """Dominant horizontal surface covering all four outer edge strips.

    Weight clipped surface area, not triangle count; raised internal tabs and
    arbitrary tessellation cannot win just by contributing more vertices.
    """
    import cv2
    low=vertices.min(axis=(0,1));high=vertices.max(axis=(0,1))
    band=max(.2,min(high[:2]-low[:2])*.015)
    x0,y0=low[:2];x1,y1=high[:2]
    strips=[[(x0,y0),(x0+band,y0),(x0+band,y1),(x0,y1)],
            [(x1-band,y0),(x1,y0),(x1,y1),(x1-band,y1)],
            [(x0,y0),(x1,y0),(x1,y0+band),(x0,y0+band)],
            [(x0,y1-band),(x1,y1-band),(x1,y1),(x0,y1)]]
    strips=[np.array(a,dtype=np.float32) for a in strips];scores={}
    flat=np.ptp(vertices[:,:,2],axis=1)<1e-4
    for face in vertices[flat]:
        z=round(float(face[:,2].mean()-low[2]),4)
        if z<=.001:continue
        xy=face[:,:2].astype(np.float32)
        if abs(cv2.contourArea(xy))<1e-8:continue
        areas=np.array([cv2.intersectConvexConvex(xy,strip)[0] for strip in strips])
        scores[z]=scores.get(z,np.zeros(4))+areas
    valid={z:areas for z,areas in scores.items() if np.all(areas>band*band)}
    if not valid:raise ValueError('STL의 네 외곽을 잇는 평평한 윗면을 찾지 못했습니다. Z축과 외곽 형상을 확인하세요.')
    return max(valid,key=lambda z:float(valid[z].sum()))


def projected_holes(vertices):
    """Normalized openings in the STL outer-rim plane, including recesses."""
    import cv2
    low=vertices.min(axis=(0,1));span=vertices.max(axis=(0,1))-low
    scale=600./max(span[:2]);size=np.ceil(span[:2]*scale).astype(int)+5
    mask=np.zeros((size[1],size[0]),np.uint8)
    rim=low[2]+outer_rim_height(vertices)
    # Only rim-level and higher faces: a cavity floor below the rim must not fill its opening.
    upper=vertices[vertices[:,:,2].min(axis=1)>=rim-1e-4]
    for tri in upper:
        xy=np.rint((tri[:,:2]-low[:2])*scale+2).astype(np.int32)
        cv2.fillConvexPoly(mask,xy,255)
    contours,hierarchy=cv2.findContours(mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    holes=[]
    if hierarchy is not None:
        for c,h in zip(contours,hierarchy[0]):
            if h[3]<0 or cv2.contourArea(c)<mask.size*.005:continue
            approx=cv2.approxPolyDP(c,1.,True).reshape(-1,2)
            holes.append(np.clip((approx-2)/scale/span[:2],0,1).tolist())
    return holes


from functools import lru_cache
@lru_cache(maxsize=1)
def legacy_holes():
    return projected_holes(triangles(ROOT/'assets/jigs/carrier_6jigs_220x148_M3_flat_lightweight.stl'))


def with_rim_metadata(model):
    model=validate_model(model)
    if 'rim_height_mm' not in model or model.get('hole_layout_version')!=2:
        vertices=triangles(resolve_asset(model))
        model['rim_height_mm']=outer_rim_height(vertices)
        model['hole_layout']=projected_holes(vertices)
        model['hole_layout_version']=2
    return model


def detection_height(model):
    model=with_rim_metadata(model)
    return model['bottom_z_mm']+model['rim_height_mm']


def model_with_meshes(scene_path,items):
    """Flatten local preview includes, then add non-colliding visual mesh bodies."""
    import xml.etree.ElementTree as ET
    import mujoco
    def expanded(path):
        path=Path(path);root=ET.parse(path).getroot()
        for parent in list(root.iter()):
            for child in list(parent):
                if child.tag=='include':
                    index=list(parent).index(child);parent.remove(child)
                    for element in expanded(path.parent/child.get('file')):parent.insert(index,element);index+=1
                elif 'file' in child.attrib:child.set('file',str((path.parent/child.get('file')).resolve()))
        return root
    root=expanded(scene_path)
    asset=ET.SubElement(root,'asset');world=ET.SubElement(root,'worldbody')
    for i,item in enumerate(items):
        if not re.fullmatch(r'[a-f0-9]{64}\.stl',item['mesh_asset']):raise ValueError('등록 STL 식별자가 올바르지 않습니다.')
        path=ASSETS/item['mesh_asset']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()+'.stl'!=item['mesh_asset']:raise ValueError('등록 STL을 다시 확인하세요.')
        ET.SubElement(asset,'mesh',name=f'jig_mesh_{i}',file=str(path),scale='.001 .001 .001')
        body=ET.SubElement(world,'body',name=f'jig_visual_{i}')
        ET.SubElement(body,'geom',type='mesh',mesh=f'jig_mesh_{i}',rgba='.92 .94 .95 1',contype='0',conaffinity='0',mass='0')
    return mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))


def place_meshes(model,items):
    for i,item in enumerate(items):
        body=model.body(f'jig_visual_{i}');a=math.radians(item['yaw'])/2
        body.pos[:]=np.array([item['x'],item['y'],item['z']])/1000.
        body.quat[:]=[math.cos(a),0,0,math.sin(a)]


def observed_scenes(targets,station,observations):
    """Measured XY/yaw; align the STL outer rim with the detection plane."""
    from vision_teaching import reference_key
    scenes=[];seen=set()
    for target in targets:
        if target.get('mode')!='rectangle':continue
        from jig_detection import validate_profile
        target={**target,**validate_profile(target)}
        key=reference_key(target)
        if key in seen or key not in observations:continue
        seen.add(key);p=observations[key]
        item={'x':p['x_mm'],'y':p['y_mm'],'yaw':p['yaw_deg'],'z':target['surface_z_mm'],
              'long':target['long_mm'],'short':target['short_mm'],'shape_model':target.get('shape_model','rectangle'),
              'reference_id':p.get('reference_id'),'nominal':False}
        visual=station.get('jigs',{}).get(key,{}).get('preview_model') or target.get('preview_model')
        if visual is not None:
            try:
                model=with_rim_metadata(visual)
                dims=np.array(model['dimensions_mm'][:2]);expected=np.array([target['long_mm'],target['short_mm']])
                if np.max(np.abs(dims-expected)/expected)>.05:raise ValueError('STL과 검출 치수가 달라 외곽만 표시')
                resolve_asset(model)
                item.update(mesh_asset=model['asset'],height=model['dimensions_mm'][2],z=target['surface_z_mm']-model['rim_height_mm'],shape_model='rectangle')
                item['height_source']='stl_outer_rim_at_detection_plane'
            except (OSError,ValueError) as exc:item['model_warning']=str(exc)
        scenes.append(item)
    return scenes
