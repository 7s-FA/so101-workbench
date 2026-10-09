"""Translate the current teaching scene's static geometry into metric assets."""
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
import fcl
from core import PROJECT, TEACH, DATA, TEST, read_json, sha

PRODUCTS = {'housing': ('하단', 4, 8), 'tpu': ('중단', 16, 20), 'cap': ('상단', 24, 28)}
PRODUCTS_DIR = PROJECT/'모델링/B 제품'

def is_contact_geometry(name):
    # Match feature tokens: plain "ear" also matches every "linear" part.
    return (name.startswith(('registered_jig','linear_','workcell_board_surface','workcell_side_panel'))
            and not any(s in name for s in ('_screw','_pin','_l12','_ear_')))

def product_orientation(kind, product):
    """Cap starts upside down on the carrier; preserve the source STL itself."""
    result = np.eye(4)
    if kind == 'cap':
        result[:3,:3] = Rotation.from_euler('x',180,degrees=True).as_matrix()
        rotated = np.asarray(product.vertices)@result[:3,:3].T
        result[2,3] = -float(rotated[:,2].min())
    return result

def transform(xyz=(0,0,0), yaw=0):
    tf = np.eye(4)
    tf[:3,:3] = Rotation.from_euler('z', yaw, degrees=True).as_matrix()
    tf[:3,3] = xyz
    return tf

def mesh(path):
    value = trimesh.load(path, force='mesh')
    if not isinstance(value, trimesh.Trimesh) or not len(value.faces):
        raise ValueError('삼각형 메시가 필요합니다: ' + str(path))
    return value

def collision_model(value):
    model = fcl.BVHModel()
    model.beginModel(len(value.vertices), len(value.faces))
    model.addSubModel(np.asarray(value.vertices, float), np.asarray(value.faces, np.int32))
    model.endModel()
    return model

def seating_height(part, jig):
    """First contact from above, centered in the original product jig (mm)."""
    moving, fixed = collision_model(part), collision_model(jig)
    other = fcl.CollisionObject(fixed)
    high = float(jig.bounds[1,2] - part.bounds[0,2] + 2)
    def hit(z):
        body = fcl.CollisionObject(moving, fcl.Transform(np.eye(3), np.array([0.,0.,z])))
        return fcl.collide(body, other, fcl.CollisionRequest(), fcl.CollisionResult()) > 0
    clear = high
    for z in np.arange(high-.25, -30, -.25):
        if hit(z):
            low = z
            for _ in range(16):
                mid = (low+clear)/2
                if hit(mid): low = mid
                else: clear = mid
            return clear + .05
        clear = z
    raise ValueError('B 부품과 지그의 초기 안착 접촉을 찾지 못했습니다.')

def estimate_mass(kind, product):
    if not product.is_watertight or product.volume <= 0:
        raise ValueError('질량 추정에는 닫힌 양의 체적 메시가 필요합니다.')
    if kind == 'tpu':
        evidence = PROJECT/'OpenCV_teach/calibration/link_mass/tpu_print_analysis.json'
        data = read_json(evidence)
        source_mesh = TEACH/'assets/so101/assets/xlerobot_fixed_finger.stl'
        grams = np.mean([r['part_mass_g'] for r in data['objects'] if r.get('name')])
        material = 'TPU 95A HF; previous finger extrusion effective density'
    else:
        evidence = TEST/'config/materials.json'
        data = read_json(evidence)['parts'][kind]
        source_mesh = TEST/'assets/so101_workcell'/f'{kind}.stl'
        grams = data['mass_kg']*1000
        material = data['material']
    original = mesh(source_mesh)
    # Both archived product and finger meshes use millimetres in their STLs.
    volume = abs(original.volume)
    density = grams/volume
    return dict(mass_kg=float(product.volume*density/1000), material=material,
                status='provisional volume-ratio estimate; not measured or resliced',
                effective_density_g_mm3=float(density), source_mass_g=float(grams),
                source_volume_mm3=float(volume), source_mesh=str(source_mesh), evidence=str(evidence))

def static_assets(ref, output):
    from so101_teach.preview import configured_scene
    from so101_teach.workcell_preview import load_placement
    from so101_teach.arm_workspace import scene_placement
    from so101_teach.height_reference import support_bottom_z
    catalog = read_json(DATA/'jigs.json')
    jigs = []
    for key, saved in ref.jig_references.items():
        cfg = catalog[key]
        if sha(cfg['stl']) != saved['stl_sha256']:
            raise ValueError('B 티칭 지그와 현재 STL이 다릅니다.')
        installation = cfg.get('physical_installation',{})
        jigs.append({**cfg, **cfg['mesh'], **saved, 'bottom_z_mm': support_bottom_z(cfg, ref.profile),
                     'platform':installation.get('platform'),
                     'platform_height_mm':installation.get('carrier_underside_height_mm'),
                     'platform_yaw_deg':installation.get('platform_yaw_deg',-90.)})
    placement = scene_placement(load_placement(DATA), ref.profile)
    scene = ET.fromstring(configured_scene(jigs, ref.profile['tcp'], workcell=placement))
    for i, jig in enumerate(jigs):
        node = scene.find(f".//body[@name='registered_jig_{i}']")
        x,y,yaw = jig['pose']
        node.set('pos', f'{x/1000} {y/1000} {jig["bottom_z_mm"]/1000}')
        q = Rotation.from_euler('z', yaw, degrees=True).as_quat()[[3,0,1,2]]
        node.set('quat', ' '.join(map(str,q)))
    assets = {n.get('name'): n for n in scene.findall('asset/mesh')}
    materials = {n.get('name'): n for n in scene.findall('asset/material')}
    textures = {n.get('name'): n for n in scene.findall('asset/texture') if n.get('name')}
    ET.ElementTree(scene).write(output/'source_preview.xml',encoding='utf-8')
    rows, sources, loaded = [], [], {}
    def local(node):
        tf = np.eye(4)
        tf[:3,3] = np.fromstring(node.get('pos','0 0 0'), sep=' ')
        if node.get('quat'):
            q = np.fromstring(node.get('quat'),sep=' ')
            tf[:3,:3] = Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
        elif node.get('euler'):
            tf[:3,:3] = Rotation.from_euler('xyz',np.fromstring(node.get('euler'),sep=' ')).as_matrix()
        return tf
    def visit(parent, tf):
        for node in parent:
            name = node.get('name','')
            if node.tag == 'body':
                if name == 'base_link' or name == 'observed_pallet': continue
                visit(node, tf@local(node))
            elif node.tag == 'geom':
                if node.get('group') == '3' or name == 'preview_floor': continue
                material = materials.get(node.get('material'),ET.Element('material'))
                rgba = np.fromstring(node.get('rgba',material.get('rgba','1 1 1 1')),sep=' ')
                if len(rgba) == 4 and rgba[3] == 0: continue
                kind = node.get('type','sphere')
                size = np.fromstring(node.get('size','.001'),sep=' ')
                if node.get('mesh'):
                    asset = assets[node.get('mesh')];path = Path(asset.get('file'))
                    if str(path) not in loaded: loaded[str(path)] = mesh(path);sources.append(str(path))
                    value = loaded[str(path)].copy()
                    value.apply_scale(np.fromstring(asset.get('scale','1 1 1'),sep=' '))
                elif kind == 'box': value = trimesh.creation.box(extents=size*2)
                elif kind == 'cylinder': value = trimesh.creation.cylinder(radius=size[0],height=size[1]*2,sections=24)
                elif kind == 'sphere': value = trimesh.creation.icosphere(subdivisions=1,radius=size[0])
                elif kind == 'plane':
                    value = trimesh.creation.box(extents=[4,4,.002]);value.apply_translation([0,0,-.001])
                else: continue  # decorative capsule axes/lines are not contacts
                value.apply_transform(tf@local(node))
                group='carrier' if name.startswith(('registered_jig_','burger_')) else 'receiver' if name.startswith('linear_') else None
                if group=='carrier':
                    jig=jigs[0]
                    semantic=transform([*jig['pose'][:2],jig['bottom_z_mm']],jig['pose'][2])
                    semantic[:3,3]/=1000
                    value.apply_transform(np.linalg.inv(semantic))
                dest = output/'assets/static'/f'{len(rows):03d}_{name or "geom"}.stl'
                dest.parent.mkdir(parents=True,exist_ok=True);value.export(dest)
                collide = is_contact_geometry(name)
                surface=dict(roughness=max(.3,1-float(material.get('shininess',0))),
                             specular=float(material.get('specular',.05)),
                             emission=float(material.get('emission',0)),source_material=node.get('material'))
                texture=textures.get(material.get('texture'))
                if texture is not None and texture.get('file'):
                    surface.update(texture=str(Path(texture.get('file')).resolve()),
                                   texrepeat=np.fromstring(material.get('texrepeat','1 1'),sep=' ').tolist(),
                                   world_from_texture=(tf@local(node)).tolist())
                    sources.append(surface['texture'])
                rows.append(dict(name=name, mesh=str(dest), color=rgba[:3].tolist(), collision=collide,surface=surface,group=group))
    visit(scene.find('worldbody'),np.eye(4))
    return rows, list(set(sources)), placement, jigs

def prepare_scene(ref, output):
    rows, sources, placement, jigs = static_assets(ref, output)
    assembly_path = TEACH/'assets/carrier_assembly/assembly.json'
    assembly = read_json(assembly_path)
    jig = jigs[0]
    carrier = transform([*jig['pose'][:2],jig['bottom_z_mm']],jig['pose'][2]+jig.get('mesh_yaw_offset_deg',0))
    parts = {}
    for kind, (label, grasp, release) in PRODUCTS.items():
        source = PRODUCTS_DIR/f'B {label}.stl'
        product = mesh(source)
        fixture = next(p for p in assembly['parts'] if p['name'] == 'b_'+kind)
        fixture_path = assembly_path.parent/fixture['source']
        orientation = product_orientation(kind,product)
        oriented = product.copy();oriented.apply_transform(orientation)
        seat = seating_height(oriented,mesh(fixture_path))
        initial = carrier@transform(fixture['translation_mm'],fixture['yaw_deg'])@transform([0,0,seat])@orientation
        # A placement reference from the taught grasp transform; not a claim of
        # correctly snapped or deformed assembled geometry.
        goal = ref.kin.fk(ref.steps[release-1]['ticks'])@np.linalg.inv(ref.kin.fk(ref.steps[grasp]['ticks']))@initial
        mass = estimate_mass(kind,product)
        destination = output/'assets'/f'b_{kind}.stl'
        product.apply_scale(.001);product.export(destination)
        parts[kind] = dict(mesh=str(destination), source=str(source), source_sha256=sha(source),
                           initial_pose_mm=initial.tolist(), target_pose_mm=goal.tolist(),
                           grasp_step=grasp, release_step=release, seating_z_mm=seat,
                           color_rgb=[1.,.48,.08] if kind in ('cap','housing') else [.15,.35,.6],
                           initial_stl_orientation=orientation.tolist(),
                           orientation_source='user confirmed cap upside down on carrier' if kind=='cap' else 'original STL', **mass)
        sources += [str(source),str(fixture_path),mass['source_mesh'],mass['evidence']]
    robot_xml=TEACH/'assets/so101/so101_modified.xml'
    xml=ET.parse(robot_xml).getroot()
    rgb=lambda name:np.fromstring(xml.find(f".//geom[@mesh='{name}']").get('rgba'),sep=' ')[:3].tolist()
    return dict(static=rows, parts=parts, source_files=sorted(set([*sources,str(assembly_path),str(robot_xml),str(TEACH/'assets/so101/inspection_scene.xml')])),
                carrier=dict(jig_id=ref.jig_id,pose=ref.jig_pose,bottom_z_mm=jig['bottom_z_mm'],
                             coordinate_frame='arm2_base',mesh_yaw_offset_deg=jig.get('mesh_yaw_offset_deg',0)),
                robot_color_rgb=rgb('base_so101_v2'),finger_color_rgb=rgb('xlerobot_fixed_finger'),
                background_friction=.5, contact_offset_m=.00015,
                target_basis='taught grasp-to-pre-release transform; not measured final assembly',
                table_z_mm=ref.profile['table_z_mm'], workcell=placement)
