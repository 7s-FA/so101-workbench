#!/usr/bin/env python3
"""Prepare separate three-arm assets; never launches a server or controls hardware.

STL files contain no unit metadata. The supplied printing assets use millimetres;
the existing simulation meshes use metres. Geometry reports retain this assumption.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import struct
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MAX_STL_BYTES = 20 * 1024 * 1024



def mesh_summary(data, scale=.001):
    """Read binary or ASCII STL, rejecting empty, truncated and non-finite meshes."""
    if len(data) > MAX_STL_BYTES:
        raise ValueError('STL exceeds 20 MiB')
    count = struct.unpack_from('<I', data, 80)[0] if len(data) >= 84 else 0
    if count and len(data) == 84 + 50 * count:
        vertices = (v for i in range(count)
                    for v in (struct.unpack_from('<3f', data, 96 + 50*i + 12*j) for j in range(3)))
    else:
        try:
            lines = data.decode('ascii').strip().splitlines()
        except UnicodeDecodeError as error:
            raise ValueError('Invalid or truncated binary STL') from error
        points = [tuple(map(float, line.split()[1:])) for line in lines
                  if line.strip().startswith('vertex ')]
        if not points or len(points) % 3 or not lines or not lines[0].strip().startswith('solid') or not any(line.strip().startswith('endsolid') for line in lines):
            raise ValueError('Invalid STL')
        count = len(points)//3
        vertices = iter(points)
    lo, hi = [math.inf]*3, [-math.inf]*3
    for vertex in vertices:
        if len(vertex) != 3 or not all(math.isfinite(v) for v in vertex):
            raise ValueError('Invalid STL vertex')
        for axis, value in enumerate(vertex):
            lo[axis] = min(lo[axis], value*scale)
            hi[axis] = max(hi[axis], value*scale)
    extent = [b-a for a, b in zip(lo, hi)]
    if min(extent) <= 0:
        raise ValueError('Empty or flat STL geometry')
    return {'sha256': hashlib.sha256(data).hexdigest(), 'triangles': count,
            'bounds_m': [lo, hi], 'extent_m': extent, 'assumed_scale_to_metres': scale}


def import_parts(archive, destination, expected):
    """Copy only named STL assets; do not extract archive paths or run its contents."""
    pending = {}
    with zipfile.ZipFile(archive) as zipped:
        for info in zipped.infolist():
            name = Path(info.filename).name
            if info.is_dir() or name not in expected:
                continue
            if name in pending:
                raise ValueError('Ambiguous duplicate STL: '+name)
            if info.file_size > MAX_STL_BYTES:
                raise ValueError('Oversized STL: '+name)
            data = zipped.read(info)
            mesh_summary(data)
            pending[name] = data
    if not pending:
        raise ValueError('No expected revision STL files found in archive')
    destination.mkdir(parents=True, exist_ok=True)
    for name, data in pending.items():
        (destination/name).write_bytes(data)
    return sorted(pending)


def prepare(output, parts_zip=None):
    cfg = json.loads((ROOT/'config/conveyor_demo.json').read_text())
    output.mkdir(parents=True, exist_ok=True)
    parts = output/'parts'
    if parts_zip:
        import_parts(parts_zip, parts, set(cfg['parts'].values()))
    original = ROOT/'assets/generated/so101.urdf'
    tree = ET.parse(original)
    tree.getroot().find("material[@name='3d_printed']/color").set('rgba','0.045 0.055 0.065 1')
    from scripts.prepare_xlerobot_gripper import prepare_gripper
    gripper = prepare_gripper(output/'gripper')
    for link,kind,oldname in [('gripper_link','fixed','wrist_roll_follower_so101_v1.stl'),
                              ('moving_jaw_so101_v1_link','moving','moving_jaw_so101_v1.stl')]:
        node=tree.getroot().find(f"link[@name='{link}']")
        for tag in ('visual','collision'):
            old=next(e for e in node.findall(tag) if e.find('geometry/mesh') is not None and Path(e.find('geometry/mesh').get('filename')).name==oldname)
            for role in ('holder','finger'):
                new=copy.deepcopy(old);new.set('name','xlerobot_'+kind+'_'+role)
                mesh=new.find('geometry/mesh');mesh.set('filename',gripper['files'][kind+'_'+role]);mesh.set('scale','0.001 0.001 0.001')
                if tag=='visual':new.find('material').set('name','fin_blue' if role=='finger' else 'sts3215')
                node.append(new)
            node.remove(old)
    for tag,names in [('link',('wrist_camera_link','wrist_camera_mount_link')),
                      ('joint',('wrist_camera_joint','wrist_camera_mount_joint'))]:
        for name in names:
            node=tree.getroot().find(f"{tag}[@name='{name}']")
            if node is not None:tree.getroot().remove(node)
    material=ET.SubElement(tree.getroot(),'material',name='fin_blue')
    ET.SubElement(material,'color',rgba='0.025 0.26 0.67 1')
    urdf = output/'so101_xlerobot.urdf'
    tree.write(urdf, encoding='utf-8', xml_declaration=True)
    models = json.loads((ROOT/'config/models.json').read_text())
    revised = {'omx': copy.deepcopy(models['omx']), 'so101_xlerobot': copy.deepcopy(models['so101'])}
    # Keep the tool frame/joint for IK, omit only its floating display cube.
    omx_tree=ET.parse(ROOT/models['omx']['urdf'])
    tool=omx_tree.getroot().find("link[@name='end_effector_link']")
    for visual in list(tool.findall('visual')):tool.remove(visual)
    omx_urdf=output/'omx_no_marker.urdf'
    omx_tree.write(omx_urdf,encoding='utf-8',xml_declaration=True)
    revised['omx']['urdf']=str(omx_urdf.resolve())
    revised['so101_xlerobot'].update(urdf=str(urdf.resolve()),
        geometry_status='XLeRobot source meshes; nominal finger mounting; rigid geometry only',
        inertia_status='inherited_stock_estimate_not_measured',
        camera_status='D435 body and printed mount; nominal transforms')
    (output/'models.json').write_text(json.dumps(revised, indent=2)+'\n')
    missing = [name for name in cfg['parts'].values() if not (parts/name).is_file()]
    report = {
        'status': 'layout_and_isolated_grasp_assets',
        'runnable_demo': False,
        'missing_parts': missing,
        'imported_parts': {name: mesh_summary((parts/name).read_bytes())
                           for name in cfg['parts'].values() if (parts/name).is_file()},
        'gripper_assembly': gripper,
        'remaining': ['physical confirmation of finger and camera mounts', 'new product assembly offsets', 'fixed-kit seating validation',
                      'four-arm motion implementation for the current purchased-table layout', 'full motion and collision verification'],
    }
    (output/'preparation.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parts-zip', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT/'assets/generated/conveyor_demo')
    args = parser.parse_args()
    report = prepare(args.output.resolve(), args.parts_zip)
    print(json.dumps({'status': report['status'], 'missing_parts': report['missing_parts'],
                      'report': str(args.output.resolve()/'preparation.json')}, ensure_ascii=False, indent=2))
