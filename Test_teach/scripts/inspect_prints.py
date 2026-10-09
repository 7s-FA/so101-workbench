"""Read Bambu 3MF metadata as data; do not execute G-code or extract archive paths."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILES = {'ball': '공.gcode.3mf', 'cap': '2단.gcode.3mf',
         'housing': '1단.gcode.3mf', 'so101_finger': 'so101 tpu그립.3mf'}
KEYS = ('filament_settings_id', 'filament_type', 'filament_density',
        'sparse_infill_density', 'sparse_infill_pattern', 'wall_loops',
        'layer_height', 'top_shell_layers', 'top_shell_thickness',
        'bottom_shell_layers', 'bottom_shell_thickness', 'fuzzy_skin',
        'fuzzy_skin_point_distance', 'fuzzy_skin_thickness')

def inspect(path):
    with zipfile.ZipFile(path) as archive:
        settings = json.loads(archive.read('Metadata/project_settings.config'))
        sliced = ET.fromstring(archive.read('Metadata/slice_info.config'))
        plates = []
        for plate in sliced.findall('plate'):
            metadata = {n.get('key'): n.get('value') for n in plate.findall('metadata')}
            plates.append({'metadata': metadata,
                           'objects': [dict(n.attrib) for n in plate.findall('object')],
                           'filaments': [dict(n.attrib) for n in plate.findall('filament')]})
        return {'filename': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'settings': {k: settings.get(k) for k in KEYS}, 'plates': plates,
                'contains_gcode': any(n.endswith('.gcode') for n in archive.namelist()),
                'mass_status': 'plate filament estimate, not measured finished-part mass'}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=Path('/home/fresh/Downloads'))
    args = parser.parse_args()
    report = {'source': 'Bambu Studio attached projects',
              'items': {kind: inspect(args.input_dir / name) for kind, name in FILES.items()},
              'limitations': ['No G-code execution.', 'Friction and gripper surface are not specified by these projects.',
                              'Housing/cap saved infill is 15%, conflicting with earlier user report of 5%.',
                              'SO101 finger project is separate from the OMX test.']}
    output = ROOT / 'reports/print_settings.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(output)

if __name__ == '__main__':
    main()
