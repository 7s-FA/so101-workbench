"""Read extrusion as data. This module never sends or executes G-code."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile

FIELDS=re.compile(r'([XYZEFIJP])\s*(-?(?:\d+(?:\.\d*)?|\.\d+))')
START=re.compile(r'; start printing object, unique label id: (\d+)')
AUXILIARY={'Brim','Skirt','Support','Support interface','Wipe tower','Custom','Prime tower'}


def extrusion(lines):
    relative=True;absolute_e=0.;debt=0.;object_id=None;feature='Custom'
    x=y=z=0.;xyz_absolute=True
    stats=defaultdict(lambda:defaultdict(float));bounds={};positive=negative=0.
    header={};counts=defaultdict(int)
    for raw in lines:
        line=raw.decode('utf-8',errors='replace').strip() if isinstance(raw,bytes) else raw.strip()
        start=START.match(line)
        if start:object_id=start.group(1);continue
        if line.startswith('; stop printing object'):object_id=None;continue
        if line.startswith('; FEATURE:'):feature=line.split(':',1)[1].strip();continue
        if line.startswith('; total filament '):
            key,val=line[2:].split(':',1);header[key.strip()]=float(val.strip());continue
        if not line or line[0]==';':continue
        code=line.split(';',1)[0].strip();command=code.split(None,1)[0]
        if command=='M82':relative=False;continue
        if command=='M83':relative=True;continue
        if command=='G90':xyz_absolute=True;continue
        if command=='G91':xyz_absolute=False;continue
        if command not in ('G0','G1','G2','G3','G92'):continue
        fields={k:float(v) for k,v in FIELDS.findall(code)}
        if command=='G92':
            if 'E' in fields:absolute_e=fields['E']
            if 'X' in fields:x=fields['X']
            if 'Y' in fields:y=fields['Y']
            if 'Z' in fields:z=fields['Z']
            continue
        nx=fields.get('X',x) if xyz_absolute else x+fields.get('X',0.)
        ny=fields.get('Y',y) if xyz_absolute else y+fields.get('Y',0.)
        z=fields.get('Z',z) if xyz_absolute else z+fields.get('Z',0.)
        spatial=(abs(nx-x)+abs(ny-y)>1e-10) or (command in ('G2','G3') and ('I' in fields or 'J' in fields))
        x,y=nx,ny
        if 'E' not in fields:continue
        amount=fields['E'] if relative else fields['E']-absolute_e
        absolute_e=absolute_e+amount if relative else fields['E']
        if amount<0:debt-=amount;negative-=amount;continue
        positive+=amount
        recovered=min(amount,debt);debt-=recovered;deposited=amount-recovered
        category=feature if spatial else 'stationary_extrusion'
        stats[object_id or 'unassigned'][category]+=deposited
        counts[command]+=1
        if deposited>0 and spatial and object_id:
            b=bounds.setdefault(object_id,[x,y,x,y,z])
            b[0]=min(b[0],x);b[1]=min(b[1],y);b[2]=max(b[2],x);b[3]=max(b[3],y)
            b[4]=max(b[4],z)
    return {'extrusion_mm_by_object_feature':{k:dict(v) for k,v in stats.items()},'path_bounds_mm':bounds,
            'header':header,'e_positive_mm':positive,'e_negative_mm':negative,'retraction_debt_mm':debt,
            'extrusion_command_counts':dict(counts)}


def analyze(path):
    path=Path(path)
    with zipfile.ZipFile(path) as archive:
        settings=json.loads(archive.read('Metadata/project_settings.config'))
        info=ET.fromstring(archive.read('Metadata/slice_info.config'))
        plate=info.find('plate')
        if plate is None:raise ValueError('No sliced plate')
        filaments=plate.findall('filament')
        if len(filaments)!=1:raise ValueError('Per-filament extrusion tracking required for this archive')
        index=int(filaments[0].get('id'))-1
        density=float(settings['filament_density'][index]);diameter=float(settings['filament_diameter'][index])
        factor=math.pi*(diameter/2)**2*density/1000
        gcodes=[n for n in archive.namelist() if n.endswith('.gcode')]
        if len(gcodes)!=1:raise ValueError('Expected one sliced plate')
        with archive.open(gcodes[0]) as stream:result=extrusion(stream)
        names={obj.get('identify_id'):obj.get('name') for obj in plate.findall('object')}
        rows=[]
        for ident,features in result['extrusion_mm_by_object_feature'].items():
            clean=sum(e for feature,e in features.items() if feature not in AUXILIARY and feature!='stationary_extrusion')
            auxiliary=sum(e for feature,e in features.items() if feature in AUXILIARY and feature!='Custom')
            rows.append({'id':ident,'name':names.get(ident),'part_mass_g':clean*factor,
                         'auxiliary_mass_g':auxiliary*factor,'all_deposited_mass_g':sum(features.values())*factor,
                         'extrusion_mm_by_feature':features,'bounds_mm':result['path_bounds_mm'].get(ident)})
        result.update(source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      density_g_cm3=density,filament_diameter_mm=diameter,
                      plate_mass_g=float(next(m.get('value') for m in plate.findall('metadata') if m.get('key')=='weight')),
                      profile={k:settings.get(k) for k in ('filament_settings_id','sparse_infill_density','sparse_infill_pattern','wall_loops','layer_height','top_shell_layers','bottom_shell_layers')},
                      objects=rows,method='Net extruded filament after retraction recovery; separate auxiliary/custom/stationary extrusion. No scaling to force plate total.')
        return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('archive');parser.add_argument('output')
    args=parser.parse_args();result=analyze(args.archive)
    Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    for obj in result['objects']:print(obj['id'],obj['name'],round(obj['part_mass_g'],6),'aux',round(obj['auxiliary_mass_g'],6),'bounds',obj['bounds_mm'])
    print('Plate',result['plate_mass_g'],'g; all net E',sum(o['all_deposited_mass_g'] for o in result['objects']),'g')
