"""Recover unchanged, base-mounted camera calibration after a motor-zero change."""
from copy import deepcopy
import json
from pathlib import Path
from vision_teaching import digest,require_intrinsics,pixels_on_plane


def device_path(source):
    text=str(source)
    path=Path('/dev/video'+text) if text.isdecimal() else Path(text)
    try:return path.resolve(strict=True)
    except OSError:return None


def recover_camera(station,folder):
    # Never replace an existing or partially configured calibration.
    if station.get('intrinsics') is not None or station.get('extrinsics') is not None:return station,None
    identity=station['identity'];camera=station['camera']
    if identity.get('demo'):return station,None
    current_device=device_path(camera['source'])
    if current_device is None:return station,None
    def physical(value):return {k:v for k,v in value.items() if k!='calibration_sha256'}
    from model_camera import SOURCE
    geometry_hash=digest(json.loads(SOURCE.read_text()))
    choices={}
    for path in sorted(Path(folder).glob('*.json'),key=lambda p:p.stat().st_mtime):
        try:
            old=json.loads(path.read_text())
            if physical(old['identity'])!=physical(identity):continue
            previous=old['camera'];ext=old.get('extrinsics') or {}
            if any(previous[k]!=camera[k] for k in ('width','height')):continue
            if device_path(previous['source'])!=current_device:continue
            if old['board']!=station['board']:continue
            # This recovery is for the existing fixed CAD mount, independent of motor zero.
            if ext.get('source')!='cad_estimate' or ext.get('geometry_sha256')!=geometry_hash:continue
            size=[previous['width'],previous['height']]
            require_intrinsics(old,size);pixels_on_plane([[size[0]/2,size[1]/2]],0,old,size)
            payload={k:deepcopy(old[k]) for k in ('camera','intrinsics','extrinsics')}
            comparison=deepcopy(payload)
            comparison['extrinsics'].pop('verified',None)  # Old informational field, not a transform.
            choices[digest(comparison)]=(payload,path)
        except (OSError,ValueError,KeyError,TypeError):continue
    if len(choices)!=1:return station,None
    payload,path=next(iter(choices.values()));result=deepcopy(station);result.update(payload)
    return result,path.name
