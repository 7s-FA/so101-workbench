#!/usr/bin/env python3
"""Verify supplied parts clear their exact jig meshes at the recorded seating pose."""
import json
import numpy as np
from arm_sim.common import ROOT
from arm_sim.grasp_geometry import stl_triangles,triangle_model,collision_object,separation


def check():
    cfg=json.loads((ROOT/'config/conveyor_demo.json').read_text());s=cfg['jig_seating'];directory=ROOT/'assets/generated/conveyor_demo/parts'
    rows={};failures=[]
    for kind in ('housing','ball','cap'):
        source=ROOT/'assets/product/02_ball_d30.stl' if kind=='ball' else directory/cfg['parts'][kind]
        part_mesh=triangle_model(stl_triangles(source)*.001)
        jig=collision_object(triangle_model(stl_triangles(directory/cfg['parts'][kind+'_jig'])*.001),np.eye(3),np.zeros(3))
        height=s['contact_height'][kind]+s['render_gap'];position=[*s['offset_xy'],height]
        part=collision_object(part_mesh,np.eye(3),position);gap,points=separation(part,jig)
        lower=collision_object(part_mesh,np.eye(3),[*s['offset_xy'],height-.0002]);lower_gap,_=separation(lower,jig)
        rows[kind]={'position_in_jig_m':position,'clearance_m':gap,'contact_if_lowered_0_2_mm':lower_gap<=1e-7,'nearest_points_m':points.tolist()}
        if not .000001<gap<.0002 or lower_gap>1e-7:failures.append(kind)
    return {'status':'seating_clear' if not failures else 'blocked','parts':rows,'failures':failures,
            'scope':'Geometric fit at rest with 0.05 mm render gap; does not establish insertion path, finger clearance or actual tolerance.'}

if __name__=='__main__':
    report=check();out=ROOT/'outputs/conveyor_preview_latest/seating_check.json';out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));raise SystemExit(bool(report['failures']))
