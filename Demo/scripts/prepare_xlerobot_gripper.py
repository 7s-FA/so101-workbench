"""Split the pinned XLeRobot printing plate and express its parts in SO-101 CAD frames.

No mesh rescaling/reshaping: the two rigid holders are registered to their
unchanged stock servo mounts. TPU finger assembly is a nominal mating-plane
fit; deformation and measured inertia are deliberately not claimed.
"""
from pathlib import Path
import hashlib
import json
import struct
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'assets/factory/xlerobot'
# Printed holder -> existing SO-101 mesh coordinates, millimetres. Registration
# used the shared servo attachment geometry; trimmed RMS 0.043 / 0.047 mm.
REGISTRATION={
 'fixed': (np.array([[-.000236768402,-.0000446186745,.999999971],[-.999999932,.000282383498,-.000236755793],[-.000282372926,-.999999959,-.0000446855310]]),np.array([-35.17301013,78.9680671,201.00105793])),
 'moving': (np.array([[.000386253162,-.999999925,.0000071618339],[-.999999780,-.000386249249,.000538569743],[-.000538566937,-.00000736985659,-.999999855]]),np.array([127.30378206,64.67345462,24.03329647])),
}


def write_stl(path,triangles):
    triangles=np.asarray(triangles,dtype=float)
    if not np.isfinite(triangles).all():raise ValueError('Non-finite gripper geometry')
    with path.open('wb') as file:
        file.write(b'XLeRobot source geometry; units mm'.ljust(80,b'\0'));file.write(struct.pack('<I',len(triangles)))
        for tri in triangles:
            n=np.cross(tri[1]-tri[0],tri[2]-tri[0]);length=np.linalg.norm(n)
            if length:n/=length
            file.write(struct.pack('<12fH',*n,*tri.ravel(),0))


def prepare_gripper(output):
    data=(SOURCE/'SO101_soft_fin.stl').read_bytes();count=struct.unpack_from('<I',data,80)[0]
    if len(data)!=84+count*50:raise ValueError('Unexpected printing plate format')
    triangles=np.array([struct.unpack_from('<9f',data,96+50*i) for i in range(count)]).reshape(-1,3,3)
    vertices,index=np.unique(np.round(triangles.reshape(-1,3),4),axis=0,return_inverse=True);faces=index.reshape(-1,3)
    a=np.r_[faces[:,0],faces[:,1],faces[:,2]];b=np.r_[faces[:,1],faces[:,2],faces[:,0]]
    _,labels=connected_components(coo_matrix((np.ones(len(a)),(a,b)),shape=(len(vertices),len(vertices))),directed=False)
    by_count={}
    for label in np.unique(labels):
        part=triangles[labels[faces[:,0]]==label]
        if len(part) in by_count:raise ValueError('Ambiguous printing plate components')
        by_count[len(part)]=part
    if set(by_count)!={3480,39690,18714}:raise ValueError('Upstream geometry changed; recheck assembly transforms')
    finger=by_count[3480]
    # Finger base plane and width axes in its print orientation.
    n=np.array([.994864,0,.101223]);n/=np.linalg.norm(n);v=np.array([0.,1.,0.]);w=np.cross(n,v)
    source_basis=np.column_stack((n,v,w));base_center=117.11*n+94.*v
    output.mkdir(parents=True,exist_ok=True);files={}
    for kind,number in [('fixed',39690),('moving',18714)]:
        r,t=REGISTRATION[kind]
        if kind=='fixed':
            target_n=np.array([0.,np.cos(np.deg2rad(5)),-np.sin(np.deg2rad(5))]);target_v=np.array([0.,np.sin(np.deg2rad(5)),np.cos(np.deg2rad(5))]);center=np.array([79.234,159.99,10.31])
        else:
            target_n=-np.array([np.cos(np.deg2rad(3)),np.sin(np.deg2rad(3)),0.]);target_v=np.array([-np.sin(np.deg2rad(3)),np.cos(np.deg2rad(3)),0.]);center=np.array([84.11,128.40,24.78])
        fit=np.column_stack((target_n,target_v,np.cross(target_n,target_v)))@source_basis.T
        local_finger=(finger-base_center)@fit.T+center
        for role,mesh in [('holder',by_count[number]),('finger',local_finger)]:
            name=f'xlerobot_{kind}_{role}.stl';write_stl(output/name,mesh@r.T+t);files[kind+'_'+role]=str((output/name).resolve())
    report={'source':json.loads((SOURCE/'source.json').read_text()),'source_file':'hardware/SO101_soft_fin.stl',
            'source_sha256':hashlib.sha256(data).hexdigest(),'files':files,'units':'mm',
            'holder_registration_trimmed_rms_mm':{'fixed':.0424,'moving':.0468},
            'finger_mount_status':'nominal mating-plane fit; screw fit requires physical confirmation',
            'deformation_simulated':False,'inertia_measured':False}
    (output/'assembly.json').write_text(json.dumps(report,indent=2)+'\n');return report

if __name__=='__main__':
    prepare_gripper(ROOT/'assets/generated/conveyor_demo/gripper')
