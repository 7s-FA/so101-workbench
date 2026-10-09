"""Shared mechanical geometry for the preview and its static audit."""
import math
import xml.etree.ElementTree as ET
from functools import lru_cache
import numpy as np
from scipy.spatial.transform import Rotation
from arm_sim.common import ROOT

# Measured support outline of the stock SO101 base STL in its URDF frame.
BASE_FOOT=(-.02236485,.06463546,-.05546243,.05546260)


def mount_plate(cfg,arm):
    """No supplemental support plate: the original URDF base rests on wood."""
    return None



def clamp_boxes(cfg,arm):
    p=cfg['arms'][arm];x,y=p['position'];z=cfg['table_height']
    f=cfg['arm_fixtures'][arm];t=next(t for t in cfg['tables'] if t['id']==f['table'])
    def clamp(i,axis,edge,side,tangent,pad_bottom,contact=None):
        rail=edge+side*.016;pad=contact if contact is not None else edge-side*f.get('clamp_pad_depth',.010)
        pad_width=f.get('clamp_pad_size',.012);upper_thickness=f.get('clamp_upper_thickness',.010)
        def pos(n,h):return (n,tangent,h) if axis==0 else (tangent,n,h)
        def size(n,t,h):return (n,t,h) if axis==0 else (t,n,h)
        return [
            (f'Pad{i}',pos(pad,pad_bottom+.001),size(pad_width,pad_width,.002)),
            (f'UpperJaw{i}',pos((rail+pad)/2,pad_bottom+.002+upper_thickness/2),size(max(.050,abs(rail-pad)+.012),f.get('clamp_jaw_width',.012),upper_thickness)),
            (f'Rail{i}',pos(rail,z-.025),size(.010,.010,.130)),
            (f'LowerPad{i}',pos(pad,z-.03101),size(.014,.014,.002)),
            (f'LowerJaw{i}',pos((rail+pad)/2,z-.03701),size(max(.050,abs(rail-pad)+.012),f.get('clamp_jaw_width',.012),.010)),
        ]
    if 'custom_pads' in f:
        pieces=[]
        bottom=z+p.get('mount_height',0.)+p['base_origin_offset']+.01512
        for i,spec in enumerate(f['custom_pads']):
            axis=0 if spec['side'] in ('left','right') else 1
            side=-1 if spec['side'] in ('left','front') else 1
            edge=t['center'][axis]+side*t['size'][axis]/2
            pieces.extend(clamp(i,axis,edge,side,spec['position'][1-axis],bottom,spec['position'][axis]))
        return pieces
    if p['model']=='omx' and 'clamp_side' not in f:
        side=-1 if f['side']=='left' else 1
        front=t['center'][1]-t['size'][1]/2
        edge=t['center'][0]+side*t['size'][0]/2
        # Grip the original 120x150mm base directly at its 13mm-high flat rim.
        return (clamp(0,1,front,-1,x-side*.040,z+.01301)+
                clamp(1,0,edge,side,y-side*.040,z+.01301))
    # Base alignment and clamp access are distinct: B/D stay at their corners,
    # with both clamps at the local rear (-X) rim, from table left/right.
    clamp_side=f.get('clamp_side',f['side'])
    axis=0 if clamp_side in ('left','right') else 1
    side=-1 if clamp_side in ('left','front') else 1
    edge=t['center'][axis]+side*t['size'][axis]/2
    pad_bottom=z+.01301 if p['model']=='omx' else z+p.get('mount_height',0.)+p['base_origin_offset']+.01512
    return sum((clamp(i,axis,edge,side,p['position'][1-axis]+delta,pad_bottom)
                for i,delta in enumerate(f.get('clamp_offsets',(-.038,.038)))),[])


@lru_cache(maxsize=1)
def d435_vertices():
    """Pinned DAE has metre units, triangle primitives, and no node transforms."""
    ns={'c':'http://www.collada.org/2005/11/COLLADASchema'}
    root=ET.parse(ROOT/'assets/factory/realsense/d435.dae').getroot()
    if root.findall('.//c:matrix',ns) or root.findall('.//c:polylist',ns):raise ValueError('Recheck changed D435 CAD')
    triangles=[]
    for mesh in root.findall('.//c:geometry/c:mesh',ns):
        sources={s.get('id'):np.fromstring(s.find('c:float_array',ns).text,sep=' ').reshape(-1,3) for s in mesh.findall('c:source',ns)}
        verts={v.get('id'):v.find("c:input[@semantic='POSITION']",ns).get('source')[1:] for v in mesh.findall('c:vertices',ns)}
        for tr in mesh.findall('c:triangles',ns):
            inputs=tr.findall('c:input',ns);stride=max(int(i.get('offset')) for i in inputs)+1
            inp=next(i for i in inputs if i.get('semantic')=='VERTEX')
            ix=np.fromstring(tr.find('c:p',ns).text,sep=' ',dtype=int).reshape(-1,stride)[:,int(inp.get('offset'))]
            triangles.append(sources[verts[inp.get('source')[1:]]][ix])
    v=np.concatenate(triangles)
    r=Rotation.from_euler('xyz',[math.pi/2,0,math.pi/2]).as_matrix()
    return v@r.T+[.0149,0,.0125]


# User confirmed five newly drilled mounting holes, with direct side mounting.
# Registration is photo-based; it need not coincide with the old access opening.
D435_HOLDER_SPACER = 0.0
# Keep both mount and camera rotations fixed. Translate 1mm toward the finger
# to match the side-profile gap in the close-up photo (approx. 4.4 +/- 1mm).
D435_MOUNT_CENTER = np.array([-.03532,0,-.0174])
D435_REAR_WASHER = .0005
D435_HORIZONTAL_ROLL = -math.pi/2 + .0486795


def d435_assembly():
    """Original bracket on the fixed opening-motor holder's left end face.

    The large bracket hole is NOT the moving finger pivot. Five newly drilled
    holes permit direct mounting on the -X side without the old 8mm spacers.
    The in-plane registration is approximate, not a measurement of the holes.
    Both outputs are in gripper_link coordinates; camera +X is optical forward.
    """
    ex=np.array([.999998657,-.00163883,0]);ex/=np.linalg.norm(ex)
    ez=np.array([.00006168,.03736868,.99930154]);ey=np.cross(ez,ex);ey/=np.linalg.norm(ey);ez=np.cross(ex,ey)
    cad=np.column_stack((ex,ey,ez))
    target=np.array([[0.,0,-1],[1,0,0],[0,-1,0]])
    r=target@cad.T
    hole=cad@np.array([-.00086328181,.17908324494,.36175903])
    mount=np.eye(4);mount[:3,:3]=r
    mount[:3,3]=D435_MOUNT_CENTER-r@hole
    rear_n=np.array([-.4211261,.90630706,-.03550106]);rear_n/=np.linalg.norm(rear_n)
    screw_center=np.array([.05476256070,.18479426958,.31990952553])
    forward=r@rear_n
    left=-r@cad[:,2];left-=forward*(left@forward);left/=np.linalg.norm(left)
    camera=np.eye(4);camera[:3,:3]=np.column_stack((forward,left,np.cross(forward,left)))
    local_contact=np.array([d435_vertices()[:,0].min(),0,.0125])
    camera[:3,3]=mount[:3,3]+r@screw_center-camera[:3,:3]@local_contact+forward*D435_REAR_WASHER
    return mount,camera


def d435_adapter_meshes():
    """Five screw heads on the bracket; two provisional camera-seat washers.

    Heads indicate the user-confirmed drilled fastening pattern. Their exact
    dimensions and the bores through the holder are not fabrication geometry.
    No holder standoffs remain. Camera rear seating is still approximate.
    """
    def ring(center,axis,length,outer,inner):
        axis=np.asarray(axis);axis=axis/np.linalg.norm(axis)
        u=np.cross(axis,[0,0,1] if abs(axis[2])<.9 else [0,1,0]);u/=np.linalg.norm(u);v=np.cross(axis,u)
        rows=[]
        for end in (-.5,.5):
            for radius in (inner,outer):
                rows.append(np.array([center+axis*length*end+radius*(u*math.cos(t)+v*math.sin(t)) for t in np.linspace(0,2*math.pi,33)[:-1]]))
        triangles=[]
        for i in range(32):
            j=(i+1)%32
            for a,b in ((0,1),(2,3),(0,2),(1,3)):
                triangles.extend([rows[a][i],rows[a][j],rows[b][j],rows[a][i],rows[b][j],rows[b][i]])
        return np.array(triangles)
    result=[]
    for i,(y,z) in enumerate(((-.0125,.0095),(.0125,.0095),(-.0125,-.0095),(.0125,-.0095),(0,-.0155))):
        # Three-millimetre bracket plate, 2mm head; 0.05mm display clearance.
        center=D435_MOUNT_CENTER+np.array([-.00405,y,z])
        result.append((f'HolderScrewHead{i}',ring(center,[-1,0,0],.002,.0027,.0008)))
    mount,camera=d435_assembly();normal=camera[:3,0]
    for i,raw in enumerate(([.05499311147,.18402061511,.29742401528],[.05453200993,.18556792404,.34239503577])):
        center=mount[:3,:3]@raw+mount[:3,3]+normal*D435_REAR_WASHER/2
        result.append((f'CameraWasher{i}',ring(center,normal,D435_REAR_WASHER-.00004,.0025,.0015)))
    return result


def d435_optical_pose(gripper_world):
    """Nominal depth imager origin, without real intrinsic/extrinsic calibration."""
    _,camera=d435_assembly();world=gripper_world@camera
    eye=world[:3,3]+world[:3,:3]@np.array([.0149,.0175,.0125])
    return eye,world[:3,0],world[:3,2]


def trident_boxes(center,size,height,prefix='',thickness=.016):
    """Three connected shelf fingers, rear crossbeam, four legs; two open fork slots."""
    x,y=center;w,d=size;h=height;result=[]
    for i,(lo,hi) in enumerate(((-w/2,-.073),(-.037,.037),(.073,w/2))):
        result.append((prefix+f'Slat{i}',(x+(lo+hi)/2,y-.008,h-thickness/2),(hi-lo,d-.016,thickness)))
    result.append((prefix+'RearBeam',(x,y+d/2-.008,h-thickness/2),(w,.016,thickness)))
    for i,(sx,sy) in enumerate(((-1,-1),(-1,1),(1,-1),(1,1))):
        result.append((prefix+f'Leg{i}',(x+sx*(w/2-.012),y+sy*(d/2-.012),(h-thickness)/2),(.024,.024,h-thickness)))
    return result


def warehouse_boxes(rack):
    """Two trident shelves on shared legs; lower shelf notched at leg footprints."""
    top=max(rack['levels']);parts=trident_boxes(rack['center'],rack['size'],top,'Top',rack.get('shelf_thickness',.016))
    # Lower shelf is attached between the common legs, above the floor. Corner reliefs accommodate
    # the common legs, so separate pieces do not occupy the same wood volume.
    x,y=rack['center'];w,d=rack['size']
    for level,h in enumerate(rack['levels'][:-1]):
        thickness=rack.get('shelf_thickness',rack.get('lower_shelf_thickness',.016))
        parts.append((f'Lower{level}Center',(x,y-.008,h-thickness/2),(.074,d-.016,thickness)))
        for side in (-1,1):
            lo,hi=.073,w/2;inner=hi-.024
            parts.extend([(f'Lower{level}Side{int(side>0)}',(x+side*(lo+inner)/2,y-.008,h-thickness/2),(inner-lo,d-.016,thickness)),
                          (f'Lower{level}Tip{int(side>0)}',(x+side*(inner+hi)/2,y,h-thickness/2),(.024,d-.048,thickness))])
        parts.append((f'Lower{level}Rear',(x,y+d/2-.008,h-thickness/2),(w-.048,.016,thickness)))
    return parts


def shipping_boxes(cfg):
    if cfg.get('trident_supports'):return trident_boxes(cfg['shipping_dock'],cfg['shipping_support_size'],cfg['shipping_height'])
    x,y=cfg['shipping_dock'];h=cfg['shipping_height'];result=[]
    for i,(lo,hi) in enumerate(((-.135,-.073),(-.037,.037),(.073,.135))):
        result.append((f'Slat{i}',(x+(lo+hi)/2,y,h-.008),(hi-lo,.125,.016)))
    for i,dx in enumerate((-.12,.12)):
        result.append((f'Leg{i}',(x+dx,y+.046,(h-.016)/2),(.024,.024,h-.016)))
    # Front feet stabilize the low stand without crossing either fork channel.
    for i,dx in enumerate((-.12,.12)):
        result.append((f'FrontLeg{i}',(x+dx,y-.046,(h-.016)/2),(.024,.024,h-.016)))
    return result


def forklift_boxes(cfg,top=None):
    """Concept bolt-on subframe + guided screw lift. Units in chassis frame."""
    w=cfg['waffle'];top=w['fork_top'] if top is None else top
    fixed=[];moving=[]
    for i,side in enumerate((-1,1)):
        fixed.extend([
            (f'ChassisRail{i}',(-.025,side*.100,.1294),(.210,.018,.012)),
            (f'Cantilever{i}',(.110,side*.100,.1294),(.080,.018,.012)),
            (f'Mast{i}',(.145,side*.079,.1205),(.018,.018,.227)),
            (f'MastAnchor{i}',(.132,side*.09,.1294),(.010,.050,.016)),
            (f'LowerBrace{i}',(.117,side*.079,.035),(.070,.018,.014)),
        ])
        moving.extend([
            (f'Slider{i}',(.163,side*.079,top+w.get('slider_offset',.042)),(.018,.030,.040)),
            (f'Heel{i}',(.187,side*.055,top+.015),(.012,.022,.036)),
            (f'Tine{i}',(.187+w['fork_length']/2,side*.055,top-w.get('fork_thickness',.006)/2),(w['fork_length'],.022,w.get('fork_thickness',.006))),
        ])
    fixed.extend([
        ('CrossMount',(.025,0,.1294),(.022,.218,.012)),
        ('TopBrace',(.145,0,.227),(.024,.176,.014)),
        ('BasePlate',(.145,0,.005),(.038,.176,.006)),
        ('DriveMotor',(.145,0,.022),(.028,.036,.028)),
        ('LeadScrew',(.145,0,.131),(.008,.008,.182)),
        ('MotorCoupler',(.145,0,.038),(.012,.012,.006)),
    ])
    moving.extend([
        ('Carriage',(.179,0,top+.032),(.014,.190,.022)),
        ('ScrewNut',(.145,0,top+w.get('slider_offset',.042)),(.020,.018,.020)),
        ('NutBridge',(.160,0,top+w.get('slider_offset',.042)),(.026,.018,.015)),
    ])
    return fixed,moving
