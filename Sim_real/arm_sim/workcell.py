"""Uploaded STL assets and a repeatable two-table PhysX workcell (metres)."""
import hashlib
import json
import math
import struct
from pathlib import Path
from arm_sim.common import ROOT

FILES = {'housing':'01_housing.stl','ball':'02_ball_d30.stl','cap':'03_cap_print_upside_down.stl','jig':'04_optional_assembly_jig.stl'}
COLORS = {'housing':(.12,.39,.68),'ball':(.95,.55,.12),'cap':(.10,.65,.58),'jig':(.27,.30,.34)}

def config():
    return json.loads((ROOT/'config/assembly.json').read_text())

def signature(models):
    h = hashlib.sha256()
    for p in [ROOT/'config/assembly.json',ROOT/'scripts/sim.py',ROOT/'config/source_revisions.json',Path(__file__),*[ROOT/'assets/product'/f for f in FILES.values()]]:
        h.update(p.read_bytes())
    h.update(json.dumps(models,sort_keys=True).encode())
    for model in models.values():
        h.update((ROOT/model['urdf']).read_bytes())
    return h.hexdigest()

def vertices(kind):
    data = (ROOT/'assets/product'/FILES[kind]).read_bytes()
    n = struct.unpack_from('<I',data,80)[0]
    if len(data)!=84+50*n:
        raise ValueError('Expected a binary STL')
    points=[]
    for i in range(n):
        values=struct.unpack_from('<12f',data,84+50*i)
        for j in (3,6,9):
            x,y,z=(v*.001 for v in values[j:j+3])
            if kind=='cap':
                y,z=-y,.0077-z  # print-upside-down -> skirt down, base at z=0
            points.append((x,y,z))
    return points

def slot_centers(tray):
    x,y=tray['center']; p=tray['pitch']/2
    return [(x+dx,y+dy) for dy in (-p,p) for dx in (-p,p)]

class Workcell:
    def __init__(self, stage, layout=None):
        from pxr import UsdGeom, UsdShade, UsdPhysics, PhysxSchema, Gf, Sdf
        self.stage=stage
        self.G,self.Sh,self.P,self.X,self.F,self.S=UsdGeom,UsdShade,UsdPhysics,PhysxSchema,Gf,Sdf
        self.cfg=config() if layout is None else layout; self.bodies={}; self.initial={}
        self.materials={}
        for name,color in {**COLORS,'tray':(.72,.75,.71),'prep':(.46,.60,.64),'finished':(.50,.64,.42),'metal':(.12,.14,.17)}.items():
            self.materials[name]=self.material(name,color)
        self.wood=[self.material('wood'+str(i),(.43+i*.012,.245+i*.009,.115+i*.006)) for i in range(6)]
        self.build()

    def material(self,name,color):
        m=self.Sh.Material.Define(self.stage,'/World/Materials/'+name)
        s=self.Sh.Shader.Define(self.stage,str(m.GetPath())+'/Shader')
        s.CreateIdAttr('UsdPreviewSurface')
        s.CreateInput('diffuseColor',self.S.ValueTypeNames.Color3f).Set(color)
        s.CreateInput('roughness',self.S.ValueTypeNames.Float).Set(.65)
        m.CreateSurfaceOutput().ConnectToSource(s.ConnectableAPI(),'surface')
        pm=self.P.MaterialAPI.Apply(m.GetPrim())
        pm.CreateStaticFrictionAttr(self.cfg['physics']['friction'])
        pm.CreateDynamicFrictionAttr(self.cfg['physics']['friction'])
        pm.CreateRestitutionAttr(0.)
        return m

    def bind(self,prim,material):
        self.Sh.MaterialBindingAPI.Apply(prim).Bind(material)
        self.Sh.MaterialBindingAPI(prim).Bind(material,materialPurpose='physics')

    def collision(self,prim):
        self.P.CollisionAPI.Apply(prim)
        p=self.X.PhysxCollisionAPI.Apply(prim)
        p.CreateContactOffsetAttr(self.cfg['physics']['contact_offset'])
        p.CreateRestOffsetAttr(0.)

    def box(self,path,pos,size,material,collide=True):
        cube=self.G.Cube.Define(self.stage,path); cube.CreateSizeAttr(1.)
        cube.AddTranslateOp().Set(self.F.Vec3d(*pos)); cube.AddScaleOp().Set(self.F.Vec3f(*size))
        self.bind(cube.GetPrim(),material)
        if collide:self.collision(cube.GetPrim())
        return cube

    def pocket(self,path,xy,width,depth,material,round_slot=False):
        x,y=xy; z=self.cfg['table_height']
        self.box(path+'/Floor',(x,y,z+.002),(width+.008,width+.008,.004),material)
        if round_slot:
            # Separate wall segments leave a real open pocket; no bounding hull fills it.
            radius=width/2+.002
            for i in range(20):
                a=i*math.tau/20
                c=self.box(path+f'/Wall{i}',(x+radius*math.cos(a),y+radius*math.sin(a),z+.004+depth/2),(.004,2*radius*math.tan(math.pi/20)+.001,depth),material)
                rotation=c.AddRotateZOp(); rotation.Set(math.degrees(a))
                ops=c.GetOrderedXformOps(); c.SetXformOpOrder([ops[0],rotation,ops[1]])
        else:
            for i,sign in enumerate((-1,1)):
                self.box(path+f'/X{i}',(x+sign*(width/2+.002),y,z+.004+depth/2),(.004,width+.008,depth),material)
                self.box(path+f'/Y{i}',(x,y+sign*(width/2+.002),z+.004+depth/2),(width,.004,depth),material)
        return z+.004

    def mesh(self,path,kind,pos,dynamic=False):
        root=self.G.Xform.Define(self.stage,path)
        root.AddTranslateOp().Set(self.F.Vec3d(*pos))
        mesh=self.G.Mesh.Define(self.stage,path+'/Visual')
        pts=vertices(kind)
        mesh.CreatePointsAttr(pts)
        mesh.CreateFaceVertexCountsAttr([3]*(len(pts)//3))
        mesh.CreateFaceVertexIndicesAttr(list(range(len(pts))))
        mesh.CreateSubdivisionSchemeAttr('none')
        self.bind(mesh.GetPrim(),self.materials[kind])
        if kind=='ball':
            sphere=self.G.Sphere.Define(self.stage,path+'/Collider')
            sphere.CreateRadiusAttr(.015); sphere.AddTranslateOp().Set(self.F.Vec3d(0,0,.015))
            sphere.CreateVisibilityAttr('invisible'); self.collision(sphere.GetPrim())
        else:
            self.collision(mesh.GetPrim())
            self.P.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('convexDecomposition' if dynamic else 'none')
            if dynamic:
                decomp=self.X.PhysxConvexDecompositionCollisionAPI.Apply(mesh.GetPrim())
                decomp.CreateMaxConvexHullsAttr(64)
                decomp.CreateHullVertexLimitAttr(64)
                decomp.CreateVoxelResolutionAttr(500000)
        if dynamic:
            self.P.RigidBodyAPI.Apply(root.GetPrim())
            self.P.MassAPI.Apply(root.GetPrim()).CreateMassAttr(self.cfg['physics']['mass_kg'][kind])
            rb=self.X.PhysxRigidBodyAPI.Apply(root.GetPrim())
            rb.CreateSolverPositionIterationCountAttr(32)
            rb.CreateSolverVelocityIterationCountAttr(8)
            rb.CreateEnableCCDAttr(True)
            self.bodies[path]=root
            self.initial[path]=list(pos)+[1.,0.,0.,0.]
        return root

    def build(self):
        z=self.cfg['table_height']
        for i,t in enumerate(self.cfg['tables']):
            x,y=t['center']; w,d=t['size']; p=f'/World/Workcell/Table{i+1}'
            # Solid collision top plus narrow coloured grain strips (visual only).
            self.box(p+'/Top',(x,y,z-.02),(w,d,.04),self.wood[2])
            for j in range(70):
                self.box(p+f'/Grain{j}',(x,y-d/2+(j+.5)*d/70,z+.00005),(w,d/70*.9,.0001),self.wood[(j*17)%6],False)
            for sx in (-1,1):
                for sy in (-1,1):
                    self.box(p+f'/Leg{sx+1}{sy+1}',(x+sx*(w/2-.04),y+sy*(d/2-.04),(z-.04)/2),(.045,.045,z-.04),self.wood[0])
        for t in self.cfg['trays']:
            kind=t['part']; x,y=t['center']; span=t['pitch']+t['slot']+.012
            path='/World/Workcell/Source_'+kind
            self.box(path+'/Base',(x,y,z+.001),(span,span,.002),self.materials['tray'])
            for i,xy in enumerate(slot_centers(t)):
                base=self.pocket(path+f'/Slot{i+1}',xy,t['slot'],t['depth'],self.materials['tray'],kind=='ball')
                self.mesh(f'/World/Parts/{kind}_{i+1}',kind,(*xy,base+.001),True)
        prep=self.cfg['preparation']; x,y=prep['center']
        self.box('/World/Workcell/Preparation/Base',(x,y,z+.001),(.074,3*prep['pitch'],.002),self.materials['prep'])
        for i,kind in enumerate(prep['parts']):
            self.pocket('/World/Workcell/Preparation/'+kind,(x,y+(i-1)*prep['pitch']),.036 if kind=='ball' else .058,.012 if kind=='ball' else .004,self.materials['prep'],kind=='ball')
        self.mesh('/World/Workcell/Jig','jig',(*self.cfg['jig'],z))
        self.pocket('/World/Workcell/Finished',self.cfg['finished'],.060,.012,self.materials['finished'])

    def initialize(self):
        from isaacsim.core.prims import RigidPrim
        self.view=RigidPrim(prim_paths_expr=list(self.bodies),name='workcell_parts',reset_xform_properties=False)
        self.view.initialize()

    def snapshot(self):
        # PhysX/Fabric changes do not necessarily update the USD authored transforms.
        positions,orientations=self.view.get_world_poses()
        return {path:positions[i].tolist()+orientations[i].tolist() for i,path in enumerate(self.view.prim_paths)}
