"""Display measured physics poses even after live USD edits invalidate Fabric.

Only visual geometry is copied. Physics APIs are removed from these copies;
the original collision bodies and their tensor states remain untouched.
"""
from pxr import Sdf, Usd, UsdGeom, Gf, UsdPhysics


class RenderPoseBridge:
    def __init__(self,env):
        self.env=env;self.bindings=[]
        stage=env.sim.stage;source_layer=stage.Flatten();root_layer=stage.GetRootLayer()
        UsdGeom.Xform.Define(stage,'/World/MeasuredDisplay')
        links={p.GetName():p for p in stage.Traverse()
               if str(p.GetPath()).startswith('/World/SO101/') and p.HasAPI(UsdPhysics.RigidBodyAPI)}
        sources=[]
        for index,name in enumerate(env.robot.body_names):
            source=links[name].GetChild('visuals')
            if source.IsValid():sources.append(('robot_'+name,source,'robot',index))
        for name in env.parts:
            sources.append(('part_'+name,stage.GetPrimAtPath('/World/Parts/'+name+'/Visual'),'part',name))
        for name,source,kind,index in sources:
            path='/World/MeasuredDisplay/'+name
            root=UsdGeom.Xform.Define(stage,path)
            dest=path+'/visuals'
            if not Sdf.CopySpec(source_layer,source.GetPath(),root_layer,Sdf.Path(dest)):
                raise RuntimeError('물리 자세 표시 메시 복사 실패: '+str(source.GetPath()))
            copied=stage.GetPrimAtPath(dest)
            # Keep material binding APIs, strip all physics APIs from the display.
            for prim in Usd.PrimRange(copied):
                apis=[a for a in prim.GetAppliedSchemas() if not a.startswith(('Physics','Physx'))]
                prim.SetMetadata('apiSchemas',Sdf.TokenListOp.CreateExplicit(apis))
                if prim.HasAPI(UsdPhysics.CollisionAPI) or prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    raise RuntimeError('표시 메시에는 물리 충돌을 만들 수 없습니다.')
            UsdGeom.Imageable(copied).MakeVisible()
            UsdGeom.Imageable(source).MakeInvisible()
            self.bindings.append((root.AddTranslateOp(),root.AddOrientOp(),kind,index))
        self.update()

    def update(self):
        env=self.env
        positions=env.robot.data.body_link_pos_w[0].cpu().numpy()
        orientations=env.robot.data.body_link_quat_w[0].cpu().numpy()
        for translate,orient,kind,index in self.bindings:
            if kind=='robot':position=positions[index];quaternion=orientations[index]
            else:
                body=env.parts[index]
                position=body.data.root_pos_w[0].cpu().numpy()
                quaternion=body.data.root_quat_w[0].cpu().numpy()
            translate.Set(Gf.Vec3d(*map(float,position)))
            orient.Set(Gf.Quatf(float(quaternion[0]),Gf.Vec3f(*map(float,quaternion[1:]))))
