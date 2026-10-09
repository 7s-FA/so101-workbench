"""Four-arm layout preview without a conveyor using the supplied September 14 product meshes.

This module authors geometry only. It does not claim that a motion plan or
compliant grasp has passed verification.
"""
import json
import math
import struct
from pathlib import Path
from arm_sim.common import ROOT
from arm_sim.factory import FactoryCell, table_legs
from arm_sim.conveyor_hardware import D435_HORIZONTAL_ROLL, BASE_FOOT
from arm_sim.overhead_camera import ARM_BASE_RISE

def cargo_jig_pose(c,col,row):
    """Jig placement relative to the deck centre, including tray orientation."""
    yaw=c.get('jig_yaw',0.);x=(col-1)*c['cargo_pitch'];y=(row-1)*c['cargo_pitch']
    return (-.032+math.cos(yaw)*x-math.sin(yaw)*y,math.sin(yaw)*x+math.cos(yaw)*y,c['deck_top']),yaw


def preview_layout():
    cfg = json.loads((ROOT/'config/factory.json').read_text())
    cfg.update(table_height=.23, show_labels=False)
    cfg['map'].update(size=[3.,2.], floor_color=[.055,.060,.065])
    # Retain the legacy 3x2m silhouette: supply left, assembly right, racks between.
    front=.400;left=.460;right=1.360;table_x=(left+right)/2
    supply_left=-1.310;supply_x=supply_left-BASE_FOOT[2]
    b_x=left+.060;d_x=right-.060
    unload=[b_x-.01125,front-.170]
    loading=[supply_x,front-.170]
    shipping=[d_x-.110,front-.070]
    forklift=[shipping[0],shipping[1]-.320,math.pi/2]
    cfg['arms'] = {
        'arm_a': {'model':'so101_xlerobot','position':[supply_x,front+.200],'yaw':-math.pi/2,'base_plate_thickness':0.,'base_origin_offset':.002402},
        'arm_b': {'model':'omx','position':[b_x,front+.075],'yaw':0.,'base_plate_thickness':0.,'base_origin_offset':0.},
        'arm_c': {'model':'so101_xlerobot','position':[table_x,front+.400+BASE_FOOT[0]-.0013],'yaw':-math.pi/2,'base_plate_thickness':0.,'base_origin_offset':.002402,'mount_height':ARM_BASE_RISE},
        'arm_d': {'model':'omx','position':[d_x,front+.075],'yaw':math.pi,'base_plate_thickness':0.,'base_origin_offset':0.},
    }
    so_home=[0.,-.8,.4,1.3,D435_HORIZONTAL_ROLL,.65]
    omx_home=[0.,-.5,-1.,1.5,0.,.65,-.65]
    cfg['rest_joints']={name:list(so_home if pose['model']=='so101_xlerobot' else omx_home) for name,pose in cfg['arms'].items()}
    cfg['tables'] = [
        {'id':'supply','center':[supply_left+.200,front+.200],'size':[.400,.400]},
        {'id':'assembly','center':[table_x,front+.200],'size':[.900,.400],'leg_inset':[.050,.09]},
    ]
    cfg['unused_worktops']=[{'size':[.400,.400],'count':1},{'size':[.600,.400],'count':1}]
    cfg['burger'].update(deck_top=.21,deck_size=[.236,.236],cargo_columns=3,cargo_pitch=.075,
                         poses=[[*loading,-math.pi/2],[-.800,.200,-math.pi/2]])
    cfg['waffle'].update(pose=forklift,fork_top=.010,mast_start=.013,mast_height=.19,fork_max_top=.15,handling_clearance=.015)
    cfg['conveyor']={'enabled':False}
    cfg['warehouses']=[{'center':[x,.790],'size':[.32,.28],'levels':[.035,.13],'post_height':.20} for x in (-.390,-.010)]
    cfg['shipping_dock']=shipping
    cfg['unload_dock']=unload
    cfg['loading_dock']=loading
    cfg['shipping_height']=.13
    cfg['assembly_jig']=[table_x+.015,front+.140]
    cfg['handoff']=[table_x+.190,front+.090]
    cfg['preparation_slots']=[{'kind':kind,'kit':kit,'position':[b_x+.060+col*.075,y]} for kit,y in enumerate((front+.250,)) for col,kind in enumerate(('housing','ball','cap'))]
    cfg['initial_materials']={'source_loaded_banks':[0],'burger_loaded_indices':[0],'parts_per_kind_per_batch':3,'total_parts':18}
    cfg['source_banks']=[[supply_left+.240,front+.110]]
    cfg['source_table']='supply'
    cfg['assembly_camera']={'attached_arm':'arm_c','count':1}
    cfg['arm_fixtures']={'arm_a':{'table':'supply','side':'left','clamp_offsets':[-.028,-.006],
                                  'clamp_pad_depth':.01546243},
                         'arm_b':{'table':'assembly','side':'left','corner':'front_left',
                                  'clamp_side':'left','clamp_offsets':[-.065,.065],
                                  'clamp_pad_depth':.018},
                         'arm_c':{'table':'assembly','side':'back',
                                  'edge_inset':.0013,'clamp_offsets':[-.037,.037],'clamp_pad_depth':.00766485,
                                  'clamp_pad_size':.008},
                         'arm_d':{'table':'assembly','side':'right','corner':'front_right',
                                  'clamp_side':'right','clamp_offsets':[-.065,.065],
                                  'clamp_pad_depth':.018}}
    cfg['process_reference']={'layout_source':'arm_sim/factory_demo.py:demo_layout (2026-09-14 final)',
        'reference_render':'outputs/final_demo_reference/top.png',
        'source':'arm_sim/factory_demo.py:unload_kit, assemble_kit, load_product, process_batch',
        'preparation':'one three-part kit; B parks empty before C takes the shared kit',
        'assembly':'C assembles all three parts, hands off, then parks',
        'output':'D transfers directly from handoff to independently supported shipping container',
        'exclusive_pairs':[['arm_b','arm_c'],['arm_c','arm_d']],
        'parallel_pairs':[['arm_a','arm_b'],['arm_a','arm_c'],['arm_a','arm_d'],['arm_b','arm_d']],
        'runtime_implemented':False}
    cfg['traffic']={'lane_width':.42,'joined_edges':True,'obstacles':[],
        'waiting_bays':[{'center':[-.800,.232],'size':[.270,.270]}],
        'shared_zones':[{'center':[.060,-.420],'size':[2.720,.440]}],
        'yield_points':[[loading[0],.05],[unload[0],.05]],
        'routes':[
            {'name':'MaterialU','points':[loading,[loading[0],-.420],[unload[0],-.420],unload]},
            {'name':'ForkliftU','points':[forklift[:2],[shipping[0],-.420],[-.010,-.420]],'edges':False,'color':[.06,.42,.43]},
            *[{'name':f'Warehouse{i}','points':([[-.010,-.420]] if x!=-.010 else [])+[[x,-.420],[x,.470]],'edges':False,'color':[.06,.42,.43]} for i,x in enumerate((-.390,-.010))],
        ],
        'mode':'visual_planned_routes_only',
        'forklift_turns':[{'center':[shipping[0],-.420],'yaw_range':[math.pi/2,math.pi]}, *[{'center':[x,-.420],'yaw_range':[math.pi,math.pi/2]} for x in (-.390,-.010)]],
        'shared_lane_policy':'reserve the whole lower corridor for one vehicle; release only after exit; controller pending',
    }
    cfg['docking_panels']=[]
    for ident,x,y,code in [('supply',loading[0],front+.003,10),('unload',unload[0],front+.003,11)]:
        cfg['docking_panels'].append({'id':ident,'center':[x,y,.09],'size':[.16,.008,.18],
                                      'marker_z':.12,'marker_size':.06,'marker_id':code})
    return cfg


class ConveyorCell(FactoryCell):
    def asset_mesh(self,path,filename,pos,material,*,scale=.001,rotation=None):
        data=Path(filename).read_bytes();count=struct.unpack_from('<I',data,80)[0]
        if len(data)!=84+count*50:raise ValueError('Binary STL required: '+str(filename))
        points=[]
        for i in range(count):
            v=struct.unpack_from('<12f',data,84+i*50)[3:]
            points.extend(tuple(scale*x for x in v[j:j+3]) for j in (0,3,6))
        root=self.G.Xform.Define(self.stage,path);root.AddTranslateOp().Set(self.F.Vec3d(*pos))
        if rotation is not None:root.AddRotateXYZOp().Set(self.F.Vec3f(*rotation))
        mesh=self.G.Mesh.Define(self.stage,path+'/Visual')
        mesh.CreatePointsAttr(points);mesh.CreateFaceVertexCountsAttr([3]*count)
        mesh.CreateFaceVertexIndicesAttr(list(range(3*count)));mesh.CreateSubdivisionSchemeAttr('none')
        self.bind(mesh.GetPrim(),material)
        return root

    def jig_asset(self,path,kind,pos):
        self.asset_mesh(path,self.parts/self.manifest[kind+'_jig'],pos,self.materials['prep'])

    def product_asset(self,path,kind,pos):
        file=ROOT/'assets/product/02_ball_d30.stl' if kind=='ball' else self.parts/self.manifest[kind]
        return self.asset_mesh(path,file,pos,self.materials[kind])

    def shipping_container(self,path,x,y,z):
        spec=self.cfg.get('finished_container',{'size':[.236,.095,.004],'pitch':.078,'jig':'pallet'})
        size=spec['size'];thickness=size[2]
        self.box(path+'/Carrier',(x,y,z+thickness/2),size,self.materials['finished'],False)
        for i in range(3):
            self.asset_mesh(path+f'/Slot{i}',self.parts/self.manifest[spec['jig']],(x+(i-1)*spec['pitch'],y,z+thickness),self.materials['finished'])

    def build(self):
        self.parts=ROOT/'assets/generated/conveyor_demo/parts'
        revision=json.loads((ROOT/'config/conveyor_demo.json').read_text())
        self.manifest=revision['parts'];seating=revision['jig_seating']
        for filename in self.manifest.values():
            if not (self.parts/filename).is_file():raise FileNotFoundError(self.parts/filename)
        z=self.cfg['table_height']
        for name,color in {'orange':(.91,.47,.10),'teal':(.03,.48,.51),'white':(.72,.75,.77)}.items():
            self.materials[name]=self.material(name,color)
        for table in self.cfg['tables']:
            x,y=table['center'];w,d=table['size'];path='/World/Factory/Tables/'+table['id']
            self.box(path+'/Top',(x,y,z-.015),(w,d,.03),self.wood[2])
            for i in range(24):
                self.box(path+f'/Grain{i}',(x,y-d/2+(i+.5)*d/24,z+.00006),(w,d/24*.8,.0001),self.wood[i%6],False)
            for i,(center,size) in enumerate(table_legs(table,z)):
                self.box(path+f'/Leg{i}',center,size,self.wood[0])
        # One 3x3 source bank: the latest jig is 68 mm, with 7 mm between jigs.
        for bank,(cx,cy) in enumerate(self.cfg['source_banks']):
            if bank not in self.cfg['initial_materials']['source_loaded_banks']:continue
            for col,kind in enumerate(('housing','ball','cap')):
                for row in range(3):
                    x,y=cx+(col-1)*self.cfg.get('source_pitch',.075),cy+(row-1)*self.cfg.get('source_pitch',.075)
                    path=f'/World/Source/Bank{bank}/{kind}{row}'
                    self.jig_asset(path+'/Jig',kind,(x,y,z))
                    self.product_asset(path+'/Part',kind,(x+seating['offset_xy'][0],y+seating['offset_xy'][1],z+seating['contact_height'][kind]+seating['render_gap']))
        # One fixed three-part kit is shared by B and C with exclusive access.
        for slot in self.cfg['preparation_slots']:
            self.jig_asset(f"/World/Preparation/Kit{slot['kit']}/{slot['kind']}",slot['kind'],(*slot['position'],z))
        self.jig_asset('/World/Handoff/Jig','housing',(*self.cfg['handoff'],z))
        self.jig_asset('/World/Assembly/Jig','housing',(*self.cfg['assembly_jig'],z))
        from arm_sim.conveyor_hardware import shipping_boxes
        for name,pos,size in shipping_boxes(self.cfg):
            self.box('/World/ShippingDock/'+name,pos,size,self.wood[2])
        # Empty shipping stand; the forklift retrieves a container after arrival.
        for j,rack in enumerate(self.cfg['warehouses']):
            x,y=rack['center'];w,d=rack['size'];p=f'/World/Warehouse{j}'
            if self.cfg.get('trident_supports'):
                from arm_sim.conveyor_hardware import warehouse_boxes
                for leaf,pos,size in warehouse_boxes(rack):self.box(p+'/'+leaf,pos,size,self.wood[2])
                self.shipping_container(p+'/EmptyContainer',x,y,rack['levels'][-1])
                continue
            for i,(sx,sy) in enumerate(((-1,-1),(-1,1),(1,-1),(1,1))):
                self.box(p+f'/Post{i}',(x+sx*(w/2-.012),y+sy*(d/2-.012),rack['post_height']/2),(.018,.018,rack['post_height']),self.materials['metal'])
            for level,h in enumerate(rack['levels']):
                for k,(lo,hi) in enumerate(((-w/2,-.073),(-.037,.037),(.073,w/2))):
                    self.box(p+f'/Shelf{level}_{k}',(x+(lo+hi)/2,y,h-.008),(hi-lo,d,.016),self.materials['white'])
            self.shipping_container(p+'/EmptyContainer',x,y,rack['levels'][1])
        self.build_traffic()
        # Parking markings only; this static preview does not reserve traffic.
        for i,bay in enumerate(self.cfg['traffic'].get('waiting_bays',[])):
            x,y=bay['center'];w,d=bay['size']
            corners=[(x-w/2,y-d/2),(x+w/2,y-d/2),(x+w/2,y+d/2),(x-w/2,y+d/2)]
            for j,(a,b) in enumerate(zip(corners,corners[1:]+corners[:1])):
                self.line(f'/World/Traffic/WaitingBay{i}/Edge{j}',a,b,self.materials['teal'],.008)
        for i,zone in enumerate(self.cfg['traffic'].get('shared_zones',[])):
            x,y=zone['center'];w,d=zone['size']
            for j,(a,b) in enumerate([((x-w/2,y-d/2),(x+w/2,y-d/2)),((x+w/2,y-d/2),(x+w/2,y+d/2)),((x+w/2,y+d/2),(x-w/2,y+d/2)),((x-w/2,y+d/2),(x-w/2,y-d/2))]):
                self.line(f'/World/Traffic/Shared{i}/Edge{j}',a,b,self.materials['orange'],.009)
        # Marker 14 is not in the old marker set; omit rather than invent code.
        markers=json.loads((ROOT/'assets/factory/aruco/markers.json').read_text())['markers']
        self.cfg['docking_panels']=[p for p in self.cfg['docking_panels'] if str(p['marker_id']) in markers]
        self.build_docking_panels()

    def attach_burger_deck(self,root):
        c=self.cfg['burger'];top=c['deck_top'];w,d=c['deck_size']
        for i,(sx,sy) in enumerate(((-1,-1),(-1,1),(1,-1),(1,1))):
            self.box(root+f'/Cargo/Post{i}',(-.032+sx*.055,sy*.06,.176),(.008,.008,.052),self.materials['metal'])
        self.box(root+'/Cargo/Deck',(-.032,0,top-.004),(w,d,.008),self.materials['white'])
        for col,kind in enumerate(('housing','ball','cap')):
            for row in range(3):
                (px,py,_),yaw=cargo_jig_pose(c,col,row)
                self.asset_mesh(root+f'/Cargo/{kind}{row}',self.parts/self.manifest[kind+'_jig'],(px,py,top),self.materials['prep'],rotation=(0,0,math.degrees(yaw)))
                vehicle_index=int(root.rsplit('_',1)[1])-1
                if vehicle_index in self.cfg['initial_materials']['burger_loaded_indices']:
                    seating=json.loads((ROOT/'config/conveyor_demo.json').read_text())['jig_seating']
                    self.product_asset(root+f'/Cargo/Parts/{kind}{row}',kind,(px+seating['offset_xy'][0],py+seating['offset_xy'][1],top+seating['contact_height'][kind]+seating['render_gap']))

    def attach_forklift(self,root):
        from arm_sim.conveyor_hardware import forklift_boxes
        fixed,moving=forklift_boxes(self.cfg)
        for group,shapes in [('Frame',fixed),('Lift',moving)]:
            for name,pos,size in shapes:
                color=self.materials['orange'] if group=='Lift' else self.materials['metal']
                self.box(root+'/'+group+'/'+name,pos,size,color)

    def attach_shipping_pallet(self,root):
        # Forklift is empty while C fills the independently supported container.
        pass

    def build_assembly_camera(self):
        from arm_sim.overhead_camera import assembly,support_boxes
        meshes,eye,direction=assembly(self.cfg)
        root='/World/Overhead/Assembly'
        for name,file,scale,tf in meshes:
            material=(self.material('camera_module',(.035,.055,.042)) if name=='Module' else
                      self.material('camera_base_yellow',(.55,.65,.025)) if name in ('ArmBase','Bottom') else
                      self.material('camera_stand_black',(.035,.040,.045)))
            shape=self.asset_mesh(root+'/'+name,file,(0,0,0),material,scale=scale)
            shape.ClearXformOpOrder();shape.AddTransformOp().Set(self.F.Matrix4d(tf.T.tolist()))
        for name,pos,size in support_boxes(self.cfg):
            self.box(root+'/Support/'+name,pos,size,self.materials['metal'])
        return eye,direction
