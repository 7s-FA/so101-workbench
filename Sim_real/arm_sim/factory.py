"""Metric factory layout. Independent of the original teaching workcell."""
import json
import math
from pathlib import Path
from arm_sim.common import ROOT
from arm_sim.workcell import Workcell


def load_layout(path=None):
    cfg = json.loads(Path(path or ROOT/'config/factory.json').read_text())
    if cfg['source']['count'] != 3 or cfg['preparation']['count'] != 3 or cfg['finished']['count'] != 3:
        raise ValueError('This scene represents exactly three complete product sets')
    for key in ('source', 'preparation', 'finished'):
        if cfg[key]['pitch'] < .09:
            raise ValueError(f'{key}: 90 mm minimum provisional finger clearance')
    check_lidar_clearance(cfg)
    if abs(cfg['table_height']-cfg['burger']['deck_top']) > .05:
        raise ValueError('Table/deck height difference exceeds this layout preview allowance')
    return cfg


def check_lidar_clearance(cfg):
    """Do not mistake vertical plate clearance for unobstructed 360 degree scans."""
    c=cfg['lidar_clearance'];b=cfg['burger'];w=cfg['waffle']
    deck_gap=b['deck_top']-.008-c['burger_body_top_z']
    fork_gap=w['fork_top']-.018-c['waffle_scan_z']
    if deck_gap+1e-9<c['deck_body_margin']:
        raise ValueError('Cargo deck is too close to the stock LDS body')
    fork_below_gap=c['waffle_scan_z']-w['fork_top']
    if max(fork_gap,fork_below_gap)+1e-9<c['fork_beam_margin']:
        raise ValueError('Fork underside intrudes into the stock scan-plane clearance')
    if b['deck_top']-.008<=b['post_start']:
        raise ValueError('Cargo supports must have positive length')
    if w['mast_start']+w['mast_height']<max(cfg['warehouse']['levels'])+.055:
        raise ValueError('Mast is too short for the upper shelf and carriage')
    return {'burger_deck_to_lidar_body_m':deck_gap,'waffle_fork_to_scan_plane_m':max(fork_gap,fork_below_gap),
            'fork_scan_side':'below' if fork_below_gap>fork_gap else 'above',
            'stock_lidar_preserved':True,'support_occlusion':'partial occlusion accepted; no scan performance claim'}


def rows(center_y, count, pitch):
    return [center_y+(i-(count-1)/2)*pitch for i in range(count)]


def arm_base_height(pose,table_height):
    rise=float(pose.get('mount_height',0.))
    if not math.isfinite(rise) or rise<0:raise ValueError('Invalid arm mount height')
    plate=float(pose.get('base_plate_thickness',.004))
    if not math.isfinite(plate) or plate<0:raise ValueError('Invalid base plate thickness')
    offset=float(pose.get('base_origin_offset',0.))
    if not math.isfinite(offset):raise ValueError('Invalid base origin offset')
    return table_height+plate+rise+offset


def table_legs(table,height):
    """Shared rendered/audited leg boxes, with room for edge-mounted clamps."""
    x,y=table['center'];w,d=table['size']
    ix,iy=table.get('leg_inset',[.025,.025])
    return [([x+sx*(w/2-ix),y+sy*(d/2-iy),(height-.03)/2],
             [.025,.025,height-.03]) for sx,sy in ((-1,-1),(-1,1),(1,-1),(1,1))]


def warehouse_world_size(cfg):
    rack=cfg['warehouse'];yaw=rack.get('yaw',0.);quarter=round(yaw/(math.pi/2))
    if not math.isclose(yaw,quarter*math.pi/2,abs_tol=1e-8):raise ValueError('Warehouse uses quarter-turn mounting orientations')
    w,d=rack['size']
    return (d,w) if quarter%2 else (w,d)


def warehouse_shelf_sections(cfg,index):
    """Two through channels let the forks withdraw under a deposited pallet."""
    rack=cfg['warehouse'];x,y=rack['center'];w,d=warehouse_world_size(cfg);z=rack['levels'][index]-.008
    if not rack.get('fork_slots') or index not in rack['fork_slots'].get('levels',[1]):
        return [([x,y,z],[w,d,.016])]
    half=rack['fork_slots']['width']/2;spacing=rack['fork_slots']['center_spacing']/2
    extent=w if rack.get('entry_side')=='bottom' else d
    intervals=[(-extent/2,-spacing-half),(-spacing+half,spacing-half),(spacing+half,extent/2)]
    if rack.get('entry_side')=='bottom':return [([x+(a+b)/2,y,z],[b-a,d,.016]) for a,b in intervals]
    return [([x,y+(a+b)/2,z],[w,b-a,.016]) for a,b in intervals]


class FactoryCell(Workcell):
    def nest(self, path, xy, kind, z=None, material=None):
        """Low cradle, with open left/right finger access; no surrounding tall walls."""
        z = self.cfg['table_height'] if z is None else z
        mat = material or self.materials['tray']
        x,y = xy
        width = .04 if kind == 'ball' else .062
        self.box(path+'/Base', (x,y,z+.002), (width,width,.004), mat)
        if kind == 'ball':
            # Four low corner pads constrain the ball without blocking side access.
            for i,(dx,dy) in enumerate(((-1,-1),(-1,1),(1,-1),(1,1))):
                self.box(path+f'/Stop{i}', (x+dx*.014,y+dy*.014,z+.006), (.008,.008,.008), mat)
        else:
            for i,sign in enumerate((-1,1)):
                self.box(path+f'/Stop{i}', (x,y+sign*.029,z+.005), (.038,.004,.006), mat)
        return z+.004

    def container(self, path, xy, kinds, pitch, z, material, filled=False, vertical=True, start_index=0, carrier_width=.080):
        self.G.Xform.Define(self.stage,path)
        x,y = xy
        size = (carrier_width, (len(kinds)-1)*pitch+carrier_width)
        if not vertical: size = size[::-1]
        self.box(path+'/Carrier', (x,y,z+.001), (*size,.002), material)
        for i,kind in enumerate(kinds):
            offset = (i-(len(kinds)-1)/2)*pitch
            p = (x,y+offset) if vertical else (x+offset,y)
            base = self.nest(path+f'/Slot{i+1}',p,kind,z,material)
            if filled:
                self.mesh(f'/World/Parts/{kind}_{i+1+start_index}',kind,(*p,base+.001),True)
        return size

    def label(self, path, text, xy, width, depth=.055, z=None):
        """Texture-backed text authored as geometry in the USD, not a screenshot overlay."""
        if not self.cfg.get('show_labels',True):return
        from PIL import Image, ImageDraw, ImageFont
        from pxr import UsdGeom, UsdShade, Sdf, Gf
        folder=ROOT/'assets/generated/factory_labels'; folder.mkdir(parents=True,exist_ok=True)
        filename=folder/(path.rsplit('/',1)[-1]+'.png')
        image=Image.new('RGB',(768,128),(234,239,240)); draw=ImageDraw.Draw(image)
        font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',38)
        draw.text((384,64),text,font=font,fill=(24,43,55),anchor='mm')
        image.save(filename)
        mesh=UsdGeom.Mesh.Define(self.stage,path)
        x,y=xy; h=self.cfg['table_height']+.010 if z is None else z
        mesh.CreatePointsAttr([(x-width/2,y-depth/2,h),(x+width/2,y-depth/2,h),(x+width/2,y+depth/2,h),(x-width/2,y+depth/2,h)])
        mesh.CreateFaceVertexCountsAttr([4]);mesh.CreateFaceVertexIndicesAttr([0,1,2,3])
        mesh.CreateSubdivisionSchemeAttr('none')
        uv=UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,UsdGeom.Tokens.vertex)
        uv.Set([(0,0),(1,0),(1,1),(0,1)])
        m=UsdShade.Material.Define(self.stage,path+'_Material')
        shader=UsdShade.Shader.Define(self.stage,path+'_Material/Surface');shader.CreateIdAttr('UsdPreviewSurface')
        tex=UsdShade.Shader.Define(self.stage,path+'_Material/Texture');tex.CreateIdAttr('UsdUVTexture')
        tex.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(str(filename))
        tex.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set('sRGB')
        st=UsdShade.Shader.Define(self.stage,path+'_Material/UV');st.CreateIdAttr('UsdPrimvarReader_float2')
        st.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st')
        tex.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(),'result')
        shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(),'rgb')
        shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.85)
        m.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(m)

    def line(self,path,start,end,color,width=.009,z=.002):
        x,y=start; a,b=end; length=math.hypot(a-x,b-y)
        obj=self.box(path,((x+a)/2,(y+b)/2,z),(length,width,.002),color,False)
        op=obj.AddRotateZOp();op.Set(math.degrees(math.atan2(b-y,a-x)))
        ops=obj.GetOrderedXformOps();obj.SetXformOpOrder([ops[0],op,ops[1]])

    def build(self):
        z=self.cfg['table_height']
        self.materials['orange']=self.material('factory_orange',(.95,.37,.06))
        self.materials['teal']=self.material('factory_teal',(.02,.55,.55))
        self.materials['white']=self.material('factory_white',(.85,.87,.88))
        for table in self.cfg['tables']:
            x,y=table['center'];w,d=table['size'];p='/World/Factory/Tables/'+table['id']
            self.box(p+'/Top',(x,y,z-.015),(w,d,.03),self.wood[2])
            for i in range(24):
                self.box(p+f'/Grain{i}',(x,y-d/2+(i+.5)*d/24,z+.00005),(w,d/24*.85,.0001),self.wood[i%6],False)
            for i,(center,size) in enumerate(table_legs(table,z)):
                self.box(p+f'/Leg{i}',center,size,self.wood[0])
            self.label('/World/Labels/Table_'+table['id'],table['id'].upper(),(x,y+d/2+.055),min(w,.40),z=.003)
        for area in ('source','preparation'):
            batches=self.cfg.get('source_batches',[self.cfg[area]]) if area=='source' else [self.cfg[area]]
            for batch_index,c in enumerate(batches):
                for kind,x in c['columns'].items():
                    self.container('/World/Factory/'+area+f'/Batch{batch_index}/'+kind,(x,c['center_y']),[kind]*c['count'],c['pitch'],z,self.materials['tray' if area=='source' else 'prep'],area=='source',start_index=c.get('start_index',0),carrier_width=c.get('carrier_width',.080))
        self.mesh('/World/Factory/AssemblyJig','jig',(*self.cfg['jig'],z+.002))
        self.nest('/World/Factory/Handoff',self.cfg['handoff'],'housing')
        c=self.cfg['finished']
        if c.get('enabled',True):
            self.container('/World/Factory/Finished',c['center'],['housing']*3,c['pitch'],z,self.materials['finished'],False,False)
            self.label('/World/Labels/Output','OUTPUT: 3 UNITS',(c['center'][0],.205+self.cfg.get('factory_y_offset',0)),.32,.035)
        # An empty shared dock, not a third material robot.
        x,y=self.cfg['unload_dock']
        for i,(a,b) in enumerate((((x-.08,y-.15),(x+.08,y-.15)),((x-.08,y+.15),(x+.08,y+.15)),((x-.08,y-.15),(x-.08,y+.15)),((x+.08,y-.15),(x+.08,y+.15)))):
            self.line('/World/Factory/Dock/Line'+str(i),a,b,self.materials['teal'])
        self.label('/World/Labels/Dock','SHARED UNLOAD',(x,y-.20),.28,z=.003)
        self.label('/World/Labels/BurgerQueue','2 BURGERS: LOAD / WAIT',(-1.04,-.72+self.cfg.get('factory_y_offset',0)),.50,z=.003)
        # Two shelf levels, each holding one 3-product container; initially empty.
        w=self.cfg['warehouse'];x,y=w['center'];sx,sy=warehouse_world_size(self.cfg)
        post_height=w.get('post_height',max(w['levels'])+.12)
        for i,(dx,dy) in enumerate(((-1,-1),(-1,1),(1,-1),(1,1))):
            self.box('/World/Factory/Warehouse/Post'+str(i),(x+dx*(sx/2-.012),y+dy*(sy/2-.012),post_height/2),(.018,.018,post_height),self.materials['metal'])
        for i,level in enumerate(w['levels']):
            for j,(center,size) in enumerate(warehouse_shelf_sections(self.cfg,i)):
                self.box(f'/World/Factory/Warehouse/Shelf{i}_{j}',center,size,self.materials['white'])
            vertical=bool(round(w.get('yaw',0.)/(math.pi/2))%2)
            self.container('/World/Factory/Warehouse/Tray'+str(i),(x,y),['housing']*3,.09,level,self.materials['finished'],False,vertical)
        self.label('/World/Labels/Warehouse','WAREHOUSE A / B',(x,min(y+.24,self.cfg['map']['size'][1]/2-.035)),.38,z=.003)
        if w.get('entry_side')=='left':
            a=x-sx/2-.18;b=x-sx/2-.035
            self.line('/World/Factory/Warehouse/EntryArrow',(a,y),(b,y),self.materials['teal'])
            for i,side in enumerate((-1,1)):self.line(f'/World/Factory/Warehouse/EntryArrow{i}',(b-.035,y+side*.025),(b,y),self.materials['teal'])

        if w.get('entry_side')=='bottom':
            a=y-sy/2-.18;b=y-sy/2-.035
            self.line('/World/Factory/Warehouse/EntryArrow',(x,a),(x,b),self.materials['teal'])
            for i,side in enumerate((-1,1)):self.line(f'/World/Factory/Warehouse/EntryArrow{i}',(x+side*.025,b-.035),(x,b),self.materials['teal'])

        self.build_docking_panels()
        self.build_traffic()

    def build_docking_panels(self):
        panels=self.cfg.get('docking_panels',[])
        if not panels:return
        markers=json.loads((ROOT/'assets/factory/aruco/markers.json').read_text())['markers']
        black=self.material('aruco_black',(.002,.002,.002))
        white=self.material('aruco_white',(.95,.95,.95))
        for panel in panels:
            root='/World/Factory/DockingPanels/'+panel['id']
            self.box(root+'/Wall',panel['center'],panel['size'],self.materials['white'])
            x,y,_=panel['center'];z=panel['marker_z'];size=panel['marker_size']
            # All four docking approaches look north (Burgers use their rear
            # camera); put the code on the wall's south face, below the deck.
            face=y-panel['size'][1]/2
            self.box(root+'/QuietZone',(x,face-.0003,z),(size*4/3,.0006,size*4/3),white,False)
            bits=markers[str(panel['marker_id'])];step=size/len(bits)
            for row,line in enumerate(bits):
                for col,value in enumerate(line):
                    if not value:
                        self.box(root+f'/Code_{row}_{col}',
                                 (x+(col-(len(bits)-1)/2)*step,face-.0007,z+((len(bits)-1)/2-row)*step),
                                 (step,.0002,step),black,False)

    def build_traffic(self):
        """Visual route graph and real pallet islands, not a navigation controller."""
        cfg=self.cfg['traffic']; half=cfg['lane_width']/2
        paint=self.material('lane_paint',(.62,.64,.64))
        yellow=self.materials['orange']
        # Paint stays just inside the configured map boundary.
        hx,hy=[v/2-.01 for v in self.cfg['map']['size']]
        for i,(a,b) in enumerate((((-hx,-hy),(hx,-hy)),((hx,-hy),(hx,hy)),((hx,hy),(-hx,hy)),((-hx,hy),(-hx,-hy)))):
            self.line('/World/Traffic/Boundary'+str(i),a,b,paint,.008)
        for route in cfg['routes']:
            center_paint=self.material('route_'+route['name'],tuple(route['color'])) if 'color' in route else yellow
            joined=route.get('edges',True) and cfg.get('joined_edges',False)
            if joined:
                points=route['points'];normals=[]
                for a,b in zip(points,points[1:]):
                    dx,dy=b[0]-a[0],b[1]-a[1];length=math.hypot(dx,dy)
                    normals.append((-dy/length,dx/length))
                for side in (-1,1):
                    border=[]
                    for j,point in enumerate(points):
                        n0=normals[max(0,j-1)];n1=normals[min(j,len(normals)-1)]
                        denom=1+n0[0]*n1[0]+n0[1]*n1[1]
                        if denom<1e-6:raise ValueError('Lane border cannot make a 180 degree reversal')
                        border.append(tuple(point[k]+side*half*(n0[k]+n1[k])/denom for k in (0,1)))
                    for j,(a,b) in enumerate(zip(border,border[1:])):
                        self.line(f"/World/Traffic/{route['name']}/Border{side+1}_{j}",a,b,paint,.005)
            for i,(a,b) in enumerate(zip(route['points'],route['points'][1:])):
                dx,dy=b[0]-a[0],b[1]-a[1];length=math.hypot(dx,dy)
                nx,ny=-dy/length,dx/length
                p='/World/Traffic/'+route['name']+f'/Segment{i}'
                for side in ((-1,1) if route.get('edges',True) and not joined else ()):
                    self.line(p+f'/Edge{side+1}',(a[0]+nx*half*side,a[1]+ny*half*side),(b[0]+nx*half*side,b[1]+ny*half*side),paint,.005)
                # Dashed centre lines keep the black floor visually dominant.
                for j in range(int(length/.12)):
                    lo=(j*.12+.025)/length;hi=min((j*.12+.070)/length,.99)
                    self.line(p+f'/Dash{j}',(a[0]+dx*lo,a[1]+dy*lo),(a[0]+dx*hi,a[1]+dy*hi),center_paint,.006)
        for i,o in enumerate(cfg['obstacles']):
            x,y=o['center'];w,d,h=o['size'];p='/World/Traffic/PalletIsland'+str(i)
            self.box(p+'/Pallet',(x,y,.015),(w+.03,d+.03,.03),self.wood[1])
            for j,side in enumerate((-1,1)):
                self.box(p+f'/Crate{j}',(x+side*w/4,y,.03+h/2),(w/2-.01,d,h),self.materials['teal'])
        for i,(x,y) in enumerate(cfg.get('yield_points', ((-.95,.0),(.10,-.05)) if cfg['routes'][0]['name']=='MaterialU' else ((-1.12,-.31),(.035,-.31),(-.40,-.31),(0,-1.02)))):
            self.line('/World/Traffic/Yield'+str(i),(x-.13,y),(x+.13,y),yellow,.02)
        self.label('/World/Labels/MapSize',f"{self.cfg['map']['size'][0]:g} m x {self.cfg['map']['size'][1]:g} m",(-.90,hy-.08),.50,.055,z=.003)
        self.label('/World/Labels/RearRoute',('WORKCELL' if cfg['routes'][0]['name']=='MaterialU' else 'REAR ACCESS'),(.15,hy-.07),.48,.05,z=.003)
        self.label('/World/Labels/Loop',('U-SHAPED SHUTTLE' if cfg['routes'][0]['name']=='MaterialU' else 'ONE-WAY LOOP / YIELD'),(.1,-hy+.07),.70,.05,z=.003)

    def attach_burger_deck(self, root):
        c=self.cfg['burger'];top=c['deck_top'];w,d=c['deck_size'];metal=self.materials['metal']
        for i,(sx,sy) in enumerate(((-1,-1),(-1,1),(1,-1),(1,1))):
            # Anchored within the original platform footprint. These posts cross
            # the scan plane; plate clearance alone does not remove that occlusion.
            hx,hy=c['post_half_span'];pw=c['post_width']
            self.box(root+f'/Cargo/Post{i}',(-.032+sx*hx,sy*hy,(top-.008+c['post_start'])/2),(pw,pw,top-.008-c['post_start']),metal)
        self.box(root+'/Cargo/Deck',(-.032,0,top-.004),(w,d,.008),self.materials['white'])
        for i in range(c.get('cargo_columns',1)):
            kind=('housing','ball','cap')[i] if c.get('cargo_columns',1)==3 else 'housing'
            x=-.032+(i-(c.get('cargo_columns',1)-1)/2)*c.get('cargo_pitch',.09)
            self.container(root+'/Cargo/Tray'+str(i),(x,0),[kind]*3,c.get('cargo_pitch',.09),top,self.materials['tray'],carrier_width=c.get('cargo_carrier_width',.08))

    def attach_forklift(self,root):
        c=self.cfg['waffle'];z=c['fork_top'];h=c['mast_height'];metal=self.materials['metal']
        for i,sy in enumerate((-1,1)):
            self.box(root+f'/Lift/Mast{i}',(.125,sy*.055,c['mast_start']+h/2),(.018,.018,h),metal)
            self.box(root+f'/Lift/Fork{i}',(.15+c['fork_length']/2,sy*.055,z-.009),(c['fork_length'],.025,.018),self.materials['orange'])
        self.box(root+'/Lift/Carriage',(.14,0,z+.02),(.018,.15,.07),self.materials['orange'])
        self.attach_shipping_pallet(root+'/Lift')

    def attach_shipping_pallet(self,root):
        c=self.cfg['waffle'];z=c['fork_top']
        self.box(root+'/Pallet',(.15+c['fork_length']/2,0,z+.004),(.18,.29,.008),self.wood[1])
        self.container(root+'/Tray',(.15+c['fork_length']/2,0),['housing']*3,.09,z+.008,self.materials['finished'])

    def report_layout(self):
        return {'part_count':len(self.initial),'material_types':3,'parts_per_type':self.cfg['source']['count'],'finished_capacity':3,
                'burger_capacity':self.cfg['burger'].get('cargo_columns',1)*3,
                'slot_pitch_m':self.cfg['source']['pitch'],'gap_between_50mm_parts_m':self.cfg['source']['pitch']-.05,
                'table_height_m':self.cfg['table_height'],'burger_deck_top_m':self.cfg['burger']['deck_top'],
                'arms':list(self.cfg['arms']),'mobile_robots':['burger_1','burger_2','waffle_forklift'],
                'map_size_m':self.cfg['map']['size'],'traffic_lane_width_m':self.cfg['traffic']['lane_width'],
                'lidar_clearance':check_lidar_clearance(self.cfg),
                'fork_top_m':self.cfg['waffle']['fork_top'],
                'mast_top_m':self.cfg['waffle']['mast_start']+self.cfg['waffle']['mast_height'],
                'scope':'held-arm, parked-vehicle physical layout preview','assumptions':self.cfg['assumptions']}
