"""Read-only presentation of the actual arm, vehicle, lift and tray schedule."""
import math
import numpy as np

NAMES={'arm_a':'로봇팔 A','arm_b':'로봇팔 B','arm_c':'로봇팔 C','arm_d':'로봇팔 D',
       'burger_1':'운반로봇 1','burger_2':'운반로봇 2','waffle_1':'지게차'}
KINDS={'housing':'하우징','ball':'볼','cap':'덮개'}

class Workflow:
    def __init__(self,plan):
        self.plan=plan;self.steps=[];self.actions=[]
        for task in plan.tasks:
            self.steps.append(dict(id=task['id'],device=task['arm'],label=task['label'].split(' · ',1)[-1],start=task['start'],end=task['end'],ack='arm'))
        a_start=min(x['start'] for x in plan.phases if x['arm']=='arm_a')
        a_end=next(x['time'] for x in plan.park_events if x['arm']=='arm_a')
        self.steps.append(dict(id='a_load',device='arm_a',label='다음 배치 재료 9개 상차 · 안전 복귀',start=a_start,end=a_end,ack='pose'))
        for job in plan.traffic_jobs:
            self.steps.append(dict(id=job['id'],device=job['vehicle'],label=job['label'].split(' · ',1)[-1],start=job['start'],end=job['end'],ack='traffic'))
        for index,event in enumerate(plan.vehicle_moves):
            tr=plan.vehicles[event['vehicle']];a=tr.at(event['start']);b=tr.at(event['end'])
            if event['mode']=='turn':label=('왼쪽' if b[2]>a[2] else '오른쪽')+' 제자리 회전 · 이동 정지'
            else:
                backward=np.dot(b[:2]-a[:2],[math.cos(a[2]),math.sin(a[2])])<0
                direction='후진' if backward else '전진'
                if abs(b[0]-a[0])>1e-6:
                    target=('작업대 쪽' if b[0]>a[0] else '창고 쪽') if event['vehicle']=='waffle_1' else ('오른쪽 통로' if b[0]>a[0] else '왼쪽 통로')
                    label=f'{direction} · {target} 이동'
                elif b[1]<a[1]:label=f'{direction} · 통로까지 이탈'
                else:label=f'{direction} · {self.location(b[:2],event["vehicle"])} 진입'
            self.actions.append(dict(id=f'move_{index}',device=event['vehicle'],label=label,start=event['start'],end=event['end'],kind=event['mode']))
        for index in range(len(plan.lift.times)-1):
            a,b=plan.lift.values[index][0],plan.lift.values[index+1][0]
            if abs(b-a)<1e-9:continue
            top=(plan.cfg['waffle']['fork_top']+b)*1000
            self.actions.append(dict(id=f'lift_{index}',device='waffle_1',label=f'포크 {"올리기" if b>a else "내리기"} · 높이 {top:.0f}mm',start=plan.lift.times[index],end=plan.lift.times[index+1],kind='lift'))
        for index,history in enumerate(plan.trays):
            for sequence,(t,owner,pose) in enumerate(history[1:],1):
                where=self.location(plan.vehicles['waffle_1'].at(t)[:2],'waffle_1')
                label=f'용기 {index+1} '+('회수' if owner=='lift' else '내려놓기')+f' · {where}'
                self.actions.append(dict(id=f'tray_{index}_{sequence}',device='waffle_1',label=label,start=t,end=t,kind='tray'))
        for index,phase in enumerate(plan.phases):
            part=phase['parts'][0];kind,kit=part.rsplit('_',1)
            item=f'제품 {kit}' if len(phase['parts'])==3 else f'세트 {kit} · {KINDS[kind]}'
            label=item+(' 집기' if phase['action']=='pick' else ' 놓기')
            self.actions.append(dict(id=f'phase_{index}',device=phase['arm'],label=label,start=phase['start'],end=phase['end'],kind='arm'))
        # Initial stationary fork preparation happens outside a traffic reservation.
        for action in self.actions:
            if action['kind']=='lift' and not any(s['device']=='waffle_1' and s['start']<=action['start']<s['end'] for s in self.steps):
                self.steps.append(dict(id='prepare_'+action['id'],device='waffle_1',label='빈 용기 회수 높이 준비 · 정지 상태',start=action['start'],end=action['end'],ack='time'))
        self.steps.sort(key=lambda s:(s['start'],s['end'],s['id']))
        for step in self.steps:
            step['children']=sorted([a for a in self.actions if a['device']==step['device'] and step['start']-1e-7<=a['start'] and a['end']<=step['end']+1e-7],key=lambda a:(a['start'],a['end']))

    def location(self,xy,device):
        c=self.plan.cfg
        if device=='waffle_1' and np.linalg.norm(np.array(xy)-c['forklift_parking_pose'][:2])<.006:return '지게차 전용 대기 위치'
        if device in c['burger_parking_poses'] and np.linalg.norm(np.array(xy)-c['burger_parking_poses'][device][:2])<.006:
            return '차량 대기 구역' if device=='burger_1' else '재료 상차·대기 위치'
        choices=[('재료 상차 위치',c['loading_dock']),('재료 하차 위치',c['unload_dock'])] if device.startswith('burger') else [('작업대 용기 받침',c['shipping_vehicle_pose'][:2])]+[(f'창고 {i+1}',self.plan.rack_dock(i)) for i in range(2)]
        for label,p in choices:
            if np.linalg.norm(np.array(xy)-p)<.006:return label
        if device.startswith('burger') and np.linalg.norm(np.array(xy)-c['burger']['poses'][1][:2])<.006:return '차량 대기 구역'
        return '가로 통로' if abs(xy[1]-c['vehicle_lane_y'])<.006 else '도킹 진입로'

    def snapshot(self,t,state,*,started,paused,done,arm_acks,traffic_acks,fault=None):
        def status(row,ack=None):
            if not started:return 'pending'
            if t>=row['end']-1e-7:
                return 'complete' if ack is None or ack else 'checking'
            if t<row['start']-1e-7:return 'pending'
            return 'fault' if fault else 'paused' if paused else 'active'
        rows=[]
        for step in self.steps:
            ack=(step['id'] in arm_acks) if step['ack']=='arm' else (step['id'] in traffic_acks) if step['ack']=='traffic' else None
            if step['ack']=='pose':ack=state['coordination']['arms']['arm_a']['safe_pose']
            children=[{**a,'status':status(a)} for a in step['children']]
            rows.append({**step,'name':NAMES[step['device']],'status':status(step,ack),'children':children})
        trays=[]
        for i,history in enumerate(self.plan.trays):
            _,owner,_=self.plan.entry(history,t);pose=self.plan.tray_pose(i,t);count=0
            for kit in range(i*3+1,i*3+4):
                n=f'housing_{kit}';part=state['parts'][n];part_owner=state['owners'][n]
                if owner=='lift':count+=int(part_owner=='lift')
                elif part_owner=='world' and np.linalg.norm(part[:2]-pose[:2])<.11 and abs(part[2]-pose[2])<.025:count+=1
            where='지게차' if owner=='lift' else '작업대 용기 받침' if np.linalg.norm(pose[:2]-self.plan.cfg['shipping_dock'])<.006 else f'창고 {i+1}'
            trays.append({'id':i,'location':where,'products':count,'owner':owner})
        vehicles=[]
        for name,tr in self.plan.vehicles.items():
            pose=state['vehicles'][name];where=self.location(pose[:2],name)
            actions=[a for a in self.actions if a['device']==name and a['start']<=t<a['end']]
            active=[s for s in rows if s['device']==name and s['status'] in ('active','paused','checking','fault')]
            future=next((s for s in rows if s['device']==name and s['start']>t),None)
            moving=next((a for a in actions if a['kind']!='lift'),None)
            phase=' · '.join(a['label'] for a in actions)
            if not phase:
                if not started:phase='시작 대기'
                elif done:phase='작업 완료 · 정지'
                elif name.startswith('burger'):
                    cargo=state['cargo'][name]
                    phase=('재료 9개 적재 완료 · 출발 대기' if cargo==9 else '상차 진행·대기') if where=='재료 상차 위치' else ('하차 진행·대기' if cargo else '하차 완료 · 정지') if where=='재료 하차 위치' else '다음 작업 대기'
                else:phase='완제품 적재·회수 대기' if any(x['location']=='작업대 용기 받침' for x in trays) else '다음 작업 대기'
            if started and paused and not done:phase='일시정지 · '+phase
            cargo=f'재료 {state["cargo"][name]} / 9개' if name.startswith('burger') else '용기 없음'
            if name=='waffle_1':
                carried=next((x for x in trays if x['owner']=='lift'),None)
                if carried:cargo=f'용기 {carried["id"]+1} · '+(f'완제품 {carried["products"]} / 3개' if carried['products'] else '빈 용기')
            heading='회전 중' if moving and moving['kind']=='turn' else ['오른쪽','위쪽','왼쪽','아래쪽'][round(pose[2]/(math.pi/2))%4]
            vehicles.append({'id':name,'name':NAMES[name],'phase':phase,'location':where,'heading':heading,'cargo':cargo,'fork_height_mm':round((self.plan.cfg['waffle']['fork_top']+state['lift'])*1000) if name=='waffle_1' else None,'job':active[0]['label'] if active else None,'next':future['label'] if future else None,'active':bool(actions) and started and not paused})
        return {'steps':rows,'vehicles':vehicles,'trays':trays,'complete':sum(s['status']=='complete' for s in rows),'total':len(rows)}
