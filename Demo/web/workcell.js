'use strict';
const $=id=>document.getElementById(id),keys=['overview','a','b','c','d'];
let state=null,received=0,pending=false;const urls={},generations={},frames={};
const labels={arm_a:'재료 상차',arm_b:'하차·분류',arm_c:'제품 조립',arm_d:'완제품 출하'};
const human=text=>String(text||'').replaceAll('housing','하우징').replaceAll('ball','볼').replaceAll('cap','덮개');
function clearFrame(k,message){$(k+'Image').hidden=true;$(k+'Empty').hidden=false;$(k+'Empty').textContent=message;$(k+'State').textContent=message;$(k+'State').className='video-state';}
function render(){
 const live=state?.simulation_available,demo=live?state?.simulation?.demo:null;
 $('connection').textContent=live?'연결됨':'연결 대기';$('connection').className='status'+(live?' good':'');
 $('notice').hidden=!!live&&!demo?.fault&&!demo?.hold_reason;
 $('notice').textContent=demo?.fault?`안전 검사로 공정 정지: ${demo.fault}`:demo?.hold_reason||'시뮬레이션 연결이 없습니다. 실행 터미널을 확인하세요.';
 $('clock').textContent='현재 시각 '+new Date().toLocaleTimeString('ko-KR',{hour12:false});
 $('time').textContent=demo?`실제 경과 ${Math.floor(demo.active_elapsed_s||0)}초 · 공정 기준 ${Math.floor(demo.time||0)} / ${Math.ceil(demo.nominal_duration_s||0)}초 · ${demo.speed}배속`:'';
 $('progress').value=demo?.progress||0;$('progressText').textContent=demo?(demo.done?'완료':demo.started===false?'시작 대기':demo.paused?'일시정지':`${Math.round(demo.progress*100)}%`):'대기';
 $('counts').textContent=`조립 완료 ${demo?.assembled??'—'} / ${demo?.goal??6} · 창고 보관 ${demo?.stored??'—'} / ${demo?.goal??6}`;
 $('cargo').textContent=`운반로봇 1 ${demo?.cargo?.burger_1??'—'} / 9 · 운반로봇 2 ${demo?.cargo?.burger_2??'—'} / 9`;
 $('demoPause').disabled=!demo||demo.done||!!demo.fault||pending;$('demoRestart').disabled=!demo||pending;
 $('demoSpeed').disabled=!demo||pending;
 if(demo&&!pending&&document.activeElement!==$('demoSpeed'))$('demoSpeed').value=String(demo.speed);
 $('speedStatus').textContent=demo?`실제 시연 속도 ${(demo.effective_speed||0).toFixed(2)}배`:'실제 시연 속도 —';
 $('demoPause').textContent=!demo||demo.started===false?'시연 시작':demo.paused?'계속하기':'일시정지';
 for(const [id,mode] of [['startLoad','load'],['startPreloaded','preloaded']]){const button=$(id);button.disabled=!demo?.start_modes?.includes(mode)||pending;button.setAttribute('aria-pressed',String(demo?.start_mode===mode))}
 $('startHint').textContent=demo&&!demo.start_modes?'시작 방식 선택은 시뮬레이션 재실행 후 사용할 수 있습니다.':'시작 방식을 고른 뒤 시연 시작을 눌러 주세요. 방식을 바꾸면 처음으로 돌아갑니다.';
 const c=demo?.coordination,acks=new Set(demo?.completed_tasks||[]);
 $('armStates').replaceChildren();
 for(const [arm,label] of Object.entries(labels)){
  const s=c?.arms?.[arm],li=document.createElement('li'),id=document.createElement('strong'),body=document.createElement('div'),small=document.createElement('small');
  id.textContent=arm.slice(-1).toUpperCase();body.textContent=label;small.textContent=s?human(s.phase):'위치 확인 대기';small.className=s?.safe_pose?'safe':'';body.append(small);li.append(id,body);$('armStates').append(li);
 }
 $('zoneState').textContent=c?(c.zones.central?c.zones.central.slice(-1).toUpperCase()+' 사용 중 · 다른 팔 진입 대기':'사용 가능 · 한 번에 한 팔만 작업'):'작업 구역 확인 대기';
 const geometryStatus=demo?.geometry_enabled===false?'실시간 간섭 검사 꺼짐':`실시간 간섭 검사 ${demo?.geometry_checks??0}회`;
 $('gateState').textContent=c?`안전 위치 복귀 확인 ${acks.size} / ${c.tasks.length} · ${geometryStatus}${demo?.hold_reason?' · '+demo.hold_reason:''}`:'안전 복귀 확인 대기';
 const traffic=demo?.traffic,vehicleNames={burger_1:'운반로봇 1',burger_2:'운반로봇 2',waffle_1:'지게차'};
 $('trafficState').textContent=traffic?`${traffic.owner?vehicleNames[traffic.owner]+' 이동 중':'통로 비어 있음'}${traffic.waiting?.length?' · '+traffic.waiting.map(n=>vehicleNames[n]).join(', ')+' 대기':''}`:'차량 위치 확인 대기';
 $('taskList').replaceChildren();
 for(const task of c?.tasks||[]){const li=document.createElement('li');const active=task.start<=demo.time&&demo.time<task.end,done=acks.has(task.id);li.className=done?'complete':active?'active':'';let reason=done?'완료 · 안전 복귀 확인':active?'진행 중':(task.deps||[]).some(d=>!acks.has(d))?'이전 작업 완료 대기':'작업 구역·로봇 준비 대기';li.textContent=human(task.label)+' — '+reason;$('taskList').append(li)}
 for(const k of keys){const cam=state?.cameras?.find(x=>x.id===$(k+'Select').value);if(!cam?.live||!live)clearFrame(k,'연결 대기');else if(frames[k]&&performance.now()-frames[k]>2500)clearFrame(k,'영상 수신 지연')}
}
async function poll(){try{const r=await fetch('/api/state',{cache:'no-store',signal:AbortSignal.timeout(2200)});if(!r.ok)throw Error();state=await r.json();received=performance.now()}catch{state=null}render();setTimeout(poll,500)}
async function frame(k){const started=performance.now(),fps=Math.min(60,Math.max(1,Number(state?.simulation?.camera_fps)||15)),delay=()=>Math.max(0,1000/fps-(performance.now()-started));const id=$(k+'Select').value,generation=generations[k]||0,cam=state?.cameras?.find(x=>x.id===id);
 if(!document.hidden&&cam?.live&&performance.now()-received<2500){try{const r=await fetch('/frame/'+encodeURIComponent(id),{cache:'no-store',signal:AbortSignal.timeout(1800)});if(!r.ok)throw Error();const blob=await r.blob();if((generations[k]||0)!==generation||$(k+'Select').value!==id)return;
 const url=URL.createObjectURL(blob),img=$(k+'Image');img.src=url;try{await img.decode()}catch{URL.revokeObjectURL(url);throw Error()}if((generations[k]||0)!==generation){URL.revokeObjectURL(url);return}if(urls[k])URL.revokeObjectURL(urls[k]);urls[k]=url;img.hidden=false;img.alt=cam.label+' 영상';$(k+'Empty').hidden=true;$(k+'State').textContent=id.endsWith('_depth')?'깊이 · 시뮬레이션':'실시간 · 시뮬레이션';$(k+'State').className='video-state live';frames[k]=performance.now();
 }catch{clearFrame(k,'영상 연결 대기')}finally{setTimeout(()=>frame(k),delay())}return;}setTimeout(()=>frame(k),250);
}
for(const k of keys){$(k+'Select').onchange=()=>{generations[k]=(generations[k]||0)+1;clearFrame(k,'영상 전환 중')};frame(k)}
async function command(command,extra={}){pending=true;$('commandError').hidden=true;render();try{const r=await fetch('/api/demo/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({command,...extra}),signal:AbortSignal.timeout(2500)});if(!r.ok)throw Error()}catch{$('commandError').textContent='명령 전달 실패. 연결 상태와 실행 터미널을 확인하세요.';$('commandError').hidden=false}finally{pending=false;render()}}
$('startLoad').onclick=()=>command('start_mode',{mode:'load'});
$('startPreloaded').onclick=()=>command('start_mode',{mode:'preloaded'});
$('demoSpeed').onchange=()=>command('speed',{speed:Number($('demoSpeed').value)});
$('demoPause').onclick=()=>command(state?.simulation?.demo?.paused?'resume':'pause');$('demoRestart').onclick=()=>command('restart');
function expand(value){document.querySelector('.overview').classList.toggle('expanded',value);document.body.classList.toggle('video-expanded',value);for(const el of document.querySelectorAll('header,.run-toolbar,.run-help,.process,.cameras,.camera-heading,.detail-row,footer,#notice,#commandError'))el.inert=value;$('fullscreen').textContent=value?'원래 크기':'크게 보기';$('fullscreen').setAttribute('aria-expanded',String(value))}
$('fullscreen').onclick=()=>expand($('fullscreen').getAttribute('aria-expanded')!=='true');window.addEventListener('keydown',e=>{if(e.key==='Escape')expand(false)});render();poll();
