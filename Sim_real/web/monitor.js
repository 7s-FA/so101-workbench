'use strict';
const $=id=>document.getElementById(id);
const names=['베이스','어깨','팔꿈치','손목 굽힘','손목 회전','그리퍼'];
let state=null,paused=false,device='so101',selected=1,lastReceived=0,omxHistory=[],lastOmxSample=null;
const fmt=(v,n=1)=>typeof v==='number'&&Number.isFinite(v)?v.toFixed(n):'—';
const unusedFields={overview:{select:'overviewSelect',image:'overviewImage',empty:'overviewEmpty',status:'overviewState'},wrist:{select:'wristSelect',image:'wristImage',empty:'wristEmpty',status:'wristState'}};
const fields={};
const urls={},frameTimes={};
for(let i=1;i<=6;i++){
 const row=document.createElement('tr');row.id='motor'+i;
 const first=document.createElement('td'),button=document.createElement('button');button.textContent=names[i-1];button.ariaLabel=names[i-1]+' 전압 기록 보기';
 button.onclick=()=>{selected=i;renderTelemetry()};first.append(button);row.append(first);
 for(const key of ['voltage','load','temperature']){const td=document.createElement('td');td.id=key+i;td.textContent='—';row.append(td)}
 $('motorRows').append(row);
}
function setStatus(id,text,kind=''){const el=$(id);el.textContent=text;el.className='status '+kind}
function options(id,cameras,preferred){
 if(!cameras.length)return;
 const el=$(id),signature=cameras.map(c=>c.id+':'+c.label).join('|');if(el.dataset.signature===signature)return;
 const old=el.value;el.replaceChildren();
 for(const c of cameras){const op=document.createElement('option');op.value=c.id;op.textContent=c.label;el.append(op)}
 el.value=cameras.some(c=>c.id===old)?old:cameras.some(c=>c.id===preferred)?preferred:cameras[0].id;
 el.dataset.signature=signature;
}
function clearFrame(key,text){const f=fields[key];$(f.image).hidden=true;$(f.empty).hidden=false;$(f.status).textContent=text;$(f.status).className='video-state';frameTimes[key]=0}
function renderTelemetry(){
 const so=state?.so101||{},omx=state?.omx||{};
 $('soPanel').hidden=device!=='so101';$('omxPanel').hidden=device!=='omx';
 const source=device==='so101'?so:omx;
 $('sourceLabel').textContent=source.scope||(device==='so101'?'SO-101 리더 1대 · 모터 6개':'OMX 리더 · 그리퍼 ID 6');
 const fault=device==='so101'?(so.motors||[]).some(m=>!m.stale&&(m.error||(m.lower!=null&&m.voltage<m.lower)||(m.upper!=null&&m.voltage>m.upper))):omx.voltage_blocked;
 $('deviceMessage').textContent=source.connected?(fault?'측정 오류 또는 전압 범위 이탈':device==='so101'?`수집 중 · ${fmt(so.rate)} Hz`:'그리퍼 상태 수집 중'):(source.message||'수집 프로그램 연결 대기');
 $('deviceMessage').className='device-message '+(source.connected?(fault?'bad':'good'):'');
 const events=[];
 for(let i=1;i<=6;i++){
  const m=(so.motors||[]).find(m=>m.id===i);
  const bad=m&&!m.stale&&(m.error||(m.lower!=null&&m.voltage<m.lower)||(m.upper!=null&&m.voltage>m.upper));
  $('motor'+i).className=(selected===i?'selected ':'')+(bad?'bad':'');
  $('motor'+i).querySelector('button').setAttribute('aria-pressed',selected===i?'true':'false');
  $('voltage'+i).textContent=fmt(m?.voltage);$('load'+i).textContent=fmt(m?.load_raw,0);$('temperature'+i).textContent=fmt(m?.temperature,0);
 }
 const motor=(so.motors||[]).find(m=>m.id===selected);
 $('selectedMotor').textContent=names[selected-1];
 $('selectedFault').textContent=!motor||motor.stale?'응답 없음':motor.error?'오류 0x'+motor.error.toString(16):(motor.lower!=null&&motor.voltage<motor.lower)||(motor.upper!=null&&motor.voltage>motor.upper)?'전압 범위 이탈':'현재 응답 정상';
 $('minimum').textContent=fmt(motor?.minimum)+' V';$('commErrors').textContent=fmt(motor?.comm_errors,0);
 $('omxVoltage').textContent=fmt(omx.voltage);$('omxCurrent').textContent=fmt(omx.motor?.present_current_mA,0);
 $('omxTemp').textContent=fmt(omx.temperature,0);$('omxGoal').textContent=fmt(omx.motor?.goal_current_mA,0);
 const feedback=omx.connected?(omx.voltage_blocked?'저전압 · 출력 해제':!omx.force_enabled?'출력 비활성':omx.motor?.torque_enable?'저항 출력 중':'출력 대기'):'측정 대기';$('feedback').textContent=feedback;
 $('chartTitle').textContent=device==='so101'?names[selected-1]+' 전압':'그리퍼 전류';
 if(device==='so101'){
  if(so.teleop)events.push(so.teleop.tracking?'가상 팔 추종 중':so.teleop.reason||'가상 팔 유지');
  for(const e of (so.events||[]).slice(0,3))events.push(`${Math.floor(e.t/60)}분 ${Math.floor(e.t%60)}초 · ${names[e.id-1]||'모터'} · ${fmt(e.voltage)} V · 오류 0x${Number(e.error).toString(16)}`);
  if(so.connected&&!events.length)events.push('현재 수집된 오류 이력이 없습니다.');
 }else if(omx.connected)events.push(feedback,'통신 데이터 '+fmt(omx.age)+'초 전 갱신');
 if(!events.length)events.push('실물 데이터 연결 대기 · 이전 수치를 현재값으로 표시하지 않습니다.');
 $('events').replaceChildren();for(const text of events){const li=document.createElement('li');li.textContent=text;$('events').append(li)}
 drawChart();
}
function drawChart(){
 const canvas=$('trend'),ctx=canvas.getContext('2d'),w=canvas.clientWidth,h=canvas.clientHeight,ratio=devicePixelRatio||1;
 canvas.width=Math.round(w*ratio);canvas.height=Math.round(h*ratio);ctx.setTransform(ratio,0,0,ratio,0,0);
 const so=state?.so101||{},m=(so.motors||[]).find(m=>m.id===selected),end=device==='so101'?(so.elapsed||0):Date.now()/1000;
 const points=(device==='so101'?(m?.history||[]):omxHistory).filter(p=>p[0]>=end-60),values=points.flatMap(p=>p.slice(1)).filter(v=>typeof v==='number'&&Number.isFinite(v));
 $('chartEmpty').hidden=!!values.length;
 const lo=device==='so101'?Math.min(3.5,...values):Math.min(0,...values),hi=device==='so101'?Math.max(5.5,...values):Math.max(20,...values);
 const L=36,R=w-7,T=8,B=h-22,x=t=>L+(t-(end-60))/60*(R-L),y=v=>B-(v-lo)/(hi-lo||1)*(B-T);
 ctx.font='11px system-ui';ctx.lineWidth=1;ctx.strokeStyle='#354150';ctx.fillStyle='#a9b6c5';
 for(let i=0;i<3;i++){const v=lo+(hi-lo)*i/2;ctx.beginPath();ctx.moveTo(L,y(v));ctx.lineTo(R,y(v));ctx.stroke();ctx.fillText(v.toFixed(device==='so101'?1:0),0,y(v)+4)}
 ctx.fillText('−60초',L,h-3);ctx.fillText('현재',R-24,h-3);
 ctx.strokeStyle='#9abfff';ctx.lineWidth=1.8;let prev=null;
 for(const p of points){if(p[1]==null){prev=null;continue}ctx.beginPath();if(prev&&p[0]-prev[0]<2)ctx.moveTo(x(prev[0]),y(prev[1]));else ctx.moveTo(x(p[0]),y(p[1]));ctx.lineTo(x(p[0]),y(p[1]));ctx.lineTo(x(p[0]),y(p[2]??p[1]));ctx.stroke();prev=p}
}
function render(){
 renderTelemetry();
 setStatus('simStatus',state?.simulation_available?'시뮬레이션 연결됨':'시뮬레이션 연결 대기',state?.simulation_available?'good':'');
 setStatus('soStatus',state?.so101.connected?'SO-101 수집 중':'SO-101 연결 대기',state?.so101.connected?'good':'');
 setStatus('omxStatus',state?.omx.connected?'OMX 수집 중':'OMX 연결 대기',state?.omx.connected?'good':'');
 $('clock').textContent='현재 시각 '+new Date().toLocaleTimeString('ko-KR',{hour12:false});
}
async function poll(){
 try{
  const r=await fetch('/api/state',{cache:'no-store',signal:AbortSignal.timeout(2200)});if(!r.ok)throw Error('state');
  const next=await r.json();lastReceived=performance.now();
  if(next.omx.connected&&next.omx.samples!==lastOmxSample){lastOmxSample=next.omx.samples;omxHistory.push([Date.now()/1000,next.omx.motor?.present_current_mA??null]);omxHistory=omxHistory.filter(p=>p[0]>Date.now()/1000-60)}
  if(!next.omx.connected&&omxHistory.length)omxHistory.push([Date.now()/1000,null]);
  if(!paused){state=next;render();$('notice').hidden=true}
 }catch{
  state=null;render();for(const key of Object.keys(fields))clearFrame(key,'모니터 연결 끊김');
  $('notice').hidden=false;$('notice').textContent='통합 모니터 연결이 끊겼습니다. 실행 터미널을 확인하세요.';
 }
 setTimeout(poll,500);
}
async function frameLoop(key){
 const f=fields[key],id=$(f.select).value,cam=state?.cameras.find(c=>c.id===id);
 if(!paused&&!document.hidden&&cam?.live&&performance.now()-lastReceived<2500){
  try{
   const r=await fetch('/frame/'+encodeURIComponent(id),{cache:'no-store',signal:AbortSignal.timeout(1800)});if(!r.ok)throw Error('frame');const blob=await r.blob();
   if(!paused&&$(f.select).value===id){
    const next=URL.createObjectURL(blob),img=$(f.image);if(urls[key])URL.revokeObjectURL(urls[key]);urls[key]=next;img.src=next;img.alt=(cam.label||id)+' 영상';img.hidden=false;$(f.empty).hidden=true;
    frameTimes[key]=performance.now();$(f.status).textContent='실시간 · 시뮬레이션';$(f.status).className='video-state live';
   }
  }catch{if(!paused)clearFrame(key,'영상 연결 대기')}
 }
 setTimeout(()=>frameLoop(key),120);
}
for(const [key,f] of Object.entries(fields))$(f.select).onchange=()=>{clearFrame(key,'영상 전환 중');if(!paused)renderCameras()};
$('pause').onclick=()=>{paused=!paused;$('pause').setAttribute('aria-pressed',String(paused));$('pause').textContent=paused?'화면 갱신 재개':'화면 일시정지';$('notice').hidden=!paused;$('notice').textContent='화면 일시정지 중 · 시뮬레이션과 실물 동작은 계속됩니다.';if(paused){for(const f of Object.values(fields)){$(f.status).textContent='화면 일시정지';$(f.status).className='video-state paused'}for(const id of ['simStatus','soStatus','omxStatus'])setStatus(id,'표시 일시정지')}else{state=null;render()}};
for(const [id,key] of [['tabSo','so101'],['tabOmx','omx']])$(id).onclick=()=>{device=key;$('tabSo').setAttribute('aria-pressed',key==='so101');$('tabOmx').setAttribute('aria-pressed',key==='omx');renderTelemetry()};
window.addEventListener('resize',drawChart);render();poll();
