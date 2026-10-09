'use strict';
const $=s=>document.querySelector(s),all=s=>[...document.querySelectorAll(s)];
let state={},episodes=[],tab='record',pending=0,selected=null,confirmAction=null,confirmData={},seekDragging=false;
const time=t=>String(Math.floor((t||0)/60)).padStart(2,'0')+':'+((t||0)%60).toFixed(1).padStart(4,'0');
const number=n=>String(n).padStart(4,'0');
const notices=new Map();
function renderNotices(){
 const text=[...new Set(notices.values())].join('\n');
 const box=$('#error');if(box.textContent!==text)box.textContent=text;box.hidden=!text;
 const stack=$('#notifications');stack.hidden=!text&&$('#recording-alert').hidden;
 const host=document.querySelector('dialog[open]')||$('#notification-home');
 if(stack.parentElement!==host)host.prepend(stack);
}
function error(message,source='request'){
 if(message)notices.set(source,ko(message));else notices.delete(source);
 renderNotices();
}
all('dialog').forEach(dialog=>dialog.addEventListener('close',renderNotices));
function statusNotices(){
 error(state.error,'control');
 const high=state.temperature?.pending||[];
 error(!state.error&&high.length?high.map(v=>`${v.joint||('ID '+v.id)}: ${v.raw??v.temperature_c}${state.temperature?.unit==='raw'?' raw':'°C'} · 온도 높음 ${v.elapsed_s.toFixed(1)}/${v.duration_s}초`).join('\n')+'\n기준 이상의 온도가 5초 연속 유지되면 중단합니다.':null,'temperature');
 error(state.dataset?.state==='error'?state.dataset.message:null,'dataset');
 error(state.model?.error?'모델 불러오기 실패: '+state.model.error:null,'model');
 error(state.sim?.state==='reconnecting'?'자세 화면을 다시 연결하고 있습니다.': ['error','stale'].includes(state.sim?.state)?'시뮬레이션 영상이 중단되었습니다. 연결 상태를 확인해 주세요.':null,'sim');
 for(const name of ['gripper','table']){
  const c=state.cameras?.[name];
  const failed=c?.state==='recovering'||(c?.error&&c.error!=='Not connected');
  error(!state.recording_interruption&&failed?`${name==='gripper'?'그리퍼':'작업대'} 카메라 영상을 받지 못하고 있습니다. 연결을 확인해 주세요.`:null,'camera-'+name);
 }
}
async function api(path,body){const response=await fetch('/api/'+path,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json','X-Studio-Command':'1'},body:JSON.stringify(body)});let data;try{data=await response.json()}catch{data={detail:'프로그램의 응답을 읽지 못했습니다'}}if(!response.ok)throw Error(data.detail||response.statusText);return data;}
async function command(op,data={}){if(pending&&!['stop','release_torque'].includes(op))throw Error('이전 요청을 처리하고 있습니다. 잠시 기다려 주세요.');pending++;error('');try{const r=await api('command/'+op,data);await updateStatus();await refreshEpisodes();await updateStatus();return r}catch(e){error(e.message);throw e}finally{pending--}}
function action(id,fn){$(id).onclick=()=>Promise.resolve().then(fn).catch(e=>error(e.message));}
// Shortcuts use the same enabled buttons and confirmation flow as mouse input.
function shortcutClick(id){const button=$(id);if(!button||button.disabled||button.closest('[hidden]')||(pending&&id!=='#global-stop'))return false;button.click();return true;}
document.addEventListener('keydown',event=>{
 if(event.defaultPrevented||event.repeat||event.isComposing||event.keyCode===229||event.ctrlKey||event.altKey||event.metaKey)return;
 if(event.code==='F8'){
  event.preventDefault();if(shortcutClick('#global-stop')){all('dialog[open]').forEach(dialog=>dialog.close());confirmAction=null;}return;
 }
 if(!$('#keyboard-enabled').checked||document.querySelector('dialog[open]'))return;
 if(event.target.closest('input,select,textarea,.app-select,.select-menu,[contenteditable]:not([contenteditable="false"]),[role="textbox"]'))return;
 const shortcuts={KeyL:'#enable-leader',KeyS:'#save',KeyP:'#preview'};
 let id=shortcuts[event.code];
 if(event.code==='KeyR')id=event.shiftKey?'#discard':state.mode==='recording'?'#stop-record':'#record';
 else if(event.shiftKey)return;
 if(id){event.preventDefault();shortcutClick(id);}
});
function switchTab(name){StudioSelect.close();tab=name;window.scrollTo({top:0,behavior:'instant'});document.body.dataset.tab=name;all('nav [data-tab]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.tab===name)));for(const name2 of ['record','episodes','train','model'])$('#'+name2+'-panel').hidden=name2!==name;$('#heading').textContent={record:'동작 기록',episodes:'기록 확인',train:'모델 학습',model:'모델 실행'}[name];$('#section-label').textContent={record:'01 / RECORD',episodes:'02 / REVIEW',train:'03 / TRAIN',model:'04 / RUN'}[name];if(name==='episodes')refreshEpisodes().catch(e=>error(e.message));renderControls();}
all('nav [data-tab]').forEach(b=>b.onclick=()=>switchTab(b.dataset.tab));
async function refreshEpisodes(){episodes=await api('episodes');const wanted=selected||state.selected;$('#episode').replaceChildren();if(!episodes.length){$('#episode').add(new Option('저장한 기록이 없습니다',''));selected=null;}for(const ep of episodes){const text=`기록 ${number(ep.id)} · ${ep.robot_type==='omx'?'이전 OMX':'SO-101'} · ${time(ep.duration)} · ${ko(ep.result)}`;$('#episode').add(new Option(text,String(ep.id)));}if(episodes.length){selected=episodes.some(e=>e.id===wanted)?wanted:episodes.some(e=>e.id===state.selected)?state.selected:episodes[episodes.length-1].id;$('#episode').value=String(selected);}$('#saved-count').textContent='저장 '+episodes.length+'개';renderEpisode();}
function renderEpisode(){const ep=episodes.find(e=>e.id===selected);if(ep)$('#episode').value=String(ep.id);$('#include').disabled=!ep;$('#include').checked=ep?.included||false;$('#episode-result').value=ep?.result||'Success';const editable=!!ep&&['idle','holding','preview'].includes(state.mode)&&!state.storage_busy;$('#include').disabled=!editable;$('#episode-result').disabled=!editable;$('#delete-episode').disabled=!ep||!['idle','holding','preview'].includes(state.mode)||!!state.storage_busy;$('#episode-task').disabled=!editable||state.dataset?.state==='writing';$('#episode-task').value=ep?.task_id||'';$('#episode-meta').textContent=ep?`${ko(ep.result)} · ${new Date(ep.created_at*1000).toLocaleString('ko-KR')}`:'';$('#replay').textContent=ep?'기록 '+number(ep.id)+' 실물 실행':'실물 재실행';StudioSelect.refresh();}
let selectionWork=Promise.resolve();
function selectEpisode(id){
 pending++;
 selectionWork=selectionWork.catch(()=>{}).then(async()=>{
  try{
   await api('command/select',{id});
   selected=id;await updateStatus();await refreshEpisodes();renderEpisode();
  }catch(e){error(e.message);renderEpisode();throw e}
  finally{pending--;}
 });
 return selectionWork;
}
async function ensureEpisodeSelected(){
 await selectionWork;
 const id=selected||Number($('#episode').value);
 if(id&&state.selected!==id)await selectEpisode(id);
}
$('#episode').onchange=()=>{const id=Number($('#episode').value);if(id)selectEpisode(id).catch(()=>{})};
$('#include').onchange=()=>command('include',{id:selected,included:$('#include').checked}).catch(()=>renderEpisode());
$('#episode-result').onchange=()=>command('change_result',{id:selected,result:$('#episode-result').value}).catch(()=>renderEpisode());
let deleteId=null;
action('#delete-episode',()=>{deleteId=selected;if(!deleteId)return;$('#delete-title').textContent='기록 '+number(deleteId)+' 삭제';$('#delete-confirm').showModal();$('#delete-cancel').focus();});
$('#delete-cancel').onclick=()=>$('#delete-confirm').close();
action('#delete-go',async()=>{const id=deleteId;$('#delete-confirm').close();if(id)await command('delete_episode',{id,confirmed:true});});
action('#preview',async()=>{await ensureEpisodeSelected();await command('preview')});
let queuedSeek=null,seeking=false;
$('#timeline').onpointerdown=()=>{seekDragging=true};
$('#timeline').oninput=()=>{seekDragging=true;$('#play-time').textContent=time(Number($('#timeline').value))};
$('#timeline').onpointercancel=()=>{seekDragging=false};
$('#timeline').onpointerup=()=>setTimeout(()=>{if(!seeking&&!queuedSeek){seekDragging=false}},0);
$('#timeline').onchange=async()=>{
 queuedSeek={t:Number($('#timeline').value),playing:true};
 if(seeking)return;
 seeking=true;pending++;error('');
 try{
  while(queuedSeek){const target=queuedSeek;queuedSeek=null;await ensureEpisodeSelected();await api('command/seek',target);await updateStatus();}
 }catch(e){queuedSeek=null;error(e.message)}
 finally{pending--;seeking=false;seekDragging=false;}
};
function confirm(op,title,description,data={}){confirmAction=op;confirmData=data;$('#confirm-title').textContent=title;$('#confirm-text').textContent=description;$('#clear').checked=false;$('#confirm-go').disabled=true;$('#confirm-go').textContent=op==='teleop'?'리더 따라가기 시작':'실물에서 실행';$('#confirm').showModal();}
$('#clear').onchange=()=>$('#confirm-go').disabled=!$('#clear').checked;$('#confirm-cancel').onclick=()=>$('#confirm').close();
action('#confirm-go',async()=>{const op=confirmAction;$('#confirm').close();await command(op,{...confirmData,workspace_clear:true})});
action('#enable-leader',()=>confirm('teleop','리더 따라가기 시작','작업용 팔이 조작용 팔의 현재 자세로 천천히 이동한 뒤 같은 관절 각도를 따라갑니다. 준비 중에는 리더를 천천히 움직여 주세요.'));
action('#replay',async()=>{await ensureEpisodeSelected();confirm('replay','기록 '+number(state.selected)+' 실물 실행','작업할 물체를 녹화 시작 때의 위치에 놓아 주세요. 팔이 시작 자세로 천천히 이동한 뒤 녹화한 속도로 움직입니다.');});
action('#discard',async()=>{const result=await command('discard');if(result&&!$('#record').disabled)$('#record').focus()});
action('#record',()=>command('record'));action('#stop-record',()=>command('stop_record'));
action('#save',async()=>{const result=await command('save',{result:$('#result').value});if(result&&!$('#record').disabled)$('#record').focus()});
for(const id of ['#global-stop','#replay-stop'])action(id,()=>command('stop'));
action('#build-dataset',()=>command('build_dataset'));
action('#details-button',()=>{const expanded=$('#details').hidden;$('#details').hidden=!expanded;$('#details-button').setAttribute('aria-expanded',String(expanded));$('#details-button span').textContent=expanded?'관절 각도 닫기':'관절 각도 보기';});
let deviceCatalog={ports:[],cameras:[],defaults:{}},releaseOperation='release_torque';
function storedDevices(){try{return JSON.parse(localStorage.getItem('so101-uvc-overhead-d435-devices')||'{}')}catch{return {}}}
function renderDevices(){
 const follower=$('#follower-port').value,leader=$('#leader-port').value;
 const grip=$('#gripper-device').value,table=$('#table-device').value;
 const problem=state.connected?'연결 중에는 장치를 바꿀 수 없습니다. 연결 해제 후 다시 선택하세요.':
  !follower?'작업용 팔을 선택하세요. 조작용 팔은 손으로 시연할 때 필요합니다.':
  follower===leader?'작업용 팔과 조작용 팔을 서로 다른 USB로 선택하세요.':
  leader&&!deviceCatalog.robot?.leader_calibration?'리더 보정 JSON을 선택하고 등록하세요.':
  grip&&grip===table?'그리퍼와 작업대 카메라를 서로 다르게 선택하세요.':
  (!grip||!table)?'자세 확인은 가능합니다. 녹화·실물 재실행에는 카메라 2대가 필요합니다.':'연결하면 자세와 영상을 확인할 수 있습니다.';
 $('#device-selection-help').textContent=problem;
 $('#connect-devices').disabled=!!state.connected||!follower||follower===leader||!!(grip&&grip===table)||!!pending||!!state.test_mode;
 $('#disconnect-devices').disabled=!state.connected||!!pending;
 for(const id of ['#follower-port','#leader-port','#gripper-device','#table-device'])$(id).disabled=!!state.connected;
 $('#device-message').textContent=state.connected?`작업용 팔 연결됨 · 모터 힘 ${StudioControls.describe(state).motor}`:state.test_mode?'화면 확인 모드 · 실물 장치 연결은 꺼져 있습니다.':'';
}
async function devices(){
 deviceCatalog=await api('devices');const saved=storedDevices();
 $('#robot-profile-status').textContent=deviceCatalog.robot?.leader_calibration?'SO-101 · 교체 그리퍼 · 리더 보정 등록됨':'SO-101 · 교체 그리퍼 · 리더 보정 파일 미등록';
 for(let [selector,paths,key]of [['#follower-port',deviceCatalog.ports,'follower'],['#leader-port',deviceCatalog.ports,'leader'],['#gripper-device',deviceCatalog.cameras,'gripper'],['#table-device',deviceCatalog.cameras,'table']]){
  const el=$(selector),previous=el.value;el.replaceChildren(new Option('선택 안 함',''));
  if(key==='gripper')paths=paths.filter(p=>!p.startsWith('realsense:'));
  if(key==='table')paths=paths.filter(p=>p.startsWith('realsense:'));
  for(const path of paths){const name=path.split('/').pop();let label=name.replace(/^usb-/,'').replace(/-video-index0$/,'').replace(/-if00$/,'');
   if(key==='follower'||key==='leader'){const serial=label.split('_').pop();label='USB 시리얼'+' · '+serial.slice(-8)+(path===deviceCatalog.defaults.follower?' · 등록된 작업용 팔':path===deviceCatalog.defaults.leader?' · 등록된 조작용 팔':'');}
   if(path.startsWith('realsense:'))label='RealSense D435 · 컬러·깊이 · '+path.split(':')[1];
   el.add(new Option(label,path));
  }
  const choices=[state.device_paths?.[key],previous,saved[key],deviceCatalog.defaults[key]];
  el.value=choices.find(p=>p&&paths.includes(p))||'';
 }
 renderDevices();
 StudioSelect.refresh();
}
action('#devices-button',async()=>{await devices();$('#devices').showModal()});action('#refresh-devices',devices);
for(const id of ['#follower-port','#leader-port','#gripper-device','#table-device'])$(id).onchange=renderDevices;
action('#connect-devices',async()=>{
 const choice={follower:$('#follower-port').value,leader:$('#leader-port').value||null,gripper:$('#gripper-device').value,table:$('#table-device').value};
 const result=await command('connect',{follower:choice.follower,leader:choice.leader,cameras:{gripper:choice.gripper,table:choice.table}});
 if(result){try{localStorage.setItem('so101-uvc-overhead-d435-devices',JSON.stringify(choice))}catch{}$('#devices').close();}
});
function askRelease(op='release_torque'){
 releaseOperation=op;const disconnect=op==='disconnect';
 $('#release-title').textContent=disconnect?'힘 끄고 연결 해제':'모터 힘 끄기';
 $('#release-go').textContent=disconnect?'힘 끄고 연결 해제':'모터 힘 끄기';
 $('#release-description').textContent='팔과 잡고 있는 물체를 받쳐 주세요. 힘을 끄면 팔이 처질 수 있습니다. '+(disconnect?'팔 연결과 카메라도 종료합니다.':'팔 연결과 카메라는 유지됩니다. 따라가기·실물 실행은 정지합니다.');
 $('#supported').checked=false;$('#release-go').disabled=true;$('#devices').close();$('#release-confirm').showModal();$('#supported').focus();
}
action('#release-motors',()=>askRelease());
action('#disconnect-devices',async()=>{if(StudioControls.describe(state).powered||state.mode==='fault')askRelease('disconnect');else {await command('disconnect');$('#devices').close()}});
$('#supported').onchange=()=>$('#release-go').disabled=!$('#supported').checked;
$('#release-cancel').onclick=()=>$('#release-confirm').close();
action('#release-go',async()=>{if(!$('#supported').checked)return;$('#release-go').disabled=true;try{const result=await command(releaseOperation,{supported:true});if(result)$('#release-confirm').close();}finally{$('#release-go').disabled=!$('#supported').checked}});
action('#help-button',()=>$('#usage-guide').showModal());
let nextAction='';
action('#next-action',()=>{if(nextAction==='#save'||nextAction==='#discard')switchTab('record');if(nextAction)shortcutClick(nextAction)});
function renderControls(){
 if(!state.mode)return;
 const view=StudioControls.describe(state,tab);
 $('#follower-state').textContent=state.connected?'연결됨':'연결 안 됨';$('#leader-state').textContent=state.leader?'연결됨':'연결 안 됨';
 $('#torque-state').textContent=view.motor;$('#torque-state').dataset.state=state.mode==='fault'?'error':view.powered?'active':'idle';
 $('#camera-state').textContent=view.cameras+' / 2 수신';
 $('#release-motors').disabled=!state.connected||(!view.powered&&state.mode!=='fault');
 $('#devices-button').textContent=state.connected?'장치 확인':'장치 연결';
 $('#enable-leader').disabled=!!view.teleopReason;$('#enable-leader').title=view.teleopReason||'모터 힘을 켜고 조작용 팔의 관절 각도를 따라갑니다.';
 $('#enable-leader').textContent=['teleop','recording'].includes(state.mode)?'리더 따라가는 중':state.armed?'리더 따라가기 재개':'리더 따라가기';
 const oldFolder=episodes.some(e=>(e.task_id||null)===(state.task_id||null)&&e.robot_type!=='so101');
 $('#record').disabled=!!view.recordReason||!!state.storage_busy||oldFolder;$('#record').title=oldFolder?'폴더 관리에서 SO-101 작업 폴더를 만들어 선택하세요.':view.recordReason;
 $('#next-step').textContent=view.next;nextAction=view.action;
 const target=nextAction?$(nextAction):null;$('#next-action').hidden=!target;$('#next-action').disabled=!target||target.disabled;
 if(target)$('#next-action').textContent=target.textContent;
 if(tab==='record'&&oldFolder&&['idle','holding','preview'].includes(state.mode)){
  $('#next-step').textContent='폴더 관리에서 SO-101 작업 폴더를 만드세요. 이전 기록과 별도로 저장합니다.';
  nextAction='#manage-folders';$('#next-action').textContent='폴더 관리';$('#next-action').hidden=false;$('#next-action').disabled=false;
 }
 $('#connection').textContent=state.test_mode?'화면 시험 · 실물 제어 꺼짐':state.connected?'팔 연결됨':'장치 연결 전';
 if(state.mode==='fault'||view.powered&&!state.armed)$('#tracking').textContent='모터 힘 '+view.motor;
 renderDevices();
 StudioSelect.refresh();
}
function renderRecordingInterruption(){
 const issue=state.recording_interruption;
 const banner=$('#recording-alert');
 const camera=issue?.reason?.startsWith('그리퍼 카메라')?'그리퍼 카메라':issue?.reason?.startsWith('작업대 카메라')?'작업대 카메라':'카메라';
 const text=issue?`녹화가 중단되었습니다 · 저장할 수 없습니다\n${camera} 영상 문제로 ${time(issue.seconds||0)} 지점에서 끝났습니다. 카메라 연결을 확인하고 ‘버리기’ 후 다시 시작해 주세요. 팔 조작은 유지됩니다.`:'';
 if(banner.textContent!==text)banner.textContent=text;
 banner.hidden=!issue;renderNotices();
}
function jointValue(q,i){return (q*(i===5&&state.robot_type==='so101'?100/1.74533:180/Math.PI)).toFixed(1)}
async function updateStatus(){state=await api('status');renderRecordingInterruption();$('#connection').textContent=state.test_mode?'화면 확인 중 · 실물 제어 꺼짐':state.armed?'모터 제어 켜짐':state.connected?'상태 확인 중':'연결 안 됨';$('#status').textContent=ko(state.message);$('#mode-label').textContent=ko(state.mode);$('#mode-dot').dataset.state=state.error?'error':['recording','model','replay','teleop'].includes(state.mode)?'active':'idle';$('#connection').dataset.state=state.armed?'active':state.connected?'connected':'idle';$('#dataset-status').textContent=state.dataset?.state==='ready'?'':state.dataset?.message||'';$('#build-dataset').disabled=state.dataset?.state==='writing'||!episodes.length;statusNotices();const live=state.mode!=='preview';$('#view-context').textContent=live?(state.connected?'작업용 팔의 현재 자세':'기본 자세 · 실물 연결 안 됨'):`저장한 움직임 · 기록 ${number(state.selected)}`;$('#sim-status').textContent=ko(state.sim.state);$('#record-time').textContent=time(state.recording_seconds);$('#draft-label').textContent=state.recording_interruption?'중단된 녹화 · 저장 불가':state.mode==='recording'?`녹화 중 · ${state.draft_count}개 시점`:state.draft?`저장 대기 · ${state.draft_count}개 시점`:'녹화 대기';$('#record').disabled=state.storage_busy||state.mode!=='teleop'||!!state.draft;$('#stop-record').disabled=state.storage_busy||state.mode!=='recording';$('#save').disabled=state.storage_busy||!!state.recording_interruption||!state.draft||state.draft_count<2;$('#discard').disabled=state.storage_busy||!state.draft||!['idle','holding','teleop','recording','preview','fault'].includes(state.mode);$('#enable-leader').disabled=!state.leader||!['idle','holding','preview'].includes(state.mode);const review=['idle','holding','preview'].includes(state.mode);$('#episode').disabled=!review;$('#include').disabled=!selected||!review||!!state.storage_busy;$('#episode-result').disabled=$('#include').disabled;$('#delete-episode').disabled=!selected||!review||!!state.storage_busy;$('#preview').disabled=!episodes.length||!review;$('#replay').disabled=episodes.find(e=>e.id===selected)?.robot_type!=='so101'||!state.connected||!episodes.length||!review||!!state.draft;$('#timeline').disabled=!state.selected||!review;$('#preview').textContent=state.previewing?'영상 일시정지':'영상·시뮬 재생';if(!seekDragging){$('#timeline').max=Math.max(.001,state.duration);$('#timeline').value=state.play_t}$('#play-time').textContent=time(state.play_t);$('#duration').textContent=time(state.duration)+' · 녹화한 속도';$('#tracking').textContent=state.armed?`최대 각도 차이 ${Math.max(...state.joint_error_deg.slice(0,5).map(Math.abs)).toFixed(1)}° · ${ko(state.mode)}`:'모터 제어 꺼짐';$('#global-stop').disabled=!state.connected&&!state.previewing&&!state.model?.id;$('#replay-stop').disabled=!state.armed&&!state.previewing;all('nav [data-tab]').forEach(b=>b.disabled=state.mode==='recording');const table=$('#joints');table.replaceChildren();['관절 / 단위','조작용 팔','실제값','목표값','차이',...state.measured.flatMap((q,i)=>[(1+i)+' '+['밑동 (°)','어깨 (°)','팔꿈치 (°)','손목 굽힘 (°)','손목 회전 (°)','그리퍼 (열림 %)'][i],state.leader_measured?jointValue(state.leader_measured[i],i):'—',jointValue(q,i),jointValue(state.goal[i],i),jointValue(state.goal[i]-q,i)])].forEach(v=>{const e=document.createElement('span');e.textContent=v;table.append(e)});$('#recovery').textContent=state.abandoned_drafts.length?`중단된 녹화 ${state.abandoned_drafts.length}개를 data/drafts에 보관했습니다. 원본은 삭제하지 않았습니다.`:'';for(const name of ['gripper','table']){const camera=state.cameras?.[name];const receiving=camera?.connected&&!camera.error&&camera.age!=null&&camera.age<=.2;const label=receiving?'영상 수신 중':camera?.state==='recovering'?'다시 연결 중':camera?.error&&camera.error!=='Not connected'?'영상 수신 중단':'연결 안 됨';$('#'+name+'-source').textContent=live?label:'저장한 영상';$('#'+name+'-placeholder').textContent=live?label:'저장한 영상을 불러오고 있습니다';}renderControls();renderDepthView();if(typeof renderLearningState==='function')renderLearningState();if(typeof renderMotorCalibration==='function')renderMotorCalibration();}
let tableView='rgb';
function renderDepthView(){
 const depth=tableView==='depth',recorded=state.mode==='preview';
 $('#table-rgb-view').setAttribute('aria-pressed',String(!depth));
 $('#table-depth-view').setAttribute('aria-pressed',String(depth));
 $('#table-image').alt=depth?'상단 D435 깊이 영상':'상단 D435 컬러 영상';
 $('#depth-legend').hidden=!depth||!state.depth_view?.available;
 if(depth){
  $('#table-placeholder').textContent=recorded?'이 기록에는 깊이 영상이 없습니다.':'D435 깊이 영상 연결 안 됨';
  $('#table-source').textContent=state.depth_view?.available?(recorded?'저장한 깊이':'깊이 수신 중'):(recorded?'깊이 기록 없음':'연결 안 됨');
 }
}
for(const [id,view] of [['#table-rgb-view','rgb'],['#table-depth-view','depth']])action(id,()=>{
 tableView=view;$('#table-image').hidden=true;$('#table-placeholder').hidden=false;renderDepthView();
});
const urls={};
async function imageLoop(name){
 const source=name==='table'&&tableView==='depth'?'table-depth':name;
 try{
  const r=await fetch('/api/image/'+source,{cache:'no-store'});
  error(r.status>=400?'영상을 불러오지 못했습니다 ('+r.status+').':null,'image-'+name);
  const blob=r.status===200?await r.blob():null;
  if(source!==(name==='table'&&tableView==='depth'?'table-depth':name))return;
  if(blob){
   const url=URL.createObjectURL(blob),img=$('#'+name+'-image');img.src=url;img.hidden=false;
   $('#'+name+'-placeholder').hidden=true;if(urls[name])URL.revokeObjectURL(urls[name]);urls[name]=url;
  }else{
   $('#'+name+'-image').hidden=true;$('#'+name+'-placeholder').hidden=false;
   if(name==='sim')$('#sim-placeholder').textContent=state.sim?.state==='disabled'?'자세 화면 없이 실행 중':state.sim?.state==='error'?'자세 화면 오류 · outputs/sim.log를 확인해 주세요':state.sim?.state==='reconnecting'?'자세 화면을 다시 연결하고 있습니다…':state.sim?.state==='stale'?'자세 화면 갱신이 멈췄습니다':'자세 화면을 준비하고 있습니다…';
  }
 }catch(e){error(`${name==='sim'?'시뮬레이션':name==='gripper'?'그리퍼 카메라':'상단 카메라'} 영상 연결이 끊겼습니다.`,'image-'+name)}
 finally{setTimeout(()=>imageLoop(name),1000/15)}
}

async function poll(){try{await api('heartbeat',{});await updateStatus();error('', 'connection')}catch(e){error('프로그램 연결이 끊겼습니다. 표시된 자세와 모터 상태는 마지막 수신 값입니다. '+ko(e.message),'connection');$('#torque-state').textContent='통신 끊김 · 확인 필요';$('#connection').textContent='프로그램 연결 끊김'}setTimeout(poll,300)}
(async()=>{await refreshEpisodes();await updateStatus();if(episodes.length){await selectEpisode(episodes[episodes.length-1].id)}})().catch(e=>error(e.message)).finally(()=>{poll();for(const name of ['sim','gripper','table'])imageLoop(name)});
