'use strict';
let learning={tasks:[],jobs:[],models:[]};
let displayedTrainingJob=null;
function fillOptions(id,items,empty,selectedValue){
 const el=$(id),previous=selectedValue===undefined?el.value:selectedValue;
 const next=[{id:'',name:empty},...items];
 const changed=el.options.length!==next.length||next.some((item,i)=>el.options[i]?.value!==item.id||el.options[i]?.text!==item.name);
 // Keep an open dropdown and its scroll position intact during status polling.
 if(changed){el.replaceChildren();for(const item of next)el.add(new Option(item.name,item.id));}
 const wanted=items.some(i=>i.id===previous)?previous:'';
 if(el.value!==wanted)el.value=wanted;
 StudioSelect.refresh(el);
}
function renderTrainingPicker(){
 const select=$('#training-job'),previous=select.value;
 const signature=JSON.stringify(learning.jobs.map(j=>[j.id,j.name,j.created_at]));
 if(select.dataset.signature!==signature){
  select.replaceChildren();
  for(const job of learning.jobs){
   const option=new Option(job.name,job.id);
   option.dataset.detail=new Date(job.created_at*1000).toLocaleString('ko-KR');
   select.add(option);
  }
  select.dataset.signature=signature;
 }
 select.value=learning.jobs.some(j=>j.id===previous)?previous:learning.jobs[0]?.id||'';
 select.disabled=!learning.jobs.length;StudioSelect.refresh(select);
}

function renderLearning(){
 const tasks=learning.tasks;
 const activeTasks=tasks.filter(t=>!episodes.some(e=>e.task_id===t.id&&e.robot_type!=='so101'));
 fillOptions('#record-task',activeTasks,activeTasks.length?'작업 폴더를 선택하세요':'폴더를 먼저 만드세요',state.task_id||'');
 $('#record-task').options[0].disabled=true;StudioSelect.refresh($('#record-task'));
 fillOptions('#episode-task',tasks,'미분류',episodes.find(e=>e.id===selected)?.task_id||'');
 for(const id of ['#train-task','#model-task'])fillOptions(id,id==='#train-task'?activeTasks:tasks,id==='#model-task'?'모든 작업':'작업을 선택하세요');
 const counts=episodes.filter(e=>e.task_id===$('#train-task').value&&e.included);
 $('#train-count').textContent=$('#train-task').value?`학습에 사용할 기록 ${counts.length}개 · ${time(counts.reduce((sum,e)=>sum+e.duration,0))} · 성공 ${counts.filter(e=>e.result==='Success').length}개 / 실패 ${counts.filter(e=>e.result==='Failure').length}개`:'기록 확인에서 폴더를 지정하고 ‘학습에 사용’을 선택하세요.';
 renderJob();renderModels();renderResume();renderExecution();renderLearningState();
}
function renderJob(){
 renderTrainingPicker();
 const job=learning.jobs.find(j=>j.id===$('#training-job').value),p=job?.progress;
 if(displayedTrainingJob!==(job?.id||null)){$('#train-log').hidden=true;$('#train-log').textContent='';displayedTrainingJob=job?.id||null;}
 error(job?.state==='error'?'학습 실패: '+job.message:null,'training');
 $('#train-status').textContent=job?.message||'아직 학습한 모델이 없습니다.';
 $('#train-progress').max=p?.total||1;$('#train-progress').value=p?.step||0;
 $('#train-metrics').textContent=p?`${p.step.toLocaleString()} / ${p.total.toLocaleString()}회 · 손실 ${p.loss.toFixed(5)} · ${time(p.elapsed)}`:'';
 $('#train-source').textContent=job?`${job.pending_episode_ids?.length?`다음 이어 학습에 추가할 기록 ${job.pending_episode_ids.length}개 · `:''}${job.task.name} · 기록 ${job.studio_ids.map(number).join(', ')} · ACT · ${job.params.device.toUpperCase()} · ${job.params.keep_latest_only?'최근 모델 1개 보관':'중간 모델 모두 보관'}`:'';
 $('#train-log-button').disabled=!job;
}
function renderModels(){
 const models=learning.models.filter(m=>!$('#model-task').value||m.task_id===$('#model-task').value);
 fillOptions('#trained-model',models.map(m=>({id:m.id,name:`${m.name} · ${m.step}회${m.robot_type==='omx'?' · 이전 OMX용':m.physical_allowed?' · SO-101':' · 시험 전용'}`})),'모델을 선택하세요');
 const model=models.find(m=>m.id===$('#trained-model').value);
 $('#model-info').textContent=model?`학습 작업: ${model.task_name} · 저장된 ${model.step}회 모델${model.physical_allowed?'':' · 시험 데이터 모델은 실물 실행 불가'}`:'저장이 완료된 모델을 선택해 주세요.';
}
function renderResume(){
 const models=learning.models.filter(m=>m.resumable&&m.robot_type==='so101');
 fillOptions('#resume-model',models.map(m=>({id:m.id,name:`${m.name} · ${Number(m.step).toLocaleString()}회`})),'저장된 모델을 선택하세요');
 const chosen=models.find(m=>m.id===$('#resume-model').value),extra=Number($('#resume-steps').value);
 $('#resume-info').textContent=chosen?`${Number(chosen.step).toLocaleString()}회부터 추가 ${Number.isFinite(extra)?extra.toLocaleString():'—'}회 학습합니다. 저장된 학습 상태에서 이어서 학습하고 새 모델로 저장합니다.`:'모델과 학습 상태를 함께 불러오며 기존 모델은 보존합니다.';
}
function renderLearningState(){
 const review=['idle','holding','preview'].includes(state.mode);
 const training=learning.jobs.some(j=>['running','starting','stopping'].includes(j.state));
 $('#manage-folders').disabled=!!state.storage_busy||(!review&&state.mode!=='teleop');$('#manage-folders').title='작업 폴더 만들기·이름 변경';
 if($('#folder-dialog').open)renderFolderDialog();
 if(!pending&&[...$('#record-task').options].some(o=>o.value===state.task_id))$('#record-task').value=state.task_id;
 $('#record-task').disabled=state.storage_busy||(!review&&state.mode!=='teleop')||!!state.draft;
 const oldRecordFolder=episodes.some(e=>(e.task_id||null)===(state.task_id||null)&&e.robot_type!=='so101');
 $('#record-folder-path').textContent=!state.task_id?'작업 폴더를 선택하면 저장 위치가 표시됩니다.':oldRecordFolder?'새 SO-101 작업 폴더를 만들어 선택하세요. 이전 기록은 기록 확인에서 볼 수 있습니다.':state.task_folder?`저장 위치: ${state.task_folder}/episodes`:'';
 const folder=learning.tasks.find(t=>t.id===$('#train-task').value);
 $('#task-folder-path').textContent=folder?`저장 위치: ${folder.folder}`:'폴더를 만들면 이 PC에 저장 공간이 생깁니다.';
 $('#episode-task').disabled=!review||!selected||!!state.storage_busy;
 const folderReasons=StudioControls.folderReasons(state,learning.tasks,$('#train-task').value,$('#task-name').value);
 $('#task-add').disabled=!!folderReasons.create;$('#task-add').title=folderReasons.create;
 $('#task-rename').disabled=!!folderReasons.rename;$('#task-rename').title=folderReasons.rename;
 $('#train-start').disabled=!review||state.torque?.release_required||state.armed||training||state.dataset?.state==='writing'||!$('#train-task').value||!episodes.some(e=>e.task_id===$('#train-task').value&&e.included);
 $('#train-stop').disabled=!training||learning.jobs.some(j=>j.state==='stopping');
 const resumeModel=learning.models.find(m=>m.id===$('#resume-model').value);
 $('#train-resume').disabled=resumeModel?.robot_type!=='so101'||!review||state.torque?.release_required||state.armed||training||!resumeModel?.resumable||($('#resume-data').value==='latest'&&state.dataset?.state==='writing');
 const chosen=learning.models.find(m=>m.id===$('#trained-model').value),model=state.model||{};
 $('#model-load').disabled=!review||state.torque?.release_required||state.armed||training||!chosen||chosen.robot_type==='omx';
 $('#model-run').disabled=!review||!state.connected||!!state.draft||!model.ready||model.id!==chosen?.id||!chosen?.physical_allowed||state.test_mode||!executionMatches(model.options,readExecution())||StudioControls.describe(state).cameras<2;
 renderExecutionState(review,chosen,model);
 $('#model-stop').disabled=!state.armed&&!model.id;
 const deleteReason=chosen?.delete_blocked_reason||(chosen&&model.id===chosen.id?'준비하거나 실행 중인 모델입니다. 동작 정지를 누른 뒤 삭제하세요.':'');
 $('#model-delete').disabled=!review||!chosen||!!deleteReason||!!pending;
 $('#model-delete-info').textContent=deleteReason;
 $('#trained-model').disabled=!review;$('#model-task').disabled=!review;
 
 const powered=state.armed||state.torque?.release_required;
 const baseReason=!review?'진행 중인 팔 조작을 먼저 정지하세요.':powered?'상단의 모터 힘 끄기를 누른 뒤 시작하세요. 연결과 영상은 유지됩니다.':training?'학습 중입니다. 중단 시에는 현재 모델을 저장한 뒤 끝납니다.':'';
 const invalidTrain=all('#train-panel input[type=number]').find(input=>input.id.startsWith('train-')&&!input.checkValidity());
 const trainInputReason=invalidTrain?`${invalidTrain.closest('label').firstChild.textContent.trim()}: ${invalidTrain.validationMessage}`:Number($('#train-n_action_steps').value)>Number($('#train-chunk_size').value)?'연속 실행 길이는 예측 길이 이하로 설정하세요.':'';
 const trainReason=baseReason||(state.dataset?.state==='writing'?'학습 파일을 만들고 있습니다. 완료 후 시작하세요.':!folder?'학습할 작업 폴더를 선택하세요.':!episodes.some(e=>e.task_id===folder.id&&e.included)?'이 폴더에 학습용 기록이 없습니다. 기록 확인에서 폴더와 학습에 사용을 지정하세요.':!$('#model-name').value.trim()?'저장할 모델 이름을 입력하세요.':trainInputReason||'학습 준비 완료 · 학습 시작을 누르세요.');
 $('#train-help').textContent=trainReason;$('#train-start').title=trainReason;
 $('#train-start').disabled=$('#train-start').disabled||!$('#model-name').value.trim()||!!trainInputReason;
 $('#resume-help').textContent=baseReason||(!resumeModel?'이어갈 모델을 선택하세요.':'추가 횟수만큼 학습하고 새 모델로 저장합니다.');
 const cameraCount=StudioControls.describe(state).cameras;
 const modelReason=baseReason||(!chosen?'실행할 모델을 선택하세요.':chosen.robot_type==='omx'?'이전 OMX 모델입니다. SO-101 기록으로 학습한 모델을 선택하세요.':model.id!==chosen.id?'모델 불러오기를 눌러 준비하세요.':!model.ready?'모델을 불러오고 있습니다.':!executionMatches(model.options,readExecution())?'설정이 바뀌었습니다. 모델을 다시 불러오세요.':!chosen.physical_allowed?'시험 데이터로 만든 모델은 실물에서 실행할 수 없습니다.':!state.connected?'장치 연결에서 작업용 팔과 카메라를 연결하세요.':cameraCount<2?'그리퍼·작업대 카메라의 정상 영상이 모두 필요합니다.':state.draft?'저장 전 녹화를 저장하거나 버린 뒤 실행하세요.':'준비 완료 · 실물에서 실행을 누르면 팔이 움직입니다.');
 $('#model-help').textContent=modelReason;$('#model-run').title=modelReason;
 const selectedEpisode=episodes.find(e=>e.id===selected);
 const replayReason=!review?'정지 (힘 유지)를 누른 뒤 기록을 선택하세요.':!selectedEpisode?'재생할 기록을 선택하세요.':selectedEpisode.robot_type!=='so101'?'SO-101 기록을 선택하세요. 이 기록은 영상으로 확인할 수 있습니다.':selectedEpisode.synthetic?'시험용 기록은 실물에서 실행할 수 없습니다.':!state.connected?'영상·시뮬은 바로 볼 수 있습니다. 실물 재실행에는 작업용 팔을 연결하세요.':cameraCount<2?'실물 재실행에는 카메라 2대의 영상이 필요합니다.':training?'기록의 실물 재실행은 학습을 중단한 뒤 사용할 수 있습니다.':state.draft?'현재 녹화를 저장하거나 버린 뒤 실행하세요.':'';
 $('#replay').disabled=!!replayReason||!!state.test_mode;$('#replay').title=replayReason;
 $('#review-help').textContent=replayReason||'선택한 기록을 원래 속도로 실행할 수 있습니다.';

 $('#model-status').textContent=model.error?`모델 불러오기 실패: ${model.error}`:state.mode==='model'?`${model.metrics?.waiting_for_inference?'추론 대기 · 현재 목표 유지':'실행 중'} ${time(model.elapsed)} / ${time(model.seconds)} · 추론 ${model.inference_ms?.toFixed(0)||'—'}ms`:model.ready?(chosen?.physical_allowed&&!state.test_mode?'모델 준비 완료 · 장치와 실행 조건을 확인하세요.':'모델 준비 완료 · 시험용 모델 또는 화면 시험 모드입니다.'):model.id?'모델을 불러오는 중…':'모델을 불러와도 팔은 움직이지 않습니다.';
 StudioSelect.refresh();
}
let executionModelId=null,previousExecutionMode='original',previousActionSteps=20;
const executionNumbers=['action_steps','prefetch_fraction','blend_ms','filter_ms','ensemble_coeff'];
function readExecution(){
 const out={mode:$('#exec-mode').value,prefetch:$('#exec-prefetch').value,gripper_filter:$('#exec-gripper_filter').checked};
 for(const key of executionNumbers)out[key]=Number($('#exec-'+key).value);
 if(out.mode==='ensemble'){out.action_steps=1;out.blend_ms=0;}
 return out;
}
function executionMatches(a,b){return !!a&&Object.keys(b).every(k=>a[k]===b[k]);}
function fillExecution(o){
 for(const key of executionNumbers){
  const el=$('#exec-'+key);
  if(el.tagName==='SELECT'&&!Array.from(el.options).some(v=>Number(v.value)===o[key]))el.add(new Option(o[key]+'ms',String(o[key])));
  el.value=o[key];
 }
 $('#exec-mode').value=o.mode;$('#exec-prefetch').value=o.prefetch;$('#exec-gripper_filter').checked=o.gripper_filter;
 previousExecutionMode=o.mode;previousActionSteps=o.mode==='ensemble'?Math.min(20,Number($('#exec-action_steps').max)):o.action_steps;
}
function renderExecution(){
 const chosen=learning.models.find(m=>m.id===$('#trained-model').value);
 if(!chosen){executionModelId=null;return;}
 if(executionModelId!==chosen.id){
  $('#exec-action_steps').max=chosen.chunk_size||100;
  fillExecution(chosen.execution_options||{mode:'original',action_steps:20,prefetch:'auto',prefetch_fraction:.5,blend_ms:0,filter_ms:0,gripper_filter:false,ensemble_coeff:.01});
  executionModelId=chosen.id;
 }
}
function validExecution(){
 for(const key of executionNumbers){const input=$('#exec-'+key);if(!input.disabled&&!input.checkValidity()){error(input.closest('label').firstChild.textContent.trim()+': '+input.validationMessage);input.focus();return false;}}
 return true;
}
function renderExecutionState(review,chosen,model){
 const config=readExecution();
 for(const el of all('#execution-settings input, #execution-settings select, #execution-settings button'))el.disabled=!review||!chosen;
 $('#exec-action_steps').disabled=!review||!chosen||config.mode==='ensemble';
 $('#exec-action_steps').min=config.mode==='prefetch'?2:1;
 $('#exec-blend_ms').disabled=!review||!chosen||config.mode==='ensemble';
 $('#exec-prefetch').disabled=!review||!chosen||config.mode!=='prefetch';
 $('#exec-prefetch_fraction').disabled=!review||!chosen||config.mode!=='prefetch'||config.prefetch!=='fixed';
 $('#exec-ensemble_coeff').disabled=!review||!chosen||config.mode!=='ensemble';
 $('#exec-description').textContent={original:'자료의 ACT 동작 묶음을 차례로 실행합니다. 추가 속도·가속도 제한은 적용하지 않습니다.',prefetch:'현재 동작이 끝나기 전에 다음 묶음을 계산합니다. 지나간 시점의 예측은 실행하지 않습니다.',ensemble:'매 단계 새 예측을 계산하고 ACT 방식으로 평균합니다. 실행 길이는 1로 고정되며 계산 부담이 커집니다.'}[config.mode];
 $('#exec-dirty').textContent=chosen&&model.id===chosen.id&&!executionMatches(model.options,config)?'설정 변경됨 · 모델 불러오기를 다시 눌러 주세요.':chosen&&!executionMatches(chosen.execution_options,config)?'설정 변경됨 · 설정 저장 또는 모델 불러오기로 적용합니다.':'';
 const m=model.metrics||{},show=v=>Number.isFinite(v)?v.toFixed(1):'—';
 $('#exec-metrics').textContent=model.id?`새 예측 계산 ${show(m.new_prediction_ms)}ms · 응답 도착 ${show(m.response_ms)}ms · 영상 지연 ${show(m.observation_age_ms)}ms · 새 예측 ${show(m.prediction_hz)}Hz · 목표 갱신 ${show(m.command_hz)}Hz · 남은 동작 ${m.remaining??'—'}개 · 동작 대기 ${m.buffer_waits??0}회 · 지난 묶음 제외 ${m.dropped_chunks??0}회`:'';
}
all('#execution-settings input, #execution-settings select').forEach(el=>el.addEventListener('change',()=>{
 if(el.id==='exec-mode'){
  if(el.value==='ensemble'){previousActionSteps=Number($('#exec-action_steps').value);$('#exec-action_steps').value=1;}
  else if(previousExecutionMode==='ensemble')$('#exec-action_steps').value=previousActionSteps;
  previousExecutionMode=el.value;
 }
 renderLearningState();
}));
action('#exec-save',async()=>{if(!validExecution())return;await learningCommand('model_settings',{id:$('#trained-model').value,options:readExecution()});});
action('#exec-reset',()=>{const chosen=learning.models.find(m=>m.id===$('#trained-model').value);fillExecution(chosen?.default_execution_options||{mode:'original',action_steps:Math.min(20,chosen?.chunk_size||100),prefetch:'auto',prefetch_fraction:.5,blend_ms:0,filter_ms:0,gripper_filter:false,ensemble_coeff:.01});renderLearningState();});
async function refreshLearning(){learning=await api('learning');error('', 'learning-connection');renderLearning();}
async function learningCommand(op,data){const result=await command(op,data);await refreshLearning();return result;}
function renderFolderDialog(){
 fillOptions('#folder-choice',learning.tasks,'새 폴더 만들기');
 const selectedFolder=learning.tasks.find(t=>t.id===$('#folder-choice').value);
 const renameBusy=state.dataset?.state==='writing';
 const reasons=StudioControls.folderReasons(state,learning.tasks,$('#folder-choice').value,$('#folder-name').value);
 $('#folder-rename').disabled=!!reasons.rename;$('#folder-rename').title=reasons.rename;
 $('#folder-create').disabled=!!reasons.create;$('#folder-create').title=reasons.create;
 $('#folder-help').textContent=selectedFolder&&renameBusy?'학습 파일 생성 중에는 이름 변경만 기다려 주세요. 새 폴더는 만들 수 있습니다.':state.draft?'저장 전 녹화의 폴더는 그대로 유지됩니다.':selectedFolder?'이름 변경은 기존 기록이 담긴 실제 폴더에 적용됩니다.':'만든 폴더는 다음 녹화의 저장 위치로 선택됩니다.';
}
action('#manage-folders',()=>{renderFolderDialog();$('#folder-choice').value=state.task_id||'';$('#folder-name').value=learning.tasks.find(t=>t.id===state.task_id)?.name||'';renderFolderDialog();$('#folder-dialog').showModal();$('#folder-name').focus()});
$('#folder-choice').onchange=()=>{$('#folder-name').value=learning.tasks.find(t=>t.id===$('#folder-choice').value)?.name||'';renderFolderDialog()};
$('#folder-name').oninput=renderFolderDialog;
action('#folder-create',async()=>{const task=await learningCommand('task_save',{name:$('#folder-name').value});if(task&&!state.draft)await learningCommand('task_select',{id:task.id});$('#folder-dialog').close()});
action('#folder-rename',async()=>{await learningCommand('task_save',{id:$('#folder-choice').value,name:$('#folder-name').value});$('#folder-dialog').close()});
$('#model-name').oninput=renderLearningState;
$('#task-name').oninput=renderLearningState;

$('#record-task').onchange=()=>learningCommand('task_select',{id:$('#record-task').value||null}).catch(e=>{error(e.message);renderLearning()});
$('#episode-task').onchange=()=>learningCommand('task_assign',{id:selected,task_id:$('#episode-task').value||null}).catch(e=>{error(e.message);renderLearning()});
$('#train-task').onchange=()=>{$('#task-name').value=learning.tasks.find(t=>t.id===$('#train-task').value)?.name||'';renderLearning()};
$('#model-task').onchange=()=>{renderModels();renderExecution();renderLearningState()};
$('#trained-model').onchange=()=>{renderModels();executionModelId=null;renderExecution();renderLearningState()};
$('#training-job').onchange=()=>{renderJob();$('#train-log').hidden=true};
action('#task-add',async()=>{const task=await command('task_save',{name:$('#task-name').value});await refreshLearning();if(task){$('#train-task').value=task.id;renderLearning()}});
action('#task-rename',()=>learningCommand('task_save',{id:$('#train-task').value,name:$('#task-name').value}));
action('#train-start',async()=>{
 const params={};for(const key of ['steps','batch_size','learning_rate','chunk_size','n_action_steps','save_freq','seed','num_workers']){const input=$('#train-'+key);if(!input.checkValidity()){error(input.closest('label').firstChild.textContent.trim()+': '+input.validationMessage);input.focus();return;}params[key]=Number(input.value)}
 params.keep_latest_only=$('#train-keep_latest_only').checked;params.device=$('#train-device').value;params.pretrained_backbone=$('#train-pretrained_backbone').checked;
 if(params.n_action_steps>params.chunk_size){error('연속 실행 길이는 예측 길이 이하로 설정하세요.');return;}
 const job=await learningCommand('train_start',{task_id:$('#train-task').value,name:$('#model-name').value.trim(),params});if(job?.id){$('#training-job').value=job.id;renderJob();}
});
action('#train-stop',()=>learningCommand('train_stop',{}));
action('#train-log-button',async()=>{try{const log=await api('training/'+$('#training-job').value+'/log');$('#train-log').textContent=log.text;$('#train-log').hidden=false}catch(e){error(e.message)}});
action('#model-load',()=>{if(!validExecution())return;return learningCommand('model_load',{id:$('#trained-model').value,device:$('#model-device').value,options:readExecution()})});
action('#model-run',()=>{if(!$('#model-seconds').checkValidity()){error('실행 시간: '+$('#model-seconds').validationMessage);$('#model-seconds').focus();return;}confirm('model_run','모델 실물 실행','카메라 위치·방향과 물체 배치를 학습할 때와 같게 맞춰 주세요. 모델이 예측한 동작으로 팔이 움직이며, 설정한 시간이 되거나 F8을 누르면 멈추고 자세를 유지합니다.',{seconds:Number($('#model-seconds').value)})});
action('#model-stop',()=>command('stop'));
async function learningPoll(){try{await refreshLearning()}catch(e){error(e.message,'learning-connection')}setTimeout(learningPoll,2000)}
learningPoll();setInterval(renderLearningState,300);

$('#resume-model').onchange=()=>{renderResume();renderLearningState()};
$('#resume-data').onchange=renderLearningState;
$('#resume-steps').oninput=renderResume;
action('#train-resume',async()=>{
 const input=$('#resume-steps');if(!input.checkValidity()){error('추가 학습 횟수: '+input.validationMessage);input.focus();return;}
 await learningCommand('train_resume',{model_id:$('#resume-model').value,additional_steps:Number(input.value),include_new:$('#resume-data').value==='latest',name:$('#resume-name').value.trim()||null,keep_latest_only:$('#resume-keep_latest_only').checked});
});

let deletingModel=null;
action('#model-delete',()=>{
 const chosen=learning.models.find(m=>m.id===$('#trained-model').value);
 if(!chosen)return;
 deletingModel={id:chosen.id,name:chosen.name,step:chosen.step};
 $('#model-delete-name').textContent=`${chosen.name} · ${Number(chosen.step).toLocaleString()}회`;
 const last=!learning.models.some(m=>m.job_id===chosen.job_id&&m.id!==chosen.id);
 $('#model-delete-description').textContent='선택한 모델과 이어 학습 상태를 영구 삭제합니다. '+(last?'이 학습의 마지막 모델이므로 학습 기록과 로그도 함께 삭제합니다.':'같은 학습의 다른 모델이 남아 있어 공통 학습 기록은 유지합니다. 마지막 모델을 삭제할 때 함께 삭제됩니다.')+' 원본 기록과 다른 학습의 모델은 유지합니다.';
 $('#model-delete-confirm').showModal();$('#model-delete-cancel').focus();
});
$('#model-delete-cancel').onclick=()=>$('#model-delete-confirm').close();
$('#model-delete-confirm').addEventListener('close',()=>{deletingModel=null;});
action('#model-delete-go',async()=>{
 const target=deletingModel;if(!target||pending)return;
 $('#model-delete-go').disabled=true;
 try{
  const result=await command('model_delete',{id:target.id,confirmed:true});
  if(result){$('#model-delete-confirm').close();await refreshLearning();$('#trained-model').focus();}
 }finally{$('#model-delete-go').disabled=false;}
});
