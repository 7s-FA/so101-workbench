'use strict';
let motorCatalog=null,calibrationTab='file',calibrationFile=null,lastCalibrationSaved=null;
const motorNames=['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper'];
const motorLabels=['밑동','어깨','팔꿈치','손목 굽힘','집게 회전','집게 벌림'];
const calibrationDialog=()=>$('#motor-calibration-dialog');
function calibrationRole(){return $('#cal-role').value}
function calibrationChoices(){
 const role=calibrationRole(),robot=motorCatalog.robot;
 const port=$('#cal-port');port.replaceChildren(new Option('USB를 선택하세요',''));
 for(const p of motorCatalog.ports)port.add(new Option(p.split('/').pop(),p));
 port.value=motorCatalog.ports.includes(robot[role+'_port'])?robot[role+'_port']:'';
 const files=$('#cal-file');files.replaceChildren(new Option('보정 파일을 선택하세요',''));
 for(const row of motorCatalog.files)files.add(new Option(row.name+' · '+row.path,row.id));
 files.value=motorCatalog.files.find(f=>f.active_roles.includes(role))?.id||'';
 calibrationFile=motorCatalog.files.find(f=>f.id===files.value)||null;
 $('#cal-name').value=(calibrationFile?.name||('so101_'+role));$('#cal-upload').value='';
 $('#cal-supported').checked=false;StudioSelect.refresh();renderMotorCalibration();
}
async function openMotorCalibration(){
 motorCatalog=await api('calibration');$('#devices').close();calibrationChoices();
 calibrationDialog().showModal();renderNotices();renderMotorCalibration();
}
action('#calibration-button',openMotorCalibration);action('#device-calibration-button',openMotorCalibration);
$('#cal-role').onchange=calibrationChoices;
$('#cal-port').onchange=renderMotorCalibration;$('#cal-supported').onchange=renderMotorCalibration;
$('#cal-file').onchange=()=>{calibrationFile=motorCatalog.files.find(f=>f.id===$('#cal-file').value)||null;$('#cal-upload').value='';renderMotorCalibration()};
$('#cal-upload').onchange=async()=>{
 try{
  const file=$('#cal-upload').files[0];if(!file)return;
  if(file.size>32768)throw Error('SO-101 관절 6개의 작은 JSON 파일을 선택하세요.');
  const values=JSON.parse(await file.text());
  if(motorNames.some(n=>!values[n]||!Number.isInteger(values[n].homing_offset)))throw Error('공식 SO-101 모터 보정 JSON이 아닙니다.');
  calibrationFile={name:file.name.replace(/\.json$/i,''),path:file.name,calibration:values};$('#cal-file').value='';StudioSelect.refresh();error(null,'calibration-file');renderMotorCalibration();
 }catch(e){calibrationFile=null;error(e.message,'calibration-file');renderMotorCalibration()}
};
for(const [id,view]of [['#cal-file-tab','file'],['#cal-new-tab','new']])action(id,()=>{calibrationTab=view;renderMotorCalibration()});
function renderMotorCalibration(){
 const c=state.motor_calibration||{},busy=!!c.busy;
 error(c.error,'motor-calibration');
 error(c.notice,'motor-calibration-notice');
 if(busy)$('#global-stop').disabled=false;
 if(!calibrationDialog().open)return;
 if(busy)calibrationTab=c.method||'new';
 const isNew=calibrationTab==='new';$('#cal-workspace').dataset.view=calibrationTab;calibrationDialog().dataset.view=calibrationTab;
 $('#cal-file-panel').hidden=isNew;$('#cal-new-panel').hidden=!isNew;
 for(const [id,name]of [['#cal-file-tab','file'],['#cal-new-tab','new']]){$(id).setAttribute('aria-pressed',String(calibrationTab===name));$(id).disabled=busy;}
 for(const id of ['#cal-role','#cal-port','#cal-name','#cal-file','#cal-upload'])$(id).disabled=busy;
 $('#cal-use').hidden=isNew;$('#cal-apply').hidden=isNew||busy;$('#cal-start').hidden=!isNew||busy;
 $('#cal-next').hidden=!busy||!['MIDPOINT','RANGE','REVIEW'].includes(c.state);
 $('#cal-resume').hidden=!busy||c.state!=='REVIEW';$('#cal-cancel').hidden=!busy;
 const actions={MIDPOINT:'중간 자세 완료 · 영점 기록',RANGE:'범위 측정 완료',REVIEW:'모터와 파일에 저장'};
 $('#cal-next').textContent=actions[c.state]||'처리 중';
 $('#cal-next').disabled=!!state.test_mode||!actions[c.state];
 $('#cal-start').disabled=!!state.test_mode||busy||!$('#cal-port').value||!$('#cal-supported').checked;$('#cal-supported').parentElement.hidden=false;$('#cal-apply').disabled=!!state.test_mode||busy||!calibrationFile||!$('#cal-port').value||!$('#cal-supported').checked;
 $('#cal-use').disabled=!!state.test_mode||busy||!calibrationFile||!$('#cal-port').value||!!state.connected;
 $('#cal-file-path').textContent=calibrationFile?'파일: '+calibrationFile.path:'기존 파일을 고르거나 JSON을 선택하세요.';
 const descriptions={CONNECTING:'연결과 모터 상태를 확인합니다.',MIDPOINT:'공식 그림처럼 모든 관절을 범위의 중간에 맞춘 뒤 영점을 기록하세요.',RANGE:'집게 회전을 제외한 관절을 하나씩 양 끝까지 천천히 움직이세요. 집게는 완전히 닫고 벌려 주세요.',REVIEW:'표의 영점과 최소·최대 값을 확인한 뒤 저장하세요.',SAVING:'모터에 저장하고 다시 읽어서 확인합니다.'};
 $('#cal-stage').textContent=state.test_mode?'화면 시험 모드 · 모터 보정과 파일 등록은 비활성화되어 있습니다.':busy?(descriptions[c.state]||c.message):isNew?(c.message||'팔을 받친 뒤 보정 시작을 누르세요.'):state.connected?'보정 파일을 바꾸려면 장치 연결을 해제하세요.':c.saved?c.message:'이 파일을 선택한 팔에 등록합니다. 모터값은 변경하지 않습니다.';
 const rows=$('#cal-rows');rows.replaceChildren();
 for(let i=0;i<6;i++){
  const n=motorNames[i],saved=isNew?c.saved?.calibration?.[n]:calibrationFile?.calibration?.[n];
  const sample=isNew?c.positions:null;
  const fixed=sample&&n==='wrist_roll';const values=[motorLabels[i],saved?.range_min??(fixed?0:sample?.mins?.[n]),sample?.current?.[n],saved?.range_max??(fixed?4095:sample?.maxes?.[n]),saved?.homing_offset??sample?.offsets?.[n]];
  const row=document.createElement('tr');for(const value of values){const cell=document.createElement('td');cell.textContent=value??'—';row.append(cell)}rows.append(row);
 }
 if(!busy&&c.saved&&c.completed_at!==lastCalibrationSaved){lastCalibrationSaved=c.completed_at;api('calibration').then(data=>{motorCatalog=data;calibrationChoices()}).catch(e=>error(e.message,'calibration-file'));}
}
action('#cal-use',async()=>{
 const body={role:calibrationRole(),port:$('#cal-port').value,name:calibrationFile.name};
 if(calibrationFile.id)body.file_id=calibrationFile.id;else body.calibration=calibrationFile.calibration;
 await command('calibration_select',body);motorCatalog=await api('calibration');calibrationChoices();
});
action('#cal-apply',()=>{const body={role:calibrationRole(),port:$('#cal-port').value,name:calibrationFile.name,supported:$('#cal-supported').checked};if(calibrationFile.id)body.file_id=calibrationFile.id;else body.calibration=calibrationFile.calibration;return command('calibration_apply',body)});
action('#cal-start',()=>command('calibration_start',{role:calibrationRole(),port:$('#cal-port').value,name:$('#cal-name').value.trim(),supported:$('#cal-supported').checked}));
action('#cal-next',()=>command('calibration_command',{action:{MIDPOINT:'middle',RANGE:'review',REVIEW:'save'}[state.motor_calibration.state]}));
action('#cal-resume',()=>command('calibration_command',{action:'resume'}));
action('#cal-cancel',()=>command('calibration_command',{action:'cancel'}));
calibrationDialog().addEventListener('close',()=>{if(state.motor_calibration?.busy)command('calibration_command',{action:'cancel'}).catch(e=>error(e.message,'motor-calibration'))});
