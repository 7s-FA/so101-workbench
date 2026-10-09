'use strict';
// Pure state descriptions: no motor command or hidden state changes here.
const StudioControls = (() => {
 const reviewing=s=>['idle','holding','preview'].includes(s.mode);
 const cameraReady=c=>!!c?.connected&&!c.error&&c.age!=null&&c.age<=.2;
 function describe(s,tab='record') {
  const connected=!!s.connected, powered=!!s.torque?.release_required||!!s.armed;
  const cameras=['gripper','table'].filter(k=>cameraReady(s.cameras?.[k])).length;
  const motor=!connected?'확인 전':s.mode==='fault'||s.torque?.state==='unknown'?'상태 확인 필요':powered?'켜짐':'꺼짐';
  let reason=!connected?'장치 연결에서 작업용 팔을 연결하세요.':!s.leader?'조작용 팔도 연결해야 리더 따라가기를 사용할 수 있습니다.':!reviewing(s)?'정지 (힘 유지)를 누르면 리더 따라가기를 다시 시작할 수 있습니다.':'';
  if(s.test_mode)reason='화면 확인 모드입니다. 실물 조작은 꺼져 있습니다.';
  let next='',action='';
  if(s.mode==='fault'){next='팔 제어가 중단됐습니다. 팔을 받친 뒤 모터 힘을 끄세요. 해제에 실패하면 USB·전원을 확인하세요.';action='#release-motors';}
  else if(s.storage_busy)next='기록을 저장하고 있습니다. 완료되면 다음 녹화를 시작할 수 있습니다.';
  else if(['checking','preparing'].includes(s.mode))next='작업용 팔을 시작 자세로 맞추고 있습니다. 정지하면 현재 위치에서 멈춥니다.';
  else if(s.mode==='recording')next='녹화 중입니다. R로 녹화를 끝내고 S로 저장하세요. 정지하면 팔 조작도 멈춥니다.';
  else if(s.mode==='replay'||s.mode==='model')next='실제 팔이 움직이고 있습니다. F8은 정지·힘 유지, 모터 힘 끄기는 받침 확인 후 힘 해제입니다.';
  else if(s.draft){next=s.recording_interruption?'끊긴 영상이 있어 저장할 수 없습니다. 이번 녹화를 버린 뒤 다시 시작하세요.':'저장하지 않은 녹화가 있습니다. 저장 또는 버리기를 선택하세요.';action=s.recording_interruption?'#discard':'#save';}
  else if(tab==='train'){next=powered?'학습을 시작하려면 모터 힘을 먼저 끄세요. 팔 연결과 카메라는 유지됩니다.':'작업 폴더와 모델 이름을 정하고 학습을 시작하세요. 실물 장치가 없어도 학습할 수 있습니다.';action=powered?'#release-motors':'';}
  else if(tab==='model'){next=powered?'모델을 불러오기 전에 모터 힘을 꺼 주세요. 불러온 뒤 실물에서 실행을 누릅니다.':'모델 선택 → 모델 불러오기 → 실물에서 실행 순서입니다. 불러오기만으로 팔은 움직이지 않습니다.';action=powered?'#release-motors':'';}
  else if(tab==='episodes'){next=s.mode==='teleop'?'기록을 보려면 리더 따라가기를 정지하세요. 모터 힘은 유지됩니다.':'영상·시뮬 재생은 확인용입니다. 실물 재실행은 팔을 움직이며 카메라 2대가 필요합니다.';action=s.mode==='teleop'?'#global-stop':'';}
  else if(s.mode==='teleop'&&!s.task_id){next='녹화할 작업 폴더를 선택하세요. 폴더 관리에서 새로 만들 수 있습니다.';action='#manage-folders';}
  else if(s.mode==='teleop'){next=cameras<2?'팔은 리더를 따라갑니다. 녹화에는 카메라 2대의 정상 영상이 필요합니다.':'리더를 움직여 연습하고, 준비되면 녹화를 시작하세요.';action=cameras===2?'#record':'#devices-button';}
  else if(!connected){next='장치 연결에서 작업용 팔·조작용 팔·카메라를 선택하세요. 연결만으로 팔이 움직이지 않습니다.';action='#devices-button';}
  else if(!s.leader){next='작업용 팔은 연결됐습니다. 손으로 시연하려면 조작용 팔도 연결하세요. 기록·모델의 실물 실행에는 리더가 필요 없습니다.';action='#devices-button';}
  else {next=powered?'정지 상태이며 모터 힘은 켜져 있습니다. 리더 따라가기로 재개하거나 모터 힘을 끄세요.':'연결됐습니다. 리더 따라가기를 누르면 모터 힘이 켜지고 작업용 팔이 움직입니다.';action='#enable-leader';}
  return {powered,cameras,motor,next,action,teleopReason:reason,
   recordReason:!s.task_id?'녹화할 작업 폴더를 먼저 선택하세요.':s.draft?'이번 녹화를 저장하거나 버려 주세요.':s.mode!=='teleop'?'리더 따라가기를 먼저 시작하세요.':cameras<2?'그리퍼·작업대 카메라의 영상이 모두 필요합니다.':''};
 }
 function folderReasons(s,tasks,id,rawName){
  const name=rawName.trim(),chosen=tasks.find(t=>t.id===id);
  const invalid=!name?'폴더 이름을 입력하세요.':name==='미분류'?'다른 폴더 이름을 입력하세요.':name.length>80||new TextEncoder().encode(name).length>240?'폴더 이름이 너무 깁니다.':name.startsWith('.')||/[\\/\x00-\x1f]/.test(name)?'폴더 이름에 경로나 제어 문자를 사용할 수 없습니다.':'';
  const busy=s.storage_busy?'기록 저장이 끝나면 변경할 수 있습니다.':'';
  return {
   create:busy||(!reviewing(s)&&s.mode!=='teleop'?'현재 작업을 정지한 뒤 만드세요.':'')||invalid||(tasks.some(t=>t.name===name)?'같은 이름의 폴더가 있습니다.':''),
   rename:busy||(!reviewing(s)?'팔 조작을 정지한 뒤 이름을 변경하세요.':'')||(s.dataset?.state==='writing'?'학습 파일 생성이 끝나면 이름을 변경할 수 있습니다.':'')||(!chosen?'이름을 바꿀 폴더를 선택하세요.':'')||invalid||(chosen?.name===name?'변경할 새 이름을 입력하세요.':'')||(tasks.some(t=>t.id!==id&&t.name===name)?'같은 이름의 폴더가 있습니다.':'')
  };
 }
 return {describe,reviewing,cameraReady,folderReasons};
})();
if(typeof module!=='undefined')module.exports=StudioControls;
