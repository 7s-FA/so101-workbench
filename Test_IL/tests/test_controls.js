const {test}=require('node:test');const assert=require('node:assert/strict');
const {describe}=require('../web/controls.js');
const cam={connected:true,error:null,age:.03};
const state={task_id:'task',mode:'idle',connected:true,leader:true,torque:{state:'off',release_required:false},cameras:{gripper:cam,table:cam}};
test('connection never implies torque enabled',()=>{assert.equal(describe(state).motor,'꺼짐');assert.match(describe(state).next,/리더 따라가기/)});
test('holding distinguishes stop from release',()=>{const s={...state,mode:'holding',armed:true};assert.equal(describe(s).motor,'켜짐');assert.match(describe(s).next,/힘은 켜져/)});
test('failed communication cannot be shown as torque off',()=>{assert.equal(describe({...state,mode:'fault'}).motor,'상태 확인 필요')});
test('stale or failed camera prevents record readiness',()=>{assert.match(describe({...state,mode:'teleop',cameras:{gripper:{...cam,age:1},table:cam}}).recordReason,/카메라/)});
test('draft has priority and remains actionable across tabs',()=>{assert.equal(describe({...state,draft:'1'},'train').action,'#save');assert.equal(describe({...state,draft:'1',recording_interruption:{}},'model').action,'#discard')});
test('training and loading guide to release without disconnect',()=>{for(const tab of ['train','model'])assert.equal(describe({...state,armed:true},tab).action,'#release-motors')});
test('disconnected preview does not ask to connect to play',()=>{assert.match(describe({...state,connected:false},'episodes').next,/확인용/) });
test('missing folder gives an actionable recording reason without disabling practice',()=>{
 const s={...state,task_id:null,mode:'teleop'};
 assert.match(describe(s).recordReason,/폴더/);assert.equal(describe(s).action,'#manage-folders');
 assert.equal(describe({...s,mode:'idle'}).teleopReason,'');
});
const {folderReasons}=require('../web/controls.js');
test('folder actions explain blank, duplicate, unchanged and invalid names',()=>{
 const tasks=[{id:'a',name:'작업 A'},{id:'b',name:'작업 B'}];
 assert.match(folderReasons(state,tasks,'a','').create,/입력/);
 assert.match(folderReasons(state,tasks,'a','작업 A').create,/같은/);
 assert.match(folderReasons(state,tasks,'a','작업 A').rename,/새 이름/);
 assert.match(folderReasons(state,tasks,'a','작업 B').rename,/같은/);
 for(const name of [' .hidden','../escape','a/b','a\\b','미분류'])assert.ok(folderReasons(state,tasks,'a',name).create);
 assert.equal(folderReasons(state,tasks,'a','새 이름').rename,'');
});
test('new folders remain available during teleop and export; rename explains wait',()=>{
 const s={...state,mode:'teleop',dataset:{state:'writing'}};
 assert.equal(folderReasons(s,[],'','새 작업').create,'');
 assert.match(folderReasons({...s,mode:'idle'},[{id:'a',name:'작업'}],'a','변경').rename,/생성/);
 assert.ok(folderReasons({...s,storage_busy:true},[],'','새 작업').create);
});
