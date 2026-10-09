'use strict';
// Keep native values/options and change handlers as the source of truth. Only the
// visible control is replaced, so recording and hardware commands stay unchanged.
const StudioSelect=(()=>{
 const pickers=new Map();let active=null;
 const text=(className,value='')=>{const el=document.createElement('span');el.className=className;el.textContent=value;return el;};
 function close(restoreFocus=false){
  if(!active)return;
  const picker=active;active=null;
  picker.list.hidePopover();picker.toggle.setAttribute('aria-expanded','false');
  if(restoreFocus&&!picker.toggle.disabled)picker.toggle.focus({preventScroll:true});
 }
 function position(){
  if(!active)return;
  const {toggle,list}=active,rect=toggle.getBoundingClientRect();
  if(!toggle.getClientRects().length||rect.bottom<0||rect.top>innerHeight){close();return;}
  const header=document.querySelector('.app-header');
  const top=toggle.closest('dialog')?12:Math.max(12,header?.getBoundingClientRect().bottom||0);
  const below=innerHeight-rect.bottom-16,above=rect.top-top-16,upward=below<200&&above>below;
  list.style.width=Math.min(rect.width,innerWidth-24)+'px';
  list.style.left=Math.max(12,Math.min(rect.left,innerWidth-list.offsetWidth-12))+'px';
  list.style.maxHeight=Math.max(44,Math.min(320,upward?above:below))+'px';
  list.style.top=upward?'auto':(rect.bottom+8)+'px';
  list.style.bottom=upward?(innerHeight-rect.top+8)+'px':'auto';
 }
 function focusOption(picker,option){
  if(!option)return;
  option.focus({preventScroll:true});
  const rect=option.getBoundingClientRect(),view=picker.list.getBoundingClientRect();
  if(rect.top<view.top+7)picker.list.scrollTop-=view.top+7-rect.top;
  else if(rect.bottom>view.bottom-7)picker.list.scrollTop+=rect.bottom-(view.bottom-7);
 }
 function refreshOne(picker){
  const {select,toggle,list,name,detail}=picker,selected=select.selectedOptions[0];
  name.textContent=selected?.text||select.dataset.empty||'선택할 항목이 없습니다';
  detail.textContent=selected?.dataset.detail||'';
  toggle.title=select.title||selected?.text||'';
  toggle.disabled=select.disabled||![...select.options].some(option=>!option.disabled);
  const options=[...select.options];
  const signature=JSON.stringify(options.map(o=>[o.value,o.text,o.dataset.detail,o.disabled]));
  if(signature!==picker.signature){
   const focused=list.contains(document.activeElement)?document.activeElement.dataset.value:null;
   const scrollTop=list.scrollTop;
   list.replaceChildren();
   for(const source of options){
    const option=document.createElement('button');option.type='button';option.role='option';option.tabIndex=-1;
    option.dataset.value=source.value;option.disabled=source.disabled;
    option.append(text('select-option-name',source.text),text('select-option-detail',source.dataset.detail||''));
    const mark=text('select-option-check','✓');mark.setAttribute('aria-hidden','true');option.append(mark);
    option.onclick=()=>{
     // Episode/device refresh can replace native options without changing labels.
     const current=[...select.options].find(o=>o.value===option.dataset.value);
     if(select.disabled||!current||current.disabled)return;
     const changed=select.value!==current.value;select.value=current.value;
     close(true);refreshOne(picker);
     if(changed){select.dispatchEvent(new Event('input',{bubbles:true}));select.dispatchEvent(new Event('change',{bubbles:true}));}
    };
    list.append(option);
   }
   picker.signature=signature;
   if(focused!==null&&active===picker){
    const next=[...list.children].find(o=>o.dataset.value===focused&&!o.disabled)||list.querySelector('button:not(:disabled)');
    next?.focus({preventScroll:true});
   }
   list.scrollTop=scrollTop;
  }
  for(const option of list.children)option.setAttribute('aria-selected',String(option.dataset.value===select.value));
  if(active===picker){if(toggle.disabled)close();else position();}
 }
 function open(picker,edge){
  refreshOne(picker);if(picker.toggle.disabled)return;
  close();active=picker;picker.search='';
  // Popovers live above panel/dialog clipping, using the same dark app theme.
  picker.list.showPopover();picker.toggle.setAttribute('aria-expanded','true');position();
  const options=[...picker.list.children].filter(o=>!o.disabled);
  focusOption(picker,edge==='first'?options[0]:edge==='last'?options.at(-1):options.find(o=>o.getAttribute('aria-selected')==='true')||options[0]);
 }
 function enhance(select){
  const labels=[...select.labels],label=select.getAttribute('aria-label')||labels.map(l=>[...l.childNodes].filter(n=>n.nodeType===Node.TEXT_NODE).map(n=>n.textContent).join(' ').trim()).join(' ')||'항목 선택';
  const wrapper=document.createElement('span');wrapper.className='app-select';wrapper.dataset.select=select.id;
  const toggle=document.createElement('button');toggle.type='button';toggle.className='select-toggle';toggle.id=select.id+'-toggle';
  const choice=text('select-choice'),name=text('select-name'),detail=text('select-detail');
  name.id=select.id+'-name';detail.id=select.id+'-detail';choice.append(name,detail);toggle.append(choice);
  const labelText=text('sr-only',label);labelText.id=select.id+'-label-text';wrapper.append(labelText);
  toggle.setAttribute('aria-labelledby',labelText.id+' '+name.id);toggle.setAttribute('aria-describedby',detail.id);
  toggle.setAttribute('aria-haspopup','listbox');toggle.setAttribute('aria-expanded','false');
  const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.setAttribute('viewBox','0 0 16 16');svg.setAttribute('aria-hidden','true');
  const path=document.createElementNS(svg.namespaceURI,'path');path.setAttribute('d','m4 6 4 4 4-4');svg.append(path);toggle.append(svg);
  const list=document.createElement('div');list.className='select-menu';list.id=select.id+'-options';list.role='listbox';list.popover='manual';list.setAttribute('aria-label',label);
  toggle.setAttribute('aria-controls',list.id);
  select.before(wrapper);wrapper.append(select,toggle);select.hidden=true;
  // A modal's list remains in its focus scope while the popover escapes its edges.
  (select.closest('dialog')||document.body).append(list);
  labels.forEach(l=>l.htmlFor=toggle.id);
  select.focus=options=>toggle.focus(options);
  const picker={select,wrapper,toggle,list,name,detail,signature:null,search:'',searchTime:0};pickers.set(select,picker);
  toggle.onclick=()=>active===picker?close():open(picker);
  toggle.addEventListener('keydown',event=>{
   if(['ArrowDown','ArrowUp','Home','End'].includes(event.key)){
    event.preventDefault();open(picker,event.key==='Home'?'first':event.key==='End'?'last':undefined);
   }
  });
  list.addEventListener('keydown',event=>{
   const options=[...list.children].filter(o=>!o.disabled),index=options.indexOf(document.activeElement);
   if(['ArrowDown','ArrowUp','Home','End'].includes(event.key)){
    event.preventDefault();const next=event.key==='Home'?0:event.key==='End'?options.length-1:Math.max(0,Math.min(options.length-1,index+(event.key==='ArrowDown'?1:-1)));
    focusOption(picker,options[next]);
   }else if(event.key==='Tab'){
    // Continue the form's tab order from its trigger, not from the portal.
    close(true);
   }else if(event.key.length===1&&event.key!==' '&&!event.ctrlKey&&!event.altKey&&!event.metaKey&&!event.isComposing){
    event.preventDefault();const now=Date.now();picker.search=(now-picker.searchTime<700?picker.search:'')+event.key.toLocaleLowerCase();picker.searchTime=now;
    const ordered=[...options.slice(index+1),...options.slice(0,index+1)];
    focusOption(picker,ordered.find(o=>o.querySelector('.select-option-name').textContent.toLocaleLowerCase().startsWith(picker.search)));
   }
  });
  select.addEventListener('change',()=>refreshOne(picker));
  new MutationObserver(()=>refreshOne(picker)).observe(select,{subtree:true,childList:true,characterData:true,attributes:true,attributeFilter:['disabled','label','value','selected','title','data-detail']});
  refreshOne(picker);
 }
 document.addEventListener('keydown',event=>{
  if(event.key==='Escape'&&active){event.preventDefault();event.stopPropagation();close(true);}
 },true);
 document.addEventListener('pointerdown',event=>{if(active&&!active.wrapper.contains(event.target)&&!active.list.contains(event.target))close();});
 document.addEventListener('focusin',event=>{if(active&&!active.wrapper.contains(event.target)&&!active.list.contains(event.target))close();});
 window.addEventListener('resize',position);
 document.addEventListener('scroll',event=>{if(active&&event.target!==active.list)position();},true);
 document.querySelectorAll('dialog').forEach(dialog=>dialog.addEventListener('close',()=>{if(active?.select.closest('dialog')===dialog)close();}));
 document.querySelectorAll('select').forEach(enhance);
 return {refresh(select){if(select){const picker=pickers.get(select);if(picker)refreshOne(picker);}else pickers.forEach(refreshOne);},close};
})();
