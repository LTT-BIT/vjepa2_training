// Synthetic DOM/localStorage unit test only: never writes real human reviews.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const html=fs.readFileSync('tools/tomato_operation_review.html','utf8');
const script=html.split('<script>')[1].split('</script>')[0];
const elements={},videos=[],cameras=['cam0','cam1','cam2'].map(value=>({value,checked:false}));
function element(id){return elements[id]??=(Object.assign({value:'',textContent:'',checked:false}, {
 replaceChildren(){if(id==='videos')videos.length=0},append(v){if(id==='videos')videos.push(v)},click(){}
}))}
const cases=[{case_id:'case-a',episode_id:'episode',event_type:'truss_stably_lifted',annotation_sha256:'sha',split:'train',suggested_frame:10,reference_transition_frame:30,reference_kind:'stage_transition'},
{case_id:'case-b',episode_id:'episode',event_type:'truss_stably_lifted',annotation_sha256:'sha',split:'train',suggested_frame:10,reference_transition_frame:30,reference_kind:'stage_transition'}];
const catalog=[{episode_id:'episode',fps:30,length:90,videos:['one','two','three']}];
const data={'tomato-operation-review:current':JSON.stringify({'case-a':{case_id:'case-a',reviewer:'fixture-A',annotation_sha256:'sha',evidence_note:'Synthetic old A'}}),
 'tomato-operation-review:past':JSON.stringify({'case-b':{case_id:'case-b',reviewer:'fixture-B',annotation_sha256:'sha',evidence_note:'Synthetic old B'}})};
const ctx=vm.createContext({console,Map,Object,Number,Math,JSON,encodeURIComponent,URL,Blob,
 document:{getElementById:element,createElement:()=>({currentTime:0,pause(){},play(){}}),querySelectorAll:s=>s==='video'?videos:s==='.camera'?cameras:s==='.camera:checked'?cameras.filter(c=>c.checked):[]},
 localStorage:{getItem:k=>data[k]??null,setItem:(k,v)=>data[k]=v},
 fetch:async p=>({json:async()=>p==='cases.json'?cases:p==='catalog.json'?catalog:{catalog_sha256:'current',previous_catalog_sha256:['past']}})});
(async()=>{
 await vm.runInContext(script,ctx);
 assert.equal(element('evidence').value,'');assert.equal(element('verdict').value,'');
 element('reviewer').value='fixture-A';element('workspace').onclick();
 assert.deepEqual(Array.from(vm.runInContext('Object.keys(store)',ctx)),['case-a']);
 assert.equal(element('evidence').value,''); // Legacy timestamp is not silently promoted.
 element('verdict').value='positive';element('note').value='Synthetic note';cameras[0].checked=true;
 element('save').onclick();assert.match(element('message').textContent,/首次证据帧/);
 element('reviewer').value='fixture-B';element('workspace').onclick();
 assert.deepEqual(Array.from(vm.runInContext('Object.keys(store)',ctx)),['case-b']);
 assert.ok(data['tomato-operation-review:current']);assert.ok(data['tomato-operation-review:past']);
 assert.notEqual(data['tomato-operation-review:current:reviewer:fixture-A'],data['tomato-operation-review:current:reviewer:fixture-B']);
 console.log('PASS: synthetic workspace isolation, preserved legacy opinions, explicit first evidence guard; no real reviews created');
})().catch(e=>{console.error(e);process.exitCode=1});
