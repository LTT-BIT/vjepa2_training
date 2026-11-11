// Read-only archive export unit fixture; never accesses browser storage.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const html=fs.readFileSync('tools/tomato_review_paused.html','utf8'),script=html.split('<script>')[1].split('</script>')[0];
assert.ok(!html.includes('id="save"'));assert.ok(!script.includes('localStorage.setItem'));
const elements={},data={'tomato-operation-review:pilot:reviewer:a':JSON.stringify({'case':{case_id:'case',reviewer:'a',annotation_sha256:'sha',evidence_note:'synthetic old opinion'}}),'tomato-operation-review:pilot:reviewer:b':JSON.stringify({'case':{case_id:'case',reviewer:'b',annotation_sha256:'sha',evidence_note:'other independent synthetic opinion'}})};
const before=JSON.stringify(data);
const ctx=vm.createContext({console,Map,Object,JSON,encodeURIComponent,URL,Blob,document:{getElementById:id=>elements[id]??={value:'',textContent:''},createElement:()=>({click(){}})},localStorage:{getItem:k=>data[k]??null},fetch:async p=>({json:async()=>p==='identity.json'?{catalog_sha256:'pilot'}:[{case_id:'case',annotation_sha256:'sha'}]})});
(async()=>{await vm.runInContext(script,ctx);assert.equal(vm.runInContext('ownReviews("a")[0].reviewer',ctx),'a');assert.equal(vm.runInContext('ownReviews("missing").length',ctx),0);assert.equal(JSON.stringify(data),before);console.log('PASS: paused page has no annotation save; own archive readout preserves storage and reviewer isolation')})().catch(e=>{console.error(e);process.exitCode=1});
