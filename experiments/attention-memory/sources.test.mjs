import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
const load = name => JSON.parse(readFileSync(new URL('./sources/'+name,import.meta.url)));
test('source snapshots match hashes and architecture',()=>{
 for(const [name, meta] of Object.entries(load('provenance.json').files)) assert.equal(createHash('sha256').update(readFileSync(new URL('./sources/'+name,import.meta.url))).digest('hex'),meta.sha256);
 const base=load('qwen-config.json').text_config, converted=load('longbook-config.json'), report=load('longbook-report.json');
 assert.equal(base.num_hidden_layers,64);assert.equal(base.layer_types.filter(x=>x==='full_attention').length,16);
 assert.deepEqual(converted.converted_layers,base.layer_types.flatMap((x,i)=>x==='full_attention'?[i]:[]));
 assert.equal(converted.layer_types.filter(x=>x==='linear_attention').length,64);
 assert.equal(report.stage_b.native_lora,true);assert.equal(report.stage_b.lora_rank,32);assert.deepEqual(report.needle,[]);assert.deepEqual(report.config.needle_lengths,[]);assert.equal(Math.max(...report.config.cache_plan.map(([n])=>n)),16384);assert.equal(converted.max_position_embeddings,262144);
});
