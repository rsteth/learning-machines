import {test} from 'node:test';
import assert from 'node:assert/strict';
import {growth, architecture, illustrate, EVENTS} from './mechanics.mjs';
test('growth boundaries and slopes', () => {
 assert.deepEqual(growth(0), {full:0,hybrid:48,recurrent:64});
 assert.deepEqual(growth(1024), {full:64,hybrid:64,recurrent:64});
 assert.deepEqual(growth(262144), {full:16384,hybrid:4144,recurrent:64});
 for(let n=0;n<262144;n+=1024) {assert.equal(growth(n+1024).full-growth(n).full,64);assert.equal(growth(n+1024).hybrid-growth(n).hybrid,16);assert.equal(growth(n).recurrent,64);}
 for(const n of [-1,262145,1.5,NaN,Infinity,'1024']) assert.throws(()=>growth(n),RangeError);
});
test('64 layers distinguish converted and later adapted layers', () => {
 assert.equal(architecture().filter(x=>x.kind==='attention').length,16);
 assert.deepEqual(architecture(true).filter(x=>x.kind==='converted').map(x=>x.index), Array.from({length:16},(_,i)=>3+4*i));
 assert.equal(architecture(true).filter(x=>x.kind==='adapted').length,48);
});
test('toy preserves history and overwrites only the current key', () => {
 assert.deepEqual(illustrate(0),{history:[],state:{}});
 assert.deepEqual(illustrate(3).state,{door:'green',cup:'red'});
 assert.deepEqual(illustrate(6).state,{door:'yellow',cup:'black'});
 assert.equal(illustrate(6).history[0].value,'blue');
 assert.deepEqual(illustrate(6),illustrate(6));
 const result=illustrate(1);result.history[0].value='changed';assert.equal(EVENTS[0].value,'blue');
 for(const n of [-1,7,0.5,NaN]) assert.throws(()=>illustrate(n),RangeError);
});
