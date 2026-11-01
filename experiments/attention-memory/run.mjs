import {growth, illustrate, architecture} from './mechanics.mjs';
console.log(JSON.stringify({kind:'teaching illustration, not benchmark',growth: [1024,16384,262144].map(tokens => ({tokens,...growth(tokens)})),converted: architecture(true).filter(x=>x.kind==='converted').map(x=>x.index),toy:illustrate(6)}, null, 2));
