#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import ort from 'onnxruntime-node';

function loadJson(p) { return JSON.parse(fs.readFileSync(p, 'utf8')); }

function encode(tokens, word2id, char2id, maxChars=24) {
  const t = tokens.length;
  const wordIds = new BigInt64Array(1 * t);
  const charIds = new BigInt64Array(1 * t * maxChars);
  for (let i = 0; i < t; i++) {
    wordIds[i] = BigInt(word2id[tokens[i]] ?? 1);
    for (let j = 0; j < Math.min(tokens[i].length, maxChars - 1); j++) {
      const ch = tokens[i][j];
      charIds[i * maxChars + j] = BigInt(char2id[ch] ?? 2);
    }
  }
  return { wordIds, charIds, t, maxChars };
}

async function main() {
  const argv = process.argv.slice(2);
  let modelPath = 'artifacts/onnx/model.onnx';
  let resDir = 'data/processed';
  let preferLex = false;
  const args = [];
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--model' && argv[i+1]) { modelPath = argv[i+1]; i++; continue; }
    if (argv[i] === '--resources' && argv[i+1]) { resDir = argv[i+1]; i++; continue; }
    if (argv[i] === '--prefer-lexicon') { preferLex = true; continue; }
    args.push(argv[i]);
  }
  if (args.length === 0) {
    console.error('usage: node examples/nodejs/analyze.mjs [--model path] [--resources dir] <token> [<token> ...]');
    process.exit(1);
  }
  if (!fs.existsSync(path.join(resDir, 'tagset.json'))) {
    const alt = path.join(process.cwd(), '../../data/processed');
    if (fs.existsSync(path.join(alt, 'tagset.json'))) resDir = alt;
  }
  const tag2id = loadJson(path.join(resDir, 'tagset.json'));
  const word2id = loadJson(path.join(resDir, 'word_vocab.json'));
  const char2id = loadJson(path.join(resDir, 'char_vocab.json'));
  const id2tag = Object.entries(tag2id).sort((a,b)=>a[1]-b[1]).map(([k])=>k);
  const id2ch = Object.entries(char2id).sort((a,b)=>a[1]-b[1]).map(([k])=>k);
  let lemmaLex = {};
  const lexPath = path.join(resDir, 'lemma_lexicon.json');
  if (fs.existsSync(lexPath)) lemmaLex = loadJson(lexPath);

  const { wordIds, charIds, t, maxChars } = encode(args, word2id, char2id);

  if (!fs.existsSync(modelPath)) {
    const altModel = path.join(process.cwd(), '../../artifacts/onnx/model.onnx');
    if (fs.existsSync(altModel)) modelPath = altModel;
  }
  const session = await ort.InferenceSession.create(modelPath);
  const feeds = {
    word_ids: new ort.Tensor('int64', wordIds, [1, t]),
    char_ids: new ort.Tensor('int64', charIds, [1, t, maxChars])
  };
  const results = await session.run(feeds);
  const tagLogits = results['tag_logits'].data;
  const lemmaLogits = results['lemma_logits'].data;

  // Helper to argmax along last dim
  function argmaxRow(row, lastDim) {
    const out = new Int32Array(row.length / lastDim);
    for (let i = 0; i < out.length; i++) {
      let maxIdx = 0, maxVal = -Infinity;
      for (let j = 0; j < lastDim; j++) {
        const v = row[i*lastDim + j];
        if (v > maxVal) { maxVal = v; maxIdx = j; }
      }
      out[i] = maxIdx;
    }
    return out;
  }

  // tag logits: [1, t, K]
  const K = tagLogits.length / t;
  const tagIdx = argmaxRow(tagLogits, K);

  // lemma logits: [1, t, L, C]
  const LxC = lemmaLogits.length / t;
  // Find C by using maxChars assumption used in export (lemma_len=24 default). We'll infer L and C by searching a factor near 24.
  let L = 24; let C = Math.floor(LxC / L);
  const lemmaIdx = new Array(t);
  const preferLex = preferLex || process.env.PREFER_LEXICON === '1' || process.env.PREFER_LEXICON === 'true';
  for (let i = 0; i < t; i++) {
    const slice = lemmaLogits.subarray(i*L*C, (i+1)*L*C);
    lemmaIdx[i] = argmaxRow(slice, C);
  }

  for (let i = 0; i < t; i++) {
    const tag = id2tag[tagIdx[i]];
    const chars = [];
    for (const cid of lemmaIdx[i]) {
      if (cid === 1) break; // EOS
      chars.push(id2ch[cid] ?? '?');
    }
    const decoded = chars.length ? chars.join('') : args[i];
    const lemma = preferLex ? (lemmaLex[args[i]] ?? decoded) : (decoded || lemmaLex[args[i]] || decoded);
    console.log(`${args[i]}\t${tag}\t${lemma}`);
  }
}

main().catch(e => { console.error(e); process.exit(1); });
