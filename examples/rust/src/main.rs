use std::{collections::HashMap, fs, path::PathBuf};
use tract_onnx::prelude::*;

fn read_json_map(path: &str) -> HashMap<String, usize> {
    let s = fs::read_to_string(path).expect("read json");
    serde_json::from_str(&s).expect("parse json")
}

fn main() -> TractResult<()> {
    let mut args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.is_empty() {
        eprintln!("usage: cargo run -- [--model path] [--resources dir] [--prefer-lexicon] <token> ...");
        std::process::exit(1);
    }
    let mut model_path = String::from("artifacts/onnx/model.onnx");
    let mut res_dir = String::from("rust/morphology_runtime/resources");
    let mut prefer_lex = false;
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--model" && i + 1 < args.len() { model_path = args[i+1].clone(); args.drain(i..=i+1); continue; }
        if args[i] == "--resources" && i + 1 < args.len() { res_dir = args[i+1].clone(); args.drain(i..=i+1); continue; }
        if args[i] == "--prefer-lexicon" { prefer_lex = true; args.drain(i..=i); continue; }
        i += 1;
    }
    let tokens = args;

    let tag2id = read_json_map(&format!("{}/tagset.json", res_dir));
    let word2id = read_json_map(&format!("{}/word_vocab.json", res_dir));
    let char2id = read_json_map(&format!("{}/char_vocab.json", res_dir));
    let mut id2tag = vec![String::new(); tag2id.len()];
    for (k, v) in tag2id { if v < id2tag.len() { id2tag[v] = k } }
    let mut id2ch = vec![String::new(); char2id.len()];
    let mut ch2id = std::collections::HashMap::<char, i64>::new();
    for (k, v) in char2id { if v < id2ch.len() { id2ch[v] = k.clone(); if let Some(ch) = k.chars().next() { ch2id.insert(ch, v as i64); } } }

    // Optional lemma lexicon
    let mut lemma_lex: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    if prefer_lex {
        let p = format!("{}/lemma_lexicon.json", res_dir);
        if let Ok(s) = fs::read_to_string(&p) { if let Ok(m) = serde_json::from_str(&s) { lemma_lex = m } }
    }

    let model = tract_onnx::onnx()
        .model_for_path(&model_path)?
        .with_input_names(vec!["word_ids", "char_ids"])?.into_optimized()?.into_runnable()?;

    let t = tokens.len();
    let max_chars = 24usize;
    let mut word_ids = tract_ndarray::Array2::<i64>::zeros((1, t));
    let mut char_ids = tract_ndarray::Array3::<i64>::zeros((1, t, max_chars));
    for (i, tok) in tokens.iter().enumerate() {
        let wid = word2id.get(tok).map(|u| *u as i64).unwrap_or(1);
        word_ids[[0, i]] = wid;
        for (j, ch) in tok.chars().take(max_chars - 1).enumerate() {
            let cid = *ch2id.get(&ch).unwrap_or(&2);
            char_ids[[0, i, j]] = cid;
        }
    }

    let out = model.run(tvec!(word_ids.into_tensor().into(), char_ids.into_tensor().into()))?;
    let tag_logits: tract_ndarray::ArrayD<f32> = out[0].to_array_view::<f32>()?.to_owned();
    let lemma_logits: tract_ndarray::ArrayD<f32> = out[1].to_array_view::<f32>()?.to_owned();
    let k = tag_logits.shape()[2] as usize;
    let l = lemma_logits.shape()[2] as usize;
    let c = lemma_logits.shape()[3] as usize;

    for i in 0..t {
        // tag
        let mut max_v = f32::NEG_INFINITY; let mut max_j = 0usize;
        for j in 0..k { let v = tag_logits[[0, i, j]]; if v > max_v { max_v = v; max_j = j; } }
        let tag = &id2tag[max_j];
        // lemma
        let mut lemma = String::new();
        for step in 0..l { let mut mv = f32::NEG_INFINITY; let mut mj = 0usize; for j in 0..c { let v = lemma_logits[[0, i, step, j]]; if v > mv { mv = v; mj = j; } } if mj == 1 { break; } let ch = &id2ch[mj]; if !ch.is_empty() { lemma.push(ch.chars().next().unwrap()); } }
        if prefer_lex { if let Some(best) = lemma_lex.get(&tokens[i]) { lemma = best.clone(); } }
        if lemma.is_empty() { lemma = tokens[i].clone(); }
        println!("{}\t{}\t{}", tokens[i], tag, lemma);
    }
    Ok(())
}
