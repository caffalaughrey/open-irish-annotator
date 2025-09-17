use serde::{Deserialize, Serialize};
use thiserror::Error;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[cfg(feature = "inference")]
use tract_onnx::prelude::*;

#[derive(Debug, Error)]
pub enum MorphError {
    #[error("resource not found: {0}")]
    ResourceNotFound(String),
    #[error("invalid model state: {0}")]
    InvalidState(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TokenAnalysis {
    pub token: String,
    pub tag: String,
    pub lemma: String,
}

#[derive(Debug)]
pub struct MorphologyRuntime {
    #[cfg(feature = "inference")]
    model: SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
    tag_id_to_str: Vec<String>,
    char_id_to_ch: Vec<char>,
    word_str_to_id: HashMap<String, i64>,
    lemma_lexicon: HashMap<String, String>,
    char_to_id: HashMap<char, i64>,
}

impl MorphologyRuntime {
    pub fn new_from_resources(model_path: &str, resources_dir: &str) -> Result<Self, MorphError> {
        #[cfg(feature = "inference")]
        let model = tract_onnx::onnx()
            .model_for_path(model_path)
            .map_err(|_| MorphError::ResourceNotFound(model_path.to_string()))?
            .with_input_names(vec!["word_ids", "char_ids"]).map_err(|e| MorphError::InvalidState(format!("names: {e}")))?
            .into_optimized().map_err(|e| MorphError::InvalidState(format!("opt: {e}")))?
            .into_runnable().map_err(|e| MorphError::InvalidState(format!("run: {e}")))?;

        // Load resources
        let tagset_path = Path::new(resources_dir).join("tagset.json");
        let word_vocab_path = Path::new(resources_dir).join("word_vocab.json");
        let char_vocab_path = Path::new(resources_dir).join("char_vocab.json");
        let lemma_lexicon_path = Path::new(resources_dir).join("lemma_lexicon.json");
        let tag_map: HashMap<String, i64> = serde_json::from_str(
            &fs::read_to_string(&tagset_path).map_err(|_| MorphError::ResourceNotFound(tagset_path.display().to_string()))?,
        )
        .map_err(|e| MorphError::InvalidState(format!("parse tagset: {e}")))?;
        let mut tag_pairs: Vec<(String, i64)> = tag_map.into_iter().collect();
        tag_pairs.sort_by_key(|(_, id)| *id);
        let tag_id_to_str: Vec<String> = tag_pairs.into_iter().map(|(s, _)| s).collect();

        let word_map: HashMap<String, i64> = serde_json::from_str(
            &fs::read_to_string(&word_vocab_path).map_err(|_| MorphError::ResourceNotFound(word_vocab_path.display().to_string()))?,
        )
        .map_err(|e| MorphError::InvalidState(format!("parse word_vocab: {e}")))?;

        let char_map: HashMap<String, i64> = serde_json::from_str(
            &fs::read_to_string(&char_vocab_path).map_err(|_| MorphError::ResourceNotFound(char_vocab_path.display().to_string()))?,
        )
        .map_err(|e| MorphError::InvalidState(format!("parse char_vocab: {e}")))?;
        // Build id->char; vocab keys are single characters or special tokens
        let mut char_pairs: Vec<(String, i64)> = char_map.into_iter().collect();
        char_pairs.sort_by_key(|(_, id)| *id);
        let mut char_id_to_ch: Vec<char> = Vec::with_capacity(char_pairs.len());
        let mut char_to_id: HashMap<char, i64> = HashMap::new();
        for (k, _) in char_pairs {
            let ch = k.chars().next().unwrap_or('?');
            char_id_to_ch.push(ch);
            let id = (char_id_to_ch.len() - 1) as i64;
            char_to_id.insert(ch, id);
        }

        // Optional lemma lexicon
        let lemma_lexicon: HashMap<String, String> = if lemma_lexicon_path.exists() {
            serde_json::from_str(&fs::read_to_string(&lemma_lexicon_path).unwrap_or_default()).unwrap_or_default()
        } else {
            HashMap::new()
        };

        Ok(Self {
            #[cfg(feature = "inference")]
            model,
            tag_id_to_str,
            char_id_to_ch,
            word_str_to_id: word_map,
            lemma_lexicon,
            char_to_id,
        })
    }

    pub fn analyze(&self, tokens: Vec<String>) -> Result<Vec<TokenAnalysis>, MorphError> {
        #[cfg(not(feature = "inference"))]
        {
            // Fallback: echo tokens with placeholders
            return Ok(tokens
                .into_iter()
                .map(|t| TokenAnalysis { token: t.clone(), tag: "X".to_string(), lemma: t })
                .collect());
        }

        #[cfg(feature = "inference")]
        {
        // Encode tokens
        let pad_char_id: i64 = 0;
        let max_chars: usize = 24;
        let tlen = tokens.len();
        let mut word_ids: Vec<i64> = Vec::with_capacity(tlen);
        let mut char_ids: Vec<i64> = Vec::with_capacity(tlen * max_chars);
        for tok in &tokens {
            let wid = *self.word_str_to_id.get(tok).unwrap_or(&1); // 1 = <unk>
            word_ids.push(wid);
            let mut chbuf = vec![pad_char_id; max_chars];
            let mut i = 0;
            for ch in tok.chars() {
                if i >= max_chars - 1 {
                    break;
                }
                // map char to id with hashmap, default to <unk>=2
                let cid = *self.char_to_id.get(&ch).unwrap_or(&2);
                chbuf[i] = cid;
                i += 1;
            }
            char_ids.extend_from_slice(&chbuf);
        }

        use tract_onnx::prelude::IntoTensor;
        let input_word: Tensor = tract_ndarray::Array2::<i64>::from_shape_vec((1, tlen), word_ids)
            .unwrap()
            .into_tensor();
        let char_tensor: Tensor = tract_ndarray::Array3::<i64>::from_shape_vec((1, tlen, max_chars), char_ids)
            .unwrap()
            .into_tensor();
        let outputs = self.model.run(tvec!(input_word.into(), char_tensor.into()))
            .map_err(|e| MorphError::InvalidState(format!("onnx run: {e}")))?;

        // Outputs as tract tensors -> ndarray
        let tag_logits: tract_ndarray::ArrayD<f32> = outputs[0].to_array_view::<f32>().unwrap().to_owned();
        let lemma_logits: tract_ndarray::ArrayD<f32> = outputs[1].to_array_view::<f32>().unwrap().to_owned();

        // Argmax over last dims
        let k = tag_logits.shape()[2] as usize;
        let l = lemma_logits.shape()[2] as usize;
        let c = lemma_logits.shape()[3] as usize;
        let mut results: Vec<TokenAnalysis> = Vec::with_capacity(tlen);
        for i in 0..tlen {
            // tag id
            let mut max_v = f32::NEG_INFINITY;
            let mut max_j = 0usize;
            for j in 0..k {
                let v = tag_logits[[0, i, j]];
                if v > max_v {
                    max_v = v;
                    max_j = j;
                }
            }
            let tag = self
                .tag_id_to_str
                .get(max_j)
                .cloned()
                .unwrap_or_else(|| "X".to_string());

            // lemma ids -> string up to EOS (id 1)
            let mut lemma_s = String::new();
            for t in 0..l {
                // pick best char
                let mut max_vc = f32::NEG_INFINITY;
                let mut max_c = 0usize;
                for j in 0..c {
                    let v = lemma_logits[[0, i, t, j]];
                    if v > max_vc {
                        max_vc = v;
                        max_c = j;
                    }
                }
                if max_c == 1 {
                    // EOS
                    break;
                }
                let ch = *self.char_id_to_ch.get(max_c).unwrap_or(&'?');
                if ch != '\0' {
                    lemma_s.push(ch);
                }
            }
            // Prefer training lexicon if available; else keep decoded lemma (trimmed)
            if let Some(best) = self.lemma_lexicon.get(&tokens[i]) {
                lemma_s = best.clone();
            }

            results.push(TokenAnalysis {
                token: tokens[i].clone(),
                tag,
                lemma: if lemma_s.is_empty() { tokens[i].clone() } else { lemma_s },
            });
        }
        Ok(results)
        }
    }
}
