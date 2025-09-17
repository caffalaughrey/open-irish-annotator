use serde::{Deserialize, Serialize};
use thiserror::Error;

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
    // Placeholder for ONNX session and resources
}

impl MorphologyRuntime {
    pub fn new_from_resources(_model_path: &str, _resources_dir: &str) -> Result<Self, MorphError> {
        // TODO: load ONNX and resources (tagset, vocabs)
        Ok(Self {})
    }

    pub fn analyze(&self, tokens: Vec<String>) -> Result<Vec<TokenAnalysis>, MorphError> {
        // Stub: echo tokens with placeholder tag/lemma
        let results = tokens
            .into_iter()
            .map(|t| TokenAnalysis {
                token: t.clone(),
                tag: "X".to_string(),
                lemma: t,
            })
            .collect();
        Ok(results)
    }
}


