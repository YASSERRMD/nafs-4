//! Textual Backpropagation Engine
//!
//! Generates textual gradients (fixes) from failure analysis.

use nafs_core::{Result, TextualGradient};
use std::collections::HashMap;

/// Analysis of failures
#[derive(Clone, Debug)]
pub struct FailureAnalysis {
    /// Total count of failures
    pub failure_count: usize,
    /// Most frequent error type
    pub most_common_error: String,
    /// Distribution of error types
    pub error_categories: HashMap<String, usize>,
}

/// Engine for conducting textual backpropagation
pub struct TextualBackpropEngine;

impl TextualBackpropEngine {
    /// Create new engine
    pub fn new() -> Self {
        Self
    }

    /// Generate textual gradients from failure analysis
    pub async fn backpropagate(
        &self,
        analysis: &FailureAnalysis,
    ) -> Result<Vec<TextualGradient>> {
        tracing::info!(
            "Generating gradients for {} failures",
            analysis.failure_count
        );

        let mut gradients = vec![];

        // For each error category, generate a gradient
        for (error_type, count) in &analysis.error_categories {
            let gradient = TextualGradient {
                id: uuid::Uuid::new_v4().to_string(),
                failed_action: format!("Action failed due to: {}", error_type),
                root_cause: format!(
                    "The agent encountered {} errors of type '{}'. Analysis shows this is a systematic issue.",
                    count, error_type
                ),
                suggested_fix: format!(
                    "Improve handling of {} by: 1) Understanding the error better, 2) Adding validation before attempting, 3) Having fallback strategies.",
                    error_type
                ),
                target_module: "system2".to_string(), // Reasoning layer
                target_field: "system_prompt".to_string(),
                confidence: 0.75,
                impact_estimate: (*count as f32) / (analysis.failure_count.max(1) as f32),
            };

            gradients.push(gradient);
        }

        Ok(gradients)
    }
}

impl Default for TextualBackpropEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_backprop_generation() {
        let engine = TextualBackpropEngine::new();

        let mut error_categories = HashMap::new();
        error_categories.insert("timeout".to_string(), 5);
        error_categories.insert("invalid_input".to_string(), 3);

        let analysis = FailureAnalysis {
            failure_count: 8,
            most_common_error: "timeout".to_string(),
            error_categories,
        };

        let gradients = engine.backpropagate(&analysis).await.unwrap();
        assert_eq!(gradients.len(), 2);
    }
}
