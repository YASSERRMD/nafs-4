//! Failure Analyzer
//!
//! Analyzes patterns in failures to guide evolution.

use nafs_core::{RecordedFailure, Result};
use std::collections::HashMap;

use crate::textual_backprop::FailureAnalysis;

/// Analyzer for identifying failure patterns
pub struct FailureAnalyzer {
    failures: Vec<RecordedFailure>,
}

impl FailureAnalyzer {
    /// Create new analyzer
    pub fn new() -> Self {
        Self {
            failures: vec![],
        }
    }

    /// Add a recorded failure
    pub fn add_failure(&mut self, failure: RecordedFailure) {
        self.failures.push(failure);
    }

    /// Analyze failure patterns
    pub fn analyze(&self, recent_failures: &[RecordedFailure]) -> Result<FailureAnalysis> {
        tracing::info!("Analyzing {} recent failures", recent_failures.len());

        if recent_failures.is_empty() {
            return Ok(FailureAnalysis {
                failure_count: 0,
                most_common_error: "none".to_string(),
                error_categories: HashMap::new(),
            });
        }

        let mut error_categories = HashMap::new();

        for failure in recent_failures {
            let category = self.categorize_error(&failure.error_message);
            *error_categories.entry(category).or_insert(0) += 1;
        }

        let most_common_error = error_categories
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(k, _)| k.clone())
            .unwrap_or_else(|| "unknown".to_string());

        Ok(FailureAnalysis {
            failure_count: recent_failures.len(),
            most_common_error,
            error_categories,
        })
    }

    fn categorize_error(&self, error_msg: &str) -> String {
        let msg_lower = error_msg.to_lowercase();
        if msg_lower.contains("timeout") {
            "timeout".to_string()
        } else if msg_lower.contains("invalid") {
            "invalid_input".to_string()
        } else if msg_lower.contains("permission") {
            "permission_denied".to_string()
        } else {
            "other".to_string()
        }
    }
}

impl Default for FailureAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    #[test]
    fn test_failure_analysis() {
        let analyzer = FailureAnalyzer::new();

        let failures = vec![
            RecordedFailure {
                id: "f1".to_string(),
                timestamp: Utc::now(),
                action_attempted: "action1".to_string(),
                error_message: "timeout".to_string(),
                context: Default::default(),
                severity: 5,
            },
            RecordedFailure {
                id: "f2".to_string(),
                timestamp: Utc::now(),
                action_attempted: "action2".to_string(),
                error_message: "timeout".to_string(),
                context: Default::default(),
                severity: 5,
            },
        ];

        let analysis = analyzer.analyze(&failures).unwrap();
        assert_eq!(analysis.failure_count, 2);
        assert_eq!(analysis.most_common_error, "timeout");
    }
}
