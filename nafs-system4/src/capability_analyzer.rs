//! Capability Analyzer
//!
//! Detects skill gaps and recommends learning targets.

use nafs_core::{CapabilityGap, Goal, RecordedFailure, Result, SelfModel};

/// Analyzer for detecting capability gaps
pub struct CapabilityAnalyzer;

impl CapabilityAnalyzer {
    /// Create new analyzer
    pub fn new() -> Self {
        Self
    }

    /// Detect capability gaps from failures and self-model
    pub fn detect_gaps(
        &self,
        self_model: &SelfModel,
        failures: &[RecordedFailure],
    ) -> Result<Vec<CapabilityGap>> {
        tracing::info!("Analyzing capability gaps from {} failures", failures.len());

        let mut gaps = vec![];

        // Map failures to skills
        for failure in failures {
            if failure.error_message.contains("timeout")
                && !self_model.capabilities.contains_key("speed")
            {
                let gap = CapabilityGap {
                    id: uuid::Uuid::new_v4().to_string(),
                    missing_skill: "execution_speed".to_string(),
                    impact_severity: 0.6,
                    learning_objective: Goal {
                        id: uuid::Uuid::new_v4().to_string(),
                        description: "Improve execution speed and responsiveness".to_string(),
                        priority: 8,
                        deadline: None,
                        success_criteria: vec!["Reduce timeout errors by 50%".to_string()],
                    },
                    recommended_training: vec![
                        "Optimize code paths".to_string(),
                        "Use caching more effectively".to_string(),
                    ],
                    estimated_learning_time: 7,
                };
                gaps.push(gap);
            }
        }

        Ok(gaps)
    }
}

impl Default for CapabilityAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gap_detection() {
        let analyzer = CapabilityAnalyzer::new();

        let failures = vec![RecordedFailure {
            id: "f1".to_string(),
            timestamp: chrono::Utc::now(),
            action_attempted: "process_data".to_string(),
            error_message: "timeout".to_string(),
            context: Default::default(),
            severity: 8,
        }];

        let self_model = SelfModel {
            agent_id: "test".to_string(),
            identity: "Test".to_string(),
            capabilities: Default::default(),
            weaknesses: vec!["speed".to_string()],
            terminal_values: vec![],
            personality: Default::default(),
        };

        let gaps = analyzer.detect_gaps(&self_model, &failures).unwrap();
        // Should detect 'speed' gap due to 'timeout' error
        assert!(gaps.len() >= 1);
        assert_eq!(gaps[0].missing_skill, "execution_speed");
    }
}
