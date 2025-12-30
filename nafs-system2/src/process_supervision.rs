//! Process Supervision: Self-Critique Mechanism
//!
//! Evaluates the quality of reasoning and planning.

use nafs_core::{Action, ChainOfThought, Result, SupervisionFeedback};

/// Process supervisor for evaluating reasoning quality
pub struct ProcessSupervisor {
    /// Minimum coherence threshold
    min_coherence: f32,
    /// Minimum relevance threshold
    min_relevance: f32,
}

impl ProcessSupervisor {
    /// Create a new process supervisor with default thresholds
    pub fn new() -> Self {
        Self {
            min_coherence: 0.5,
            min_relevance: 0.5,
        }
    }

    /// Create with custom thresholds
    pub fn with_thresholds(min_coherence: f32, min_relevance: f32) -> Self {
        Self {
            min_coherence: min_coherence.clamp(0.0, 1.0),
            min_relevance: min_relevance.clamp(0.0, 1.0),
        }
    }

    /// Supervise reasoning quality
    pub fn supervise(
        &self,
        cot: &ChainOfThought,
        actions: &[Action],
    ) -> Result<SupervisionFeedback> {
        tracing::info!("Running process supervision");

        let reasoning_coherence = self.check_reasoning_coherence(cot);
        let action_relevance = self.check_action_relevance(cot, actions);
        let plan_completeness = self.check_plan_completeness(cot, actions);

        let mut feedback = SupervisionFeedback::new(
            reasoning_coherence,
            action_relevance,
            plan_completeness,
        );

        // Add issues based on checks
        if reasoning_coherence < self.min_coherence {
            feedback.add_issue(format!(
                "Low reasoning coherence: {:.2} (minimum: {:.2})",
                reasoning_coherence, self.min_coherence
            ));
            feedback.add_suggestion("Add more reasoning steps to improve chain-of-thought quality");
        }

        if action_relevance < self.min_relevance {
            feedback.add_issue(format!(
                "Low action relevance: {:.2} (minimum: {:.2})",
                action_relevance, self.min_relevance
            ));
            feedback.add_suggestion("Ensure actions are directly tied to goal success criteria");
        }

        if plan_completeness < 0.6 {
            feedback.add_issue("Plan may not fully address the goal");
            feedback.add_suggestion("Verify all success criteria have corresponding actions");
        }

        // Check for empty plans with goals
        if actions.is_empty() && !cot.goal.success_criteria.is_empty() {
            feedback.add_issue("No actions generated despite having success criteria");
            feedback.add_suggestion("Generate at least one action per success criterion");
        }

        // Log supervision result
        tracing::debug!(
            "Supervision complete - coherence: {:.2}, relevance: {:.2}, completeness: {:.2}",
            reasoning_coherence,
            action_relevance,
            plan_completeness
        );

        Ok(feedback)
    }

    /// Check reasoning coherence (are steps logically connected?)
    fn check_reasoning_coherence(&self, cot: &ChainOfThought) -> f32 {
        if cot.steps.is_empty() {
            return 0.0;
        }

        // Use the precomputed reasoning quality
        let base_quality = cot.reasoning_quality;

        // Bonus for having multiple steps (shows more thorough thinking)
        let step_bonus = (cot.steps.len() as f32 / 10.0).min(0.2);

        // Bonus for high-confidence steps
        let confidence_avg = cot.steps.iter().map(|s| s.confidence).sum::<f32>()
            / cot.steps.len() as f32;

        ((base_quality + step_bonus + confidence_avg) / 2.0).min(1.0)
    }

    /// Check action relevance (are actions tied to the goal?)
    fn check_action_relevance(&self, cot: &ChainOfThought, actions: &[Action]) -> f32 {
        if actions.is_empty() {
            // Empty actions might be okay for simple goals
            return if cot.goal.success_criteria.is_empty() {
                0.8
            } else {
                0.3
            };
        }

        // Base relevance from reasoning quality
        let base = cot.reasoning_quality;

        // Bonus if actions match criteria count
        let criteria_count = cot.goal.success_criteria.len();
        let action_count = actions.len();

        let coverage = if criteria_count > 0 {
            (action_count as f32 / criteria_count as f32).min(1.0)
        } else {
            0.7
        };

        ((base + coverage) / 2.0).min(1.0)
    }

    /// Check plan completeness (does plan address all criteria?)
    fn check_plan_completeness(&self, cot: &ChainOfThought, actions: &[Action]) -> f32 {
        let criteria_count = cot.goal.success_criteria.len();

        if criteria_count == 0 {
            // No criteria = plan is complete by default
            return 1.0;
        }

        let action_count = actions.len();
        let reasoning_steps = cot.steps.len();

        // Completeness based on having enough actions for criteria
        let action_coverage = (action_count as f32 / criteria_count as f32).min(1.0);

        // Bonus for thorough reasoning
        let reasoning_bonus = (reasoning_steps as f32 / (criteria_count as f32 * 2.0)).min(0.3);

        (action_coverage * 0.7 + reasoning_bonus + 0.3 * cot.reasoning_quality).min(1.0)
    }

    /// Quick quality check (returns true if feedback is acceptable)
    pub fn passes_quality(&self, feedback: &SupervisionFeedback) -> bool {
        feedback.overall_score() >= 0.5 && feedback.issues.is_empty()
    }
}

impl Default for ProcessSupervisor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nafs_core::{Goal, ReasoningStep};

    #[test]
    fn test_supervision_empty_cot() {
        let supervisor = ProcessSupervisor::new();
        let cot = ChainOfThought::new(Goal::new("Test", 5));

        let feedback = supervisor.supervise(&cot, &[]).unwrap();
        assert!(feedback.reasoning_coherence >= 0.0);
    }

    #[test]
    fn test_supervision_with_steps() {
        let supervisor = ProcessSupervisor::new();
        let goal = Goal::new("Test", 5);
        let mut cot = ChainOfThought::new(goal);

        cot.add_step(ReasoningStep::new("Step 1", "Reason 1").with_confidence(0.8));
        cot.add_step(ReasoningStep::new("Step 2", "Reason 2").with_confidence(0.9));

        let feedback = supervisor.supervise(&cot, &[]).unwrap();
        assert!(feedback.reasoning_coherence > 0.0);
    }

    #[test]
    fn test_supervision_with_actions() {
        let supervisor = ProcessSupervisor::new();
        let goal = Goal::new("Do task", 5).with_criterion("Task done");
        let mut cot = ChainOfThought::new(goal);
        cot.add_step(ReasoningStep::new("Plan action", "Need to act").with_confidence(0.85));

        let actions = vec![Action::new("do_task", serde_json::json!({}))];

        let feedback = supervisor.supervise(&cot, &actions).unwrap();
        assert!(feedback.action_relevance > 0.5);
    }

    #[test]
    fn test_low_coherence_warning() {
        let supervisor = ProcessSupervisor::with_thresholds(0.9, 0.5);
        let cot = ChainOfThought::new(Goal::new("Test", 5));

        let feedback = supervisor.supervise(&cot, &[]).unwrap();
        // Should have issue about low coherence
        assert!(feedback.issues.iter().any(|i| i.contains("coherence")));
    }

    #[test]
    fn test_passes_quality() {
        let supervisor = ProcessSupervisor::new();
        
        let good_feedback = SupervisionFeedback::new(0.8, 0.8, 0.8);
        assert!(supervisor.passes_quality(&good_feedback));

        let mut bad_feedback = SupervisionFeedback::new(0.3, 0.3, 0.3);
        bad_feedback.add_issue("Problem");
        assert!(!supervisor.passes_quality(&bad_feedback));
    }

    #[test]
    fn test_suggestions_provided() {
        let supervisor = ProcessSupervisor::with_thresholds(0.9, 0.9);
        let cot = ChainOfThought::new(Goal::new("Test", 5));

        let feedback = supervisor.supervise(&cot, &[]).unwrap();
        assert!(!feedback.suggestions.is_empty());
    }
}
