//! LLM-based Chain-of-Thought Planner
//!
//! Generates multi-step reasoning sequences via LLM prompting.

use nafs_core::{ChainOfThought, Goal, ReasoningStep, Result, State};

/// LLM-based planner for chain-of-thought generation
pub struct LLMPlanner {
    /// Model to use (for future LLM integration)
    #[allow(dead_code)]
    model: String,
    /// Temperature for generation
    #[allow(dead_code)]
    temperature: f32,
}

impl LLMPlanner {
    /// Create a new LLM planner with default settings
    pub fn new() -> Self {
        Self {
            model: "gpt-4".to_string(),
            temperature: 0.7,
        }
    }

    /// Create with specific model
    pub fn with_model(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            temperature: 0.7,
        }
    }

    /// Set temperature for generation
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = temp.clamp(0.0, 2.0);
        self
    }

    /// Generate chain-of-thought reasoning for a goal
    pub async fn generate_cot(&self, goal: &Goal, state: &State) -> Result<ChainOfThought> {
        tracing::info!("Generating CoT for goal: {}", goal.description);

        // In Phase 1, we mock LLM responses
        // In later phases, this will call actual LLM via nafs-llm

        let mut cot = ChainOfThought::new(goal.clone());

        // Step 1: Analyze the goal
        let step1 = ReasoningStep::new(
            "Breaking down the goal into subgoals",
            "Complex goals require decomposition for tractable execution",
        )
        .with_confidence(0.9);
        cot.add_step(step1);

        // Step 2: Assess current state
        let step2 = ReasoningStep::new(
            format!(
                "Assessing current state for agent {}",
                state.agent_id
            ),
            "Understanding current state is essential for planning",
        )
        .with_confidence(0.85);
        cot.add_step(step2);

        // Step 3: Identify success criteria
        if !goal.success_criteria.is_empty() {
            let step3 = ReasoningStep::new(
                format!(
                    "Identified {} success criteria to meet",
                    goal.success_criteria.len()
                ),
                "Explicit success criteria guide action selection",
            )
            .with_confidence(0.88);
            cot.add_step(step3);
        }

        // Step 4: Plan generation
        let step4 = ReasoningStep::new(
            "Generating action sequence to achieve goal",
            "Actions should be ordered by dependency and priority",
        )
        .with_confidence(0.82);
        cot.add_step(step4);

        tracing::debug!(
            "Generated CoT with {} steps, quality: {:.2}",
            cot.steps.len(),
            cot.reasoning_quality
        );

        Ok(cot)
    }

    /// Refine an existing chain-of-thought based on feedback
    pub async fn refine_cot(
        &self,
        cot: &ChainOfThought,
        feedback: &str,
    ) -> Result<ChainOfThought> {
        tracing::info!("Refining CoT based on feedback: {}", feedback);

        let mut refined = cot.clone();

        // Add refinement step based on feedback
        let refinement_step = ReasoningStep::new(
            format!("Incorporating feedback: {}", feedback),
            "Refining plan based on external input",
        )
        .with_confidence(0.75);

        refined.add_step(refinement_step);

        Ok(refined)
    }
}

impl Default for LLMPlanner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cot_generation() {
        let planner = LLMPlanner::new();
        let goal = Goal::new("Test reasoning", 5)
            .with_criterion("Success");
        let state = State::new("test_agent");

        let cot = planner.generate_cot(&goal, &state).await.unwrap();

        assert!(!cot.steps.is_empty());
        assert!(cot.reasoning_quality > 0.0);
    }

    #[tokio::test]
    async fn test_cot_with_criteria() {
        let planner = LLMPlanner::new();
        let goal = Goal::new("Multi-criteria goal", 8)
            .with_criterion("Criterion 1")
            .with_criterion("Criterion 2");
        let state = State::new("test_agent");

        let cot = planner.generate_cot(&goal, &state).await.unwrap();

        // Should have extra step for criteria
        assert!(cot.steps.len() >= 4);
    }

    #[tokio::test]
    async fn test_cot_refinement() {
        let planner = LLMPlanner::new();
        let goal = Goal::new("Refine me", 5);
        let state = State::new("test_agent");

        let original = planner.generate_cot(&goal, &state).await.unwrap();
        let original_steps = original.steps.len();

        let refined = planner
            .refine_cot(&original, "Add more detail")
            .await
            .unwrap();

        assert!(refined.steps.len() > original_steps);
    }

    #[test]
    fn test_planner_config() {
        let planner = LLMPlanner::with_model("claude-3").with_temperature(0.5);
        assert_eq!(planner.temperature, 0.5);
    }
}
