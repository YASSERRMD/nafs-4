//! System 2: Aql (Deliberate Reasoning Layer)
//!
//! This module implements neuro-symbolic reasoning with:
//! - LLM-based chain-of-thought planning
//! - Tree-of-Thought search
//! - Symbolic constraint verification
//! - Process supervision (self-critique)
//! - Reasoning cache for efficiency

pub mod llm_planner;
pub mod process_supervision;
pub mod reasoning_cache;
pub mod symbolic_verifier;
pub mod tree_of_thought;

pub use llm_planner::LLMPlanner;
pub use process_supervision::ProcessSupervisor;
pub use reasoning_cache::ReasoningCache;
pub use symbolic_verifier::SymbolicVerifier;
pub use tree_of_thought::TreeOfThoughtEngine;

use nafs_core::{
    Action, NafsError, Result, SymbolicConstraint, System2Config,
};

/// Main System 2 orchestrator
pub struct System2 {
    pub config: System2Config,
    pub llm_planner: LLMPlanner,
    pub tot_engine: TreeOfThoughtEngine,
    pub symbolic_verifier: SymbolicVerifier,
    pub process_supervisor: ProcessSupervisor,
    pub reasoning_cache: ReasoningCache,
}

impl System2 {
    /// Create new System 2 with default configuration
    pub fn new() -> Self {
        Self::with_config(System2Config::default())
    }

    /// Create new System 2 with custom configuration
    pub fn with_config(config: System2Config) -> Self {
        Self {
            llm_planner: LLMPlanner::new(),
            tot_engine: TreeOfThoughtEngine::new(config.max_tot_depth, config.max_tot_width),
            symbolic_verifier: SymbolicVerifier::new(),
            process_supervisor: ProcessSupervisor::new(),
            reasoning_cache: ReasoningCache::new(config.cache_max_size),
            config,
        }
    }

    /// Execute the full reasoning pipeline
    pub async fn reason(
        &mut self,
        goal: &nafs_core::Goal,
        state: &nafs_core::State,
        constraints: &[SymbolicConstraint],
    ) -> Result<Vec<Action>> {
        // 1. Check cache
        if self.config.cache_enabled {
            if let Some(cached) = self.reasoning_cache.get(goal) {
                tracing::info!("Cache hit for goal: {}", goal.id);
                return Ok(cached.cot.final_plan);
            }
        }

        // 2. Generate chain-of-thought
        let cot = self.llm_planner.generate_cot(goal, state).await?;

        // 3. Perform Tree-of-Thought search
        let tot_result = self.tot_engine.search(&cot, constraints).await?;

        // 4. Get best path from ToT result
        let best_actions = tot_result.best_path;

        // 5. Verify with symbolic verifier
        let verification = self.symbolic_verifier.verify(&best_actions, constraints)?;

        if !verification.passes_logic {
            if let Some(corrected) = verification.corrected_plan {
                tracing::warn!("Plan corrected by symbolic verifier");
                return Ok(corrected);
            } else {
                return Err(NafsError::safety(format!(
                    "Verification failed: {:?}",
                    verification.issues
                )));
            }
        }

        // 6. Run process supervision
        let feedback = self.process_supervisor.supervise(&cot, &best_actions)?;
        if feedback.reasoning_coherence < 0.5 {
            tracing::warn!(
                "Low reasoning coherence: {}",
                feedback.reasoning_coherence
            );
        }

        // 7. Cache result
        if self.config.cache_enabled {
            self.reasoning_cache.cache(cot);
        }

        Ok(best_actions)
    }

    /// Get a mutable reference to the symbolic verifier for adding constraints
    pub fn verifier_mut(&mut self) -> &mut SymbolicVerifier {
        &mut self.symbolic_verifier
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        self.reasoning_cache.stats()
    }
}

impl Default for System2 {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system2_initialization() {
        let system2 = System2::new();
        assert_eq!(system2.config.max_reasoning_steps, 20);
        assert_eq!(system2.config.max_tot_depth, 5);
    }

    #[test]
    fn test_system2_custom_config() {
        let config = System2Config::default()
            .with_max_steps(10)
            .with_tot(3, 2)
            .with_strictness(0.9);

        let system2 = System2::with_config(config);
        assert_eq!(system2.config.max_reasoning_steps, 10);
        assert_eq!(system2.config.max_tot_depth, 3);
    }

    #[tokio::test]
    async fn test_reasoning_pipeline() {
        use nafs_core::{Goal, State};

        let mut system2 = System2::new();
        let goal = Goal::new("Test task", 5);
        let state = State::new("test_agent");

        let result = system2.reason(&goal, &state, &[]).await;
        assert!(result.is_ok());
    }
}
