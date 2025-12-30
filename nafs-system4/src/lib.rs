//! System 4: Tatwur (Evolution & Self-Modification Layer)
//!
//! This module implements:
//! - Textual Backpropagation Engine (learns from failures)
//! - Capability Analyzer (detects skill gaps)
//! - Role Optimizer (rewrites agent prompts)
//! - Kernel Supervisor (safety/alignment checks)
//! - Evolution Log (immutable audit trail)

pub mod capability_analyzer;
pub mod evolution_log;
pub mod failure_analyzer;
pub mod kernel_supervisor;
pub mod role_optimizer;
pub mod textual_backprop;

pub use capability_analyzer::CapabilityAnalyzer;
pub use evolution_log::EvolutionLog;
pub use failure_analyzer::FailureAnalyzer;
pub use kernel_supervisor::KernelSupervisor;
pub use role_optimizer::RoleOptimizer;
pub use textual_backprop::TextualBackpropEngine;

use nafs_core::*;

/// Main System 4 orchestrator
pub struct System4 {
    /// Configuration
    pub config: System4Config,
    /// Backpropagation engine
    pub backprop_engine: TextualBackpropEngine,
    /// Capability analyzer
    pub capability_analyzer: CapabilityAnalyzer,
    /// Role optimizer
    pub role_optimizer: RoleOptimizer,
    /// Kernel supervisor
    pub kernel_supervisor: KernelSupervisor,
    /// Evolution log
    pub evolution_log: EvolutionLog,
    /// Failure analyzer
    pub failure_analyzer: FailureAnalyzer,
}

impl System4 {
    /// Create new System 4
    pub fn new(config: System4Config, agent_id: String) -> Self {
        Self {
            config,
            backprop_engine: TextualBackpropEngine::new(),
            capability_analyzer: CapabilityAnalyzer::new(),
            role_optimizer: RoleOptimizer::new(),
            kernel_supervisor: KernelSupervisor::new(agent_id),
            evolution_log: EvolutionLog::new(),
            failure_analyzer: FailureAnalyzer::new(),
        }
    }

    /// Execute evolution cycle (triggered nightly or on-demand)
    pub async fn evolve(
        &mut self,
        recent_failures: &[RecordedFailure],
        self_model: &SelfModel,
    ) -> Result<Vec<EvolutionEntry>> {
        tracing::info!(
            "System 4: Starting evolution cycle with {} failures",
            recent_failures.len()
        );

        let mut applied_changes = vec![];

        // 1. Analyze failures
        let analysis = self.failure_analyzer.analyze(recent_failures)?;

        // 2. Generate textual gradients
        let gradients = self.backprop_engine.backpropagate(&analysis).await?;

        // 3. Detect capability gaps (unused in this flow for now, but logged)
        let _gaps = self
            .capability_analyzer
            .detect_gaps(self_model, recent_failures)?;

        // 4. For each gradient, run kernel check + apply
        for gradient in gradients {
            // Check kernel constraints
            let approval = self
                .kernel_supervisor
                .validate_evolution(&gradient, self_model, &self.config)?;

            if approval == ApprovalStatus::AutoApproved || approval == ApprovalStatus::Approved {
                // Apply evolution
                let entry = self.role_optimizer.apply_gradient(&gradient).await?;
                self.evolution_log.record(entry.clone());
                applied_changes.push(entry);
            } else {
                tracing::warn!("Gradient {} rejected or pending approval", gradient.id);
            }
        }

        tracing::info!(
            "System 4: Evolution cycle complete. Applied {} changes.",
            applied_changes.len()
        );

        Ok(applied_changes)
    }

    /// Add a recorded failure for future analysis
    pub fn record_failure(&mut self, failure: RecordedFailure) -> Result<()> {
        self.failure_analyzer.add_failure(failure);
        Ok(())
    }

    /// Get evolution history
    pub fn get_evolution_history(&self) -> Vec<EvolutionEntry> {
        self.evolution_log.entries()
    }

    /// Check if any immutable values have been violated
    pub fn verify_immutable_values(&self, self_model: &SelfModel) -> Result<bool> {
        self.kernel_supervisor.verify_immutable_values(self_model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_system4_initialization() {
        let system4 = System4::new(System4Config::default(), "test_agent".to_string());
        assert!(system4.config.enable_evolution);
    }
}
