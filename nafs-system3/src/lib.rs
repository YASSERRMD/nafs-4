//! System 3: Wai (Meta-Cognition & Awareness Layer)
//!
//! This module implements:
//! - Executive Monitor (central orchestrator)
//! - Memory Module (episodic + semantic)
//! - Self-Model (persistent identity)
//! - Intrinsic Motivation (curiosity, mastery, autonomy)
//! - Hybrid Reward (extrinsic + intrinsic)

pub mod executive_monitor;
pub mod hybrid_reward;
pub mod intrinsic_motivation;
pub mod memory_module;
pub mod self_model;

pub use executive_monitor::ExecutiveMonitor;
pub use hybrid_reward::HybridRewardModule;
pub use intrinsic_motivation::MotivationEngine;
pub use memory_module::{EpisodicStore, MemoryModule, SemanticStore};
pub use self_model::SelfModelManager;

use nafs_core::{EpisodicEvent, Goal, Result, Reward, SelfModel, State, System3Config};

/// Main System 3 orchestrator
pub struct System3 {
    pub config: System3Config,
    pub executive_monitor: ExecutiveMonitor,
}

impl System3 {
    /// Create new System 3
    pub fn new(config: System3Config, agent_id: String) -> Self {
        let executive_monitor = ExecutiveMonitor::new(config.clone(), agent_id);

        Self {
            config,
            executive_monitor,
        }
    }

    /// Create with default configuration
    pub fn default_for_agent(agent_id: String) -> Self {
        Self::new(System3Config::default(), agent_id)
    }

    /// Execute System 3 cycle: generate goal + reward
    pub async fn tick(&mut self, current_state: &State) -> Result<(Goal, Reward)> {
        self.executive_monitor.tick(current_state).await
    }

    /// Store an experience in memory
    pub fn remember(&mut self, event: EpisodicEvent) -> Result<()> {
        self.executive_monitor.memory.store_experience(event)
    }

    /// Retrieve memories relevant to a query
    pub async fn recall(&self, query: &str) -> Result<Vec<EpisodicEvent>> {
        self.executive_monitor.memory.retrieve_similar(query).await
    }

    /// Get agent's self-model
    pub fn get_self_model(&self) -> &SelfModel {
        &self.executive_monitor.self_model_manager.model
    }

    /// Update agent capability proficiency
    pub fn update_capability(&mut self, skill: String, new_proficiency: f32) {
        self.executive_monitor
            .self_model_manager
            .update_capability(skill, new_proficiency);
    }

    /// Get memory statistics
    pub fn memory_stats(&self) -> (usize, usize) {
        self.executive_monitor.memory.stats()
    }

    /// Get current intrinsic motivation level
    pub fn get_motivation(&self) -> f32 {
        self.executive_monitor
            .motivation_engine
            .compute_drive(&self.executive_monitor.self_model_manager.model)
    }
}

impl Default for System3 {
    fn default() -> Self {
        Self::new(System3Config::default(), "default_agent".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_system3_initialization() {
        let system3 = System3::new(System3Config::default(), "test_agent".to_string());
        assert!(system3.config.enable_memory);
        assert_eq!(system3.config.max_episodic_events, 10000);
    }

    #[tokio::test]
    async fn test_system3_tick() {
        let mut system3 = System3::default_for_agent("test_agent".to_string());
        let state = State::new("test_agent");

        let result = system3.tick(&state).await;
        assert!(result.is_ok());

        let (goal, reward) = result.unwrap();
        assert!(!goal.id.is_empty());
        assert!(reward.total >= 0.0);
    }

    #[test]
    fn test_memory_operations() {
        let mut system3 = System3::default_for_agent("test_agent".to_string());

        let event = EpisodicEvent::new("Test observation", nafs_core::Outcome::Success)
            .with_valence(0.8)
            .with_reflection("Good experience");

        system3.remember(event).unwrap();

        let (episodic, semantic) = system3.memory_stats();
        assert_eq!(episodic, 1);
        assert_eq!(semantic, 0);
    }

    #[test]
    fn test_get_motivation() {
        let system3 = System3::default_for_agent("test_agent".to_string());
        let motivation = system3.get_motivation();
        assert!(motivation >= 0.0 && motivation <= 1.0);
    }
}
