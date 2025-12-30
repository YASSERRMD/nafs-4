//! Executive Monitor: Central Orchestrator
//!
//! Coordinates memory, self-model, motivation, and goal generation.

use crate::{HybridRewardModule, MemoryModule, MotivationEngine, SelfModelManager};
use nafs_core::{
    EnvironmentEvent, ExtrinsicReward, Goal, IntrinsicReward, Result, Reward, State,
    System3Config,
};
use std::collections::VecDeque;

/// Executive Monitor coordinates all System 3 components
pub struct ExecutiveMonitor {
    pub config: System3Config,
    pub agent_id: String,
    pub memory: MemoryModule,
    pub self_model_manager: SelfModelManager,
    pub motivation_engine: MotivationEngine,
    pub hybrid_reward: HybridRewardModule,
    pub event_queue: VecDeque<EnvironmentEvent>,
}

impl ExecutiveMonitor {
    /// Create a new executive monitor
    pub fn new(config: System3Config, agent_id: String) -> Self {
        let mut self_model_manager = SelfModelManager::new(agent_id.clone());
        self_model_manager.initialize_default();

        Self {
            memory: MemoryModule::new(config.max_episodic_events),
            self_model_manager,
            motivation_engine: MotivationEngine::new(config.intrinsic_motivation_weights),
            hybrid_reward: HybridRewardModule::new(
                config.extrinsic_reward_weight,
                config.intrinsic_reward_weight,
            ),
            event_queue: VecDeque::new(),
            config,
            agent_id,
        }
    }

    /// Main executive cycle
    pub async fn tick(&mut self, current_state: &State) -> Result<(Goal, Reward)> {
        tracing::info!("System 3 tick for agent: {}", self.agent_id);

        // 1. Process event queue
        while let Some(event) = self.event_queue.pop_front() {
            self.process_event(event);
        }

        // 2. Retrieve relevant memories
        let memories = self.memory.retrieve_recent(10).await?;
        tracing::debug!("Retrieved {} recent memories", memories.len());

        // 3. Consult self-model
        let agent_identity = self.self_model_manager.model.identity.clone();
        tracing::debug!("Agent identity: {}", agent_identity);

        // 4. Compute intrinsic motivation
        let intrinsic_drive = self
            .motivation_engine
            .compute_drive(&self.self_model_manager.model);
        tracing::debug!("Intrinsic motivation: {:.2}", intrinsic_drive);

        // 5. Generate goal
        let goal = self.generate_goal(current_state, intrinsic_drive)?;

        // 6. Compute hybrid reward
        let extrinsic = ExtrinsicReward {
            value: 0.5, // Placeholder - real value from System 1 feedback
            success_flag: true,
            feedback_text: Some("Ongoing".to_string()),
        };

        let intrinsic = IntrinsicReward {
            curiosity: intrinsic_drive * 0.4,
            mastery: intrinsic_drive * 0.35,
            autonomy: intrinsic_drive * 0.25,
        };

        let intrinsic_sum = intrinsic.curiosity + intrinsic.mastery + intrinsic.autonomy;
        let total_reward = self
            .hybrid_reward
            .compute_total(extrinsic.value, intrinsic_sum);

        let reward = Reward {
            extrinsic,
            intrinsic,
            total: total_reward,
            timestamp: chrono::Utc::now(),
        };

        Ok((goal, reward))
    }

    /// Generate next goal based on current state and motivation
    fn generate_goal(&self, state: &State, intrinsic_drive: f32) -> Result<Goal> {
        // If there's an active goal, continue with it
        if let Some(current_goal) = &state.current_goal {
            return Ok(current_goal.clone());
        }

        // Generate a new goal based on motivation level
        let priority = (intrinsic_drive * 10.0) as u8;
        let priority = priority.clamp(1, 10);

        Ok(Goal::new(
            "Maintain persistent identity and learn from interactions",
            priority,
        )
        .with_criterion("Store experience in memory")
        .with_criterion("Update self-model based on outcomes")
        .with_criterion("Optimize future actions"))
    }

    /// Process an environment event
    fn process_event(&mut self, event: EnvironmentEvent) {
        tracing::debug!("Processing event: {} (type: {})", event.id, event.event_type);

        // Update self-model based on event type
        match event.event_type.as_str() {
            "success" => {
                // Slightly increase relevant capabilities
                for (skill, prof) in self.self_model_manager.model.capabilities.clone() {
                    let new_prof = (prof + 0.01).min(1.0);
                    self.self_model_manager.update_capability(skill, new_prof);
                }
            }
            "failure" => {
                // Note potential weaknesses
                if let Some(context) = event.data.get("context") {
                    if let Some(ctx_str) = context.as_str() {
                        self.self_model_manager.add_weakness(ctx_str.to_string());
                    }
                }
            }
            _ => {}
        }
    }

    /// Add event to queue
    pub fn queue_event(&mut self, event: EnvironmentEvent) {
        self.event_queue.push_back(event);
    }

    /// Get pending event count
    pub fn pending_events(&self) -> usize {
        self.event_queue.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_executive_monitor_tick() {
        let mut monitor =
            ExecutiveMonitor::new(System3Config::default(), "test_agent".to_string());

        let state = State::new("test_agent");

        let (goal, reward) = monitor.tick(&state).await.unwrap();
        assert!(!goal.id.is_empty());
        assert!(reward.total >= 0.0 && reward.total <= 1.0);
    }

    #[test]
    fn test_event_queue() {
        let mut monitor =
            ExecutiveMonitor::new(System3Config::default(), "test_agent".to_string());

        let event = EnvironmentEvent::new("test_event", serde_json::json!({"key": "value"}));

        monitor.queue_event(event);
        assert_eq!(monitor.pending_events(), 1);
    }

    #[tokio::test]
    async fn test_event_processing() {
        let mut monitor =
            ExecutiveMonitor::new(System3Config::default(), "test_agent".to_string());

        let event = EnvironmentEvent::new("success", serde_json::json!({}));
        monitor.queue_event(event);

        let state = State::new("test_agent");
        let _ = monitor.tick(&state).await;

        // Event should be processed
        assert_eq!(monitor.pending_events(), 0);
    }

    #[test]
    fn test_goal_generation() {
        let monitor = ExecutiveMonitor::new(System3Config::default(), "test_agent".to_string());

        let state = State::new("test_agent");
        let goal = monitor.generate_goal(&state, 0.7).unwrap();

        assert!(!goal.success_criteria.is_empty());
        assert!(goal.priority >= 1 && goal.priority <= 10);
    }
}
