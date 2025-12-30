//! NAFS-4 Orchestrator: Central Coordinator
//!
//! This module implements:
//! - Agent lifecycle management
//! - Multi-agent coordination
//! - State persistence
//! - Event routing
//! - Configuration management

pub mod agent_manager;
pub mod event_bus;
pub mod health_monitor;
pub mod state_persistence;

pub use agent_manager::AgentManager;
pub use event_bus::EventBus;
pub use health_monitor::HealthMonitor;
pub use state_persistence::StatePersistence;

use nafs_core::*;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Main orchestrator
pub struct NafsOrchestrator {
    /// Configuration
    pub config: OrchestratorConfig,
    /// Agent manager with concurrent access
    pub agent_manager: Arc<RwLock<AgentManager>>,
    /// State persistence
    pub persistence: Arc<StatePersistence>,
    /// Event bus
    pub event_bus: Arc<EventBus>,
    /// Health monitor
    pub health_monitor: Arc<HealthMonitor>,
}

impl NafsOrchestrator {
    /// Create new orchestrator
    pub async fn new(config: OrchestratorConfig) -> Result<Self> {
        tracing::info!("Initializing NAFS-4 Orchestrator");

        let persistence = Arc::new(StatePersistence::new(&config).await?);
        let event_bus = Arc::new(EventBus::new());
        let health_monitor = Arc::new(HealthMonitor::new());

        Ok(Self {
            config,
            agent_manager: Arc::new(RwLock::new(AgentManager::new())),
            persistence,
            event_bus,
            health_monitor,
        })
    }

    /// Create a new agent
    pub async fn create_agent(&self, name: String, role: AgentRole) -> Result<AgentInstance> {
        let mut manager = self.agent_manager.write().await;

        let agent = AgentInstance {
            id: uuid::Uuid::new_v4().to_string(),
            name,
            role,
            self_model: SelfModel::default(),
            created_at: chrono::Utc::now(),
            last_activity: chrono::Utc::now(),
            is_active: true,
            metadata: HashMap::new(),
        };

        manager.add_agent(agent.clone())?;
        self.persistence.save_agent(&agent).await?;
        self.event_bus.publish("agent_created", &agent.id)?;

        tracing::info!("Agent created: {}", agent.id);
        Ok(agent)
    }

    /// Execute agent query
    pub async fn execute_request(&self, request: AgentRequest) -> Result<AgentResponse> {
        let manager = self.agent_manager.read().await;
        let _agent = manager.get_agent(&request.agent_id)?; // Verify agent exists

        // In Phase 4, this is a simple placeholder
        // Phase 5 will integrate the full System 1-4 pipeline
        let result = format!("Processed query: {}", request.query);
        let execution_time_ms = 100;

        let response = AgentResponse {
            request_id: request.id.clone(),
            agent_id: request.agent_id.clone(),
            result,
            success: true,
            error: None,
            execution_time_ms,
            metadata: HashMap::new(),
        };

        self.event_bus
            .publish("request_completed", &response.request_id)?;
        Ok(response)
    }

    /// Get system health
    pub async fn health_check(&self) -> HealthStatus {
        let manager = self.agent_manager.read().await;

        HealthStatus {
            is_healthy: true,
            active_agents: manager.count_active(),
            uptime_seconds: self.health_monitor.uptime_seconds(),
            memory_usage_mb: 0.0,
            last_error: None,
            timestamp: chrono::Utc::now(),
        }
    }

    /// Get system statistics
    pub async fn get_stats(&self) -> SystemStats {
        SystemStats {
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            average_response_time_ms: 0.0,
            total_memories_stored: 0,
            total_evolutions: 0,
            agents_by_status: HashMap::new(),
        }
    }

    /// List all agents
    pub async fn list_agents(&self) -> Result<Vec<AgentInstance>> {
        let manager = self.agent_manager.read().await;
        Ok(manager.list_all())
    }

    /// Delete agent
    pub async fn delete_agent(&self, agent_id: &str) -> Result<()> {
        let mut manager = self.agent_manager.write().await;
        manager.remove_agent(agent_id)?;
        self.persistence.delete_agent(agent_id).await?;
        self.event_bus.publish("agent_deleted", agent_id)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_orchestrator_creation() {
        let orchestrator = NafsOrchestrator::new(OrchestratorConfig::default())
            .await
            .unwrap();

        let health = orchestrator.health_check().await;
        assert!(health.is_healthy);
    }
}
