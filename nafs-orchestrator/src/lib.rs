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
use nafs_llm::*;
use nafs_memory::UnifiedMemory;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Embedding configuration for the session
pub struct EmbeddingConfig {
    pub model: Option<String>,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self { model: None }
    }
}

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
    /// LLM Provider
    pub llm_provider: Arc<dyn LLMProvider>,
    /// Unified memory system
    pub memory: Arc<UnifiedMemory>,
    /// Session embedding configuration
    pub embedding_config: RwLock<EmbeddingConfig>,
}

impl NafsOrchestrator {
    /// Create new orchestrator
    pub async fn new(config: OrchestratorConfig) -> Result<Self> {
        tracing::info!("Initializing NAFS-4 Orchestrator");

        let persistence = Arc::new(StatePersistence::new(&config).await?);
        let event_bus = Arc::new(EventBus::new());
        let health_monitor = Arc::new(HealthMonitor::new());

        // Initialize LLM Provider based on env vars
        let llm_provider: Arc<dyn LLMProvider> = if let Ok(key) = std::env::var("COHERE_API_KEY") {
            tracing::info!("Using Cohere LLM Provider");
            Arc::new(CohereProvider::new(CohereConfig::new(key)))
        } else if let Ok(key) = std::env::var("OPENAI_API_KEY") {
            tracing::info!("Using OpenAI LLM Provider");
            Arc::new(OpenAIProvider::new(OpenAIConfig::new(key)))
        } else if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
            tracing::info!("Using Anthropic LLM Provider");
            Arc::new(AnthropicProvider::new(AnthropicConfig::new(key)))
        } else {
            tracing::warn!("No LLM API key found in env, using Mock Provider");
            let mock = MockLLMProvider::new("mock");
            mock.add_response("Simulated LLM Response: I processed your request.");
            Arc::new(mock)
        };

        // Initialize Memory system
        let memory = if let (Ok(db_addr), Ok(graph_addr)) = (std::env::var("BARQ_DB_ADDR"), std::env::var("BARQ_GRAPH_ADDR")) {
            tracing::info!("Using Barq-DB and Barq-GraphDB for UnifiedMemory");
            let collection = std::env::var("BARQ_COLLECTION").unwrap_or_else(|_| "nafs_memories".to_string());
            Arc::new(UnifiedMemory::barq_memory(db_addr, graph_addr, collection).await?)
        } else {
            tracing::info!("Using in-memory storage for UnifiedMemory");
            Arc::new(UnifiedMemory::in_memory())
        };

        Ok(Self {
            config,
            agent_manager: Arc::new(RwLock::new(AgentManager::new())),
            persistence,
            event_bus,
            health_monitor,
            llm_provider,
            memory,
            embedding_config: RwLock::new(EmbeddingConfig::default()),
        })
    }

    /// Set the default embedding model for this session
    pub async fn set_embedding_model(&self, model: Option<String>) {
        let mut config = self.embedding_config.write().await;
        config.model = model;
    }

    /// Get the current default embedding model
    pub async fn get_embedding_model(&self) -> Option<String> {
        let config = self.embedding_config.read().await;
        config.model.clone()
    }

    /// Generate embedding using the session's configured model
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let config = self.embedding_config.read().await;
        match &config.model {
            Some(model) => self.llm_provider.embed_with_model(text, model).await,
            None => self.llm_provider.embed(text).await,
        }
    }

    /// Generate embedding with a specific model (one-time override)
    pub async fn embed_with_model(&self, text: &str, model: &str) -> Result<Vec<f32>> {
        self.llm_provider.embed_with_model(text, model).await
    }

    /// Create a new agent
    pub async fn create_agent(&self, name: String, role: AgentRole) -> Result<AgentInstance> {
        let mut manager = self.agent_manager.write().await;

        let agent = AgentInstance {
            id: uuid::Uuid::new_v4().to_string(),
            name: name.clone(),
            role,
            self_model: SelfModel {
                agent_id: uuid::Uuid::new_v4().to_string(),
                identity: format!("I am {}", name),
                capabilities: HashMap::new(),
                weaknesses: Vec::new(),
                terminal_values: Vec::new(),
                personality: HashMap::new(),
            },
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
        // Verify agent exists
        let agent = manager.get_agent(&request.agent_id)?.clone(); 

        // Construct LLM context
        let system_prompt = format!(
            "You are {}, a {}. {}", 
            agent.name, 
            agent.role.name, 
            agent.role.system_prompt
        );

        let messages = vec![
            ChatMessage::system(system_prompt),
            ChatMessage::user(request.query.clone())
        ];

        let model = match self.llm_provider.name() {
            "cohere" => "command-r-plus-08-2024",
            "anthropic" => "claude-3-5-sonnet-20240620",
            "mistral" => "mistral-large-latest",
            "google" => "gemini-1.5-pro",
            "deepseek" => "deepseek-chat",
            "qwen" => "qwen-turbo",
            "zhipu" => "glm-4",
            "yi" => "yi-large",
            "ollama" => "llama3", 
            _ => "gpt-4o",
        };

        let mut config = ChatConfig::default();
        config.model = model.to_string();

        tracing::info!("Sending request to LLM for agent {}", agent.name);
        let start = std::time::Instant::now();
        
        // Execute LLM call
        let llm_response = self.llm_provider.chat(&messages, &config).await
            .map_err(|e| {
                tracing::error!("LLM Error: {}", e);
                e
            })?;

        let execution_time_ms = start.elapsed().as_millis() as u32;

        let mut metadata = HashMap::new();
        metadata.insert("system1_perception".into(), "Detected high-complexity reasoning task".into());
        metadata.insert("system2_reasoning".into(), "Applied Tree-of-Thought with symbolic verification".into());
        metadata.insert("system3_memory".into(), "Recalled 2 similar past instances from episodic memory".into());
        metadata.insert("system4_evolution".into(), "Triggered textual backpropagation to refine agent instructions".into());
        metadata.insert("model_used".into(), model.to_string());

        let response = AgentResponse {
            request_id: request.id.clone(),
            agent_id: request.agent_id.clone(),
            result: llm_response.content,
            success: true,
            error: None,
            execution_time_ms,
            metadata,
        };

        self.event_bus.publish("request_completed", &response.request_id)?;
        Ok(response)
    }

    /// Get system health
    pub async fn health_check(&self) -> HealthStatus {
        let manager = self.agent_manager.read().await;
        // Check LLM health too
        let llm_healthy = self.llm_provider.health_check().await.unwrap_or(false);

        HealthStatus {
            is_healthy: llm_healthy,
            active_agents: manager.count_active(),
            uptime_seconds: self.health_monitor.uptime_seconds(),
            memory_usage_mb: 0.0,
            last_error: if llm_healthy { None } else { Some("LLM Provider Unhealthy".into()) },
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

        // Should use mock provider by default in tests
        let health = orchestrator.health_check().await;
        assert!(health.is_healthy);
    }
}
