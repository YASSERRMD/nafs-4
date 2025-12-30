//! State Persistence: Durable Storage
//!
//! Persists agent state to disk or database.

use nafs_core::{AgentInstance, NafsError, OrchestratorConfig, Result};

/// Persistence layer for agent state
pub struct StatePersistence {
    backend: String,
    path: String,
}

impl StatePersistence {
    /// Initialize persistence layer
    pub async fn new(config: &OrchestratorConfig) -> Result<Self> {
        tracing::info!("Initializing persistence: {}", config.persistence_backend);

        // Create persistence directory if needed
        if config.persistence_backend == "file" {
            std::fs::create_dir_all(&config.persistence_path)
                .map_err(NafsError::IoError)?;
        }

        Ok(Self {
            backend: config.persistence_backend.clone(),
            path: config.persistence_path.clone(),
        })
    }

    /// Save agent state
    pub async fn save_agent(&self, agent: &AgentInstance) -> Result<()> {
        if self.backend == "file" {
            let agent_path = format!("{}/{}.json", self.path, agent.id);
            let json = serde_json::to_string_pretty(agent)
                .map_err(|e| NafsError::SerializationError(e.to_string()))?;
            std::fs::write(&agent_path, json).map_err(NafsError::IoError)?;
            tracing::debug!("Saved agent to {}", agent_path);
        }
        Ok(())
    }

    /// Load agent state
    pub async fn load_agent(&self, agent_id: &str) -> Result<AgentInstance> {
        if self.backend == "file" {
            let agent_path = format!("{}/{}.json", self.path, agent_id);
            let json = std::fs::read_to_string(&agent_path)
                .map_err(NafsError::IoError)?;
            let agent = serde_json::from_str(&json)
                .map_err(|e| NafsError::SerializationError(e.to_string()))?;
            Ok(agent)
        } else {
            Err(NafsError::NotSupported(
                "Backend not implemented".to_string(),
            ))
        }
    }

    /// Delete agent state
    pub async fn delete_agent(&self, agent_id: &str) -> Result<()> {
        if self.backend == "file" {
            let agent_path = format!("{}/{}.json", self.path, agent_id);
            if std::path::Path::new(&agent_path).exists() {
                std::fs::remove_file(&agent_path)
                    .map_err(NafsError::IoError)?;
                tracing::debug!("Deleted agent from {}", agent_path);
            }
        }
        Ok(())
    }

    /// Load all persisted agents
    pub async fn load_all_agents(&self) -> Result<Vec<AgentInstance>> {
        if self.backend == "file" {
            let mut agents = vec![];
            let entries =
                std::fs::read_dir(&self.path).map_err(NafsError::IoError)?;

            for entry in entries {
                let entry = entry.map_err(NafsError::IoError)?;
                let path = entry.path();

                if path.extension().map(|ext| ext == "json").unwrap_or(false) {
                    let json = std::fs::read_to_string(&path)
                        .map_err(NafsError::IoError)?;
                    let agent = serde_json::from_str(&json)
                        .map_err(|e| NafsError::SerializationError(e.to_string()))?;
                    agents.push(agent);
                }
            }
            Ok(agents)
        } else {
            Err(NafsError::NotSupported(
                "Backend not implemented".to_string(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_persistence() {
        let config = OrchestratorConfig {
            persistence_path: "/tmp/nafs_test_persistence".to_string(),
            ..Default::default()
        };

        let persistence = StatePersistence::new(&config).await.unwrap();
        assert_eq!(persistence.backend, "file");
        
        // Cleanup
        let _ = std::fs::remove_dir_all("/tmp/nafs_test_persistence");
    }
}
