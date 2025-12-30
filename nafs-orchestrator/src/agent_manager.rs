//! Agent Manager: Lifecycle & Coordination
//!
//! Manages agent creation, deletion, and lookup.

use nafs_core::{AgentInstance, NafsError, Result};
use std::collections::HashMap;

/// Manager for handling agent instances
pub struct AgentManager {
    agents: HashMap<String, AgentInstance>,
}

impl AgentManager {
    /// Create new agent manager
    pub fn new() -> Self {
        Self {
            agents: HashMap::new(),
        }
    }

    /// Add a new agent instance
    pub fn add_agent(&mut self, agent: AgentInstance) -> Result<()> {
        if self.agents.contains_key(&agent.id) {
            return Err(NafsError::AlreadyExists(agent.id.clone()));
        }
        self.agents.insert(agent.id.clone(), agent);
        Ok(())
    }

    /// Retrieve an agent by ID
    pub fn get_agent(&self, id: &str) -> Result<AgentInstance> {
        self.agents
            .get(id)
            .cloned()
            .ok_or_else(|| NafsError::NotFound(id.to_string()))
    }

    /// Remove an agent by ID
    pub fn remove_agent(&mut self, id: &str) -> Result<()> {
        self.agents
            .remove(id)
            .ok_or_else(|| NafsError::NotFound(id.to_string()))?;
        Ok(())
    }

    /// List all agents
    pub fn list_all(&self) -> Vec<AgentInstance> {
        self.agents.values().cloned().collect()
    }

    /// Count active agents
    pub fn count_active(&self) -> usize {
        self.agents.values().filter(|a| a.is_active).count()
    }

    /// Count total agents
    pub fn count_total(&self) -> usize {
        self.agents.len()
    }
}

impl Default for AgentManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nafs_core::{AgentRole, SelfModel};

    #[test]
    fn test_agent_manager() {
        let mut manager = AgentManager::new();

        let agent = AgentInstance {
            id: "agent_1".to_string(),
            name: "Test Agent".to_string(),
            role: AgentRole {
                id: "role_1".to_string(),
                name: "Test Role".to_string(),
                system_prompt: "Test".to_string(),
                version: 1,
                capabilities: vec![],
                constraints: vec![],
                evolution_lineage: vec![],
                created_at: chrono::Utc::now(),
                last_updated: chrono::Utc::now(),
            },
            self_model: SelfModel::default(),
            created_at: chrono::Utc::now(),
            last_activity: chrono::Utc::now(),
            is_active: true,
            metadata: Default::default(),
        };

        manager.add_agent(agent.clone()).unwrap();
        assert_eq!(manager.count_total(), 1);
        assert_eq!(manager.get_agent("agent_1").unwrap().id, "agent_1");
    }
}
