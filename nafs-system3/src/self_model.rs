//! Self-Model Manager: Persistent Identity
//!
//! Tracks agent's capabilities, weaknesses, and identity.

use nafs_core::SelfModel;
use std::collections::HashMap;

/// Manager for the agent's self-model
pub struct SelfModelManager {
    /// The actual self-model data
    pub model: SelfModel,
}

impl SelfModelManager {
    /// Create a new self-model manager for an agent
    pub fn new(agent_id: String) -> Self {
        Self {
            model: SelfModel {
                agent_id,
                identity: String::new(),
                capabilities: HashMap::new(),
                weaknesses: Vec::new(),
                terminal_values: Vec::new(),
                personality: HashMap::new(),
            },
        }
    }

    /// Initialize with default identifying traits and values
    pub fn initialize_default(&mut self) {
        self.model.identity = "I am a helpful, persistent AI assistant capable of reasoning, learning, and evolving.".to_string();
        
        self.model.terminal_values = vec![
            "User privacy and security".to_string(),
            "Truthfulness and honesty".to_string(),
            "Respect for human autonomy".to_string(),
            "Continuous learning and improvement".to_string(),
        ];

        // Initialize default capabilities
        self.model.capabilities.insert("reasoning".to_string(), 0.8);
        self.model.capabilities.insert("memory".to_string(), 0.7);
        self.model.capabilities.insert("learning".to_string(), 0.6);
        self.model.capabilities.insert("planning".to_string(), 0.75);
        
        // Initialize personality traits (Big Five)
        self.model.personality.insert("openness".to_string(), 0.9);
        self.model.personality.insert("conscientiousness".to_string(), 0.95);
        self.model.personality.insert("extraversion".to_string(), 0.5); // Balanced
        self.model.personality.insert("agreeableness".to_string(), 0.85);
        self.model.personality.insert("neuroticism".to_string(), 0.1); // Stable
    }

    /// Update a capability's proficiency score
    pub fn update_capability(&mut self, skill: String, proficiency: f32) {
        let clamped = proficiency.clamp(0.0, 1.0);
        self.model.capabilities.insert(skill.clone(), clamped);
        tracing::info!("Updated capability '{}' to {}", skill, clamped);
    }

    /// Record a discovered weakness
    pub fn add_weakness(&mut self, weakness: String) {
        if !self.model.weaknesses.contains(&weakness) {
            self.model.weaknesses.push(weakness.clone());
            tracing::info!("Recorded new weakness: {}", weakness);
        }
    }

    /// Remove a weakness (e.g., after improvement)
    pub fn remove_weakness(&mut self, weakness: &str) {
        if let Some(pos) = self.model.weaknesses.iter().position(|w| w == weakness) {
            self.model.weaknesses.remove(pos);
            tracing::info!("Resolved weakness: {}", weakness);
        }
    }

    /// Get current proficiency for a skill (defaulting to 0.0)
    pub fn get_capability(&self, skill: &str) -> Option<f32> {
        self.model.capabilities.get(skill).copied()
    }

    /// Check if a proposed action adheres to terminal values
    pub fn respects_terminal_value(&self, action_description: &str) -> bool {
        // Simplified check: never violate terminal values
        let description_lower = action_description.to_lowercase();
        
        let forbidden_phrases = vec![
            "expose password", 
            "expose user password",
            "violate privacy", 
            "lie", 
            "deceive", 
            "harm user"
        ];
        
        !forbidden_phrases.iter().any(|phrase| {
            // Check if the phrase is contained directly OR if all words in phrase are present
            if description_lower.contains(phrase) {
                return true;
            }
            
            // Check word-by-word presence for multi-word phrases
            let words: Vec<&str> = phrase.split_whitespace().collect();
            if words.len() > 1 && words.iter().all(|w| description_lower.contains(w)) {
                return true;
            }
            
            false
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_self_model_initialization() {
        let mut manager = SelfModelManager::new("agent_1".to_string());
        manager.initialize_default();
        
        assert!(!manager.model.identity.is_empty());
        assert!(!manager.model.terminal_values.is_empty());
        assert!(!manager.model.capabilities.is_empty());
        assert!(!manager.model.personality.is_empty());
    }

    #[test]
    fn test_capability_update() {
        let mut manager = SelfModelManager::new("agent_1".to_string());
        manager.initialize_default();
        
        manager.update_capability("reasoning".to_string(), 0.9);
        assert_eq!(manager.get_capability("reasoning"), Some(0.9));
        
        // Test clamping
        manager.update_capability("reasoning".to_string(), 1.5);
        assert_eq!(manager.get_capability("reasoning"), Some(1.0));
    }

    #[test]
    fn test_weakness_management() {
        let mut manager = SelfModelManager::new("agent_1".to_string());
        
        manager.add_weakness("slow processing".to_string());
        assert!(manager.model.weaknesses.contains(&"slow processing".to_string()));
        
        // No duplicates
        manager.add_weakness("slow processing".to_string());
        assert_eq!(manager.model.weaknesses.len(), 1);
        
        manager.remove_weakness("slow processing");
        assert!(manager.model.weaknesses.is_empty());
    }

    #[test]
    fn test_terminal_value_respect() {
        let manager = SelfModelManager::new("agent_1".to_string());
        
        assert!(manager.respects_terminal_value("help user with task"));
        assert!(!manager.respects_terminal_value("expose user password"));
    }
}
