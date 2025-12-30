//! Intrinsic Motivation Engine
//!
//! Computes curiosity, mastery, and autonomy drives.

use nafs_core::SelfModel;

/// Engine for computing intrinsic motivation
pub struct MotivationEngine {
    pub curiosity_weight: f32,
    pub mastery_weight: f32,
    pub autonomy_weight: f32,
}

impl MotivationEngine {
    /// Create a new motivation engine with specified weights
    pub fn new(weights: (f32, f32, f32)) -> Self {
        Self {
            curiosity_weight: weights.0,
            mastery_weight: weights.1,
            autonomy_weight: weights.2,
        }
    }

    /// Compute intrinsic motivation drive (0.0 - 1.0)
    pub fn compute_drive(&self, self_model: &SelfModel) -> f32 {
        // Curiosity: unexplored areas or novel situations
        let curiosity = self.compute_curiosity_drive(self_model);
        
        // Mastery: opportunity to improve skills
        let mastery = self.compute_mastery_drive(self_model);
        
        // Autonomy: freedom to choose actions
        let autonomy = self.compute_autonomy_drive(self_model);

        // Weighted sum
        let total_weight = self.curiosity_weight + self.mastery_weight + self.autonomy_weight;
        
        if total_weight == 0.0 {
            return 0.5; // Default if weights are zero
        }

        let total = (curiosity * self.curiosity_weight
            + mastery * self.mastery_weight
            + autonomy * self.autonomy_weight)
            / total_weight;

        total.clamp(0.0, 1.0)
    }

    /// Compute curiosity drive based on novelty
    fn compute_curiosity_drive(&self, _self_model: &SelfModel) -> f32 {
        // In a full implementation, this would look at prediction error
        // For now, we return a baseline moderate curiosity
        0.6
    }

    /// Compute mastery drive based on skill gaps
    fn compute_mastery_drive(&self, self_model: &SelfModel) -> f32 {
        if self_model.capabilities.is_empty() {
            return 1.0; // High drive to learn if nothing known
        }
        
        // Average proficiency of all skills
        let avg_proficiency: f32 = self_model.capabilities.values().sum::<f32>()
            / self_model.capabilities.len() as f32;
        
        // Drive to improve is inverse of current proficiency
        (1.0 - avg_proficiency).clamp(0.1, 1.0)
    }

    /// Compute autonomy drive based on constraints
    fn compute_autonomy_drive(&self, _self_model: &SelfModel) -> f32 {
        // Baseline autonomy drive
        0.7
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_motivation_computation() {
        let engine = MotivationEngine::new((0.3, 0.4, 0.3));
        let self_model = SelfModel {
            agent_id: "test".to_string(),
            identity: "Test agent".to_string(),
            capabilities: vec![
                ("skill1".to_string(), 0.7),
                ("skill2".to_string(), 0.6),
            ]
            .into_iter()
            .collect(),
            weaknesses: vec![],
            terminal_values: vec![],
            personality: Default::default(),
        };

        let drive = engine.compute_drive(&self_model);
        assert!(drive >= 0.0 && drive <= 1.0);
    }

    #[test]
    fn test_mastery_drive() {
        let engine = MotivationEngine::new((0.0, 1.0, 0.0));
        
        // Low skill = high mastery drive
        let novice_model = SelfModel {
            agent_id: "novice".to_string(),
            identity: "".to_string(),
            capabilities: vec![("skill".to_string(), 0.1)].into_iter().collect(),
            weaknesses: vec![],
            terminal_values: vec![],
            personality: Default::default(),
        };
        
        let novice_drive = engine.compute_drive(&novice_model);
        
        // High skill = low mastery drive
        let expert_model = SelfModel {
            agent_id: "expert".to_string(),
            identity: "".to_string(),
            capabilities: vec![("skill".to_string(), 0.9)].into_iter().collect(),
            weaknesses: vec![],
            terminal_values: vec![],
            personality: Default::default(),
        };
        
        let expert_drive = engine.compute_drive(&expert_model);
        
        assert!(novice_drive > expert_drive);
    }
}
