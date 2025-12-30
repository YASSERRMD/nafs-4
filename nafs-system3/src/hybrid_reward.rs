//! Hybrid Reward Module: Extrinsic + Intrinsic
//!
//! Combines task success with learning/curiosity rewards.

/// Module for calculating hybrid rewards
pub struct HybridRewardModule {
    pub extrinsic_weight: f32,
    pub intrinsic_weight: f32,
}

impl HybridRewardModule {
    /// Create a new hybrid reward module
    pub fn new(extrinsic_weight: f32, intrinsic_weight: f32) -> Self {
        Self {
            extrinsic_weight,
            intrinsic_weight,
        }
    }

    /// Compute total weighted reward (0.0 - 1.0)
    pub fn compute_total(&self, extrinsic_value: f32, intrinsic_value: f32) -> f32 {
        let total_weight = self.extrinsic_weight + self.intrinsic_weight;
        if total_weight == 0.0 {
            return 0.5; // Default safety
        }

        (extrinsic_value * self.extrinsic_weight + intrinsic_value * self.intrinsic_weight)
            / total_weight
    }

    /// Analyze reward balance and return description
    pub fn analyze_balance(&self, extrinsic: f32, intrinsic: f32) -> String {
        let total = self.compute_total(extrinsic, intrinsic);
        
        // Normalize components for comparison
        let w_ext = extrinsic * self.extrinsic_weight;
        let w_int = intrinsic * self.intrinsic_weight;
        
        if (w_ext - w_int).abs() < 0.1 {
            format!("Balanced: {:.2}", total)
        } else if w_ext > w_int {
            format!("Task-focused: {:.2}", total)
        } else {
            format!("Learning-focused: {:.2}", total)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid_reward_computation() {
        let module = HybridRewardModule::new(0.6, 0.4);
        let total = module.compute_total(0.8, 0.5);
        // (0.8*0.6 + 0.5*0.4) / 1.0 = 0.48 + 0.20 = 0.68
        assert!((total - 0.68).abs() < 0.001);
    }

    #[test]
    fn test_reward_balance_analysis() {
        let module = HybridRewardModule::new(0.5, 0.5);
        
        // Balanced
        assert!(module.analyze_balance(0.5, 0.5).contains("Balanced"));
        
        // Task focused
        assert!(module.analyze_balance(0.9, 0.1).contains("Task-focused"));
        
        // Learning focused
        assert!(module.analyze_balance(0.1, 0.9).contains("Learning-focused"));
    }
}
