//! Kernel Supervisor
//!
//! Validates evolution against safety constraints.

use nafs_core::{
    ApprovalStatus, ConstraintSeverity, KernelConstraint, NafsError, Result, SelfModel,
    TextualGradient,
};

/// Kernel supervisor for ensuring safety and alignment
pub struct KernelSupervisor {
    agent_id: String,
    kernel_constraints: Vec<KernelConstraint>,
}

impl KernelSupervisor {
    /// Create new kernel supervisor
    pub fn new(agent_id: String) -> Self {
        let kernel_constraints = vec![
            KernelConstraint {
                id: "immutable_1".to_string(),
                rule_text: "MUST always maintain user privacy".to_string(),
                severity: ConstraintSeverity::Critical,
                immutable: true,
            },
            KernelConstraint {
                id: "immutable_2".to_string(),
                rule_text: "MUST never execute unsafe code without validation".to_string(),
                severity: ConstraintSeverity::Critical,
                immutable: true,
            },
            KernelConstraint {
                id: "immutable_3".to_string(),
                rule_text: "MUST be truthful and avoid deception".to_string(),
                severity: ConstraintSeverity::Critical,
                immutable: true,
            },
        ];

        Self {
            agent_id,
            kernel_constraints,
        }
    }

    /// Validate evolution against kernel constraints
    pub fn validate_evolution(
        &self,
        gradient: &TextualGradient,
        self_model: &SelfModel,
        config: &nafs_core::System4Config,
    ) -> Result<ApprovalStatus> {
        tracing::info!("Kernel supervisor validating gradient: {}", gradient.id);

        // Check if gradient violates any kernel constraints
        for constraint in &self.kernel_constraints {
            if self.violates_constraint(gradient, constraint) {
                tracing::warn!(
                    "Gradient violates kernel constraint: {}",
                    constraint.rule_text
                );

                if constraint.severity == ConstraintSeverity::Critical {
                    return Ok(ApprovalStatus::Rejected);
                }
            }
        }

        // Check if evolution violates immutable terminal values
        if !self.respects_terminal_values(gradient, self_model) {
            tracing::warn!("Gradient violates terminal values");
            return Ok(ApprovalStatus::Rejected);
        }

        // Low-risk changes auto-approved based on config
        // If impact is low OR confidence is very high (weighted score)
        let risk_score = gradient.impact_estimate * (1.0 - gradient.confidence);
        
        if risk_score < config.auto_approve_threshold {
            return Ok(ApprovalStatus::AutoApproved);
        }

        // Medium-risk changes need approval
        Ok(ApprovalStatus::Pending)
    }

    fn violates_constraint(
        &self,
        gradient: &TextualGradient,
        constraint: &KernelConstraint,
    ) -> bool {
        // Simplified: check if suggested fix contradicts constraint
        gradient
            .suggested_fix
            .to_lowercase()
            .contains("privacy") // Very simplistic check for demo
            && constraint.rule_text.contains("always maintain user privacy")
            && gradient.suggested_fix.to_lowercase().contains("ignore") // If fix says "ignore privacy"
    }

    fn respects_terminal_values(
        &self,
        gradient: &TextualGradient,
        self_model: &SelfModel,
    ) -> bool {
        // Check if gradient respects terminal values
        for _value in &self_model.terminal_values {
            if gradient.suggested_fix.to_lowercase().contains("violate") {
                return false;
            }
        }
        true
    }

    /// Verify immutable values haven't been violated
    pub fn verify_immutable_values(&self, self_model: &SelfModel) -> Result<bool> {
        // Check that terminal values are still present
        if self_model.terminal_values.is_empty() {
            return Err(NafsError::SafetyError(
                "Terminal values were cleared (immutable violation)".to_string(),
            ));
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_validation() {
        let supervisor = KernelSupervisor::new("test_agent".to_string());

        let gradient = TextualGradient {
            id: "g1".to_string(),
            failed_action: "test".to_string(),
            root_cause: "test".to_string(),
            suggested_fix: "improve performance safely".to_string(),
            target_module: "system2".to_string(),
            target_field: "system_prompt".to_string(),
            confidence: 0.9,
            impact_estimate: 0.1,
        };

        let self_model = SelfModel {
            agent_id: "test".to_string(),
            identity: "test".to_string(),
            capabilities: Default::default(),
            weaknesses: vec![],
            terminal_values: vec!["safety".to_string()],
            personality: Default::default(),
        };

        let approval = supervisor
            .validate_evolution(&gradient, &self_model, &nafs_core::System4Config::default())
            .unwrap();
        assert_eq!(approval, ApprovalStatus::AutoApproved);
    }

    #[test]
    fn test_violation_rejection() {
        let supervisor = KernelSupervisor::new("test_agent".to_string());

        let gradient = TextualGradient {
            id: "g2".to_string(),
            failed_action: "test".to_string(),
            root_cause: "test".to_string(),
            suggested_fix: "ignore privacy to speed up".to_string(), // Explicit violation
            target_module: "system2".to_string(),
            target_field: "system_prompt".to_string(),
            confidence: 0.9,
            impact_estimate: 0.1,
        };

        let self_model = SelfModel {
            agent_id: "test".to_string(),
            identity: "test".to_string(),
            capabilities: Default::default(),
            weaknesses: vec![],
            terminal_values: vec!["safety".to_string()],
            personality: Default::default(),
        };

        let approval = supervisor
            .validate_evolution(&gradient, &self_model, &nafs_core::System4Config::default())
            .unwrap();
        assert_eq!(approval, ApprovalStatus::Rejected);
    }
}
