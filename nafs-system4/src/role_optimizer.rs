//! Role Optimizer
//!
//! Rewrites agent system prompts and capability definitions.

use nafs_core::{ApprovalStatus, EvolutionEntry, Result, TextualGradient};

/// Optimizer for agent role evolution
pub struct RoleOptimizer;

impl RoleOptimizer {
    /// Create new optimizer
    pub fn new() -> Self {
        Self
    }

    /// Apply a gradient to optimize agent role
    pub async fn apply_gradient(&self, gradient: &TextualGradient) -> Result<EvolutionEntry> {
        tracing::info!(
            "Applying gradient to target: {}.{}",
            gradient.target_module,
            gradient.target_field
        );

        // In real implementation, this would call LLM to rewrite system prompt
        // For Phase 3, we simulate the changes

        let applied_changes = format!(
            "Updated {} to address: {}",
            gradient.target_field, gradient.root_cause
        );

        let entry = EvolutionEntry {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: chrono::Utc::now(),
            gradient: gradient.clone(),
            approval_status: ApprovalStatus::AutoApproved,
            approved_by: Some("kernel_supervisor".to_string()),
            applied_changes,
            performance_delta_before: 0.5,
            performance_delta_after: 0.65,
            rollback_available: true,
        };

        Ok(entry)
    }
}

impl Default for RoleOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_gradient_application() {
        let optimizer = RoleOptimizer::new();

        let gradient = TextualGradient {
            id: "g1".to_string(),
            failed_action: "test".to_string(),
            root_cause: "test".to_string(),
            suggested_fix: "improve".to_string(),
            target_module: "system2".to_string(),
            target_field: "system_prompt".to_string(),
            confidence: 0.8,
            impact_estimate: 0.3,
        };

        let entry = optimizer.apply_gradient(&gradient).await.unwrap();
        assert_eq!(entry.approval_status, ApprovalStatus::AutoApproved);
    }
}
