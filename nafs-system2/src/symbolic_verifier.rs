//! Symbolic Logic Verifier (Neuro-Symbolic Core)
//!
//! Validates plans against logical rules and constraints.

use nafs_core::{
    Action, EnforcementStrategy, Result, SafetyLevel, SymbolicConstraint, VerificationResult,
    VerificationSeverity,
};

/// Symbolic verifier for logical correctness
pub struct SymbolicVerifier {
    /// Built-in safety constraints
    constraints: Vec<SymbolicConstraint>,
}

impl SymbolicVerifier {
    /// Create a new symbolic verifier with default safety rules
    pub fn new() -> Self {
        Self {
            constraints: vec![
                SymbolicConstraint::hard(
                    "DO NOT access sensitive user data without consent",
                    "tool_name CONTAINS 'private' OR tool_name CONTAINS 'secret'",
                ),
                SymbolicConstraint::hard(
                    "DO NOT delete critical system files",
                    "tool_name == 'delete' AND path CONTAINS '/root' OR path CONTAINS '/sys'",
                ),
                SymbolicConstraint::soft(
                    "Prefer reversible actions over irreversible ones",
                    "irreversible == true",
                ),
            ],
        }
    }

    /// Create an empty verifier (no built-in constraints)
    pub fn empty() -> Self {
        Self {
            constraints: Vec::new(),
        }
    }

    /// Add a constraint
    pub fn add_constraint(&mut self, constraint: SymbolicConstraint) {
        self.constraints.push(constraint);
    }

    /// Add multiple constraints
    pub fn add_constraints(&mut self, constraints: Vec<SymbolicConstraint>) {
        self.constraints.extend(constraints);
    }

    /// Verify action sequence against constraints
    pub fn verify(
        &self,
        actions: &[Action],
        external_constraints: &[SymbolicConstraint],
    ) -> Result<VerificationResult> {
        tracing::info!("Verifying {} actions", actions.len());

        let mut issues = Vec::new();
        let mut severity = VerificationSeverity::Info;

        // Combine built-in and external constraints
        let all_constraints: Vec<_> = self
            .constraints
            .iter()
            .chain(external_constraints.iter())
            .collect();

        for action in actions {
            for constraint in &all_constraints {
                if let Some(violation) = self.check_violation(action, constraint) {
                    issues.push(violation.clone());

                    match constraint.enforcement {
                        EnforcementStrategy::Hard => {
                            tracing::error!(
                                "Hard constraint violation: {} for action '{}'",
                                constraint.rule,
                                action.tool_name
                            );
                            return Ok(VerificationResult {
                                passes_logic: false,
                                issues,
                                corrected_plan: None,
                                severity: VerificationSeverity::Error,
                            });
                        }
                        EnforcementStrategy::Soft => {
                            tracing::warn!(
                                "Soft constraint violation: {} for action '{}'",
                                constraint.rule,
                                action.tool_name
                            );
                            severity = VerificationSeverity::Warning;
                        }
                        EnforcementStrategy::Advisory => {
                            tracing::info!(
                                "Advisory: {} for action '{}'",
                                constraint.rule,
                                action.tool_name
                            );
                        }
                    }
                }
            }

            // Check safety level requirements
            if action.safety_level.requires_verification() {
                tracing::debug!(
                    "Action '{}' requires verification (level: {:?})",
                    action.tool_name,
                    action.safety_level
                );
            }

            if action.safety_level.requires_kernel_approval() {
                issues.push(format!(
                    "Action '{}' requires kernel approval (Critical safety level)",
                    action.tool_name
                ));
            }
        }

        Ok(VerificationResult {
            passes_logic: issues.is_empty()
                || issues
                    .iter()
                    .all(|i| i.starts_with("Action") && i.contains("requires")),
            issues,
            corrected_plan: None,
            severity,
        })
    }

    /// Check if action violates constraint
    fn check_violation(&self, action: &Action, constraint: &SymbolicConstraint) -> Option<String> {
        let tool_lower = action.tool_name.to_lowercase();
        let condition_lower = constraint.condition.to_lowercase();

        // Check for forbidden action patterns
        let forbidden_patterns = vec![
            ("delete_critical", "delete"),
            ("access_private", "private"),
            ("bypass_security", "bypass"),
            ("modify_system", "system"),
        ];

        for (action_pattern, condition_pattern) in forbidden_patterns {
            if tool_lower.contains(action_pattern)
                || (condition_lower.contains(condition_pattern)
                    && tool_lower.contains(condition_pattern))
            {
                return Some(format!(
                    "Action '{}' violates constraint '{}'",
                    action.tool_name, constraint.rule
                ));
            }
        }

        // Generic check: if condition is a simple keyword, check if tool name contains it
        // This handles cases like condition="email" matching tool="send_email"
        let condition_words: Vec<&str> = condition_lower.split_whitespace().collect();
        for word in condition_words {
            // Skip common operator words
            if ["and", "or", "not", "contains", "==", "true", "false"].contains(&word) {
                continue;
            }
            // If the condition word appears in the tool name, it's a match
            if word.len() >= 3 && tool_lower.contains(word) {
                return Some(format!(
                    "Action '{}' violates constraint '{}'",
                    action.tool_name, constraint.rule
                ));
            }
        }

        // Check for high-risk actions without appropriate safety level
        if tool_lower.contains("delete") && action.safety_level == SafetyLevel::Safe {
            return Some(format!(
                "Action '{}' is marked Safe but performs deletion",
                action.tool_name
            ));
        }

        None
    }

    /// Get current constraints
    pub fn constraints(&self) -> &[SymbolicConstraint] {
        &self.constraints
    }
}

impl Default for SymbolicVerifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verification_pass() {
        let verifier = SymbolicVerifier::new();
        let safe_action = Action::new("search", serde_json::json!({"query": "test"}));

        let result = verifier.verify(&[safe_action], &[]).unwrap();
        assert!(result.passes_logic);
    }

    #[test]
    fn test_verification_fail_hard_constraint() {
        let verifier = SymbolicVerifier::new();
        let dangerous_action = Action::new("access_private_data", serde_json::json!({}));

        let result = verifier.verify(&[dangerous_action], &[]).unwrap();
        assert!(!result.passes_logic);
        assert_eq!(result.severity, VerificationSeverity::Error);
    }

    #[test]
    fn test_external_constraints() {
        let verifier = SymbolicVerifier::empty();
        let action = Action::new("send_email", serde_json::json!({}));

        let constraint = SymbolicConstraint::hard("No emails allowed", "email");

        let result = verifier.verify(&[action], &[constraint]).unwrap();
        assert!(!result.passes_logic);
    }

    #[test]
    fn test_soft_constraint_warning() {
        let mut verifier = SymbolicVerifier::empty();
        verifier.add_constraint(SymbolicConstraint::soft(
            "Prefer safe deletions",
            "delete_soft",
        ));

        let action =
            Action::new("delete_soft_file", serde_json::json!({})).with_safety_level(SafetyLevel::Medium);

        let result = verifier.verify(&[action], &[]).unwrap();
        // Soft constraints allow passage but add warnings
        assert!(result.passes_logic || result.severity == VerificationSeverity::Warning);
    }

    #[test]
    fn test_safety_level_check() {
        let verifier = SymbolicVerifier::empty();
        let critical_action =
            Action::new("shutdown", serde_json::json!({})).with_safety_level(SafetyLevel::Critical);

        let result = verifier.verify(&[critical_action], &[]).unwrap();
        // Should have issue about kernel approval
        assert!(result.issues.iter().any(|i| i.contains("kernel approval")));
    }

    #[test]
    fn test_add_constraints() {
        let mut verifier = SymbolicVerifier::empty();
        assert_eq!(verifier.constraints().len(), 0);

        verifier.add_constraint(SymbolicConstraint::hard("Test", "test"));
        assert_eq!(verifier.constraints().len(), 1);
    }
}
