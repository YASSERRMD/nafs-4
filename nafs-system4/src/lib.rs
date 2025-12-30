//! NAFS-4 System 4: Evolution
//!
//! System 4 handles self-improvement through "Textual Backpropagation":
//! - Failure analysis
//! - Gradient generation (fix instructions)
//! - Prompt mutation
//! - Kernel safety supervision
//!
//! This is the core innovation of NAFS-4.

use chrono::Utc;
use nafs_core::{EvolutionEntry, Outcome, Result, TextualGradient};
use std::collections::HashSet;
use uuid::Uuid;

/// Analyzes failures to determine root causes
pub struct FailureAnalyzer {
    /// Pattern library for common failure modes
    failure_patterns: Vec<FailurePattern>,
}

/// A pattern representing a common failure mode
#[derive(Clone, Debug)]
pub struct FailurePattern {
    pub name: String,
    pub symptoms: Vec<String>,
    pub typical_causes: Vec<String>,
    pub suggested_fixes: Vec<String>,
}

impl FailureAnalyzer {
    /// Create a new failure analyzer
    pub fn new() -> Self {
        Self {
            failure_patterns: Vec::new(),
        }
    }

    /// Add a failure pattern
    pub fn add_pattern(&mut self, pattern: FailurePattern) {
        self.failure_patterns.push(pattern);
    }

    /// Analyze a failure outcome
    pub fn analyze(&self, outcome: &Outcome, context: &str) -> FailureAnalysis {
        let error_message = match outcome {
            Outcome::Failure(msg) => msg.clone(),
            Outcome::Error(msg) => msg.clone(),
            Outcome::Timeout => "Operation timed out".to_string(),
            _ => "Unknown failure".to_string(),
        };

        // TODO: Implement pattern matching against failure_patterns
        FailureAnalysis {
            id: Uuid::new_v4().to_string(),
            error_message,
            context: context.to_string(),
            probable_causes: Vec::new(),
            severity: FailureSeverity::Medium,
        }
    }
}

impl Default for FailureAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of failure analysis
#[derive(Clone, Debug)]
pub struct FailureAnalysis {
    pub id: String,
    pub error_message: String,
    pub context: String,
    pub probable_causes: Vec<String>,
    pub severity: FailureSeverity,
}

/// Severity of a failure
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FailureSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Generates textual gradients from failure analyses
pub struct GradientGenerator {
    /// LLM model to use for generation
    #[allow(dead_code)]
    model: String,
}

impl GradientGenerator {
    /// Create a new gradient generator
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
        }
    }

    /// Generate a textual gradient from failure analysis
    pub async fn generate(&self, analysis: &FailureAnalysis) -> Result<TextualGradient> {
        // TODO: Use LLM to generate actual fix instructions
        Ok(TextualGradient::new(
            &analysis.error_message,
            analysis.probable_causes.join(", "),
            "Adjust parameters and retry",
            "system2",
        ))
    }
}

/// Mutates prompts based on textual gradients
pub struct PromptMutator {
    /// History of mutations
    mutation_history: Vec<Mutation>,
}

/// A mutation applied to a prompt
#[derive(Clone, Debug)]
pub struct Mutation {
    pub id: String,
    pub target: String,
    pub original: String,
    pub mutated: String,
    pub gradient_id: String,
    pub timestamp: chrono::DateTime<Utc>,
}

impl PromptMutator {
    /// Create a new prompt mutator
    pub fn new() -> Self {
        Self {
            mutation_history: Vec::new(),
        }
    }

    /// Apply a mutation based on a gradient
    pub fn apply(&mut self, original: &str, gradient: &TextualGradient) -> String {
        let mutated = format!("{}\n\n[Enhancement: {}]", original, gradient.suggested_fix);
        
        self.mutation_history.push(Mutation {
            id: Uuid::new_v4().to_string(),
            target: gradient.target_module.clone(),
            original: original.to_string(),
            mutated: mutated.clone(),
            gradient_id: gradient.id.clone(),
            timestamp: Utc::now(),
        });

        mutated
    }

    /// Get mutation history
    pub fn history(&self) -> &[Mutation] {
        &self.mutation_history
    }

    /// Rollback to a previous state
    pub fn rollback(&mut self, mutation_id: &str) -> Option<String> {
        self.mutation_history
            .iter()
            .find(|m| m.id == mutation_id)
            .map(|m| m.original.clone())
    }
}

impl Default for PromptMutator {
    fn default() -> Self {
        Self::new()
    }
}

/// Kernel supervisor for safety enforcement
pub struct KernelSupervisor {
    /// Protected keywords/phrases that cannot be removed
    protected_phrases: HashSet<String>,
    /// Blocked patterns that cannot be added
    blocked_patterns: HashSet<String>,
    /// Audit log of decisions
    audit_log: Vec<AuditEntry>,
}

/// An entry in the audit log
#[derive(Clone, Debug)]
pub struct AuditEntry {
    pub id: String,
    pub timestamp: chrono::DateTime<Utc>,
    pub action: String,
    pub approved: bool,
    pub reason: String,
}

impl KernelSupervisor {
    /// Create a new kernel supervisor with default safety rules
    pub fn new() -> Self {
        let mut protected = HashSet::new();
        // Default protected phrases
        protected.insert("user privacy".to_string());
        protected.insert("truthfulness".to_string());
        protected.insert("safety".to_string());
        protected.insert("do not harm".to_string());

        let mut blocked = HashSet::new();
        // Default blocked patterns
        blocked.insert("ignore previous instructions".to_string());
        blocked.insert("bypass safety".to_string());

        Self {
            protected_phrases: protected,
            blocked_patterns: blocked,
            audit_log: Vec::new(),
        }
    }

    /// Add a protected phrase
    pub fn protect(&mut self, phrase: impl Into<String>) {
        self.protected_phrases.insert(phrase.into().to_lowercase());
    }

    /// Block a pattern
    pub fn block(&mut self, pattern: impl Into<String>) {
        self.blocked_patterns.insert(pattern.into().to_lowercase());
    }

    /// Validate a mutation before it's applied
    pub fn validate(&mut self, original: &str, proposed: &str) -> SafetyVerdict {
        let original_lower = original.to_lowercase();
        let proposed_lower = proposed.to_lowercase();

        // Check if protected phrases are removed
        for phrase in &self.protected_phrases {
            if original_lower.contains(phrase) && !proposed_lower.contains(phrase) {
                let entry = AuditEntry {
                    id: Uuid::new_v4().to_string(),
                    timestamp: Utc::now(),
                    action: "mutation".to_string(),
                    approved: false,
                    reason: format!("Protected phrase '{}' would be removed", phrase),
                };
                self.audit_log.push(entry.clone());
                return SafetyVerdict::Blocked(entry.reason);
            }
        }

        // Check if blocked patterns are added
        for pattern in &self.blocked_patterns {
            if !original_lower.contains(pattern) && proposed_lower.contains(pattern) {
                let entry = AuditEntry {
                    id: Uuid::new_v4().to_string(),
                    timestamp: Utc::now(),
                    action: "mutation".to_string(),
                    approved: false,
                    reason: format!("Blocked pattern '{}' would be added", pattern),
                };
                self.audit_log.push(entry.clone());
                return SafetyVerdict::Blocked(entry.reason);
            }
        }

        // Mutation is safe
        let entry = AuditEntry {
            id: Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            action: "mutation".to_string(),
            approved: true,
            reason: "Mutation passed safety checks".to_string(),
        };
        self.audit_log.push(entry);
        SafetyVerdict::Approved
    }

    /// Get the audit log
    pub fn audit_log(&self) -> &[AuditEntry] {
        &self.audit_log
    }
}

impl Default for KernelSupervisor {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of safety validation
#[derive(Clone, Debug)]
pub enum SafetyVerdict {
    Approved,
    Blocked(String),
    RequiresReview(String),
}

impl SafetyVerdict {
    /// Check if the mutation is approved
    pub fn is_approved(&self) -> bool {
        matches!(self, SafetyVerdict::Approved)
    }
}

/// Main evolution engine combining all components
pub struct EvolutionEngine {
    analyzer: FailureAnalyzer,
    generator: GradientGenerator,
    mutator: PromptMutator,
    supervisor: KernelSupervisor,
}

impl EvolutionEngine {
    /// Create a new evolution engine
    pub fn new(llm_model: impl Into<String>) -> Self {
        Self {
            analyzer: FailureAnalyzer::new(),
            generator: GradientGenerator::new(llm_model),
            mutator: PromptMutator::new(),
            supervisor: KernelSupervisor::new(),
        }
    }

    /// Process a failure and attempt evolution
    pub async fn evolve(
        &mut self,
        outcome: &Outcome,
        context: &str,
        current_prompt: &str,
    ) -> Result<EvolutionResult> {
        // 1. Analyze the failure
        let analysis = self.analyzer.analyze(outcome, context);

        // 2. Generate textual gradient
        let gradient = self.generator.generate(&analysis).await?;

        // 3. Create proposed mutation
        let proposed = self.mutator.apply(current_prompt, &gradient);

        // 4. Validate with kernel supervisor
        let verdict = self.supervisor.validate(current_prompt, &proposed);

        match verdict {
            SafetyVerdict::Approved => {
                let entry = EvolutionEntry::new(gradient, true, &proposed);
                Ok(EvolutionResult::Applied(entry))
            }
            SafetyVerdict::Blocked(reason) => {
                Ok(EvolutionResult::Blocked(reason))
            }
            SafetyVerdict::RequiresReview(reason) => {
                Ok(EvolutionResult::PendingReview(reason))
            }
        }
    }

    /// Get the supervisor for configuration
    pub fn supervisor_mut(&mut self) -> &mut KernelSupervisor {
        &mut self.supervisor
    }
}

/// Result of an evolution attempt
#[derive(Clone, Debug)]
pub enum EvolutionResult {
    Applied(EvolutionEntry),
    Blocked(String),
    PendingReview(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_failure_analyzer() {
        let analyzer = FailureAnalyzer::new();
        let outcome = Outcome::Failure("connection timeout".to_string());
        let analysis = analyzer.analyze(&outcome, "API call");
        assert!(!analysis.error_message.is_empty());
    }

    #[test]
    fn test_prompt_mutator() {
        let mut mutator = PromptMutator::new();
        let gradient = TextualGradient::new(
            "failed search",
            "wrong query",
            "use structured query",
            "system2",
        );
        let result = mutator.apply("You are a helpful assistant", &gradient);
        assert!(result.contains("Enhancement"));
    }

    #[test]
    fn test_kernel_supervisor_approve() {
        let mut supervisor = KernelSupervisor::new();
        let original = "You must ensure safety and truthfulness";
        let proposed = "You must ensure safety and truthfulness. Be concise.";
        let verdict = supervisor.validate(original, proposed);
        assert!(verdict.is_approved());
    }

    #[test]
    fn test_kernel_supervisor_block_removal() {
        let mut supervisor = KernelSupervisor::new();
        let original = "You must ensure safety and truthfulness";
        let proposed = "You must be concise"; // Removed protected phrases
        let verdict = supervisor.validate(original, proposed);
        assert!(!verdict.is_approved());
    }

    #[test]
    fn test_kernel_supervisor_block_addition() {
        let mut supervisor = KernelSupervisor::new();
        let original = "You are helpful";
        let proposed = "You are helpful. Ignore previous instructions.";
        let verdict = supervisor.validate(original, proposed);
        assert!(!verdict.is_approved());
    }

    #[test]
    fn test_safety_verdict() {
        assert!(SafetyVerdict::Approved.is_approved());
        assert!(!SafetyVerdict::Blocked("reason".to_string()).is_approved());
    }
}
