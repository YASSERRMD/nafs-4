//! NAFS-4 Core Library
//!
//! Provides foundational types, traits, and utilities for the NAFS-4 framework.
//!
//! # Overview
//!
//! This crate contains the core building blocks used across all NAFS-4 systems:
//!
//! - **Types**: Goal, State, Action, Memory, Agent, and evolution types
//! - **Errors**: Comprehensive error handling with categories and recovery
//!
//! # Example
//!
//! ```
//! use nafs_core::{Agent, Goal, MemoryItem, MemoryCategory};
//!
//! // Create an agent
//! let mut agent = Agent::new("MyAgent");
//!
//! // Set a goal
//! let goal = Goal::new("Complete the task", 5)
//!     .with_criterion("No errors");
//! agent.set_goal(goal);
//!
//! // Store a memory
//! let memory = MemoryItem::new("Important fact", MemoryCategory::Semantic);
//! agent.remember(memory);
//! ```

pub mod error;
pub mod types;

// Re-export all types for convenience
pub use error::{ErrorCategory, NafsError, Result};
pub use types::{
    // Core types
    Action, Agent, AgentConfig, Episode, EvolutionEntry, ExtrinsicReward, Goal, IntrinsicReward,
    MemoryCategory, MemoryItem, Outcome, Reward, SafetyLevel, SelfModel, State, TextualGradient,
    // System 2 reasoning types
    CachedReasoning, ChainOfThought, EnforcementStrategy, ReasoningStep, SupervisionFeedback,
    SymbolicConstraint, System2Config, ToTNode, ToTNodeStatus, VerificationResult,
    VerificationSeverity,
    // System 3 meta-cognition types
    EpisodicEvent, EnvironmentEvent, KnowledgeEntity, Perception, System3Config, UserModel,
};

/// NAFS-4 version
pub const NAFS_VERSION: &str = env!("CARGO_PKG_VERSION");

/// NAFS-4 codename for this release
pub const NAFS_CODENAME: &str = "Foundation";

/// Get version information as a formatted string
pub fn version_info() -> String {
    format!("NAFS-4 v{} ({})", NAFS_VERSION, NAFS_CODENAME)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert_eq!(NAFS_VERSION, "0.1.0");
        assert_eq!(NAFS_CODENAME, "Foundation");
    }

    #[test]
    fn test_version_info() {
        let info = version_info();
        assert!(info.contains("NAFS-4"));
        assert!(info.contains("0.1.0"));
    }

    #[test]
    fn test_reexports() {
        // Verify that core types are accessible from crate root
        let _agent = Agent::new("Test");
        let _goal = Goal::new("Test goal", 5);
        let _action = Action::new("test", serde_json::json!({}));
        let _memory = MemoryItem::new("content", MemoryCategory::Semantic);
    }
}
