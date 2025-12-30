//! Error handling for NAFS-4
//!
//! Provides a comprehensive error type hierarchy for all NAFS-4 operations.

use thiserror::Error;

/// Main error type for NAFS-4 operations
#[derive(Debug, Error)]
pub enum NafsError {
    /// Configuration-related errors
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Runtime execution errors
    #[error("Runtime error: {0}")]
    RuntimeError(String),

    /// Timeout errors
    #[error("Timeout: {0}")]
    TimeoutError(String),

    /// Validation errors
    #[error("Validation error: {0}")]
    ValidationError(String),

    /// LLM-related errors
    #[error("LLM error: {0}")]
    LLMError(String),

    /// Memory operation errors
    #[error("Memory error: {0}")]
    MemoryError(String),

    /// Tool execution errors
    #[error("Tool error: {0}")]
    ToolError(String),

    /// Safety constraint violations
    #[error("Safety violation: {0}")]
    SafetyError(String),

    /// Planning errors
    #[error("Planning error: {0}")]
    PlanningError(String),

    /// Evolution system errors
    #[error("Evolution error: {0}")]
    EvolutionError(String),

    /// Kernel supervisor errors
    #[error("Kernel error: {0}")]
    KernelError(String),

    /// IO errors
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// JSON serialization errors
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    /// Resource not found
    #[error("Not found: {0}")]
    NotFound(String),

    /// Resource already exists
    #[error("Already exists: {0}")]
    AlreadyExists(String),

    /// Operation not supported
    #[error("Not supported: {0}")]
    NotSupported(String),

    /// General serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Unknown/unexpected errors
    #[error("Unknown error: {0}")]
    Unknown(String),
}

impl NafsError {
    /// Create a configuration error
    pub fn config(msg: impl Into<String>) -> Self {
        NafsError::ConfigError(msg.into())
    }

    /// Create a runtime error
    pub fn runtime(msg: impl Into<String>) -> Self {
        NafsError::RuntimeError(msg.into())
    }

    /// Create a timeout error
    pub fn timeout(msg: impl Into<String>) -> Self {
        NafsError::TimeoutError(msg.into())
    }

    /// Create a validation error
    pub fn validation(msg: impl Into<String>) -> Self {
        NafsError::ValidationError(msg.into())
    }

    /// Create an LLM error
    pub fn llm(msg: impl Into<String>) -> Self {
        NafsError::LLMError(msg.into())
    }

    /// Create a memory error
    pub fn memory(msg: impl Into<String>) -> Self {
        NafsError::MemoryError(msg.into())
    }

    /// Create a tool error
    pub fn tool(msg: impl Into<String>) -> Self {
        NafsError::ToolError(msg.into())
    }

    /// Create a safety error
    pub fn safety(msg: impl Into<String>) -> Self {
        NafsError::SafetyError(msg.into())
    }

    /// Create a planning error
    pub fn planning(msg: impl Into<String>) -> Self {
        NafsError::PlanningError(msg.into())
    }

    /// Create an evolution error
    pub fn evolution(msg: impl Into<String>) -> Self {
        NafsError::EvolutionError(msg.into())
    }

    /// Create a kernel error
    pub fn kernel(msg: impl Into<String>) -> Self {
        NafsError::KernelError(msg.into())
    }

    /// Check if this is a recoverable error
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            NafsError::TimeoutError(_) |
            NafsError::ToolError(_) |
            NafsError::LLMError(_)
        )
    }

    /// Check if this is a safety-critical error
    pub fn is_safety_critical(&self) -> bool {
        matches!(
            self,
            NafsError::SafetyError(_) | NafsError::KernelError(_)
        )
    }
}

/// Error category for grouping
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    Configuration,
    Execution,
    Safety,
    External,
    Internal,
}

impl NafsError {
    /// Get the category of this error
    pub fn category(&self) -> ErrorCategory {
        match self {
            NafsError::ConfigError(_) | NafsError::ValidationError(_) => ErrorCategory::Configuration,
            NafsError::RuntimeError(_) | NafsError::TimeoutError(_) | NafsError::PlanningError(_) => {
                ErrorCategory::Execution
            }
            NafsError::SafetyError(_) | NafsError::KernelError(_) => ErrorCategory::Safety,
            NafsError::LLMError(_) | NafsError::IoError(_) => ErrorCategory::External,
            _ => ErrorCategory::Internal,
        }
    }
}

/// Result type alias for NAFS-4 operations
pub type Result<T> = std::result::Result<T, NafsError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = NafsError::config("invalid setting");
        assert!(matches!(err, NafsError::ConfigError(_)));
    }

    #[test]
    fn test_error_display() {
        let err = NafsError::safety("blocked mutation");
        assert_eq!(err.to_string(), "Safety violation: blocked mutation");
    }

    #[test]
    fn test_is_recoverable() {
        assert!(NafsError::timeout("slow response").is_recoverable());
        assert!(!NafsError::safety("blocked").is_recoverable());
    }

    #[test]
    fn test_is_safety_critical() {
        assert!(NafsError::safety("violation").is_safety_critical());
        assert!(NafsError::kernel("blocked").is_safety_critical());
        assert!(!NafsError::runtime("error").is_safety_critical());
    }

    #[test]
    fn test_error_category() {
        assert_eq!(NafsError::config("x").category(), ErrorCategory::Configuration);
        assert_eq!(NafsError::safety("x").category(), ErrorCategory::Safety);
        assert_eq!(NafsError::runtime("x").category(), ErrorCategory::Execution);
    }
}
