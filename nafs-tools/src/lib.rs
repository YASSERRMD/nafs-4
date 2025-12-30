//! NAFS-4 Tool Management
//!
//! Provides infrastructure for defining, registering, and executing tools:
//! - Tool trait for implementing custom tools
//! - ToolRegistry for managing available tools
//! - ToolExecutor for running tools safely

use async_trait::async_trait;
use nafs_core::{Action, NafsError, Result, SafetyLevel};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Definition of a tool parameter
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolParameter {
    pub name: String,
    pub description: String,
    pub param_type: ParameterType,
    pub required: bool,
    pub default: Option<serde_json::Value>,
}

/// Type of a tool parameter
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ParameterType {
    String,
    Integer,
    Float,
    Boolean,
    Array,
    Object,
}

/// Definition of a tool for LLM function calling
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Vec<ToolParameter>,
    pub safety_level: SafetyLevel,
}

/// Result of tool execution
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolResult {
    pub success: bool,
    pub output: serde_json::Value,
    pub error: Option<String>,
    pub duration_ms: u64,
}

impl ToolResult {
    /// Create a successful result
    pub fn success(output: serde_json::Value) -> Self {
        Self {
            success: true,
            output,
            error: None,
            duration_ms: 0,
        }
    }

    /// Create a failed result
    pub fn failure(error: impl Into<String>) -> Self {
        Self {
            success: false,
            output: serde_json::Value::Null,
            error: Some(error.into()),
            duration_ms: 0,
        }
    }

    /// Set duration
    pub fn with_duration(mut self, ms: u64) -> Self {
        self.duration_ms = ms;
        self
    }
}

/// Trait for implementing tools
#[async_trait]
pub trait Tool: Send + Sync {
    /// Get the tool definition
    fn definition(&self) -> ToolDefinition;

    /// Execute the tool with given parameters
    async fn execute(&self, params: serde_json::Value) -> Result<ToolResult>;

    /// Validate parameters before execution
    fn validate(&self, params: &serde_json::Value) -> Result<()> {
        // Default validation just checks if params is an object
        if !params.is_object() && !params.is_null() {
            return Err(NafsError::validation("Parameters must be an object"));
        }
        Ok(())
    }
}

/// Registry for managing available tools
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Register a tool
    pub fn register(&mut self, tool: Arc<dyn Tool>) {
        let def = tool.definition();
        self.tools.insert(def.name.clone(), tool);
    }

    /// Get a tool by name
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    /// List all registered tools
    pub fn list(&self) -> Vec<ToolDefinition> {
        self.tools.values().map(|t| t.definition()).collect()
    }

    /// Get the number of registered tools
    pub fn count(&self) -> usize {
        self.tools.len()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Executor for running tools with safety checks
pub struct ToolExecutor {
    registry: ToolRegistry,
    timeout_ms: u64,
}

impl ToolExecutor {
    /// Create a new executor with the given registry
    pub fn new(registry: ToolRegistry) -> Self {
        Self {
            registry,
            timeout_ms: 30000, // 30 seconds default
        }
    }

    /// Set the execution timeout
    pub fn with_timeout(mut self, ms: u64) -> Self {
        self.timeout_ms = ms;
        self
    }

    /// Execute an action using registered tools
    pub async fn execute(&self, action: &Action) -> Result<ToolResult> {
        let tool = self.registry.get(&action.tool_name).ok_or_else(|| {
            NafsError::tool(format!("Tool '{}' not found", action.tool_name))
        })?;

        // Validate parameters
        tool.validate(&action.parameters)?;

        let start = std::time::Instant::now();

        // Execute the tool
        let result = tool.execute(action.parameters.clone()).await?;

        Ok(result.with_duration(start.elapsed().as_millis() as u64))
    }
}

/// A simple echo tool for testing
pub struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "echo".to_string(),
            description: "Echoes the input message back".to_string(),
            parameters: vec![ToolParameter {
                name: "message".to_string(),
                description: "Message to echo".to_string(),
                param_type: ParameterType::String,
                required: true,
                default: None,
            }],
            safety_level: SafetyLevel::Safe,
        }
    }

    async fn execute(&self, params: serde_json::Value) -> Result<ToolResult> {
        let message = params
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or("No message provided");
        
        Ok(ToolResult::success(serde_json::json!({
            "echo": message
        })))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_registry() {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(EchoTool));
        
        assert_eq!(registry.count(), 1);
        assert!(registry.get("echo").is_some());
        assert!(registry.get("unknown").is_none());
    }

    #[test]
    fn test_tool_definition() {
        let tool = EchoTool;
        let def = tool.definition();
        
        assert_eq!(def.name, "echo");
        assert_eq!(def.safety_level, SafetyLevel::Safe);
    }

    #[tokio::test]
    async fn test_echo_tool() {
        let tool = EchoTool;
        let params = serde_json::json!({"message": "Hello, World!"});
        
        let result = tool.execute(params).await.unwrap();
        assert!(result.success);
        assert_eq!(result.output["echo"], "Hello, World!");
    }

    #[tokio::test]
    async fn test_tool_executor() {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(EchoTool));
        
        let executor = ToolExecutor::new(registry);
        let action = Action::new("echo", serde_json::json!({"message": "Test"}));
        
        let result = executor.execute(&action).await.unwrap();
        assert!(result.success);
    }

    #[test]
    fn test_tool_result_success() {
        let result = ToolResult::success(serde_json::json!({"data": 123}));
        assert!(result.success);
        assert!(result.error.is_none());
    }

    #[test]
    fn test_tool_result_failure() {
        let result = ToolResult::failure("Something went wrong");
        assert!(!result.success);
        assert!(result.error.is_some());
    }
}
