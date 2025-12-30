//! NAFS-4 LLM Integration
//!
//! Provides abstracted interfaces for LLM providers:
//! - OpenAI (GPT-4, etc.)
//! - Anthropic (Claude)
//! - Local models (Ollama, etc.)
//!
//! Designed for easy provider switching and fallback chains.

use async_trait::async_trait;
use nafs_core::{NafsError, Result};
use serde::{Deserialize, Serialize};

/// A message in a conversation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: MessageRole,
    pub content: String,
}

/// Role of a message sender
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    System,
    User,
    Assistant,
}

impl ChatMessage {
    /// Create a system message
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::System,
            content: content.into(),
        }
    }

    /// Create a user message
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: content.into(),
        }
    }

    /// Create an assistant message
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: content.into(),
        }
    }
}

/// Configuration for chat completion
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatConfig {
    pub model: String,
    pub temperature: f32,
    pub max_tokens: usize,
    pub top_p: f32,
    pub stop_sequences: Vec<String>,
}

impl Default for ChatConfig {
    fn default() -> Self {
        Self {
            model: "gpt-4".to_string(),
            temperature: 0.7,
            max_tokens: 2000,
            top_p: 1.0,
            stop_sequences: Vec::new(),
        }
    }
}

impl ChatConfig {
    /// Create a config for a specific model
    pub fn for_model(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            ..Default::default()
        }
    }

    /// Set temperature
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = temp.clamp(0.0, 2.0);
        self
    }

    /// Set max tokens
    pub fn with_max_tokens(mut self, tokens: usize) -> Self {
        self.max_tokens = tokens;
        self
    }
}

/// Response from chat completion
#[derive(Clone, Debug)]
pub struct ChatResponse {
    pub content: String,
    pub finish_reason: FinishReason,
    pub usage: TokenUsage,
}

/// Reason for completion finish
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
    ContentFilter,
    ToolCall,
    Unknown,
}

/// Token usage statistics
#[derive(Clone, Debug, Default)]
pub struct TokenUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// Trait for LLM providers
#[async_trait]
pub trait LLMProvider: Send + Sync {
    /// Get the provider name
    fn name(&self) -> &str;

    /// List available models
    fn available_models(&self) -> Vec<String>;

    /// Complete a chat conversation
    async fn chat(&self, messages: &[ChatMessage], config: &ChatConfig) -> Result<ChatResponse>;

    /// Generate embeddings for text
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Check if the provider is available
    async fn health_check(&self) -> Result<bool>;
}

/// Mock LLM provider for testing
pub struct MockLLMProvider {
    name: String,
    responses: std::sync::RwLock<Vec<String>>,
}

impl MockLLMProvider {
    /// Create a new mock provider
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            responses: std::sync::RwLock::new(Vec::new()),
        }
    }

    /// Add a canned response
    pub fn add_response(&self, response: impl Into<String>) {
        let mut responses = self.responses.write().unwrap();
        responses.push(response.into());
    }
}

#[async_trait]
impl LLMProvider for MockLLMProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn available_models(&self) -> Vec<String> {
        vec!["mock-model".to_string()]
    }

    async fn chat(&self, _messages: &[ChatMessage], _config: &ChatConfig) -> Result<ChatResponse> {
        let mut responses = self.responses.write().unwrap();
        let content = responses.pop().unwrap_or_else(|| "Mock response".to_string());
        
        Ok(ChatResponse {
            content,
            finish_reason: FinishReason::Stop,
            usage: TokenUsage::default(),
        })
    }

    async fn embed(&self, _text: &str) -> Result<Vec<f32>> {
        // Return a simple mock embedding
        Ok(vec![0.1, 0.2, 0.3, 0.4, 0.5])
    }

    async fn health_check(&self) -> Result<bool> {
        Ok(true)
    }
}

/// OpenAI provider configuration
#[derive(Clone, Debug)]
pub struct OpenAIConfig {
    pub api_key: String,
    pub base_url: String,
    pub organization: Option<String>,
}

impl OpenAIConfig {
    /// Create from API key
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://api.openai.com/v1".to_string(),
            organization: None,
        }
    }

    /// Set organization
    pub fn with_organization(mut self, org: impl Into<String>) -> Self {
        self.organization = Some(org.into());
        self
    }

    /// Set custom base URL
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }
}

/// OpenAI provider (stub for now)
pub struct OpenAIProvider {
    config: OpenAIConfig,
    client: reqwest::Client,
}

impl OpenAIProvider {
    /// Create a new OpenAI provider
    pub fn new(config: OpenAIConfig) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl LLMProvider for OpenAIProvider {
    fn name(&self) -> &str {
        "openai"
    }

    fn available_models(&self) -> Vec<String> {
        vec![
            "gpt-4".to_string(),
            "gpt-4-turbo".to_string(),
            "gpt-3.5-turbo".to_string(),
        ]
    }

    async fn chat(&self, messages: &[ChatMessage], config: &ChatConfig) -> Result<ChatResponse> {
        // TODO: Implement actual OpenAI API call
        tracing::info!("OpenAI chat request with {} messages", messages.len());
        Err(NafsError::llm("OpenAI provider not yet implemented"))
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        // TODO: Implement actual OpenAI embeddings API call
        tracing::info!("OpenAI embed request for text of len {}", text.len());
        Err(NafsError::llm("OpenAI embeddings not yet implemented"))
    }

    async fn health_check(&self) -> Result<bool> {
        // TODO: Implement actual health check
        Ok(false)
    }
}

/// Provider chain for fallback behavior
pub struct ProviderChain {
    providers: Vec<Box<dyn LLMProvider>>,
}

impl ProviderChain {
    /// Create a new provider chain
    pub fn new() -> Self {
        Self {
            providers: Vec::new(),
        }
    }

    /// Add a provider to the chain
    pub fn add(mut self, provider: Box<dyn LLMProvider>) -> Self {
        self.providers.push(provider);
        self
    }

    /// Try providers in order until one succeeds
    pub async fn chat(&self, messages: &[ChatMessage], config: &ChatConfig) -> Result<ChatResponse> {
        for provider in &self.providers {
            match provider.chat(messages, config).await {
                Ok(response) => return Ok(response),
                Err(e) => {
                    tracing::warn!("Provider {} failed: {}, trying next", provider.name(), e);
                }
            }
        }
        Err(NafsError::llm("All providers failed"))
    }
}

impl Default for ProviderChain {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_message_creation() {
        let system = ChatMessage::system("You are helpful");
        let user = ChatMessage::user("Hello");
        let assistant = ChatMessage::assistant("Hi there!");
        
        assert_eq!(system.role, MessageRole::System);
        assert_eq!(user.role, MessageRole::User);
        assert_eq!(assistant.role, MessageRole::Assistant);
    }

    #[test]
    fn test_chat_config() {
        let config = ChatConfig::for_model("gpt-4")
            .with_temperature(0.5)
            .with_max_tokens(1000);
        
        assert_eq!(config.model, "gpt-4");
        assert_eq!(config.temperature, 0.5);
        assert_eq!(config.max_tokens, 1000);
    }

    #[tokio::test]
    async fn test_mock_provider() {
        let provider = MockLLMProvider::new("test");
        provider.add_response("Hello from mock!");
        
        let messages = vec![ChatMessage::user("Hi")];
        let config = ChatConfig::default();
        
        let response = provider.chat(&messages, &config).await.unwrap();
        assert_eq!(response.content, "Hello from mock!");
    }

    #[tokio::test]
    async fn test_mock_embeddings() {
        let provider = MockLLMProvider::new("test");
        let embedding = provider.embed("test text").await.unwrap();
        assert_eq!(embedding.len(), 5);
    }

    #[tokio::test]
    async fn test_mock_health_check() {
        let provider = MockLLMProvider::new("test");
        assert!(provider.health_check().await.unwrap());
    }
}
