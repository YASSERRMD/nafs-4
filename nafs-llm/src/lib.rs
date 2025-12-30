//! NAFS-4 LLM Integration
//!
//! Provides abstracted interfaces for LLM providers:
//! - OpenAI (and compatible APIs like Ollama, LocalAI)
//! - Anthropic (Claude)
//! - Google (Gemini)
//!
//! Designed for easy provider switching and fallback chains.

use async_trait::async_trait;
use nafs_core::{NafsError, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;

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
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: MessageRole::System, content: content.into() }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: MessageRole::User, content: content.into() }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: MessageRole::Assistant, content: content.into() }
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
    pub fn for_model(model: impl Into<String>) -> Self {
        Self { model: model.into(), ..Default::default() }
    }
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = temp.clamp(0.0, 2.0);
        self
    }
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
    ContentFilter,
    ToolCall,
    Unknown,
}

#[derive(Clone, Debug, Default)]
pub struct TokenUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[async_trait]
pub trait LLMProvider: Send + Sync {
    fn name(&self) -> &str;
    fn available_models(&self) -> Vec<String>;
    async fn chat(&self, messages: &[ChatMessage], config: &ChatConfig) -> Result<ChatResponse>;
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;
    async fn health_check(&self) -> Result<bool>;
}

// ==========================================
// OpenAI Provider (Standard & Compatible)
// ==========================================

#[derive(Clone, Debug)]
pub struct OpenAIConfig {
    pub api_key: String,
    pub base_url: String, // e.g., "https://api.openai.com/v1" or "http://localhost:11434/v1"
    pub organization: Option<String>,
}

impl OpenAIConfig {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://api.openai.com/v1".to_string(),
            organization: None,
        }
    }
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }
    pub fn with_organization(mut self, org: impl Into<String>) -> Self {
        self.organization = Some(org.into());
        self
    }
}

pub struct OpenAIProvider {
    config: OpenAIConfig,
    client: reqwest::Client,
}

impl OpenAIProvider {
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
        vec!["gpt-4".to_string(), "gpt-3.5-turbo".to_string(), "gpt-4-turbo".to_string()]
    }

    async fn chat(&self, messages: &[ChatMessage], config: &ChatConfig) -> Result<ChatResponse> {
        let url = format!("{}/chat/completions", self.config.base_url.trim_end_matches('/'));
        
        // Convert messages
        let msgs: Vec<_> = messages.iter().map(|m| json!({
            "role": match m.role {
                MessageRole::System => "system",
                MessageRole::User => "user",
                MessageRole::Assistant => "assistant",
            },
            "content": m.content
        })).collect();

        let body = json!({
            "model": config.model,
            "messages": msgs,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p,
        });

        let mut req = self.client.post(&url)
            .header("Content-Type", "application/json");
            
        // Add auth if key is present (some local backends don't need it)
        if !self.config.api_key.is_empty() {
            req = req.header("Authorization", format!("Bearer {}", self.config.api_key));
        }
        if let Some(org) = &self.config.organization {
            req = req.header("OpenAI-Organization", org);
        }

        let resp = req.json(&body)
            .send()
            .await
            .map_err(|e| NafsError::llm(format!("Request failed: {}", e)))?;

        if !resp.status().is_success() {
            let error_text = resp.text().await.unwrap_or_default();
            return Err(NafsError::llm(format!("OpenAI API error: {}", error_text)));
        }

        let data: serde_json::Value = resp.json()
            .await
            .map_err(|e| NafsError::llm(format!("Parse error: {}", e)))?;

        let choice = data["choices"].get(0)
            .ok_or_else(|| NafsError::llm("No choices in response"))?;
            
        let content = choice["message"]["content"].as_str()
            .unwrap_or_default()
            .to_string();
            
        let finish_reason = match choice["finish_reason"].as_str() {
            Some("stop") => FinishReason::Stop,
            Some("length") => FinishReason::Length,
            Some("content_filter") => FinishReason::ContentFilter,
            _ => FinishReason::Unknown,
        };

        let usage = data["usage"].as_object();
        let usage_stats = TokenUsage {
            prompt_tokens: usage.and_then(|u| u["prompt_tokens"].as_u64()).unwrap_or(0) as usize,
            completion_tokens: usage.and_then(|u| u["completion_tokens"].as_u64()).unwrap_or(0) as usize,
            total_tokens: usage.and_then(|u| u["total_tokens"].as_u64()).unwrap_or(0) as usize,
        };

        Ok(ChatResponse {
            content,
            finish_reason,
            usage: usage_stats,
        })
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let url = format!("{}/embeddings", self.config.base_url.trim_end_matches('/'));
        let body = json!({
            "model": "text-embedding-ada-002", // Default for now
            "input": text
        });

        let resp = self.client.post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .json(&body)
            .send()
            .await
            .map_err(|e| NafsError::llm(format!("Request failed: {}", e)))?;

        if !resp.status().is_success() {
            return Err(NafsError::llm(format!("OpenAI Embeddings error: {}", resp.text().await.unwrap_or_default())));
        }

        let data: serde_json::Value = resp.json().await.unwrap();
        let embedding = data["data"][0]["embedding"].as_array()
            .ok_or_else(|| NafsError::llm("Invalid embedding response"))?
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0) as f32)
            .collect();
            
        Ok(embedding)
    }

    async fn health_check(&self) -> Result<bool> {
        // Try listing models as health check
        let url = format!("{}/models", self.config.base_url.trim_end_matches('/'));
        let resp = self.client.get(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .send()
            .await;
            
        Ok(resp.map(|r| r.status().is_success()).unwrap_or(false))
    }
}

// ==========================================
// Anthropic Provider
// ==========================================

#[derive(Clone, Debug)]
pub struct AnthropicConfig {
    pub api_key: String,
    pub base_url: String, 
}

impl AnthropicConfig {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://api.anthropic.com/v1".to_string(),
        }
    }
}

pub struct AnthropicProvider {
    config: AnthropicConfig,
    client: reqwest::Client,
}

impl AnthropicProvider {
    pub fn new(config: AnthropicConfig) -> Self {
        Self { config, client: reqwest::Client::new() }
    }
}

#[async_trait]
impl LLMProvider for AnthropicProvider {
    fn name(&self) -> &str {
        "anthropic"
    }

    fn available_models(&self) -> Vec<String> {
        vec!["claude-3-opus-20240229".to_string(), "claude-3-sonnet-20240229".to_string()]
    }

    async fn chat(&self, messages: &[ChatMessage], config: &ChatConfig) -> Result<ChatResponse> {
        let url = format!("{}/messages", self.config.base_url.trim_end_matches('/'));
        
        let mut system_prompt = String::new();
        let mut anthropic_msgs = Vec::new();
        
        for msg in messages {
            match msg.role {
                MessageRole::System => system_prompt = msg.content.clone(),
                _ => anthropic_msgs.push(json!({
                    "role": if msg.role == MessageRole::User { "user" } else { "assistant" },
                    "content": msg.content
                }))
            }
        }

        let body = json!({
            "model": config.model,
            "messages": anthropic_msgs,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "system": system_prompt
        });

        let resp = self.client.post(&url)
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| NafsError::llm(format!("Request failed: {}", e)))?;

        if !resp.status().is_success() {
            return Err(NafsError::llm(format!("Anthropic API error: {}", resp.text().await.unwrap_or_default())));
        }

        let data: serde_json::Value = resp.json().await.unwrap();
        let content = data["content"][0]["text"].as_str().unwrap_or_default().to_string();
        
        Ok(ChatResponse {
            content,
            finish_reason: FinishReason::Stop,
            usage: TokenUsage::default(), // Anthropic usage parsing can be added
        })
    }

    async fn embed(&self, _text: &str) -> Result<Vec<f32>> {
        Err(NafsError::llm("Anthropic does not support embeddings endpoint natively yet"))
    }

    async fn health_check(&self) -> Result<bool> {
        // No simple list models endpoint public without parsing, so assume true if struct built
        Ok(true) 
    }
}

// ==========================================
// Google Gemini Provider
// ==========================================

#[derive(Clone, Debug)]
pub struct GeminiConfig {
    pub api_key: String,
}

impl GeminiConfig {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self { api_key: api_key.into() }
    }
}

pub struct GeminiProvider {
    config: GeminiConfig,
    client: reqwest::Client,
}

impl GeminiProvider {
    pub fn new(config: GeminiConfig) -> Self {
        Self { config, client: reqwest::Client::new() }
    }
}

#[async_trait]
impl LLMProvider for GeminiProvider {
    fn name(&self) -> &str { "google" }
    fn available_models(&self) -> Vec<String> { vec!["gemini-pro".to_string()] }

    async fn chat(&self, messages: &[ChatMessage], config: &ChatConfig) -> Result<ChatResponse> {
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}", 
            config.model, self.config.api_key
        );

        let contents: Vec<_> = messages.iter().map(|m| json!({
            "role": if m.role == MessageRole::User { "user" } else { "model" },
            "parts": [{ "text": m.content }]
        })).collect();

        let body = json!({
            "contents": contents,
            "generationConfig": {
                "temperature": config.temperature,
                "maxOutputTokens": config.max_tokens,
            }
        });

        let resp = self.client.post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| NafsError::llm(format!("Request failed: {}", e)))?;

        if !resp.status().is_success() {
            return Err(NafsError::llm(format!("Gemini API error: {}", resp.text().await.unwrap_or_default())));
        }

        let data: serde_json::Value = resp.json().await.unwrap();
        let content = data["candidates"][0]["content"]["parts"][0]["text"]
            .as_str()
            .unwrap_or_default()
            .to_string();

        Ok(ChatResponse {
            content,
            finish_reason: FinishReason::Stop,
            usage: TokenUsage::default(),
        })
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
         let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent?key={}", 
            self.config.api_key
        );
        let body = json!({
            "model": "models/embedding-001",
            "content": { "parts": [{ "text": text }] }
        });

        let resp = self.client.post(&url).json(&body).send().await
            .map_err(|e| NafsError::llm(format!("Gemini Embed failed: {}", e)))?;

        let data: serde_json::Value = resp.json().await.unwrap();
        let embedding = data["embedding"]["values"].as_array()
            .ok_or_else(|| NafsError::llm("Invalid embedding"))?
            .iter().map(|v| v.as_f64().unwrap() as f32).collect();

        Ok(embedding)
    }

    async fn health_check(&self) -> Result<bool> { Ok(true) }
}

// ==========================================
// Mock & Chain (Preserved)
// ==========================================

pub struct MockLLMProvider {
    name: String,
    responses: std::sync::RwLock<Vec<String>>,
}

impl MockLLMProvider {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), responses: std::sync::RwLock::new(Vec::new()) }
    }
    pub fn add_response(&self, response: impl Into<String>) {
        self.responses.write().unwrap().push(response.into());
    }
}

#[async_trait]
impl LLMProvider for MockLLMProvider {
    fn name(&self) -> &str { &self.name }
    fn available_models(&self) -> Vec<String> { vec!["mock".to_string()] }
    async fn chat(&self, _msgs: &[ChatMessage], _cfg: &ChatConfig) -> Result<ChatResponse> {
        let content = self.responses.write().unwrap().pop().unwrap_or("Mock".to_string());
        Ok(ChatResponse { content, finish_reason: FinishReason::Stop, usage: Default::default() })
    }
    async fn embed(&self, _text: &str) -> Result<Vec<f32>> { Ok(vec![0.0; 1536]) }
    async fn health_check(&self) -> Result<bool> { Ok(true) }
}

pub struct ProviderChain {
    providers: Vec<Box<dyn LLMProvider>>,
}

impl ProviderChain {
    pub fn new() -> Self { Self { providers: Vec::new() } }
    pub fn add(mut self, provider: Box<dyn LLMProvider>) -> Self {
        self.providers.push(provider);
        self
    }
    pub async fn chat(&self, msgs: &[ChatMessage], cfg: &ChatConfig) -> Result<ChatResponse> {
        for p in &self.providers {
            match p.chat(msgs, cfg).await {
                Ok(r) => return Ok(r),
                Err(e) => tracing::warn!("{} failed: {}", p.name(), e),
            }
        }
        Err(NafsError::llm("All providers failed"))
    }
}
