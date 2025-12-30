//! Core type system for NAFS-4
//!
//! Defines all fundamental types used across the framework:
//! - Goal: What the agent wants to achieve
//! - State: Current internal state of the agent
//! - Action: What the agent can do
//! - Reward: Feedback signal (extrinsic + intrinsic)
//! - Memory: Episodic and semantic storage
//! - Agent: The entity orchestrating everything

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use uuid::Uuid;

/// ============================================================================
/// CORE TYPES (Foundation of NAFS-4)
/// ============================================================================

/// Represents a goal the agent wants to achieve
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Goal {
    pub id: String,
    pub description: String,
    pub priority: u8,                    // 1-10, higher = more urgent
    pub deadline: Option<DateTime<Utc>>,
    pub success_criteria: Vec<String>,
}

impl Goal {
    /// Create a new goal with generated ID
    pub fn new(description: impl Into<String>, priority: u8) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            description: description.into(),
            priority,
            deadline: None,
            success_criteria: Vec::new(),
        }
    }

    /// Add a success criterion
    pub fn with_criterion(mut self, criterion: impl Into<String>) -> Self {
        self.success_criteria.push(criterion.into());
        self
    }

    /// Set a deadline
    pub fn with_deadline(mut self, deadline: DateTime<Utc>) -> Self {
        self.deadline = Some(deadline);
        self
    }
}

/// Represents the internal state of the agent at any moment
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct State {
    pub agent_id: String,
    pub timestamp: DateTime<Utc>,
    pub current_goal: Option<Goal>,
    pub context: HashMap<String, serde_json::Value>,
    pub internal_state: HashMap<String, f32>,
}

impl State {
    /// Create a new state for an agent
    pub fn new(agent_id: impl Into<String>) -> Self {
        Self {
            agent_id: agent_id.into(),
            timestamp: Utc::now(),
            current_goal: None,
            context: HashMap::new(),
            internal_state: HashMap::new(),
        }
    }

    /// Set the current goal
    pub fn with_goal(mut self, goal: Goal) -> Self {
        self.current_goal = Some(goal);
        self
    }

    /// Add context data
    pub fn with_context(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.context.insert(key.into(), value);
        self
    }
}

/// Represents an action the agent can take
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Action {
    pub id: String,
    pub tool_name: String,
    pub parameters: serde_json::Value,
    pub safety_level: SafetyLevel,
    pub estimated_cost: f32,             // Time or resource cost
}

impl Action {
    /// Create a new action
    pub fn new(tool_name: impl Into<String>, parameters: serde_json::Value) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            tool_name: tool_name.into(),
            parameters,
            safety_level: SafetyLevel::Medium,
            estimated_cost: 1.0,
        }
    }

    /// Set safety level
    pub fn with_safety_level(mut self, level: SafetyLevel) -> Self {
        self.safety_level = level;
        self
    }

    /// Set estimated cost
    pub fn with_cost(mut self, cost: f32) -> Self {
        self.estimated_cost = cost;
        self
    }
}

/// Safety level of an action
#[derive(Clone, Debug, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SafetyLevel {
    Critical,  // Needs kernel approval
    High,      // Needs symbolic verification
    #[default]
    Medium,    // Standard verification
    Low,       // Minimal checks
    Safe,      // No verification needed
}

impl SafetyLevel {
    /// Check if this level requires verification
    pub fn requires_verification(&self) -> bool {
        matches!(self, SafetyLevel::Critical | SafetyLevel::High | SafetyLevel::Medium)
    }

    /// Check if this level requires kernel approval
    pub fn requires_kernel_approval(&self) -> bool {
        matches!(self, SafetyLevel::Critical)
    }
}

/// External (task) reward
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExtrinsicReward {
    pub value: f32,                      // Typically 0.0-1.0
    pub success_flag: bool,
    pub feedback_text: Option<String>,
}

impl Default for ExtrinsicReward {
    fn default() -> Self {
        Self {
            value: 0.0,
            success_flag: false,
            feedback_text: None,
        }
    }
}

/// Internal (learning/curiosity) reward
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct IntrinsicReward {
    pub curiosity: f32,                  // Novelty of state
    pub mastery: f32,                    // Skill improvement
    pub autonomy: f32,                   // Freedom to choose
}

/// Combined reward signal
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Reward {
    pub extrinsic: ExtrinsicReward,
    pub intrinsic: IntrinsicReward,
    pub total: f32,                      // Weighted sum
    pub timestamp: DateTime<Utc>,
}

impl Reward {
    /// Create a new reward with computed total
    pub fn new(extrinsic: ExtrinsicReward, intrinsic: IntrinsicReward) -> Self {
        let total = extrinsic.value * 0.7 + 
                    (intrinsic.curiosity + intrinsic.mastery + intrinsic.autonomy) / 3.0 * 0.3;
        Self {
            extrinsic,
            intrinsic,
            total,
            timestamp: Utc::now(),
        }
    }
}

/// An episode in the agent's experience
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Episode {
    pub id: String,
    pub goal: Goal,
    pub action: Action,
    pub outcome: Outcome,
    pub reward: Reward,
    pub reflection: Option<String>,      // Agent's interpretation
    pub timestamp: DateTime<Utc>,
}

impl Episode {
    /// Create a new episode
    pub fn new(goal: Goal, action: Action, outcome: Outcome, reward: Reward) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            goal,
            action,
            outcome,
            reward,
            reflection: None,
            timestamp: Utc::now(),
        }
    }

    /// Add reflection to the episode
    pub fn with_reflection(mut self, reflection: impl Into<String>) -> Self {
        self.reflection = Some(reflection.into());
        self
    }
}

/// Outcome of an action
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Outcome {
    Success,
    PartialSuccess(f32),  // Percentage
    Failure(String),      // Reason
    Timeout,
    Error(String),
}

impl Outcome {
    /// Check if outcome is successful (fully or partially)
    pub fn is_success(&self) -> bool {
        matches!(self, Outcome::Success | Outcome::PartialSuccess(_))
    }

    /// Get success percentage (1.0 for full success, 0.0 for failure)
    pub fn success_rate(&self) -> f32 {
        match self {
            Outcome::Success => 1.0,
            Outcome::PartialSuccess(rate) => *rate,
            _ => 0.0,
        }
    }
}

/// Represents a memory item (episodic or semantic)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemoryItem {
    pub id: String,
    pub content: String,
    pub embedding: Vec<f32>,             // Vector representation for RAG
    pub category: MemoryCategory,
    pub relevance: f32,                  // For retrieval ranking
    pub timestamp: DateTime<Utc>,
}

impl MemoryItem {
    /// Create a new memory item
    pub fn new(content: impl Into<String>, category: MemoryCategory) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            content: content.into(),
            embedding: Vec::new(),
            category,
            relevance: 1.0,
            timestamp: Utc::now(),
        }
    }

    /// Set embedding vector
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = embedding;
        self
    }

    /// Set relevance score
    pub fn with_relevance(mut self, relevance: f32) -> Self {
        self.relevance = relevance;
        self
    }
}

/// Types of memory
#[derive(Clone, Debug, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum MemoryCategory {
    #[default]
    Episodic,                            // Specific experiences
    Semantic,                            // General knowledge
    Procedural,                          // How to do things
    MetaCognitive,                       // Knowledge about knowledge
}

/// Agent's self-model
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SelfModel {
    pub agent_id: String,
    pub identity: String,                // "I am a helpful assistant..."
    pub capabilities: HashMap<String, f32>, // Skill → proficiency (0.0-1.0)
    pub weaknesses: Vec<String>,
    pub terminal_values: Vec<String>,    // Immutable values/constraints
    pub personality: HashMap<String, f32>, // Big Five traits, etc.
}

impl SelfModel {
    /// Create a new self-model for an agent
    pub fn new(agent_id: impl Into<String>, identity: impl Into<String>) -> Self {
        Self {
            agent_id: agent_id.into(),
            identity: identity.into(),
            capabilities: HashMap::new(),
            weaknesses: Vec::new(),
            terminal_values: vec![
                "User privacy".to_string(),
                "Truthfulness".to_string(),
                "Safety".to_string(),
            ],
            personality: HashMap::new(),
        }
    }

    /// Add a capability with proficiency level
    pub fn with_capability(mut self, name: impl Into<String>, proficiency: f32) -> Self {
        self.capabilities.insert(name.into(), proficiency.clamp(0.0, 1.0));
        self
    }

    /// Add a weakness
    pub fn with_weakness(mut self, weakness: impl Into<String>) -> Self {
        self.weaknesses.push(weakness.into());
        self
    }

    /// Get proficiency for a capability (0.0 if unknown)
    pub fn proficiency(&self, capability: &str) -> f32 {
        self.capabilities.get(capability).copied().unwrap_or(0.0)
    }
}

/// Textual gradient for evolution
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TextualGradient {
    pub id: String,
    pub failed_action: String,
    pub root_cause: String,              // LLM analysis of why it failed
    pub suggested_fix: String,           // LLM suggestion for improvement
    pub target_module: String,           // Which module to update
    pub confidence: f32,                 // How confident in the fix (0.0-1.0)
}

impl TextualGradient {
    /// Create a new textual gradient
    pub fn new(
        failed_action: impl Into<String>,
        root_cause: impl Into<String>,
        suggested_fix: impl Into<String>,
        target_module: impl Into<String>,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            failed_action: failed_action.into(),
            root_cause: root_cause.into(),
            suggested_fix: suggested_fix.into(),
            target_module: target_module.into(),
            confidence: 0.5,
        }
    }

    /// Set confidence level
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }
}

/// Evolution log entry
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvolutionEntry {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub gradient: TextualGradient,
    pub approved_by_kernel: bool,
    pub applied_changes: String,         // What was changed
    pub performance_delta: f32,          // Impact on success rate
}

impl EvolutionEntry {
    /// Create a new evolution entry
    pub fn new(gradient: TextualGradient, approved: bool, changes: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            gradient,
            approved_by_kernel: approved,
            applied_changes: changes.into(),
            performance_delta: 0.0,
        }
    }

    /// Set performance delta
    pub fn with_performance_delta(mut self, delta: f32) -> Self {
        self.performance_delta = delta;
        self
    }
}

/// Configuration for the agent
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentConfig {
    pub agent_name: String,
    pub identity: String,
    pub llm_provider: String,            // "openai", "anthropic", "local"
    pub llm_model: String,
    pub temperature: f32,
    pub max_tokens: usize,
    pub memory_backend: String,          // "vectordb", "filesystem", "graph"
    pub enable_evolution: bool,
    pub evolution_schedule: String,      // Cron format: "0 0 * * *" (daily)
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            agent_name: "NAFS-Agent".to_string(),
            identity: "I am a helpful and persistent AI assistant.".to_string(),
            llm_provider: "openai".to_string(),
            llm_model: "gpt-4".to_string(),
            temperature: 0.7,
            max_tokens: 2000,
            memory_backend: "vectordb".to_string(),
            enable_evolution: true,
            evolution_schedule: "0 0 * * *".to_string(),
        }
    }
}

impl AgentConfig {
    /// Create a new config with a custom name
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            agent_name: name.into(),
            ..Default::default()
        }
    }

    /// Set temperature
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = temp.clamp(0.0, 2.0);
        self
    }

    /// Set LLM provider and model
    pub fn with_llm(mut self, provider: impl Into<String>, model: impl Into<String>) -> Self {
        self.llm_provider = provider.into();
        self.llm_model = model.into();
        self
    }

    /// Enable or disable evolution
    pub fn with_evolution(mut self, enabled: bool) -> Self {
        self.enable_evolution = enabled;
        self
    }
}

/// Main Agent struct
#[derive(Debug)]
pub struct Agent {
    pub id: String,
    pub config: AgentConfig,
    pub self_model: SelfModel,
    pub state: State,
    pub memory: VecDeque<MemoryItem>,
    pub evolution_log: VecDeque<EvolutionEntry>,
}

impl Agent {
    /// Create a new agent with the given name
    pub fn new(name: impl Into<String>) -> Self {
        let name = name.into();
        let agent_id = Uuid::new_v4().to_string();
        let config = AgentConfig::new(&name);

        let self_model = SelfModel::new(&agent_id, &config.identity);
        let state = State::new(&agent_id);

        Self {
            id: agent_id,
            config,
            self_model,
            state,
            memory: VecDeque::new(),
            evolution_log: VecDeque::new(),
        }
    }

    /// Create an agent with custom configuration
    pub fn with_config(config: AgentConfig) -> Self {
        let agent_id = Uuid::new_v4().to_string();
        let self_model = SelfModel::new(&agent_id, &config.identity);
        let state = State::new(&agent_id);

        Self {
            id: agent_id,
            config,
            self_model,
            state,
            memory: VecDeque::new(),
            evolution_log: VecDeque::new(),
        }
    }

    /// Get agent name
    pub fn name(&self) -> &str {
        &self.config.agent_name
    }

    /// Get agent identity
    pub fn identity(&self) -> &str {
        &self.self_model.identity
    }

    /// Store a memory item
    pub fn remember(&mut self, item: MemoryItem) {
        self.memory.push_back(item);
        // Limit memory size (simple FIFO eviction)
        if self.memory.len() > 1000 {
            self.memory.pop_front();
        }
    }

    /// Recall memories by category
    pub fn recall_by_category(&self, category: MemoryCategory) -> Vec<&MemoryItem> {
        self.memory
            .iter()
            .filter(|m| m.category == category)
            .collect()
    }

    /// Record an evolution entry
    pub fn log_evolution(&mut self, entry: EvolutionEntry) {
        self.evolution_log.push_back(entry);
    }

    /// Get all capabilities with their proficiency levels
    pub fn capabilities(&self) -> Vec<(String, f32)> {
        self.self_model
            .capabilities
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect()
    }

    /// Set the current goal
    pub fn set_goal(&mut self, goal: Goal) {
        self.state.current_goal = Some(goal);
        self.state.timestamp = Utc::now();
    }

    /// Get memory count
    pub fn memory_count(&self) -> usize {
        self.memory.len()
    }

    /// Get evolution log count
    pub fn evolution_count(&self) -> usize {
        self.evolution_log.len()
    }
}

/// ============================================================================
/// REASONING LAYER TYPES (System 2 - Aql)
/// ============================================================================

/// A single reasoning step in the chain-of-thought
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReasoningStep {
    pub id: String,
    pub thought: String,                 // The thought/insight
    pub justification: String,           // Why this thought matters
    pub next_action: Option<Action>,     // Proposed action
    pub confidence: f32,                 // Confidence in this step (0.0-1.0)
    pub timestamp: DateTime<Utc>,
}

impl ReasoningStep {
    /// Create a new reasoning step
    pub fn new(thought: impl Into<String>, justification: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            thought: thought.into(),
            justification: justification.into(),
            next_action: None,
            confidence: 0.5,
            timestamp: Utc::now(),
        }
    }

    /// Set confidence level
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Set the proposed action
    pub fn with_action(mut self, action: Action) -> Self {
        self.next_action = Some(action);
        self
    }
}

/// A complete chain-of-thought sequence
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChainOfThought {
    pub id: String,
    pub goal: Goal,
    pub steps: Vec<ReasoningStep>,
    pub final_plan: Vec<Action>,
    pub reasoning_quality: f32,          // Overall quality score (0.0-1.0)
    pub generated_at: DateTime<Utc>,
}

impl ChainOfThought {
    /// Create a new chain-of-thought for a goal
    pub fn new(goal: Goal) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            goal,
            steps: Vec::new(),
            final_plan: Vec::new(),
            reasoning_quality: 0.0,
            generated_at: Utc::now(),
        }
    }

    /// Add a reasoning step
    pub fn add_step(&mut self, step: ReasoningStep) {
        self.steps.push(step);
        // Update quality based on average confidence
        if !self.steps.is_empty() {
            self.reasoning_quality = self.steps.iter().map(|s| s.confidence).sum::<f32>()
                / self.steps.len() as f32;
        }
    }

    /// Set the final plan
    pub fn with_plan(mut self, plan: Vec<Action>) -> Self {
        self.final_plan = plan;
        self
    }
}

/// A node in the Tree-of-Thought search
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToTNode {
    pub id: String,
    pub depth: usize,
    pub parent_id: Option<String>,
    pub children_ids: Vec<String>,
    pub partial_plan: Vec<Action>,
    pub value_estimate: f32,             // Estimated value (0.0-1.0)
    pub status: ToTNodeStatus,
}

impl ToTNode {
    /// Create a root node
    pub fn root() -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            depth: 0,
            parent_id: None,
            children_ids: Vec::new(),
            partial_plan: Vec::new(),
            value_estimate: 1.0,
            status: ToTNodeStatus::Pending,
        }
    }

    /// Create a child node
    pub fn child(parent: &ToTNode) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            depth: parent.depth + 1,
            parent_id: Some(parent.id.clone()),
            children_ids: Vec::new(),
            partial_plan: parent.partial_plan.clone(),
            value_estimate: 0.5,
            status: ToTNodeStatus::Pending,
        }
    }

    /// Set value estimate
    pub fn with_value(mut self, value: f32) -> Self {
        self.value_estimate = value.clamp(0.0, 1.0);
        self
    }

    /// Update status
    pub fn set_status(&mut self, status: ToTNodeStatus) {
        self.status = status;
    }

    /// Check if this is a leaf node
    pub fn is_leaf(&self) -> bool {
        self.children_ids.is_empty()
    }
}

/// Status of a ToT node
#[derive(Clone, Debug, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ToTNodeStatus {
    #[default]
    Pending,                             // Not yet evaluated
    Approved,                            // Passed verification
    Pruned,                              // Failed verification
    Selected,                            // Chosen as best path
}

/// A symbolic constraint the agent must respect
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SymbolicConstraint {
    pub id: String,
    pub rule: String,                    // Human-readable rule
    pub condition: String,               // Logic to check
    pub enforcement: EnforcementStrategy,
}

impl SymbolicConstraint {
    /// Create a new hard constraint
    pub fn hard(rule: impl Into<String>, condition: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            rule: rule.into(),
            condition: condition.into(),
            enforcement: EnforcementStrategy::Hard,
        }
    }

    /// Create a soft constraint
    pub fn soft(rule: impl Into<String>, condition: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            rule: rule.into(),
            condition: condition.into(),
            enforcement: EnforcementStrategy::Soft,
        }
    }
}

/// How strictly to enforce a constraint
#[derive(Clone, Debug, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum EnforcementStrategy {
    #[default]
    Hard,       // Always block violations
    Soft,       // Log warnings but allow
    Advisory,   // Just inform user
}

/// Result of verification
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VerificationResult {
    pub passes_logic: bool,
    pub issues: Vec<String>,             // List of problems found
    pub corrected_plan: Option<Vec<Action>>, // If correctable
    pub severity: VerificationSeverity,
}

impl VerificationResult {
    /// Create a passing result
    pub fn pass() -> Self {
        Self {
            passes_logic: true,
            issues: Vec::new(),
            corrected_plan: None,
            severity: VerificationSeverity::Info,
        }
    }

    /// Create a failing result
    pub fn fail(issues: Vec<String>) -> Self {
        Self {
            passes_logic: false,
            issues,
            corrected_plan: None,
            severity: VerificationSeverity::Error,
        }
    }

    /// Add a corrected plan
    pub fn with_correction(mut self, plan: Vec<Action>) -> Self {
        self.corrected_plan = Some(plan);
        self
    }
}

/// Severity of verification issues
#[derive(Clone, Debug, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum VerificationSeverity {
    Error,
    Warning,
    #[default]
    Info,
}

/// Process supervision feedback
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SupervisionFeedback {
    pub reasoning_coherence: f32,        // 0.0-1.0
    pub action_relevance: f32,           // Are actions relevant to goal?
    pub plan_completeness: f32,          // Does plan fully address goal?
    pub issues: Vec<String>,
    pub suggestions: Vec<String>,
}

impl SupervisionFeedback {
    /// Create new supervision feedback
    pub fn new(coherence: f32, relevance: f32, completeness: f32) -> Self {
        Self {
            reasoning_coherence: coherence.clamp(0.0, 1.0),
            action_relevance: relevance.clamp(0.0, 1.0),
            plan_completeness: completeness.clamp(0.0, 1.0),
            issues: Vec::new(),
            suggestions: Vec::new(),
        }
    }

    /// Add an issue
    pub fn add_issue(&mut self, issue: impl Into<String>) {
        self.issues.push(issue.into());
    }

    /// Add a suggestion
    pub fn add_suggestion(&mut self, suggestion: impl Into<String>) {
        self.suggestions.push(suggestion.into());
    }

    /// Get overall quality score
    pub fn overall_score(&self) -> f32 {
        (self.reasoning_coherence + self.action_relevance + self.plan_completeness) / 3.0
    }
}

/// Cached reasoning result
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CachedReasoning {
    pub id: String,
    pub goal_hash: String,               // Hash of goal for matching
    pub cot: ChainOfThought,
    pub hits: u32,                       // Times this cache was used
    pub created_at: DateTime<Utc>,
    pub last_used: DateTime<Utc>,
}

impl CachedReasoning {
    /// Create a new cached reasoning entry
    pub fn new(goal_hash: impl Into<String>, cot: ChainOfThought) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            goal_hash: goal_hash.into(),
            cot,
            hits: 0,
            created_at: now,
            last_used: now,
        }
    }

    /// Record a cache hit
    pub fn record_hit(&mut self) {
        self.hits += 1;
        self.last_used = Utc::now();
    }
}

/// System 2 configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct System2Config {
    pub max_reasoning_steps: usize,      // Max chain-of-thought steps
    pub max_tot_depth: usize,            // Max tree depth for search
    pub max_tot_width: usize,            // Max branches at each node
    pub verification_strictness: f32,    // 0.0 (lenient) to 1.0 (strict)
    pub cache_enabled: bool,
    pub cache_max_size: usize,
}

impl Default for System2Config {
    fn default() -> Self {
        Self {
            max_reasoning_steps: 20,
            max_tot_depth: 5,
            max_tot_width: 3,
            verification_strictness: 0.8,
            cache_enabled: true,
            cache_max_size: 1000,
        }
    }
}

impl System2Config {
    /// Create a new config with custom max steps
    pub fn with_max_steps(mut self, steps: usize) -> Self {
        self.max_reasoning_steps = steps;
        self
    }

    /// Set ToT parameters
    pub fn with_tot(mut self, depth: usize, width: usize) -> Self {
        self.max_tot_depth = depth;
        self.max_tot_width = width;
        self
    }

    /// Set verification strictness
    pub fn with_strictness(mut self, strictness: f32) -> Self {
        self.verification_strictness = strictness.clamp(0.0, 1.0);
        self
    }

    /// Disable caching
    pub fn without_cache(mut self) -> Self {
        self.cache_enabled = false;
        self
    }
}

/// ============================================================================
/// META-COGNITION LAYER TYPES (System 3 - Wai)
/// ============================================================================

/// An episode in episodic memory
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EpisodicEvent {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub goal: Option<Goal>,
    pub action: Option<Action>,
    pub observation: String,          // What happened
    pub outcome: Outcome,
    pub reward: Option<Reward>,
    pub reflection: Option<String>,   // Agent's interpretation
    pub emotional_valence: f32,       // -1.0 (bad) to 1.0 (good)
}

impl EpisodicEvent {
    /// Create a new episodic event
    pub fn new(observation: impl Into<String>, outcome: Outcome) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            goal: None,
            action: None,
            observation: observation.into(),
            outcome,
            reward: None,
            reflection: None,
            emotional_valence: 0.0,
        }
    }

    /// Set associated goal
    pub fn with_goal(mut self, goal: Goal) -> Self {
        self.goal = Some(goal);
        self
    }

    /// Set associated action
    pub fn with_action(mut self, action: Action) -> Self {
        self.action = Some(action);
        self
    }

    /// Set emotional valence
    pub fn with_valence(mut self, valence: f32) -> Self {
        self.emotional_valence = valence.clamp(-1.0, 1.0);
        self
    }

    /// Add reflection
    pub fn with_reflection(mut self, reflection: impl Into<String>) -> Self {
        self.reflection = Some(reflection.into());
        self
    }

    /// Check if this was a positive experience
    pub fn is_positive(&self) -> bool {
        self.emotional_valence > 0.0 || self.outcome.is_success()
    }
}

/// An entity in the knowledge graph
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KnowledgeEntity {
    pub id: String,
    pub entity_type: String,           // "Person", "Object", "Concept", "Place"
    pub name: String,
    pub attributes: HashMap<String, String>,
    pub relationships: Vec<(String, String)>, // (relation_type, target_entity_id)
    pub confidence: f32,
}

impl KnowledgeEntity {
    /// Create a new knowledge entity
    pub fn new(entity_type: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            entity_type: entity_type.into(),
            name: name.into(),
            attributes: HashMap::new(),
            relationships: Vec::new(),
            confidence: 1.0,
        }
    }

    /// Add attribute
    pub fn with_attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.attributes.insert(key.into(), value.into());
        self
    }

    /// Add relationship
    pub fn with_relationship(mut self, relation: impl Into<String>, target_id: impl Into<String>) -> Self {
        self.relationships.push((relation.into(), target_id.into()));
        self
    }

    /// Set confidence
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }
}

/// User model: what the agent believes about the user
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UserModel {
    pub user_id: String,
    pub name: Option<String>,
    pub preferences: HashMap<String, String>,
    pub goals: Vec<Goal>,
    pub communication_style: String,
    pub expertise_level: HashMap<String, f32>, // Topic → proficiency
    pub emotional_state: Option<String>,
}

impl UserModel {
    /// Create a new user model
    pub fn new(user_id: impl Into<String>) -> Self {
        Self {
            user_id: user_id.into(),
            name: None,
            preferences: HashMap::new(),
            goals: Vec::new(),
            communication_style: "neutral".to_string(),
            expertise_level: HashMap::new(),
            emotional_state: None,
        }
    }

    /// Set user name
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Add preference
    pub fn with_preference(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.preferences.insert(key.into(), value.into());
        self
    }

    /// Set expertise level for a topic
    pub fn with_expertise(mut self, topic: impl Into<String>, level: f32) -> Self {
        self.expertise_level.insert(topic.into(), level.clamp(0.0, 1.0));
        self
    }
}

/// System 3 configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct System3Config {
    pub enable_memory: bool,
    pub max_episodic_events: usize,
    pub memory_retention_days: u32,
    pub enable_rag: bool,
    pub vector_embedding_dim: usize,
    pub intrinsic_motivation_weights: (f32, f32, f32), // (curiosity, mastery, autonomy)
    pub extrinsic_reward_weight: f32,
    pub intrinsic_reward_weight: f32,
}

impl Default for System3Config {
    fn default() -> Self {
        Self {
            enable_memory: true,
            max_episodic_events: 10000,
            memory_retention_days: 90,
            enable_rag: true,
            vector_embedding_dim: 384,
            intrinsic_motivation_weights: (0.3, 0.4, 0.3), // curiosity, mastery, autonomy
            extrinsic_reward_weight: 0.6,
            intrinsic_reward_weight: 0.4,
        }
    }
}

impl System3Config {
    /// Set memory capacity
    pub fn with_max_events(mut self, max: usize) -> Self {
        self.max_episodic_events = max;
        self
    }

    /// Set motivation weights
    pub fn with_motivation_weights(mut self, curiosity: f32, mastery: f32, autonomy: f32) -> Self {
        self.intrinsic_motivation_weights = (curiosity, mastery, autonomy);
        self
    }

    /// Set reward weights
    pub fn with_reward_weights(mut self, extrinsic: f32, intrinsic: f32) -> Self {
        self.extrinsic_reward_weight = extrinsic;
        self.intrinsic_reward_weight = intrinsic;
        self
    }

    /// Disable memory
    pub fn without_memory(mut self) -> Self {
        self.enable_memory = false;
        self
    }

    /// Disable RAG
    pub fn without_rag(mut self) -> Self {
        self.enable_rag = false;
        self
    }
}

/// Raw event from environment
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EnvironmentEvent {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub event_type: String,
    pub data: serde_json::Value,
}

impl EnvironmentEvent {
    /// Create a new environment event
    pub fn new(event_type: impl Into<String>, data: serde_json::Value) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            event_type: event_type.into(),
            data,
        }
    }
}

/// Perception of a state
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Perception {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub sensory_modality: String,      // "visual", "textual", "audio", "api"
    pub raw_data: Vec<u8>,
    pub interpreted: String,           // Meaning extracted
    pub confidence: f32,
}

impl Perception {
    /// Create a new perception
    pub fn new(modality: impl Into<String>, interpreted: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            sensory_modality: modality.into(),
            raw_data: Vec::new(),
            interpreted: interpreted.into(),
            confidence: 1.0,
        }
    }

    /// Set raw data
    pub fn with_raw_data(mut self, data: Vec<u8>) -> Self {
        self.raw_data = data;
        self
    }

    /// Set confidence
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }
}

/// ============================================================================
/// UNIT TESTS
/// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_creation() {
        let agent = Agent::new("TestAgent");
        assert_eq!(agent.name(), "TestAgent");
        assert!(!agent.id.is_empty());
    }

    #[test]
    fn test_goal_creation() {
        let goal = Goal::new("Complete task", 5)
            .with_criterion("Task done")
            .with_criterion("No errors");
        
        assert_eq!(goal.priority, 5);
        assert_eq!(goal.success_criteria.len(), 2);
    }

    #[test]
    fn test_action_creation() {
        let action = Action::new("search", serde_json::json!({"query": "test"}))
            .with_safety_level(SafetyLevel::High)
            .with_cost(2.5);
        
        assert_eq!(action.tool_name, "search");
        assert_eq!(action.safety_level, SafetyLevel::High);
        assert_eq!(action.estimated_cost, 2.5);
    }

    #[test]
    fn test_safety_level() {
        assert!(SafetyLevel::Critical.requires_kernel_approval());
        assert!(SafetyLevel::High.requires_verification());
        assert!(!SafetyLevel::Safe.requires_verification());
    }

    #[test]
    fn test_memory_storage() {
        let mut agent = Agent::new("TestAgent");
        let item = MemoryItem::new("Important fact", MemoryCategory::Semantic)
            .with_embedding(vec![0.1, 0.2, 0.3])
            .with_relevance(0.9);
        
        agent.remember(item);
        assert_eq!(agent.memory_count(), 1);
    }

    #[test]
    fn test_recall_by_category() {
        let mut agent = Agent::new("TestAgent");
        
        agent.remember(MemoryItem::new("Fact 1", MemoryCategory::Semantic));
        agent.remember(MemoryItem::new("Episode 1", MemoryCategory::Episodic));
        agent.remember(MemoryItem::new("Fact 2", MemoryCategory::Semantic));
        
        let semantic = agent.recall_by_category(MemoryCategory::Semantic);
        assert_eq!(semantic.len(), 2);
    }

    #[test]
    fn test_self_model() {
        let agent = Agent::new("TestAgent");
        assert!(agent.self_model.terminal_values.contains(&"Safety".to_string()));
    }

    #[test]
    fn test_self_model_capabilities() {
        let model = SelfModel::new("agent1", "I am helpful")
            .with_capability("coding", 0.8)
            .with_capability("writing", 0.9);
        
        assert_eq!(model.proficiency("coding"), 0.8);
        assert_eq!(model.proficiency("unknown"), 0.0);
    }

    #[test]
    fn test_outcome() {
        assert!(Outcome::Success.is_success());
        assert!(Outcome::PartialSuccess(0.5).is_success());
        assert!(!Outcome::Failure("error".to_string()).is_success());
        
        assert_eq!(Outcome::Success.success_rate(), 1.0);
        assert_eq!(Outcome::PartialSuccess(0.7).success_rate(), 0.7);
    }

    #[test]
    fn test_textual_gradient() {
        let gradient = TextualGradient::new(
            "search failed",
            "wrong query format",
            "use structured query",
            "system2",
        ).with_confidence(0.8);
        
        assert_eq!(gradient.confidence, 0.8);
    }

    #[test]
    fn test_agent_config() {
        let config = AgentConfig::new("MyAgent")
            .with_temperature(0.5)
            .with_llm("anthropic", "claude-3")
            .with_evolution(false);
        
        assert_eq!(config.temperature, 0.5);
        assert_eq!(config.llm_provider, "anthropic");
        assert!(!config.enable_evolution);
    }

    #[test]
    fn test_reward_calculation() {
        let extrinsic = ExtrinsicReward {
            value: 1.0,
            success_flag: true,
            feedback_text: Some("Good job!".to_string()),
        };
        let intrinsic = IntrinsicReward {
            curiosity: 0.5,
            mastery: 0.6,
            autonomy: 0.4,
        };
        
        let reward = Reward::new(extrinsic, intrinsic);
        // 1.0 * 0.7 + (0.5 + 0.6 + 0.4) / 3 * 0.3 = 0.7 + 0.15 = 0.85
        assert!((reward.total - 0.85).abs() < 0.01);
    }
}
