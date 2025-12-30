# NAFS-4: Neuro-Agentic Framework System (Version 4)

## Executive Summary

NAFS-4 is a **self-evolving agent framework** built in Rust that implements a multi-system cognitive architecture inspired by neuroscience and modern AI research. The framework enables agents to reason, learn, and adapt through "textual backpropagation" - a novel mechanism for self-improvement.

---

## Architecture Overview

### The Four Systems

```
┌─────────────────────────────────────────────────────────────────────┐
│                         NAFS-4 Architecture                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐│
│  │  System 1   │  │  System 2   │  │  System 3   │  │  System 4   ││
│  │  (Reactive) │  │ (Reasoning) │  │ (Awareness) │  │ (Evolution) ││
│  │             │  │             │  │             │  │             ││
│  │ • Fast      │  │ • Slow      │  │ • Memory    │  │ • Backprop  ││
│  │ • Intuitive │  │ • Logical   │  │ • Self-Model│  │ • Mutation  ││
│  │ • Heuristic │  │ • Verify    │  │ • Monitor   │  │ • Safety    ││
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘│
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                         Core Layer                              ││
│  │  Goals │ Actions │ States │ Messages │ AgentRole │ Config      ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                      Infrastructure                             ││
│  │  LLM Provider │ Vector DB │ Graph DB │ Observability │ Python  ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

---

## Crate Architecture (11 Crates)

### Core Layer
1. **nafs-core** - Foundational types (Goal, State, Action, Message, AgentRole)
2. **nafs-config** - Configuration management and validation
3. **nafs-error** - Error types and handling

### Cognitive Systems
4. **nafs-system1** - Reactive system (fast heuristics, pattern matching)
5. **nafs-system2** - Reasoning system (symbolic verification, LLM planning, ToT)
6. **nafs-system3** - Awareness system (memory, self-model, executive monitor)
7. **nafs-system4** - Evolution system (textual backprop, kernel safety)

### Infrastructure
8. **nafs-llm** - LLM provider abstraction (OpenAI, Anthropic, local)
9. **nafs-memory** - Vector DB + Graph DB interfaces
10. **nafs-observe** - Tracing, metrics, and observability
11. **nafs-python** - PyO3 bindings for Python interop

---

## Type Definitions

### Core Types (nafs-core)

```rust
/// Represents a goal the agent is trying to achieve
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Goal {
    pub id: GoalId,
    pub description: String,
    pub priority: Priority,
    pub status: GoalStatus,
    pub sub_goals: Vec<Goal>,
    pub constraints: Vec<Constraint>,
    pub deadline: Option<Timestamp>,
    pub created_at: Timestamp,
    pub metadata: HashMap<String, Value>,
}

/// The current state of the agent's world model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct State {
    pub id: StateId,
    pub observations: Vec<Observation>,
    pub beliefs: BeliefSet,
    pub active_goals: Vec<GoalId>,
    pub timestamp: Timestamp,
    pub confidence: f64,
}

/// An action the agent can take
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Action {
    pub id: ActionId,
    pub name: String,
    pub parameters: HashMap<String, Value>,
    pub preconditions: Vec<Condition>,
    pub effects: Vec<Effect>,
    pub cost: f64,
    pub reversible: bool,
}

/// Messages exchanged between systems
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub id: MessageId,
    pub source: SystemId,
    pub target: SystemId,
    pub payload: MessagePayload,
    pub timestamp: Timestamp,
    pub priority: Priority,
}

/// Defines the role and behavior of an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRole {
    pub id: RoleId,
    pub name: String,
    pub system_prompt: String,
    pub capabilities: Vec<Capability>,
    pub constraints: Vec<Constraint>,
    pub version: u64,
    pub mutated_at: Option<Timestamp>,
}
```

---

## System 2: Reasoning Engine

### Components

#### SymbolicVerifier
Performs logical verification of reasoning chains.

```rust
pub struct SymbolicVerifier {
    rules: Vec<LogicRule>,
    axioms: Vec<Axiom>,
}

impl SymbolicVerifier {
    pub fn verify(&self, chain: &ReasoningChain) -> VerificationResult;
    pub fn check_consistency(&self, beliefs: &BeliefSet) -> ConsistencyReport;
}
```

#### LLMPlanner
Generates plans using Chain-of-Thought reasoning.

```rust
pub struct LLMPlanner {
    provider: Box<dyn LLMProvider>,
    cot_config: ChainOfThoughtConfig,
}

impl LLMPlanner {
    pub async fn plan(&self, goal: &Goal, state: &State) -> Result<Plan, PlanError>;
    pub async fn refine(&self, plan: &Plan, feedback: &Feedback) -> Result<Plan, PlanError>;
}
```

#### TreeOfThought
Implements tree search for complex reasoning.

```rust
pub struct TreeOfThought {
    max_depth: usize,
    branching_factor: usize,
    evaluator: Box<dyn ThoughtEvaluator>,
}

impl TreeOfThought {
    pub async fn search(&self, problem: &Problem) -> Result<Solution, SearchError>;
    pub fn prune(&mut self, threshold: f64);
}
```

---

## System 3: Awareness Engine

### Components

#### MemoryModule
Unified interface for vector and graph memory.

```rust
pub struct MemoryModule {
    vector_db: Box<dyn VectorStore>,
    graph_db: Box<dyn GraphStore>,
    indexer: MemoryIndexer,
}

impl MemoryModule {
    pub async fn store(&self, memory: Memory) -> Result<MemoryId, MemoryError>;
    pub async fn recall(&self, query: &Query, limit: usize) -> Result<Vec<Memory>, MemoryError>;
    pub async fn associate(&self, from: MemoryId, to: MemoryId, relation: Relation) -> Result<(), MemoryError>;
    pub async fn forget(&self, id: MemoryId) -> Result<(), MemoryError>;
}
```

#### SelfModel
Tracks the agent's capabilities and limitations.

```rust
pub struct SelfModel {
    capabilities: HashMap<CapabilityId, CapabilityProfile>,
    performance_history: Vec<PerformanceRecord>,
    confidence_model: ConfidenceModel,
}

impl SelfModel {
    pub fn assess_capability(&self, action: &Action) -> CapabilityAssessment;
    pub fn update_from_outcome(&mut self, action: &Action, outcome: &Outcome);
    pub fn predict_success(&self, plan: &Plan) -> f64;
}
```

#### ExecutiveMonitor
The main cognitive loop controller.

```rust
pub struct ExecutiveMonitor {
    systems: SystemRegistry,
    scheduler: TaskScheduler,
    attention: AttentionController,
}

impl ExecutiveMonitor {
    pub async fn run(&mut self) -> Result<(), MonitorError>;
    pub fn allocate_attention(&mut self, task: &Task) -> AttentionAllocation;
    pub fn interrupt(&mut self, priority: Priority) -> bool;
}
```

---

## System 4: Evolution Engine (Core Innovation)

### Textual Backpropagation

The key innovation of NAFS-4 is "textual backpropagation" - using natural language gradients to improve the agent.

```rust
pub struct TextualBackpropagator {
    analyzer: FailureAnalyzer,
    gradient_generator: GradientGenerator,
    mutator: PromptMutator,
    kernel: KernelSupervisor,
}

impl TextualBackpropagator {
    /// Catch and analyze a runtime failure
    pub fn catch_failure(&self, failure: &Failure) -> FailureAnalysis;
    
    /// Generate textual "gradients" (fix instructions)
    pub fn compute_gradient(&self, analysis: &FailureAnalysis) -> TextualGradient;
    
    /// Update AgentRole system prompts
    pub fn apply_mutation(&mut self, role: &mut AgentRole, gradient: &TextualGradient) -> MutationResult;
}
```

### Kernel Supervisor (Safety)

Blocks unsafe mutations to protect core agent behavior.

```rust
pub struct KernelSupervisor {
    protected_regions: Vec<ProtectedRegion>,
    invariants: Vec<SafetyInvariant>,
    audit_log: AuditLog,
}

impl KernelSupervisor {
    /// Check if a mutation is safe to apply
    pub fn validate_mutation(&self, mutation: &Mutation) -> SafetyVerdict;
    
    /// Enforce safety invariants
    pub fn enforce(&self, role: &AgentRole) -> Result<(), SafetyViolation>;
    
    /// Rollback to a safe state
    pub fn rollback(&mut self, checkpoint: &Checkpoint) -> Result<(), RollbackError>;
}
```

---

## Python Bindings (nafs-python)

### Exposed Types

```python
from nafs import Goal, State, Action, AgentRole, MemoryModule

# Create a goal
goal = Goal(
    description="Analyze the dataset",
    priority="high",
    constraints=["memory < 1GB", "time < 60s"]
)

# Query memory
memories = memory_module.recall("previous analysis results", limit=10)

# Inject goal into agent
agent.submit_goal(goal)

# Get evolution metrics
metrics = agent.get_evolution_metrics()
print(f"Mutations applied: {metrics.mutation_count}")
print(f"Success rate delta: {metrics.success_rate_delta}")
```

### Integration Points

- **Goal Injection**: Python can create and submit goals
- **Memory Access**: Read/write to the memory module
- **Metrics Export**: Retrieve evolution metrics for NumPy/Pandas analysis
- **Callback Hooks**: Register Python callbacks for agent events

---

## Implementation Phases

### Phase 0: Foundation
- [ ] Initialize Cargo workspace with 11 crates
- [ ] Implement core types (Goal, State, Action, Message, AgentRole)
- [ ] Create error types and configuration
- [ ] Generate Workspace_Structure_Report.md

### Phase 1: Reasoning (System 2)
- [ ] Implement SymbolicVerifier
- [ ] Implement LLMPlanner with Chain-of-Thought
- [ ] Implement TreeOfThought search
- [ ] Add verification tests

### Phase 2: Awareness (System 3)
- [ ] Implement MemoryModule (VectorDB interface)
- [ ] Implement MemoryModule (GraphDB interface)
- [ ] Implement SelfModel
- [ ] Implement ExecutiveMonitor

### Phase 3: Evolution (System 4)
- [ ] Implement FailureAnalyzer
- [ ] Implement TextualBackpropagator
- [ ] Implement PromptMutator
- [ ] Implement KernelSupervisor

### Phase 4: Python Bindings
- [ ] Setup PyO3/maturin
- [ ] Expose core types as PyClass
- [ ] Implement Goal injection
- [ ] Implement Memory access

### Phase 5: Optimization & Polish
- [ ] Performance optimization
- [ ] Documentation
- [ ] Integration tests
- [ ] Examples

---

## Dependencies

```toml
[workspace.dependencies]
tokio = { version = "1.35", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
async-trait = "0.1"
thiserror = "1.0"
tracing = "0.1"
tracing-subscriber = "0.3"
uuid = { version = "1.6", features = ["v4", "serde"] }
chrono = { version = "0.4", features = ["serde"] }
pyo3 = { version = "0.20", features = ["extension-module"] }
```

---

## License

MIT License

---

*NAFS-4: Where cognition meets code evolution.*
