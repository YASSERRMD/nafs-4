# NAFS-4: Self-Evolving Neuro-Symbolic Agent Framework

**Phase 0: Foundation** ✅

A Rust-based, self-evolving agent framework implementing a multi-system cognitive architecture.

## 🏗️ Architecture

NAFS-4 implements a four-system cognitive model:

| System | Name | Description |
|--------|------|-------------|
| **System 1** | Perception/Action | Fast, intuitive heuristic responses |
| **System 2** | Reasoning | Slow, deliberate symbolic verification & LLM planning |
| **System 3** | Meta-Cognition | Self-awareness, memory, executive monitoring |
| **System 4** | Evolution | Self-improvement via "Textual Backpropagation" |

## 📦 Project Structure

```
nafs-4/
├── nafs-core/       # Core types & traits (Goal, State, Action, Agent)
├── nafs-system1/    # Perception & fast heuristics
├── nafs-system2/    # Reasoning (SymbolicVerifier, LLMPlanner, TreeOfThought)
├── nafs-system3/    # Meta-cognition (Memory, SelfModel, ExecutiveMonitor)
├── nafs-system4/    # Evolution (TextualBackprop, KernelSupervisor)
├── nafs-memory/     # Vector DB & Graph DB interfaces
├── nafs-llm/        # LLM provider abstraction
├── nafs-tools/      # Tool management (registry, executor)
├── nafs-logging/    # Observability (tracing, metrics)
├── nafs-cli/        # Command-line interface
└── nafs-server/     # REST API server
```

## 🚀 Quick Start

### Build
```bash
cargo build --all
```

### Run Tests
```bash
cargo test --all
```

### CLI
```bash
cargo run --bin nafs -- --help
cargo run --bin nafs -- new --name "MyAgent"
cargo run --bin nafs -- version
```

### Server
```bash
cargo run --bin nafs-server
# API available at http://127.0.0.1:8080
```

## 🧠 Core Concepts

### Agent
The central entity combining all systems:
```rust
use nafs_core::{Agent, Goal, MemoryItem, MemoryCategory};

let mut agent = Agent::new("MyAgent");

// Set a goal
let goal = Goal::new("Complete the task", 5)
    .with_criterion("No errors");
agent.set_goal(goal);

// Store a memory
let memory = MemoryItem::new("Important fact", MemoryCategory::Semantic);
agent.remember(memory);
```

### Evolution (System 4)
The core innovation - "Textual Backpropagation":
1. Catch runtime failures
2. Generate textual "gradients" (fix instructions)
3. Mutate system prompts
4. Kernel supervisor blocks unsafe mutations

## 📊 Phase 0 Metrics

| Metric | Value |
|--------|-------|
| Crates | 11 |
| Lines of Rust | 3,000+ |
| Tests | 58 |
| Build Time | < 10s |

## 🛣️ Roadmap

- [x] **Phase 0**: Foundation (Workspace, Core Types)
- [ ] **Phase 1**: System 2 Implementation (Full Reasoning)
- [ ] **Phase 2**: System 3 Implementation (Full Awareness)
- [ ] **Phase 3**: System 4 Implementation (Full Evolution)
- [ ] **Phase 4**: Python Bindings (PyO3)
- [ ] **Phase 5**: Optimization & Polish

## 📄 License

Apache-2.0

---

*NAFS-4: Where cognition meets code evolution.*
