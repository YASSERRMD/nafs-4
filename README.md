# NAFS-4: Neuromorphic AI Framework System

NAFS-4 is a production-grade, self-evolving cognitive architecture framework implemented in Rust. It integrates four distinct cognitive systems into a cohesive pipeline, enabling the creation of autonomous agents capable of fast reaction, deliberate reasoning, meta-cognition, and self-evolution through textual backpropagation.

## Architecture

The framework implements a hierarchical cognitive model inspired by dual-process theory and meta-learning research:

### System 1: Instinct and Perception
Provides fast, heuristic-based responses to environmental stimuli. It utilizes caching and pattern matching to handle routine tasks efficiently without invoking expensive reasoning resources.

### System 2: Deliberate Reasoning
Implements a Tree of Thought (ToT) reasoning engine. It performs symbolic verification, planning, and multi-step problem solving. This system engages when System 1 heuristics are insufficient or when high-stakes decisions require logical validation.

### System 3: Meta-Cognition and Memory
Maintains the agent's self-model (`identity`, `capabilities`, `values`) and manages a comprehensive memory system (episodic and semantic). It features an Executive Monitor that tracks internal state, motivation, and goal alignment.

### System 4: Evolution
A novel "Textual Backpropagation" mechanism allowing agents to improve over time. By analyzing failure patterns, System 4 generates textual gradients—instructions for self-modification—that selectively update the agent's prompt architecture and capabilities, subject to strict Kernel Supervisor constraints.

### Orchestrator
The central coordination layer that routes requests, manages agent lifecycles, and ensures seamless data flow between all cognitive systems.

## Supported LLM Providers

NAFS-4 includes native support for a wide range of global and regional LLM providers:
- **Major Platforms:** OpenAI, Anthropic (Claude), Google (Gemini), Azure OpenAI
- **Specialized:** Mistral AI, Cohere
- **Chinese Providers:** DeepSeek, Alibaba Cloud Qwen, Zhipu AI (ChatGLM), 01.AI (Yi)
- **Local/Private:** Ollama, LocalAI, and any OpenAI-compatible endpoint

## Memory & Storage

NAFS-4 features a unified memory subsystem that abstracts vector and graph storage:
- **Vector Storage:** Native integration with **Barq-DB** for high-dimensional similarity search.
- **Graph Storage:** Native integration with **Barq-GraphDB** for complex relational memory models and hybrid traversal.
- **Embedded:** Supports flexible in-memory storage for development and testing.

## Key Features

- **Neuro-Symbolic Reasoning**: Combines Large Language Model (LLM) flexibility with symbolic logic verification.
- **Autonomous Self-Evolution**: Agents iteratively refine their own behavior based on operational outcomes.
- **Robust Persistence**: Distributed state management for long-running agent processes.
- **Kernel Safety**: Hard constraints prevent agents from evolving potentially unsafe behaviors.
- **High Performance**: Built on Rust's async runtime (Tokio) for concurrent agent execution.

## Installation

Ensure you have Rust and Cargo installed (1.70+).

```bash
git clone https://github.com/nafs-framework/nafs-4.git
cd nafs-4
cargo build --release
```

## Component Usage

The framework exposes its functionality through a Command Line Interface (CLI) and a REST API.

### Command Line Interface (CLI)

The `nafs` binary provides interactive management capabilities.

**Manage Agents**
```bash
# Create a new agent
nafs agent create --name "Analyst_01" --role "Data Analyst"

# List active agents
nafs agent list

# Query an agent directly
nafs agent query --agent-id "agent-uuid" --query "Analyze the provided dataset."
```

**System Operations**
```bash
# Check system health and metrics
nafs system health

# Start the interactive REPL
nafs repl
```

### REST API

The `nafs-api` service enables remote integration.

**Start the Server**
```bash
cargo run --release -p nafs-api
# Server listening on http://127.0.0.1:3000
```

**API Endpoints**

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | System health status |
| `POST` | `/agents` | Inspect/Create new agent instances |
| `POST` | `/agents/:id/query` | Submit a cognitive task to an agent |
| `GET` | `/agents/:id/memory` | Retrieve agent memory context |
| `POST` | `/agents/:id/evolve` | Trigger an evolution cycle manually |

## Development and Testing

The repository includes a comprehensive test suite covering unit logic and end-to-end integration.

```bash
# Run all unit tests
cargo test --workspace

# Run integration pipeline tests
cargo test -p nafs-integration
```

## Project Structure

- `nafs-core`: Fundamental types and traits.
- `nafs-system1`: Perception and heuristic modules.
- `nafs-system2`: Reasoning engines and symbolic verification.
- `nafs-system3`: Memory systems and meta-cognitive monitors.
- `nafs-system4`: Evolutionary algorithms and safety kernels.
- `nafs-orchestrator`: Lifecycle management and event routing.
- `nafs-cli`: Terminal user interface implementation.
- `nafs-api`: HTTP/WebSocket API implementation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
