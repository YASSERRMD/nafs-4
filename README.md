# NAFS-4: Neuromorphic AI Framework System

A production-grade, self-evolving cognitive architecture framework with native Python bindings. Build autonomous AI agents with fast perception, deliberate reasoning, meta-cognition, and self-evolution.

## 🚀 Quick Start (Python)

### Installation

```bash
pip install git+https://github.com/YASSERRMD/nafs-4.git#subdirectory=nafs-python
```

### Basic Usage

```python
import asyncio
import nafs

async def main():
    # Initialize orchestrator
    orch = await nafs.Orchestrator.create()
    
    # Create an agent
    agent_id = await orch.create_agent("MyAgent", "Research Assistant")
    
    # Query the agent
    response = await orch.query(agent_id, "Explain quantum computing")
    print(response.result)
    
    # Generate embeddings
    embedding = await orch.embed("Hello, world!")
    print(f"Embedding dimensions: {len(embedding)}")

asyncio.run(main())
```

### Configure LLM Provider

Set one environment variable to select your provider:

```bash
# Choose ONE provider
export COHERE_API_KEY=your_key        # Cohere (recommended)
export OPENAI_API_KEY=your_key        # OpenAI
export TOGETHER_API_KEY=your_key      # Together.ai
export GROQ_API_KEY=your_key          # Groq (fastest)
export HUGGINGFACE_API_KEY=your_key   # HuggingFace
export VOYAGE_API_KEY=your_key        # Voyage AI (embeddings only)
export JINA_API_KEY=your_key          # Jina AI (embeddings only)
export OLLAMA_URL=http://localhost:11434  # Local Ollama
```

---

## 📚 Python Examples

| Example | Description |
|---------|-------------|
| [simple_agent.py](nafs-python/examples/simple_agent.py) | Basic agent creation and querying |
| [multi_agent_workflow.py](nafs-python/examples/multi_agent_workflow.py) | Coordinating multiple agents |
| [research_assistant.py](nafs-python/examples/research_assistant.py) | RAG-powered research agent |
| [parallel_cognitive_system.py](nafs-python/examples/parallel_cognitive_system.py) | Concurrent agents with Systems 1-4 |
| [self_evolving_tutor.py](nafs-python/examples/self_evolving_tutor.py) | Agent evolution demonstration |
| [embedding_rag_demo.py](nafs-python/examples/embedding_rag_demo.py) | Multi-provider embedding demo |
| [knowledge_graph_demo.py](nafs-python/examples/knowledge_graph_demo.py) | Graph-enhanced RAG with Barq-GraphDB |
| [complete_research_pipeline.py](nafs-python/examples/complete_research_pipeline.py) | **Full showcase** - All features |

Run any example:
```bash
export COHERE_API_KEY=your_key
python nafs-python/examples/simple_agent.py
```

---

## 🔌 Supported Providers

### LLM + Embeddings
| Provider | LLM Models | Embedding Models |
|----------|------------|------------------|
| **Cohere** | command-r-plus | embed-english-v3.0, embed-multilingual-v3.0 |
| **OpenAI** | gpt-4, gpt-3.5-turbo | text-embedding-3-small/large |
| **Together.ai** | Llama-3, Mixtral, Qwen | m2-bert, bge-large |
| **Fireworks** | Llama, Mixtral | nomic-embed, UAE |
| **HuggingFace** | Mistral, Llama | all-MiniLM, bge, e5 |
| **Ollama** | llama3, mistral, phi | nomic-embed-text |

### Embeddings Only
| Provider | Models |
|----------|--------|
| **Voyage AI** | voyage-3, voyage-code-3, voyage-law-2 |
| **Jina AI** | jina-embeddings-v3, jina-colbert-v2 |

### LLM Only
| Provider | Models |
|----------|--------|
| **Groq** | llama-3.1-70b (fastest inference!) |
| **Anthropic** | claude-3 |

---

## 🧠 Embedding API

```python
# Get available models
models = orch.get_embedding_models()
print(models)  # ['embed-english-v3.0', ...]

# Set session default model
await orch.set_embedding_model("embed-multilingual-v3.0")

# Generate embedding (uses session model)
embedding = await orch.embed("Your text here")

# One-time model override
embedding = await orch.embed_with_model("Text", "embed-english-light-v3.0")

# Reset to provider default
await orch.set_embedding_model(None)
```

---

## 🗄️ Memory & Storage

NAFS-4 integrates with external databases for persistent memory:

| Storage | Purpose | Environment Variable |
|---------|---------|---------------------|
| **Barq-DB** | Vector storage for embeddings | `BARQ_DB_ADDR=http://localhost:50051` |
| **Barq-GraphDB** | Knowledge graph storage | `BARQ_GRAPH_ADDR=http://localhost:8081` |

```bash
# Enable external storage
export BARQ_DB_ADDR=http://localhost:50051
export BARQ_GRAPH_ADDR=http://localhost:8081
```

---

## 🏗️ Architecture

NAFS-4 implements a 4-system cognitive architecture:

```
┌─────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                         │
├─────────────┬─────────────┬─────────────┬──────────────┤
│  System 1   │  System 2   │  System 3   │  System 4    │
│  Perception │  Reasoning  │  Memory     │  Evolution   │
│  (Fast)     │  (ToT/CoT)  │  (Episodic) │  (Learning)  │
└─────────────┴─────────────┴─────────────┴──────────────┘
                          │
            ┌─────────────┴─────────────┐
            │   Unified Memory Layer    │
            ├───────────┬───────────────┤
            │  Barq-DB  │ Barq-GraphDB  │
            │  (Vector) │   (Graph)     │
            └───────────┴───────────────┘
```

- **System 1**: Fast pattern matching and heuristic responses
- **System 2**: Tree-of-Thought reasoning with symbolic verification
- **System 3**: Episodic/semantic memory with self-model
- **System 4**: Textual backpropagation for self-improvement

---

## 🔧 Rust Development

For Rust developers building from source:

```bash
# Clone and build
git clone https://github.com/YASSERRMD/nafs-4.git
cd nafs-4
cargo build --release

# Run tests
cargo test --workspace

# Start REST API server
cargo run --release -p nafs-server
```

### Project Structure

| Crate | Description |
|-------|-------------|
| `nafs-core` | Fundamental types and traits |
| `nafs-system1` | Perception and heuristics |
| `nafs-system2` | Reasoning engines |
| `nafs-system3` | Memory systems |
| `nafs-system4` | Evolution and safety |
| `nafs-orchestrator` | Agent lifecycle management |
| `nafs-llm` | LLM provider integrations |
| `nafs-memory` | Vector/graph storage |
| `nafs-python` | Python bindings |

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.
