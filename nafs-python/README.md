# NAFS-4 Python Bindings

Python bindings for the NAFS-4 Neuromorphic AI Framework System.

## Installation

```bash
pip install git+https://github.com/YASSERRMD/nafs-4.git#subdirectory=nafs-python
```

## Quick Start

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

## Configure LLM Provider

Set one environment variable to select your provider:

```bash
# Choose ONE provider
export COHERE_API_KEY=your_key        # Cohere (recommended)
export OPENAI_API_KEY=your_key        # OpenAI
export TOGETHER_API_KEY=your_key      # Together.ai
export GROQ_API_KEY=your_key          # Groq (fastest)
export HUGGINGFACE_API_KEY=your_key   # HuggingFace
export FIREWORKS_API_KEY=your_key     # Fireworks AI
export VOYAGE_API_KEY=your_key        # Voyage AI (embeddings only)
export JINA_API_KEY=your_key          # Jina AI (embeddings only)
export OLLAMA_URL=http://localhost:11434  # Local Ollama
```

---

## Examples

| Example | Description |
|---------|-------------|
| [simple_agent.py](examples/simple_agent.py) | Basic agent creation and querying |
| [multi_agent_workflow.py](examples/multi_agent_workflow.py) | Coordinating multiple agents |
| [research_assistant.py](examples/research_assistant.py) | RAG-powered research agent |
| [parallel_cognitive_system.py](examples/parallel_cognitive_system.py) | Concurrent agents with Systems 1-4 |
| [self_evolving_tutor.py](examples/self_evolving_tutor.py) | Agent evolution demonstration |
| [embedding_rag_demo.py](examples/embedding_rag_demo.py) | Multi-provider embedding demo |
| [knowledge_graph_demo.py](examples/knowledge_graph_demo.py) | Graph-enhanced RAG with Barq-GraphDB |
| [complete_research_pipeline.py](examples/complete_research_pipeline.py) | Full showcase - All features |

Run any example:
```bash
export COHERE_API_KEY=your_key
python examples/simple_agent.py
```

---

## Supported Providers

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

## API Reference

### Orchestrator

```python
# Create orchestrator
orch = await nafs.Orchestrator.create()

# Health check
is_healthy = await orch.health()

# Get provider info
provider_name = orch.get_provider_name()  # "cohere", "openai", etc.
```

### Agent Management

```python
# Create an agent
agent_id = await orch.create_agent(name="AgentName", role="Role Description")

# Query an agent
response = await orch.query(agent_id, "Your question here")
print(response.result)
print(response.execution_time_ms)
print(response.metadata)
```

### Embedding API

```python
# Get available models
models = orch.get_embedding_models()
# ['embed-english-v3.0', 'embed-english-light-v3.0', ...]

# Set session default model
await orch.set_embedding_model("embed-multilingual-v3.0")

# Get current model
current = await orch.get_embedding_model()

# Generate embedding (uses session model)
embedding = await orch.embed("Your text here")

# One-time model override
embedding = await orch.embed_with_model("Text", "embed-english-light-v3.0")

# Reset to provider default
await orch.set_embedding_model(None)
```

---

## Memory and Storage

Enable external databases for persistent memory:

```bash
# Vector storage (Barq-DB)
export BARQ_DB_ADDR=http://localhost:50051

# Knowledge graph storage (Barq-GraphDB)
export BARQ_GRAPH_ADDR=http://localhost:8081

# Collection name (optional)
export BARQ_COLLECTION=my_memories
```

---

## Development

Build from source:

```bash
cd nafs-python

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install maturin
pip install maturin

# Build and install
maturin develop

# Run tests
python -m pytest
```

---

## License

MIT License - see [LICENSE](../LICENSE) for details.
