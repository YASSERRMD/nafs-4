# NAFS-4 Python Bindings

Python bindings for the NAFS-4 Neuromorphic AI Framework.

## Installation

You need [maturin](https://github.com/PyO3/maturin) to build and install this package.

```bash
pip install maturin
maturin develop
```

## Usage

```python
import nafs
import asyncio

async def main():
    # Create orchestrator
    orchestrator = await nafs.Orchestrator.create()
    
    # Check health
    healthy = await orchestrator.health()
    print(f"System healthy: {healthy}")
    
    # Create agent
    agent_id = await orchestrator.create_agent("PyAgent", "Assistant")
    print(f"Created agent: {agent_id}")
    
    # Query
    result = await orchestrator.query(agent_id, "Hello from Python!")
    print(f"Result: {result}")

if __name__ == "__main__":
    asyncio.run(main())
```
