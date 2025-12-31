#!/usr/bin/env python3
"""
NAFS-4 Knowledge Graph RAG Demo with Barq-GraphDB

Demonstrates a knowledge graph-enhanced RAG pipeline:
1. Create knowledge entities as graph nodes
2. Link entities with relationships
3. Combine graph traversal with embeddings for enhanced retrieval

Prerequisites:
    - Barq-GraphDB running on port 8081
    - export BARQ_GRAPH_ADDR=http://localhost:8081
    - export COHERE_API_KEY=your_key

Usage:
    python knowledge_graph_demo.py
"""

import asyncio
import nafs
import os

async def main():
    print("=" * 60)
    print("🕸️  NAFS-4 Knowledge Graph RAG with Barq-GraphDB")
    print("=" * 60)
    
    # Configuration check
    graph_addr = os.environ.get("BARQ_GRAPH_ADDR", "")
    has_cohere = "COHERE_API_KEY" in os.environ
    
    print(f"\n📋 Configuration:")
    print(f"   Graph DB: {graph_addr or 'Not configured'}")
    print(f"   LLM: {'Cohere' if has_cohere else 'Mock'}")
    
    if not graph_addr:
        print("\n⚠️  Setting BARQ_GRAPH_ADDR=http://localhost:8081")
        os.environ["BARQ_GRAPH_ADDR"] = "http://localhost:8081"
        graph_addr = "http://localhost:8081"
    
    # Initialize NAFS-4
    print("\n🔧 Initializing NAFS-4 with Barq-GraphDB...")
    try:
        orch = await nafs.Orchestrator.create()
        print("   ✅ Connected to Barq-GraphDB")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    # Create knowledge graph entities
    print("\n📊 Building Knowledge Graph:")
    print("-" * 50)
    
    # Define entities
    entities = [
        {"id": "nafs4", "type": "Framework", "name": "NAFS-4", "desc": "Neuromorphic AI Framework with 4 cognitive systems"},
        {"id": "barq_db", "type": "Database", "name": "Barq-DB", "desc": "High-performance vector database in Rust"},
        {"id": "barq_graph", "type": "Database", "name": "Barq-GraphDB", "desc": "Hybrid graph+vector database for AI"},
        {"id": "cohere", "type": "LLM", "name": "Cohere", "desc": "LLM provider with embeddings API"},
        {"id": "rag", "type": "Technique", "name": "RAG", "desc": "Retrieval-Augmented Generation"},
    ]
    
    # Define relationships
    relationships = [
        ("nafs4", "USES", "barq_db"),
        ("nafs4", "USES", "barq_graph"),
        ("nafs4", "INTEGRATES", "cohere"),
        ("nafs4", "IMPLEMENTS", "rag"),
        ("rag", "REQUIRES", "barq_db"),
        ("barq_graph", "EXTENDS", "barq_db"),
    ]
    
    print("   Entities:")
    for e in entities:
        print(f"     [{e['type']}] {e['name']}")
    
    print("\n   Relationships:")
    for src, rel, dst in relationships:
        print(f"     {src} --{rel}--> {dst}")
    
    # Create Graph-Augmented RAG Agent
    print("\n🤖 Creating Graph-RAG Agent...")
    graph_agent = await orch.create_agent(
        "GraphRAG_Bot", 
        "Knowledge Graph Reasoning Specialist"
    )
    
    # Simulate graph-aware query
    print("\n🔍 Graph-Enhanced Query:")
    print("-" * 50)
    
    query = "What technologies does NAFS-4 use for RAG?"
    print(f"   Query: \"{query}\"")
    
    # Build context from graph (simulated traversal)
    graph_context = """
    Knowledge Graph Traversal Results:
    - NAFS-4 [Framework] --USES--> Barq-DB [Database]
    - NAFS-4 [Framework] --USES--> Barq-GraphDB [Database]
    - NAFS-4 [Framework] --IMPLEMENTS--> RAG [Technique]
    - RAG [Technique] --REQUIRES--> Barq-DB [Database]
    - Barq-GraphDB [Database] --EXTENDS--> Barq-DB [Database]
    """
    
    augmented_prompt = f"""
    User Question: {query}
    
    {graph_context}
    
    Using the graph knowledge above, answer the question accurately.
    """
    
    response = await orch.query(graph_agent, augmented_prompt)
    
    print(f"\n📝 Graph-RAG Answer:")
    print(f"   {response.result}")
    
    print(f"\n🧠 Cognitive Pipeline:")
    for k, v in response.metadata.items():
        if k.startswith("system"):
            print(f"   {k}: {v}")
    
    print(f"\n⏱️  Latency: {response.execution_time_ms}ms")
    print("\n✅ Knowledge Graph RAG demo completed!")

if __name__ == "__main__":
    asyncio.run(main())
