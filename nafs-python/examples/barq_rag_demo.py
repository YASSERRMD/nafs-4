#!/usr/bin/env python3
"""
NAFS-4 RAG Demo with Barq-DB Vector Storage

This example demonstrates using NAFS-4 with Barq-DB as the vector database
for Retrieval-Augmented Generation (RAG) workflows.

Prerequisites:
1. Start Barq-DB server: `cargo run --bin barq-server` (default port: 50051)
2. Set environment variables:
   - BARQ_DB_ADDR=http://localhost:50051
   - BARQ_GRAPH_ADDR=http://localhost:3000 (optional, for graph queries)
   - BARQ_COLLECTION=nafs_memories (optional, defaults to this)
   - COHERE_API_KEY=your_key (or other LLM provider)

Usage:
    export BARQ_DB_ADDR=http://localhost:50051
    export COHERE_API_KEY=your_key
    python barq_rag_demo.py
"""

import asyncio
import nafs
import os
import sys

# Configuration
BARQ_DB_DEFAULT = "http://localhost:50051"
BARQ_GRAPH_DEFAULT = "http://localhost:3000"

async def main():
    print("=" * 60)
    print("🚀 NAFS-4 RAG System with Barq-DB Vector Storage")
    print("=" * 60)
    
    # Check environment configuration
    barq_db = os.environ.get("BARQ_DB_ADDR", "")
    barq_graph = os.environ.get("BARQ_GRAPH_ADDR", "")
    llm_provider = "Mock"
    
    if os.environ.get("COHERE_API_KEY"):
        llm_provider = "Cohere"
    elif os.environ.get("OPENAI_API_KEY"):
        llm_provider = "OpenAI"
    elif os.environ.get("ANTHROPIC_API_KEY"):
        llm_provider = "Anthropic"
    
    print("\n📋 Configuration:")
    print(f"   Vector DB (Barq-DB): {barq_db or 'In-Memory (set BARQ_DB_ADDR to use Barq-DB)'}")
    print(f"   Graph DB (Barq-GraphDB): {barq_graph or 'In-Memory'}")
    print(f"   LLM Provider: {llm_provider}")
    
    if not barq_db:
        print("\n⚠️  BARQ_DB_ADDR not set. Using in-memory vector storage.")
        print(f"   To use Barq-DB, run: export BARQ_DB_ADDR={BARQ_DB_DEFAULT}")
    
    # Initialize orchestrator
    print("\n🔧 Initializing NAFS-4 Orchestrator...")
    try:
        orch = await nafs.Orchestrator.create()
        print("   ✅ Orchestrator ready")
    except Exception as e:
        print(f"   ❌ Failed to initialize: {e}")
        return
    
    # Create specialized RAG agents
    print("\n👥 Creating RAG Pipeline Agents...")
    
    # Agent 1: Document Processor - handles ingestion and chunking
    doc_agent = await orch.create_agent(
        "DocProcessor", 
        "Document Ingestion Specialist"
    )
    print(f"   [+] DocProcessor: {doc_agent[:8]}")
    
    # Agent 2: Retrieval Agent - searches vector store
    retrieval_agent = await orch.create_agent(
        "Retriever",
        "Semantic Search Specialist"
    )
    print(f"   [+] Retriever: {retrieval_agent[:8]}")
    
    # Agent 3: Answer Generator - synthesizes responses
    answer_agent = await orch.create_agent(
        "AnswerBot",
        "Answer Synthesis Expert"
    )
    print(f"   [+] AnswerBot: {answer_agent[:8]}")
    
    # Simulate document ingestion
    print("\n📄 Phase 1: Document Ingestion")
    print("-" * 40)
    
    documents = [
        "Barq-DB is a high-performance vector database built in Rust for AI applications.",
        "NAFS-4 implements a four-system cognitive architecture for autonomous agents.",
        "Vector databases enable semantic search by comparing embedding similarities.",
    ]
    
    for i, doc in enumerate(documents, 1):
        # In production, this would embed and store in Barq-DB
        response = await orch.query(
            doc_agent,
            f"Process and prepare this document for storage: {doc}"
        )
        print(f"   [{i}/{len(documents)}] Processed: {doc[:50]}...")
    
    print("\n   ✅ Documents ingested into vector store")
    
    # RAG Query Flow
    print("\n🔍 Phase 2: RAG Query Execution")
    print("-" * 40)
    
    user_query = "What is the purpose of vector databases in AI?"
    print(f"   User Query: \"{user_query}\"")
    
    # Step 1: Retrieve relevant context
    print("\n   [Step 1] Retrieving relevant documents...")
    retrieval_response = await orch.query(
        retrieval_agent,
        f"Find documents relevant to: {user_query}"
    )
    print(f"   Retrieved context: {retrieval_response.result[:100]}...")
    
    # Step 2: Generate answer with context
    print("\n   [Step 2] Generating answer with retrieved context...")
    augmented_prompt = f"""
    User Question: {user_query}
    
    Retrieved Context:
    {retrieval_response.result}
    
    Based on the context above, provide a concise and accurate answer.
    """
    
    final_response = await orch.query(answer_agent, augmented_prompt)
    
    print("\n" + "=" * 60)
    print("📝 Final RAG Response:")
    print("=" * 60)
    print(final_response.result)
    
    # Display system metadata
    print("\n🧠 Cognitive Pipeline Metadata:")
    for key, value in final_response.metadata.items():
        print(f"   {key}: {value}")
    
    print(f"\n⏱️  Total LLM latency: {final_response.execution_time_ms}ms")
    print("\n✅ RAG pipeline completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
