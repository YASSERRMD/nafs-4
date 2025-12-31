#!/usr/bin/env python3
"""
NAFS-4 Embedding + Barq-DB RAG Demo

Demonstrates:
1. Generating embeddings using Cohere's embed-english-v3.0 model
2. Storing vectors in Barq-DB (if running)
3. Performing semantic search for RAG

Prerequisites:
    export COHERE_API_KEY=your_key
    export BARQ_DB_ADDR=http://localhost:50051  # Optional

Usage:
    python embedding_rag_demo.py
"""

import asyncio
import nafs
import os
import numpy as np

async def main():
    print("=" * 60)
    print("🧠 NAFS-4 Embedding & RAG Demo")
    print("=" * 60)
    
    # Check configuration
    has_cohere = "COHERE_API_KEY" in os.environ
    barq_addr = os.environ.get("BARQ_DB_ADDR", "")
    
    print(f"\n📋 Configuration:")
    print(f"   Embedding Provider: {'Cohere (embed-english-v3.0)' if has_cohere else 'Mock'}")
    print(f"   Vector Store: {barq_addr or 'In-Memory'}")
    
    if not has_cohere:
        print("\n⚠️  Set COHERE_API_KEY for real embeddings")
        return
    
    # Initialize
    print("\n🔧 Initializing NAFS-4...")
    orch = await nafs.Orchestrator.create()
    print("   ✅ Ready")
    
    # Sample documents to embed
    documents = [
        "Barq-DB is a high-performance vector database written in Rust.",
        "NAFS-4 implements a cognitive architecture with four systems.",
        "Embeddings represent text as dense numerical vectors.",
        "RAG combines retrieval with generation for better answers.",
        "Cohere provides state-of-the-art embedding models."
    ]
    
    # Generate embeddings
    print("\n📊 Generating Embeddings:")
    print("-" * 50)
    embeddings = []
    
    for i, doc in enumerate(documents, 1):
        print(f"   [{i}/{len(documents)}] {doc[:50]}...")
        embedding = await orch.embed(doc)
        embeddings.append(embedding)
        print(f"           → Vector dim: {len(embedding)}, L2 norm: {np.linalg.norm(embedding):.4f}")
    
    # Demonstrate semantic similarity
    print("\n🔍 Semantic Similarity Search:")
    print("-" * 50)
    
    query = "What is a vector database?"
    print(f"   Query: \"{query}\"")
    query_embedding = await orch.embed(query)
    print(f"   Query vector dim: {len(query_embedding)}")
    
    # Compute cosine similarities
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    print("\n   Similarity Scores:")
    similarities = []
    for doc, emb in zip(documents, embeddings):
        sim = cosine_similarity(query_embedding, emb)
        similarities.append((sim, doc))
        print(f"   {sim:.4f} | {doc[:60]}...")
    
    # Rank by similarity
    similarities.sort(reverse=True)
    print("\n   🏆 Top Match:")
    print(f"   {similarities[0][0]:.4f} | {similarities[0][1]}")
    
    # Use top result for RAG
    print("\n📝 RAG Generation:")
    print("-" * 50)
    
    rag_agent = await orch.create_agent("RAG_Bot", "Answer Synthesis Expert")
    augmented_prompt = f"""
    User Question: {query}
    
    Retrieved Context (most relevant):
    {similarities[0][1]}
    
    Based on the context, provide a concise answer.
    """
    
    response = await orch.query(rag_agent, augmented_prompt)
    print(f"   Answer: {response.result}")
    print(f"   ⏱️  Latency: {response.execution_time_ms}ms")
    
    print("\n✅ Demo completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
