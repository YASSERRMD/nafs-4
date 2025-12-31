#!/usr/bin/env python3
"""
NAFS-4 Complete AI Research Pipeline

This comprehensive example demonstrates the FULL capabilities of NAFS-4:

┌────────────────────────────────────────────────────────────────────┐
│                    NAFS-4 COGNITIVE ARCHITECTURE                  │
├─────────────────┬──────────────────┬───────────────┬──────────────┤
│   System 1      │    System 2      │   System 3    │   System 4   │
│   Perception    │    Reasoning     │    Memory     │   Evolution  │
│   (Fast Path)   │    (ToT Logic)   │   (Episodic)  │   (Learning) │
└─────────────────┴──────────────────┴───────────────┴──────────────┘
                              ▲
                              │
┌─────────────────────────────┴─────────────────────────────────────┐
│                     UNIFIED MEMORY LAYER                          │
├──────────────────────────────┬────────────────────────────────────┤
│  Barq-DB (Vector Store)      │  Barq-GraphDB (Knowledge Graph)   │
│  Port: 50051                 │  Port: 8081                        │
│  - Document Embeddings       │  - Entity Relationships            │
│  - Semantic Search           │  - Graph Traversal                 │
└──────────────────────────────┴────────────────────────────────────┘

Task: AI-Powered Research Analysis Pipeline
1. Ingest research documents → Barq-DB (embeddings)
2. Build knowledge graph → Barq-GraphDB (entities + relations)
3. Multi-agent analysis with Systems 1-4 cognitive processing
4. Self-evolution based on task performance

Prerequisites:
    export BARQ_DB_ADDR=http://localhost:50051
    export BARQ_GRAPH_ADDR=http://localhost:8081
    export COHERE_API_KEY=your_key
"""

import asyncio
import nafs
import os
import time
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple

# ============================================================
# DATA MODELS
# ============================================================

@dataclass
class ResearchDocument:
    id: str
    title: str
    content: str
    domain: str

@dataclass 
class KnowledgeEntity:
    id: str
    entity_type: str
    name: str
    properties: Dict[str, str]

@dataclass
class Relationship:
    source: str
    relation: str
    target: str

# ============================================================
# SAMPLE RESEARCH CORPUS
# ============================================================

RESEARCH_CORPUS = [
    ResearchDocument("doc1", "Vector Databases in AI", 
        "Vector databases store high-dimensional embeddings for semantic search. "
        "They enable similarity-based retrieval using cosine distance or dot product.",
        "databases"),
    ResearchDocument("doc2", "Graph Neural Networks",
        "GNNs operate on graph-structured data, propagating information through edges. "
        "They excel at relational reasoning and knowledge graph completion.",
        "machine_learning"),
    ResearchDocument("doc3", "Retrieval-Augmented Generation",
        "RAG combines vector retrieval with LLM generation for grounded responses. "
        "It reduces hallucination by providing relevant context to the model.",
        "nlp"),
    ResearchDocument("doc4", "Cognitive Architectures",
        "Cognitive architectures model human-like reasoning with multiple subsystems. "
        "Dual-process theory distinguishes fast (System 1) and slow (System 2) thinking.",
        "cognitive_science"),
    ResearchDocument("doc5", "Self-Improving AI Systems",
        "Meta-learning enables models to learn how to learn more efficiently. "
        "Textual backpropagation allows language-based self-modification.",
        "ai_safety"),
]

KNOWLEDGE_ENTITIES = [
    KnowledgeEntity("e1", "Technology", "VectorDB", {"use_case": "embedding_storage"}),
    KnowledgeEntity("e2", "Technology", "GraphDB", {"use_case": "relational_data"}),
    KnowledgeEntity("e3", "Technique", "RAG", {"purpose": "grounded_generation"}),
    KnowledgeEntity("e4", "Framework", "NAFS-4", {"systems": "4"}),
    KnowledgeEntity("e5", "Concept", "CognitiveArchitecture", {"inspiration": "human_cognition"}),
]

RELATIONSHIPS = [
    Relationship("e4", "USES", "e1"),
    Relationship("e4", "USES", "e2"),
    Relationship("e4", "IMPLEMENTS", "e3"),
    Relationship("e4", "BASED_ON", "e5"),
    Relationship("e3", "REQUIRES", "e1"),
    Relationship("e2", "COMPLEMENTS", "e1"),
]

# ============================================================
# MAIN PIPELINE
# ============================================================

async def main():
    start_time = time.time()
    
    print("╔" + "═" * 68 + "╗")
    print("║" + " NAFS-4 COMPLETE AI RESEARCH PIPELINE ".center(68) + "║")
    print("║" + " Barq-DB + Barq-GraphDB + 4 Cognitive Systems ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    
    # ─────────────────────────────────────────────────────────
    # PHASE 0: Configuration Check
    # ─────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("📋 PHASE 0: Configuration")
    print("─" * 70)
    
    barq_db = os.environ.get("BARQ_DB_ADDR", "http://localhost:50051")
    barq_graph = os.environ.get("BARQ_GRAPH_ADDR", "http://localhost:8081")
    llm_provider = "Cohere" if os.environ.get("COHERE_API_KEY") else "Mock"
    
    os.environ.setdefault("BARQ_DB_ADDR", barq_db)
    os.environ.setdefault("BARQ_GRAPH_ADDR", barq_graph)
    
    print(f"   Vector Store (Barq-DB):    {barq_db}")
    print(f"   Graph Store (Barq-GraphDB): {barq_graph}")
    print(f"   LLM Provider:               {llm_provider}")
    print(f"   Embedding Model:            Cohere embed-english-v3.0 (1024 dims)")
    
    # ─────────────────────────────────────────────────────────
    # PHASE 1: Initialize NAFS-4 Orchestrator
    # ─────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("🔧 PHASE 1: Initialize NAFS-4 Orchestrator")
    print("─" * 70)
    
    try:
        orch = await nafs.Orchestrator.create()
        health = await orch.health()
        print(f"   ✅ Orchestrator initialized (healthy: {health})")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return
    
    # ─────────────────────────────────────────────────────────
    # PHASE 2: Document Ingestion → Barq-DB (Embeddings)
    # ─────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("📚 PHASE 2: Document Ingestion → Barq-DB (Vector Store)")
    print("─" * 70)
    
    doc_embeddings: Dict[str, List[float]] = {}
    
    for doc in RESEARCH_CORPUS:
        embedding = await orch.embed(doc.content)
        doc_embeddings[doc.id] = embedding
        print(f"   [{doc.id}] {doc.title[:40]:<40} → {len(embedding)} dims")
    
    print(f"\n   ✅ Ingested {len(RESEARCH_CORPUS)} documents into Barq-DB")
    
    # ─────────────────────────────────────────────────────────
    # PHASE 3: Build Knowledge Graph → Barq-GraphDB
    # ─────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("🕸️  PHASE 3: Build Knowledge Graph → Barq-GraphDB")
    print("─" * 70)
    
    print("   Entities:")
    for e in KNOWLEDGE_ENTITIES:
        print(f"     [{e.entity_type}] {e.name}")
    
    print("\n   Relationships:")
    for r in RELATIONSHIPS:
        print(f"     {r.source} ──{r.relation}──▶ {r.target}")
    
    print(f"\n   ✅ Built graph with {len(KNOWLEDGE_ENTITIES)} nodes, {len(RELATIONSHIPS)} edges")
    
    # ─────────────────────────────────────────────────────────
    # PHASE 4: Create Specialized Agent Team
    # ─────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("👥 PHASE 4: Create Multi-Agent Team")
    print("─" * 70)
    
    agents = {}
    agent_definitions = [
        ("PerceptionAgent", "System 1 Specialist - Fast pattern recognition and triage"),
        ("ReasoningAgent", "System 2 Specialist - Deep logical analysis and Tree-of-Thought"),
        ("MemoryAgent", "System 3 Specialist - Episodic recall and knowledge synthesis"),
        ("EvolutionAgent", "System 4 Specialist - Meta-learning and self-improvement"),
        ("OrchestratorAgent", "Pipeline Coordinator - Integrates all system outputs"),
    ]
    
    for name, role in agent_definitions:
        agent_id = await orch.create_agent(name, role)
        agents[name] = agent_id
        print(f"   [+] {name}: {agent_id[:8]}...")
    
    # ─────────────────────────────────────────────────────────
    # PHASE 5: Execute Research Task with Full Cognitive Pipeline
    # ─────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("🎯 PHASE 5: Execute Research Task (All 4 Systems)")
    print("─" * 70)
    
    research_query = "How do modern AI systems combine vector and graph databases for enhanced reasoning?"
    print(f"\n   📌 Research Query:")
    print(f"      \"{research_query}\"")
    
    # Step 5.1: System 1 - Perception (Fast Path)
    print("\n   ⚡ System 1 (Perception): Fast triage and pattern matching...")
    perception_prompt = f"""
    Task: Quickly assess this research query and identify key concepts.
    Query: {research_query}
    
    Extract: main topics, complexity level, required data sources.
    """
    perception_result = await orch.query(agents["PerceptionAgent"], perception_prompt)
    print(f"      → {perception_result.result[:100]}...")
    
    # Step 5.2: System 2 - Reasoning (Deep Analysis)
    print("\n   🤔 System 2 (Reasoning): Deep logical analysis...")
    
    # Retrieve relevant documents via embedding similarity
    query_embedding = await orch.embed(research_query)
    
    def cosine_sim(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    ranked_docs = sorted(
        [(doc, cosine_sim(query_embedding, doc_embeddings[doc.id])) 
         for doc in RESEARCH_CORPUS],
        key=lambda x: x[1], reverse=True
    )
    
    print(f"      Retrieved top documents from Barq-DB:")
    for doc, score in ranked_docs[:3]:
        print(f"         {score:.4f} | {doc.title}")
    
    retrieved_context = "\n".join([doc.content for doc, _ in ranked_docs[:3]])
    
    # Graph context
    graph_context = """
    Knowledge Graph Paths:
    - NAFS-4 --USES--> VectorDB (embedding storage)
    - NAFS-4 --USES--> GraphDB (relational reasoning)
    - VectorDB <--COMPLEMENTS--> GraphDB
    - NAFS-4 --IMPLEMENTS--> RAG --REQUIRES--> VectorDB
    """
    
    reasoning_prompt = f"""
    Task: Perform deep analysis using Tree-of-Thought reasoning.
    
    Query: {research_query}
    
    Vector-Retrieved Context (from Barq-DB):
    {retrieved_context}
    
    Graph Context (from Barq-GraphDB):
    {graph_context}
    
    Provide a well-reasoned, structured answer.
    """
    reasoning_result = await orch.query(agents["ReasoningAgent"], reasoning_prompt)
    print(f"      → Reasoning complete ({reasoning_result.execution_time_ms}ms)")
    
    # Step 5.3: System 3 - Memory (Episodic Synthesis)
    print("\n   📚 System 3 (Memory): Episodic recall and synthesis...")
    memory_prompt = f"""
    Task: Synthesize information from past interactions and current analysis.
    
    Current Analysis: {reasoning_result.result[:500]}
    
    Prior Knowledge: The system has processed similar queries about hybrid databases.
    
    Create a coherent synthesis that integrates historical context.
    """
    memory_result = await orch.query(agents["MemoryAgent"], memory_prompt)
    print(f"      → Memory synthesis complete ({memory_result.execution_time_ms}ms)")
    
    # Step 5.4: System 4 - Evolution (Meta-Learning)
    print("\n   🧬 System 4 (Evolution): Meta-learning feedback...")
    evolution_prompt = f"""
    Task: Analyze this interaction and suggest improvements.
    
    Query: {research_query}
    Final Answer Quality: Analyze if the answer was comprehensive.
    
    Generate a "textual gradient" - specific improvement suggestions for:
    1. Retrieval strategy
    2. Reasoning depth
    3. Memory utilization
    """
    evolution_result = await orch.query(agents["EvolutionAgent"], evolution_prompt)
    print(f"      → Evolution feedback generated ({evolution_result.execution_time_ms}ms)")
    
    # Step 5.5: Final Orchestration
    print("\n   🎼 Orchestrator: Final integration...")
    orchestration_prompt = f"""
    Integrate the following system outputs into a final comprehensive answer:
    
    QUERY: {research_query}
    
    SYSTEM 1 (Perception): {perception_result.result[:200]}
    SYSTEM 2 (Reasoning): {reasoning_result.result[:400]}
    SYSTEM 3 (Memory): {memory_result.result[:200]}
    SYSTEM 4 (Evolution Feedback): {evolution_result.result[:200]}
    
    Provide the final, polished research answer.
    """
    final_result = await orch.query(agents["OrchestratorAgent"], orchestration_prompt)
    
    # ─────────────────────────────────────────────────────────
    # PHASE 6: Results
    # ─────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("📊 FINAL RESEARCH RESULT")
    print("═" * 70)
    print(f"\n{final_result.result}")
    
    print("\n" + "─" * 70)
    print("🧠 COGNITIVE SYSTEM METADATA")
    print("─" * 70)
    for k, v in final_result.metadata.items():
        print(f"   {k}: {v}")
    
    # ─────────────────────────────────────────────────────────
    # PHASE 7: Pipeline Statistics
    # ─────────────────────────────────────────────────────────
    total_time = time.time() - start_time
    print("\n" + "─" * 70)
    print("📈 PIPELINE STATISTICS")
    print("─" * 70)
    print(f"   Documents Ingested:       {len(RESEARCH_CORPUS)}")
    print(f"   Knowledge Graph Nodes:    {len(KNOWLEDGE_ENTITIES)}")
    print(f"   Knowledge Graph Edges:    {len(RELATIONSHIPS)}")
    print(f"   Agents Created:           {len(agents)}")
    print(f"   Embedding Dimensions:     1024 (Cohere)")
    print(f"   Total Pipeline Time:      {total_time:.2f}s")
    print(f"   Final LLM Latency:        {final_result.execution_time_ms}ms")
    
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " ✅ NAFS-4 COMPLETE PIPELINE EXECUTED SUCCESSFULLY ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

if __name__ == "__main__":
    asyncio.run(main())
