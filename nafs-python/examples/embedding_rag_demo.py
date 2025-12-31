#!/usr/bin/env python3
"""
NAFS-4 Configurable Embedding Demo

Demonstrates:
1. Setting the embedding model for the session
2. Switching between different embedding models
3. One-time model override vs session default

Supported Embedding Models:
    
    Cohere:
        - embed-english-v3.0 (1024 dims) - Default
        - embed-english-light-v3.0 (384 dims)  
        - embed-multilingual-v3.0 (1024 dims)
        - embed-multilingual-light-v3.0 (384 dims)
    
    OpenAI:
        - text-embedding-3-small (1536 dims) - Default
        - text-embedding-3-large (3072 dims)
        - text-embedding-ada-002 (1536 dims)

Prerequisites:
    export COHERE_API_KEY=your_key   # For Cohere models
    # OR
    export OPENAI_API_KEY=your_key   # For OpenAI models
"""

import asyncio
import nafs
import os
import numpy as np

async def main():
    print("=" * 70)
    print("🔧 NAFS-4 Configurable Embedding Demo")
    print("=" * 70)
    
    # Initialize orchestrator
    orch = await nafs.Orchestrator.create()
    
    # Get provider info
    provider = orch.get_provider_name()
    models = orch.get_embedding_models()
    
    print(f"\n📋 Provider: {provider}")
    print(f"   Available Models: {', '.join(models)}")
    
    if not models:
        print("\n⚠️  No embedding models available.")
        return
    
    test_text = "NAFS-4 is a cognitive architecture framework."
    
    # ─────────────────────────────────────────────────────────
    # 1. Default model (provider's default)
    # ─────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("1️⃣  Using Provider's Default Model")
    print("─" * 70)
    
    current = await orch.get_embedding_model()
    print(f"   Current session model: {current if current else '(provider default)'}")
    
    embedding = await orch.embed(test_text)
    print(f"   Result: {len(embedding)} dimensions")
    
    # ─────────────────────────────────────────────────────────
    # 2. Set session default to a specific model
    # ─────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("2️⃣  Setting Session Default Model")
    print("─" * 70)
    
    # Pick a different model if available
    if len(models) > 1:
        new_model = models[1]  # Pick second model
        print(f"   Setting session model to: {new_model}")
        await orch.set_embedding_model(new_model)
        
        current = await orch.get_embedding_model()
        print(f"   Current session model: {current}")
        
        embedding = await orch.embed(test_text)
        print(f"   Result: {len(embedding)} dimensions")
    else:
        print("   (Only one model available, skipping)")
    
    # ─────────────────────────────────────────────────────────
    # 3. One-time override (doesn't change session default)
    # ─────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("3️⃣  One-Time Model Override")
    print("─" * 70)
    
    override_model = models[0]  # Use first model as override
    print(f"   Using one-time override: {override_model}")
    
    embedding = await orch.embed_with_model(test_text, override_model)
    print(f"   Result: {len(embedding)} dimensions")
    
    # Verify session default is unchanged
    current = await orch.get_embedding_model()
    print(f"   Session model still: {current}")
    
    # ─────────────────────────────────────────────────────────
    # 4. Reset to provider default
    # ─────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("4️⃣  Reset to Provider Default")
    print("─" * 70)
    
    await orch.set_embedding_model(None)
    current = await orch.get_embedding_model()
    print(f"   Session model: {current if current else '(provider default)'}")
    
    embedding = await orch.embed(test_text)
    print(f"   Result: {len(embedding)} dimensions")
    
    # ─────────────────────────────────────────────────────────
    # 5. Benchmark all models
    # ─────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("5️⃣  Benchmark All Models")
    print("─" * 70)
    
    for model in models:
        try:
            embedding = await orch.embed_with_model(test_text, model)
            print(f"   ✅ {model:<35} → {len(embedding):>5} dims")
        except Exception as e:
            print(f"   ❌ {model:<35} → Error: {e}")
    
    print("\n" + "=" * 70)
    print("✅ Configurable embedding demo completed!")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())
