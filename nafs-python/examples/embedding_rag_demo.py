#!/usr/bin/env python3
"""
NAFS-4 Multi-Model Embedding Demo

Demonstrates:
1. Querying available embedding models from the provider
2. Generating embeddings with different models
3. Comparing embedding dimensions across models

Supported Embedding Models:
    
    Cohere:
        - embed-english-v3.0 (1024 dims)
        - embed-english-light-v3.0 (384 dims)  
        - embed-multilingual-v3.0 (1024 dims)
        - embed-multilingual-light-v3.0 (384 dims)
    
    OpenAI:
        - text-embedding-3-small (1536 dims)
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
    print("=" * 65)
    print("🧠 NAFS-4 Multi-Model Embedding Demo")
    print("=" * 65)
    
    # Initialize
    orch = await nafs.Orchestrator.create()
    
    # Get provider info
    provider = orch.get_provider_name()
    models = orch.get_embedding_models()
    
    print(f"\n📋 Configuration:")
    print(f"   LLM Provider: {provider}")
    print(f"   Available Embedding Models:")
    for model in models:
        print(f"      • {model}")
    
    if not models:
        print("\n⚠️  No embedding models available for this provider.")
        return
    
    # Test text
    test_text = "NAFS-4 is a cognitive architecture framework implementing four distinct systems."
    
    print(f"\n📝 Test Text:")
    print(f'   "{test_text}"')
    
    # Generate embeddings with each available model
    print("\n📊 Embedding Results:")
    print("-" * 65)
    
    results = []
    
    for model in models:
        try:
            print(f"\n   Model: {model}")
            embedding = await orch.embed_with_model(test_text, model)
            dim = len(embedding)
            norm = np.linalg.norm(embedding)
            results.append((model, dim, norm, embedding))
            print(f"      ✅ Dimensions: {dim}, L2 Norm: {norm:.4f}")
        except Exception as e:
            print(f"      ❌ Error: {e}")
    
    # Compare embeddings if we have multiple
    if len(results) >= 2:
        print("\n🔍 Model Comparison (Cosine Similarity):")
        print("-" * 65)
        
        def cosine_sim(a, b):
            # Handle different dimensions by padding
            if len(a) != len(b):
                return float('nan')
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        
        for i, (m1, d1, _, e1) in enumerate(results):
            for m2, d2, _, e2 in results[i+1:]:
                sim = cosine_sim(e1, e2)
                if np.isnan(sim):
                    print(f"   {m1[:25]:<25} vs {m2[:25]:<25}: N/A (different dims: {d1} vs {d2})")
                else:
                    print(f"   {m1[:25]:<25} vs {m2[:25]:<25}: {sim:.4f}")
    
    # Show default model usage
    print("\n� Default Model Usage:")
    print("-" * 65)
    default_embedding = await orch.embed(test_text)
    print(f"   orch.embed(text) → {len(default_embedding)} dimensions")
    print(f"   (Uses provider's default model)")
    
    print("\n✅ Multi-model embedding demo completed!")

if __name__ == "__main__":
    asyncio.run(main())
