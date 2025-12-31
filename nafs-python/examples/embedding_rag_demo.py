#!/usr/bin/env python3
"""
NAFS-4 Multi-Provider Embedding Demo

Demonstrates configuring different LLM/Embedding providers.

Supported Providers & Models:

┌─────────────────┬───────────────────────────────────────────────────┐
│ Provider        │ Embedding Models                                  │
├─────────────────┼───────────────────────────────────────────────────┤
│ Cohere          │ embed-english-v3.0, embed-multilingual-v3.0      │
│                 │ embed-english-light-v3.0, embed-multilingual-light│
├─────────────────┼───────────────────────────────────────────────────┤
│ OpenAI          │ text-embedding-3-small, text-embedding-3-large   │
│                 │ text-embedding-ada-002                            │
├─────────────────┼───────────────────────────────────────────────────┤
│ HuggingFace     │ sentence-transformers/all-MiniLM-L6-v2           │
│                 │ BAAI/bge-small-en-v1.5, intfloat/e5-base-v2       │
├─────────────────┼───────────────────────────────────────────────────┤
│ Ollama (local)  │ nomic-embed-text, mxbai-embed-large              │
│                 │ all-minilm, snowflake-arctic-embed                │
└─────────────────┴───────────────────────────────────────────────────┘

Environment Variables (select ONE provider):
    export HUGGINGFACE_API_KEY=your_key    # Use HuggingFace
    export OLLAMA_URL=http://localhost:11434  # Use local Ollama
    export COHERE_API_KEY=your_key         # Use Cohere
    export OPENAI_API_KEY=your_key         # Use OpenAI
    export ANTHROPIC_API_KEY=your_key      # Use Anthropic (no embeddings)
"""

import asyncio
import nafs
import os

async def main():
    print("=" * 70)
    print("🔌 NAFS-4 Multi-Provider Demo")
    print("=" * 70)
    
    # Show provider selection rules
    print("\n📋 Provider Selection Priority:")
    print("   1. HUGGINGFACE_API_KEY → HuggingFace Inference API")
    print("   2. OLLAMA_URL → Local Ollama")
    print("   3. COHERE_API_KEY → Cohere")
    print("   4. OPENAI_API_KEY → OpenAI")
    print("   5. ANTHROPIC_API_KEY → Anthropic")
    print("   6. (none) → Mock Provider")
    
    # Initialize
    orch = await nafs.Orchestrator.create()
    
    # Show active provider
    provider = orch.get_provider_name()
    models = orch.get_embedding_models()
    
    print(f"\n🎯 Active Provider: {provider.upper()}")
    print(f"   Available Embedding Models ({len(models)}):")
    for m in models[:5]:
        print(f"      • {m}")
    if len(models) > 5:
        print(f"      ... and {len(models) - 5} more")
    
    # Test embedding
    test_text = "NAFS-4 is a cognitive architecture for AI agents."
    
    print(f"\n📝 Test Embedding:")
    print(f'   Text: "{test_text}"')
    
    # Default model
    current_model = await orch.get_embedding_model()
    print(f"\n   Session Model: {current_model if current_model else '(provider default)'}")
    
    embedding = await orch.embed(test_text)
    print(f"   Result: {len(embedding)} dimensions")
    
    # Try different models
    if len(models) >= 2:
        print(f"\n🔄 Switching Models:")
        for model in models[:3]:
            try:
                await orch.set_embedding_model(model)
                emb = await orch.embed(test_text)
                print(f"   ✅ {model:<40} → {len(emb):>5} dims")
            except Exception as e:
                print(f"   ❌ {model:<40} → {str(e)[:30]}")
    
    # Reset to default
    await orch.set_embedding_model(None)
    
    print("\n" + "=" * 70)
    print("💡 To switch providers, set a different environment variable and restart.")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())
