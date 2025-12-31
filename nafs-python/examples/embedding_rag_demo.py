#!/usr/bin/env python3
"""
NAFS-4 Complete Multi-Provider Demo

Supports 11+ LLM/Embedding Providers:

┌─────────────────┬─────────────────────────────────────────────────────────────┐
│ Provider        │ Features                                                    │
├─────────────────┼─────────────────────────────────────────────────────────────┤
│ Together.ai     │ LLM: Llama-3, Mixtral, Qwen  |  Embed: m2-bert, bge        │
│ Groq            │ LLM: Llama-3.1, Mixtral (fast!)  |  Embed: ❌              │
│ Fireworks       │ LLM: Llama, Mixtral, Qwen  |  Embed: nomic, UAE            │
│ Voyage AI       │ LLM: ❌  |  Embed: voyage-3, voyage-code, voyage-law       │
│ Jina AI         │ LLM: ❌  |  Embed: jina-v3, jina-colbert, jina-clip        │
│ HuggingFace     │ LLM: Mistral, Llama  |  Embed: all-MiniLM, bge, e5         │
│ Ollama          │ LLM: llama3, mistral, phi  |  Embed: nomic, mxbai          │
│ Cohere          │ LLM: command-r  |  Embed: embed-v3 (english/multilingual) │
│ OpenAI          │ LLM: gpt-4, gpt-3.5  |  Embed: text-embedding-3-small/large│
│ Anthropic       │ LLM: claude-3  |  Embed: ❌                                │
│ Azure OpenAI    │ LLM: deployed models  |  Embed: deployed models            │
└─────────────────┴─────────────────────────────────────────────────────────────┘

Environment Variables (set ONE to select provider):
    TOGETHER_API_KEY      → Together.ai
    GROQ_API_KEY          → Groq (fastest LLM)
    FIREWORKS_API_KEY     → Fireworks AI
    VOYAGE_API_KEY        → Voyage AI (embeddings only)
    JINA_API_KEY          → Jina AI (embeddings only)
    HUGGINGFACE_API_KEY   → HuggingFace Inference
    OLLAMA_URL            → Local Ollama (e.g., http://localhost:11434)
    COHERE_API_KEY        → Cohere
    OPENAI_API_KEY        → OpenAI
    ANTHROPIC_API_KEY     → Anthropic
"""

import asyncio
import nafs
import os

async def main():
    print("=" * 75)
    print("🌐 NAFS-4 Multi-Provider System")
    print("=" * 75)
    
    # Initialize
    orch = await nafs.Orchestrator.create()
    
    # Get provider info
    provider = orch.get_provider_name()
    embedding_models = orch.get_embedding_models()
    
    print(f"\n🎯 Active Provider: {provider.upper()}")
    
    # Show capabilities
    if embedding_models:
        print(f"\n📊 Available Embedding Models ({len(embedding_models)}):")
        for m in embedding_models[:6]:
            print(f"   • {m}")
        if len(embedding_models) > 6:
            print(f"   ... and {len(embedding_models) - 6} more")
    else:
        print(f"\n⚠️  {provider} does not support embeddings")
    
    # Test embedding if available
    if embedding_models:
        test_text = "NAFS-4 cognitive architecture for autonomous AI agents."
        print(f"\n📝 Test Embedding:")
        print(f'   "{test_text[:50]}..."')
        
        try:
            embedding = await orch.embed(test_text)
            print(f"   ✅ Success: {len(embedding)} dimensions")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # Try setting a different model
        if len(embedding_models) >= 2:
            alt_model = embedding_models[1]
            print(f"\n🔄 Switching to: {alt_model}")
            await orch.set_embedding_model(alt_model)
            
            try:
                embedding = await orch.embed(test_text)
                print(f"   ✅ Success: {len(embedding)} dimensions")
            except Exception as e:
                print(f"   ❌ Error: {e}")
    
    # Show all available providers
    print("\n" + "─" * 75)
    print("📋 All Supported Providers:")
    print("─" * 75)
    
    providers = [
        ("TOGETHER_API_KEY", "Together.ai", "LLM + Embeddings"),
        ("GROQ_API_KEY", "Groq", "LLM only (fastest)"),
        ("FIREWORKS_API_KEY", "Fireworks AI", "LLM + Embeddings"),
        ("VOYAGE_API_KEY", "Voyage AI", "Embeddings only"),
        ("JINA_API_KEY", "Jina AI", "Embeddings only"),
        ("HUGGINGFACE_API_KEY", "HuggingFace", "LLM + Embeddings"),
        ("OLLAMA_URL", "Ollama (local)", "LLM + Embeddings"),
        ("COHERE_API_KEY", "Cohere", "LLM + Embeddings"),
        ("OPENAI_API_KEY", "OpenAI", "LLM + Embeddings"),
        ("ANTHROPIC_API_KEY", "Anthropic", "LLM only"),
    ]
    
    for env_var, name, capabilities in providers:
        is_active = provider.lower() == name.lower().split()[0]
        marker = "→" if is_active else " "
        status = "✓" if os.environ.get(env_var) else " "
        print(f"   {marker} [{status}] {name:<20} ({capabilities})")
    
    print("\n" + "=" * 75)
    print("💡 Set an environment variable and restart to switch providers.")
    print("=" * 75)

if __name__ == "__main__":
    asyncio.run(main())
