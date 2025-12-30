import asyncio
import nafs
import os
import sys

async def main():
    print("🔬 Initializing NAFS Research Assistant System...")
    
    # Check for API Key
    if "COHERE_API_KEY" in os.environ:
        print("✅ COHERE_API_KEY found. Using real Cohere LLM.")
    else:
        print("⚠️  Warning: COHERE_API_KEY not found. Operations will use Mock LLM.")

    try:
        orch = await nafs.Orchestrator.create()
        print("🚀 System Online.")

        # Create Specialized Agents
        print("\n👥 Creating Agents...")
        planner = await orch.create_agent("Planner_Lead", "Assistant") 
        print(f"  [+] Planner ready: {planner}")
        
        writer = await orch.create_agent("Writer_Bot", "Assistant")
        print(f"  [+] Writer ready:  {writer}")
        
        # Phase 1: Planning
        print("\n📝 Phase 1: Planning")
        topic = "The Future of Neuromorphic AI"
        prompt_plan = f"Create a brief 3-point research outline for: '{topic}'. Be concise."
        
        print(f"   Requesting plan for '{topic}'...")
        plan = await orch.query(planner, prompt_plan)
        print(f"\n📋 Research Plan:\n{'-'*20}\n{plan}\n{'-'*20}")
        
        # Phase 2: Writing (Using context from Phase 1)
        print("\n✍️  Phase 2: Writing")
        prompt_write = f"Using this outline, write a short introductory paragraph:\n{plan}"
        
        print(f"   Drafting content based on plan...")
        content = await orch.query(writer, prompt_write)
        print(f"\n📄 Final Draft:\n{'-'*20}\n{content}\n{'-'*20}")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
