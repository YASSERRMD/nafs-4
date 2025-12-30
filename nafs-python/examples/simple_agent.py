import asyncio
import nafs
import sys
import time

async def main():
    print("🚀 Initializing NAFS Orchestrator through Python Bindings...")
    start_time = time.time()
    
    try:
        # 1. Create the Orchestrator instance
        # This initializes the underlying Rust system (Tokio runtime, etc.)
        orchestrator = await nafs.Orchestrator.create()
        print(f"✅ Orchestrator started in {time.time() - start_time:.2f}s")

        # 2. Check System Health
        is_healthy = await orchestrator.health()
        print(f"🏥 System Health: {'Healthy' if is_healthy else 'Unhealthy'}")

        # 3. Create a new Agent
        agent_name = "PyAgent_Alpha"
        agent_role = "Assistant"
        print(f"\n🤖 Creating agent '{agent_name}' with role '{agent_role}'...")
        
        agent_id = await orchestrator.create_agent(agent_name, agent_role)
        print(f"✅ Agent created successfully! ID: {agent_id}")

        # 4. Interact with the Agent
        query_text = "Please verify your operational status."
        print(f"\n📨 Sending query to agent: '{query_text}'")
        
        result = await orchestrator.query(agent_id, query_text)
        print(f"\n📝 Response from Agent:\n{'-'*20}\n{result}\n{'-'*20}")

    except Exception as e:
        print(f"❌ Error during execution: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Ensure Windows compatibility if needed, though this is mac
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    asyncio.run(main())
