import asyncio
import nafs
import os
import sys
import time
from typing import List

# Use actual keys if available, otherwise it falls back to Mock logically in Orchestrator
COHERE_KEY = os.environ.get("COHERE_API_KEY", "")

async def process_threat(orchestrator: nafs.Orchestrator, agent_id: str, threat_data: str):
    """Worker function to process a specific threat independently."""
    print(f"🔍 [Agent {agent_id[:8]}] Analyzing: {threat_data[:40]}...")
    
    start = time.time()
    # Call the orchestrator. In the background, NAFS runs System 1->2->3->4
    response = await orchestrator.query(agent_id, threat_data)
    elapsed = time.time() - start
    
    return {
        "agent_id": agent_id,
        "result": response.result,
        "metadata": response.metadata,
        "latency": response.execution_time_ms,
        "wall_time": elapsed
    }

async def main():
    print("🛡️ NAFS-4 Autonomous Cybersecurity Response Center")
    print("="*60)
    
    if not COHERE_KEY:
        print("⚠️  Running in MOCK mode (No COHERE_API_KEY found)")
    else:
        print("🚀 Systems Online with Cohere Intelligence")

    # 1. Initialize Orchestrator
    try:
        orch = await nafs.Orchestrator.create()
    except Exception as e:
        print(f"❌ Initialization Failed: {e}")
        return

    # 2. Spawn Specialized Cyber-Response Agents
    print("\n👥 Spawning Parallel Response Units...")
    
    # We spawn them in parallel for speed
    agent_types = [
        ("NIDS_Watcher", "Packet Inspection Specialist"),
        ("IAM_Guard", "Authentication Auditor"),
        ("Kernel_Shield", "System Call Monitor")
    ]
    
    creation_tasks = [orch.create_agent(name, role) for name, role in agent_types]
    agent_ids = await asyncio.gather(*creation_tasks)
    
    for (name, _), aid in zip(agent_types, agent_ids):
        print(f"  [+] {name} activated (ID: {aid[:8]})")

    # 3. Simulated Concurrent Threat Intel
    threats = [
        "Incoming brute-force attempt detected on SSH port 22 from IP 192.168.1.105",
        "Anomalous outbound traffic spike to unknown domain 'malicious-c2.top'",
        "Unauthorized sudo attempt by restricted user 'developer_04'"
    ]

    print(f"\n⚡ Processing {len(threats)} threats concurrently across Systems 1-4...")
    print("-" * 60)

    # 4. Execute Parallel Processing
    processing_tasks = [
        process_threat(orch, aid, threat) 
        for aid, threat in zip(agent_ids, threats)
    ]
    
    results = await asyncio.gather(*processing_tasks)

    # 5. Review Multi-System Cognitive Results
    for res in results:
        metadata = res["metadata"]
        print(f"\n📊 Threat Analysis Report (Agent: {res['agent_id'][:8]})")
        print(f"   Response: {res['result'][:100]}...")
        print(f"   ⏱️  LLM Latency: {res['latency']}ms")
        
        print("\n   🧠 Cognitive Pipeline Logs:")
        print(f"     ⚡ System 1 (Perception): {metadata.get('system1_perception', 'N/A')}")
        print(f"     🤔 System 2 (Reasoning):  {metadata.get('system2_reasoning', 'N/A')}")
        print(f"     📚 System 3 (Memory):     {metadata.get('system3_memory', 'N/A')}")
        print(f"     🌱 System 4 (Evolution):  {metadata.get('system4_evolution', 'N/A')}")
        print("-" * 40)

    print("\n✅ All incident response units completed their cognitive cycles.")

if __name__ == "__main__":
    asyncio.run(main())
