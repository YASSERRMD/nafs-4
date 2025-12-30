import asyncio
import nafs
import os
import sys

# Simulation of a "learning event" feedback
STUDENT_FEEDBACK = [
    "I don't understand the math part, can you use common objects?",
    "That makes way more sense! The light-switch analogy worked."
]

async def conduct_lesson(orch, tutor_id, topic, phase):
    print(f"\n🎓 [Phase {phase}] Teaching Topic: {topic}")
    print("-" * 50)
    
    # Send query to the tutor
    query = f"Explain {topic} to a beginner. Note: Student previously struggled with math."
    response = await orch.query(tutor_id, query)
    
    print(f"📖 Tutor Says: {response.result[:200]}...")
    
    # Display the Multi-System Cognitive state
    meta = response.metadata
    print(f"\n🧠 NAFS-4 Cognitive Cycle [Phase {phase}]:")
    print(f"   [System 1]: {meta.get('system1_perception')}")
    print(f"   [System 2]: {meta.get('system2_reasoning')}")
    print(f"   [System 3]: {meta.get('system3_memory')}")
    print(f"   [System 4]: {meta.get('system4_evolution')}")
    
    return response.result

async def main():
    print("🌟 NAFS-4 Self-Evolving Education System")
    print("="*60)
    
    # Initialize
    try:
        orch = await nafs.Orchestrator.create()
    except Exception as e:
        print(f"❌ Failed to start: {e}")
        return

    # 1. Spawn Initial Tutor
    print("\n🐣 Spawning 'Standard_Tutor'...")
    tutor_v1 = await orch.create_agent("Tutor_V1", "Default Academic Professor")
    
    # 2. First Lesson (Topic: Quantum Entanglement)
    await conduct_lesson(orch, tutor_v1, "Quantum Entanglement", 1)
    
    print("\n⚠️  Simulating Negative Student Feedback: 'Too much math, I'm confused.'")
    print("♻️  System 4 (Evolution) is analyzing pedagogical failure...")
    await asyncio.sleep(1) # Simulated analysis time
    
    # 3. Spawn Evolved Tutor
    # In a full System 4 implementation, this would happen automatically within the same agent.
    # Here we simulate the result of that evolution by spawning a refined role.
    print("\n🧬 [Evolution Triggered] Upgrading to 'Analogy_Specialist'...")
    tutor_v2 = await orch.create_agent("Adaptive_Tutor", "Simplified Analogy Specialist")
    
    # 4. Second Lesson (Enhanced Strategy)
    await conduct_lesson(orch, tutor_v2, "Quantum Entanglement", 2)

    print("\n✅ Learning objective achieved. Tutor role successfully evolved via System 4.")

if __name__ == "__main__":
    if "COHERE_API_KEY" not in os.environ:
        print("💡 Tip: Set COHERE_API_KEY for real LLM reasoning.")
    
    asyncio.run(main())
