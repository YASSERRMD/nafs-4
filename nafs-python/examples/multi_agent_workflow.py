import asyncio
import nafs
import time

async def main():
    print("🚀 Starting Multi-Agent Workflow Simulation...\n")

    # 1. Initialize System
    orch = await nafs.Orchestrator.create()
    if not await orch.health():
        print("❌ System unhealthy, aborting.")
        return

    # 2. Spawn Agents
    print("👥 Spawning Team...")
    
    # Ingestion Agent
    ingest_id = await orch.create_agent("Ingestor_01", "Assistant")
    print(f"  [+] Ingestion Agent ready: {ingest_id}")
    
    # Analyst Agent
    analyst_id = await orch.create_agent("Analyst_Alpha", "Assistant")
    print(f"  [+] Analyst Agent ready:   {analyst_id}")
    
    # Reporter Agent
    reporter_id = await orch.create_agent("Reporter_X", "Assistant")
    print(f"  [+] Reporter Agent ready:  {reporter_id}")
    
    print("\n🎬 Executing Workflow Pipeline:")
    print("="*40)

    # Step 1: Ingestion
    print("\n🔹 Step 1: Data Ingestion")
    raw_data_query = "Fetch daily transaction logs for processing."
    data_response = await orch.query(ingest_id, raw_data_query)
    print(f"   Input: '{raw_data_query}'")
    print(f"   Output: {data_response}") 
    # (In a real system, we'd parse this output)

    # Step 2: Analysis
    print("\n🔹 Step 2: Data Analysis")
    analysis_query = f"Analyze the following data for anomalies: [Simulated Data Block]"
    analysis_result = await orch.query(analyst_id, analysis_query)
    print(f"   Input: Sent data to Analyst...")
    print(f"   Output: {analysis_result}")

    # Step 3: Reporting
    print("\n🔹 Step 3: Final Report Generation")
    report_query = f"Generate executive summary based on findings: {analysis_result}"
    final_report = await orch.query(reporter_id, report_query)
    print(f"   Input: Requesting summary...")
    print(f"   Output: {final_report}")

    print("\n"+"="*40)
    print("✅ Workflow Complete!")

if __name__ == "__main__":
    asyncio.run(main())
