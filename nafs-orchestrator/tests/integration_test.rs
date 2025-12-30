use nafs_core::*;
use nafs_orchestrator::*;
use std::collections::HashMap;

#[tokio::test]
async fn test_agent_lifecycle() {
    let config = OrchestratorConfig {
        persistence_path: "/tmp/nafs_lifecycle_test".to_string(),
        ..Default::default()
    };
    let orchestrator = NafsOrchestrator::new(config).await.unwrap();

    // Create agent
    let role = AgentRole {
        id: "role_1".to_string(),
        name: "TestRole".to_string(),
        system_prompt: "Test".to_string(),
        version: 1,
        capabilities: vec![],
        constraints: vec![],
        evolution_lineage: vec![],
        created_at: chrono::Utc::now(),
        last_updated: chrono::Utc::now(),
    };
    
    let agent = orchestrator.create_agent("Agent1".to_string(), role).await.unwrap();
    assert_eq!(agent.name, "Agent1");

    // List agents
    let agents = orchestrator.list_agents().await.unwrap();
    assert_eq!(agents.len(), 1);

    // Get health
    let health = orchestrator.health_check().await;
    assert_eq!(health.active_agents, 1);

    // Delete agent
    orchestrator.delete_agent(&agent.id).await.unwrap();
    let agents_after = orchestrator.list_agents().await.unwrap();
    assert_eq!(agents_after.len(), 0);
    
    // Cleanup
    let _ = std::fs::remove_dir_all("/tmp/nafs_lifecycle_test");
}

#[tokio::test]
async fn test_request_execution() {
    let config = OrchestratorConfig {
        persistence_path: "/tmp/nafs_request_test".to_string(),
        ..Default::default()
    };
    let orchestrator = NafsOrchestrator::new(config).await.unwrap();

    let role = AgentRole {
        id: "role_1".to_string(),
        name: "TestRole".to_string(),
        system_prompt: "Test".to_string(),
        version: 1,
        capabilities: vec![],
        constraints: vec![],
        evolution_lineage: vec![],
        created_at: chrono::Utc::now(),
        last_updated: chrono::Utc::now(),
    };
    let agent = orchestrator.create_agent("Agent1".to_string(), role).await.unwrap();

    let request = AgentRequest {
        id: "req_1".to_string(),
        agent_id: agent.id.clone(),
        query: "Echo this".to_string(),
        context: HashMap::new(),
        priority: 1,
        timeout_ms: 1000,
        metadata: HashMap::new(),
    };

    let response = orchestrator.execute_request(request).await.unwrap();
    assert!(response.success);
    assert!(response.result.contains("Echo this"));
    
    // Cleanup
    let _ = std::fs::remove_dir_all("/tmp/nafs_request_test");
}
