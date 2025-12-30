use nafs_core::OrchestratorConfig;
use nafs_server::app;
use tokio::net::TcpListener;

#[tokio::test]
async fn test_end_to_end_pipeline() {
    // 1. Start Server
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let port = addr.port();
    
    // Create a temporary directory for persistence to avoid conflicts
    let tmp_dir = format!("/tmp/nafs_e2e_{}", uuid::Uuid::new_v4());
    let config = OrchestratorConfig {
        persistence_path: tmp_dir.clone(),
        ..Default::default()
    };
    
    let app = app(config).await.unwrap();
    
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let client = reqwest::Client::new();
    let base_url = format!("http://127.0.0.1:{}", port);

    // Give server a moment to start
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // 2. Check Health
    let resp = client.get(format!("{}/health", base_url))
        .send().await.unwrap();
    assert!(resp.status().is_success());
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["status"], "healthy");

    // 3. Create Agent
    let agent_name = "E2E_Test_Agent";
    let resp = client.post(format!("{}/agents", base_url))
        .json(&serde_json::json!({
            "name": agent_name,
            "role": "Assistant"
        }))
        .send().await.unwrap();
    assert!(resp.status().is_success());
    let agent: serde_json::Value = resp.json().await.unwrap();
    let agent_id = agent["id"].as_str().unwrap().to_string();
    assert_eq!(agent["name"], agent_name);

    // 4. Query Agent
    let resp = client.post(format!("{}/agents/{}/query", base_url, agent_id))
        .json(&serde_json::json!({
            "query": "Hello NAFS"
        }))
        .send().await.unwrap();
    assert!(resp.status().is_success());
    let result: serde_json::Value = resp.json().await.unwrap();
    assert!(result["success"].as_bool().unwrap());
    
    // 5. List Agents
    let resp = client.get(format!("{}/agents", base_url)).send().await.unwrap();
    let list: serde_json::Value = resp.json().await.unwrap();
    let agents = list["agents"].as_array().unwrap();
    assert!(agents.iter().any(|a| a["id"] == agent_id));

    // 6. Delete Agent
    let resp = client.delete(format!("{}/agents/{}", base_url, agent_id))
        .send().await.unwrap();
    assert!(resp.status().is_success());

    // 7. Verify Deletion
    let resp = client.get(format!("{}/agents/{}", base_url, agent_id))
        .send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::NOT_FOUND);
    
    // Cleanup
    let _ = std::fs::remove_dir_all(&tmp_dir);
}
