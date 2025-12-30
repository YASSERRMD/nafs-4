use axum::{
    extract::{Path, Json, State},
    http::StatusCode,
    routing::{get, post, delete},
    Router,
    response::IntoResponse,
};
use nafs_core::*;
use nafs_orchestrator::NafsOrchestrator;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde_json::json;
use tracing;

type SharedOrchestrator = Arc<RwLock<NafsOrchestrator>>;

pub async fn app(config: OrchestratorConfig) -> Result<Router> {
    let orchestrator = Arc::new(RwLock::new(
        NafsOrchestrator::new(config).await?
    ));

    let app = Router::new()
        .route("/health", get(health_check))
        .route("/stats", get(system_stats))
        .route("/ready", get(readiness_check))
        .route("/agents", post(create_agent).get(list_agents))
        .route("/agents/:id", get(get_agent).delete(delete_agent))
        .route("/agents/:id/query", post(query_agent))
        .route("/agents/:id/memory/search", post(search_memory))
        .route("/agents/:id/memory/recall", get(recall_memory))
        .route("/agents/:id/memory/export", get(export_memory))
        .route("/agents/:id/evolve", post(evolve_agent))
        .route("/agents/:id/evolution/history", get(evolution_history))
        .route("/agents/:id/evolution/rollback", post(rollback_evolution))
        .with_state(orchestrator);

    Ok(app)
}


// Handlers (copied from main.rs, made pub or local) - Actually I should move them here.

async fn health_check(State(orchestrator): State<SharedOrchestrator>) -> Json<serde_json::Value> {
    let orch = orchestrator.read().await;
    let health = orch.health_check().await;
    
    Json(json!({
        "status": if health.is_healthy { "healthy" } else { "unhealthy" },
        "active_agents": health.active_agents,
        "uptime_seconds": health.uptime_seconds,
        "timestamp": health.timestamp.to_rfc3339()
    }))
}

async fn system_stats(State(orchestrator): State<SharedOrchestrator>) -> Json<serde_json::Value> {
    let orch = orchestrator.read().await;
    let stats = orch.get_stats().await;
    
    Json(json!({
        "total_requests": stats.total_requests,
        "successful_requests": stats.successful_requests,
        "failed_requests": stats.failed_requests,
        "memories_stored": stats.total_memories_stored,
        "evolutions": stats.total_evolutions
    }))
}

async fn readiness_check() -> StatusCode {
    StatusCode::OK
}

async fn create_agent(
    State(orchestrator): State<SharedOrchestrator>,
    Json(payload): Json<serde_json::Value>
) -> impl IntoResponse {
    let name = payload.get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("Unknown Agent")
        .to_string();

    let role = AgentRole {
        id: uuid::Uuid::new_v4().to_string(),
        name: name.clone(),
        system_prompt: "Default agent system prompt".to_string(),
        version: 1,
        capabilities: vec![],
        constraints: vec![],
        evolution_lineage: vec![],
        created_at: chrono::Utc::now(),
        last_updated: chrono::Utc::now(),
    };

    let mut orch = orchestrator.write().await;
    match orch.create_agent(name, role).await {
        Ok(agent) => (StatusCode::CREATED, Json(json!({
            "id": agent.id,
            "name": agent.name,
            "created_at": agent.created_at.to_rfc3339()
        }))),
        Err(_) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({
            "error": "Failed to create agent"
        })))
    }
}

async fn list_agents(State(orchestrator): State<SharedOrchestrator>) -> Json<serde_json::Value> {
    let orch = orchestrator.read().await;
    match orch.list_agents().await {
        Ok(agents) => Json(json!({
            "agents": agents.iter().map(|a| json!({
                "id": a.id,
                "name": a.name,
                "active": a.is_active,
                "created_at": a.created_at.to_rfc3339()
            })).collect::<Vec<_>>()
        })),
        Err(_) => Json(json!({ "agents": [] }))
    }
}

async fn get_agent(
    State(orchestrator): State<SharedOrchestrator>,
    Path(id): Path<String>
) -> impl IntoResponse {
    let manager_lock = {
        let orch = orchestrator.read().await;
        orch.agent_manager.clone()
    };
    
    let result = match manager_lock.read().await.get_agent(&id) {
        Ok(agent) => (StatusCode::OK, Json(json!({
            "id": agent.id,
            "name": agent.name,
            "active": agent.is_active,
            "created_at": agent.created_at.to_rfc3339(),
            "last_activity": agent.last_activity.to_rfc3339()
        }))).into_response(),
        Err(_) => (StatusCode::NOT_FOUND, Json(json!({
            "error": "Agent not found"
        }))).into_response()
    };
    result
}

async fn delete_agent(
    State(orchestrator): State<SharedOrchestrator>,
    Path(id): Path<String>
) -> StatusCode {
    let orch = orchestrator.write().await;
    match orch.delete_agent(&id).await {
        Ok(_) => StatusCode::NO_CONTENT,
        Err(_) => StatusCode::NOT_FOUND
    }
}

async fn query_agent(
    State(orchestrator): State<SharedOrchestrator>,
    Path(id): Path<String>,
    Json(payload): Json<serde_json::Value>
) -> impl IntoResponse {
    let query = payload.get("query")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let request = AgentRequest {
        id: uuid::Uuid::new_v4().to_string(),
        agent_id: id.clone(),
        query: query.clone(),
        context: Default::default(),
        priority: 5,
        timeout_ms: 5000,
        metadata: Default::default(),
    };

    let orch = orchestrator.read().await;
    match orch.execute_request(request).await {
        Ok(response) => (StatusCode::OK, Json(json!({
            "result": response.result,
            "success": response.success,
            "execution_time_ms": response.execution_time_ms
        }))),
        Err(_) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({
            "error": "Query execution failed"
        })))
    }
}

async fn search_memory(
    Path(id): Path<String>,
    Json(payload): Json<serde_json::Value>
) -> Json<serde_json::Value> {
    let query = payload.get("query")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    Json(json!({ "agent_id": id, "query": query, "results": [] }))
}

async fn recall_memory(Path(id): Path<String>) -> Json<serde_json::Value> {
    Json(json!({ "agent_id": id, "memories": [] }))
}

async fn export_memory(Path(id): Path<String>) -> Json<serde_json::Value> {
    Json(json!({ "agent_id": id, "format": "json", "export_url": format!("/exports/memory_{}.json", id) }))
}

async fn evolve_agent(
    State(_): State<SharedOrchestrator>,
    Path(id): Path<String>
) -> Json<serde_json::Value> {
    tracing::info!("Evolution triggered for agent: {}", id);
    Json(json!({ "agent_id": id, "status": "evolution_started", "timestamp": chrono::Utc::now().to_rfc3339() }))
}

async fn evolution_history(Path(id): Path<String>) -> Json<serde_json::Value> {
    Json(json!({ "agent_id": id, "entries": [] }))
}

async fn rollback_evolution(
    Path(id): Path<String>,
    Json(payload): Json<serde_json::Value>
) -> Json<serde_json::Value> {
    let steps = payload.get("steps").and_then(|v| v.as_u64()).unwrap_or(1);
    Json(json!({ "agent_id": id, "steps_rolled_back": steps, "timestamp": chrono::Utc::now().to_rfc3339() }))
}
