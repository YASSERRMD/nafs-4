use axum::{
    extract::{Path, State},
    http::StatusCode,
    routing::{get, post, delete},
    Json, Router,
};
use nafs_core::*;
use nafs_orchestrator::NafsOrchestrator;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::collections::HashMap;
use tower_http::trace::TraceLayer;

struct AppState {
    orchestrator: Arc<NafsOrchestrator>,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    // Initialize orchestrator
    let config = OrchestratorConfig {
        persistence_backend: "file".to_string(),
        persistence_path: "./nafs_data".to_string(),
        ..Default::default()
    };
    
    let orchestrator = Arc::new(NafsOrchestrator::new(config).await?);
    let state = Arc::new(AppState { orchestrator });

    // Build router
    let app = Router::new()
        .route("/health", get(health_check))
        .route("/agents", get(list_agents).post(create_agent))
        .route("/agents/:id", delete(delete_agent))
        .route("/agents/:id/query", post(query_agent))
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.map_err(|e| NafsError::IoError(e))?;
    tracing::info!("NAFS-4 API Server listening on {}", listener.local_addr().map_err(|e| NafsError::IoError(e))?);
    
    axum::serve(listener, app).await.map_err(|e| NafsError::IoError(e))?;

    Ok(())
}

// Handler DTOs
#[derive(Deserialize)]
struct CreateAgentRequest {
    name: String,
    role: String,
}

#[derive(Deserialize)]
struct QueryAgentRequest {
    query: String,
}

// Handlers

async fn health_check(
    State(state): State<Arc<AppState>>,
) -> Json<HealthStatus> {
    let health = state.orchestrator.health_check().await;
    Json(health)
}

async fn list_agents(
    State(state): State<Arc<AppState>>,
) -> std::result::Result<Json<Vec<AgentInstance>>, StatusCode> {
    let agents = state.orchestrator.list_agents().await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(agents))
}

async fn create_agent(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<CreateAgentRequest>,
) -> std::result::Result<Json<AgentInstance>, StatusCode> {
    let role = AgentRole {
        id: uuid::Uuid::new_v4().to_string(),
        name: payload.role,
        system_prompt: "Standard agent prompt".to_string(),
        version: 1,
        capabilities: vec![],
        constraints: vec![],
        evolution_lineage: vec![],
        created_at: chrono::Utc::now(),
        last_updated: chrono::Utc::now(),
    };

    let agent = state.orchestrator.create_agent(payload.name, role).await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    Ok(Json(agent))
}

async fn delete_agent(
    Path(id): Path<String>,
    State(state): State<Arc<AppState>>,
) -> StatusCode {
    match state.orchestrator.delete_agent(&id).await {
        Ok(_) => StatusCode::NO_CONTENT,
        Err(_) => StatusCode::NOT_FOUND,
    }
}

async fn query_agent(
    Path(id): Path<String>,
    State(state): State<Arc<AppState>>,
    Json(payload): Json<QueryAgentRequest>,
) -> std::result::Result<Json<AgentResponse>, StatusCode> {
    let request = AgentRequest {
        id: uuid::Uuid::new_v4().to_string(),
        agent_id: id,
        query: payload.query,
        context: HashMap::new(),
        priority: 1,
        timeout_ms: 5000,
        metadata: HashMap::new(),
    };

    let response = state.orchestrator.execute_request(request).await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        
    Ok(Json(response))
}
