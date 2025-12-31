use pyo3::prelude::*;
use nafs_orchestrator::NafsOrchestrator;
use nafs_core::{OrchestratorConfig, AgentRole, AgentRequest};
use std::sync::Arc;
use std::collections::HashMap;

#[pyclass]
#[derive(Clone)]
struct Orchestrator {
    inner: Arc<NafsOrchestrator>,
}

#[pyclass]
#[derive(Clone)]
struct AgentResponse {
    #[pyo3(get)]
    pub result: String,
    #[pyo3(get)]
    pub metadata: HashMap<String, String>,
    #[pyo3(get)]
    pub execution_time_ms: u32,
}

#[pymethods]
impl Orchestrator {
    #[staticmethod]
    fn create(py: Python<'_>) -> PyResult<&PyAny> {
        pyo3_asyncio::tokio::future_into_py(py, async {
            let config = OrchestratorConfig::default();
            let orchestrator = NafsOrchestrator::new(config).await
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok(Orchestrator { inner: Arc::new(orchestrator) })
        })
    }

    fn create_agent<'a>(&self, py: Python<'a>, name: String, role: String) -> PyResult<&'a PyAny> {
        let orch = self.inner.clone();
        pyo3_asyncio::tokio::future_into_py(py, async move {
            let role_struct = AgentRole {
                 id: uuid::Uuid::new_v4().to_string(),
                 name: role.clone(),
                 system_prompt: format!("You are a {}", role),
                 version: 1,
                 capabilities: vec![],
                 constraints: vec![],
                 evolution_lineage: vec![],
                 created_at: chrono::Utc::now(),
                 last_updated: chrono::Utc::now(),
            };
            let agent = orch.create_agent(name, role_struct).await
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok(agent.id)
        })
    }
    
    fn query<'a>(&self, py: Python<'a>, agent_id: String, query: String) -> PyResult<&'a PyAny> {
        let orch = self.inner.clone();
        pyo3_asyncio::tokio::future_into_py(py, async move {
             let req = AgentRequest {
                id: uuid::Uuid::new_v4().to_string(),
                agent_id,
                query,
                context: HashMap::new(),
                priority: 1,
                timeout_ms: 30000,
                metadata: HashMap::new(),
             };
             let resp = orch.execute_request(req).await
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
             
             Ok(AgentResponse {
                 result: resp.result,
                 metadata: resp.metadata,
                 execution_time_ms: resp.execution_time_ms,
             })
        })
    }
    
    fn health<'a>(&self, py: Python<'a>) -> PyResult<&'a PyAny> {
        let orch = self.inner.clone();
        pyo3_asyncio::tokio::future_into_py(py, async move {
            let status = orch.health_check().await;
            Ok(status.is_healthy)
        })
    }

    /// Generate embeddings using the default model for the configured provider
    fn embed<'a>(&self, py: Python<'a>, text: String) -> PyResult<&'a PyAny> {
        let orch = self.inner.clone();
        pyo3_asyncio::tokio::future_into_py(py, async move {
            let embedding = orch.llm_provider.embed(&text).await
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok(embedding)
        })
    }

    /// Generate embeddings using a specific model
    #[pyo3(signature = (text, model))]
    fn embed_with_model<'a>(&self, py: Python<'a>, text: String, model: String) -> PyResult<&'a PyAny> {
        let orch = self.inner.clone();
        pyo3_asyncio::tokio::future_into_py(py, async move {
            let embedding = orch.llm_provider.embed_with_model(&text, &model).await
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok(embedding)
        })
    }

    /// Get the name of the current LLM provider
    fn get_provider_name(&self) -> String {
        self.inner.llm_provider.name().to_string()
    }

    /// Get available embedding models for the current provider
    fn get_embedding_models(&self) -> Vec<String> {
        self.inner.llm_provider.available_embedding_models()
    }
}

#[pymodule]
fn nafs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Orchestrator>()?;
    m.add_class::<AgentResponse>()?;
    Ok(())
}
