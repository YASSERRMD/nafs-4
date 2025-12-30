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
             Ok(resp.result)
        })
    }
    
    fn health<'a>(&self, py: Python<'a>) -> PyResult<&'a PyAny> {
        let orch = self.inner.clone();
        pyo3_asyncio::tokio::future_into_py(py, async move {
            let status = orch.health_check().await;
            Ok(status.is_healthy)
        })
    }
}

#[pymodule]
fn nafs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Orchestrator>()?;
    Ok(())
}
