//! NAFS-4 Server Library
//!
//! Provides REST API functionality for the framework.

use axum::{routing::get, Router};
use std::net::SocketAddr;

/// Server configuration
#[derive(Clone, Debug)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8080,
        }
    }
}

impl ServerConfig {
    /// Get the socket address
    pub fn addr(&self) -> SocketAddr {
        format!("{}:{}", self.host, self.port)
            .parse()
            .expect("Invalid address")
    }
}

/// Create the application router
pub fn create_router() -> Router {
    Router::new()
        .route("/", get(root_handler))
        .route("/health", get(health_handler))
        .route("/version", get(version_handler))
}

async fn root_handler() -> &'static str {
    "NAFS-4 API Server"
}

async fn health_handler() -> &'static str {
    "OK"
}

async fn version_handler() -> String {
    nafs_core::version_info()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_config_default() {
        let config = ServerConfig::default();
        assert_eq!(config.host, "127.0.0.1");
        assert_eq!(config.port, 8080);
    }

    #[test]
    fn test_server_config_addr() {
        let config = ServerConfig::default();
        let addr = config.addr();
        assert_eq!(addr.port(), 8080);
    }

    #[tokio::test]
    async fn test_root_handler() {
        let response = root_handler().await;
        assert!(response.contains("NAFS-4"));
    }

    #[tokio::test]
    async fn test_health_handler() {
        let response = health_handler().await;
        assert_eq!(response, "OK");
    }
}
