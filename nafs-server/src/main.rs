//! NAFS-4 REST API Server
//!
//! Entry point for the `nafs-server` binary.

use nafs_logging::init_logging;
use nafs_server::{create_router, ServerConfig};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    init_logging();

    let config = ServerConfig::default();
    let app = create_router();

    tracing::info!("Starting NAFS-4 API Server on {}", config.addr());
    println!("🚀 NAFS-4 API Server running at http://{}", config.addr());

    let listener = tokio::net::TcpListener::bind(config.addr()).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
