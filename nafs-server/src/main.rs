use nafs_core::OrchestratorConfig;
use nafs_server::app;

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let config = OrchestratorConfig::default();
    let app = app(config).await?;

    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000").await?;
    println!("🚀 NAFS-4 API Server listening on http://127.0.0.1:3000");
    println!("   Health check: GET /health");
    println!("   API docs: http://127.0.0.1:3000/docs");

    axum::serve(listener, app).await?;

    Ok(())
}
