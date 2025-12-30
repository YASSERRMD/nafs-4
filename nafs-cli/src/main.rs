use clap::{Parser, Subcommand};
use nafs_core::*;
use nafs_orchestrator::NafsOrchestrator;
use std::collections::HashMap;

#[derive(Parser)]
#[command(name = "nafs")]
#[command(about = "NAFS-4 Agent Framework CLI", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Agent management commands
    Agent {
        #[command(subcommand)]
        action: AgentAction,
    },
    /// System commands
    System {
        #[command(subcommand)]
        action: SystemAction,
    },
}

#[derive(Subcommand)]
enum AgentAction {
    /// Create a new agent
    Create {
        /// Name of the agent
        #[arg(short, long)]
        name: String,
        /// Role name
        #[arg(short, long, default_value = "General Assistant")]
        role: String,
    },
    /// List all agents
    List,
    /// Send a request to an agent
    Query {
        /// Agent ID
        #[arg(short, long)]
        id: String,
        /// The query text
        #[arg(short, long)]
        query: String,
    },
    /// Delete an agent
    Delete {
        /// Agent ID
        #[arg(short, long)]
        id: String,
    },
}

#[derive(Subcommand)]
enum SystemAction {
    /// Check system health
    Health,
    /// View system stats
    Stats,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();
    
    // Initialize orchestrator (embedded for CLI tool)
    // In a real deployed scenario, this might connect to a daemon.
    let config = OrchestratorConfig {
        persistence_backend: "file".to_string(),
        persistence_path: "./nafs_data".to_string(),
        ..Default::default()
    };
    
    let orchestrator = NafsOrchestrator::new(config).await?;

    match cli.command {
        Commands::Agent { action } => match action {
            AgentAction::Create { name, role } => {
                let agent_role = AgentRole {
                    id: uuid::Uuid::new_v4().to_string(),
                    name: role,
                    system_prompt: "You are a helpful NAFS-4 agent.".to_string(),
                    version: 1,
                    capabilities: vec![],
                    constraints: vec![],
                    evolution_lineage: vec![],
                    created_at: chrono::Utc::now(),
                    last_updated: chrono::Utc::now(),
                };
                
                let agent = orchestrator.create_agent(name, agent_role).await?;
                println!("Agent created successfully!");
                println!("ID: {}", agent.id);
                println!("Name: {}", agent.name);
            }
            AgentAction::List => {
                let agents = orchestrator.list_agents().await?;
                if agents.is_empty() {
                    println!("No agents found.");
                } else {
                    println!("Found {} agents:", agents.len());
                    for agent in agents {
                        println!("- {} ({}) [Active: {}]", agent.name, agent.id, agent.is_active);
                    }
                }
            }
            AgentAction::Query { id, query } => {
                let request = AgentRequest {
                    id: uuid::Uuid::new_v4().to_string(),
                    agent_id: id,
                    query,
                    context: HashMap::new(),
                    priority: 1,
                    timeout_ms: 5000,
                    metadata: HashMap::new(),
                };
                
                println!("Sending query to agent...");
                match orchestrator.execute_request(request).await {
                    Ok(response) => {
                        println!("Response received:");
                        println!("{}", response.result);
                    }
                    Err(e) => {
                        eprintln!("Error executing request: {}", e);
                    }
                }
            }
            AgentAction::Delete { id } => {
                orchestrator.delete_agent(&id).await?;
                println!("Agent {} deleted.", id);
            }
        },
        Commands::System { action } => match action {
            SystemAction::Health => {
                let health = orchestrator.health_check().await;
                println!("System Health Status:");
                println!("  Healthy: {}", health.is_healthy);
                println!("  Active Agents: {}", health.active_agents);
                println!("  Uptime: {}s", health.uptime_seconds);
            }
            SystemAction::Stats => {
                println!("System statistics not fully implemented yet.");
            }
        },
    }

    Ok(())
}
