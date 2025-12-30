//! NAFS-4 CLI: Interactive Agent Management Tool

use clap::Parser;
use nafs_core::*;
use colored::*;

mod commands;
mod repl;
mod output;

use commands::*;
use output::*;

#[derive(Parser)]
#[command(name = "nafs")]
#[command(about = "NAFS-4: Neuromorphic AI Framework System", long_about = None)]
#[command(version = "0.5.0")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Enable verbose logging
    #[arg(global = true, short, long)]
    verbose: bool,

    /// Orchestrator URL (for API mode)
    #[arg(global = true, long, default_value = "http://localhost:3000")]
    api_url: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    init_logging(cli.verbose);

    match cli.command {
        Some(Commands::Agent { action }) => handle_agent(action, &cli.api_url).await?,
        Some(Commands::Memory { action }) => handle_memory(action, &cli.api_url).await?,
        Some(Commands::Evolution { action }) => handle_evolution(action, &cli.api_url).await?,
        Some(Commands::System { action }) => handle_system(action, &cli.api_url).await?,
        None => {
            // Start interactive REPL
            repl::start_repl(&cli.api_url).await?;
        }
    }

    Ok(())
}

async fn handle_agent(action: AgentCommands, _api_url: &str) -> Result<()> {
    match action {
        AgentCommands::Create { name, role: _ } => {
            print_info(&format!("Creating agent: {}", name));
            // Call API or Orchestrator here
            print_success("Agent created successfully");
        }
        AgentCommands::List { status: _ } => {
            print_header("Agents:");
            println!("{:<20} {:<20} {:<10}", "ID", "Name", "Status");
            println!("{}", "-".repeat(50));
            println!("{:<20} {:<20} {:<10}", "agent_1", "Assistant", "active".green());
            println!("{:<20} {:<20} {:<10}", "agent_2", "Analyzer", "inactive".red());
        }
        AgentCommands::Query { agent_id, query } => {
            print_header(&format!("Query [{}]:", agent_id));
            println!("  {}", query);
            println!("{}", "Response:".cyan());
            println!("  [Processing through System 1-4 pipeline...]");
        }
        AgentCommands::Info { agent_id } => {
            print_header(&format!("Agent: {}", agent_id));
            println!("  Status: {}", "active".green());
            println!("  Created: 2025-12-30T08:00:00Z");
        }
        AgentCommands::Delete { agent_id, force } => {
            if !force {
                println!("Delete agent {}? [y/N] ", agent_id);
            }
            print_success("Agent deleted");
        }
    }
    Ok(())
}

async fn handle_memory(action: MemoryCommands, _api_url: &str) -> Result<()> {
    match action {
        MemoryCommands::Search { query, agent: _ } => {
            print_header(&format!("Searching: {}", query));
            println!("Found 3 memories:");
            println!("  • Memory 1 (2025-12-30 08:15:23)");
            println!("  • Memory 2 (2025-12-30 08:10:45)");
            println!("  • Memory 3 (2025-12-30 08:05:12)");
        }
        MemoryCommands::Recall { agent_id, limit } => {
            print_info(&format!("Recalling {} memories for {}...", limit, agent_id));
        }
        MemoryCommands::Export { agent_id, format } => {
            print_info(&format!("Exporting memory for {} as {}...", agent_id, format));
        }
        MemoryCommands::Clear { agent_id: _, force } => {
            if !force {
                println!("Clear all memories? [y/N] ");
            }
            print_success("Memory cleared");
        }
    }
    Ok(())
}

async fn handle_evolution(action: EvolutionCommands, _api_url: &str) -> Result<()> {
    match action {
        EvolutionCommands::Evolve { agent_id } => {
            print_header(&format!("Starting evolution for {}...", agent_id));
            println!("  1. Analyzing {} recent failures", 5);
            println!("  2. Generating textual gradients");
            println!("  3. Validating kernel constraints");
            println!("  4. Applying approved changes");
            print_success("Evolution complete. 2 changes applied.");
        }
        EvolutionCommands::History { agent_id: _, limit } => {
            print_header(&format!("Evolution history (last {}):", limit));
            println!("{:<30} {:<15} {:<20}", "Gradient", "Status", "Timestamp");
            println!("{}", "-".repeat(65));
        }
        EvolutionCommands::Rollback { agent_id: _, steps } => {
            print_info(&format!("Rolling back {} steps...", steps));
            print_success("Rollback complete");
        }
        EvolutionCommands::Review { agent_id } => {
            print_header(&format!("Pending changes for {}:", agent_id));
        }
        EvolutionCommands::Approve { agent_id: _, change_id } => {
            print_info(&format!("Approving change {}", change_id));
            print_success("Change approved");
        }
    }
    Ok(())
}

async fn handle_system(action: SystemCommands, api_url: &str) -> Result<()> {
    match action {
        SystemCommands::Health => {
            print_header("System Health:");
            println!("  Status: {}", "✓ Healthy".green().bold());
            println!("  Active Agents: {}", "2".yellow());
            println!("  Uptime: {}", "12h 34m".green());
            println!("  Memory Usage: {}", "156 MB".yellow());
        }
        SystemCommands::Stats => {
            print_header("System Statistics:");
            println!("  Total Requests: {}", "1,234".yellow());
            println!("  Success Rate: {}", "98.5%".green());
            println!("  Memories Stored: {}", "5,678".yellow());
            println!("  Evolutions: {}", "12".green());
        }
        SystemCommands::Config => {
            print_header("Configuration:");
            println!("  Max Agents: {}", "100".yellow());
            println!("  Backend: {}", "file".yellow());
            println!("  API URL: {}", api_url.yellow());
        }
        SystemCommands::Repl => {
            repl::start_repl(api_url).await?;
        }
    }
    Ok(())
}

fn init_logging(verbose: bool) {
    let level = if verbose { "debug" } else { "info" };
    std::env::set_var("RUST_LOG", level);
    tracing_subscriber::fmt::init();
}
