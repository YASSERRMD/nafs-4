//! NAFS-4 Command-Line Interface
//!
//! Entry point for the `nafs` CLI tool.

use clap::{Parser, Subcommand};
use nafs_cli::CliConfig;
use nafs_core::Agent;
use nafs_logging::init_logging;

#[derive(Parser)]
#[command(name = "nafs")]
#[command(author = "NAFS Team")]
#[command(version = nafs_core::NAFS_VERSION)]
#[command(about = "NAFS-4: Self-Evolving Neuro-Symbolic Agent Framework")]
struct Cli {
    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Create a new agent
    New {
        /// Name of the agent
        #[arg(short, long, default_value = "NAFS-Agent")]
        name: String,
    },

    /// Run an agent interactively
    Run {
        /// Agent name
        #[arg(short, long, default_value = "NAFS-Agent")]
        name: String,
    },

    /// Show version information
    Version,

    /// Show system status
    Status,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    if cli.verbose {
        std::env::set_var("RUST_LOG", "debug");
    }
    init_logging();

    match cli.command {
        Commands::New { name } => {
            let agent = Agent::new(&name);
            println!("✅ Created agent: {}", agent.name());
            println!("   ID: {}", agent.id);
            println!("   Identity: {}", agent.identity());
        }

        Commands::Run { name } => {
            println!("🚀 Starting agent: {}", name);
            println!("   (Interactive mode not yet implemented)");
        }

        Commands::Version => {
            println!("{}", nafs_core::version_info());
        }

        Commands::Status => {
            println!("📊 NAFS-4 Status");
            println!("   Version: {}", nafs_core::NAFS_VERSION);
            println!("   Codename: {}", nafs_core::NAFS_CODENAME);
            println!("   Status: Ready");
        }
    }

    Ok(())
}
