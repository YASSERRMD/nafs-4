use clap::Subcommand;

#[derive(Subcommand)]
pub enum Commands {
    /// Agent management
    Agent {
        #[command(subcommand)]
        action: AgentCommands,
    },
    /// Memory operations
    Memory {
        #[command(subcommand)]
        action: MemoryCommands,
    },
    /// Evolution control
    Evolution {
        #[command(subcommand)]
        action: EvolutionCommands,
    },
    /// System operations
    System {
        #[command(subcommand)]
        action: SystemCommands,
    },
}

#[derive(Subcommand)]
pub enum AgentCommands {
    /// Create new agent
    Create {
        /// Agent name
        #[arg(short, long)]
        name: String,
        /// Optional role template
        #[arg(short, long)]
        role: Option<String>,
    },
    /// List all agents
    List {
        /// Filter by status (active/inactive)
        #[arg(short, long)]
        status: Option<String>,
    },
    /// Query an agent
    Query {
        /// Agent ID
        #[arg(short, long)]
        agent_id: String,
        /// Query text
        #[arg(short, long)]
        query: String,
    },
    /// Get agent details
    Info {
        /// Agent ID
        #[arg(short, long)]
        agent_id: String,
    },
    /// Delete agent
    Delete {
        /// Agent ID
        #[arg(short, long)]
        agent_id: String,
        /// Skip confirmation
        #[arg(short, long)]
        force: bool,
    },
}

#[derive(Subcommand)]
pub enum MemoryCommands {
    /// Search memory
    Search {
        /// Query string
        #[arg(short, long)]
        query: String,
        /// Agent ID (optional)
        #[arg(short, long)]
        agent: Option<String>,
    },
    /// Recall recent memories
    Recall {
        /// Agent ID
        #[arg(short, long)]
        agent_id: String,
        /// Number of memories
        #[arg(short, long, default_value = "10")]
        limit: usize,
    },
    /// Export memory
    Export {
        /// Agent ID
        #[arg(short, long)]
        agent_id: String,
        /// Output format (json/csv)
        #[arg(short, long, default_value = "json")]
        format: String,
    },
    /// Clear memory
    Clear {
        /// Agent ID
        #[arg(short, long)]
        agent_id: String,
        /// Skip confirmation
        #[arg(short, long)]
        force: bool,
    },
}

#[derive(Subcommand)]
pub enum EvolutionCommands {
    /// Trigger evolution cycle
    Evolve {
        /// Agent ID
        #[arg(short, long)]
        agent_id: String,
    },
    /// Show evolution history
    History {
        /// Agent ID
        #[arg(short, long)]
        agent_id: String,
        /// Number of recent entries
        #[arg(short, long, default_value = "20")]
        limit: usize,
    },
    /// Rollback evolution
    Rollback {
        /// Agent ID
        #[arg(short, long)]
        agent_id: String,
        /// Number of steps to rollback
        #[arg(short, long, default_value = "1")]
        steps: usize,
    },
    /// Review pending changes
    Review {
        /// Agent ID
        #[arg(short, long)]
        agent_id: String,
    },
    /// Approve pending changes
    Approve {
        /// Agent ID
        #[arg(short, long)]
        agent_id: String,
        /// Change ID
        #[arg(short, long)]
        change_id: String,
    },
}

#[derive(Subcommand)]
pub enum SystemCommands {
    /// Check system health
    Health,
    /// Show system statistics
    Stats,
    /// Show configuration
    Config,
    /// Start interactive REPL
    Repl,
}
