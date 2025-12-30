//! NAFS-4 Observability
//!
//! Provides logging, tracing, and metrics infrastructure:
//! - Structured logging with tracing
//! - Metric collection
//! - Span management for distributed tracing

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;
use tracing_subscriber::prelude::*;

/// Initialize the logging subsystem
pub fn init_logging() {
    let subscriber = tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer().with_target(true))
        .with(tracing_subscriber::EnvFilter::from_default_env());

    tracing::subscriber::set_global_default(subscriber).ok();
}

/// Initialize logging with custom format
pub fn init_logging_with_format(format: LogFormat) {
    match format {
        LogFormat::Pretty => {
            let subscriber = tracing_subscriber::registry()
                .with(tracing_subscriber::fmt::layer().pretty())
                .with(tracing_subscriber::EnvFilter::from_default_env());
            tracing::subscriber::set_global_default(subscriber).ok();
        }
        LogFormat::Json => {
            let subscriber = tracing_subscriber::registry()
                .with(tracing_subscriber::fmt::layer().json())
                .with(tracing_subscriber::EnvFilter::from_default_env());
            tracing::subscriber::set_global_default(subscriber).ok();
        }
        LogFormat::Compact => {
            let subscriber = tracing_subscriber::registry()
                .with(tracing_subscriber::fmt::layer().compact())
                .with(tracing_subscriber::EnvFilter::from_default_env());
            tracing::subscriber::set_global_default(subscriber).ok();
        }
    }
}

/// Log format options
#[derive(Clone, Copy, Debug)]
pub enum LogFormat {
    Pretty,
    Json,
    Compact,
}

/// A metric value
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum MetricValue {
    Counter(u64),
    Gauge(f64),
    Histogram(Vec<f64>),
}

/// Metrics collector for the framework
pub struct MetricsCollector {
    counters: RwLock<HashMap<String, AtomicU64>>,
    gauges: RwLock<HashMap<String, f64>>,
    histograms: RwLock<HashMap<String, Vec<f64>>>,
}

impl MetricsCollector {
    /// Create a new metrics collector
    pub fn new() -> Self {
        Self {
            counters: RwLock::new(HashMap::new()),
            gauges: RwLock::new(HashMap::new()),
            histograms: RwLock::new(HashMap::new()),
        }
    }

    /// Increment a counter
    pub fn increment(&self, name: &str) {
        self.increment_by(name, 1);
    }

    /// Increment a counter by a specific amount
    pub fn increment_by(&self, name: &str, amount: u64) {
        let counters = self.counters.read().unwrap();
        if let Some(counter) = counters.get(name) {
            counter.fetch_add(amount, Ordering::Relaxed);
        } else {
            drop(counters);
            let mut counters = self.counters.write().unwrap();
            counters
                .entry(name.to_string())
                .or_insert_with(|| AtomicU64::new(0))
                .fetch_add(amount, Ordering::Relaxed);
        }
    }

    /// Set a gauge value
    pub fn set_gauge(&self, name: &str, value: f64) {
        let mut gauges = self.gauges.write().unwrap();
        gauges.insert(name.to_string(), value);
    }

    /// Record a histogram value
    pub fn record_histogram(&self, name: &str, value: f64) {
        let mut histograms = self.histograms.write().unwrap();
        histograms
            .entry(name.to_string())
            .or_insert_with(Vec::new)
            .push(value);
    }

    /// Get counter value
    pub fn get_counter(&self, name: &str) -> u64 {
        let counters = self.counters.read().unwrap();
        counters
            .get(name)
            .map(|c| c.load(Ordering::Relaxed))
            .unwrap_or(0)
    }

    /// Get gauge value
    pub fn get_gauge(&self, name: &str) -> Option<f64> {
        let gauges = self.gauges.read().unwrap();
        gauges.get(name).copied()
    }

    /// Get histogram values
    pub fn get_histogram(&self, name: &str) -> Vec<f64> {
        let histograms = self.histograms.read().unwrap();
        histograms.get(name).cloned().unwrap_or_default()
    }

    /// Get all metrics as a snapshot
    pub fn snapshot(&self) -> MetricsSnapshot {
        let counters = self.counters.read().unwrap();
        let gauges = self.gauges.read().unwrap();
        let histograms = self.histograms.read().unwrap();

        MetricsSnapshot {
            timestamp: Utc::now(),
            counters: counters
                .iter()
                .map(|(k, v)| (k.clone(), v.load(Ordering::Relaxed)))
                .collect(),
            gauges: gauges.clone(),
            histograms: histograms.clone(),
        }
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// A snapshot of metrics at a point in time
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    pub timestamp: DateTime<Utc>,
    pub counters: HashMap<String, u64>,
    pub gauges: HashMap<String, f64>,
    pub histograms: HashMap<String, Vec<f64>>,
}

/// Event types for structured logging
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum EventType {
    AgentCreated,
    GoalSet,
    ActionStarted,
    ActionCompleted,
    ActionFailed,
    EvolutionTriggered,
    EvolutionApplied,
    EvolutionBlocked,
    MemoryStored,
    MemoryRecalled,
    Error,
}

/// A structured log event
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LogEvent {
    pub timestamp: DateTime<Utc>,
    pub event_type: EventType,
    pub agent_id: Option<String>,
    pub message: String,
    pub metadata: HashMap<String, serde_json::Value>,
}

impl LogEvent {
    /// Create a new log event
    pub fn new(event_type: EventType, message: impl Into<String>) -> Self {
        Self {
            timestamp: Utc::now(),
            event_type,
            agent_id: None,
            message: message.into(),
            metadata: HashMap::new(),
        }
    }

    /// Set agent ID
    pub fn with_agent(mut self, agent_id: impl Into<String>) -> Self {
        self.agent_id = Some(agent_id.into());
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Emit the event using tracing
    pub fn emit(&self) {
        match self.event_type {
            EventType::Error | EventType::ActionFailed | EventType::EvolutionBlocked => {
                tracing::error!(
                    event_type = ?self.event_type,
                    agent_id = ?self.agent_id,
                    message = %self.message,
                    "NAFS Event"
                );
            }
            _ => {
                tracing::info!(
                    event_type = ?self.event_type,
                    agent_id = ?self.agent_id,
                    message = %self.message,
                    "NAFS Event"
                );
            }
        }
    }
}

/// Convenience macros for common metrics
#[macro_export]
macro_rules! count {
    ($collector:expr, $name:expr) => {
        $collector.increment($name)
    };
    ($collector:expr, $name:expr, $amount:expr) => {
        $collector.increment_by($name, $amount)
    };
}

#[macro_export]
macro_rules! gauge {
    ($collector:expr, $name:expr, $value:expr) => {
        $collector.set_gauge($name, $value)
    };
}

#[macro_export]
macro_rules! histogram {
    ($collector:expr, $name:expr, $value:expr) => {
        $collector.record_histogram($name, $value)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_counter() {
        let collector = MetricsCollector::new();
        collector.increment("requests");
        collector.increment("requests");
        assert_eq!(collector.get_counter("requests"), 2);
    }

    #[test]
    fn test_metrics_gauge() {
        let collector = MetricsCollector::new();
        collector.set_gauge("cpu_usage", 0.75);
        assert_eq!(collector.get_gauge("cpu_usage"), Some(0.75));
    }

    #[test]
    fn test_metrics_histogram() {
        let collector = MetricsCollector::new();
        collector.record_histogram("latency", 100.0);
        collector.record_histogram("latency", 150.0);
        let values = collector.get_histogram("latency");
        assert_eq!(values.len(), 2);
    }

    #[test]
    fn test_metrics_snapshot() {
        let collector = MetricsCollector::new();
        collector.increment("test");
        collector.set_gauge("temp", 25.0);
        
        let snapshot = collector.snapshot();
        assert_eq!(snapshot.counters.get("test"), Some(&1));
        assert_eq!(snapshot.gauges.get("temp"), Some(&25.0));
    }

    #[test]
    fn test_log_event() {
        let event = LogEvent::new(EventType::AgentCreated, "Agent initialized")
            .with_agent("agent-123")
            .with_metadata("version", serde_json::json!("0.1.0"));
        
        assert_eq!(event.agent_id, Some("agent-123".to_string()));
        assert!(event.metadata.contains_key("version"));
    }
}
