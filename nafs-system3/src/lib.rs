//! NAFS-4 System 3: Meta-Cognition
//!
//! System 3 handles self-awareness and monitoring:
//! - Memory management (episodic/semantic)
//! - Self-model maintenance
//! - Executive monitoring
//! - Attention allocation
//!
//! This system provides the agent with awareness of its own state.

use async_trait::async_trait;
use nafs_core::{MemoryCategory, MemoryItem, Result, SelfModel, State};
use std::collections::HashMap;

/// Memory module for storing and retrieving memories
pub struct MemoryModule {
    /// In-memory storage (will be replaced with vector DB)
    memories: Vec<MemoryItem>,
    /// Maximum number of memories to retain
    capacity: usize,
}

impl MemoryModule {
    /// Create a new memory module with given capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            memories: Vec::new(),
            capacity,
        }
    }

    /// Store a memory item
    pub fn store(&mut self, item: MemoryItem) {
        self.memories.push(item);
        // Evict oldest if over capacity
        if self.memories.len() > self.capacity {
            self.memories.remove(0);
        }
    }

    /// Recall memories by query (simple substring match for now)
    pub fn recall(&self, query: &str, limit: usize) -> Vec<&MemoryItem> {
        self.memories
            .iter()
            .filter(|m| m.content.contains(query))
            .take(limit)
            .collect()
    }

    /// Recall memories by category
    pub fn recall_by_category(&self, category: MemoryCategory, limit: usize) -> Vec<&MemoryItem> {
        self.memories
            .iter()
            .filter(|m| m.category == category)
            .take(limit)
            .collect()
    }

    /// Get total memory count
    pub fn count(&self) -> usize {
        self.memories.len()
    }

    /// Clear all memories
    pub fn clear(&mut self) {
        self.memories.clear();
    }
}

impl Default for MemoryModule {
    fn default() -> Self {
        Self::new(10000)
    }
}

/// Executive monitor for controlling attention and resources
pub struct ExecutiveMonitor {
    /// Current attention focus
    attention_focus: Option<String>,
    /// Resource allocations
    allocations: HashMap<String, f32>,
    /// Interrupt threshold
    interrupt_threshold: f32,
}

impl ExecutiveMonitor {
    /// Create a new executive monitor
    pub fn new() -> Self {
        Self {
            attention_focus: None,
            allocations: HashMap::new(),
            interrupt_threshold: 0.8,
        }
    }

    /// Set attention focus
    pub fn focus_on(&mut self, target: impl Into<String>) {
        self.attention_focus = Some(target.into());
    }

    /// Get current focus
    pub fn current_focus(&self) -> Option<&str> {
        self.attention_focus.as_deref()
    }

    /// Allocate resources to a task
    pub fn allocate(&mut self, task: impl Into<String>, amount: f32) {
        self.allocations.insert(task.into(), amount.clamp(0.0, 1.0));
    }

    /// Check if an interrupt should be allowed
    pub fn should_interrupt(&self, priority: f32) -> bool {
        priority >= self.interrupt_threshold
    }

    /// Set interrupt threshold
    pub fn set_interrupt_threshold(&mut self, threshold: f32) {
        self.interrupt_threshold = threshold.clamp(0.0, 1.0);
    }

    /// Get total allocated resources
    pub fn total_allocated(&self) -> f32 {
        self.allocations.values().sum()
    }
}

impl Default for ExecutiveMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Self-model tracker for maintaining agent identity
pub struct SelfModelTracker {
    /// The current self-model
    model: SelfModel,
    /// Performance history
    performance_history: Vec<PerformanceRecord>,
}

/// A record of performance on a task
#[derive(Clone, Debug)]
pub struct PerformanceRecord {
    pub task_type: String,
    pub success: bool,
    pub duration_ms: u64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl SelfModelTracker {
    /// Create a new self-model tracker
    pub fn new(model: SelfModel) -> Self {
        Self {
            model,
            performance_history: Vec::new(),
        }
    }

    /// Get the current self-model
    pub fn model(&self) -> &SelfModel {
        &self.model
    }

    /// Record a performance outcome
    pub fn record_performance(&mut self, record: PerformanceRecord) {
        // Update capability based on outcome
        let current = self.model.capabilities.get(&record.task_type).copied().unwrap_or(0.5);
        let delta = if record.success { 0.01 } else { -0.01 };
        let new_value = (current + delta).clamp(0.0, 1.0);
        self.model.capabilities.insert(record.task_type.clone(), new_value);

        self.performance_history.push(record);

        // Keep history bounded
        if self.performance_history.len() > 1000 {
            self.performance_history.remove(0);
        }
    }

    /// Get success rate for a task type
    pub fn success_rate(&self, task_type: &str) -> f32 {
        let relevant: Vec<_> = self.performance_history
            .iter()
            .filter(|r| r.task_type == task_type)
            .collect();
        
        if relevant.is_empty() {
            return 0.5; // Default uncertainty
        }

        let successes = relevant.iter().filter(|r| r.success).count();
        successes as f32 / relevant.len() as f32
    }

    /// Predict success probability for a task
    pub fn predict_success(&self, task_type: &str) -> f32 {
        self.model.capabilities
            .get(task_type)
            .copied()
            .unwrap_or(0.5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nafs_core::MemoryCategory;

    #[test]
    fn test_memory_module() {
        let mut mem = MemoryModule::new(100);
        let item = MemoryItem::new("test content", MemoryCategory::Semantic);
        mem.store(item);
        assert_eq!(mem.count(), 1);
    }

    #[test]
    fn test_memory_recall() {
        let mut mem = MemoryModule::new(100);
        mem.store(MemoryItem::new("hello world", MemoryCategory::Semantic));
        mem.store(MemoryItem::new("goodbye world", MemoryCategory::Semantic));
        
        let results = mem.recall("hello", 10);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_executive_monitor() {
        let mut monitor = ExecutiveMonitor::new();
        monitor.focus_on("planning");
        assert_eq!(monitor.current_focus(), Some("planning"));
    }

    #[test]
    fn test_interrupt_threshold() {
        let monitor = ExecutiveMonitor::new();
        assert!(monitor.should_interrupt(0.9));
        assert!(!monitor.should_interrupt(0.5));
    }

    #[test]
    fn test_self_model_tracker() {
        let model = SelfModel::new("agent1", "I am helpful");
        let mut tracker = SelfModelTracker::new(model);
        
        tracker.record_performance(PerformanceRecord {
            task_type: "coding".to_string(),
            success: true,
            duration_ms: 1000,
            timestamp: chrono::Utc::now(),
        });

        assert!(tracker.predict_success("coding") > 0.5);
    }
}
