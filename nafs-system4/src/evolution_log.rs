//! Evolution Log
//!
//! Maintains immutable audit trail of all self-modifications.

use nafs_core::{ApprovalStatus, EvolutionEntry, TextualGradient};
use std::collections::VecDeque;

/// Log for tracking system evolution
pub struct EvolutionLog {
    entries: VecDeque<EvolutionEntry>,
    max_size: usize,
}

impl EvolutionLog {
    /// Create new evolution log
    pub fn new() -> Self {
        Self {
            entries: VecDeque::new(),
            max_size: 10000,
        }
    }

    /// Record an evolution entry
    pub fn record(&mut self, entry: EvolutionEntry) {
        self.entries.push_back(entry);
        if self.entries.len() > self.max_size {
            self.entries.pop_front();
        }
        tracing::info!(
            "Evolution recorded. Total entries: {}",
            self.entries.len()
        );
    }

    /// Get all entries
    pub fn entries(&self) -> Vec<EvolutionEntry> {
        self.entries.iter().cloned().collect()
    }

    /// Get recent entries
    pub fn get_recent(&self, n: usize) -> Vec<EvolutionEntry> {
        self.entries.iter().rev().take(n).cloned().collect()
    }

    /// Get approved entries
    pub fn get_approved(&self) -> Vec<EvolutionEntry> {
        self.entries
            .iter()
            .filter(|e| e.approval_status == ApprovalStatus::Approved)
            .cloned()
            .collect()
    }

    /// Calculate total performance improvement
    pub fn total_performance_delta(&self) -> f32 {
        self.entries
            .iter()
            .map(|e| e.performance_delta_after - e.performance_delta_before)
            .sum()
    }

    /// Get log size
    pub fn size(&self) -> usize {
        self.entries.len()
    }
}

impl Default for EvolutionLog {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evolution_logging() {
        let mut log = EvolutionLog::new();

        let entry = EvolutionEntry {
            id: "e1".to_string(),
            timestamp: chrono::Utc::now(),
            gradient: TextualGradient {
                id: "g1".to_string(),
                failed_action: "test".to_string(),
                root_cause: "test".to_string(),
                suggested_fix: "test".to_string(),
                target_module: "test".to_string(),
                target_field: "test".to_string(),
                confidence: 0.8,
                impact_estimate: 0.3,
            },
            approval_status: ApprovalStatus::Approved,
            approved_by: Some("test".to_string()),
            applied_changes: "test".to_string(),
            performance_delta_before: 0.5,
            performance_delta_after: 0.65,
            rollback_available: true,
        };

        log.record(entry);
        assert_eq!(log.size(), 1);
    }
}
