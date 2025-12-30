//! NAFS-4 System 1: Perception and Action
//!
//! System 1 handles fast, intuitive processing:
//! - Pattern matching
//! - Reactive behaviors
//! - Heuristic-based decisions
//!
//! This is the "fast thinking" system that handles routine tasks
//! without engaging the more expensive System 2 reasoning.

use nafs_core::{Action, Result, State};

/// System 1 processor for fast, intuitive responses
pub struct System1 {
    /// Heuristic rules for pattern matching
    heuristics: Vec<Heuristic>,
    /// Cached patterns for fast lookup
    pattern_cache: Vec<Pattern>,
}

/// A heuristic rule for quick decision making
#[derive(Clone, Debug)]
pub struct Heuristic {
    pub name: String,
    pub pattern: String,
    pub action_template: String,
    pub confidence: f32,
}

/// A pattern for matching against state
#[derive(Clone, Debug)]
pub struct Pattern {
    pub id: String,
    pub trigger: String,
    pub response: String,
}

impl System1 {
    /// Create a new System 1 processor
    pub fn new() -> Self {
        Self {
            heuristics: Vec::new(),
            pattern_cache: Vec::new(),
        }
    }

    /// Add a heuristic rule
    pub fn add_heuristic(&mut self, heuristic: Heuristic) {
        self.heuristics.push(heuristic);
    }

    /// Process state and return an action if pattern matches
    pub fn process(&self, _state: &State) -> Option<Action> {
        // TODO: Implement pattern matching logic
        None
    }

    /// Check if System 1 can handle this state (fast path)
    pub fn can_handle(&self, _state: &State) -> bool {
        // TODO: Implement pattern detection
        false
    }
}

impl Default for System1 {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system1_creation() {
        let system = System1::new();
        assert!(system.heuristics.is_empty());
    }

    #[test]
    fn test_add_heuristic() {
        let mut system = System1::new();
        system.add_heuristic(Heuristic {
            name: "greet".to_string(),
            pattern: "hello".to_string(),
            action_template: "respond_greeting".to_string(),
            confidence: 0.9,
        });
        assert_eq!(system.heuristics.len(), 1);
    }
}
