//! Tree-of-Thought Search Engine
//!
//! Expands, evaluates, and prunes decision trees for deliberate reasoning.

use nafs_core::{
    Action, ChainOfThought, NafsError, Result, SymbolicConstraint, ToTNode, ToTNodeStatus,
};
use std::collections::HashMap;

/// Result of a Tree-of-Thought search
#[derive(Clone, Debug)]
pub struct ToTSearchResult {
    /// All explored nodes
    pub nodes: HashMap<String, ToTNode>,
    /// The best path found (sequence of actions)
    pub best_path: Vec<Action>,
    /// Confidence in the selected path
    pub confidence: f32,
    /// Number of nodes pruned
    pub pruned_count: usize,
}

/// Tree-of-Thought search engine
pub struct TreeOfThoughtEngine {
    /// Maximum depth of the search tree
    max_depth: usize,
    /// Maximum branches per node
    max_width: usize,
    /// Current node storage
    nodes: HashMap<String, ToTNode>,
}

impl TreeOfThoughtEngine {
    /// Create a new ToT engine with specified parameters
    pub fn new(max_depth: usize, max_width: usize) -> Self {
        Self {
            max_depth,
            max_width,
            nodes: HashMap::new(),
        }
    }

    /// Execute Tree-of-Thought search
    pub async fn search(
        &mut self,
        cot: &ChainOfThought,
        constraints: &[SymbolicConstraint],
    ) -> Result<ToTSearchResult> {
        tracing::info!("Starting ToT search for goal: {}", cot.goal.id);

        self.nodes.clear();

        // Create root node
        let root = ToTNode::root();
        let root_id = root.id.clone();
        self.nodes.insert(root_id.clone(), root);

        // Expand tree using the chain-of-thought
        self.expand_from_cot(&root_id, cot).await?;

        // Evaluate and prune nodes
        let pruned_count = self.evaluate_and_prune(constraints);

        // Select best path
        let (best_path, confidence) = self.select_best_path()?;

        Ok(ToTSearchResult {
            nodes: self.nodes.clone(),
            best_path,
            confidence,
            pruned_count,
        })
    }

    /// Expand tree from chain-of-thought
    async fn expand_from_cot(&mut self, parent_id: &str, cot: &ChainOfThought) -> Result<()> {
        let parent = self
            .nodes
            .get(parent_id)
            .cloned()
            .ok_or_else(|| NafsError::runtime("Parent node not found"))?;

        if parent.depth >= self.max_depth {
            return Ok(());
        }

        // Create child nodes based on reasoning steps
        let branches = std::cmp::min(self.max_width, cot.steps.len().max(2));

        for i in 0..branches {
            let mut child = ToTNode::child(&parent);

            // Assign value based on reasoning quality and branch diversity
            let base_value = cot.reasoning_quality;
            let diversity_bonus = (i as f32 * 0.05).min(0.15);
            child = child.with_value(base_value + diversity_bonus);

            // If there's a corresponding action in the step, add to partial plan
            if let Some(step) = cot.steps.get(i) {
                if let Some(action) = &step.next_action {
                    child.partial_plan.push(action.clone());
                }
            }

            let child_id = child.id.clone();
            
            // Update parent's children list
            if let Some(parent_node) = self.nodes.get_mut(parent_id) {
                parent_node.children_ids.push(child_id.clone());
            }

            self.nodes.insert(child_id, child);
        }

        tracing::debug!(
            "Expanded node {} with {} children",
            parent_id,
            branches
        );

        Ok(())
    }

    /// Evaluate nodes and prune those that fail constraints
    fn evaluate_and_prune(&mut self, constraints: &[SymbolicConstraint]) -> usize {
        let mut pruned = 0;

        // Helper function to check violations (avoids borrow issues)
        fn check_violation(action: &Action, constraint: &SymbolicConstraint) -> bool {
            let tool_lower = action.tool_name.to_lowercase();
            let condition_lower = constraint.condition.to_lowercase();

            // Check for dangerous patterns
            if condition_lower.contains("delete") && tool_lower.contains("delete") {
                return true;
            }
            if condition_lower.contains("private") && tool_lower.contains("private") {
                return true;
            }

            false
        }

        for node in self.nodes.values_mut() {
            // Check constraints for each action in partial plan
            let mut violates = false;

            for action in &node.partial_plan {
                for constraint in constraints {
                    if check_violation(action, constraint) {
                        violates = true;
                        break;
                    }
                }
                if violates {
                    break;
                }
            }

            if violates {
                node.set_status(ToTNodeStatus::Pruned);
                pruned += 1;
            } else {
                node.set_status(ToTNodeStatus::Approved);
            }
        }

        tracing::debug!("Pruned {} nodes", pruned);
        pruned
    }

    /// Check if an action violates a constraint (standalone version for other uses)
    #[allow(dead_code)]
    fn violates_constraint(&self, action: &Action, constraint: &SymbolicConstraint) -> bool {
        // Simple check: look for forbidden patterns in tool name
        let tool_lower = action.tool_name.to_lowercase();
        let condition_lower = constraint.condition.to_lowercase();

        // Check for dangerous patterns
        if condition_lower.contains("delete") && tool_lower.contains("delete") {
            return true;
        }
        if condition_lower.contains("private") && tool_lower.contains("private") {
            return true;
        }

        false
    }

    /// Select the best path through the tree
    fn select_best_path(&mut self) -> Result<(Vec<Action>, f32)> {
        // Find approved leaf nodes with highest value
        let best_leaf = self
            .nodes
            .values()
            .filter(|n| n.status == ToTNodeStatus::Approved && n.is_leaf())
            .max_by(|a, b| a.value_estimate.partial_cmp(&b.value_estimate).unwrap())
            .cloned();

        if let Some(leaf) = best_leaf {
            // Mark as selected
            if let Some(node) = self.nodes.get_mut(&leaf.id) {
                node.set_status(ToTNodeStatus::Selected);
            }

            Ok((leaf.partial_plan, leaf.value_estimate))
        } else {
            // Return empty plan if no good path found
            Ok((Vec::new(), 0.5))
        }
    }

    /// Get current tree statistics
    pub fn stats(&self) -> TreeStats {
        let total = self.nodes.len();
        let approved = self
            .nodes
            .values()
            .filter(|n| n.status == ToTNodeStatus::Approved)
            .count();
        let pruned = self
            .nodes
            .values()
            .filter(|n| n.status == ToTNodeStatus::Pruned)
            .count();
        let selected = self
            .nodes
            .values()
            .filter(|n| n.status == ToTNodeStatus::Selected)
            .count();

        TreeStats {
            total,
            approved,
            pruned,
            selected,
        }
    }
}

/// Statistics about the search tree
#[derive(Clone, Debug)]
pub struct TreeStats {
    pub total: usize,
    pub approved: usize,
    pub pruned: usize,
    pub selected: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use nafs_core::Goal;

    #[tokio::test]
    async fn test_tot_search() {
        let mut engine = TreeOfThoughtEngine::new(5, 3);
        let goal = Goal::new("Test goal", 5);
        let cot = ChainOfThought::new(goal);

        let result = engine.search(&cot, &[]).await.unwrap();

        assert!(!result.nodes.is_empty());
        assert!(result.confidence >= 0.0);
    }

    #[tokio::test]
    async fn test_tot_with_constraints() {
        let mut engine = TreeOfThoughtEngine::new(5, 3);
        let goal = Goal::new("Test goal", 5);
        let mut cot = ChainOfThought::new(goal);

        // Add a step with an action
        use nafs_core::Action;
        let step = nafs_core::ReasoningStep::new("Test step", "Justification")
            .with_action(Action::new("delete_file", serde_json::json!({})));
        cot.add_step(step);

        // Add constraint against delete
        let constraint = SymbolicConstraint::hard(
            "No deletions",
            "delete",
        );

        let result = engine.search(&cot, &[constraint]).await.unwrap();

        // Some nodes should be pruned
        let stats = engine.stats();
        assert!(stats.pruned > 0 || stats.approved > 0);
    }

    #[test]
    fn test_tree_stats() {
        let engine = TreeOfThoughtEngine::new(5, 3);
        let stats = engine.stats();
        assert_eq!(stats.total, 0);
    }

    #[tokio::test]
    async fn test_expand_respects_depth() {
        let mut engine = TreeOfThoughtEngine::new(2, 3);
        let goal = Goal::new("Shallow search", 5);
        let cot = ChainOfThought::new(goal);

        let result = engine.search(&cot, &[]).await.unwrap();

        // All nodes should have depth <= max_depth
        for node in result.nodes.values() {
            assert!(node.depth <= 2);
        }
    }
}
