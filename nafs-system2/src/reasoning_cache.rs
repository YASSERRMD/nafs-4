//! Reasoning Cache: Reuse successful plans
//!
//! Caches chain-of-thought results to avoid recomputation.

use nafs_core::{CachedReasoning, ChainOfThought, Goal};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// Cache for storing and retrieving reasoning results
pub struct ReasoningCache {
    /// Cached entries indexed by goal hash
    cache: HashMap<String, CachedReasoning>,
    /// Maximum cache size
    max_size: usize,
}

impl ReasoningCache {
    /// Create a new reasoning cache with specified max size
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: HashMap::new(),
            max_size,
        }
    }

    /// Get cached reasoning for a goal (if exists)
    pub fn get(&mut self, goal: &Goal) -> Option<CachedReasoning> {
        let goal_hash = self.hash_goal(goal);

        if let Some(entry) = self.cache.get_mut(&goal_hash) {
            entry.record_hit();
            tracing::debug!("Cache hit for goal hash: {}", goal_hash);
            Some(entry.clone())
        } else {
            tracing::debug!("Cache miss for goal hash: {}", goal_hash);
            None
        }
    }

    /// Cache a chain-of-thought result
    pub fn cache(&mut self, cot: ChainOfThought) {
        let goal_hash = self.hash_goal(&cot.goal);

        // Evict if at capacity
        if self.cache.len() >= self.max_size {
            self.evict_lru();
        }

        let cached = CachedReasoning::new(goal_hash.clone(), cot);
        self.cache.insert(goal_hash.clone(), cached);

        tracing::debug!("Cached reasoning result with hash: {}", goal_hash);
    }

    /// Evict the least recently used entry
    fn evict_lru(&mut self) {
        if let Some(lru_key) = self
            .cache
            .iter()
            .min_by_key(|(_, v)| v.last_used)
            .map(|(k, _)| k.clone())
        {
            self.cache.remove(&lru_key);
            tracing::debug!("Evicted cache entry: {}", lru_key);
        }
    }

    /// Hash a goal for cache lookup
    fn hash_goal(&self, goal: &Goal) -> String {
        let mut hasher = DefaultHasher::new();
        goal.description.hash(&mut hasher);
        goal.priority.hash(&mut hasher);

        // Include criteria in hash
        for criterion in &goal.success_criteria {
            criterion.hash(&mut hasher);
        }

        format!("{:x}", hasher.finish())
    }

    /// Get cache statistics
    pub fn stats(&self) -> (usize, usize) {
        let total_hits: u32 = self.cache.values().map(|v| v.hits).sum();
        (self.cache.len(), total_hits as usize)
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.cache.clear();
        tracing::debug!("Cache cleared");
    }

    /// Check if a goal is cached
    pub fn contains(&self, goal: &Goal) -> bool {
        let goal_hash = self.hash_goal(goal);
        self.cache.contains_key(&goal_hash)
    }

    /// Get current cache size
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Remove a specific goal from cache
    pub fn invalidate(&mut self, goal: &Goal) -> bool {
        let goal_hash = self.hash_goal(goal);
        self.cache.remove(&goal_hash).is_some()
    }
}

impl Default for ReasoningCache {
    fn default() -> Self {
        Self::new(1000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_goal(desc: &str) -> Goal {
        Goal::new(desc, 5)
    }

    fn make_cot(goal: Goal) -> ChainOfThought {
        ChainOfThought::new(goal)
    }

    #[test]
    fn test_cache_miss() {
        let mut cache = ReasoningCache::new(100);
        let goal = make_goal("Test goal");

        assert!(cache.get(&goal).is_none());
    }

    #[test]
    fn test_cache_hit() {
        let mut cache = ReasoningCache::new(100);
        let goal = make_goal("Test goal");
        let cot = make_cot(goal.clone());

        cache.cache(cot);

        let cached = cache.get(&goal);
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().hits, 1);
    }

    #[test]
    fn test_cache_hit_count() {
        let mut cache = ReasoningCache::new(100);
        let goal = make_goal("Test goal");
        let cot = make_cot(goal.clone());

        cache.cache(cot);

        // Multiple hits
        let _ = cache.get(&goal);
        let _ = cache.get(&goal);
        let cached = cache.get(&goal);

        assert_eq!(cached.unwrap().hits, 3);
    }

    #[test]
    fn test_cache_eviction() {
        let mut cache = ReasoningCache::new(2);

        // Fill cache
        cache.cache(make_cot(make_goal("Goal 1")));
        cache.cache(make_cot(make_goal("Goal 2")));
        assert_eq!(cache.len(), 2);

        // Add one more (should evict)
        cache.cache(make_cot(make_goal("Goal 3")));
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_cache_stats() {
        let mut cache = ReasoningCache::new(100);
        let goal = make_goal("Stats test");
        let cot = make_cot(goal.clone());

        cache.cache(cot);
        let _ = cache.get(&goal);
        let _ = cache.get(&goal);

        let (size, hits) = cache.stats();
        assert_eq!(size, 1);
        assert_eq!(hits, 2);
    }

    #[test]
    fn test_cache_clear() {
        let mut cache = ReasoningCache::new(100);
        cache.cache(make_cot(make_goal("Goal 1")));
        cache.cache(make_cot(make_goal("Goal 2")));

        assert_eq!(cache.len(), 2);
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_contains() {
        let mut cache = ReasoningCache::new(100);
        let goal = make_goal("Contains test");

        assert!(!cache.contains(&goal));
        cache.cache(make_cot(goal.clone()));
        assert!(cache.contains(&goal));
    }

    #[test]
    fn test_cache_invalidate() {
        let mut cache = ReasoningCache::new(100);
        let goal = make_goal("Invalidate test");

        cache.cache(make_cot(goal.clone()));
        assert!(cache.contains(&goal));

        let removed = cache.invalidate(&goal);
        assert!(removed);
        assert!(!cache.contains(&goal));
    }

    #[test]
    fn test_hash_includes_criteria() {
        let cache = ReasoningCache::new(100);

        let goal1 = Goal::new("Same desc", 5);
        let goal2 = Goal::new("Same desc", 5).with_criterion("Extra");

        let hash1 = cache.hash_goal(&goal1);
        let hash2 = cache.hash_goal(&goal2);

        assert_ne!(hash1, hash2);
    }
}
