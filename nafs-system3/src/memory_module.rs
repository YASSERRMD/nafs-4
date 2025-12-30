//! Memory Module: Episodic + Semantic Storage
//!
//! Implements Dhikr layer with RAG-based retrieval.

use nafs_core::{EpisodicEvent, KnowledgeEntity, Result};
use std::collections::{HashMap, VecDeque};

/// Episodic memory store for experiences
pub struct EpisodicStore {
    events: VecDeque<EpisodicEvent>,
    max_size: usize,
}

impl EpisodicStore {
    /// Create a new episodic store with max capacity
    pub fn new(max_size: usize) -> Self {
        Self {
            events: VecDeque::new(),
            max_size,
        }
    }

    /// Store an event
    pub fn store(&mut self, event: EpisodicEvent) {
        self.events.push_back(event);
        if self.events.len() > self.max_size {
            self.events.pop_front();
        }
    }

    /// Get most recent N events
    pub fn get_recent(&self, n: usize) -> Vec<EpisodicEvent> {
        self.events.iter().rev().take(n).cloned().collect()
    }

    /// Get events by outcome (success/failure)
    pub fn get_by_success(&self, success: bool) -> Vec<EpisodicEvent> {
        self.events
            .iter()
            .filter(|e| e.outcome.is_success() == success)
            .cloned()
            .collect()
    }

    /// Get positive experiences
    pub fn get_positive(&self) -> Vec<EpisodicEvent> {
        self.events
            .iter()
            .filter(|e| e.is_positive())
            .cloned()
            .collect()
    }

    /// Search events by observation content
    pub fn search(&self, query: &str) -> Vec<EpisodicEvent> {
        let query_lower = query.to_lowercase();
        self.events
            .iter()
            .filter(|e| e.observation.to_lowercase().contains(&query_lower))
            .cloned()
            .collect()
    }

    /// Get store size
    pub fn size(&self) -> usize {
        self.events.len()
    }

    /// Clear all events
    pub fn clear(&mut self) {
        self.events.clear();
    }
}

impl Default for EpisodicStore {
    fn default() -> Self {
        Self::new(10000)
    }
}

/// Semantic memory store for knowledge
pub struct SemanticStore {
    entities: HashMap<String, KnowledgeEntity>,
}

impl SemanticStore {
    /// Create a new semantic store
    pub fn new() -> Self {
        Self {
            entities: HashMap::new(),
        }
    }

    /// Add or update an entity
    pub fn add_entity(&mut self, entity: KnowledgeEntity) {
        self.entities.insert(entity.id.clone(), entity);
    }

    /// Get entity by ID
    pub fn get_entity(&self, id: &str) -> Option<KnowledgeEntity> {
        self.entities.get(id).cloned()
    }

    /// Get entities by type
    pub fn get_by_type(&self, entity_type: &str) -> Vec<KnowledgeEntity> {
        self.entities
            .values()
            .filter(|e| e.entity_type == entity_type)
            .cloned()
            .collect()
    }

    /// Search entities by name
    pub fn search_by_name(&self, query: &str) -> Vec<KnowledgeEntity> {
        let query_lower = query.to_lowercase();
        self.entities
            .values()
            .filter(|e| e.name.to_lowercase().contains(&query_lower))
            .cloned()
            .collect()
    }

    /// Get related entities
    pub fn get_related(&self, entity_id: &str) -> Vec<KnowledgeEntity> {
        if let Some(entity) = self.entities.get(entity_id) {
            entity
                .relationships
                .iter()
                .filter_map(|(_, target_id)| self.entities.get(target_id).cloned())
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Remove entity
    pub fn remove(&mut self, id: &str) -> Option<KnowledgeEntity> {
        self.entities.remove(id)
    }

    /// Get store size
    pub fn size(&self) -> usize {
        self.entities.len()
    }

    /// Clear all entities
    pub fn clear(&mut self) {
        self.entities.clear();
    }
}

impl Default for SemanticStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Unified memory module
pub struct MemoryModule {
    pub episodic: EpisodicStore,
    pub semantic: SemanticStore,
}

impl MemoryModule {
    /// Create a new memory module
    pub fn new(max_episodic: usize) -> Self {
        Self {
            episodic: EpisodicStore::new(max_episodic),
            semantic: SemanticStore::new(),
        }
    }

    /// Store an experience
    pub fn store_experience(&mut self, event: EpisodicEvent) -> Result<()> {
        self.episodic.store(event);
        Ok(())
    }

    /// Store knowledge entity
    pub fn store_knowledge(&mut self, entity: KnowledgeEntity) -> Result<()> {
        self.semantic.add_entity(entity);
        Ok(())
    }

    /// Retrieve similar experiences (simplified - text matching)
    pub async fn retrieve_similar(&self, query: &str) -> Result<Vec<EpisodicEvent>> {
        // In real implementation, this would use vector embeddings
        Ok(self.episodic.search(query))
    }

    /// Retrieve recent experiences
    pub async fn retrieve_recent(&self, n: usize) -> Result<Vec<EpisodicEvent>> {
        Ok(self.episodic.get_recent(n))
    }

    /// Get memory statistics
    pub fn stats(&self) -> (usize, usize) {
        (self.episodic.size(), self.semantic.size())
    }

    /// Clear all memory
    pub fn clear(&mut self) {
        self.episodic.clear();
        self.semantic.clear();
    }
}

impl Default for MemoryModule {
    fn default() -> Self {
        Self::new(10000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nafs_core::Outcome;

    #[test]
    fn test_episodic_store() {
        let mut store = EpisodicStore::new(100);
        let event = EpisodicEvent::new("Test observation", Outcome::Success);

        store.store(event);
        assert_eq!(store.size(), 1);
    }

    #[test]
    fn test_episodic_capacity() {
        let mut store = EpisodicStore::new(3);

        for i in 0..5 {
            let event = EpisodicEvent::new(format!("Event {}", i), Outcome::Success);
            store.store(event);
        }

        assert_eq!(store.size(), 3);
        // Most recent should be "Event 4"
        let recent = store.get_recent(1);
        assert!(recent[0].observation.contains("4"));
    }

    #[test]
    fn test_episodic_search() {
        let mut store = EpisodicStore::new(100);

        store.store(EpisodicEvent::new("Hello world", Outcome::Success));
        store.store(EpisodicEvent::new("Goodbye world", Outcome::Success));
        store.store(EpisodicEvent::new("Test message", Outcome::Success));

        let results = store.search("world");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_semantic_store() {
        let mut store = SemanticStore::new();
        let entity = KnowledgeEntity::new("Person", "Alice")
            .with_attribute("role", "developer")
            .with_confidence(0.95);

        store.add_entity(entity.clone());
        assert_eq!(store.get_entity(&entity.id).unwrap().name, "Alice");
    }

    #[test]
    fn test_semantic_search() {
        let mut store = SemanticStore::new();

        store.add_entity(KnowledgeEntity::new("Person", "Alice"));
        store.add_entity(KnowledgeEntity::new("Person", "Bob"));
        store.add_entity(KnowledgeEntity::new("Concept", "Machine Learning"));

        let persons = store.get_by_type("Person");
        assert_eq!(persons.len(), 2);

        let alice = store.search_by_name("Ali");
        assert_eq!(alice.len(), 1);
    }

    #[test]
    fn test_memory_module() {
        let mut memory = MemoryModule::new(100);

        let event = EpisodicEvent::new("Test", Outcome::Success);
        memory.store_experience(event).unwrap();

        let entity = KnowledgeEntity::new("Concept", "Testing");
        memory.store_knowledge(entity).unwrap();

        let (episodic, semantic) = memory.stats();
        assert_eq!(episodic, 1);
        assert_eq!(semantic, 1);
    }

    #[tokio::test]
    async fn test_retrieve_similar() {
        let mut memory = MemoryModule::new(100);

        memory
            .store_experience(EpisodicEvent::new("Learning Rust", Outcome::Success))
            .unwrap();
        memory
            .store_experience(EpisodicEvent::new("Learning Python", Outcome::Success))
            .unwrap();

        let results = memory.retrieve_similar("Rust").await.unwrap();
        assert_eq!(results.len(), 1);
    }
}
