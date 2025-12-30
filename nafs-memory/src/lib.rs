//! NAFS-4 Memory Subsystem
//!
//! Provides unified interfaces for:
//! - Vector database (similarity search)
//! - Graph database (relational queries)
//!
//! Abstracted behind traits to allow different backends.

use async_trait::async_trait;
use nafs_core::{MemoryItem, Result};
use std::collections::HashMap;

/// Trait for vector database operations
#[async_trait]
pub trait VectorStore: Send + Sync {
    /// Store a vector with associated metadata
    async fn store(&self, id: &str, vector: &[f32], metadata: &HashMap<String, String>) -> Result<()>;

    /// Search for similar vectors
    async fn search(&self, query: &[f32], limit: usize) -> Result<Vec<SearchResult>>;

    /// Delete a vector by ID
    async fn delete(&self, id: &str) -> Result<()>;

    /// Get vector count
    async fn count(&self) -> Result<usize>;
}

/// Result of a vector search
#[derive(Clone, Debug)]
pub struct SearchResult {
    pub id: String,
    pub score: f32,
    pub metadata: HashMap<String, String>,
}

/// In-memory vector store for development/testing
pub struct InMemoryVectorStore {
    vectors: std::sync::RwLock<HashMap<String, StoredVector>>,
}

#[derive(Clone)]
struct StoredVector {
    vector: Vec<f32>,
    metadata: HashMap<String, String>,
}

impl InMemoryVectorStore {
    /// Create a new in-memory vector store
    pub fn new() -> Self {
        Self {
            vectors: std::sync::RwLock::new(HashMap::new()),
        }
    }

    /// Compute cosine similarity between two vectors
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        dot / (norm_a * norm_b)
    }
}

impl Default for InMemoryVectorStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl VectorStore for InMemoryVectorStore {
    async fn store(&self, id: &str, vector: &[f32], metadata: &HashMap<String, String>) -> Result<()> {
        let mut vectors = self.vectors.write().unwrap();
        vectors.insert(
            id.to_string(),
            StoredVector {
                vector: vector.to_vec(),
                metadata: metadata.clone(),
            },
        );
        Ok(())
    }

    async fn search(&self, query: &[f32], limit: usize) -> Result<Vec<SearchResult>> {
        let vectors = self.vectors.read().unwrap();
        let mut results: Vec<_> = vectors
            .iter()
            .map(|(id, stored)| {
                let score = Self::cosine_similarity(query, &stored.vector);
                SearchResult {
                    id: id.clone(),
                    score,
                    metadata: stored.metadata.clone(),
                }
            })
            .collect();
        
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        Ok(results)
    }

    async fn delete(&self, id: &str) -> Result<()> {
        let mut vectors = self.vectors.write().unwrap();
        vectors.remove(id);
        Ok(())
    }

    async fn count(&self) -> Result<usize> {
        let vectors = self.vectors.read().unwrap();
        Ok(vectors.len())
    }
}

/// Trait for graph database operations
#[async_trait]
pub trait GraphStore: Send + Sync {
    /// Create a node
    async fn create_node(&self, id: &str, labels: &[&str], properties: &HashMap<String, String>) -> Result<()>;

    /// Create an edge between nodes
    async fn create_edge(&self, from: &str, to: &str, relation: &str, properties: &HashMap<String, String>) -> Result<()>;

    /// Query nodes by label
    async fn query_by_label(&self, label: &str) -> Result<Vec<GraphNode>>;

    /// Get neighbors of a node
    async fn get_neighbors(&self, id: &str, relation: Option<&str>) -> Result<Vec<GraphNode>>;

    /// Delete a node
    async fn delete_node(&self, id: &str) -> Result<()>;
}

/// A node in the graph
#[derive(Clone, Debug)]
pub struct GraphNode {
    pub id: String,
    pub labels: Vec<String>,
    pub properties: HashMap<String, String>,
}

/// An edge in the graph
#[derive(Clone, Debug)]
pub struct GraphEdge {
    pub from: String,
    pub to: String,
    pub relation: String,
    pub properties: HashMap<String, String>,
}

/// In-memory graph store for development/testing
pub struct InMemoryGraphStore {
    nodes: std::sync::RwLock<HashMap<String, GraphNode>>,
    edges: std::sync::RwLock<Vec<GraphEdge>>,
}

impl InMemoryGraphStore {
    /// Create a new in-memory graph store
    pub fn new() -> Self {
        Self {
            nodes: std::sync::RwLock::new(HashMap::new()),
            edges: std::sync::RwLock::new(Vec::new()),
        }
    }
}

impl Default for InMemoryGraphStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl GraphStore for InMemoryGraphStore {
    async fn create_node(&self, id: &str, labels: &[&str], properties: &HashMap<String, String>) -> Result<()> {
        let mut nodes = self.nodes.write().unwrap();
        nodes.insert(
            id.to_string(),
            GraphNode {
                id: id.to_string(),
                labels: labels.iter().map(|s| s.to_string()).collect(),
                properties: properties.clone(),
            },
        );
        Ok(())
    }

    async fn create_edge(&self, from: &str, to: &str, relation: &str, properties: &HashMap<String, String>) -> Result<()> {
        let mut edges = self.edges.write().unwrap();
        edges.push(GraphEdge {
            from: from.to_string(),
            to: to.to_string(),
            relation: relation.to_string(),
            properties: properties.clone(),
        });
        Ok(())
    }

    async fn query_by_label(&self, label: &str) -> Result<Vec<GraphNode>> {
        let nodes = self.nodes.read().unwrap();
        Ok(nodes
            .values()
            .filter(|n| n.labels.contains(&label.to_string()))
            .cloned()
            .collect())
    }

    async fn get_neighbors(&self, id: &str, relation: Option<&str>) -> Result<Vec<GraphNode>> {
        let nodes = self.nodes.read().unwrap();
        let edges = self.edges.read().unwrap();
        
        let neighbor_ids: Vec<_> = edges
            .iter()
            .filter(|e| e.from == id || e.to == id)
            .filter(|e| relation.map_or(true, |r| e.relation == r))
            .map(|e| if e.from == id { &e.to } else { &e.from })
            .collect();

        Ok(neighbor_ids
            .iter()
            .filter_map(|id| nodes.get(*id).cloned())
            .collect())
    }

    async fn delete_node(&self, id: &str) -> Result<()> {
        let mut nodes = self.nodes.write().unwrap();
        let mut edges = self.edges.write().unwrap();
        nodes.remove(id);
        edges.retain(|e| e.from != id && e.to != id);
        Ok(())
    }
}

/// Unified memory interface combining vector and graph stores
pub struct UnifiedMemory {
    vector_store: Box<dyn VectorStore>,
    graph_store: Box<dyn GraphStore>,
}

impl UnifiedMemory {
    /// Create a new unified memory with the given stores
    pub fn new(vector_store: Box<dyn VectorStore>, graph_store: Box<dyn GraphStore>) -> Self {
        Self {
            vector_store,
            graph_store,
        }
    }

    /// Create an in-memory unified memory for testing
    pub fn in_memory() -> Self {
        Self {
            vector_store: Box::new(InMemoryVectorStore::new()),
            graph_store: Box::new(InMemoryGraphStore::new()),
        }
    }

    /// Store a memory item
    pub async fn store(&self, item: &MemoryItem) -> Result<()> {
        // Store in vector DB
        let mut metadata = HashMap::new();
        metadata.insert("content".to_string(), item.content.clone());
        metadata.insert("category".to_string(), format!("{:?}", item.category));
        
        self.vector_store.store(&item.id, &item.embedding, &metadata).await?;

        // Store in graph DB as a node
        self.graph_store.create_node(
            &item.id,
            &["Memory", &format!("{:?}", item.category)],
            &metadata,
        ).await?;

        Ok(())
    }

    /// Search for similar memories
    pub async fn search(&self, query_embedding: &[f32], limit: usize) -> Result<Vec<SearchResult>> {
        self.vector_store.search(query_embedding, limit).await
    }

    /// Get the vector store
    pub fn vector_store(&self) -> &dyn VectorStore {
        &*self.vector_store
    }

    /// Get the graph store
    pub fn graph_store(&self) -> &dyn GraphStore {
        &*self.graph_store
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nafs_core::MemoryCategory;

    #[tokio::test]
    async fn test_in_memory_vector_store() {
        let store = InMemoryVectorStore::new();
        let metadata = HashMap::new();
        
        store.store("test1", &[1.0, 0.0, 0.0], &metadata).await.unwrap();
        store.store("test2", &[0.0, 1.0, 0.0], &metadata).await.unwrap();
        
        let results = store.search(&[1.0, 0.0, 0.0], 2).await.unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "test1");
    }

    #[tokio::test]
    async fn test_in_memory_graph_store() {
        let store = InMemoryGraphStore::new();
        let props = HashMap::new();
        
        store.create_node("node1", &["Person"], &props).await.unwrap();
        store.create_node("node2", &["Person"], &props).await.unwrap();
        store.create_edge("node1", "node2", "knows", &props).await.unwrap();
        
        let neighbors = store.get_neighbors("node1", None).await.unwrap();
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].id, "node2");
    }

    #[tokio::test]
    async fn test_unified_memory() {
        let memory = UnifiedMemory::in_memory();
        let item = MemoryItem::new("Test content", MemoryCategory::Semantic)
            .with_embedding(vec![1.0, 0.0, 0.0]);
        
        memory.store(&item).await.unwrap();
        
        let results = memory.search(&[1.0, 0.0, 0.0], 10).await.unwrap();
        assert_eq!(results.len(), 1);
    }
}
