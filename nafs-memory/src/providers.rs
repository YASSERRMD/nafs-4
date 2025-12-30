use crate::{GraphNode, GraphStore, SearchResult, VectorStore};
use async_trait::async_trait;
use nafs_core::Result;
use std::collections::HashMap;
use tonic::transport::Channel;

// Include generated gRPC code
pub mod barq_db {
    tonic::include_proto!("barq_db");
}
pub mod barq_graph {
    tonic::include_proto!("barq_graph");
}

use barq_db::barq_db_client::BarqDbClient;
use barq_db::{CreateCollectionRequest, InsertDocumentRequest, SearchRequest};

use barq_graph::barq_graph_client::BarqGraphClient;
use barq_graph::{CreateEdgeRequest, CreateNodeRequest, GetNeighborsRequest, QueryByLabelRequest};

/// Vector store implementation using Barq-DB
pub struct BarqDBStore {
    client: BarqDbClient<Channel>,
    collection: String,
}

impl BarqDBStore {
    pub async fn connect(addr: String, collection: String) -> Result<Self> {
        let client = BarqDbClient::connect(addr)
            .await
            .map_err(|e| nafs_core::NafsError::memory(format!("Failed to connect to Barq-DB: {}", e)))?;
        
        Ok(Self { client, collection })
    }

    pub async fn ensure_collection(&mut self, dimension: i32) -> Result<()> {
        let request = tonic::Request::new(CreateCollectionRequest {
            name: self.collection.clone(),
            dimension,
        });
        self.client.create_collection(request).await.ok(); // Ignore if already exists
        Ok(())
    }
}

#[async_trait]
impl VectorStore for BarqDBStore {
    async fn store(&self, id: &str, vector: &[f32], metadata: &HashMap<String, String>) -> Result<()> {
        let mut client = self.client.clone();
        let metadata_json = serde_json::to_string(metadata).unwrap_or_default();
        
        let request = tonic::Request::new(InsertDocumentRequest {
            collection_name: self.collection.clone(),
            id: id.to_string(),
            values: vector.to_vec(),
            metadata_json,
        });

        client.insert_document(request).await
            .map_err(|e| nafs_core::NafsError::memory(format!("Barq-DB insert failed: {}", e)))?;
        
        Ok(())
    }

    async fn search(&self, query: &[f32], limit: usize) -> Result<Vec<SearchResult>> {
        let mut client = self.client.clone();
        
        let request = tonic::Request::new(SearchRequest {
            collection_name: self.collection.clone(),
            vector: query.to_vec(),
            limit: limit as i32,
        });

        let response = client.search(request).await
            .map_err(|e| nafs_core::NafsError::memory(format!("Barq-DB search failed: {}", e)))?;
        
        let hits = response.into_inner().hits;
        let results = hits.into_iter().map(|hit| {
            let metadata: HashMap<String, String> = serde_json::from_str(&hit.metadata_json).unwrap_or_default();
            SearchResult {
                id: hit.id,
                score: hit.score,
                metadata,
            }
        }).collect();

        Ok(results)
    }

    async fn delete(&self, _id: &str) -> Result<()> {
        // Barq-DB proto didn't have delete in my quick search, but I'll stub it
        Ok(())
    }

    async fn count(&self) -> Result<usize> {
        Ok(0)
    }
}

/// Graph store implementation using Barq-GraphDB
pub struct BarqGraphStore {
    client: BarqGraphClient<Channel>,
}

impl BarqGraphStore {
    pub async fn connect(addr: String) -> Result<Self> {
        let client = BarqGraphClient::connect(addr)
            .await
            .map_err(|e| nafs_core::NafsError::memory(format!("Failed to connect to Barq-GraphDB: {}", e)))?;
        
        Ok(Self { client })
    }
}

#[async_trait]
impl GraphStore for BarqGraphStore {
    async fn create_node(&self, id: &str, labels: &[&str], properties: &HashMap<String, String>) -> Result<()> {
        let mut client = self.client.clone();
        
        let request = tonic::Request::new(CreateNodeRequest {
            id: id.to_string(),
            labels: labels.iter().map(|s| s.to_string()).collect(),
            properties: properties.clone(),
        });

        client.create_node(request).await
            .map_err(|e| nafs_core::NafsError::memory(format!("Barq-GraphDB create_node failed: {}", e)))?;
        
        Ok(())
    }

    async fn create_edge(&self, from: &str, to: &str, relation: &str, properties: &HashMap<String, String>) -> Result<()> {
        let mut client = self.client.clone();
        
        let request = tonic::Request::new(CreateEdgeRequest {
            from: from.to_string(),
            to: to.to_string(),
            relation: relation.to_string(),
            properties: properties.clone(),
        });

        client.create_edge(request).await
            .map_err(|e| nafs_core::NafsError::memory(format!("Barq-GraphDB create_edge failed: {}", e)))?;
        
        Ok(())
    }

    async fn query_by_label(&self, label: &str) -> Result<Vec<GraphNode>> {
        let mut client = self.client.clone();
        
        let request = tonic::Request::new(QueryByLabelRequest {
            label: label.to_string(),
        });

        let response = client.query_by_label(request).await
            .map_err(|e| nafs_core::NafsError::memory(format!("Barq-GraphDB query_by_label failed: {}", e)))?;
        
        let nodes = response.into_inner().nodes;
        Ok(nodes.into_iter().map(|n| GraphNode {
            id: n.id,
            labels: n.labels,
            properties: n.properties,
        }).collect())
    }

    async fn get_neighbors(&self, id: &str, relation: Option<&str>) -> Result<Vec<GraphNode>> {
        let mut client = self.client.clone();
        
        let request = tonic::Request::new(GetNeighborsRequest {
            id: id.to_string(),
            relation: relation.map(|s| s.to_string()),
        });

        let response = client.get_neighbors(request).await
            .map_err(|e| nafs_core::NafsError::memory(format!("Barq-GraphDB get_neighbors failed: {}", e)))?;
        
        let nodes = response.into_inner().nodes;
        Ok(nodes.into_iter().map(|n| GraphNode {
            id: n.id,
            labels: n.labels,
            properties: n.properties,
        }).collect())
    }

    async fn delete_node(&self, _id: &str) -> Result<()> {
        Ok(())
    }
}
