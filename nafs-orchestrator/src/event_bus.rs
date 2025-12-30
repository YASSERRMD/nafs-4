//! Event Bus: System-wide Pub/Sub
//!
//! Routes events across system components.

use nafs_core::Result;

/// Simple event bus for intra-system communication
pub struct EventBus;

impl EventBus {
    /// Create new event bus
    pub fn new() -> Self {
        Self
    }

    /// Publish an event
    pub fn publish(&self, event_type: &str, data: &str) -> Result<()> {
        tracing::debug!("Event published: {} -> {}", event_type, data);
        // In Phase 4, this is a simple logger
        // Phase 5 will add full event queue + subscribers
        Ok(())
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_bus() {
        let bus = EventBus::new();
        bus.publish("test_event", "test_data").unwrap();
    }
}
