//! Health Monitor: System Status Tracking
//!
//! Monitors system health and uptime.

use std::time::{SystemTime, UNIX_EPOCH};

/// System health monitor
pub struct HealthMonitor {
    start_time: u64,
}

impl HealthMonitor {
    /// Create new health monitor
    pub fn new() -> Self {
        let start_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Self { start_time }
    }

    /// Get uptime in seconds
    pub fn uptime_seconds(&self) -> u64 {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        now - self.start_time
    }
}

impl Default for HealthMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_monitor() {
        let monitor = HealthMonitor::new();
        let uptime = monitor.uptime_seconds();
        assert!(uptime >= 0);
    }
}
