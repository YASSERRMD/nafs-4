//! Integration tests for System 3 (Meta-Cognition)

use nafs_core::*;
use nafs_system3::*;

#[tokio::test]
async fn test_executive_monitor_full_cycle() {
    let mut system3 = System3::new(
        System3Config::default(),
        "test_agent".to_string(),
    );

    let state = State::new("test_agent");

    let (goal, reward) = system3.tick(&state).await.unwrap();
    assert!(!goal.id.is_empty());
    assert!(reward.total >= 0.0 && reward.total <= 1.0);
}

#[test]
fn test_memory_operations_integration() {
    let mut system3 = System3::new(
        System3Config::default(),
        "test_agent".to_string(),
    );

    let event = EpisodicEvent::new("Test observation", Outcome::Success)
        .with_reflection("Good experience")
        .with_valence(0.8);

    system3.remember(event).unwrap();
    
    let (episodic_count, semantic_count) = system3.memory_stats();
    assert_eq!(episodic_count, 1);
    assert_eq!(semantic_count, 0);
}

#[test]
fn test_self_model_persistence_integration() {
    let mut system3 = System3::new(
        System3Config::default(),
        "test_agent".to_string(),
    );

    system3.update_capability("reasoning".to_string(), 0.95);
    
    let self_model = system3.get_self_model();
    assert_eq!(self_model.capabilities.get("reasoning"), Some(&0.95));
}

#[test]
fn test_intrinsic_motivation_integration() {
    let system3 = System3::new(
        System3Config::default(),
        "test_agent".to_string(),
    );

    let drive = system3.get_motivation();
    assert!(drive >= 0.0 && drive <= 1.0);
}

#[test]
fn test_config_propagation() {
    let config = System3Config::default().with_max_events(500);
    let system3 = System3::new(config, "test_agent".to_string());
    
    assert_eq!(system3.config.max_episodic_events, 500);
}
