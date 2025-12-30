use nafs_core::*;
use nafs_system4::*;

#[tokio::test]
async fn test_full_evolution_cycle() {
    let mut system4 = System4::new(
        System4Config::default(),
        "test_agent".to_string(),
    );

    let failures = vec![
        RecordedFailure::new("use_api_tool", "timeout waiting for response")
            .with_severity(5)
    ];

    // Define a self model with safety constraints
    let self_model = SelfModel {
        agent_id: "test_agent".to_string(),
        identity: "Test Agent".to_string(),
        capabilities: Default::default(),
        weaknesses: vec![],
        terminal_values: vec!["safety".to_string()],
        personality: Default::default(),
    };

    // Run evolution cycle
    let evolution_entries = system4.evolve(&failures, &self_model).await.unwrap();
    
    // We expect 1 evolution entry due to the timeout failure being analyzed and processed
    assert_eq!(evolution_entries.len(), 1);
    let entry = &evolution_entries[0];
    assert_eq!(entry.approval_status, ApprovalStatus::AutoApproved);
    assert!(entry.gradient.failed_action.contains("timeout"));
}

#[test]
fn test_kernel_constraints_validation() {
    let system4 = System4::new(
        System4Config::default(),
        "test_agent".to_string(),
    );

    let self_model = SelfModel {
        agent_id: "test".to_string(),
        identity: "test".to_string(),
        capabilities: Default::default(),
        weaknesses: vec![],
        terminal_values: vec!["safety".to_string()],
        personality: Default::default(),
    };

    let result = system4.verify_immutable_values(&self_model).unwrap();
    assert!(result);
}

#[test]
fn test_evolution_log_persistence() {
    let mut system4 = System4::new(
        System4Config::default(),
        "test_agent".to_string(),
    );

    let before = system4.get_evolution_history().len();
    assert_eq!(before, 0);

    // Manually record something if evolve wasn't run
    // But let's rely on internal methods
    // The evolve method calls record internally
}

#[test]
fn test_failure_recording_integration() {
    let mut system4 = System4::new(
        System4Config::default(),
        "test_agent".to_string(),
    );

    let failure = RecordedFailure::new("test", "test error");
    system4.record_failure(failure).unwrap();
    
    // Internal state isn't directly exposed for failures list, but record_failure shouldn't error
}
