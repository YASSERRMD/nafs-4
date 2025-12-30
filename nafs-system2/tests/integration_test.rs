//! Integration tests for System 2 (Reasoning Layer)

use nafs_core::*;
use nafs_system2::*;

#[tokio::test]
async fn test_full_reasoning_pipeline() {
    let mut system2 = System2::new();

    let goal = Goal::new("Complete a multi-step task", 8)
        .with_criterion("Step 1 done")
        .with_criterion("Step 2 done");

    let state = State::new("test_agent");

    let constraints = vec![];

    let actions = system2.reason(&goal, &state, &constraints).await;
    assert!(actions.is_ok());
}

#[tokio::test]
async fn test_reasoning_with_constraints() {
    let mut system2 = System2::new();

    let goal = Goal::new("Perform safe operations", 7);
    let state = State::new("test_agent");

    let constraints = vec![
        SymbolicConstraint::hard("No unsafe operations", "unsafe"),
    ];

    let result = system2.reason(&goal, &state, &constraints).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_caching_behavior() {
    let mut system2 = System2::with_config(
        System2Config::default().with_max_steps(5),
    );

    let goal = Goal::new("Cached task", 5);
    let state = State::new("test_agent");

    // First call
    let result1 = system2.reason(&goal, &state, &[]).await;
    assert!(result1.is_ok());

    // Check cache has entry
    let (cache_size, _) = system2.cache_stats();
    assert_eq!(cache_size, 1);

    // Second call should hit cache
    let result2 = system2.reason(&goal, &state, &[]).await;
    assert!(result2.is_ok());

    let (_, hits) = system2.cache_stats();
    assert!(hits >= 1);
}

#[test]
fn test_symbolic_verification_integration() {
    let verifier = SymbolicVerifier::new();

    // Safe action should pass
    let safe_action = Action::new("search", serde_json::json!({"query": "test"}));
    let result = verifier.verify(&[safe_action], &[]).unwrap();
    assert!(result.passes_logic);

    // Dangerous action should fail
    let dangerous_action = Action::new("access_private_data", serde_json::json!({}));
    let result = verifier.verify(&[dangerous_action], &[]).unwrap();
    assert!(!result.passes_logic);
}

#[test]
fn test_reasoning_cache_integration() {
    let mut cache = ReasoningCache::new(100);

    let goal = Goal::new("Cached goal", 5)
        .with_criterion("Success");

    let cot = ChainOfThought::new(goal.clone());

    // Cache and retrieve
    cache.cache(cot);

    let cached = cache.get(&goal);
    assert!(cached.is_some());
    assert_eq!(cached.unwrap().hits, 1);

    // Second retrieval increases hits
    let cached2 = cache.get(&goal);
    assert!(cached2.is_some());
    assert_eq!(cached2.unwrap().hits, 2);
}

#[tokio::test]
async fn test_tree_of_thought_integration() {
    let mut engine = TreeOfThoughtEngine::new(5, 3);
    let goal = Goal::new("ToT test", 5);
    let cot = ChainOfThought::new(goal);

    let result = engine.search(&cot, &[]).await.unwrap();

    assert!(!result.nodes.is_empty());
    assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
}

#[test]
fn test_process_supervision_integration() {
    let supervisor = ProcessSupervisor::new();

    let goal = Goal::new("Supervised task", 5)
        .with_criterion("Done correctly");

    let mut cot = ChainOfThought::new(goal);
    cot.add_step(
        ReasoningStep::new("Analyze", "Understanding the task")
            .with_confidence(0.85),
    );
    cot.add_step(
        ReasoningStep::new("Plan", "Creating action plan")
            .with_confidence(0.9),
    );

    let actions = vec![Action::new("execute_task", serde_json::json!({}))];

    let feedback = supervisor.supervise(&cot, &actions).unwrap();

    assert!(feedback.reasoning_coherence > 0.0);
    assert!(feedback.action_relevance > 0.0);
    assert!(feedback.plan_completeness > 0.0);
}

#[tokio::test]
async fn test_llm_planner_integration() {
    let planner = LLMPlanner::new();

    let goal = Goal::new("Generate reasoning", 6)
        .with_criterion("Good quality")
        .with_criterion("Complete");

    let state = State::new("test_agent");

    let cot = planner.generate_cot(&goal, &state).await.unwrap();

    assert!(!cot.steps.is_empty());
    assert!(cot.reasoning_quality > 0.0);
    assert_eq!(cot.goal.id, goal.id);
}

#[tokio::test]
async fn test_system2_custom_config() {
    let config = System2Config::default()
        .with_max_steps(10)
        .with_tot(3, 2)
        .with_strictness(0.9)
        .without_cache();

    let mut system2 = System2::with_config(config);

    assert_eq!(system2.config.max_reasoning_steps, 10);
    assert_eq!(system2.config.max_tot_depth, 3);
    assert_eq!(system2.config.max_tot_width, 2);
    assert!(!system2.config.cache_enabled);

    let goal = Goal::new("No cache test", 5);
    let state = State::new("test_agent");

    let _ = system2.reason(&goal, &state, &[]).await;

    // Cache should remain empty since disabled
    let (cache_size, _) = system2.cache_stats();
    assert_eq!(cache_size, 0);
}
