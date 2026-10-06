use numerica::numerical_integration::qmc::{QmcAccumulator, QmcPartial, QmcPlan, Rank1Rule};

#[test]
fn coverage_survives_unrepresentable_shift_sum_and_retains_gaps() {
    let plan = QmcPlan::new(Rank1Rule::new(2, vec![1]).unwrap(), 2, 1, 0).unwrap();
    let packages: Vec<_> = plan.packages(1).unwrap().collect();
    let mut accumulator = QmcAccumulator::new(plan.clone(), 1).unwrap();
    for &index in &[3, 0, 2] {
        let mut partial = QmcPartial::new(&plan, packages[index], 1).unwrap();
        partial.push(&[1e308]).unwrap();
        accumulator.merge(partial.finish().unwrap()).unwrap();
    }
    assert_eq!(accumulator.complete_shift_ids(), vec![1]);
    assert_eq!(accumulator.completed_points(), 3);
    assert!(accumulator.shift_estimates().is_err());
    let mut missing = QmcPartial::new(&plan, packages[1], 1).unwrap();
    missing.push(&[1e308]).unwrap();
    accumulator.merge(missing.finish().unwrap()).unwrap();
    assert_eq!(accumulator.complete_shift_ids(), vec![0, 1]);
    assert!(accumulator.is_complete());
    assert!(accumulator.estimate().is_err());
}
