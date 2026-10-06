//! Serial evaluation and caller-owned threads using the same canonical packages.
use numerica::numerical_integration::qmc::{
    Korobov3, QmcAccumulator, QmcPartial, QmcPlan, QmcWorkPackage, Rank1Rule,
};

fn evaluate(plan: &QmcPlan, work: QmcWorkPackage) -> QmcPartial {
    let mut points = vec![0.0; work.point_count() as usize * plan.dimension()];
    plan.fill_points(work, &mut points).unwrap();
    let mut partial = QmcPartial::new(plan, work, 2).unwrap();
    for point in points.chunks_exact_mut(plan.dimension()) {
        let weight = Korobov3::transform_in_place(point).unwrap();
        let value = point[0] * point[1]; // integral = 1/4
        partial
            .push(&[weight * value, weight * (1.0 - value)])
            .unwrap();
    }
    partial.finish().unwrap()
}

fn main() {
    let plan = QmcPlan::new(Rank1Rule::kuo(4096, 2).unwrap(), 32, 42, 0).unwrap();
    let packages: Vec<_> = plan.packages(1024).unwrap().collect();

    let mut serial = QmcAccumulator::new(plan.clone(), 2).unwrap();
    for &work in &packages {
        serial.merge(evaluate(&plan, work)).unwrap();
    }

    // The application owns the thread count, dispatch and synchronization.
    // Changing it never changes package boundaries or randomizations.
    let workers = 4;
    let returns = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|worker| {
                let plan = &plan;
                let packages = &packages;
                scope.spawn(move || {
                    packages
                        .iter()
                        .skip(worker)
                        .step_by(workers)
                        .map(|&work| evaluate(plan, work))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>()
    });
    let mut parallel = QmcAccumulator::new(plan, 2).unwrap();
    for partial in returns.into_iter().rev() {
        parallel.merge(partial).unwrap();
    }
    assert!(parallel.is_complete());
    let result = parallel.estimate().unwrap();
    assert_eq!(result, serial.estimate().unwrap());
    println!(
        "Integral: {:.9} ± {:.2e}; complementary integral: {:.9} ± {:.2e}",
        result.mean[0], result.standard_error[0], result.mean[1], result.standard_error[1]
    );
    println!(
        "{} independent shifts, {} lattice evaluations",
        result.complete_shifts, result.used_points
    );
}
