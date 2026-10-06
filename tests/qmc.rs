use numerica::numerical_integration::qmc::{
    KUO_MAX_DIMENSION, KUO_MAX_POINTS, Korobov3, QmcAccumulator, QmcError, QmcEstimate, QmcPartial,
    QmcPlan, QmcWorkPackage, Rank1Rule, RuleSource,
};

fn plan(n: u64, shifts: u32) -> QmcPlan {
    QmcPlan::new(Rank1Rule::new(n, vec![1]).unwrap(), shifts, 123, 0).unwrap()
}

fn evaluate(
    plan: &QmcPlan,
    work: QmcWorkPackage,
    outputs: usize,
    mut f: impl FnMut(u64, &[f64]) -> Vec<f64>,
) -> QmcPartial {
    let mut points = vec![0.0; work.point_count() as usize * plan.dimension()];
    plan.fill_points(work, &mut points).unwrap();
    let mut partial = QmcPartial::new(plan, work, outputs).unwrap();
    for (offset, point) in points.chunks_exact(plan.dimension()).enumerate() {
        partial
            .push(&f(work.start() + offset as u64, point))
            .unwrap();
    }
    partial.finish().unwrap()
}

fn integrate(
    plan: QmcPlan,
    package: u64,
    outputs: usize,
    mut f: impl FnMut(u64, &[f64]) -> Vec<f64>,
) -> QmcAccumulator {
    let mut accumulator = QmcAccumulator::new(plan.clone(), outputs).unwrap();
    for work in plan.packages(package).unwrap() {
        accumulator
            .merge(evaluate(&plan, work, outputs, &mut f))
            .unwrap();
    }
    accumulator
}

#[test]
fn catalog_has_published_vector_and_construction_limits() {
    let rule = Rank1Rule::kuo(KUO_MAX_POINTS, KUO_MAX_DIMENSION).unwrap();
    assert_eq!(rule.source(), RuleSource::Kuo33002);
    assert_eq!(
        &rule.generator()[..10],
        &[
            1, 182667, 213731, 255351, 96013, 116671, 479315, 424089, 271103, 464421
        ]
    );
    assert_eq!(rule.generator().len(), 9125);
    assert!(
        rule.generator()
            .iter()
            .all(|&x| x & 1 == 1 && x < KUO_MAX_POINTS)
    );
    for n in [0, 16, 1000, 1023, 1025, KUO_MAX_POINTS + 1, 1 << 21] {
        assert!(Rank1Rule::kuo(n, 2).is_err());
    }
    assert!(Rank1Rule::kuo(1024, 0).is_err());
    assert!(Rank1Rule::kuo(1024, 9126).is_err());
    assert_eq!(
        Rank1Rule::kuo(4096, 10).unwrap().generator(),
        rule.generator()[..10]
            .iter()
            .map(|z| z % 4096)
            .collect::<Vec<_>>()
    );
}

#[test]
fn supplied_vectors_validate_and_normalize() {
    assert_eq!(Rank1Rule::new(7, vec![8, 10]).unwrap().generator(), &[1, 3]);
    for (n, z) in [
        (1, vec![1]),
        (1 << 54, vec![1]),
        (7, vec![]),
        (7, vec![0]),
        (8, vec![2]),
    ] {
        assert!(Rank1Rule::new(n, z).is_err());
    }
    assert!(Rank1Rule::new(12, vec![1, 5, 7, 11]).is_ok());
}

#[test]
fn known_rank_one_points_and_exact_supplied_shifts() {
    let plan = QmcPlan::with_shifts(
        Rank1Rule::new(7, vec![1, 3]).unwrap(),
        vec![vec![0.0, 0.0], vec![0.25, 0.75]],
    )
    .unwrap();
    let mut point = [0.0; 2];
    plan.point(4, &mut point).unwrap();
    assert_eq!(point, [4.0 / 7.0, 5.0 / 7.0]);
    plan.point(7, &mut point).unwrap();
    assert_eq!(point, [0.25, 0.75]);
    plan.point(13, &mut point).unwrap();
    assert!((point[0] - (6.0 / 7.0 + 0.25 - 1.0)).abs() < 1e-15);
    assert!((point[1] - (4.0 / 7.0 + 0.75 - 1.0)).abs() < 1e-15);
    assert!(plan.point(14, &mut point).is_err());
    assert!(plan.point(0, &mut [0.0]).is_err());
}

#[test]
fn modular_products_do_not_overflow() {
    let n = (1u64 << 53) - 1;
    let plan = QmcPlan::with_shifts(
        Rank1Rule::new(n, vec![n - 1]).unwrap(),
        vec![vec![0.0], vec![0.0]],
    )
    .unwrap();
    let mut point = [0.0];
    plan.point(n - 1, &mut point).unwrap();
    assert_eq!(point[0], 1.0 / n as f64);
    let n = 1u64 << 53;
    let plan = QmcPlan::with_shifts(
        Rank1Rule::new(n, vec![n - 1]).unwrap(),
        vec![vec![0.0], vec![0.0]],
    )
    .unwrap();
    plan.point(n - 1, &mut point).unwrap();
    assert_eq!(point[0], 1.0 / n as f64);
}

#[test]
fn periodic_shift_preserves_small_residues_at_maximum_modulus() {
    let n = 1u64 << 53;
    let shifts = [1, 1 << 51, 1 << 52, 3 << 51, n - 1];
    let plan = QmcPlan::with_shifts(
        Rank1Rule::new(n, vec![1]).unwrap(),
        shifts.iter().map(|s| vec![*s as f64 / n as f64]).collect(),
    )
    .unwrap();
    let mut point = [0.0];
    for (shift_id, &shift) in shifts.iter().enumerate() {
        for index in [0, 1, (1 << 51) - 1, 1 << 51, (1 << 51) + 1, n - 2, n - 1] {
            plan.point(shift_id as u64 * n + index, &mut point).unwrap();
            assert_eq!(point[0], ((index + shift) % n) as f64 / n as f64);
        }
    }
    // Arbitrary supplied shifts can carry information finer than the lattice
    // spacing. Keep the tiny wrapped residue instead of first rounding to one.
    let offset = (1.0 / n as f64).next_up();
    let plan =
        QmcPlan::with_shifts(Rank1Rule::new(n, vec![1]).unwrap(), vec![vec![offset]]).unwrap();
    plan.point(n - 1, &mut point).unwrap();
    assert_eq!(point[0], offset - 1.0 / n as f64);
    // A sum just below one may round up. It must remain below the boundary,
    // rather than spuriously wrapping from almost one to zero.
    let plan = QmcPlan::with_shifts(
        Rank1Rule::new(2, vec![1]).unwrap(),
        vec![vec![0.5_f64.next_down()]],
    )
    .unwrap();
    plan.point(1, &mut point).unwrap();
    assert_eq!(point[0], 1.0_f64.next_down());
}

#[test]
fn shift_prefixes_do_not_depend_on_dimension_count_or_workers() {
    let small = QmcPlan::new(Rank1Rule::new(127, vec![1, 3]).unwrap(), 3, 987, 2).unwrap();
    let large = QmcPlan::new(Rank1Rule::new(127, vec![1, 3, 5, 7, 9]).unwrap(), 8, 987, 2).unwrap();
    for (a, b) in small.shifts().iter().zip(large.shifts()) {
        assert_eq!(a, &b[..2]);
    }
    assert_eq!(
        small,
        QmcPlan::new(Rank1Rule::new(127, vec![1, 3]).unwrap(), 3, 987, 2).unwrap()
    );
    let other = QmcPlan::new(small.rule().clone(), 3, 987, 3).unwrap();
    assert_ne!(small.shifts(), other.shifts());
    assert!(
        large
            .shifts()
            .iter()
            .flatten()
            .all(|x| (0.0..1.0).contains(x))
    );
}

#[test]
fn invalid_plans_and_packages_are_rejected_without_clamping() {
    let rule = Rank1Rule::new(7, vec![1]).unwrap();
    assert!(QmcPlan::new(rule.clone(), 0, 0, 0).is_err());
    for shifts in [
        vec![vec![0.0], vec![1.0]],
        vec![vec![0.0], vec![f64::NAN]],
        vec![vec![], vec![]],
    ] {
        assert!(QmcPlan::with_shifts(rule.clone(), shifts).is_err());
    }
    assert!(QmcPlan::new(Rank1Rule::new(1 << 53, vec![1]).unwrap(), u32::MAX, 0, 0).is_err());
    let p = plan(7, 3);
    assert!(p.work(0, 0).is_err());
    assert!(p.work(20, 2).is_err());
    assert!(p.work(u64::MAX, 2).is_err());
    assert!(p.packages(0).is_err());
    assert!(p.fill_points(p.work(0, 2).unwrap(), &mut [0.0]).is_err());
    let other = QmcPlan::new(rule, 3, 123, 1).unwrap();
    assert!(
        p.fill_points(other.work(0, 2).unwrap(), &mut [0.0; 2])
            .is_err()
    );
}

#[test]
fn streamed_generation_is_identical_to_whole_array_for_every_partition() {
    let plan = QmcPlan::new(Rank1Rule::new(11, vec![1, 3, 7]).unwrap(), 5, 91, 0).unwrap();
    let mut whole = vec![0.0; plan.total_points() as usize * 3];
    plan.fill_points(plan.work(0, plan.total_points()).unwrap(), &mut whole)
        .unwrap();
    for size in [1, 4, 11, 12, 17, 55, 100] {
        let mut pieces = Vec::new();
        for work in plan.packages(size).unwrap() {
            let mut points = vec![0.0; work.point_count() as usize * 3];
            plan.fill_points(work, &mut points).unwrap();
            pieces.extend(points);
        }
        assert_eq!(whole, pieces);
    }
    assert!(whole.iter().all(|x| (0.0..1.0).contains(x)));
}

#[test]
fn uncertainty_and_covariance_use_complete_shift_means() {
    let accumulator = integrate(plan(7, 3), 5, 2, |i, _| {
        let mean = 2.0 * (i / 7 + 1) as f64;
        vec![mean, -2.0 * mean]
    });
    let estimate = accumulator.estimate().unwrap();
    assert_eq!(estimate.mean, vec![4.0, -8.0]);
    assert_eq!(estimate.complete_shifts, 3);
    assert_eq!(estimate.used_points, 21);
    assert!((estimate.standard_error[0] - (4.0_f64 / 3.0).sqrt()).abs() < 1e-15);
    assert_eq!(
        estimate.covariance_of_mean,
        vec![4.0 / 3.0, -8.0 / 3.0, -8.0 / 3.0, 16.0 / 3.0]
    );
    // Positive pointwise variance but identical replica means: QMC error is zero.
    let identical = integrate(plan(4, 4), 3, 1, |i, _| vec![(i % 2 * 2) as f64]);
    assert_eq!(identical.estimate().unwrap().standard_error, vec![0.0]);
}

#[test]
fn covariance_retains_small_variations_on_a_large_absolute_offset() {
    let estimate =
        QmcEstimate::from_shift_means(&[vec![1e16, -1e16], vec![1e16 + 2.0, -1e16 - 2.0]]).unwrap();
    assert_eq!(estimate.covariance_of_mean, vec![1.0, -1.0, -1.0, 1.0]);
    assert_eq!(estimate.standard_error, vec![1.0, 1.0]);
    // A representable mean need not require summing large absolute values.
    let constant = QmcEstimate::from_shift_means(&[vec![1e308], vec![1e308]]).unwrap();
    assert_eq!(constant.mean, vec![1e308]);
    assert_eq!(constant.standard_error, vec![0.0]);
}

#[test]
fn overlap_gaps_duplicates_and_incomplete_returns_are_detected() {
    let p = plan(7, 3);
    let mut accumulator = QmcAccumulator::new(p.clone(), 1).unwrap();
    let first = evaluate(&p, p.work(0, 5).unwrap(), 1, |_, _| vec![1.0]);
    accumulator.merge(first.clone()).unwrap();
    assert_eq!(
        accumulator.merge(first).unwrap_err(),
        QmcError::OverlappingWork
    );
    for range in [(3, 4), (0, 7), (4, 8)] {
        let partial = evaluate(&p, p.work(range.0, range.1).unwrap(), 1, |_, _| vec![1.0]);
        assert_eq!(
            accumulator.merge(partial).unwrap_err(),
            QmcError::OverlappingWork
        );
    }
    accumulator
        .merge(evaluate(&p, p.work(14, 7).unwrap(), 1, |_, _| vec![2.0]))
        .unwrap();
    assert_eq!(accumulator.missing_ranges(), vec![5..14]);
    assert!(!accumulator.is_complete());
    assert_eq!(accumulator.shift_estimates().unwrap()[0].shift, 2);
    assert!(matches!(
        accumulator.estimate(),
        Err(QmcError::InsufficientShifts { complete: 1 })
    ));
    let incomplete = QmcPartial::new(&p, p.work(5, 9).unwrap(), 1).unwrap();
    assert!(incomplete.clone().finish().is_err());
    assert!(accumulator.merge(incomplete).is_err());
    accumulator
        .merge(evaluate(&p, p.work(5, 9).unwrap(), 1, |_, _| vec![1.0]))
        .unwrap();
    assert!(accumulator.is_complete());
    assert!(accumulator.missing_ranges().is_empty());
}

#[test]
fn cross_rule_and_cross_randomization_partials_cannot_merge() {
    let p = plan(7, 3);
    let mut accumulator = QmcAccumulator::new(p.clone(), 1).unwrap();
    for other in [
        plan(11, 3),
        QmcPlan::new(p.rule().clone(), 3, 123, 1).unwrap(),
        QmcPlan::new(Rank1Rule::new(7, vec![3]).unwrap(), 3, 123, 0).unwrap(),
    ] {
        let partial = evaluate(&other, other.work(0, 4).unwrap(), 1, |_, _| vec![1.0]);
        assert!(accumulator.merge(partial).is_err());
    }
    assert_eq!(accumulator.completed_points(), 0);
}

#[test]
fn partial_rejects_bad_values_and_overflow() {
    let p = plan(7, 3);
    let mut partial = QmcPartial::new(&p, p.work(0, 2).unwrap(), 2).unwrap();
    assert!(partial.push(&[1.0]).is_err());
    assert!(partial.push(&[1.0, f64::NAN]).is_err());
    assert_eq!(partial.processed_points(), 0);
    partial.push(&[1.0, 2.0]).unwrap();
    partial.push(&[2.0, 3.0]).unwrap();
    assert!(partial.push(&[1.0, 2.0]).is_err());
    partial.finish().unwrap();
    let mut overflow = QmcPartial::new(&p, p.work(0, 2).unwrap(), 1).unwrap();
    overflow.push(&[f64::MAX]).unwrap();
    assert_eq!(
        overflow.push(&[f64::MAX]).unwrap_err(),
        QmcError::NumericOverflow
    );
    assert!(overflow.finish().is_err());
}

#[test]
fn accepted_package_layout_is_available_in_index_order() {
    let p = plan(7, 3);
    let packages: Vec<_> = p.packages(5).unwrap().collect();
    let mut accumulator = QmcAccumulator::new(p.clone(), 1).unwrap();
    for &work in packages.iter().rev() {
        accumulator
            .merge(evaluate(&p, work, 1, |_, _| vec![1.0]))
            .unwrap();
    }
    assert_eq!(
        accumulator.completed_work_packages().collect::<Vec<_>>(),
        packages
    );
}

#[test]
fn compensated_sums_preserve_large_cancellations() {
    let accumulator = integrate(plan(4, 3), 3, 1, |i, _| {
        vec![[1e16, 1.0, -1e16, 3.0][(i % 4) as usize]]
    });
    assert_eq!(accumulator.estimate().unwrap().mean, vec![1.0]);
    assert_eq!(accumulator.estimate().unwrap().standard_error, vec![0.0]);
}

#[test]
fn completed_shifts_exclude_partially_evaluated_lattices() {
    let p = plan(7, 4);
    let mut accumulator = QmcAccumulator::new(p.clone(), 1).unwrap();
    accumulator
        .merge(evaluate(&p, p.work(0, 16).unwrap(), 1, |i, _| {
            vec![(i / 7) as f64]
        }))
        .unwrap();
    let estimate = accumulator.estimate().unwrap();
    assert_eq!(estimate.mean, vec![0.5]);
    assert_eq!(estimate.standard_error, vec![0.5]);
    assert_eq!(estimate.complete_shifts, 2);
    assert_eq!(estimate.completed_points, 16);
    assert_eq!(estimate.used_points, 14);
}

#[test]
fn a_single_shift_generates_points_but_cannot_report_uncertainty() {
    let accumulator = integrate(plan(7, 1), 7, 1, |_, x| vec![x[0]]);
    assert!(accumulator.is_complete());
    assert_eq!(accumulator.shift_estimates().unwrap().len(), 1);
    assert!(matches!(
        accumulator.estimate(),
        Err(QmcError::InsufficientShifts { complete: 1 })
    ));
}

#[test]
fn caller_threads_and_arrival_order_preserve_exact_result() {
    let p = plan(127, 8);
    let work: Vec<_> = p.packages(53).unwrap().collect();
    let f = |_: u64, x: &[f64]| vec![x[0].exp(), x[0] * x[0]];
    let serial = integrate(p.clone(), 53, 2, f).estimate().unwrap();
    for workers in [1, 2, 4] {
        let results = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..workers)
                .map(|worker| {
                    let p = &p;
                    let work = &work;
                    scope.spawn(move || {
                        work.iter()
                            .skip(worker)
                            .step_by(workers)
                            .map(|&package| evaluate(p, package, 2, f))
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().unwrap())
                .collect::<Vec<_>>()
        });
        let mut accumulator = QmcAccumulator::new(p.clone(), 2).unwrap();
        for partial in results.into_iter().rev() {
            accumulator.merge(partial).unwrap();
        }
        assert_eq!(serial, accumulator.estimate().unwrap());
    }
}

#[test]
fn korobov_map_symmetry_normalization_and_domain() {
    assert_eq!(Korobov3::map(0.0), 0.0);
    assert_eq!(Korobov3::map(1.0), 1.0);
    assert_eq!(Korobov3::map(0.5), 0.5);
    assert_eq!(Korobov3::jacobian(0.0), 0.0);
    assert_eq!(Korobov3::jacobian(1.0), 0.0);
    let mut integral = 0.0;
    for i in 0..10000 {
        let x = (i as f64 + 0.5) / 10000.0;
        integral += Korobov3::jacobian(x) / 10000.0;
        assert!((Korobov3::map(x) + Korobov3::map(1.0 - x) - 1.0).abs() < 1e-14);
        assert!((0.0..=1.0).contains(&Korobov3::map(x)));
    }
    assert!((integral - 1.0).abs() < 1e-12);
    assert_eq!(Korobov3::transform_in_place(&mut []).unwrap(), 1.0);
    let mut bad = [0.25, f64::NAN];
    assert!(Korobov3::transform_in_place(&mut bad).is_err());
    assert_eq!(bad[0], 0.25);
}

#[test]
fn periodized_polynomial_integral_matches_analytic_value() {
    let p = QmcPlan::new(Rank1Rule::kuo(4096, 3).unwrap(), 16, 51, 0).unwrap();
    let result = integrate(p, 4096, 1, |_, point| {
        let mut x = point.to_vec();
        let w = Korobov3::transform_in_place(&mut x).unwrap();
        vec![w * (x[0] * x[0] + x[1] * x[2])]
    })
    .estimate()
    .unwrap();
    let residual = (result.mean[0] - 7.0 / 12.0).abs();
    assert!(residual < 1e-4, "{result:?}");
    assert!(
        residual < 5.0 * result.standard_error[0] + 1e-12,
        "{result:?}"
    );
}

#[test]
fn periodization_rejects_interior_underflow_without_confusing_exact_endpoints() {
    // Every one-dimensional Jacobian is positive and representable. Their
    // product is about 3e-686, so the old transform silently returned zero.
    let mut point = vec![0.001; 100];
    assert!(point.iter().all(|x| Korobov3::jacobian(*x) > 0.0));
    assert_eq!(
        point
            .iter()
            .map(|x| Korobov3::jacobian(*x))
            .product::<f64>(),
        0.0
    );
    assert_eq!(
        Korobov3::transform_in_place(&mut point),
        Err(QmcError::NumericUnderflow)
    );
    // A single Jacobian can underflow for a strictly interior coordinate too.
    assert_eq!(
        Korobov3::transform_in_place(&mut [1e-120]),
        Err(QmcError::NumericUnderflow)
    );
    for endpoint in [0.0, 1.0] {
        let mut point = vec![0.001; 100];
        point.push(endpoint);
        assert_eq!(Korobov3::transform_in_place(&mut point).unwrap(), 0.0);
        assert_eq!(point.last(), Some(&endpoint));
    }
}

#[test]
fn periodization_rejects_lossy_attenuation_even_when_amplification_recovers_range() {
    let mut point = vec![0.5; 54];
    point.extend(vec![0.001; 46]);
    // An amplification-first product remains normal (~1e-297). Attenuation
    // first enters the subnormal range and loses accuracy before recovering.
    let amplification_first = point
        .iter()
        .map(|x| Korobov3::jacobian(*x))
        .product::<f64>();
    assert!(amplification_first.is_normal());
    let small_product = vec![0.001; 46]
        .iter()
        .map(|x| Korobov3::jacobian(*x))
        .product::<f64>();
    assert!(small_product.is_subnormal());
    let mut reversed = point.iter().rev().copied().collect::<Vec<_>>();
    assert_eq!(
        Korobov3::transform_in_place(&mut point),
        Err(QmcError::NumericUnderflow)
    );
    assert_eq!(
        Korobov3::transform_in_place(&mut reversed),
        Err(QmcError::NumericUnderflow)
    );
}

#[test]
fn independent_seed_ensemble_has_calibrated_shift_errors() {
    let mut squared_error = 0.0;
    let mut estimated_variance = 0.0;
    let mut covered = 0;
    for seed in 0..40 {
        let p = QmcPlan::new(Rank1Rule::new(127, vec![1]).unwrap(), 8, seed, 0).unwrap();
        let estimate = integrate(p, 127, 1, |_, x| vec![x[0] * x[0]])
            .estimate()
            .unwrap();
        let error = estimate.mean[0] - 1.0 / 3.0;
        squared_error += error * error;
        estimated_variance += estimate.standard_error[0].powi(2);
        covered += usize::from(error.abs() <= 3.0 * estimate.standard_error[0]);
    }
    assert!(
        covered >= 34,
        "only {covered}/40 three-sigma intervals covered"
    );
    let ratio = squared_error / estimated_variance;
    assert!(
        (0.3..3.0).contains(&ratio),
        "variance calibration ratio {ratio}"
    );
}

#[test]
fn correlated_sector_sums_preserve_cancellation() {
    let p = plan(127, 8);
    let a = integrate(p.clone(), 53, 1, |_, x| vec![x[0].exp()])
        .shift_estimates()
        .unwrap();
    let b = integrate(p, 53, 1, |_, x| vec![-x[0].exp()])
        .shift_estimates()
        .unwrap();
    let sums: Vec<_> = a
        .iter()
        .zip(b)
        .map(|(a, b)| {
            assert_eq!(a.shift, b.shift);
            vec![a.mean[0] + b.mean[0]]
        })
        .collect();
    let total = QmcEstimate::from_shift_means(&sums).unwrap();
    assert_eq!(total.mean, vec![0.0]);
    assert_eq!(total.standard_error, vec![0.0]);
}

#[cfg(feature = "serde")]
#[test]
fn checkpoint_resume_preserves_points_partials_and_estimate() {
    let p = QmcPlan::new(Rank1Rule::kuo(1024, 2).unwrap(), 4, 53, 8).unwrap();
    let restored: QmcPlan = serde_json::from_str(&serde_json::to_string(&p).unwrap()).unwrap();
    assert_eq!(p, restored);
    let mut accumulator = QmcAccumulator::new(p.clone(), 1).unwrap();
    let packages: Vec<_> = p.packages(713).unwrap().collect();
    for &work in packages.iter().step_by(2) {
        accumulator
            .merge(evaluate(&p, work, 1, |_, x| vec![x[0] * x[1]]))
            .unwrap();
    }
    let json = serde_json::to_string(&accumulator).unwrap();
    let mut resumed: QmcAccumulator = serde_json::from_str(&json).unwrap();
    assert_eq!(accumulator.missing_ranges(), resumed.missing_ranges());
    for &work in packages.iter().skip(1).step_by(2) {
        resumed
            .merge(evaluate(&p, work, 1, |_, x| vec![x[0] * x[1]]))
            .unwrap();
    }
    let direct = integrate(p.clone(), 713, 1, |_, x| vec![x[0] * x[1]]);
    assert_eq!(resumed.estimate().unwrap(), direct.estimate().unwrap());
    let mut partial = QmcPartial::new(&p, p.work(0, 2).unwrap(), 1).unwrap();
    partial.push(&[0.125]).unwrap();
    let mut partial: QmcPartial =
        serde_json::from_str(&serde_json::to_string(&partial).unwrap()).unwrap();
    partial.push(&[0.25]).unwrap();
    partial.finish().unwrap();
}

#[cfg(feature = "serde")]
#[test]
fn malformed_checkpoints_cannot_bypass_constructor_invariants() {
    let p = plan(7, 3);
    let mut json = serde_json::to_value(&p).unwrap();
    json["rule"]["modulus"] = serde_json::json!(0);
    assert!(serde_json::from_value::<QmcPlan>(json).is_err());
    let partial = evaluate(&p, p.work(0, 4).unwrap(), 1, |_, _| vec![1.0]);
    let mut json = serde_json::to_value(&partial).unwrap();
    json["counts"][0] = serde_json::json!(500);
    assert!(serde_json::from_value::<QmcPartial>(json).is_err());
    let mut accumulator = QmcAccumulator::new(p, 1).unwrap();
    accumulator.merge(partial).unwrap();
    let mut json = serde_json::to_value(&accumulator).unwrap();
    let duplicate = json["partials"][0].clone();
    json["partials"].as_array_mut().unwrap().push(duplicate);
    assert!(serde_json::from_value::<QmcAccumulator>(json).is_err());
}
