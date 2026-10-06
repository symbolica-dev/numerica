use numerica::{
    domains::rational::Rational,
    numerical_integration::qmc::{
        Korobov2, PublishedLattice, QmcAccumulator, QmcError, QmcPartial, QmcPlan, QmcWorkPackage,
        Rank1Rule,
    },
};

#[test]
fn korobov2_endpoints_reflection_and_derivative() {
    for x in [0.0, 0.5, 1.0] {
        assert_eq!(Korobov2::map(x), x);
    }
    assert_eq!(Korobov2::map(0.25), Rational::from((53, 512)).to_f64());
    assert_eq!(
        Korobov2::jacobian(0.25),
        Rational::from((135, 128)).to_f64()
    );
    assert_eq!(Korobov2::jacobian(0.5), Rational::from((15, 8)).to_f64());
    assert_eq!(Korobov2::jacobian(0.0), 0.0);
    assert_eq!(Korobov2::jacobian(1.0), 0.0);
    let mut previous = 0.0;
    for i in 0..=1024 {
        let x = i as f64 / 1024.0;
        let mapped = Korobov2::map(x);
        assert!((previous..=1.0).contains(&mapped));
        assert_eq!(mapped + Korobov2::map(1.0 - x), 1.0);
        assert_eq!(Korobov2::jacobian(x), Korobov2::jacobian(1.0 - x));
        previous = mapped;
    }
    for x in [0.125, 0.25, 0.5, 0.75, 0.875] {
        let h = 1e-5;
        let difference = (Korobov2::map(x + h) - Korobov2::map(x - h)) / (2.0 * h);
        assert!((difference - Korobov2::jacobian(x)).abs() < 6e-10);
    }
    let near_zero = 2.0_f64.powi(-30);
    assert!(Korobov2::map(near_zero) > 0.0);
    assert_eq!(
        Korobov2::map(1.0 - near_zero),
        1.0 - Korobov2::map(near_zero)
    );
}

#[test]
fn korobov2_multidimensional_weight_and_invalid_coordinates() {
    assert_eq!(Korobov2::transform_in_place(&mut []).unwrap(), 1.0);
    let mut point = [0.25, 0.5, 0.75];
    let expected = Rational::from((135, 128)).pow(2) * Rational::from((15, 8));
    assert_eq!(
        Korobov2::transform_in_place(&mut point).unwrap(),
        expected.to_f64()
    );
    assert_eq!(point, [53.0 / 512.0, 0.5, 459.0 / 512.0]);
    for bad in [-0.1, 1.1, f64::NAN, f64::NEG_INFINITY, f64::INFINITY] {
        let mut point = [0.25, bad];
        assert!(matches!(
            Korobov2::transform_in_place(&mut point),
            Err(QmcError::InvalidWork(_))
        ));
        assert_eq!(point[0], 0.25);
    }
}

#[test]
fn korobov2_rejects_lost_interior_range_but_preserves_exact_endpoints() {
    for mut point in [vec![1e-160], vec![1e-170], vec![0.001; 100]] {
        assert_eq!(
            Korobov2::transform_in_place(&mut point),
            Err(QmcError::NumericUnderflow)
        );
    }
    for endpoint in [0.0, 1.0] {
        let mut point = vec![0.001; 100];
        point.push(endpoint);
        assert_eq!(Korobov2::transform_in_place(&mut point).unwrap(), 0.0);
        assert_eq!(point.last(), Some(&endpoint));
    }
    assert_eq!(
        Korobov2::transform_in_place(&mut vec![0.5; 2000]),
        Err(QmcError::NumericOverflow)
    );
}

#[test]
fn korobov2_attenuation_guard_is_not_hidden_by_amplifying_axes() {
    let mut point = vec![0.5; 64];
    point.extend(vec![0.001; 69]);
    let product = |xs: &[f64]| xs.iter().map(|&x| Korobov2::jacobian(x)).product::<f64>();
    assert!(product(&point).is_normal());
    assert!(product(&point[64..]).is_subnormal());
    let mut reversed = point.iter().rev().copied().collect::<Vec<_>>();
    for point in [&mut point, &mut reversed] {
        assert_eq!(
            Korobov2::transform_in_place(point),
            Err(QmcError::NumericUnderflow)
        );
    }
}

fn evaluate(
    plan: &QmcPlan,
    work: QmcWorkPackage,
    outputs: usize,
    f: impl Fn(&[f64], f64) -> Vec<f64>,
) -> QmcPartial {
    // The caller owns each point buffer and generates/transforms it on its worker.
    let mut point = vec![0.0; plan.dimension()];
    let mut partial = QmcPartial::new(plan, work, outputs).unwrap();
    for index in work.start()..work.start() + work.point_count() {
        plan.point(index, &mut point).unwrap();
        let weight = Korobov2::transform_in_place(&mut point).unwrap();
        partial.push(&f(&point, weight)).unwrap();
    }
    partial.finish().unwrap()
}

#[test]
fn korobov2_normalization_and_transformed_moments() {
    let plan = QmcPlan::new(Rank1Rule::new(4096, vec![1]).unwrap(), 8, 17, 0).unwrap();
    let mut accumulator = QmcAccumulator::new(plan.clone(), 4).unwrap();
    for work in plan.packages(257).unwrap() {
        accumulator
            .merge(evaluate(&plan, work, 4, |x, weight| {
                [0, 1, 2, 5]
                    .into_iter()
                    .map(|power| weight * x[0].powi(power))
                    .collect()
            }))
            .unwrap();
    }
    let estimate = accumulator.estimate().unwrap();
    for (i, denominator) in [1, 2, 3, 6].into_iter().enumerate() {
        let expected = Rational::from((1, denominator)).to_f64();
        assert!((estimate.mean[i] - expected).abs() < 2e-10, "{estimate:?}");
    }
}

#[test]
fn korobov2_worker_local_evaluation_preserves_vectors_and_covariance() {
    let plan = QmcPlan::new(
        Rank1Rule::published(PublishedLattice::HkknAlpha3, 4096, 3).unwrap(),
        16,
        51,
        0,
    )
    .unwrap();
    let work: Vec<_> = plan.packages(257).unwrap().collect();
    let f = |x: &[f64], weight| vec![weight, weight * x[0], weight * x.iter().product::<f64>()];
    let mut serial = QmcAccumulator::new(plan.clone(), 3).unwrap();
    for &package in &work {
        serial.merge(evaluate(&plan, package, 3, f)).unwrap();
    }
    let expected = serial.estimate().unwrap();
    for (i, denominator) in [1, 2, 8].into_iter().enumerate() {
        let exact = Rational::from((1, denominator)).to_f64();
        assert!((expected.mean[i] - exact).abs() < 1e-4, "{expected:?}");
    }
    for workers in [2, 4] {
        let returns = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..workers)
                .map(|worker| {
                    let plan = &plan;
                    let work = &work;
                    scope.spawn(move || {
                        work.iter()
                            .skip(worker)
                            .step_by(workers)
                            .map(|&package| evaluate(plan, package, 3, f))
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().unwrap())
                .collect::<Vec<_>>()
        });
        let mut parallel = QmcAccumulator::new(plan.clone(), 3).unwrap();
        for partial in returns.into_iter().rev() {
            parallel.merge(partial).unwrap();
        }
        assert_eq!(parallel.estimate().unwrap(), expected);
    }
}
