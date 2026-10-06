use numerica::numerical_integration::qmc::{
    Korobov3, PublishedLattice, QmcAccumulator, QmcPartial, QmcPlan, Rank1Rule, RuleSource,
};

#[test]
fn published_data_and_bounds_match_the_attributed_catalogues() {
    for (catalogue, source, prefix) in [
        (
            PublishedLattice::Kuo38005,
            RuleSource::Kuo38005,
            [1, 433461, 103659, 481853, 186513],
        ),
        (
            PublishedLattice::Kuo39101,
            RuleSource::Kuo39101,
            [1, 182667, 279195, 223491, 205755],
        ),
        (
            PublishedLattice::HkknAlpha3,
            RuleSource::HkknAlpha3,
            [1, 364981, 245389, 97823, 488939],
        ),
    ] {
        let largest =
            Rank1Rule::published(catalogue, catalogue.max_points(), catalogue.max_dimension())
                .unwrap();
        assert_eq!(largest.source(), source);
        assert_eq!(&largest.generator()[..5], prefix);
        assert_eq!(largest.generator().len(), catalogue.max_dimension());
        assert!(
            largest
                .generator()
                .iter()
                .all(|z| z % 2 == 1 && *z < catalogue.max_points())
        );
        for n in [catalogue.min_points(), 8192, catalogue.max_points()] {
            let rule = Rank1Rule::published(catalogue, n, 5).unwrap();
            assert_eq!(rule.generator(), prefix.map(|z| z % n));
        }
        for n in [
            0,
            1,
            catalogue.min_points() - 1,
            8193,
            catalogue.max_points() * 2,
        ] {
            assert!(Rank1Rule::published(catalogue, n, 1).is_err());
        }
        assert!(Rank1Rule::published(catalogue, 8192, 0).is_err());
        assert!(Rank1Rule::published(catalogue, 8192, catalogue.max_dimension() + 1).is_err());
    }
    assert_eq!(
        Rank1Rule::kuo(8192, 9).unwrap(),
        Rank1Rule::published(PublishedLattice::Kuo33002, 8192, 9).unwrap()
    );
}

#[cfg(feature = "serde")]
#[test]
fn published_provenance_is_verified_on_restore_and_old_kuo_is_unchanged() {
    let old = Rank1Rule::kuo(8192, 5).unwrap();
    assert_eq!(serde_json::to_value(&old).unwrap()["source"], "Kuo33002");
    for catalogue in [
        PublishedLattice::HkknAlpha3,
        PublishedLattice::Kuo38005,
        PublishedLattice::Kuo39101,
    ] {
        let rule = Rank1Rule::published(catalogue, 8192, 5).unwrap();
        let value = serde_json::to_value(&rule).unwrap();
        assert_eq!(
            serde_json::from_value::<Rank1Rule>(value.clone()).unwrap(),
            rule
        );
        let mut wrong = value.clone();
        wrong["generator"][1] = serde_json::json!(3);
        assert!(serde_json::from_value::<Rank1Rule>(wrong).is_err());
        let mut wrong = value;
        wrong["source"] = serde_json::json!("Kuo33002");
        assert!(serde_json::from_value::<Rank1Rule>(wrong).is_err());
    }
}

#[test]
fn published_rule_avoids_periodized_five_dimensional_constant_plateau() {
    // Physical constant1 has exact integral1. Periodization introduces a
    // five-way Jacobian product, which is not controlled by order3 weights.
    // This regression exercises the actual published data, transform, point
    // packages and shift covariance, independently of any Feynman kernel.
    let integrate = |catalogue| {
        let plan = QmcPlan::new(
            Rank1Rule::published(catalogue, 8192, 5).unwrap(),
            32,
            20261005,
            0,
        )
        .unwrap();
        let mut accumulator = QmcAccumulator::new(plan.clone(), 1).unwrap();
        let mut returns = Vec::new();
        for work in plan.packages(1024).unwrap() {
            let mut points = vec![0.; work.point_count() as usize * 5];
            plan.fill_points(work, &mut points).unwrap();
            let mut partial = QmcPartial::new(&plan, work, 1).unwrap();
            for point in points.as_chunks_mut::<5>().0 {
                partial
                    .push(&[Korobov3::transform_in_place(point).unwrap()])
                    .unwrap();
            }
            returns.push(partial.finish().unwrap());
        }
        let mut reverse = accumulator.clone();
        for partial in &returns {
            accumulator.merge(partial.clone()).unwrap();
        }
        for partial in returns.into_iter().rev() {
            reverse.merge(partial).unwrap();
        }
        let estimate = accumulator.estimate().unwrap();
        assert_eq!(estimate, reverse.estimate().unwrap());
        estimate
    };
    let historical = integrate(PublishedLattice::Kuo33002);
    let alternative = integrate(PublishedLattice::HkknAlpha3);
    assert!(historical.standard_error[0] > 1e-4);
    assert!(alternative.standard_error[0] < 2e-6);
    assert!((alternative.mean[0] - 1.).abs() < 5. * alternative.standard_error[0]);
}
