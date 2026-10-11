use numerica::domains::{
    float::{
        DoubleFloat, ErrorPropagatingFloat as E, Float, FloatLike, Real, RealLike, SingleFloat,
    },
    rational::Rational,
};

fn exact(value: Float) -> E<Float> {
    E::new_with_accuracy(value, f64::INFINITY)
}

#[test]
fn scale_down_and_back_up_does_not_erase_uncertainty() {
    for bits in [512, 1024, 2048] {
        let seed = E::new(Float::with_val(bits, 1), 150.);
        let down = seed * exact(Float::with_val(bits, 1e-200));
        assert!(down.get_absolute_error().is_nan());
        let restored = down * exact(Float::with_val(bits, 1e200));
        assert!((restored.to_f64() - 1.).abs() < 1e-14);
        assert!(restored.get_absolute_error().is_nan());
        assert!(restored.get_relative_error().is_nan());
        assert!(restored.get_precision().unwrap().is_nan());
    }
}

#[test]
fn finite_requested_errors_and_native_constants_refuse_range_loss() {
    let context = E::new(Float::with_val(2048, 1), 20.);
    for number in [
        context.zero(),
        context.one(),
        context.from_i64(2),
        context.from_usize(3),
        context.pi(),
        context.e(),
    ] {
        assert!(number.get_absolute_error().is_nan());
    }
    assert!(
        E::new(Float::with_val(512, 1e-200), 200.)
            .get_absolute_error()
            .is_nan()
    );
    assert!(
        E::new_with_accuracy(Float::with_val(512, 1), 400.)
            .get_absolute_error()
            .is_nan()
    );
    assert!(
        E::new_with_accuracy(Float::with_val(512, 1), -400.)
            .get_absolute_error()
            .is_nan()
    );
    assert_eq!(exact(Float::with_val(2048, 1)).get_absolute_error(), 0.);
}

#[test]
fn nonzero_native_scales_are_not_mistaken_for_zero() {
    let small = Float::with_val(2048, 1e-200).pow(2);
    let large = Float::with_val(2048, 1e200).pow(2);
    assert!(!small.is_zero());
    assert_eq!(small.to_f64(), 0.);
    assert!(large.is_finite());
    assert!(!large.to_f64().is_finite());
    for value in [small, large] {
        assert!(E::new(value.clone(), 20.).get_absolute_error().is_nan());
        let tracked = E::new_with_accuracy(value, 20.);
        for transformed in [
            tracked.clone() * tracked.one(),
            tracked.inv(),
            tracked.pow(2),
            tracked.powf(&exact(Float::with_val(2048, 2))),
            tracked.log(),
        ] {
            assert!(transformed.get_absolute_error().is_nan());
        }
    }
}

#[test]
fn reciprocal_and_power_sensitivities_do_not_silently_underflow() {
    let large = E::new_with_accuracy(Float::with_val(512, 1e200), 100.);
    assert!(large.inv().get_absolute_error().is_nan());
    let tiny = E::new_with_accuracy(Float::with_val(512, 1e-200), 210.);
    assert!(tiny.pow(2).get_absolute_error().is_nan());
    assert!(
        tiny.powf(&exact(Float::with_val(512, 2)))
            .get_absolute_error()
            .is_nan()
    );
    let recovered = tiny.clone() * tiny * exact(Float::with_val(512, 1e200));
    assert!(recovered.to_f64() > 0.);
    assert!(recovered.get_absolute_error().is_nan());
}

#[test]
fn invalid_tracking_is_sticky_through_arithmetic_and_truncation() {
    let bad = E::new_with_accuracy(1f64, 400.);
    let exact_zero = E::new_with_accuracy(0f64, f64::INFINITY);
    let exact_one = E::new_with_accuracy(1f64, f64::INFINITY);
    for result in [
        bad.truncate(),
        bad * exact_zero,
        bad + exact_one,
        bad.pow(2),
        bad.powf(&exact_one),
        -bad,
        bad.norm(),
    ] {
        assert!(result.get_absolute_error().is_nan());
    }
    let zero_bad = E::new_with_accuracy(0f64, 400.);
    assert!(zero_bad.powf(&exact_one).get_absolute_error().is_nan());
}

#[test]
fn precision_reset_keeps_its_explicit_native_semantics() {
    let mut value = E::new_with_accuracy(Float::with_val(2048, 1), 400.);
    assert!(value.get_absolute_error().is_nan());
    value.set_precision(128);
    assert_eq!(value.get_absolute_error(), 2f64.powi(-128));
    value.set_precision(2048);
    assert!(value.get_absolute_error().is_nan());
}

#[test]
fn ordinary_tracking_and_signed_rational_scaling_remain_useful() {
    let x = E::new_with_accuracy(2f64, 6.);
    let y = E::new_with_accuracy(3f64, 6.);
    assert!(((x * y).get_absolute_error() - 5e-6).abs() < 1e-20);
    let rational_context = E::new(Float::with_val(128, 2), 20.);
    assert!((rational_context.clone() * Rational::from(-2)).get_absolute_error() > 0.);
    let huge = Rational::from(10).pow(400);
    assert!(
        (rational_context.clone() * huge.clone())
            .get_absolute_error()
            .is_nan()
    );
    assert!(
        (rational_context * huge.inv())
            .get_absolute_error()
            .is_nan()
    );
    assert!(x.from_rational(&Rational::from(-2)).get_absolute_error() > 0.);
    let high = E::new(DoubleFloat::from(2.), 20.);
    assert!((high * high).get_absolute_error().is_finite());
    let multiprecision = E::new(Float::with_val(128, 2), 20.);
    assert!(
        (multiprecision.clone() * multiprecision)
            .get_absolute_error()
            .is_finite()
    );
    let zero = E::new_with_accuracy(0f64, f64::INFINITY);
    let one = E::new_with_accuracy(1f64, f64::INFINITY);
    assert_eq!((zero * one).get_absolute_error(), 0.);
}

#[test]
fn native_transcendental_sensitivities_survive_rounded_scales() {
    let x = E::new_with_accuracy(Float::with_val(512, 100), 20.);
    let result = x.tanh();
    assert_eq!(result.to_f64(), 1.);
    assert!(result.get_absolute_error().is_finite());
    assert!(result.get_absolute_error() > 1e-110);
    let amplified = result * exact(Float::with_val(512, 1e100));
    assert!(amplified.get_absolute_error() > 1e-10);
    for x in [1000., -1000.] {
        assert!(
            E::new_with_accuracy(Float::with_val(512, x), 20.)
                .tanh()
                .get_absolute_error()
                .is_nan()
        );
        assert!(
            E::new_with_accuracy(x, 20.)
                .tanh()
                .get_absolute_error()
                .is_nan()
        );
    }
    let base = Float::with_val(512, 1) + Float::with_val(512, 1e-100);
    assert_eq!(base.to_f64(), 1.);
    assert!(!base.log().is_zero());
    let exponent = E::new_with_accuracy(Float::with_val(512, 2), 20.);
    let power = exact(base).powf(&exponent);
    assert!(power.get_absolute_error().is_finite());
    assert!(power.get_absolute_error() > 1e-125);
    assert!((power * exact(Float::with_val(512, 1e100))).get_absolute_error() > 1e-25);
    let exact_zero = exact(Float::with_val(512, 0));
    assert!(
        !exact_zero
            .powf(&exact(Float::with_val(512, -1)))
            .to_f64()
            .is_finite()
    );
}

#[test]
fn native_operation_underflow_is_not_exact_zero_sensitivity() {
    let tiny = E::new_with_accuracy(1e-200f64, 200.);
    for result in [
        tiny.pow(2),
        tiny.powf(&E::new_with_accuracy(2., f64::INFINITY)),
        E::new_with_accuracy(-1000., 20.).exp(),
        E::new_with_accuracy(f64::INFINITY, 20.).inv(),
    ] {
        assert_eq!(result.to_f64(), 0.);
        assert!(result.get_absolute_error().is_nan());
        assert!(
            (result * E::new_with_accuracy(1e200, f64::INFINITY))
                .get_absolute_error()
                .is_nan()
        );
    }
}
