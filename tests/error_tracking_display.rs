use numerica::domains::float::{DoubleFloat, ErrorPropagatingFloat, Float};

#[test]
fn exact_fixed_precision_values_use_native_display() {
    for value in [1., -2.5, f64::MIN_POSITIVE, f64::MAX] {
        let tracked = ErrorPropagatingFloat::new_with_accuracy(value, f64::INFINITY);
        assert_eq!(tracked.get_absolute_error(), 0.);
        assert_eq!(format!("{tracked}"), format!("{value}"));
        assert_eq!(format!("{tracked:+12.3}"), format!("{value:+12.3}"));
        // LowerExp already delegates to Display for this wrapper.
        assert_eq!(format!("{tracked:e}"), format!("{tracked}"));
    }
    let value = DoubleFloat::from(-2.5);
    let tracked = ErrorPropagatingFloat::new_with_accuracy(value, f64::INFINITY);
    assert_eq!(format!("{tracked}"), format!("{value}"));
}

#[test]
fn exact_multiprecision_values_use_native_display() {
    for bits in [53, 128, 256] {
        for input in [1., -2.5] {
            let value = Float::with_val(bits, input);
            let tracked = ErrorPropagatingFloat::new_with_accuracy(value.clone(), f64::INFINITY);
            assert_eq!(tracked.get_absolute_error(), 0.);
            assert_eq!(format!("{tracked}"), format!("{value}"));
            assert_eq!(format!("{tracked:.8}"), format!("{value:.8}"));
        }
    }
}

#[test]
fn finite_uncertainty_and_zero_keep_their_display() {
    assert_eq!(
        format!("{}", ErrorPropagatingFloat::new(1.25f64, 3.)),
        "1.250"
    );
    assert_eq!(format!("{}", ErrorPropagatingFloat::new(1.25f64, -1.)), "0");
    assert_eq!(
        format!(
            "{}",
            ErrorPropagatingFloat::new_with_accuracy(0f64, f64::INFINITY)
        ),
        "0"
    );
}
