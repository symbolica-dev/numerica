use std::panic::{AssertUnwindSafe, catch_unwind};

use numerica::{
    create_hyperdual_single_derivative,
    domains::{
        Ring,
        dual::{DualNumberStructure, HyperDual},
        finite_field::{FiniteField, FiniteFieldCore, Mersenne32, Mersenne64, Z2, Zp, Zp64},
        float::{
            Complex, DoubleFloat, ErrorPropagatingFloat, F64, Float, FloatLike, RealBall, RealLike,
            SingleFloat,
        },
        integer::{Integer, MultiPrecisionInteger, Z},
        rational::{Q, Rational},
    },
    numerical_integration::{ContinuousGrid, DiscreteGrid, Grid, MonteCarloRng, Sample},
    tensors::{
        matrix::{Matrix, MatrixError, Vector},
        sparse::SparseMatrix,
    },
};

#[test]
fn nan_preserves_precision_and_components() {
    assert!(0.0_f64.nan().unwrap().is_nan());
    assert!(F64(0.0).nan().unwrap().to_f64().is_nan());
    assert!(DoubleFloat::from(0.0).nan().unwrap().to_f64().is_nan());
    assert!(Rational::from(1).nan().is_none());
    for precision in [53, 100, 256] {
        let nan = Float::new(precision).nan().unwrap();
        assert_eq!(nan.prec(), precision);
        assert!(nan.to_f64().is_nan());
    }
    let complex = Complex::new(Float::new(80), Float::new(120)).nan().unwrap();
    assert_eq!((complex.re.prec(), complex.im.prec()), (80, 120));
    assert!(complex.re.to_f64().is_nan() && complex.im.to_f64().is_nan());
    assert!(
        Complex::new(Rational::from(1), Rational::from(0))
            .nan()
            .is_none()
    );
    let ball = RealBall::new(Float::new(80), Float::new(120))
        .nan()
        .unwrap();
    assert_eq!((ball.center.prec(), ball.radius.prec()), (80, 120));
    assert!(!ball.is_finite());
    let tracked = ErrorPropagatingFloat::new(Float::new(100), 20.0)
        .nan()
        .unwrap();
    assert!(tracked.get_num().to_f64().is_nan());
    assert!(tracked.get_absolute_error().is_nan());
    assert_eq!(tracked.get_num().prec(), 100);
    for v in wide::f64x4::splat(1.0).nan().unwrap().to_array() {
        assert!(v.is_nan());
    }
}

create_hyperdual_single_derivative!(TestDual, 2);

#[test]
fn dual_nan_preserves_shape() {
    let shape = vec![vec![0], vec![1]];
    let dual = HyperDual::from_values(shape, vec![Float::new(80), Float::new(120)]);
    let nan = dual.nan().unwrap();
    assert_eq!(nan.get_shape(), dual.get_shape());
    assert_eq!(
        nan.values.iter().map(Float::prec).collect::<Vec<_>>(),
        [80, 120]
    );
    assert!(nan.values.iter().all(|v| v.to_f64().is_nan()));
    let fixed = TestDual::<f64>::new_variable(0, 1.0).nan().unwrap();
    assert!(fixed.values.iter().all(|v| v.is_nan()));
    assert!(
        TestDual::<Rational>::new_variable(0, 1.into())
            .nan()
            .is_none()
    );
}

#[test]
fn parse_counts_significant_digits_independently_of_notation() {
    for s in [
        "123456789012345678901234567890",
        "-123456789012345678901234567890",
        "+123456789012345678901234567890",
        "1.23456789012345678901234567890e29",
        "-0.000123456789012345678901234567890E33",
        "  123456789012345678901234567890  ",
        "123456789012345678901234567890`",
    ] {
        let value = Float::parse(s, None).unwrap();
        assert_eq!(value.prec(), 100, "{s}");
        let expected: Rational = "123456789012345678901234567890"
            .parse::<Integer>()
            .unwrap()
            .into();
        let expected = if s.trim_start().starts_with('-') {
            -expected
        } else {
            expected
        };
        assert_eq!(value.try_to_rational(), Some(expected), "{s}");
    }
    for s in ["0", "-0.0000", "1e100", "1.25E-100", "NaN"] {
        assert_eq!(Float::parse(s, None).unwrap().prec(), 53, "{s}");
    }
    assert!(Float::parse("NaN", Some(80)).unwrap().to_f64().is_nan());
    assert_eq!(
        Float::parse("-Infinity", None).unwrap().to_f64(),
        f64::NEG_INFINITY
    );
    assert_eq!(Float::parse("1.25`40", None).unwrap().prec(), 133);
    assert_eq!(Float::parse("1.25`40", Some(80)).unwrap().prec(), 80);
    for s in [
        "", "-", "1e", "1.2.3", "1`0", "1`-1", "1`NaN", "1`inf", "1`1`2",
    ] {
        assert!(Float::parse(s, None).is_err(), "{s}");
        assert!(Float::parse(s, Some(80)).is_err(), "{s}");
    }
    assert!(Float::parse("1", Some(0)).is_err());
}

fn check_zero_powers<F: Ring>(field: F, period: u64) {
    assert_eq!(field.pow(&field.zero(), 0), field.one());
    for exponent in [1, 2, period, period.saturating_mul(2), u64::MAX] {
        assert_eq!(field.pow(&field.zero(), exponent), field.zero());
        assert_eq!(field.pow(&field.one(), exponent), field.one());
    }
}

#[test]
fn zero_powers_are_correct_in_all_finite_field_representations() {
    check_zero_powers(Zp::new(7), 6);
    check_zero_powers(Zp64::new(7), 6);
    check_zero_powers(Zp::new_non_prime(9), 8);
    check_zero_powers(
        FiniteField::<Mersenne32>::new(Mersenne32::new()),
        (1 << 31) - 2,
    );
    check_zero_powers(
        FiniteField::<Mersenne64>::new(Mersenne64::new()),
        (1 << 61) - 2,
    );
    check_zero_powers(Z2, 1);
    check_zero_powers(FiniteField::<Integer>::new_non_prime(7.into()), 6);
    check_zero_powers(
        FiniteField::<MultiPrecisionInteger>::new_non_prime(7.into()),
        6,
    );
}

