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

