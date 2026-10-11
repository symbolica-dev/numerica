use std::{
    fmt::{self, Debug, Display, Formatter, LowerExp, Write},
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use rand::Rng;

use super::{FixedPrecision, FloatLike, Real, RealLike, SingleFloat};
use crate::domains::{integer::Integer, rational::Rational};

/// A float that does linear error propagation.
///
/// The absolute error has the range of `f64`, even for multiprecision values.
/// If error tracking leaves that range, the absolute error is `NaN`; numerical
/// consumers must reject nonfinite uncertainty. This is not a certified bound.
#[derive(Copy, Clone)]
pub struct ErrorPropagatingFloat<T: FloatLike> {
    value: T,
    abs_err: f64,
}

// Refuse unsupported tracking instead of turning lost uncertainty into zero.
// These helpers intentionally do not replace the range-limited error field.
#[inline]
fn error_add(a: f64, b: f64) -> f64 {
    let r = a + b;
    if !r.is_finite() || a < 0. || b < 0. {
        f64::NAN
    } else {
        r
    }
}
#[inline]
fn error_mul(a: f64, b: f64) -> f64 {
    let r = a * b.abs();
    if !a.is_finite()
        || a < 0.
        || !b.is_finite()
        || !r.is_finite()
        || (r == 0. && a != 0. && b != 0.)
    {
        f64::NAN
    } else {
        r
    }
}
#[inline]
fn error_div(a: f64, b: f64) -> f64 {
    let r = a / b.abs();
    if !a.is_finite()
        || a < 0.
        || !b.is_finite()
        || b == 0.
        || !r.is_finite()
        || (r == 0. && a != 0.)
    {
        f64::NAN
    } else {
        r
    }
}
#[inline]
fn error_power(a: f64, exponent: f64) -> f64 {
    let r = a.powf(exponent);
    if !a.is_finite() || a < 0. || !exponent.is_finite() || !r.is_finite() || (r == 0. && a != 0.) {
        f64::NAN
    } else {
        r
    }
}
#[inline]
fn decimal_error(accuracy: f64) -> f64 {
    if accuracy == f64::INFINITY {
        return 0.;
    }
    let r = 10f64.powf(-accuracy);
    if accuracy.is_finite() && r.is_finite() && r != 0. {
        r
    } else {
        f64::NAN
    }
}
#[inline]
fn binary_error(bits: u32) -> f64 {
    let r = 2f64.powf(-f64::from(bits));
    if r == 0. { f64::NAN } else { r }
}
#[inline]
fn tracked_scale<T: RealLike>(value: &T) -> f64 {
    let r = value.to_f64();
    if !r.is_finite() || (r == 0. && !value.is_zero()) {
        f64::NAN
    } else {
        r
    }
}

#[inline]
fn rational_scale(value: &Rational) -> f64 {
    let r = value.to_f64().abs();
    if !r.is_finite() || (r == 0. && !value.is_zero()) {
        f64::NAN
    } else {
        r
    }
}

impl<T: FloatLike + From<f64>> From<f64> for ErrorPropagatingFloat<T> {
    fn from(value: f64) -> Self {
        if value == 0. {
            ErrorPropagatingFloat {
                value: value.into(),
                abs_err: f64::EPSILON,
            }
        } else {
            ErrorPropagatingFloat {
                value: value.into(),
                abs_err: error_mul(f64::EPSILON, value.abs()),
            }
        }
    }
}

impl<T: FloatLike> Neg for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        ErrorPropagatingFloat {
            value: -self.value,
            abs_err: self.abs_err,
        }
    }
}

impl<T: FloatLike> Add<&ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: &Self) -> Self::Output {
        ErrorPropagatingFloat {
            abs_err: error_add(self.abs_err, rhs.abs_err),
            value: self.value + &rhs.value,
        }
    }
}

impl<T: FloatLike> Add<ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        self + &rhs
    }
}

impl<T: FloatLike> Sub<&ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: &Self) -> Self::Output {
        self - rhs.clone()
    }
}

impl<T: FloatLike> Sub<ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl<T: RealLike> Mul<&ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: &Self) -> Self::Output {
        let value = self.value.clone() * &rhs.value;
        let r = tracked_scale(&rhs.value).abs();
        let s = tracked_scale(&self.value).abs();

        if s == 0. && r == 0. {
            ErrorPropagatingFloat {
                value,
                abs_err: error_mul(self.abs_err, rhs.abs_err),
            }
        } else {
            ErrorPropagatingFloat {
                value,
                abs_err: error_add(error_mul(self.abs_err, r), error_mul(rhs.abs_err, s)),
            }
        }
    }
}

impl<T: RealLike + Add<Rational, Output = T>, R: Into<Rational>> Add<R>
    for ErrorPropagatingFloat<T>
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: R) -> Self::Output {
        ErrorPropagatingFloat {
            abs_err: self.abs_err,
            value: self.value + rhs.into(),
        }
    }
}

impl<T: RealLike + Add<Rational, Output = T>, R: Into<Rational>> Sub<R>
    for ErrorPropagatingFloat<T>
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: R) -> Self::Output {
        self + -rhs.into()
    }
}

impl<T: RealLike + Mul<Rational, Output = T>, R: Into<Rational>> Mul<R>
    for ErrorPropagatingFloat<T>
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: R) -> Self::Output {
        let rhs = rhs.into();
        ErrorPropagatingFloat {
            abs_err: error_mul(self.abs_err, rational_scale(&rhs)),
            value: self.value * rhs,
        }
        .truncate()
    }
}

impl<T: RealLike + Div<Rational, Output = T>, R: Into<Rational>> Div<R>
    for ErrorPropagatingFloat<T>
{
    type Output = Self;

    #[inline]
    fn div(self, rhs: R) -> Self::Output {
        let rhs = rhs.into();
        ErrorPropagatingFloat {
            abs_err: error_mul(self.abs_err, rational_scale(&rhs.inv())),
            value: self.value.clone() / rhs,
        }
        .truncate()
    }
}

impl<T: FloatLike + From<f64>> Add<f64> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: f64) -> Self::Output {
        self + Self::from(rhs)
    }
}

impl<T: FloatLike + From<f64>> Sub<f64> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: f64) -> Self::Output {
        self - Self::from(rhs)
    }
}

impl<T: RealLike + From<f64>> Mul<f64> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: f64) -> Self::Output {
        self * Self::from(rhs)
    }
}

impl<T: RealLike + From<f64>> Div<f64> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: f64) -> Self::Output {
        self / Self::from(rhs)
    }
}

impl<T: RealLike> Mul<ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        self * &rhs
    }
}

impl<T: RealLike> Div<&ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: &Self) -> Self::Output {
        self * rhs.inv()
    }
}

impl<T: RealLike> Div<ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        self / &rhs
    }
}

impl<T: RealLike> AddAssign<&ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    #[inline]
    fn add_assign(&mut self, rhs: &ErrorPropagatingFloat<T>) {
        // TODO: optimize
        *self = self.clone() + rhs;
    }
}

impl<T: RealLike> AddAssign<ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    #[inline]
    fn add_assign(&mut self, rhs: ErrorPropagatingFloat<T>) {
        self.add_assign(&rhs)
    }
}

impl<T: RealLike> SubAssign<&ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: &ErrorPropagatingFloat<T>) {
        // TODO: optimize
        *self = self.clone() - rhs;
    }
}

impl<T: RealLike> SubAssign<ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: ErrorPropagatingFloat<T>) {
        self.sub_assign(&rhs)
    }
}

impl<T: RealLike> MulAssign<&ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: &ErrorPropagatingFloat<T>) {
        // TODO: optimize
        *self = self.clone() * rhs;
    }
}

impl<T: RealLike> MulAssign<ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: ErrorPropagatingFloat<T>) {
        self.mul_assign(&rhs)
    }
}

impl<T: RealLike> DivAssign<&ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    #[inline]
    fn div_assign(&mut self, rhs: &ErrorPropagatingFloat<T>) {
        // TODO: optimize
        *self = self.clone() / rhs;
    }
}

impl<T: RealLike> DivAssign<ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    #[inline]
    fn div_assign(&mut self, rhs: ErrorPropagatingFloat<T>) {
        self.div_assign(&rhs)
    }
}

impl<T: RealLike> ErrorPropagatingFloat<T> {
    /// Create a new precision tracking float with a number of precise decimal digits `prec`.
    /// The `prec` must be smaller than the precision of the underlying float.
    ///
    /// If the value provided is 0, the precision argument is interpreted as an accuracy (
    /// the number of digits of the absolute error).
    pub fn new(value: T, prec: f64) -> Self {
        let r = tracked_scale(&value).abs();

        if r == 0. {
            ErrorPropagatingFloat {
                abs_err: decimal_error(prec),
                value,
            }
        } else {
            ErrorPropagatingFloat {
                abs_err: error_mul(decimal_error(prec), r),
                value,
            }
        }
    }

    pub fn get_absolute_error(&self) -> f64 {
        self.abs_err
    }

    pub fn get_relative_error(&self) -> f64 {
        error_div(self.abs_err, tracked_scale(&self.value).abs())
    }

    /// Get the precision in number of decimal digits.
    #[inline(always)]
    pub fn get_precision(&self) -> Option<f64> {
        let r = tracked_scale(&self.value).abs();
        if r == 0. {
            None
        } else {
            Some(-error_div(self.abs_err, r).log10())
        }
    }

    /// Get the accuracy in number of decimal digits.
    #[inline(always)]
    pub fn get_accuracy(&self) -> f64 {
        -self.abs_err.log10()
    }

    /// Truncate the precision to the maximal number of stable decimal digits
    /// of the underlying float.
    #[inline(always)]
    pub fn truncate(mut self) -> Self {
        if self.value.fixed_precision() && !self.abs_err.is_nan() {
            let rounding = error_mul(self.value.get_epsilon(), tracked_scale(&self.value).abs());
            self.abs_err = if rounding.is_nan() {
                rounding
            } else {
                self.abs_err.max(rounding)
            };
        }
        self
    }
}

impl<T: FloatLike> ErrorPropagatingFloat<T> {
    pub fn new_with_accuracy(value: T, acc: f64) -> Self {
        ErrorPropagatingFloat {
            value,
            abs_err: decimal_error(acc),
        }
    }

    /// Get the number.
    #[inline(always)]
    pub fn get_num(&self) -> &T {
        &self.value
    }
}

impl<T: RealLike> fmt::Display for ErrorPropagatingFloat<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if let Some(p) = self.get_precision() {
            if p < 0. {
                f.write_char('0')
            } else {
                f.write_fmt(format_args!("{0:.1$}", self.value, p as usize))
            }
        } else {
            f.write_char('0')
        }
    }
}

impl<T: RealLike> Debug for ErrorPropagatingFloat<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.value, f)?;

        if let Some(p) = self.get_precision() {
            f.write_fmt(format_args!("`{p:.2}"))
        } else {
            f.write_fmt(format_args!("``{:.2}", -self.abs_err.log10()))
        }
    }
}

impl<T: RealLike> LowerExp for ErrorPropagatingFloat<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(self, f)
    }
}

impl<T: FloatLike> PartialEq for ErrorPropagatingFloat<T> {
    fn eq(&self, other: &Self) -> bool {
        // TODO: ignore precision for partial equality?
        self.value == other.value
    }
}

impl<T: FloatLike + PartialOrd> PartialOrd for ErrorPropagatingFloat<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl<T: RealLike> FloatLike for ErrorPropagatingFloat<T> {
    fn nan(&self) -> Option<Self> {
        Some(Self {
            value: self.value.nan()?,
            abs_err: f64::NAN,
        })
    }

    fn set_from(&mut self, other: &Self) {
        self.value.set_from(&other.value);
        self.abs_err = other.abs_err;
    }

    fn mul_add(&self, a: &Self, b: &Self) -> Self {
        self.clone() * a + b
    }

    fn neg(&self) -> Self {
        -self.clone()
    }

    fn zero(&self) -> Self {
        ErrorPropagatingFloat {
            value: self.value.zero(),
            abs_err: binary_error(self.value.get_precision()),
        }
    }

    fn new_zero() -> Self {
        ErrorPropagatingFloat {
            value: T::new_zero(),
            abs_err: 2f64.powi(-53),
        }
    }

    fn one(&self) -> Self {
        ErrorPropagatingFloat {
            value: self.value.one(),
            abs_err: binary_error(self.value.get_precision()),
        }
    }

    fn pow(&self, e: u64) -> Self {
        let i = tracked_scale(&self.value).abs();

        if i == 0. {
            return ErrorPropagatingFloat {
                value: self.value.pow(e),
                abs_err: error_power(self.abs_err, e as f64),
            };
        }

        let r = self.value.pow(e);
        // A nonzero base has a nonzero integer power; native underflow must
        // not turn its propagated uncertainty into an exact zero.
        let scale = if r.is_zero() {
            f64::NAN
        } else {
            tracked_scale(&r).abs()
        };
        ErrorPropagatingFloat {
            abs_err: error_div(error_mul(error_mul(self.abs_err, e as f64), scale), i),
            value: r,
        }
    }

    fn inv(&self) -> Self {
        let r = self.value.inv();
        let rr = if r.is_zero() {
            f64::NAN
        } else {
            tracked_scale(&r).abs()
        };
        ErrorPropagatingFloat {
            abs_err: error_mul(error_mul(self.abs_err, rr), rr),
            value: r,
        }
    }

    /// Convert from a `usize`.
    fn from_usize(&self, a: usize) -> Self {
        let v = self.value.from_usize(a);
        let r = tracked_scale(&v).abs();
        if r == 0. {
            ErrorPropagatingFloat {
                value: v,
                abs_err: binary_error(self.value.get_precision()),
            }
        } else {
            ErrorPropagatingFloat {
                value: v,
                abs_err: error_mul(binary_error(self.value.get_precision()), r),
            }
        }
    }

    /// Convert from a `i64`.
    fn from_i64(&self, a: i64) -> Self {
        let v = self.value.from_i64(a);
        let r = tracked_scale(&v).abs();
        if r == 0. {
            ErrorPropagatingFloat {
                value: v,
                abs_err: binary_error(self.value.get_precision()),
            }
        } else {
            ErrorPropagatingFloat {
                value: v,
                abs_err: error_mul(binary_error(self.value.get_precision()), r),
            }
        }
    }

    fn get_precision(&self) -> u32 {
        // return the precision of the underlying float instead
        // of the current tracked precision
        self.value.get_precision()
    }

    fn set_precision(&mut self, precision: u32) {
        self.value.set_precision(precision);
        // This explicitly resets the tracked error, as documented by FloatLike.
        // The newly assumed error must still fit the tracking representation.
        let scale = tracked_scale(&self.value).abs();
        self.abs_err = error_mul(
            binary_error(self.value.get_precision()),
            if scale == 0. { 1. } else { scale },
        );
    }

    fn get_epsilon(&self) -> f64 {
        2.0f64.powi(-(self.value.get_precision() as i32))
    }

    #[inline(always)]
    fn fixed_precision(&self) -> bool {
        self.value.fixed_precision()
    }

    fn sample_unit<R: Rng + ?Sized>(&self, rng: &mut R) -> Self {
        let v = self.value.sample_unit(rng);
        ErrorPropagatingFloat {
            abs_err: error_mul(self.abs_err, tracked_scale(&v).abs()),
            value: v,
        }
    }

    fn is_fully_zero(&self) -> bool {
        self.value.is_fully_zero()
    }
}

impl<T: RealLike> SingleFloat for ErrorPropagatingFloat<T> {
    fn is_zero(&self) -> bool {
        self.value.is_zero()
    }

    fn is_one(&self) -> bool {
        self.value.is_one()
    }

    fn is_finite(&self) -> bool {
        self.value.is_finite()
    }

    fn from_rational(&self, rat: &Rational) -> Self {
        if rat.is_zero() {
            ErrorPropagatingFloat {
                value: self.value.from_rational(rat),
                abs_err: self.abs_err,
            }
        } else {
            ErrorPropagatingFloat {
                value: self.value.from_rational(rat),
                abs_err: error_mul(self.abs_err, rational_scale(rat)),
            }
        }
    }
}

impl<T: RealLike> RealLike for ErrorPropagatingFloat<T> {
    fn to_usize_clamped(&self) -> usize {
        self.value.to_usize_clamped()
    }

    fn to_f64(&self) -> f64 {
        self.value.to_f64()
    }

    fn round_to_nearest_integer(&self) -> Integer {
        // TODO: what does this do with the error?
        self.value.round_to_nearest_integer()
    }
}

impl<T: Real + RealLike> Real for ErrorPropagatingFloat<T> {
    fn pi(&self) -> Self {
        let v = self.value.pi();
        ErrorPropagatingFloat {
            abs_err: error_mul(
                binary_error(self.value.get_precision()),
                tracked_scale(&v).abs(),
            ),
            value: v,
        }
    }

    fn e(&self) -> Self {
        let v = self.value.e();
        ErrorPropagatingFloat {
            abs_err: error_mul(
                binary_error(self.value.get_precision()),
                tracked_scale(&v).abs(),
            ),
            value: v,
        }
    }

    fn euler(&self) -> Self {
        let v = self.value.euler();
        ErrorPropagatingFloat {
            abs_err: error_mul(
                binary_error(self.value.get_precision()),
                tracked_scale(&v).abs(),
            ),
            value: v,
        }
    }

    fn phi(&self) -> Self {
        let v = self.value.phi();
        ErrorPropagatingFloat {
            abs_err: error_mul(
                binary_error(self.value.get_precision()),
                tracked_scale(&v).abs(),
            ),
            value: v,
        }
    }

    #[inline(always)]
    fn i(&self) -> Option<Self> {
        Some(ErrorPropagatingFloat {
            value: self.value.i()?,
            abs_err: binary_error(self.value.get_precision()),
        })
    }

    fn conj(&self) -> Self {
        ErrorPropagatingFloat {
            abs_err: self.abs_err,
            value: self.value.conj(),
        }
    }

    fn norm(&self) -> Self {
        ErrorPropagatingFloat {
            abs_err: self.abs_err,
            value: self.value.norm(),
        }
    }

    fn sqrt(&self) -> Self {
        let v = self.value.sqrt();
        let r = tracked_scale(&v).abs();

        ErrorPropagatingFloat {
            abs_err: error_div(self.abs_err, error_mul(2., r)),
            value: v,
        }
        .truncate()
    }

    fn log(&self) -> Self {
        let r = self.value.log();
        ErrorPropagatingFloat {
            abs_err: error_div(self.abs_err, tracked_scale(&self.value).abs()),
            value: r,
        }
        .truncate()
    }

    fn exp(&self) -> Self {
        let v = self.value.exp();
        let scale = if v.is_zero() {
            f64::NAN
        } else {
            tracked_scale(&v).abs()
        };
        ErrorPropagatingFloat {
            abs_err: error_mul(self.abs_err, scale),
            value: v,
        }
        .truncate()
    }

    fn sin(&self) -> Self {
        ErrorPropagatingFloat {
            abs_err: error_mul(self.abs_err, tracked_scale(&self.value).cos().abs()),
            value: self.value.sin(),
        }
        .truncate()
    }

    fn cos(&self) -> Self {
        ErrorPropagatingFloat {
            abs_err: error_mul(self.abs_err, tracked_scale(&self.value).sin().abs()),
            value: self.value.cos(),
        }
        .truncate()
    }

    fn tan(&self) -> Self {
        let t = self.value.tan();
        let tt = tracked_scale(&t).abs();

        ErrorPropagatingFloat {
            abs_err: error_mul(self.abs_err, 1. + error_mul(tt, tt)),
            value: t,
        }
        .truncate()
    }

    fn asin(&self) -> Self {
        let v = tracked_scale(&self.value);
        let t = self.value.asin();
        let tt = (1. - error_mul(v.abs(), v)).sqrt();
        ErrorPropagatingFloat {
            abs_err: error_div(self.abs_err, tt),
            value: t,
        }
        .truncate()
    }

    fn acos(&self) -> Self {
        let v = tracked_scale(&self.value);
        let t = self.value.acos();
        let tt = (1. - error_mul(v.abs(), v)).sqrt();
        ErrorPropagatingFloat {
            abs_err: error_div(self.abs_err, tt),
            value: t,
        }
        .truncate()
    }

    fn atan2(&self, x: &Self) -> Self {
        let t = self.value.atan2(&x.value);
        let r = self.clone() / x;
        let r2 = tracked_scale(&r.value).abs();

        let tt = 1. + error_mul(r2, r2);
        ErrorPropagatingFloat {
            abs_err: error_div(r.abs_err, tt),
            value: t,
        }
        .truncate()
    }

    fn sinh(&self) -> Self {
        ErrorPropagatingFloat {
            abs_err: error_mul(self.abs_err, tracked_scale(&self.value.cosh()).abs()),
            value: self.value.sinh(),
        }
        .truncate()
    }

    fn cosh(&self) -> Self {
        ErrorPropagatingFloat {
            abs_err: error_mul(self.abs_err, tracked_scale(&self.value.sinh()).abs()),
            value: self.value.cosh(),
        }
        .truncate()
    }

    fn tanh(&self) -> Self {
        let t = self.value.tanh();
        // 1 - tanh(x)^2 can round to zero while the native derivative is
        // nonzero. The equivalent cosh denominator avoids that cancellation.
        let c = tracked_scale(&self.value.cosh()).abs();
        ErrorPropagatingFloat {
            abs_err: error_div(error_div(self.abs_err, c), c),
            value: t,
        }
        .truncate()
    }

    fn asinh(&self) -> Self {
        let v = tracked_scale(&self.value);
        let t = self.value.asinh();
        let tt = (1. + error_mul(v.abs(), v)).sqrt();
        ErrorPropagatingFloat {
            abs_err: error_div(self.abs_err, tt),
            value: t,
        }
        .truncate()
    }

    fn acosh(&self) -> Self {
        let v = tracked_scale(&self.value);
        let t = self.value.acosh();
        let tt = (error_mul(v.abs(), v) - 1.).sqrt();
        ErrorPropagatingFloat {
            abs_err: error_div(self.abs_err, tt),
            value: t,
        }
        .truncate()
    }

    fn atanh(&self) -> Self {
        let v = tracked_scale(&self.value);
        let t = self.value.atanh();
        let tt = 1. - error_mul(v.abs(), v);
        ErrorPropagatingFloat {
            abs_err: error_div(self.abs_err, tt),
            value: t,
        }
        .truncate()
    }

    fn powf(&self, e: &Self) -> Self {
        let i = tracked_scale(&self.value).abs();

        if i == 0. {
            return ErrorPropagatingFloat {
                value: self.value.powf(&e.value),
                abs_err: if self.abs_err == 0. && e.abs_err == 0. {
                    0.
                } else {
                    f64::NAN
                },
            };
        }

        let r = self.value.powf(&e.value);
        let scale = if r.is_zero() {
            f64::NAN
        } else {
            tracked_scale(&r).abs()
        };
        ErrorPropagatingFloat {
            abs_err: error_div(
                error_mul(
                    error_add(
                        error_mul(self.abs_err, tracked_scale(&e.value).abs()),
                        error_mul(
                            error_mul(e.abs_err, i),
                            tracked_scale(&self.value.norm().log()).abs(),
                        ),
                    ),
                    scale,
                ),
                i,
            ),
            value: r,
        }
        .truncate()
    }
}

impl<T: FloatLike + FixedPrecision> FixedPrecision for ErrorPropagatingFloat<T> {
    const BINARY_PRECISION: usize = T::BINARY_PRECISION;
    const DECIMAL_PRECISION: usize = T::DECIMAL_PRECISION;
}
