use super::QmcError;

/// Symmetric Korobov periodization with exponent two.
///
/// `phi(t) = t^3 (10 - 15 t + 6 t^2)` and
/// `phi'(t) = 30 t^2 (1-t)^2`. Evaluate the lower half and reflect to
/// avoid polynomial cancellation near one. The transformed integrand must
/// include the returned product Jacobian. Endpoint values are not clipped.
/// Like [`Korobov3`], this is an explicit caller choice; its suitability and
/// convergence depend on the integrand and lattice.
#[derive(Debug, Clone, Copy, Default)]
pub struct Korobov2;

impl Korobov2 {
    pub fn map(t: f64) -> f64 {
        let x = if t > 0.5 { 1.0 - t } else { t };
        let y = x * x * x * (10.0 + x * (-15.0 + 6.0 * x));
        if t > 0.5 { 1.0 - y } else { y }
    }

    pub fn jacobian(t: f64) -> f64 {
        let p = t * (1.0 - t);
        30.0 * p * p
    }

    /// Transform one point and return its product Jacobian, with the same
    /// coordinate validation, exact-endpoint and normal-range guarantees as
    /// [`Korobov3::transform_in_place`].
    pub fn transform_in_place(point: &mut [f64]) -> Result<f64, QmcError> {
        transform_in_place(point, Self::map, Self::jacobian)
    }
}

/// Symmetric Korobov periodization with exponent three.
///
/// `phi(t) = 35 t^4 - 84 t^5 + 70 t^6 - 20 t^7` and
/// `phi'(t) = 140 t^3 (1-t)^3`. Evaluate the lower half and reflect to
/// avoid polynomial cancellation near one. The transformed integrand must
/// include the returned product Jacobian. Endpoint values are not clipped.
#[derive(Debug, Clone, Copy, Default)]
pub struct Korobov3;

impl Korobov3 {
    pub fn map(t: f64) -> f64 {
        let x = if t > 0.5 { 1.0 - t } else { t };
        let x2 = x * x;
        let y = x2 * x2 * (35.0 + x * (-84.0 + x * (70.0 - 20.0 * x)));
        if t > 0.5 { 1.0 - y } else { y }
    }

    pub fn jacobian(t: f64) -> f64 {
        let p = t * (1.0 - t);
        140.0 * p * p * p
    }

    /// Transform one point, returning the product Jacobian (one in dimension zero).
    /// A zero Jacobian is returned only for an original coordinate exactly at
    /// an endpoint. Loss of normal binary64 range at a strictly interior point
    /// is an explicit error, since a large integrand can amplify that loss.
    /// This conservative guard rejects subnormal attenuation even when later
    /// amplifying factors could make the final product representable. Acceptance
    /// must not depend on which parameter happens to appear first.
    pub fn transform_in_place(point: &mut [f64]) -> Result<f64, QmcError> {
        transform_in_place(point, Self::map, Self::jacobian)
    }
}

fn transform_in_place(
    point: &mut [f64],
    map: impl Fn(f64) -> f64,
    jacobian: impl Fn(f64) -> f64,
) -> Result<f64, QmcError> {
    if point
        .iter()
        .any(|x| !x.is_finite() || !(0.0..=1.0).contains(x))
    {
        return Err(QmcError::InvalidWork(
            "periodization requires coordinates in [0, 1]".into(),
        ));
    }
    let endpoint = point.iter().any(|x| *x == 0.0 || *x == 1.0);
    let mut weight = 1.0;
    let mut attenuation = 1.0;
    for x in point {
        if !endpoint {
            let jacobian = jacobian(*x);
            if jacobian < 1.0 {
                attenuation *= jacobian;
                if attenuation < f64::MIN_POSITIVE {
                    return Err(QmcError::NumericUnderflow);
                }
            }
            weight *= jacobian;
        }
        *x = map(*x);
    }
    if endpoint {
        return Ok(0.0);
    }
    if !weight.is_finite() {
        return Err(QmcError::NumericOverflow);
    }
    if weight < f64::MIN_POSITIVE {
        return Err(QmcError::NumericUnderflow);
    }
    Ok(weight)
}
