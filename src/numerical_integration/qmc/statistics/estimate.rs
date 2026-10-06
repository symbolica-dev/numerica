use super::super::QmcError;
use super::sum::CompensatedSum;

/// One complete lattice mean with its original shift identity.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct ShiftEstimate {
    pub shift: usize,
    pub mean: Vec<f64>,
}

/// Statistical result from independent, complete shifted-lattice means.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct QmcEstimate {
    pub mean: Vec<f64>,
    pub standard_error: Vec<f64>,
    /// Row-major covariance of the estimated mean, not of individual points.
    pub covariance_of_mean: Vec<f64>,
    pub complete_shifts: usize,
    /// All evaluated points merged, including those in incomplete shifts.
    pub completed_points: u64,
    /// Points belonging to the complete shifts used by this estimate.
    pub used_points: u64,
}

impl QmcEstimate {
    /// Reduce equally weighted independent replica means.
    ///
    /// For correlated sectors, first sum each sector's complete mean for the
    /// same shift ID, then pass those aggregate vectors here. All replicas must
    /// use the same production design. The point counters remain zero because
    /// this constructor receives no raw-point information.
    pub fn from_shift_means(means: &[Vec<f64>]) -> Result<Self, QmcError> {
        let replicas = means.len();
        if replicas < 2 {
            return Err(QmcError::InsufficientShifts { complete: replicas });
        }
        let outputs = means[0].len();
        if outputs == 0 {
            return Err(QmcError::InvalidWork(
                "output dimension must be positive".into(),
            ));
        }
        let covariance_size = outputs
            .checked_mul(outputs)
            .ok_or_else(|| QmcError::InvalidWork("covariance shape overflow".into()))?;
        let origin = &means[0];
        let mut sums = vec![CompensatedSum::default(); outputs];
        for row in means {
            if row.len() != outputs {
                return Err(QmcError::OutputDimension {
                    expected: outputs,
                    actual: row.len(),
                });
            }
            for ((sum, &value), &anchor) in sums.iter_mut().zip(row).zip(origin) {
                if !value.is_finite() {
                    return Err(QmcError::NonFiniteValue);
                }
                sum.add(value - anchor)?;
            }
        }
        // Keep the center as an offset: adding it back to a large absolute
        // origin can round away variations that must remain in the covariance.
        let center: Vec<_> = sums
            .iter()
            .map(|sum| sum.total() / replicas as f64)
            .collect();
        let mean: Vec<_> = origin.iter().zip(&center).map(|(a, b)| a + b).collect();
        if mean.iter().any(|x| !x.is_finite()) {
            return Err(QmcError::NumericOverflow);
        }
        let mut covariance = vec![CompensatedSum::default(); covariance_size];
        for row in means {
            for i in 0..outputs {
                for j in 0..=i {
                    covariance[i * outputs + j].add(
                        ((row[i] - origin[i]) - center[i]) * ((row[j] - origin[j]) - center[j]),
                    )?;
                }
            }
        }
        let denominator = replicas as f64 * (replicas - 1) as f64;
        let mut covariance_of_mean = vec![0.0; covariance_size];
        for i in 0..outputs {
            for j in 0..=i {
                let value = covariance[i * outputs + j].total() / denominator;
                covariance_of_mean[i * outputs + j] = value;
                covariance_of_mean[j * outputs + i] = value;
            }
        }
        let standard_error = (0..outputs)
            .map(|i| covariance_of_mean[i * outputs + i].sqrt())
            .collect();
        Ok(Self {
            mean,
            standard_error,
            covariance_of_mean,
            complete_shifts: replicas,
            completed_points: 0,
            used_points: 0,
        })
    }
}
