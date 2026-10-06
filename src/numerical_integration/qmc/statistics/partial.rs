use super::super::{QmcError, QmcPlan, QmcWorkPackage};
use super::sum::CompensatedSum;

/// A worker's vector-valued sums, grouped by randomized shift.
///
/// Add observations in package order. Values must already include any
/// periodization Jacobian or importance weight. This type never treats points
/// inside a lattice as independent observations for error estimation.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(try_from = "PartialState"))]
#[derive(Debug, Clone)]
pub struct QmcPartial {
    pub(super) work: QmcWorkPackage,
    pub(super) points_per_shift: u64,
    pub(super) output_count: usize,
    pub(super) processed: u64,
    pub(super) counts: Vec<u64>,
    pub(super) sums: Vec<Vec<CompensatedSum>>,
    failed: bool,
}

impl QmcPartial {
    pub fn new(
        plan: &QmcPlan,
        work: QmcWorkPackage,
        output_count: usize,
    ) -> Result<Self, QmcError> {
        let range = plan.validate_work(work)?;
        if output_count == 0 {
            return Err(QmcError::InvalidWork(
                "output dimension must be positive".into(),
            ));
        }
        let n = plan.rule().points();
        let groups = ((range.end - 1) / n - range.start / n + 1) as usize;
        groups
            .checked_mul(output_count)
            .ok_or_else(|| QmcError::InvalidWork("partial buffer length overflow".into()))?;
        Ok(Self {
            work,
            points_per_shift: n,
            output_count,
            processed: 0,
            counts: vec![0; groups],
            sums: vec![vec![CompensatedSum::default(); output_count]; groups],
            failed: false,
        })
    }

    pub fn work(&self) -> QmcWorkPackage {
        self.work
    }
    pub fn processed_points(&self) -> u64 {
        self.processed
    }
    pub fn output_count(&self) -> usize {
        self.output_count
    }

    /// Add the next point's complete output vector without allocating.
    ///
    /// Nonfinite input and wrong dimensions leave the partial unchanged. An
    /// arithmetic overflow invalidates the partial permanently; discard it.
    pub fn push(&mut self, values: &[f64]) -> Result<(), QmcError> {
        if self.failed {
            return Err(QmcError::NumericOverflow);
        }
        if values.len() != self.output_count {
            return Err(QmcError::OutputDimension {
                expected: self.output_count,
                actual: values.len(),
            });
        }
        if values.iter().any(|x| !x.is_finite()) {
            return Err(QmcError::NonFiniteValue);
        }
        if self.processed >= self.work.count {
            return Err(QmcError::InvalidWork(
                "partial already contains every package point".into(),
            ));
        }
        let group = ((self.work.start + self.processed) / self.points_per_shift
            - self.work.start / self.points_per_shift) as usize;
        for (sum, &value) in self.sums[group].iter_mut().zip(values) {
            if let Err(error) = sum.add(value) {
                self.failed = true;
                return Err(error);
            }
        }
        self.counts[group] += 1;
        self.processed += 1;
        Ok(())
    }

    /// Mark the worker return boundary, rejecting missing observations.
    pub fn finish(self) -> Result<Self, QmcError> {
        self.validate(true)?;
        Ok(self)
    }

    pub(super) fn first_shift(&self) -> usize {
        (self.work.start / self.points_per_shift) as usize
    }

    pub(super) fn validate(&self, complete: bool) -> Result<(), QmcError> {
        if self.failed {
            return Err(QmcError::NumericOverflow);
        }
        if !(2..=1 << 53).contains(&self.points_per_shift) || self.output_count == 0 {
            return Err(QmcError::InvalidWork("invalid partial shape".into()));
        }
        let end = self
            .work
            .start
            .checked_add(self.work.count)
            .ok_or_else(|| QmcError::InvalidWork("partial interval overflow".into()))?;
        if self.work.count == 0 || self.processed > self.work.count {
            return Err(QmcError::InvalidWork("invalid partial point count".into()));
        }
        let n = self.points_per_shift;
        let first_shift = self.work.start / n;
        let groups = usize::try_from((end - 1) / n - first_shift + 1)
            .map_err(|_| QmcError::InvalidWork("partial group count overflow".into()))?;
        if self.counts.len() != groups || self.sums.len() != groups {
            return Err(QmcError::InvalidWork(
                "invalid partial group buffers".into(),
            ));
        }
        for group in 0..groups {
            let start = self.work.start.max((first_shift + group as u64) * n);
            let capacity = (n - start % n).min(end - start);
            let expected = (self.work.start + self.processed)
                .saturating_sub(start)
                .min(capacity);
            if self.counts[group] != expected || self.sums[group].len() != self.output_count {
                return Err(QmcError::InvalidWork(
                    "partial counts or output buffers differ from interval".into(),
                ));
            }
            if self.sums[group].iter().any(|sum| !sum.is_finite()) {
                return Err(QmcError::NonFiniteValue);
            }
            if expected == 0
                && self.sums[group]
                    .iter()
                    .any(|sum| sum.sum != 0.0 || sum.correction != 0.0)
            {
                return Err(QmcError::InvalidWork(
                    "empty partial group has nonzero sums".into(),
                ));
            }
        }
        if complete && self.processed != self.work.count {
            return Err(QmcError::IncompleteWork {
                expected: self.work.count,
                actual: self.processed,
            });
        }
        Ok(())
    }
}

#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
struct PartialState {
    work: QmcWorkPackage,
    points_per_shift: u64,
    output_count: usize,
    processed: u64,
    counts: Vec<u64>,
    sums: Vec<Vec<CompensatedSum>>,
    failed: bool,
}

#[cfg(feature = "serde")]
impl TryFrom<PartialState> for QmcPartial {
    type Error = QmcError;
    fn try_from(state: PartialState) -> Result<Self, Self::Error> {
        let partial = Self {
            work: state.work,
            points_per_shift: state.points_per_shift,
            output_count: state.output_count,
            processed: state.processed,
            counts: state.counts,
            sums: state.sums,
            failed: state.failed,
        };
        partial.validate(false)?;
        Ok(partial)
    }
}
