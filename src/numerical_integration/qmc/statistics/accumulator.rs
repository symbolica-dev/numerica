use std::{collections::BTreeMap, ops::Range};

use super::super::{QmcError, QmcPlan};
use super::{QmcEstimate, QmcPartial, ShiftEstimate, sum::CompensatedSum};

/// Merge worker returns in any arrival order, rejecting overlap and duplicates.
///
/// Partials are retained and reduced in increasing point-index order. With a
/// fixed canonical package layout, this makes estimates bitwise independent of
/// worker count and completion order. Storage grows with returned packages and
/// output components, not with individual lattice points. Completed packages
/// and remaining ranges can be checkpointed by enabling `serde`.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(try_from = "AccumulatorState", into = "AccumulatorState")
)]
#[derive(Debug, Clone)]
pub struct QmcAccumulator {
    plan: QmcPlan,
    output_count: usize,
    partials: BTreeMap<u64, QmcPartial>,
}

impl QmcAccumulator {
    pub fn new(plan: QmcPlan, output_count: usize) -> Result<Self, QmcError> {
        if output_count == 0 {
            return Err(QmcError::InvalidWork(
                "output dimension must be positive".into(),
            ));
        }
        output_count
            .checked_mul(plan.shift_count())
            .ok_or_else(|| QmcError::InvalidWork("replica buffer length overflow".into()))?;
        Ok(Self {
            plan,
            output_count,
            partials: BTreeMap::new(),
        })
    }

    pub fn plan(&self) -> &QmcPlan {
        &self.plan
    }
    pub fn output_count(&self) -> usize {
        self.output_count
    }

    /// Merge a completed package. Errors leave the accumulator unchanged.
    pub fn merge(&mut self, partial: QmcPartial) -> Result<(), QmcError> {
        let range = self.plan.validate_work(partial.work)?;
        partial.validate(true)?;
        if partial.points_per_shift != self.plan.rule().points()
            || partial.output_count != self.output_count
        {
            return Err(QmcError::InvalidWork(
                "partial shape differs from accumulator".into(),
            ));
        }
        if let Some((_, previous)) = self.partials.range(..=range.start).next_back()
            && previous.work.start + previous.work.count > range.start
        {
            return Err(QmcError::OverlappingWork);
        }
        if let Some((&next, _)) = self.partials.range(range.start..).next()
            && next < range.end
        {
            return Err(QmcError::OverlappingWork);
        }
        self.partials.insert(range.start, partial);
        Ok(())
    }

    pub fn completed_points(&self) -> u64 {
        self.partials.values().map(|p| p.processed).sum()
    }
    pub fn is_complete(&self) -> bool {
        self.completed_points() == self.plan.total_points()
    }

    /// Accepted package intervals in canonical index order. This lets a caller
    /// validate its own fixed packaging and bookkeeping when restoring state.
    pub fn completed_work_packages(
        &self,
    ) -> impl Iterator<Item = super::super::QmcWorkPackage> + '_ {
        self.partials.values().map(|partial| partial.work())
    }

    /// Unmerged intervals, in increasing order. No partial interval is counted twice.
    pub fn missing_ranges(&self) -> Vec<Range<u64>> {
        let mut missing = Vec::new();
        let mut next = 0;
        for partial in self.partials.values() {
            if next < partial.work.start {
                missing.push(next..partial.work.start);
            }
            next = partial.work.start + partial.work.count;
        }
        if next < self.plan.total_points() {
            missing.push(next..self.plan.total_points());
        }
        missing
    }

    /// Complete replica means only. A gap anywhere inside a shift excludes it.
    pub fn shift_estimates(&self) -> Result<Vec<ShiftEstimate>, QmcError> {
        let counts = self.shift_counts();
        let mut sums =
            vec![vec![CompensatedSum::default(); self.output_count]; self.plan.shift_count()];
        for partial in self.partials.values() {
            let first = partial.first_shift();
            for (group, row) in partial.sums.iter().enumerate() {
                for (sum, &value) in sums[first + group].iter_mut().zip(row) {
                    sum.merge(value)?;
                }
            }
        }
        let n = self.plan.rule().points();
        Ok(counts
            .iter()
            .zip(sums)
            .enumerate()
            .filter_map(|(shift, (&count, sums))| {
                (count == n).then(|| ShiftEstimate {
                    shift,
                    mean: sums.into_iter().map(|sum| sum.total() / n as f64).collect(),
                })
            })
            .collect())
    }

    /// Identities of fully accepted shifts, without evaluating their numerical
    /// sums. Coverage remains observable even if finite partial observations
    /// overflow when combined into a mean or covariance.
    pub fn complete_shift_ids(&self) -> Vec<usize> {
        let n = self.plan.rule().points();
        self.shift_counts()
            .into_iter()
            .enumerate()
            .filter_map(|(shift, count)| (count == n).then_some(shift))
            .collect()
    }

    fn shift_counts(&self) -> Vec<u64> {
        let mut counts = vec![0u64; self.plan.shift_count()];
        for partial in self.partials.values() {
            let first = partial.first_shift();
            for (group, &count) in partial.counts.iter().enumerate() {
                counts[first + group] += count;
            }
        }
        counts
    }

    pub fn estimate(&self) -> Result<QmcEstimate, QmcError> {
        let means: Vec<_> = self
            .shift_estimates()?
            .into_iter()
            .map(|s| s.mean)
            .collect();
        let mut estimate = QmcEstimate::from_shift_means(&means)?;
        estimate.completed_points = self.completed_points();
        estimate.used_points = estimate.complete_shifts as u64 * self.plan.rule().points();
        Ok(estimate)
    }
}

#[cfg(feature = "serde")]
#[derive(serde::Serialize, serde::Deserialize)]
struct AccumulatorState {
    plan: QmcPlan,
    output_count: usize,
    partials: Vec<QmcPartial>,
}

#[cfg(feature = "serde")]
impl From<QmcAccumulator> for AccumulatorState {
    fn from(accumulator: QmcAccumulator) -> Self {
        Self {
            plan: accumulator.plan,
            output_count: accumulator.output_count,
            partials: accumulator.partials.into_values().collect(),
        }
    }
}

#[cfg(feature = "serde")]
impl TryFrom<AccumulatorState> for QmcAccumulator {
    type Error = QmcError;
    fn try_from(state: AccumulatorState) -> Result<Self, Self::Error> {
        let mut accumulator = Self::new(state.plan, state.output_count)?;
        for partial in state.partials {
            accumulator.merge(partial)?;
        }
        Ok(accumulator)
    }
}
