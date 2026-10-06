use std::ops::Range;

use rand::{RngCore, SeedableRng};

use crate::numerical_integration::MonteCarloRng;

use super::{QmcError, Rank1Rule};

/// A stable, shift-major interval of a plan's lattice points.
///
/// Packages contain no coordinate buffers. Their identity includes all rule
/// components and all random shifts. Deserialized packages are checked against
/// the receiving plan before use. They are not authenticated messages.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QmcWorkPackage {
    pub(super) plan_id: [u64; 2],
    pub(super) start: u64,
    pub(super) count: u64,
}

impl QmcWorkPackage {
    pub fn start(&self) -> u64 {
        self.start
    }
    pub fn point_count(&self) -> u64 {
        self.count
    }
}

/// Immutable lattice and randomization information shared with workers.
///
/// Coordinates use linear order within each shift. A flat index `k` means
/// shift `k / rule.points()` and lattice index `k % rule.points()`. Only the
/// `shifts * dimension` shift coordinates are materialized, never a full lattice.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(try_from = "PlanConfig", into = "PlanConfig"))]
#[derive(Debug, Clone, PartialEq)]
pub struct QmcPlan {
    rule: Rank1Rule,
    shifts: Vec<Vec<f64>>,
    total_points: u64,
    id: [u64; 2],
}

impl QmcPlan {
    /// Create deterministic independent shifts using Havana's RNG primitives.
    ///
    /// A shift's coordinates do not depend on the requested dimension, worker
    /// count or batching. Increasing dimensions preserves every existing shift's
    /// coordinate prefix; increasing the shift count preserves earlier shifts.
    /// `stream` namespaces independent integration runs, not worker IDs.
    pub fn new(rule: Rank1Rule, shifts: u32, seed: u64, stream: u64) -> Result<Self, QmcError> {
        Self::validate_size(&rule, shifts as usize)?;
        // Expand the two seeds separately with Havana's existing SplitMix64.
        // Xoshiro jump spaces independent shifts 2^128 RNG steps apart.
        let mut seed_state = seed;
        let mut stream_state = stream;
        let mut state = [0u8; 32];
        for chunk in state.as_chunks_mut::<8>().0 {
            let a = MonteCarloRng::splitmix64_next(&mut seed_state);
            let b = MonteCarloRng::splitmix64_next(&mut stream_state);
            chunk.copy_from_slice(&a.wrapping_add(b.rotate_left(29)).to_le_bytes());
        }
        let mut shift_rng = MonteCarloRng::from_seed(state);
        let mut values = Vec::with_capacity(shifts as usize);
        for _ in 0..shifts {
            let mut coordinate_rng = shift_rng.clone();
            values.push(
                (0..rule.dimension())
                    .map(|_| (coordinate_rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64))
                    .collect(),
            );
            shift_rng.jump();
        }
        Self::with_shifts(rule, values)
    }

    /// Supply exact random shifts, for cross-language comparisons or custom RNGs.
    ///
    /// The caller is responsible for independence and uniformity. Coordinates
    /// must be finite and in [0,1). A single shift permits point generation;
    /// at least two completed shifts are required for statistical errors.
    pub fn with_shifts(rule: Rank1Rule, shifts: Vec<Vec<f64>>) -> Result<Self, QmcError> {
        let total_points = Self::validate_size(&rule, shifts.len())?;
        if shifts.iter().any(|v| v.len() != rule.dimension()) {
            return Err(QmcError::InvalidPlan(
                "shift dimension differs from rule".into(),
            ));
        }
        if shifts
            .iter()
            .flatten()
            .any(|x| !x.is_finite() || !(0.0..1.0).contains(x))
        {
            return Err(QmcError::InvalidPlan(
                "shift coordinates must lie in [0,1)".into(),
            ));
        }
        // A stable two-word fingerprint for accidental cross-plan mixing, not a
        // cryptographic digest. Include the entire vector and every shift bit.
        let mut id = [0xcbf29ce484222325u64, 0x84222325cbf29ce4u64];
        let words = [
            1,
            rule.points(),
            rule.dimension() as u64,
            shifts.len() as u64,
        ]
        .into_iter()
        .chain(rule.generator().iter().copied())
        .chain(shifts.iter().flatten().map(|x| x.to_bits()));
        for word in words {
            for byte in word.to_le_bytes() {
                id[0] = (id[0] ^ byte as u64).wrapping_mul(0x100000001b3);
                id[1] = (id[1] ^ byte as u64).wrapping_mul(0x9e3779b185ebca87);
            }
        }
        Ok(Self {
            rule,
            shifts,
            total_points,
            id,
        })
    }

    fn validate_size(rule: &Rank1Rule, shifts: usize) -> Result<u64, QmcError> {
        if !(1..=u32::MAX as usize).contains(&shifts) {
            return Err(QmcError::InvalidPlan(
                "shift count must be in 1..=u32::MAX".into(),
            ));
        }
        shifts
            .checked_mul(rule.dimension())
            .ok_or_else(|| QmcError::InvalidPlan("shift buffer length overflow".into()))?;
        rule.points()
            .checked_mul(shifts as u64)
            .ok_or_else(|| QmcError::InvalidPlan("total point count overflow".into()))
    }

    pub fn rule(&self) -> &Rank1Rule {
        &self.rule
    }
    pub fn dimension(&self) -> usize {
        self.rule.dimension()
    }
    pub fn shift_count(&self) -> usize {
        self.shifts.len()
    }
    pub fn shifts(&self) -> &[Vec<f64>] {
        &self.shifts
    }
    pub fn total_points(&self) -> u64 {
        self.total_points
    }

    /// Describe a nonempty interval. Invalid intervals are never clamped.
    pub fn work(&self, start: u64, count: u64) -> Result<QmcWorkPackage, QmcError> {
        let work = QmcWorkPackage {
            plan_id: self.id,
            start,
            count,
        };
        self.validate_work(work)?;
        Ok(work)
    }

    /// Split a plan into stable canonical packages, independent of worker count.
    ///
    /// Keep `package_points` fixed when requiring bitwise repeatable reduction.
    /// Different package sizes preserve points but may change floating sums.
    pub fn packages(
        &self,
        package_points: u64,
    ) -> Result<impl Iterator<Item = QmcWorkPackage> + '_, QmcError> {
        if package_points == 0 {
            return Err(QmcError::InvalidWork(
                "package size must be positive".into(),
            ));
        }
        let mut start = 0;
        Ok(std::iter::from_fn(move || {
            if start == self.total_points {
                return None;
            }
            let count = package_points.min(self.total_points - start);
            let work = QmcWorkPackage {
                plan_id: self.id,
                start,
                count,
            };
            start += count;
            Some(work)
        }))
    }

    pub(super) fn validate_work(&self, work: QmcWorkPackage) -> Result<Range<u64>, QmcError> {
        let end = work
            .start
            .checked_add(work.count)
            .ok_or_else(|| QmcError::InvalidWork("interval overflow".into()))?;
        if work.plan_id != self.id {
            return Err(QmcError::InvalidWork(
                "package belongs to another rule or randomization".into(),
            ));
        }
        if work.count == 0 || end > self.total_points {
            return Err(QmcError::InvalidWork(
                "interval is empty or exceeds plan".into(),
            ));
        }
        Ok(work.start..end)
    }

    fn point_unchecked(&self, flat_index: u64, output: &mut [f64]) {
        let shift = (flat_index / self.rule.points()) as usize;
        let index = flat_index % self.rule.points();
        let n = self.rule.points() as f64;
        for (axis, x) in output.iter_mut().enumerate() {
            let lattice = self.rule.numerator(index, axis) as f64 / n;
            let offset = self.shifts[shift][axis];
            let shifted = lattice + offset;
            *x = if shifted >= 1.0 {
                // Subtract before adding across the periodic boundary. Taking
                // the complement of the larger addend is exact (Sterbenz),
                // and retains tiny residues lost by `(lattice + offset) - 1`.
                let residue = if lattice >= offset {
                    offset - (1.0 - lattice)
                } else {
                    lattice - (1.0 - offset)
                };
                if residue >= 0.0 {
                    residue
                } else {
                    // The exact sum lay just below one but rounded up to one.
                    // Select its nearest representable coordinate in [0,1).
                    1.0_f64.next_down()
                }
            } else {
                shifted
            };
        }
    }

    /// Generate one point into a caller-owned buffer of `dimension()` elements.
    pub fn point(&self, flat_index: u64, output: &mut [f64]) -> Result<(), QmcError> {
        if flat_index >= self.total_points {
            return Err(QmcError::InvalidWork("point index exceeds plan".into()));
        }
        if output.len() != self.dimension() {
            return Err(QmcError::OutputDimension {
                expected: self.dimension(),
                actual: output.len(),
            });
        }
        self.point_unchecked(flat_index, output);
        Ok(())
    }

    /// Generate a worker's row-major batch without allocating a point array.
    /// The buffer length must equal `work.point_count() * dimension()`.
    pub fn fill_points(&self, work: QmcWorkPackage, output: &mut [f64]) -> Result<(), QmcError> {
        let range = self.validate_work(work)?;
        let expected = usize::try_from(work.count)
            .ok()
            .and_then(|n| n.checked_mul(self.dimension()))
            .ok_or_else(|| QmcError::InvalidWork("point buffer length overflow".into()))?;
        if output.len() != expected {
            return Err(QmcError::OutputDimension {
                expected,
                actual: output.len(),
            });
        }
        for (index, point) in range.zip(output.chunks_exact_mut(self.dimension())) {
            self.point_unchecked(index, point);
        }
        Ok(())
    }
}

#[cfg(feature = "serde")]
#[derive(serde::Serialize, serde::Deserialize)]
struct PlanConfig {
    version: u8,
    rule: Rank1Rule,
    // Preserve every coordinate bit even with serializers whose decimal-float
    // parser does not promise a lossless round trip.
    shifts: Vec<Vec<u64>>,
}

#[cfg(feature = "serde")]
impl From<QmcPlan> for PlanConfig {
    fn from(plan: QmcPlan) -> Self {
        Self {
            version: 1,
            rule: plan.rule,
            shifts: plan
                .shifts
                .into_iter()
                .map(|row| row.into_iter().map(f64::to_bits).collect())
                .collect(),
        }
    }
}

#[cfg(feature = "serde")]
impl TryFrom<PlanConfig> for QmcPlan {
    type Error = QmcError;
    fn try_from(config: PlanConfig) -> Result<Self, Self::Error> {
        if config.version != 1 {
            return Err(QmcError::InvalidPlan(
                "unsupported serialized QMC plan version".into(),
            ));
        }
        Self::with_shifts(
            config.rule,
            config
                .shifts
                .into_iter()
                .map(|row| row.into_iter().map(f64::from_bits).collect())
                .collect(),
        )
    }
}
