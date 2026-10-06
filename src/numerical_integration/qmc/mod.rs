//! Randomized rank-one lattice integration with caller-owned evaluation and scheduling.
//!
//! A [`QmcPlan`] describes a rule and independent random shifts. Its work packages
//! contain indices, not points: workers generate coordinates locally into reusable
//! buffers, evaluate the integrand, and return [`QmcPartial`] sums. Merge these in
//! a [`QmcAccumulator`]. Uncertainty is computed from **complete shift means**,
//! never from the mutually dependent points in one lattice.
//!
//! The library creates no threads, chooses no stopping rule, and invokes no
//! integrand. See `examples/qmc.rs` for serial and caller-owned threaded loops.
//! Existing Monte Carlo grids and their random sequences are unchanged.
//!
//! This lane uses `f64` coordinates and vector outputs. Represent a complex
//! result by two components (real and imaginary); their covariance is retained.
//! To correlate several integrands, use the same seed and stream and combine
//! their means **within each shift** before calling [`QmcEstimate::from_shift_means`].
//! Different streams give independent randomizations. Do not combine estimates
//! from different lattice sizes or an adaptation pilot as equal production shifts.
//!
//! With the `serde` feature, plans and accumulation state preserve floating-point
//! bits even through JSON. The plan encoding is versioned; checkpoint coordinates
//! and sums use integer bit patterns, whereas displayed estimates use numbers.

mod catalogue;
mod error;
mod rule;
mod statistics;
mod transform;
mod work;

pub use catalogue::PublishedLattice;
pub use error::QmcError;
pub use rule::{KUO_MAX_DIMENSION, KUO_MAX_POINTS, KUO_MIN_POINTS, Rank1Rule, RuleSource};
pub use statistics::{QmcAccumulator, QmcEstimate, QmcPartial, ShiftEstimate};
pub use transform::{Korobov2, Korobov3};
pub use work::{QmcPlan, QmcWorkPackage};
