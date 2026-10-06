use std::fmt;

/// An invalid QMC rule, work package, observation, or statistical estimate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QmcError {
    InvalidRule(String),
    InvalidPlan(String),
    InvalidWork(String),
    OutputDimension { expected: usize, actual: usize },
    NonFiniteValue,
    NumericOverflow,
    NumericUnderflow,
    IncompleteWork { expected: u64, actual: u64 },
    OverlappingWork,
    InsufficientShifts { complete: usize },
}

impl fmt::Display for QmcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRule(s) => write!(f, "invalid rank-one rule: {s}"),
            Self::InvalidPlan(s) => write!(f, "invalid QMC plan: {s}"),
            Self::InvalidWork(s) => write!(f, "invalid QMC work package: {s}"),
            Self::OutputDimension { expected, actual } => {
                write!(
                    f,
                    "expected {expected} output components, received {actual}"
                )
            }
            Self::NonFiniteValue => f.write_str("QMC observations must be finite"),
            Self::NumericOverflow => f.write_str("QMC accumulation exceeded finite f64 range"),
            Self::NumericUnderflow => {
                f.write_str("positive QMC periodization attenuation lost normal f64 range")
            }
            Self::IncompleteWork { expected, actual } => {
                write!(
                    f,
                    "incomplete QMC work: expected {expected} points, received {actual}"
                )
            }
            Self::OverlappingWork => f.write_str("QMC work overlaps an already merged package"),
            Self::InsufficientShifts { complete } => {
                write!(
                    f,
                    "QMC uncertainty requires two complete shifts, received {complete}"
                )
            }
        }
    }
}

impl std::error::Error for QmcError {}
