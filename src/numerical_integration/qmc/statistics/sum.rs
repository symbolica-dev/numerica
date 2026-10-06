use super::super::QmcError;

/// Neumaier summation, retaining cancellation terms lost by naive addition.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(try_from = "SumState", into = "SumState"))]
#[derive(Debug, Default, Clone, Copy)]
pub(super) struct CompensatedSum {
    pub sum: f64,
    pub correction: f64,
}

#[cfg(feature = "serde")]
#[derive(serde::Serialize, serde::Deserialize)]
struct SumState {
    sum: u64,
    correction: u64,
}

#[cfg(feature = "serde")]
impl From<CompensatedSum> for SumState {
    fn from(value: CompensatedSum) -> Self {
        Self {
            sum: value.sum.to_bits(),
            correction: value.correction.to_bits(),
        }
    }
}

#[cfg(feature = "serde")]
impl TryFrom<SumState> for CompensatedSum {
    type Error = QmcError;
    fn try_from(state: SumState) -> Result<Self, Self::Error> {
        let sum = Self {
            sum: f64::from_bits(state.sum),
            correction: f64::from_bits(state.correction),
        };
        if !sum.is_finite() {
            return Err(QmcError::NonFiniteValue);
        }
        Ok(sum)
    }
}

impl CompensatedSum {
    pub fn add(&mut self, value: f64) -> Result<(), QmcError> {
        let next = self.sum + value;
        self.correction += if self.sum.abs() >= value.abs() {
            (self.sum - next) + value
        } else {
            (value - next) + self.sum
        };
        self.sum = next;
        if !self.is_finite() {
            return Err(QmcError::NumericOverflow);
        }
        Ok(())
    }

    pub fn merge(&mut self, other: Self) -> Result<(), QmcError> {
        self.add(other.sum)?;
        self.add(other.correction)
    }

    pub fn total(self) -> f64 {
        self.sum + self.correction
    }

    pub fn is_finite(self) -> bool {
        self.sum.is_finite() && self.correction.is_finite() && self.total().is_finite()
    }
}
