use super::rule::RuleSource;

/// A published generating-vector catalogue, not a construction algorithm.
///
/// Each catalogue supplies prefixes of one extensible vector. Only complete
/// power-of-two point sets in its published range are supported here. Bounds
/// describe available data, not an integrand-independent accuracy guarantee.
/// Provenance, original text, licenses and payload hashes are in `qmc/data`.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PublishedLattice {
    /// Extensible order-three weights; the historical `Rank1Rule::kuo` rule.
    Kuo33002,
    /// Equal product weights, gamma_j = 0.05.
    Kuo38005,
    /// Decaying product weights, gamma_j = 1/j.
    Kuo39101,
    /// Ten-dimensional equal-weight Korobov alpha-three rule (HKKN 2011).
    HkknAlpha3,
}

impl PublishedLattice {
    /// Smallest complete set supported by this catalogue constructor.
    pub const fn min_points(self) -> u64 {
        match self {
            Self::HkknAlpha3 => 2,
            _ => 1 << 10,
        }
    }

    pub const fn max_points(self) -> u64 {
        1 << 20
    }

    pub const fn max_dimension(self) -> usize {
        match self {
            Self::Kuo33002 => 9125,
            Self::Kuo38005 => 5000,
            Self::Kuo39101 => 3600,
            Self::HkknAlpha3 => 10,
        }
    }

    pub(super) fn data(self) -> &'static [u8] {
        match self {
            Self::Kuo33002 => include_bytes!("data/kuo-33002.u64le"),
            Self::Kuo38005 => include_bytes!("data/kuo-38005.u64le"),
            Self::Kuo39101 => include_bytes!("data/kuo-39101.u64le"),
            Self::HkknAlpha3 => include_bytes!("data/hkkn-alpha3.u64le"),
        }
    }

    pub(super) const fn source(self) -> RuleSource {
        match self {
            Self::Kuo33002 => RuleSource::Kuo33002,
            Self::Kuo38005 => RuleSource::Kuo38005,
            Self::Kuo39101 => RuleSource::Kuo39101,
            Self::HkknAlpha3 => RuleSource::HkknAlpha3,
        }
    }
}
