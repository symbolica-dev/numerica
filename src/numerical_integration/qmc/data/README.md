# Bundled lattice data

`kuo-33002.u64le` contains 9,125 little-endian unsigned 64-bit integers, with no
header. These are mathematical generating-vector data, not integration code.
The point generator, randomizations and statistical estimators are independent
Rust implementations in Numerica. No Python package is needed to use the data.

The original rule is Frances Kuo's
`lattice-33002-1024-1048576.9125`: an extensible base-two rank-one rule with
order-three weights (Gamma_1 = Gamma_2 = 1, Gamma_3 = 0.5), constructed for powers
of two from 1,024 through 1,048,576 points. The published range is enforced; it
must not be confused with the integer word size or extrapolated silently.

- Author's publication and description: <https://web.maths.unsw.edu.au/~fkuo/lattice/>
- Related paper: R. Cools, F. Y. Kuo and D. Nuyens, *Constructing embedded lattice
  rules for multivariate integration*, SIAM J. Sci. Comput. 28(6), 2162–2188 (2006).
- Distribution used here: QMCPy, commit
  `0f6d3c28f5fdd7effb1c883bc5d0351d50987427`, file
  `qmcpy/discrete_distribution/lattice/generating_vectors/kuo.lattice-33002-1024-1048576.9125.npy`.
- Source URL: <https://github.com/QMCSoftware/qmcpy/blob/0f6d3c28f5fdd7effb1c883bc5d0351d50987427/qmcpy/discrete_distribution/lattice/generating_vectors/kuo.lattice-33002-1024-1048576.9125.npy>

The third-party data are distributed under QMCPy's Apache License 2.0, included
as `LICENSE-APACHE-2.0`. Copyright 2021 Illinois Institute of Technology.
The surrounding Numerica implementation remains MIT-licensed. This file records
the format change: the original NumPy 1.0 header (128 bytes, dtype `<u8`, shape
`(9125,)`, C order) was removed; every payload byte is unchanged.

Source SHA-256:
`dbb76538e5dc9249f1a8de2b73df6f3d8c4ffec86f07b32df5e481441e712a3b`

Payload SHA-256:
`f1ea2884947828c718ddff18540d55ef5415cfb9aec55c1b88b8e6f50c707fbb`

Reproduce the conversion with `dd if=SOURCE.npy of=kuo-33002.u64le bs=1 skip=128`.
This is a one-time data import; neither builds nor execution download data.

## Additional explicit catalogues

`Rank1Rule::published` also supports these attributed numerical datasets:

| Catalogue | Source text | Dimension | Complete power-of-two counts |
| --- | --- | --- | --- |
| Kuo38005 | `kuo.lattice-38005-1024-1048576.5000.txt` | 5000 | 1024 through 1048576 |
| Kuo39101 | `kuo.lattice-39101-1024-1048576.3600.txt` | 3600 | 1024 through 1048576 |
| HKKN alpha3 | `mps.exew_base2_m20_a3_HKKN.txt` | 10 | 2 through 1048576 |

Kuo38005 has equal product weights gamma_j=0.05; Kuo39101 has decaying
product weights gamma_j=1/j. The author describes both at
<https://web.maths.unsw.edu.au/~fkuo/lattice/>. HKKN alpha3 has equal weights
in a Korobov space of smoothness three, from F. J. Hickernell, P. Kritzer,
F. Y. Kuo and D. Nuyens, *Weighted compound integration rules with higher
order convergence for all N*, Numerical Algorithms 59, 161–183 (2012),
<https://doi.org/10.1007/s11075-011-9482-5>. The author's numerical file is
<https://people.cs.kuleuven.be/~dirk.nuyens/qmc-generators/LATSEQ/exew_base2_m20_a3_HKKN.txt>.
Only complete base-two sets are exposed here; arbitrary all-N sequence ordering
is not implemented. Small-count support is a capability, not an accuracy claim.

The imported source is the QMCSoftware-maintained LDData distribution, revision
`5c55b76bf6b6aba3415a0dece9cc1269ff1be883`:
<https://huggingface.co/datasets/Sou-Cheng/LDData/tree/5c55b76bf6b6aba3415a0dece9cc1269ff1be883>.
Its Apache2 license notice is preserved in `LDDATA-LICENSE-NOTICE.txt`; the
complete license is `LICENSE-APACHE-2.0`. Original text, attribution comments
and headers are preserved alongside the compact payloads. Every source
component was independently compared against the original author's file;
all 10/5000/3600 integers match.

The format change discards comments and the first two numeric records
(dimension and maximum count), then writes each vector component as an
unsigned 64-bit little-endian integer. No numerical value is changed. Point
generation, randomization and statistics remain the independent Rust code in
Numerica; no external integration code is copied or needed.

| File | SHA256 |
| --- | --- |
| HKKN source text | `c423a2d92f8d891fff91aebd5ceb224b0a6c154f39da1964b55245296689364c` |
| `hkkn-alpha3.u64le` | `e961023a9667d081a4fe782a5f9bb624e2c99e2fc0e20a875552c5a86939013f` |
| Kuo38005 source text | `d6558c9dac142f871d92f83c0d32f83da500021c2580d1c954e4570362515e8b` |
| `kuo-38005.u64le` | `101369d627429d77ac61b5290dffdc27ea97b0fa71632c27d4fbe088db0345be` |
| Kuo39101 source text | `2a17256bba283d44ea3c8488c5ccb5a964a79150bc5a288ba54c8330ac0e7d2d` |
| `kuo-39101.u64le` | `682ebbdea068592cf5d702f24b9033e76880ad0b21c74cfdac4877316d6e66f8` |

The historical `kuo` constructor and `Kuo33002` serialized provenance keep
their original meaning. These options do not change the default, search for
a better vector or silently fall back when a catalogue dimension is exceeded.
