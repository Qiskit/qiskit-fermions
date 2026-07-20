// This code is a Qiskit project.
//
// (C) Copyright IBM 2026.
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! Full configuration interaction (FCI) string-addressing primitives.
//!
//! An FCI state vector for a fixed particle-number sector enumerates all determinants with a given
//! number of occupied orbitals. Applying fermionic ladder operators to such a vector requires a
//! bijection between an occupation string and its linear address, plus the fermionic sign that
//! accompanies each creation/annihilation. This module provides exactly those primitives, matching
//! the conventions of `pyscf.fci.cistring` (which `ffsim` and other FCI codes use), so that a native
//! matrix-vector kernel built on top of them produces a state vector in the same basis ordering.
//!
//! Conventions (verified against `pyscf.fci.cistring`):
//!
//! * An occupation string is a bitmask where bit `p` is set iff spatial orbital `p` is occupied.
//! * Determinants are ordered by ascending bitmask value; equivalently, for occupied orbitals
//!   ``p_0 < p_1 < ... < p_{k-1}`` the address is the combinatorial-number-system rank
//!   ``addr = sum_j C(p_j, j + 1)``.
//! * A creation on orbital `p` (valid only if `p` is empty) and an annihilation on orbital `p`
//!   (valid only if `p` is occupied) both carry the sign ``(-1)^m`` where `m` is the number of
//!   occupied orbitals with index strictly greater than `p`.

use num_complex::Complex64;

/// The largest number of spatial orbitals supported by the bitmask representation.
///
/// Occupation strings are stored in a `u64`, so at most 64 orbitals can be represented. FCI
/// dimensions become astronomically large long before this limit, so it is not a practical
/// restriction.
pub const MAX_ORBITALS: u32 = 64;

/// A precomputed table of binomial coefficients ``C(n, k)``.
///
/// The table is triangular with rows ``0..=norb`` and, in row `n`, columns ``0..=n``. It is used
/// both to size the FCI space (``C(norb, nocc)``) and to compute combinatorial-number-system ranks.
#[derive(Clone, Debug)]
pub struct BinomialTable {
    norb: u32,
    // Row-major storage of a triangular table; `offsets[n]` is the start of row `n`.
    data: Vec<usize>,
    offsets: Vec<usize>,
}

impl BinomialTable {
    /// Builds a binomial table covering ``C(n, k)`` for all ``0 <= k <= n <= norb``.
    pub fn new(norb: u32) -> Self {
        assert!(
            norb <= MAX_ORBITALS,
            "norb={norb} exceeds MAX_ORBITALS={MAX_ORBITALS}"
        );
        let norb_usize = norb as usize;
        let mut offsets = Vec::with_capacity(norb_usize + 2);
        let mut total = 0;
        for n in 0..=norb_usize {
            offsets.push(total);
            total += n + 1;
        }
        offsets.push(total);

        let mut data = vec![0usize; total];
        // Pascal's triangle: C(n, 0) = C(n, n) = 1, C(n, k) = C(n-1, k-1) + C(n-1, k).
        for n in 0..=norb_usize {
            let row = offsets[n];
            data[row] = 1;
            data[row + n] = 1;
            if n >= 2 {
                let prev = offsets[n - 1];
                for k in 1..n {
                    data[row + k] = data[prev + k - 1] + data[prev + k];
                }
            }
        }
        Self {
            norb,
            data,
            offsets,
        }
    }

    /// Returns ``C(n, k)``, or `0` when `k > n` (as is conventional for counting).
    #[inline]
    pub fn comb(&self, n: u32, k: u32) -> usize {
        if k > n {
            return 0;
        }
        debug_assert!(
            n <= self.norb,
            "n={n} out of range for table built with norb={}",
            self.norb
        );
        self.data[self.offsets[n as usize] + k as usize]
    }

    /// Returns the dimension ``C(norb, nocc)`` of the FCI space for `nocc` occupied orbitals.
    #[inline]
    pub fn num_strings(&self, norb: u32, nocc: u32) -> usize {
        self.comb(norb, nocc)
    }

    /// The occupation strings of the ``(norb, nocc)`` sector, addressed by their linear index.
    ///
    /// Element `addr` is the occupation string with [`str2addr`] equal to `addr`. Precomputing this
    /// list once per sector avoids repeated unranking in the inner matvec loop.
    pub fn sector_strings(&self, norb: u32, nocc: u32) -> Vec<u64> {
        (0..self.num_strings(norb, nocc))
            .map(|addr| addr2str(self, norb, nocc, addr))
            .collect()
    }
}

/// Returns the highest orbital index set in `string` if it lies outside `0..norb`, else `None`.
///
/// Used to validate occupation masks whose set bits must all address orbitals below `norb` before
/// they are ranked by [`str2addr`] (an out-of-range bit would rank past the sector dimension).
#[inline]
fn mode_out_of_range(string: u64, norb: u32) -> Option<u32> {
    // `1u64 << 64` would overflow; norb >= 64 admits every 64-bit mask (norb == 64 is the ceiling
    // enforced by `BinomialTable::new`, and higher is rejected upstream).
    if norb >= 64 || string >> norb == 0 {
        None
    } else {
        // 63 minus the leading zeros is the highest set bit; it is `>= norb` by the check above.
        Some(63 - string.leading_zeros())
    }
}

/// Computes the combinatorial-number-system address of an occupation `string`.
///
/// `string`'s set bits must all lie within ``0..norb`` (the caller ensures this); the address is
/// ``sum_j C(p_j, j + 1)`` over the occupied orbitals `p_j` in ascending order, matching
/// `pyscf.fci.cistring.str2addr`. Called in the inner scatter loops of [`SpinlessSector::compile`]
/// and [`SpinfulSector::compile`], hence `#[inline]`.
#[inline]
pub fn str2addr(table: &BinomialTable, string: u64) -> usize {
    let mut addr = 0;
    let mut remaining = string;
    let mut which = 1u32; // 1-based position of the occupied orbital (j + 1)
    while remaining != 0 {
        let orb = remaining.trailing_zeros();
        addr += table.comb(orb, which);
        which += 1;
        remaining &= remaining - 1; // clear the lowest set bit
    }
    addr
}

/// Computes the occupation string at address `addr` in the ``(norb, nocc)`` sector.
///
/// This inverts [`str2addr`]: it walks orbitals from high to low, greedily placing each of the
/// `nocc` electrons in the highest orbital whose combinatorial weight does not exceed the remaining
/// address. Matches `pyscf.fci.cistring.addr2str`.
pub fn addr2str(table: &BinomialTable, norb: u32, nocc: u32, addr: usize) -> u64 {
    debug_assert!(
        addr < table.num_strings(norb, nocc),
        "addr={addr} out of range for C({norb}, {nocc})"
    );
    let mut string = 0u64;
    let mut remaining = addr;
    // Assign the electrons from the highest-indexed (position `nocc`) down to the first.
    for which in (1..=nocc).rev() {
        // Find the largest orbital `orb` with C(orb, which) <= remaining. Iterating `(0..norb).rev()`
        // (rather than an unbounded `loop { orb -= 1; .. }`) keeps `orb` from underflowing past 0 in
        // release builds when `addr` is out of range for the sector -- the `debug_assert!` above
        // catches that in debug, and the explicit `expect` below turns the same violation into a
        // clear panic in release instead of a `u32` subtract-overflow or an out-of-bounds table read.
        let orb = (0..norb)
            .rev()
            .find(|&orb| table.comb(orb, which) <= remaining)
            .expect("addr out of range for the (norb, nocc) sector");
        string |= 1u64 << orb;
        remaining -= table.comb(orb, which);
    }
    string
}

/// The result of applying a single ladder operator to an occupation string.
///
/// `Vanishes` indicates the operator annihilates the state (creating in an occupied orbital or
/// destroying an empty one). `Maps` carries the resulting string and the accompanying fermionic
/// sign (`+1` or `-1`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LadderResult {
    /// The operator maps the input string to `string` with the given `sign`.
    Maps { string: u64, sign: i8 },
    /// The operator annihilates the state (result is the zero vector).
    Vanishes,
}

/// The fermionic sign accompanying a ladder operator acting on orbital `p` of `string`.
///
/// Both creation and annihilation carry ``(-1)^m`` where `m` is the number of occupied orbitals
/// with index strictly greater than `p` (the `pyscf.fci.cistring` convention). Only bits above `p`
/// contribute, so the input `string` may be taken either before or after toggling bit `p`.
#[inline]
fn ladder_sign(string: u64, p: u32) -> i8 {
    // Mask selecting bits strictly above p; for p == 63 this is 0.
    let above_mask = if p >= 63 { 0 } else { !0u64 << (p + 1) };
    let parity = (string & above_mask).count_ones() & 1;
    if parity == 0 { 1 } else { -1 }
}

/// Applies a single ladder operator on orbital `p` to `string`.
///
/// With `is_creation == true` this is a creation operator (vanishing when `p` is already occupied);
/// otherwise an annihilation operator (vanishing when `p` is empty). The two differ only in that
/// vanish predicate: on the surviving branch each simply toggles bit `p` (`string ^ bit`) and carries
/// the same [`ladder_sign`]. Returns [`LadderResult::Vanishes`] when the operator annihilates the
/// state.
#[inline]
pub fn apply_ladder_op(string: u64, p: u32, is_creation: bool) -> LadderResult {
    let bit = 1u64 << p;
    // Creation vanishes on an occupied orbital; annihilation vanishes on an empty one.
    if (string & bit != 0) == is_creation {
        return LadderResult::Vanishes;
    }
    LadderResult::Maps {
        string: string ^ bit,
        sign: ladder_sign(string, p),
    }
}

// -------------------------------------------------------------------------------------------------
// Matrix-vector kernel
// -------------------------------------------------------------------------------------------------
//
// The kernel applies a `FermionOperator` to an FCI state vector, producing `out += op @ vec` in the
// same basis ordering that `pyscf.fci.cistring` (and hence `ffsim`) use. Two mode interpretations
// are supported, selected by the caller (mirroring ffsim's `norb, nelec` convention):
//
// * **Spinful** (`n_beta` is `Some`): the operator's `2 * norb` modes are spin-orbitals under the
//   block-spin ordering `alpha 0..norb`, then `beta 0..norb`. Mode `m < norb` acts on alpha orbital
//   `m`; mode `m >= norb` acts on beta orbital `m - norb`. The state vector has length
//   `dim_a * dim_b` with the flat index `addr_a * dim_b + addr_b` (alpha slow, beta fast), where
//   `dim_s = C(norb, n_s)`.
// * **Spinless** (`n_beta` is `None`): the operator's `norb` modes are orbitals directly. The state
//   vector has length `C(norb, n_alpha)` indexed by `str2addr` of the occupation string.
//
// Within a spin sector the fermionic sign of a single ladder operator matches `pyscf`'s
// `gen_cre_str_index`/`gen_des_str_index` (see [`apply_ladder_op`]), so composing the per-op maps
// of a term right-to-left reproduces ffsim's per-sector sign. The only cross-sector contribution is
// that, under block-spin ordering, every occupied alpha orbital sits below every beta orbital:
// hence each *beta* ladder operator additionally carries `(-1)^{n_alpha}` (the current alpha
// electron count, which is invariant across a particle-conserving term).

/// Errors that can arise while applying a `FermionOperator` to an FCI state vector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FciMatvecError {
    /// The provided vector length does not match the FCI dimension implied by `(norb, nelec)`.
    DimensionMismatch { expected: usize, actual: usize },
    /// A term acts on a mode outside the range implied by `norb` (`[0, norb)` when spinless, else
    /// `[0, 2 * norb)`).
    ModeOutOfRange { mode: u32, num_modes: u32 },
    /// The spinful FCI dimension `C(norb, n_alpha) * C(norb, n_beta)` overflows `usize`.
    DimensionOverflow { dim_a: usize, dim_b: usize },
}

impl std::fmt::Display for FciMatvecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FciMatvecError::DimensionMismatch { expected, actual } => write!(
                f,
                "state vector length {actual} does not match FCI dimension {expected}"
            ),
            FciMatvecError::ModeOutOfRange { mode, num_modes } => write!(
                f,
                "mode {mode} is outside the range [0, {num_modes}) implied by the number of orbitals"
            ),
            FciMatvecError::DimensionOverflow { dim_a, dim_b } => write!(
                f,
                "spinful FCI dimension {dim_a} * {dim_b} overflows the addressable range"
            ),
        }
    }
}

impl std::error::Error for FciMatvecError {}

/// Computes the spinful FCI dimension `C(norb, n_alpha) * C(norb, n_beta)`, checking for overflow.
///
/// The per-sector dimensions each fit in `usize`, but their product need not; a silent wrap could
/// defeat the vector-length check and corrupt the matvec, so overflow is reported as an error.
pub fn spinful_dim(
    table: &BinomialTable,
    norb: u32,
    n_alpha: u32,
    n_beta: u32,
) -> Result<usize, FciMatvecError> {
    let dim_a = table.num_strings(norb, n_alpha);
    let dim_b = table.num_strings(norb, n_beta);
    dim_a
        .checked_mul(dim_b)
        .ok_or(FciMatvecError::DimensionOverflow { dim_a, dim_b })
}

/// Builds the FCI state vector of a single occupation determinant (a one-hot at its address).
///
/// The occupied orbitals of each spin sector are given as bitmasks (bit `p` set iff orbital `p` is
/// occupied), matching [`str2addr`]'s convention. `beta_str` selects the mode interpretation exactly
/// as the matvec kernels do:
///
/// * **Spinless** (`beta_str` is `None`): the vector has length `C(norb, n_alpha)` and a single `1`
///   at `str2addr(alpha_str)`, where `n_alpha` is the population count of `alpha_str`.
/// * **Spinful** (`beta_str` is `Some`): the vector has length `dim_a * dim_b` with the flat index
///   `str2addr(alpha_str) * dim_b + str2addr(beta_str)` (alpha slow, beta fast), matching the
///   block-spin ordering used throughout this module.
///
/// This is the seeding counterpart of the matvec kernels: it produces, in the same basis ordering,
/// the determinant that a Jordan-Wigner occupation prepares from the vacuum. The result is a genuine
/// determinant (sign `+1`) because the occupied orbitals are addressed directly by their bitmask,
/// which is inherently sorted.
///
/// The population counts of the masks define the target sector; the caller is responsible for
/// ensuring they match the intended `(n_alpha, n_beta)`. Errors are limited to a spinful dimension
/// that overflows `usize` (via [`spinful_dim`]).
pub fn slater_determinant_statevector(
    norb: u32,
    alpha_str: u64,
    beta_str: Option<u64>,
) -> Result<Vec<Complex64>, FciMatvecError> {
    // Reject occupations that set a bit outside `0..norb`; such a string would rank past the sector
    // dimension and index the output vector out of bounds. The highest set bit must be below `norb`.
    if let Some(highest) = mode_out_of_range(alpha_str, norb) {
        return Err(FciMatvecError::ModeOutOfRange {
            mode: highest,
            num_modes: norb,
        });
    }
    if let Some(beta_str) = beta_str
        && let Some(highest) = mode_out_of_range(beta_str, norb)
    {
        return Err(FciMatvecError::ModeOutOfRange {
            mode: highest,
            num_modes: norb,
        });
    }

    let table = BinomialTable::new(norb);
    let n_alpha = alpha_str.count_ones();
    let addr_a = str2addr(&table, alpha_str);

    match beta_str {
        None => {
            let dim = table.num_strings(norb, n_alpha);
            let mut vec = vec![Complex64::new(0.0, 0.0); dim];
            vec[addr_a] = Complex64::new(1.0, 0.0);
            Ok(vec)
        }
        Some(beta_str) => {
            let n_beta = beta_str.count_ones();
            let dim = spinful_dim(&table, norb, n_alpha, n_beta)?;
            let dim_b = table.num_strings(norb, n_beta);
            let addr_b = str2addr(&table, beta_str);
            let mut vec = vec![Complex64::new(0.0, 0.0); dim];
            vec[addr_a * dim_b + addr_b] = Complex64::new(1.0, 0.0);
            Ok(vec)
        }
    }
}

/// Applies a single term's ladder operators (right-to-left) to one sector's occupation `string`.
///
/// `ops` are `(is_creation, orbital)` pairs in the term's written (left-to-right) order; they are
/// applied to the ket from the right. Returns the resulting `(string, sign)`, or `None` if the term
/// annihilates the state (creating in an occupied orbital or destroying an empty one).
#[inline]
fn apply_ops_to_string(string: u64, ops: &[(bool, u32)]) -> Option<(u64, i8)> {
    let mut string = string;
    let mut sign: i8 = 1;
    for &(is_creation, orb) in ops.iter().rev() {
        match apply_ladder_op(string, orb, is_creation) {
            LadderResult::Maps {
                string: next,
                sign: s,
            } => {
                string = next;
                sign *= s;
            }
            LadderResult::Vanishes => return None,
        }
    }
    Some((string, sign))
}

/// A coalesced list of single-block transitions `out[dst] += scale * vec[src]`.
///
/// The three arrays are parallel: entry `k` is `(src[k], dst[k], scale[k])`. Produced by
/// [`coalesce`], which sums the `scale` of every raw transition sharing a `(src, dst)` key, so no two
/// entries here repeat a `(src, dst)` pair. Used for the spinful [`CompiledKind::Spinful`] fast paths
/// where one spin block is the identity (`alpha_only` / `beta_only`).
#[derive(Clone, Debug, Default)]
struct ScaledTransitions {
    src: Vec<usize>,
    dst: Vec<usize>,
    scale: Vec<Complex64>,
}

/// One spin block of the mixed terms, as a CSR-segmented list of phased transitions.
///
/// Mixed terms (acting non-trivially on both spins) are kept *factored*: rather than materializing
/// the full `|alpha_trans| * |beta_trans|` cross product of block-spin flat entries, each term stores
/// its alpha transitions here and its beta transitions in a parallel [`PhasedTransitions`], and the
/// two are contracted at apply-time. Term `t` owns the transitions `indptr[t]..indptr[t + 1]`; entry
/// `k` in that range is `(src[k], dst[k])` with fermionic phase `phase[k]` (always +/-1, so an `i8`).
/// The per-term complex coefficient (with the cross-sector sign folded in) lives alongside in
/// [`CompiledKind::Spinful::mixed_coeffs`]. This trades the dense product's memory/time for a
/// per-term double loop, iterating the shorter block outer.
#[derive(Clone, Debug, Default)]
struct PhasedTransitions {
    indptr: Vec<usize>,
    src: Vec<usize>,
    dst: Vec<usize>,
    phase: Vec<i8>,
}

impl PhasedTransitions {
    /// Starts a fresh CSR built term-by-term. `indptr` begins with the single boundary `[0]`.
    fn new() -> Self {
        Self {
            indptr: vec![0],
            ..Default::default()
        }
    }

    /// Appends one transition to the segment currently being built (before [`Self::finish_term`]).
    fn push(&mut self, src: usize, dst: usize, phase: i8) {
        self.src.push(src);
        self.dst.push(dst);
        self.phase.push(phase);
    }

    /// Closes the current term's segment, recording its end boundary in `indptr`.
    fn finish_term(&mut self) {
        self.indptr.push(self.src.len());
    }
}

/// Coalesces raw `(src, dst, weight)` transitions, summing the weights of every pair sharing the same
/// `(src, dst)` key into a single [`ScaledTransitions`] entry.
///
/// This is the Rust analog of ffsim's `lexsort` + `add.reduceat`: distinct operator terms (or distinct
/// source determinants) that scatter into the same `(src, dst)` slot contribute one fused float
/// multiply per matvec instead of one each. Entries whose fused weight is exactly zero are dropped.
fn coalesce(mut raw: Vec<(usize, usize, Complex64)>) -> ScaledTransitions {
    raw.sort_unstable_by_key(|&(src, dst, _)| (src, dst));
    let mut out = ScaledTransitions::default();
    for (src, dst, weight) in raw {
        if let (Some(&last_src), Some(&last_dst)) = (out.src.last(), out.dst.last())
            && last_src == src
            && last_dst == dst
        {
            *out.scale.last_mut().unwrap() += weight;
        } else {
            out.src.push(src);
            out.dst.push(dst);
            out.scale.push(weight);
        }
    }
    // Drop exactly-zero fused weights (e.g. `n_0 - n_0`); they contribute nothing to any matvec.
    let mut k = 0;
    for i in 0..out.scale.len() {
        if out.scale[i] != Complex64::new(0.0, 0.0) {
            out.src[k] = out.src[i];
            out.dst[k] = out.dst[i];
            out.scale[k] = out.scale[i];
            k += 1;
        }
    }
    out.src.truncate(k);
    out.dst.truncate(k);
    out.scale.truncate(k);
    out
}

/// The compiled scatter data, specialized by sector kind.
///
/// The spinless case is a single coalesced flat scatter. The spinful case is *categorized* by which
/// spin blocks a term acts on non-trivially, so that identity blocks are swept cheaply rather than
/// materialized as dense diagonal entries:
///
/// * `scalar` -- terms that are the identity on both spins, summed into one coefficient (applied as
///   `out += scalar * vec`).
/// * `alpha_only` / `beta_only` -- terms that are the identity on the *other* spin, kept as coalesced
///   single-block transitions and applied with an inner sweep over the untouched block.
/// * `mixed` -- terms acting non-trivially on both spins, kept *factored* into per-block CSR
///   transitions (`mixed_alpha` / `mixed_beta`) plus a per-term coefficient (`mixed_coeffs`), and
///   contracted at apply-time. This avoids materializing the `|alpha_trans| * |beta_trans|` dense
///   cross product per term. See [`PhasedTransitions`].
#[derive(Clone, Debug)]
enum CompiledKind {
    Spinless {
        entries: Vec<(usize, usize, Complex64)>,
    },
    // Boxed to keep the two variants close in size (the spinful payload carries several vectors);
    // the box is dereferenced once per `contract` call, never in the hot loop.
    Spinful(Box<SpinfulCompiled>),
}

/// The compiled spinful payload (see [`CompiledKind::Spinful`]).
#[derive(Clone, Debug)]
struct SpinfulCompiled {
    dim_a: usize,
    dim_b: usize,
    scalar: Complex64,
    alpha_only: ScaledTransitions,
    beta_only: ScaledTransitions,
    /// One complex coefficient per mixed term (with the cross-sector sign folded in), parallel to
    /// the per-term segments of `mixed_alpha` / `mixed_beta`.
    mixed_coeffs: Vec<Complex64>,
    mixed_alpha: PhasedTransitions,
    mixed_beta: PhasedTransitions,
}

/// An operator compiled into a reusable scatter map for a fixed sector.
///
/// Built once by [`SpinlessSector::compile`] or [`SpinfulSector::compile`] and reused across the
/// repeated matvecs (and rmatvecs, via [`Self::apply_conj`]) of a single `expm_multiply`, this
/// replaces the per-call ladder walk / conservation check / `str2addr` rank with a plain
/// gather-scatter. Addresses are sector-local: for the spinless case they are `str2addr` ranks; for
/// the spinful case they are the block-spin flat index `addr_a * dim_b + addr_b`. See [`CompiledKind`]
/// for how spinful terms are categorized so identity spin blocks cost a cheap sweep.
#[derive(Clone, Debug)]
pub struct CompiledSector {
    dim: usize,
    kind: CompiledKind,
}

impl CompiledSector {
    /// The FCI dimension of the sector this map was compiled for.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Applies the compiled operator to a state vector: `out = op @ vec`.
    ///
    /// `vec` must have length [`Self::dim`]. Validated bit-for-bit in the tests against an
    /// independent naive Jordan-Wigner reference matvec for the same operator and sector.
    pub fn apply(&self, vec: &[Complex64]) -> Result<Vec<Complex64>, FciMatvecError> {
        self.contract(vec, false)
    }

    /// Applies the compiled operator's conjugate transpose to a state vector: `out = op^H @ vec`.
    ///
    /// The conjugate transpose of a scatter `src -> dst` with weight `w` is `dst -> src` with weight
    /// `conj(w)`, so the same compiled data backs both the forward apply and this adjoint apply
    /// (`matvec` and `rmatvec`) -- no separate adjoint operator need be built or held. `vec` must
    /// have length [`Self::dim`].
    pub fn apply_conj(&self, vec: &[Complex64]) -> Result<Vec<Complex64>, FciMatvecError> {
        self.contract(vec, true)
    }

    /// Shared contraction for [`Self::apply`] (`conj = false`) and [`Self::apply_conj`]
    /// (`conj = true`). When `conj`, every scatter runs `src -> dst` reversed with the weight
    /// conjugated. Fermionic phases are real +/-1, so conjugation only touches the complex parts:
    /// the folded `scale` of the single-block paths and the per-term `mixed_coeffs` of the mixed path.
    fn contract(&self, vec: &[Complex64], conj: bool) -> Result<Vec<Complex64>, FciMatvecError> {
        if vec.len() != self.dim {
            return Err(FciMatvecError::DimensionMismatch {
                expected: self.dim,
                actual: vec.len(),
            });
        }
        let zero = Complex64::new(0.0, 0.0);
        let mut out = vec![zero; self.dim];
        match &self.kind {
            CompiledKind::Spinless { entries } => {
                for &(src, dst, weight) in entries {
                    let weight = if conj { weight.conj() } else { weight };
                    let (src, dst) = if conj { (dst, src) } else { (src, dst) };
                    out[dst] += weight * vec[src];
                }
            }
            CompiledKind::Spinful(spinful) => {
                let SpinfulCompiled {
                    dim_a,
                    dim_b,
                    scalar,
                    alpha_only,
                    beta_only,
                    mixed_coeffs,
                    mixed_alpha,
                    mixed_beta,
                } = spinful.as_ref();
                // Scalar (identity on both spins): a single diagonal scale over the whole vector.
                let scalar = if conj { scalar.conj() } else { *scalar };
                if scalar != zero {
                    for (o, v) in out.iter_mut().zip(vec) {
                        *o += scalar * v;
                    }
                }
                // Alpha-only (beta identity): each alpha transition sweeps the untouched beta block.
                for k in 0..alpha_only.scale.len() {
                    let scale = if conj {
                        alpha_only.scale[k].conj()
                    } else {
                        alpha_only.scale[k]
                    };
                    let (sa, da) = if conj {
                        (alpha_only.dst[k], alpha_only.src[k])
                    } else {
                        (alpha_only.src[k], alpha_only.dst[k])
                    };
                    let (src_off, dst_off) = (sa * dim_b, da * dim_b);
                    for b in 0..*dim_b {
                        out[dst_off + b] += scale * vec[src_off + b];
                    }
                }
                // Beta-only (alpha identity): sweep the untouched alpha rows, applying beta transitions.
                for a in 0..*dim_a {
                    let row = a * dim_b;
                    for k in 0..beta_only.scale.len() {
                        let scale = if conj {
                            beta_only.scale[k].conj()
                        } else {
                            beta_only.scale[k]
                        };
                        let (sb, db) = if conj {
                            (beta_only.dst[k], beta_only.src[k])
                        } else {
                            (beta_only.src[k], beta_only.dst[k])
                        };
                        out[row + db] += scale * vec[row + sb];
                    }
                }
                // Mixed (both spins active): contract each term's factored alpha/beta blocks.
                // `out[a_dst*dim_b + b_dst] += coeff * a_phase * b_phase * vec[a_src*dim_b + b_src]`,
                // where the alpha and beta transitions are the term's CSR segments. The `conj` /
                // shorter-block dispatch is resolved once per term (not per inner iteration), and the
                // inner block's +/-1 phase is applied by conditional negation rather than a complex
                // multiply -- both keep the innermost loop a single fused-multiply-add.
                for (t, &coeff) in mixed_coeffs.iter().enumerate() {
                    let coeff = if conj { coeff.conj() } else { coeff };
                    let (a0, a1) = (mixed_alpha.indptr[t], mixed_alpha.indptr[t + 1]);
                    let (b0, b1) = (mixed_beta.indptr[t], mixed_beta.indptr[t + 1]);
                    // Iterate the shorter block outer so `coeff * outer_phase` is hoisted for the
                    // smaller factor and the inner loop runs over the longer one. `(o_block, i_block)`
                    // and the `(outer, inner) -> (alpha, beta)` remap are chosen once per term.
                    let dim_b = *dim_b;
                    let alpha_outer = a1 - a0 <= b1 - b0;
                    let (o0, o1, i0, i1) = if alpha_outer {
                        (a0, a1, b0, b1)
                    } else {
                        (b0, b1, a0, a1)
                    };
                    let (o_block, i_block) = if alpha_outer {
                        (mixed_alpha, mixed_beta)
                    } else {
                        (mixed_beta, mixed_alpha)
                    };
                    for o in o0..o1 {
                        let (o_src, o_dst) = if conj {
                            (o_block.dst[o], o_block.src[o])
                        } else {
                            (o_block.src[o], o_block.dst[o])
                        };
                        // Fold the outer block's +/-1 phase into the coefficient once per outer entry.
                        let scale = coeff * f64::from(o_block.phase[o]);
                        for i in i0..i1 {
                            let (i_src, i_dst) = if conj {
                                (i_block.dst[i], i_block.src[i])
                            } else {
                                (i_block.src[i], i_block.dst[i])
                            };
                            // Remap (outer, inner) back to (alpha, beta) for the block-spin flat index
                            // `alpha_addr * dim_b + beta_addr`.
                            let (a_src, b_src, a_dst, b_dst) = if alpha_outer {
                                (o_src, i_src, o_dst, i_dst)
                            } else {
                                (i_src, o_src, i_dst, o_dst)
                            };
                            out[a_dst * dim_b + b_dst] +=
                                scale * f64::from(i_block.phase[i]) * vec[a_src * dim_b + b_src];
                        }
                    }
                }
            }
        }
        Ok(out)
    }
}

/// A spinless FCI sector's precomputed geometry, reusable across many matvec calls.
///
/// Owns the [`BinomialTable`] and the sector's occupation [`sector_strings`], both of which depend
/// only on `(norb, nocc)` -- not on the operator or the input vector. Building them once and reusing
/// the context across the repeated matvecs of a single `expm_multiply` (the evolution path) avoids
/// rebuilding this identical geometry on every call.
#[derive(Clone, Debug)]
pub struct SpinlessSector {
    norb: u32,
    nocc: u32,
    table: BinomialTable,
    strings: Vec<u64>,
}

impl SpinlessSector {
    /// Precomputes the `(norb, nocc)` sector geometry (binomial table + occupation strings).
    pub fn new(norb: u32, nocc: u32) -> Self {
        let table = BinomialTable::new(norb);
        let strings = table.sector_strings(norb, nocc);
        Self {
            norb,
            nocc,
            table,
            strings,
        }
    }

    /// The FCI dimension `C(norb, nocc)` of this sector.
    #[inline]
    pub fn dim(&self) -> usize {
        self.table.num_strings(self.norb, self.nocc)
    }

    /// Compiles an operator's terms into a reusable [`CompiledSector`] scatter map: `out = op @ vec`
    /// via [`CompiledSector::apply`].
    ///
    /// The operator's modes must lie in `[0, norb)`. Each term is supplied as `(coeff, actions,
    /// modes)` -- the native slice layout of a
    /// [`crate::operators::fermion_operator::FermionOperatorTermView`] -- where `actions[k]` is
    /// `true` for a creation and `modes[k]` is its orbital, in the term's written (left-to-right)
    /// order.
    ///
    /// The `(source, destination, weight)` scatter is a pure function of the operator and the sector
    /// -- it does not depend on the input vector -- so compiling it once lets the many matvecs of a
    /// single `expm_multiply` reuse the flat scatter, replacing the per-call combinatorics (the
    /// ladder walk, the conservation check, and the `str2addr` rank of each destination) with a plain
    /// gather-scatter. Non-conserving terms are silently dropped (they contribute no entries),
    /// matching the kernel's in-sector projection. Returns an error on a mode out of range.
    pub fn compile<'a>(
        &self,
        terms: impl IntoIterator<Item = (Complex64, &'a [bool], &'a [u32])>,
    ) -> Result<CompiledSector, FciMatvecError> {
        let (norb, nocc) = (self.norb, self.nocc);
        let mut entries: Vec<(usize, usize, Complex64)> = Vec::new();
        let mut ops: Vec<(bool, u32)> = Vec::new();
        for (coeff, actions, modes) in terms {
            ops.clear();
            for (&is_creation, &orb) in actions.iter().zip(modes) {
                if orb >= norb {
                    return Err(FciMatvecError::ModeOutOfRange {
                        mode: orb,
                        num_modes: norb,
                    });
                }
                ops.push((is_creation, orb));
            }
            // Record, for every source determinant, the destination and weight the term produces --
            // independent of any input vector, so the whole map is built once here.
            for (src_addr, &string) in self.strings.iter().enumerate() {
                if let Some((out_string, sign)) = apply_ops_to_string(string, &ops) {
                    // Drop terms that leave the fixed `nocc` sector (the kernel's in-sector projection).
                    if out_string.count_ones() != nocc {
                        continue;
                    }
                    let dst_addr = str2addr(&self.table, out_string);
                    entries.push((src_addr, dst_addr, coeff * f64::from(sign)));
                }
            }
        }
        // Coalesce transitions sharing a `(src, dst)` key so each is one fused multiply per matvec.
        let coalesced = coalesce(entries);
        let entries = (0..coalesced.scale.len())
            .map(|k| (coalesced.src[k], coalesced.dst[k], coalesced.scale[k]))
            .collect();
        Ok(CompiledSector {
            dim: self.dim(),
            kind: CompiledKind::Spinless { entries },
        })
    }
}

/// A spinful FCI sector's precomputed geometry, reusable across many matvec calls.
///
/// Owns the [`BinomialTable`], both spin sectors' occupation [`sector_strings`], and the derived
/// dimensions -- all of which depend only on `(norb, n_alpha, n_beta)`. Building them once and
/// reusing the context across the repeated matvecs of a single `expm_multiply` (the evolution path)
/// avoids rebuilding this identical geometry on every call.
#[derive(Clone, Debug)]
pub struct SpinfulSector {
    norb: u32,
    n_alpha: u32,
    n_beta: u32,
    table: BinomialTable,
    dim: usize,
    dim_b: usize,
    alpha_strings: Vec<u64>,
    beta_strings: Vec<u64>,
}

impl SpinfulSector {
    /// Precomputes the `(norb, n_alpha, n_beta)` sector geometry.
    ///
    /// Returns [`FciMatvecError::DimensionOverflow`] if the spinful dimension
    /// `C(norb, n_alpha) * C(norb, n_beta)` overflows `usize` (see [`spinful_dim`]).
    pub fn new(norb: u32, n_alpha: u32, n_beta: u32) -> Result<Self, FciMatvecError> {
        let table = BinomialTable::new(norb);
        let dim_b = table.num_strings(norb, n_beta);
        let dim = spinful_dim(&table, norb, n_alpha, n_beta)?;
        let alpha_strings = table.sector_strings(norb, n_alpha);
        let beta_strings = table.sector_strings(norb, n_beta);
        Ok(Self {
            norb,
            n_alpha,
            n_beta,
            table,
            dim,
            dim_b,
            alpha_strings,
            beta_strings,
        })
    }

    /// The FCI dimension `C(norb, n_alpha) * C(norb, n_beta)` of this sector.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Compiles an operator's terms into a reusable [`CompiledSector`] scatter map: `out = op @ vec`
    /// via [`CompiledSector::apply`].
    ///
    /// The operator's modes must lie in `[0, 2 * norb)` under the block-spin convention (mode
    /// `m < norb` is alpha orbital `m`; mode `m >= norb` is beta orbital `m - norb`), and the applied
    /// vector length must equal [`Self::dim`] with flat index `addr_a * dim_b + addr_b`. Each term is
    /// supplied as `(coeff, actions, modes)` (see [`SpinlessSector::compile`] for the layout). Each
    /// beta operator additionally contributes `(-1)^{n_alpha}` because, under block-spin ordering,
    /// all `n_alpha` occupied alpha orbitals precede every beta orbital in the Jordan-Wigner string.
    ///
    /// As with [`SpinlessSector::compile`], the flat `(source, destination, weight)` scatter is a pure
    /// function of the operator and the sector, so compiling it once lets an `expm_multiply`'s many
    /// matvecs reuse it. The per-term construction (alpha/beta split, the `(-1)^{n_alpha}`
    /// cross-sector sign folded into the beta action, and the per-spin conservation drops) is
    /// described inline below.
    pub fn compile<'a>(
        &self,
        terms: impl IntoIterator<Item = (Complex64, &'a [bool], &'a [u32])>,
    ) -> Result<CompiledSector, FciMatvecError> {
        let (norb, n_alpha, n_beta) = (self.norb, self.n_alpha, self.n_beta);
        let dim_b = self.dim_b;
        let num_modes = 2 * norb;
        let beta_alpha_parity: i8 = if n_alpha.is_multiple_of(2) { 1 } else { -1 };

        // We categorize each term by which spin blocks it touches, without normal-ordering the
        // operator first. Normal ordering is unnecessary here: `apply_ops_to_string` applies a term's
        // ladder ops right-to-left and computes the exact Jordan-Wigner sign regardless of the written
        // order, and the alpha/beta split below is purely by mode index. A mixed term's weight
        // factorizes as `coeff * a_sign(a) * (cross_sign * b_sign(b))` -- `a_sign` depends only on the
        // alpha transition, `b_sign` only on the beta transition, and `cross_sign` is a per-term
        // constant -- so each block can be compiled independently.
        let mut scalar = Complex64::new(0.0, 0.0);
        let mut alpha_only_raw: Vec<(usize, usize, Complex64)> = Vec::new();
        let mut beta_only_raw: Vec<(usize, usize, Complex64)> = Vec::new();
        let mut mixed_coeffs: Vec<Complex64> = Vec::new();
        let mut mixed_alpha = PhasedTransitions::new();
        let mut mixed_beta = PhasedTransitions::new();
        let mut alpha_ops: Vec<(bool, u32)> = Vec::new();
        let mut beta_ops: Vec<(bool, u32)> = Vec::new();
        // Reused per term: the surviving alpha transitions of a mixed term, as `(src, dst, phase)`.
        // (Beta transitions are appended straight into `mixed_beta`.)
        let mut alpha_trans: Vec<(usize, usize, i8)> = Vec::new();
        for (coeff, actions, modes) in terms {
            alpha_ops.clear();
            beta_ops.clear();
            for (&is_creation, &mode) in actions.iter().zip(modes) {
                if mode >= num_modes {
                    return Err(FciMatvecError::ModeOutOfRange { mode, num_modes });
                }
                if mode < norb {
                    alpha_ops.push((is_creation, mode));
                } else {
                    beta_ops.push((is_creation, mode - norb));
                }
            }

            let cross_sign = if beta_ops.len().is_multiple_of(2) {
                1
            } else {
                beta_alpha_parity
            };

            match (alpha_ops.is_empty(), beta_ops.is_empty()) {
                // Identity on both spins: fold into the scalar diagonal.
                (true, true) => scalar += coeff,
                // Beta identity: an alpha-only term. Weight = coeff * a_sign.
                (false, true) => {
                    for (a_addr, &a_string) in self.alpha_strings.iter().enumerate() {
                        let Some((a_out, a_sign)) = apply_ops_to_string(a_string, &alpha_ops)
                        else {
                            continue;
                        };
                        if a_out.count_ones() != n_alpha {
                            continue;
                        }
                        let a_out_addr = str2addr(&self.table, a_out);
                        alpha_only_raw.push((a_addr, a_out_addr, coeff * f64::from(a_sign)));
                    }
                }
                // Alpha identity: a beta-only term. Weight = coeff * cross_sign * b_sign.
                (true, false) => {
                    for (b_addr, &b_string) in self.beta_strings.iter().enumerate() {
                        let Some((b_out, b_sign)) = apply_ops_to_string(b_string, &beta_ops) else {
                            continue;
                        };
                        if b_out.count_ones() != n_beta {
                            continue;
                        }
                        let b_out_addr = str2addr(&self.table, b_out);
                        beta_only_raw.push((
                            b_addr,
                            b_out_addr,
                            coeff * f64::from(b_sign * cross_sign),
                        ));
                    }
                }
                // Mixed: both spins active. Keep the two blocks *factored* -- one CSR segment of alpha
                // transitions and one of beta transitions -- instead of materializing the
                // `|alpha_trans| * |beta_trans|` dense cross product. The `cross_sign` (a per-term
                // constant) is folded into the coefficient, so the alpha phase is `a_sign` and the beta
                // phase is `b_sign`; the apply-time contraction recombines them.
                (false, false) => {
                    alpha_trans.clear();
                    for (a_addr, &a_string) in self.alpha_strings.iter().enumerate() {
                        let Some((a_out, a_sign)) = apply_ops_to_string(a_string, &alpha_ops)
                        else {
                            continue;
                        };
                        if a_out.count_ones() != n_alpha {
                            continue;
                        }
                        alpha_trans.push((a_addr, str2addr(&self.table, a_out), a_sign));
                    }
                    // A block with no surviving transitions annihilates the sector: drop the whole term
                    // (emit no alpha or beta segment) rather than a segment that never contributes.
                    if alpha_trans.is_empty() {
                        continue;
                    }
                    let beta_start = mixed_beta.src.len();
                    for (b_addr, &b_string) in self.beta_strings.iter().enumerate() {
                        let Some((b_out, b_sign)) = apply_ops_to_string(b_string, &beta_ops) else {
                            continue;
                        };
                        if b_out.count_ones() != n_beta {
                            continue;
                        }
                        mixed_beta.push(b_addr, str2addr(&self.table, b_out), b_sign);
                    }
                    if mixed_beta.src.len() == beta_start {
                        // Beta annihilates the sector; roll back nothing (we appended none) and skip.
                        continue;
                    }
                    for &(a_src, a_dst, a_phase) in &alpha_trans {
                        mixed_alpha.push(a_src, a_dst, a_phase);
                    }
                    mixed_alpha.finish_term();
                    mixed_beta.finish_term();
                    mixed_coeffs.push(coeff * f64::from(cross_sign));
                }
            }
        }
        Ok(CompiledSector {
            dim: self.dim,
            kind: CompiledKind::Spinful(Box::new(SpinfulCompiled {
                dim_a: self.alpha_strings.len(),
                dim_b,
                scalar,
                // Coalesce transitions sharing a `(src, dst)` key into one fused multiply per matvec.
                alpha_only: coalesce(alpha_only_raw),
                beta_only: coalesce(beta_only_raw),
                mixed_coeffs,
                mixed_alpha,
                mixed_beta,
            })),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference binomial coefficient computed independently of the table.
    fn comb_ref(n: u64, k: u64) -> u64 {
        if k > n {
            return 0;
        }
        let k = k.min(n - k);
        let mut num = 1u64;
        let mut den = 1u64;
        for i in 0..k {
            num *= n - i;
            den *= i + 1;
        }
        num / den
    }

    #[test]
    fn binomial_matches_reference() {
        let table = BinomialTable::new(20);
        for n in 0..=20u32 {
            for k in 0..=20u32 {
                assert_eq!(
                    table.comb(n, k) as u64,
                    comb_ref(n as u64, k as u64),
                    "C({n}, {k}) mismatch"
                );
            }
        }
    }

    /// Enumerates all `u64` bitmasks with exactly `nocc` bits set within `0..norb`, in ascending
    /// bitmask order (the pyscf/cistring determinant ordering).
    fn enumerate_strings(norb: u32, nocc: u32) -> Vec<u64> {
        (0u64..(1u64 << norb))
            .filter(|s| s.count_ones() == nocc)
            .collect()
    }

    #[test]
    fn str2addr_is_identity_over_ascending_enumeration() {
        // Because determinants are ordered by ascending bitmask, str2addr of the i-th such string
        // must equal i.
        let table = BinomialTable::new(10);
        for norb in 0..=10u32 {
            for nocc in 0..=norb {
                for (i, string) in enumerate_strings(norb, nocc).into_iter().enumerate() {
                    assert_eq!(
                        str2addr(&table, string),
                        i,
                        "norb={norb} nocc={nocc} string={string:b}"
                    );
                }
            }
        }
    }

    #[test]
    fn addr2str_inverts_str2addr() {
        let table = BinomialTable::new(10);
        for norb in 0..=10u32 {
            for nocc in 0..=norb {
                let dim = table.num_strings(norb, nocc);
                for addr in 0..dim {
                    let string = addr2str(&table, norb, nocc, addr);
                    assert_eq!(string.count_ones(), nocc, "wrong popcount");
                    assert_eq!(str2addr(&table, string), addr, "round-trip failed");
                }
            }
        }
    }

    #[test]
    fn slater_determinant_spinless_is_one_hot_at_str2addr() {
        // The spinless seed is a one-hot at str2addr(occupation), which (per the ascending-bitmask
        // ordering) equals the index of that occupation in the enumeration.
        let table = BinomialTable::new(8);
        for norb in 0..=8u32 {
            for nocc in 0..=norb {
                let dim = table.num_strings(norb, nocc);
                for (i, string) in enumerate_strings(norb, nocc).into_iter().enumerate() {
                    let vec = slater_determinant_statevector(norb, string, None).unwrap();
                    assert_eq!(vec.len(), dim, "norb={norb} nocc={nocc}");
                    for (addr, amp) in vec.iter().enumerate() {
                        let want = if addr == i {
                            Complex64::new(1.0, 0.0)
                        } else {
                            Complex64::new(0.0, 0.0)
                        };
                        assert_eq!(*amp, want, "norb={norb} nocc={nocc} string={string:b}");
                    }
                }
            }
        }
    }

    #[test]
    fn slater_determinant_spinful_matches_block_spin_flat_index() {
        // The spinful seed is a one-hot at addr_a * dim_b + addr_b (alpha slow, beta fast).
        let table = BinomialTable::new(5);
        for norb in 1..=5u32 {
            for n_alpha in 0..=norb {
                for n_beta in 0..=norb {
                    let dim_b = table.num_strings(norb, n_beta);
                    let dim = table.num_strings(norb, n_alpha) * dim_b;
                    for alpha in enumerate_strings(norb, n_alpha) {
                        for beta in enumerate_strings(norb, n_beta) {
                            let vec =
                                slater_determinant_statevector(norb, alpha, Some(beta)).unwrap();
                            assert_eq!(vec.len(), dim);
                            let addr_a = str2addr(&table, alpha);
                            let addr_b = str2addr(&table, beta);
                            let flat = addr_a * dim_b + addr_b;
                            for (addr, amp) in vec.iter().enumerate() {
                                let want = if addr == flat {
                                    Complex64::new(1.0, 0.0)
                                } else {
                                    Complex64::new(0.0, 0.0)
                                };
                                assert_eq!(
                                    *amp, want,
                                    "norb={norb} a={alpha:b} b={beta:b} flat={flat}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn strings_match_pyscf_golden() {
        // Golden values captured from pyscf.fci.cistring.make_strings(range(norb), nocc):
        // the bitmask at each address, in order.
        let table = BinomialTable::new(6);
        let cases: &[(u32, u32, &[u64])] = &[
            (4, 2, &[3, 5, 6, 9, 10, 12]),
            (5, 2, &[3, 5, 6, 9, 10, 12, 17, 18, 20, 24]),
            (
                6,
                3,
                &[
                    7, 11, 13, 14, 19, 21, 22, 25, 26, 28, 35, 37, 38, 41, 42, 44, 49, 50, 52, 56,
                ],
            ),
        ];
        for &(norb, nocc, golden) in cases {
            assert_eq!(table.num_strings(norb, nocc), golden.len());
            for (addr, &string) in golden.iter().enumerate() {
                assert_eq!(str2addr(&table, string), addr);
                assert_eq!(addr2str(&table, norb, nocc, addr), string);
            }
        }
    }

    /// Independent reference for the ladder sign: count occupied orbitals strictly above `p`.
    fn sign_ref(string: u64, p: u32) -> i8 {
        let mut count = 0;
        for o in (p + 1)..64 {
            if string & (1u64 << o) != 0 {
                count += 1;
            }
        }
        if count % 2 == 0 { 1 } else { -1 }
    }

    #[test]
    fn creation_and_annihilation_signs_and_targets() {
        let table = BinomialTable::new(6);
        let norb = 6u32;
        for nocc in 0..norb {
            for string in enumerate_strings(norb, nocc) {
                for p in 0..norb {
                    let occupied = string & (1u64 << p) != 0;
                    match apply_ladder_op(string, p, true) {
                        LadderResult::Vanishes => {
                            assert!(occupied, "cre should vanish only if occupied")
                        }
                        LadderResult::Maps { string: out, sign } => {
                            assert!(!occupied);
                            assert_eq!(out, string | (1u64 << p));
                            assert_eq!(sign, sign_ref(string, p));
                            // creating raises the sector; address is valid there
                            assert_eq!(out.count_ones(), nocc + 1);
                            let _ = str2addr(&table, out);
                        }
                    }
                    match apply_ladder_op(string, p, false) {
                        LadderResult::Vanishes => {
                            assert!(!occupied, "des should vanish only if empty")
                        }
                        LadderResult::Maps { string: out, sign } => {
                            assert!(occupied);
                            assert_eq!(out, string & !(1u64 << p));
                            assert_eq!(sign, sign_ref(string, p));
                            assert_eq!(out.count_ones(), nocc - 1);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn creation_signs_match_pyscf_golden() {
        // Golden (nocc, src_bitmask, orb, expected_tgt_addr, expected_sign) captured from
        // pyscf.fci.cistring.gen_cre_str_index(range(4), nocc) on norb=4.
        let table = BinomialTable::new(4);
        let golden: &[(u32, u64, u32, usize, i8)] = &[
            (0, 0b0000, 0, 0, 1),
            (0, 0b0000, 1, 1, 1),
            (0, 0b0000, 3, 3, 1),
            (1, 0b0001, 1, 0, 1),
            (1, 0b0001, 3, 3, 1),
            (1, 0b0010, 0, 0, -1),
            (1, 0b0010, 3, 4, 1),
            (2, 0b0110, 0, 0, 1),
            (2, 0b0110, 3, 3, 1),
        ];
        for &(nocc, src, orb, tgt, sign) in golden {
            match apply_ladder_op(src, orb, true) {
                LadderResult::Maps {
                    string: out,
                    sign: s,
                } => {
                    assert_eq!(s, sign, "cre sign nocc={nocc} src={src:b} orb={orb}");
                    assert_eq!(
                        str2addr(&table, out),
                        tgt,
                        "cre target nocc={nocc} src={src:b} orb={orb}"
                    );
                }
                LadderResult::Vanishes => panic!("unexpected vanish for cre"),
            }
        }
    }

    #[test]
    fn annihilation_signs_match_pyscf_golden() {
        // Golden (nocc, src_bitmask, orb, expected_tgt_addr, expected_sign) captured from
        // pyscf.fci.cistring.gen_des_str_index(range(4), nocc) on norb=4 (orbital = column 1).
        let table = BinomialTable::new(4);
        let golden: &[(u32, u64, u32, usize, i8)] = &[
            (2, 0b0011, 0, 1, -1),
            (2, 0b0011, 1, 0, 1),
            (2, 0b0110, 1, 2, -1),
            (2, 0b0110, 2, 1, 1),
            (2, 0b1100, 2, 3, -1),
            (2, 0b1100, 3, 2, 1),
        ];
        for &(nocc, src, orb, tgt, sign) in golden {
            match apply_ladder_op(src, orb, false) {
                LadderResult::Maps {
                    string: out,
                    sign: s,
                } => {
                    assert_eq!(s, sign, "des sign nocc={nocc} src={src:b} orb={orb}");
                    assert_eq!(
                        str2addr(&table, out),
                        tgt,
                        "des target nocc={nocc} src={src:b} orb={orb}"
                    );
                }
                LadderResult::Vanishes => panic!("unexpected vanish for des"),
            }
        }
    }

    // ---------------------------------------------------------------------------------------------
    // Matvec tests
    // ---------------------------------------------------------------------------------------------

    /// Independent, deliberately naive reference matvec via a single block-spin Jordan-Wigner walk.
    ///
    /// This is the exact algorithm empirically verified against `ffsim.linear_operator` (block-spin
    /// ordering, ladder ops applied right-to-left, per-op sign `(-1)^{occupied modes below m}`). It
    /// treats the whole `2 * norb`-mode (or `norb`-mode spinless) register as one occupation bitmask,
    /// making no alpha/beta split -- so it cross-checks the split-and-recombine production kernel.
    fn reference_matvec(
        norb: u32,
        n_alpha: u32,
        n_beta: Option<u32>,
        terms: &[(Complex64, Vec<bool>, Vec<u32>)],
        vec: &[Complex64],
    ) -> Vec<Complex64> {
        let table = BinomialTable::new(norb);
        let spinless = n_beta.is_none();
        let n_beta = n_beta.unwrap_or(0);
        let dim_a = table.num_strings(norb, n_alpha);
        let dim_b = if spinless {
            1
        } else {
            table.num_strings(norb, n_beta)
        };
        // `dim_b == 1` in the spinless case, so this covers both.
        let dim = dim_a * dim_b;

        // Map a flat address to a combined occupation bitmask over 2*norb (or norb) modes.
        let to_mask = |addr: usize| -> u64 {
            if spinless {
                addr2str(&table, norb, n_alpha, addr)
            } else {
                let a = addr / dim_b;
                let b = addr % dim_b;
                let a_str = addr2str(&table, norb, n_alpha, a);
                let b_str = addr2str(&table, norb, n_beta, b);
                a_str | (b_str << norb)
            }
        };
        // Map a combined occupation bitmask back to a flat address.
        let from_mask = |mask: u64| -> usize {
            if spinless {
                str2addr(&table, mask)
            } else {
                let low = if norb >= 64 {
                    !0u64
                } else {
                    (1u64 << norb) - 1
                };
                let a_str = mask & low;
                let b_str = mask >> norb;
                let a = str2addr(&table, a_str);
                let b = str2addr(&table, b_str);
                a * dim_b + b
            }
        };

        let mut out = vec![Complex64::new(0.0, 0.0); dim];
        for (coeff, actions, modes) in terms {
            for (src_addr, &amp) in vec.iter().enumerate() {
                if amp == Complex64::new(0.0, 0.0) {
                    continue;
                }
                let mut mask = to_mask(src_addr);
                let mut sign: i8 = 1;
                let mut ok = true;
                // ladder ops act right-to-left; `actions`/`modes` are parallel slices (a term's
                // native split layout), so zip and reverse them together.
                for (&is_creation, &mode) in actions.iter().zip(modes).rev() {
                    let bit = 1u64 << mode;
                    let occupied = mask & bit != 0;
                    if is_creation == occupied {
                        // creating in an occupied mode or destroying an empty one
                        ok = false;
                        break;
                    }
                    // Sign from occupied modes strictly below `mode` in the single JW string.
                    let below_mask = if mode == 0 { 0 } else { (1u64 << mode) - 1 };
                    let below = (mask & below_mask).count_ones();
                    if below % 2 == 1 {
                        sign = -sign;
                    }
                    mask ^= bit;
                }
                if ok {
                    out[from_mask(mask)] += *coeff * f64::from(sign) * amp;
                }
            }
        }
        out
    }

    fn complex_vec(reals: &[f64]) -> Vec<Complex64> {
        reals.iter().map(|&r| Complex64::new(r, 0.0)).collect()
    }

    /// Borrows owned `(coeff, actions, modes)` terms (a term's native split layout, mirroring
    /// `FermionOperatorTermView`) as the `(coeff, &[bool], &[u32])` iterator the `compile` entry point
    /// takes. The owned terms must outlive the borrowed views.
    fn term_views(
        terms: &[(Complex64, Vec<bool>, Vec<u32>)],
    ) -> impl Iterator<Item = (Complex64, &[bool], &[u32])> {
        terms
            .iter()
            .map(|(c, a, m)| (*c, a.as_slice(), m.as_slice()))
    }

    fn assert_vec_close(a: &[Complex64], b: &[Complex64]) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!(
                (x - y).norm() < 1e-12,
                "mismatch at {i}: {x} vs {y}\n  got:      {a:?}\n  expected: {b:?}"
            );
        }
    }

    #[test]
    fn spinless_matvec_matches_reference() {
        // Cover a range of orbital counts / fillings and several kinds of terms.
        let cases: &[(u32, u32)] = &[(3, 1), (4, 2), (5, 2), (5, 3), (6, 3)];
        // Terms are in the native split `(coeff, actions, modes)` layout (parallel actions/modes
        // slices), the same shape `compile` consumes; `actions[k]` is `true` for a creation.
        let terms: Vec<(Complex64, Vec<bool>, Vec<u32>)> = vec![
            // hopping a†_0 a_1
            (Complex64::new(0.7, 0.2), vec![true, false], vec![0, 1]),
            // its conjugate a†_1 a_0
            (Complex64::new(0.7, -0.2), vec![true, false], vec![1, 0]),
            // number operator n_2 = a†_2 a_2
            (Complex64::new(0.5, 0.0), vec![true, false], vec![2, 2]),
            // density-density n_0 n_2 (written interleaved to exercise sign)
            (
                Complex64::new(1.3, 0.0),
                vec![true, false, true, false],
                vec![0, 0, 2, 2],
            ),
        ];
        for &(norb, nocc) in cases {
            let dim = BinomialTable::new(norb).num_strings(norb, nocc);
            // A deterministic non-trivial input vector.
            let vec: Vec<Complex64> = (0..dim)
                .map(|i| Complex64::new((i as f64 + 1.0) * 0.31, (i as f64) * -0.17))
                .collect();
            let expected = reference_matvec(norb, nocc, None, &terms, &vec);
            let got = SpinlessSector::new(norb, nocc)
                .compile(term_views(&terms))
                .unwrap()
                .apply(&vec)
                .unwrap();
            assert_vec_close(&got, &expected);
        }
    }

    #[test]
    fn spinful_matvec_matches_reference() {
        // Modes: alpha in [0, norb), beta in [norb, 2*norb).
        let norb = 3u32;
        let cases: &[(u32, u32)] = &[(1, 1), (2, 1), (1, 2), (2, 2)];
        // A mixture of same-spin and cross-spin terms, some written interleaved so the block-spin
        // reordering sign is genuinely exercised.
        // Split `(coeff, actions, modes)` layout; beta modes are `norb + orbital`.
        let make_terms = |norb: u32| -> Vec<(Complex64, Vec<bool>, Vec<u32>)> {
            vec![
                // alpha hopping a†_0 a_1
                (Complex64::new(0.9, 0.0), vec![true, false], vec![0, 1]),
                // beta hopping a†_0 a_1  (beta modes are norb + orbital)
                (
                    Complex64::new(0.4, 0.1),
                    vec![true, false],
                    vec![norb, norb + 1],
                ),
                // cross-spin density-density n^a_0 n^b_0 = a†_0 a_0 a†_{norb} a_{norb}
                (
                    Complex64::new(1.1, 0.0),
                    vec![true, false, true, false],
                    vec![0, 0, norb, norb],
                ),
                // interleaved cross-spin: a†_0(a) a†_{norb+2}(b) a_{norb}(b) a_2(a)  -- spin exchange
                // pattern that moves an alpha 2->0 and a beta 0->2, ops interleaved a,b,b,a.
                (
                    Complex64::new(0.6, -0.3),
                    vec![true, true, false, false],
                    vec![0, norb + 2, norb, 2],
                ),
            ]
        };
        let terms = make_terms(norb);
        for &(n_a, n_b) in cases {
            let table = BinomialTable::new(norb);
            let dim = table.num_strings(norb, n_a) * table.num_strings(norb, n_b);
            let vec: Vec<Complex64> = (0..dim)
                .map(|i| Complex64::new((i as f64 + 0.5) * 0.23, (i as f64) * 0.11))
                .collect();
            let expected = reference_matvec(norb, n_a, Some(n_b), &terms, &vec);
            let got = SpinfulSector::new(norb, n_a, n_b)
                .unwrap()
                .compile(term_views(&terms))
                .unwrap()
                .apply(&vec)
                .unwrap();
            assert_vec_close(&got, &expected);
        }
    }

    #[test]
    fn spinful_matvec_cross_sign_depends_on_alpha_parity() {
        // A single beta number operator on orbital 0, applied with 1 vs 2 alpha electrons. Because a
        // number operator has an even count of beta ladder ops (k_beta = 2), the cross factor is +1
        // in both cases -- this is a regression guard that the kernel does NOT erroneously apply an
        // odd power. We compare against the reference for both parities.
        let norb = 3u32;
        for &(n_a, n_b) in &[(1u32, 1u32), (2, 1)] {
            let table = BinomialTable::new(norb);
            let dim = table.num_strings(norb, n_a) * table.num_strings(norb, n_b);
            let terms: Vec<(Complex64, Vec<bool>, Vec<u32>)> = vec![(
                Complex64::new(1.0, 0.0),
                vec![true, false],
                vec![norb, norb],
            )];
            let vec: Vec<Complex64> = (0..dim)
                .map(|i| Complex64::new(i as f64 + 1.0, 0.0))
                .collect();
            let expected = reference_matvec(norb, n_a, Some(n_b), &terms, &vec);
            let got = SpinfulSector::new(norb, n_a, n_b)
                .unwrap()
                .compile(term_views(&terms))
                .unwrap()
                .apply(&vec)
                .unwrap();
            assert_vec_close(&got, &expected);
        }
    }

    #[test]
    fn matvec_dimension_and_mode_errors() {
        // Wrong vector length is caught by `apply` (dimension is a property of the vector, not the
        // compiled map).
        let bad = complex_vec(&[1.0, 2.0]);
        let err = SpinlessSector::new(4, 2)
            .compile(std::iter::once((
                Complex64::new(1.0, 0.0),
                [true, false].as_slice(),
                [0u32, 0].as_slice(),
            )))
            .unwrap()
            .apply(&bad)
            .unwrap_err();
        assert!(matches!(err, FciMatvecError::DimensionMismatch { .. }));

        // Mode out of range (spinless: mode must be < norb) is caught at compile time.
        let err = SpinlessSector::new(4, 2)
            .compile(std::iter::once((
                Complex64::new(1.0, 0.0),
                [true, false].as_slice(),
                [9u32, 0].as_slice(),
            )))
            .unwrap_err();
        assert!(matches!(
            err,
            FciMatvecError::ModeOutOfRange {
                mode: 9,
                num_modes: 4
            }
        ));

        // Spinful: mode must be < 2*norb.
        let err = SpinfulSector::new(3, 1, 1)
            .unwrap()
            .compile(std::iter::once((
                Complex64::new(1.0, 0.0),
                [true].as_slice(),
                [6u32].as_slice(),
            )))
            .unwrap_err();
        assert!(matches!(
            err,
            FciMatvecError::ModeOutOfRange {
                mode: 6,
                num_modes: 6
            }
        ));
    }

    #[test]
    fn spinful_dim_reports_overflow() {
        let table = BinomialTable::new(64);
        // Each half-filled sector is ~1.8e18 (< usize::MAX), but their product ~3.3e36 overflows.
        let err = spinful_dim(&table, 64, 32, 32).unwrap_err();
        assert!(matches!(err, FciMatvecError::DimensionOverflow { .. }));

        // A representable product still succeeds and equals the plain product.
        let dim = spinful_dim(&table, 4, 2, 2).unwrap();
        assert_eq!(dim, table.num_strings(4, 2) * table.num_strings(4, 2));
    }

    #[test]
    fn spinless_matvec_drops_non_conserving_terms() {
        // norb=4, nocc=1: a bare creation a†_2 maps |0b0001> -> |0b0101> (popcount 2), which
        // `str2addr` would rank at C(0,1)+C(2,2) = 1, but the two-electron string is outside the
        // one-electron sector. Before the guard this indexed `out` out of bounds (str2addr of
        // 0b1100 in a dim-4 vector is 5). It must now be dropped, leaving the zero vector.
        let dim = BinomialTable::new(4).num_strings(4, 1); // C(4,1) = 4
        let vec = complex_vec(&vec![1.0; dim]);
        let got = SpinlessSector::new(4, 1)
            .compile(std::iter::once((
                Complex64::new(1.0, 0.0),
                [true].as_slice(),
                [2u32].as_slice(),
            )))
            .unwrap()
            .apply(&vec)
            .unwrap();
        assert_vec_close(&got, &complex_vec(&vec![0.0; dim]));

        // The number operator a†_2 a_2 (conserving) still acts: it keeps determinants occupying
        // orbital 2 and zeros the rest, so the guard does not over-reject.
        let got = SpinlessSector::new(4, 1)
            .compile(std::iter::once((
                Complex64::new(1.0, 0.0),
                [true, false].as_slice(),
                [2u32, 2].as_slice(),
            )))
            .unwrap()
            .apply(&vec)
            .unwrap();
        // Only the determinant |0b0100> (orbital 2 occupied) survives; its address is C(2,1) = 2.
        let mut expected = vec![Complex64::new(0.0, 0.0); dim];
        expected[str2addr(&BinomialTable::new(4), 0b0100)] = Complex64::new(1.0, 0.0);
        assert_vec_close(&got, &expected);
    }

    #[test]
    fn spinful_matvec_drops_sector_changing_terms() {
        // norb=2, (n_alpha, n_beta) = (1, 1). The spin-flip a†_{0a} a_{0b} (modes 0 and norb+0=2)
        // conserves total particle number but moves an electron from beta to alpha, leaving the
        // (1, 1) sector. Its alpha sublist raises to two alpha electrons (popcount 2 != n_alpha),
        // so it must be dropped rather than ranked out of bounds.
        let t = BinomialTable::new(2);
        let dim = t.num_strings(2, 1) * t.num_strings(2, 1); // 2 * 2 = 4
        let vec = complex_vec(&vec![1.0; dim]);
        let got = SpinfulSector::new(2, 1, 1)
            .unwrap()
            .compile(std::iter::once((
                Complex64::new(1.0, 0.0),
                [true, false].as_slice(),
                [0u32, 2].as_slice(),
            )))
            .unwrap()
            .apply(&vec)
            .unwrap();
        assert_vec_close(&got, &complex_vec(&vec![0.0; dim]));
    }

    /// The adjoint of a split `(coeff, actions, modes)` term: reverse the ladder-operator order, swap
    /// creation/annihilation, and conjugate the coefficient. `(c * o_1 o_2 ... o_k)^H = conj(c) *
    /// o_k^H ... o_1^H`, and a single ladder operator's adjoint just flips creation<->annihilation on
    /// the same orbital. Used to build an independent oracle for `apply_conj` (op^H @ vec): compiling
    /// the adjoint terms and applying them forward must reproduce `apply_conj` of the original.
    fn adjoint_term(term: &(Complex64, Vec<bool>, Vec<u32>)) -> (Complex64, Vec<bool>, Vec<u32>) {
        let (coeff, actions, modes) = term;
        let adj_actions = actions
            .iter()
            .rev()
            .map(|&is_creation| !is_creation)
            .collect();
        let adj_modes = modes.iter().rev().copied().collect();
        (coeff.conj(), adj_actions, adj_modes)
    }

    #[test]
    fn compiled_apply_conj_matches_adjoint_oracle() {
        // `apply_conj` (op^H @ vec) must reproduce the *forward* apply of the compiled adjoint
        // operator across several vectors -- the reversed-scatter/conjugated-weight rmatvec must equal
        // building and applying op^H directly. This is an independent oracle for `apply_conj`; the
        // forward `apply` is validated against the naive `reference_matvec` in the `*_matches_reference`
        // tests. Sector-changing terms are included: they compile to *dropped* entries (no
        // contribution) on both the original and its adjoint, matching the kernel's in-sector
        // projection.

        // --- Spinless ---
        let (norb, nocc) = (5u32, 3u32);
        let sector = SpinlessSector::new(norb, nocc);
        let dim = sector.dim();
        let terms: Vec<(Complex64, Vec<bool>, Vec<u32>)> = vec![
            (Complex64::new(0.7, 0.2), vec![true, false], vec![0, 1]),
            (Complex64::new(0.5, 0.0), vec![true, false], vec![2, 2]),
            // sector-changing (raises particle number): must be dropped on compile.
            (Complex64::new(1.3, -0.4), vec![true], vec![4]),
        ];
        let adj_terms: Vec<(Complex64, Vec<bool>, Vec<u32>)> =
            terms.iter().map(adjoint_term).collect();
        let compiled = sector.compile(term_views(&terms)).unwrap();
        let compiled_adj = sector.compile(term_views(&adj_terms)).unwrap();
        assert_eq!(compiled.dim(), dim);
        for scale in [0.31f64, -1.7] {
            let vec: Vec<Complex64> = (0..dim)
                .map(|i| Complex64::new((i as f64 + 1.0) * scale, (i as f64) * 0.13))
                .collect();
            // apply_conj == forward apply of the compiled adjoint operator
            let via_conj = compiled.apply_conj(&vec).unwrap();
            let via_adj = compiled_adj.apply(&vec).unwrap();
            assert_vec_close(&via_conj, &via_adj);
        }

        // --- Spinful ---
        let (norb, n_a, n_b) = (3u32, 2u32, 1u32);
        let sector = SpinfulSector::new(norb, n_a, n_b).unwrap();
        let dim = sector.dim();
        let terms: Vec<(Complex64, Vec<bool>, Vec<u32>)> = vec![
            (Complex64::new(0.9, 0.0), vec![true, false], vec![0, 1]),
            (
                Complex64::new(1.1, 0.3),
                vec![true, false, true, false],
                vec![0, 0, norb, norb],
            ),
            // sector-changing spin flip a†_{0a} a_{0b}: conserves total N but leaves the (2,1) sector;
            // must be dropped on compile.
            (Complex64::new(0.5, 0.0), vec![true, false], vec![0, norb]),
        ];
        let adj_terms: Vec<(Complex64, Vec<bool>, Vec<u32>)> =
            terms.iter().map(adjoint_term).collect();
        let compiled = sector.compile(term_views(&terms)).unwrap();
        let compiled_adj = sector.compile(term_views(&adj_terms)).unwrap();
        assert_eq!(compiled.dim(), dim);
        for scale in [0.23f64, 0.91] {
            let vec: Vec<Complex64> = (0..dim)
                .map(|i| Complex64::new((i as f64 + 0.5) * scale, (i as f64) * 0.07))
                .collect();
            let via_conj = compiled.apply_conj(&vec).unwrap();
            let via_adj = compiled_adj.apply(&vec).unwrap();
            assert_vec_close(&via_conj, &via_adj);
        }
    }

    #[test]
    fn compiled_apply_reports_dimension_mismatch() {
        // A wrong-length vector must be rejected by both `apply` and `apply_conj`, mirroring `matvec`.
        let sector = SpinlessSector::new(4, 2);
        let compiled = sector
            .compile(std::iter::once((
                Complex64::new(1.0, 0.0),
                [true, false].as_slice(),
                [0u32, 1].as_slice(),
            )))
            .unwrap();
        let bad = complex_vec(&[1.0, 2.0]);
        assert!(matches!(
            compiled.apply(&bad).unwrap_err(),
            FciMatvecError::DimensionMismatch { .. }
        ));
        assert!(matches!(
            compiled.apply_conj(&bad).unwrap_err(),
            FciMatvecError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn compiled_rejects_out_of_range_mode() {
        // Mode validation happens at compile time, mirroring the lazy kernel's per-call check.
        let sector = SpinlessSector::new(4, 2);
        let err = sector
            .compile(std::iter::once((
                Complex64::new(1.0, 0.0),
                [true, false].as_slice(),
                [9u32, 0].as_slice(),
            )))
            .unwrap_err();
        assert!(matches!(
            err,
            FciMatvecError::ModeOutOfRange {
                mode: 9,
                num_modes: 4
            }
        ));
    }

    #[test]
    fn slater_determinant_rejects_out_of_range_bit() {
        // A bit set at orbital 2 with norb=2 addresses a nonexistent orbital; it must error rather
        // than rank past the sector dimension and panic on the vector write.
        let err = slater_determinant_statevector(2, 0b100, None).unwrap_err();
        assert!(matches!(
            err,
            FciMatvecError::ModeOutOfRange {
                mode: 2,
                num_modes: 2
            }
        ));

        // The beta mask is validated too.
        let err = slater_determinant_statevector(2, 0b01, Some(0b100)).unwrap_err();
        assert!(matches!(
            err,
            FciMatvecError::ModeOutOfRange {
                mode: 2,
                num_modes: 2
            }
        ));

        // A valid occupation still succeeds.
        let vec = slater_determinant_statevector(2, 0b01, Some(0b10)).unwrap();
        assert_eq!(vec.iter().filter(|c| c.norm() > 0.0).count(), 1);
    }

    #[test]
    fn coalescing_preserves_result() {
        // Terms that scatter into the same `(src, dst)` slot are fused by `coalesce`. The fused map
        // must reproduce the naive reference bit-for-bit (linearity), and the fused entry count must
        // be smaller than the raw count -- otherwise coalescing bought nothing.

        // --- Spinless: three number operators on orbital 0 collapse to one diagonal entry per
        // occupied determinant (n_0 written three times), plus a hop written twice. ---
        let (norb, nocc) = (5u32, 3u32);
        let sector = SpinlessSector::new(norb, nocc);
        let dim = sector.dim();
        let terms: Vec<(Complex64, Vec<bool>, Vec<u32>)> = vec![
            (Complex64::new(1.0, 0.0), vec![true, false], vec![0, 0]),
            (Complex64::new(0.5, 0.0), vec![true, false], vec![0, 0]),
            (Complex64::new(-0.25, 0.3), vec![true, false], vec![0, 0]),
            (Complex64::new(0.7, 0.2), vec![true, false], vec![1, 2]),
            (Complex64::new(0.1, -0.4), vec![true, false], vec![1, 2]),
        ];
        let compiled = sector.compile(term_views(&terms)).unwrap();
        let vec: Vec<Complex64> = (0..dim)
            .map(|i| Complex64::new(i as f64 + 0.5, (i as f64) * 0.11))
            .collect();
        assert_vec_close(
            &compiled.apply(&vec).unwrap(),
            &reference_matvec(norb, nocc, None, &terms, &vec),
        );
        // Coalesced entries (<= 2 distinct (src,dst) families) vs the raw scatter count.
        let CompiledKind::Spinless { entries } = &compiled.kind else {
            panic!("expected a spinless compiled kind");
        };
        let raw_spinless = terms
            .iter()
            .map(|(_, actions, modes)| {
                let ops: Vec<(bool, u32)> =
                    actions.iter().copied().zip(modes.iter().copied()).collect();
                sector
                    .strings
                    .iter()
                    .filter(|&&s| {
                        apply_ops_to_string(s, &ops)
                            .is_some_and(|(out, _)| out.count_ones() == nocc)
                    })
                    .count()
            })
            .sum::<usize>();
        assert!(
            entries.len() < raw_spinless,
            "coalescing did not shrink the spinless scatter: {} !< {raw_spinless}",
            entries.len()
        );

        // --- Spinful: the same fusion applies to the alpha-only bucket (a repeated alpha number
        // operator) and the beta-only bucket (a repeated beta hop). ---
        let (norb, n_a, n_b) = (3u32, 2u32, 1u32);
        let sector = SpinfulSector::new(norb, n_a, n_b).unwrap();
        let dim = sector.dim();
        let terms: Vec<(Complex64, Vec<bool>, Vec<u32>)> = vec![
            (Complex64::new(1.0, 0.0), vec![true, false], vec![0, 0]),
            (Complex64::new(0.5, 0.2), vec![true, false], vec![0, 0]),
            (
                Complex64::new(0.9, 0.0),
                vec![true, false],
                vec![norb, norb + 1],
            ),
            (
                Complex64::new(0.3, -0.1),
                vec![true, false],
                vec![norb, norb + 1],
            ),
        ];
        let compiled = sector.compile(term_views(&terms)).unwrap();
        let vec: Vec<Complex64> = (0..dim)
            .map(|i| Complex64::new(i as f64 + 0.3, (i as f64) * 0.07))
            .collect();
        assert_vec_close(
            &compiled.apply(&vec).unwrap(),
            &reference_matvec(norb, n_a, Some(n_b), &terms, &vec),
        );
        let CompiledKind::Spinful(spinful) = &compiled.kind else {
            panic!("expected a spinful compiled kind");
        };
        let (alpha_only, beta_only) = (&spinful.alpha_only, &spinful.beta_only);
        // Two alpha-number terms fused into one alpha_only family per surviving determinant, and two
        // beta-hop terms fused into one beta_only family -- each strictly fewer than the 2x raw pushes.
        let alpha_survivors = sector.alpha_strings.iter().filter(|&&s| s & 1 != 0).count();
        assert_eq!(alpha_only.scale.len(), alpha_survivors);
        assert!(beta_only.scale.len() < 2 * sector.beta_strings.len());
    }

    #[test]
    fn categories_match_reference_without_normal_ordering() {
        // A spinful operator mixing all four term categories -- identity/scalar, alpha-only,
        // beta-only, and cross-spin mixed -- with a term written in a NON-normal order (annihilation
        // before creation on distinct alpha modes). Categorization + the factored fast paths must
        // reproduce the naive JW reference across a symmetric and an asymmetric sector, proving the
        // kernel needs no normal-ordering pre-pass.
        let norb = 3u32;
        let terms: Vec<(Complex64, Vec<bool>, Vec<u32>)> = vec![
            // scalar / identity on both spins.
            (Complex64::new(1.3, -0.2), vec![], vec![]),
            // alpha-only, deliberately NON-normal-ordered: a_1 a†_0 (annihilation first).
            (Complex64::new(0.7, 0.4), vec![false, true], vec![1, 0]),
            // beta-only hop a†_{0b} a_{1b}.
            (
                Complex64::new(0.5, 0.0),
                vec![true, false],
                vec![norb, norb + 1],
            ),
            // cross-spin density-density n^a_2 n^b_2.
            (
                Complex64::new(1.1, 0.0),
                vec![true, false, true, false],
                vec![2, 2, norb + 2, norb + 2],
            ),
            // cross-spin, interleaved and non-normal-ordered: a_{1b} a†_{0a} a†_{1a}(no) -- use a real
            // spin exchange a†_{0a} a†_{2b} a_{0b} a_{2a} written interleaved.
            (
                Complex64::new(0.6, -0.3),
                vec![true, true, false, false],
                vec![0, norb + 2, norb, 2],
            ),
        ];
        for &(n_a, n_b) in &[(2u32, 2u32), (2, 1)] {
            let sector = SpinfulSector::new(norb, n_a, n_b).unwrap();
            let dim = sector.dim();
            let vec: Vec<Complex64> = (0..dim)
                .map(|i| Complex64::new((i as f64 + 0.5) * 0.23, (i as f64) * 0.11))
                .collect();
            let compiled = sector.compile(term_views(&terms)).unwrap();
            assert_vec_close(
                &compiled.apply(&vec).unwrap(),
                &reference_matvec(norb, n_a, Some(n_b), &terms, &vec),
            );
        }
    }

    #[test]
    fn mixed_factored_contraction_exercises_both_shorter_block_branches() {
        // The factored mixed contraction iterates whichever spin block has fewer transitions in the
        // outer loop. On an asymmetric sector (n_alpha != n_beta), a term whose alpha block is a hop
        // (few survivors) and beta block is a number operator (many survivors) drives one branch; a
        // term with the roles reversed drives the other. Both must reproduce the naive JW reference for
        // `apply` and `apply_conj`, and we assert the two branches are genuinely both taken.
        let norb = 4u32;
        let (n_a, n_b) = (3u32, 1u32); // asymmetric: a number operator has many survivors in the
        // 3-electron block but a hop has few in the 1-electron block, so the shorter block flips with
        // which spin carries the number operator.
        let sector = SpinfulSector::new(norb, n_a, n_b).unwrap();
        let dim = sector.dim();
        let terms: Vec<(Complex64, Vec<bool>, Vec<u32>)> = vec![
            // alpha number operator n^a_0 (n_a = 3: many alpha survivors) with a beta hop 1->2 (n_b = 1:
            // 1 beta survivor). Expect alpha_len > beta_len -> beta-outer branch.
            (
                Complex64::new(0.8, 0.2),
                vec![true, false, true, false],
                vec![0, 0, norb + 2, norb + 1],
            ),
            // alpha hop 1->2 (n_a = 3: few alpha survivors) with a beta number operator n^b_0 (n_b = 1:
            // 1 beta survivor). Expect alpha_len <= beta_len -> alpha-outer branch.
            (
                Complex64::new(0.5, -0.4),
                vec![true, false, true, false],
                vec![2, 1, norb, norb],
            ),
        ];
        let compiled = sector.compile(term_views(&terms)).unwrap();

        // White-box: confirm both the alpha-outer (a_len <= b_len) and beta-outer (a_len > b_len)
        // branches are represented across the compiled mixed terms.
        let CompiledKind::Spinful(spinful) = &compiled.kind else {
            panic!("expected a spinful compiled kind");
        };
        let (mixed_alpha, mixed_beta, mixed_coeffs) = (
            &spinful.mixed_alpha,
            &spinful.mixed_beta,
            &spinful.mixed_coeffs,
        );
        assert_eq!(mixed_coeffs.len(), 2, "both mixed terms should compile");
        let mut saw_alpha_outer = false;
        let mut saw_beta_outer = false;
        for t in 0..mixed_coeffs.len() {
            let a_len = mixed_alpha.indptr[t + 1] - mixed_alpha.indptr[t];
            let b_len = mixed_beta.indptr[t + 1] - mixed_beta.indptr[t];
            if a_len <= b_len {
                saw_alpha_outer = true;
            } else {
                saw_beta_outer = true;
            }
        }
        assert!(
            saw_alpha_outer && saw_beta_outer,
            "expected both shorter-block branches to be exercised (alpha-outer={saw_alpha_outer}, \
             beta-outer={saw_beta_outer})"
        );

        let vec: Vec<Complex64> = (0..dim)
            .map(|i| Complex64::new((i as f64 + 0.5) * 0.19, (i as f64) * 0.13))
            .collect();
        assert_vec_close(
            &compiled.apply(&vec).unwrap(),
            &reference_matvec(norb, n_a, Some(n_b), &terms, &vec),
        );
        // apply_conj against the adjoint operator's forward reference.
        let adj_terms: Vec<(Complex64, Vec<bool>, Vec<u32>)> =
            terms.iter().map(adjoint_term).collect();
        assert_vec_close(
            &compiled.apply_conj(&vec).unwrap(),
            &reference_matvec(norb, n_a, Some(n_b), &adj_terms, &vec),
        );
    }
}
