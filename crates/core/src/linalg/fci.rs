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
}

/// Computes the combinatorial-number-system address of an occupation `string`.
///
/// `string` must have exactly `nocc` bits set, all within ``0..norb``. The address is
/// ``sum_j C(p_j, j + 1)`` over the occupied orbitals `p_j` in ascending order, matching
/// `pyscf.fci.cistring.str2addr`.
pub fn str2addr(table: &BinomialTable, _norb: u32, _nocc: u32, string: u64) -> usize {
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
        // Find the largest orbital `orb` with C(orb, which) <= remaining.
        let mut orb = norb;
        loop {
            orb -= 1;
            let weight = table.comb(orb, which);
            if weight <= remaining {
                string |= 1u64 << orb;
                remaining -= weight;
                break;
            }
        }
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

/// Applies a creation operator on orbital `p` to `string`.
///
/// Returns [`LadderResult::Vanishes`] if orbital `p` is already occupied.
#[inline]
pub fn apply_creation(string: u64, p: u32) -> LadderResult {
    let bit = 1u64 << p;
    if string & bit != 0 {
        return LadderResult::Vanishes;
    }
    LadderResult::Maps {
        string: string | bit,
        sign: ladder_sign(string, p),
    }
}

/// Applies an annihilation operator on orbital `p` to `string`.
///
/// Returns [`LadderResult::Vanishes`] if orbital `p` is empty.
#[inline]
pub fn apply_annihilation(string: u64, p: u32) -> LadderResult {
    let bit = 1u64 << p;
    if string & bit == 0 {
        return LadderResult::Vanishes;
    }
    LadderResult::Maps {
        string: string & !bit,
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
// `gen_cre_str_index`/`gen_des_str_index` (see [`apply_creation`]/[`apply_annihilation`]), so
// composing the per-op maps of a term right-to-left reproduces ffsim's per-sector sign. The only
// cross-sector contribution is that, under block-spin ordering, every occupied alpha orbital sits
// below every beta orbital: hence each *beta* ladder operator additionally carries `(-1)^{n_alpha}`
// (the current alpha electron count, which is invariant across a particle-conserving term).

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

/// The occupation strings of a fixed-particle-number sector, addressed by their linear index.
///
/// Element `addr` is the occupation string with [`str2addr`] equal to `addr`. Precomputing this list
/// once per sector avoids repeated unranking in the inner matvec loop.
fn sector_strings(table: &BinomialTable, norb: u32, nocc: u32) -> Vec<u64> {
    let dim = table.num_strings(norb, nocc);
    (0..dim)
        .map(|addr| addr2str(table, norb, nocc, addr))
        .collect()
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
        let result = if is_creation {
            apply_creation(string, orb)
        } else {
            apply_annihilation(string, orb)
        };
        match result {
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

/// Applies an operator's terms to a spinless FCI state vector: `out = op @ vec`.
///
/// * `norb` is the number of orbitals; the operator's modes must lie in `[0, norb)`.
/// * `nocc` is the (integer) electron count; the vector length must equal `C(norb, nocc)`.
///
/// Each term is supplied as `(coeff, actions, modes)` -- the native slice layout of a
/// [`crate::operators::fermion_operator::FermionOperatorTermView`] -- where `actions[k]` is `true`
/// for a creation and `modes[k]` is its orbital, in the term's written (left-to-right) order.
/// Returns the transformed vector, or an error on a dimension/mode mismatch.
pub fn spinless_matvec<'a>(
    norb: u32,
    nocc: u32,
    terms: impl IntoIterator<Item = (Complex64, &'a [bool], &'a [u32])>,
    vec: &[Complex64],
) -> Result<Vec<Complex64>, FciMatvecError> {
    let table = BinomialTable::new(norb);
    let dim = table.num_strings(norb, nocc);
    if vec.len() != dim {
        return Err(FciMatvecError::DimensionMismatch {
            expected: dim,
            actual: vec.len(),
        });
    }
    let strings = sector_strings(&table, norb, nocc);
    let mut out = vec![Complex64::new(0.0, 0.0); dim];
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
        // Iterate over every source determinant, apply the term, and scatter to the destination.
        for (src_addr, &string) in strings.iter().enumerate() {
            let amp = vec[src_addr];
            if amp == Complex64::new(0.0, 0.0) {
                continue;
            }
            if let Some((out_string, sign)) = apply_ops_to_string(string, &ops) {
                let dst_addr = str2addr(&table, norb, nocc, out_string);
                out[dst_addr] += coeff * f64::from(sign) * amp;
            }
        }
    }
    Ok(out)
}

/// Applies an operator's terms to a spinful FCI state vector: `out = op @ vec`.
///
/// * `norb` is the number of spatial orbitals; the operator's modes must lie in `[0, 2 * norb)`
///   under the block-spin convention (mode `m < norb` is alpha orbital `m`; mode `m >= norb` is beta
///   orbital `m - norb`).
/// * `n_alpha`, `n_beta` are the per-spin electron counts; the vector length must equal
///   `C(norb, n_alpha) * C(norb, n_beta)` with flat index `addr_a * dim_b + addr_b`.
///
/// Each term is supplied as `(coeff, actions, modes)` (see [`spinless_matvec`] for the layout).
/// Each beta operator additionally contributes `(-1)^{n_alpha}` because, under block-spin ordering,
/// all `n_alpha` occupied alpha orbitals precede every beta orbital in the Jordan-Wigner string.
pub fn spinful_matvec<'a>(
    norb: u32,
    n_alpha: u32,
    n_beta: u32,
    terms: impl IntoIterator<Item = (Complex64, &'a [bool], &'a [u32])>,
    vec: &[Complex64],
) -> Result<Vec<Complex64>, FciMatvecError> {
    let table = BinomialTable::new(norb);
    let dim_b = table.num_strings(norb, n_beta);
    let dim = spinful_dim(&table, norb, n_alpha, n_beta)?;
    if vec.len() != dim {
        return Err(FciMatvecError::DimensionMismatch {
            expected: dim,
            actual: vec.len(),
        });
    }
    let alpha_strings = sector_strings(&table, norb, n_alpha);
    let beta_strings = sector_strings(&table, norb, n_beta);
    let num_modes = 2 * norb;

    // Whether an odd number of alpha electrons flips the sign of each beta operator.
    let beta_alpha_parity: i8 = if n_alpha.is_multiple_of(2) { 1 } else { -1 };

    let mut out = vec![Complex64::new(0.0, 0.0); dim];
    let mut alpha_ops: Vec<(bool, u32)> = Vec::new();
    let mut beta_ops: Vec<(bool, u32)> = Vec::new();
    // Per beta-source address, the term's action on that beta string: (out_addr, sign), or `None`
    // if it annihilates. Precomputed once per term so it is not recomputed for every alpha string.
    let mut beta_action: Vec<Option<(usize, i8)>> = Vec::new();
    for (coeff, actions, modes) in terms {
        // Split the term's ops into alpha- and beta-sector sublists (order preserved), validating
        // modes as we go.
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

        // The cross-sector sign is `(-1)^{n_alpha}` once per beta operator. Because each beta op is
        // applied while the alpha string still holds all `n_alpha` electrons (the alpha ops act on a
        // different sector and do not change the alpha count within a particle-conserving term), the
        // factor is constant across the term.
        let cross_sign = if beta_ops.len().is_multiple_of(2) {
            1
        } else {
            beta_alpha_parity
        };

        // Precompute the beta sector's action, folding in the cross-sector sign.
        beta_action.clear();
        beta_action.extend(beta_strings.iter().map(|&b_string| {
            apply_ops_to_string(b_string, &beta_ops)
                .map(|(b_out, b_sign)| (str2addr(&table, norb, n_beta, b_out), b_sign * cross_sign))
        }));

        for (a_addr, &a_string) in alpha_strings.iter().enumerate() {
            let Some((a_out, a_sign)) = apply_ops_to_string(a_string, &alpha_ops) else {
                continue;
            };
            let a_out_addr = str2addr(&table, norb, n_alpha, a_out);
            let src_row = a_addr * dim_b;
            let dst_row = a_out_addr * dim_b;
            for (b_addr, beta) in beta_action.iter().enumerate() {
                let amp = vec[src_row + b_addr];
                if amp == Complex64::new(0.0, 0.0) {
                    continue;
                }
                let Some((b_out_addr, b_sign)) = beta else {
                    continue;
                };
                let sign = a_sign * b_sign;
                out[dst_row + b_out_addr] += coeff * f64::from(sign) * amp;
            }
        }
    }
    Ok(out)
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
                        str2addr(&table, norb, nocc, string),
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
                    assert_eq!(
                        str2addr(&table, norb, nocc, string),
                        addr,
                        "round-trip failed"
                    );
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
                assert_eq!(str2addr(&table, norb, nocc, string), addr);
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
                    match apply_creation(string, p) {
                        LadderResult::Vanishes => {
                            assert!(occupied, "cre should vanish only if occupied")
                        }
                        LadderResult::Maps { string: out, sign } => {
                            assert!(!occupied);
                            assert_eq!(out, string | (1u64 << p));
                            assert_eq!(sign, sign_ref(string, p));
                            // creating raises the sector; address is valid there
                            assert_eq!(out.count_ones(), nocc + 1);
                            let _ = str2addr(&table, norb, nocc + 1, out);
                        }
                    }
                    match apply_annihilation(string, p) {
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
        let norb = 4u32;
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
            match apply_creation(src, orb) {
                LadderResult::Maps {
                    string: out,
                    sign: s,
                } => {
                    assert_eq!(s, sign, "cre sign nocc={nocc} src={src:b} orb={orb}");
                    assert_eq!(
                        str2addr(&table, norb, nocc + 1, out),
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
        let norb = 4u32;
        let golden: &[(u32, u64, u32, usize, i8)] = &[
            (2, 0b0011, 0, 1, -1),
            (2, 0b0011, 1, 0, 1),
            (2, 0b0110, 1, 2, -1),
            (2, 0b0110, 2, 1, 1),
            (2, 0b1100, 2, 3, -1),
            (2, 0b1100, 3, 2, 1),
        ];
        for &(nocc, src, orb, tgt, sign) in golden {
            match apply_annihilation(src, orb) {
                LadderResult::Maps {
                    string: out,
                    sign: s,
                } => {
                    assert_eq!(s, sign, "des sign nocc={nocc} src={src:b} orb={orb}");
                    assert_eq!(
                        str2addr(&table, norb, nocc - 1, out),
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
        terms: &[(Complex64, Vec<(bool, u32)>)],
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
        let dim = if spinless { dim_a } else { dim_a * dim_b };

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
                str2addr(&table, norb, n_alpha, mask)
            } else {
                let low = if norb >= 64 {
                    !0u64
                } else {
                    (1u64 << norb) - 1
                };
                let a_str = mask & low;
                let b_str = mask >> norb;
                let a = str2addr(&table, norb, n_alpha, a_str);
                let b = str2addr(&table, norb, n_beta, b_str);
                a * dim_b + b
            }
        };

        let mut out = vec![Complex64::new(0.0, 0.0); dim];
        for (coeff, ops) in terms {
            for (src_addr, &amp) in vec.iter().enumerate() {
                if amp == Complex64::new(0.0, 0.0) {
                    continue;
                }
                let mut mask = to_mask(src_addr);
                let mut sign: i8 = 1;
                let mut ok = true;
                for &(is_creation, mode) in ops.iter().rev() {
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

    /// Splits `(is_creation, orbital)` pairs into the parallel `(actions, modes)` slices that the
    /// production kernels consume, mirroring a `FermionOperatorTermView`'s native layout.
    fn split_ops(ops: &[(bool, u32)]) -> (Vec<bool>, Vec<u32>) {
        (
            ops.iter().map(|&(a, _)| a).collect(),
            ops.iter().map(|&(_, m)| m).collect(),
        )
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
        let terms: Vec<(Complex64, Vec<(bool, u32)>)> = vec![
            // hopping a†_0 a_1
            (Complex64::new(0.7, 0.2), vec![(true, 0), (false, 1)]),
            // its conjugate a†_1 a_0
            (Complex64::new(0.7, -0.2), vec![(true, 1), (false, 0)]),
            // number operator n_2 = a†_2 a_2
            (Complex64::new(0.5, 0.0), vec![(true, 2), (false, 2)]),
            // density-density n_0 n_2 (written interleaved to exercise sign)
            (
                Complex64::new(1.3, 0.0),
                vec![(true, 0), (false, 0), (true, 2), (false, 2)],
            ),
        ];
        for &(norb, nocc) in cases {
            let dim = BinomialTable::new(norb).num_strings(norb, nocc);
            // A deterministic non-trivial input vector.
            let vec: Vec<Complex64> = (0..dim)
                .map(|i| Complex64::new((i as f64 + 1.0) * 0.31, (i as f64) * -0.17))
                .collect();
            let expected = reference_matvec(norb, nocc, None, &terms, &vec);
            let split: Vec<(Complex64, Vec<bool>, Vec<u32>)> = terms
                .iter()
                .map(|(c, ops)| {
                    let (a, m) = split_ops(ops);
                    (*c, a, m)
                })
                .collect();
            let got = spinless_matvec(
                norb,
                nocc,
                split
                    .iter()
                    .map(|(c, a, m)| (*c, a.as_slice(), m.as_slice())),
                &vec,
            )
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
        let make_terms = |norb: u32| -> Vec<(Complex64, Vec<(bool, u32)>)> {
            vec![
                // alpha hopping a†_0 a_1
                (Complex64::new(0.9, 0.0), vec![(true, 0), (false, 1)]),
                // beta hopping a†_0 a_1  (beta modes are norb + orbital)
                (
                    Complex64::new(0.4, 0.1),
                    vec![(true, norb), (false, norb + 1)],
                ),
                // cross-spin density-density n^a_0 n^b_0 = a†_0 a_0 a†_{norb} a_{norb}
                (
                    Complex64::new(1.1, 0.0),
                    vec![(true, 0), (false, 0), (true, norb), (false, norb)],
                ),
                // interleaved cross-spin: a†_0(a) a†_{norb+2}(b) a_{norb}(b) a_2(a)  -- spin exchange
                // pattern that moves an alpha 2->0 and a beta 0->2, ops interleaved a,b,b,a.
                (
                    Complex64::new(0.6, -0.3),
                    vec![(true, 0), (true, norb + 2), (false, norb), (false, 2)],
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
            let split: Vec<(Complex64, Vec<bool>, Vec<u32>)> = terms
                .iter()
                .map(|(c, ops)| {
                    let (a, m) = split_ops(ops);
                    (*c, a, m)
                })
                .collect();
            let got = spinful_matvec(
                norb,
                n_a,
                n_b,
                split
                    .iter()
                    .map(|(c, a, m)| (*c, a.as_slice(), m.as_slice())),
                &vec,
            )
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
            let terms: Vec<(Complex64, Vec<(bool, u32)>)> =
                vec![(Complex64::new(1.0, 0.0), vec![(true, norb), (false, norb)])];
            let vec: Vec<Complex64> = (0..dim)
                .map(|i| Complex64::new(i as f64 + 1.0, 0.0))
                .collect();
            let expected = reference_matvec(norb, n_a, Some(n_b), &terms, &vec);
            let split: Vec<(Complex64, Vec<bool>, Vec<u32>)> = terms
                .iter()
                .map(|(c, ops)| {
                    let (a, m) = split_ops(ops);
                    (*c, a, m)
                })
                .collect();
            let got = spinful_matvec(
                norb,
                n_a,
                n_b,
                split
                    .iter()
                    .map(|(c, a, m)| (*c, a.as_slice(), m.as_slice())),
                &vec,
            )
            .unwrap();
            assert_vec_close(&got, &expected);
        }
    }

    #[test]
    fn matvec_dimension_and_mode_errors() {
        // Wrong vector length.
        let bad = complex_vec(&[1.0, 2.0]);
        let err = spinless_matvec(
            4,
            2,
            std::iter::once((
                Complex64::new(1.0, 0.0),
                [true, false].as_slice(),
                [0u32, 0].as_slice(),
            )),
            &bad,
        )
        .unwrap_err();
        assert!(matches!(err, FciMatvecError::DimensionMismatch { .. }));

        // Mode out of range (spinless: mode must be < norb).
        let dim = BinomialTable::new(4).num_strings(4, 2);
        let vec = complex_vec(&vec![1.0; dim]);
        let err = spinless_matvec(
            4,
            2,
            std::iter::once((
                Complex64::new(1.0, 0.0),
                [true, false].as_slice(),
                [9u32, 0].as_slice(),
            )),
            &vec,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            FciMatvecError::ModeOutOfRange {
                mode: 9,
                num_modes: 4
            }
        ));

        // Spinful: mode must be < 2*norb.
        let dim = {
            let t = BinomialTable::new(3);
            t.num_strings(3, 1) * t.num_strings(3, 1)
        };
        let vec = complex_vec(&vec![1.0; dim]);
        let err = spinful_matvec(
            3,
            1,
            1,
            std::iter::once((
                Complex64::new(1.0, 0.0),
                [true].as_slice(),
                [6u32].as_slice(),
            )),
            &vec,
        )
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
}
