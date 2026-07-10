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
}
