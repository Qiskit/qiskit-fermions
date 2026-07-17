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

use crate::operators::{CoherenceError, OperatorMacro, OperatorTrait, TermSortKey};
use num_complex::{Complex64, ComplexFloat};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::iter::zip;
use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign,
};

pub type FermionAction<'a> = (&'a bool, &'a u32);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FermionOperatorTermView<'a> {
    pub coeff: Complex64,
    pub actions: &'a [bool],
    pub modes: &'a [u32],
}

impl FermionOperatorTermView<'_> {
    pub fn iter(&'_ self) -> impl ExactSizeIterator<Item = FermionAction<'_>> + '_ {
        zip(self.actions, self.modes)
    }

    pub fn to_vec(&'_ self) -> Vec<FermionAction<'_>> {
        zip(self.actions, self.modes).collect()
    }

    pub fn into_vec(&'_ self) -> Vec<(bool, u32)> {
        zip(self.actions.to_vec(), self.modes.to_vec()).collect()
    }
}

impl TermSortKey for FermionOperatorTermView<'_> {
    fn sort_key(&self) -> impl Ord {
        // Compare the operator string position-by-position, each factor being an (action, mode)
        // pair. This matches `into_vec` (and hence the sorted display order), so the canonical
        // order agrees with how terms are printed.
        self.into_vec()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FermionOperatorGroupTermView<'a> {
    pub coeff: Complex64,
    pub actions: &'a [bool],
    pub modes: &'a [u32],
    pub group: u32,
}

impl FermionOperatorGroupTermView<'_> {
    pub fn iter(&'_ self) -> impl ExactSizeIterator<Item = FermionAction<'_>> + '_ {
        zip(self.actions, self.modes)
    }

    pub fn to_vec(&'_ self) -> Vec<FermionAction<'_>> {
        zip(self.actions, self.modes).collect()
    }

    pub fn into_vec(&'_ self) -> Vec<(bool, u32)> {
        zip(self.actions.to_vec(), self.modes.to_vec()).collect()
    }
}

impl TermSortKey for FermionOperatorGroupTermView<'_> {
    fn sort_key(&self) -> impl Ord {
        // Match the ungrouped `TermView` key exactly (ignoring `group`), so ordering a grouped
        // operator agrees with ordering the same terms ungrouped.
        self.into_vec()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FermionOperator {
    pub coeffs: Vec<Complex64>,
    pub actions: Vec<bool>,
    pub modes: Vec<u32>,
    pub boundaries: Vec<usize>,
    pub groups: Option<Vec<u32>>,
}

crate::impl_operator_macro!(FermionOperator);

impl FermionOperator {
    #[inline]
    pub fn coeffs(&self) -> &[Complex64] {
        &self.coeffs
    }

    #[inline]
    pub fn actions(&self) -> &[bool] {
        &self.actions
    }

    #[inline]
    pub fn modes(&self) -> &[u32] {
        &self.modes
    }

    #[inline]
    pub fn boundaries(&self) -> &[usize] {
        &self.boundaries
    }

    /// Appends a single term to the operator. Low-level building block for construction.
    ///
    /// # Warning
    ///
    /// This does **not** maintain `groups`: it pushes a coefficient and a boundary but no group
    /// index. Only call it while `self.groups` is `None` (e.g. on a fresh [`Self::zero`]); calling
    /// it on an operator that tracks groups leaves `coeffs.len()` out of sync with `groups.len()`,
    /// after which [`iter_with_groups`](Self::iter_with_groups) silently drops the trailing terms.
    pub fn _append_term(&mut self, coeff: Complex64, actions: &[bool], modes: &[u32]) {
        self.coeffs.push(coeff);
        self.actions.extend_from_slice(actions);
        self.modes.extend_from_slice(modes);
        self.boundaries.push(self.modes.len());
    }

    pub fn num_groups(&self) -> Option<u32> {
        let self_groups = self.groups.as_ref()?;
        if self_groups.is_empty() {
            Some(0)
        } else {
            Some(self_groups.iter().max().unwrap() + 1)
        }
    }

    pub fn split_out_groups(&self) -> Option<Vec<Self>> {
        let mut groups = vec![Self::zero(); self.num_groups()? as usize];
        for term in self.iter_with_groups() {
            groups[term.group as usize]._append_term(term.coeff, term.actions, term.modes);
        }
        Some(groups)
    }

    pub fn normal_ordered(&self, sandwich: Option<bool>) -> Self {
        let mut result = Self::zero();
        self.iter()
            .for_each(|term| result.__iadd__(&_normal_ordered_term(term, sandwich)));
        result
    }

    pub fn is_hermitian(&self, atol: f64) -> bool {
        let mut diff = (self.__sub__(&self.adjoint())).normal_ordered(None);
        diff.ichop(atol);
        diff.equiv(&Self::zero(), atol)
    }

    pub fn max_rank(&self) -> u32 {
        self.boundaries
            .windows(2)
            .map(|p| p[1] - p[0])
            .max()
            .unwrap_or(0) as u32
    }

    pub fn conserves_particle_number(&self) -> bool {
        // Particle-number conservation is sector conservation with all modes in one block.
        self.conserves_sector(&[])
    }

    /// Returns whether every term conserves particle number within each mode block.
    ///
    /// `block_sizes` partitions the mode range into consecutive, non-overlapping blocks: block `b`
    /// spans modes `[start_b, start_b + block_sizes[b])` where `start_b = sum(block_sizes[..b])`. A
    /// term conserves the sector iff, in *every* block, its number of creation operators equals its
    /// number of annihilation operators. A term touching a mode that lies beyond the last block does
    /// not conserve the sector (there is no block whose count it could balance).
    ///
    /// An empty `block_sizes` treats all modes as a single unbounded block, making this equivalent to
    /// [`Self::conserves_particle_number`].
    ///
    /// This underlies the fixed-sector FCI kernel: `[norb]` checks plain particle-number conservation
    /// (spinless), while `[norb, norb]` checks that the alpha block `[0, norb)` and beta block
    /// `[norb, 2 * norb)` are *each* conserved, i.e. strict conservation of both particle number and
    /// the z-component of spin.
    pub fn conserves_sector(&self, block_sizes: &[u32]) -> bool {
        // The exclusive upper bound of each block; the last entry is the total mode count. Empty
        // `block_sizes` yields no bounds, which the block lookup below treats as one unbounded block.
        let bounds: Vec<u32> = block_sizes
            .iter()
            .scan(0, |acc, size| {
                *acc += size;
                Some(*acc)
            })
            .collect();
        // Net (creations - annihilations) per block, reused across terms.
        let mut net = vec![0i64; block_sizes.len().max(1)];
        for term in self.iter() {
            net.iter_mut().for_each(|n| *n = 0);
            let mut in_range = true;
            for (&action, &mode) in term.iter() {
                // The block containing `mode`: the first bound strictly greater than it. With no
                // bounds every mode falls into the single block 0.
                let block = match bounds.iter().position(|&b| mode < b) {
                    Some(b) => b,
                    None if bounds.is_empty() => 0,
                    None => {
                        in_range = false;
                        break;
                    }
                };
                net[block] += if action { 1 } else { -1 };
            }
            if !in_range || net.iter().any(|&n| n != 0) {
                return false;
            }
        }
        true
    }

    /// Applies this operator to a spinless FCI state vector: returns `self @ vec`.
    ///
    /// The operator's `norb` modes are treated as spinless orbitals. `vec` is a state vector of the
    /// `nocc`-particle sector, of length `C(norb, nocc)`, addressed by the combinatorial-number-system
    /// convention of [`crate::linalg::fci`] (matching `pyscf.fci.cistring`, hence `ffsim`). This is the
    /// operator-specific entry point; the addressing and sign arithmetic live in [`crate::linalg::fci`].
    pub fn fci_matvec_spinless(
        &self,
        norb: u32,
        nocc: u32,
        vec: &[Complex64],
    ) -> Result<Vec<Complex64>, crate::linalg::fci::FciMatvecError> {
        crate::linalg::fci::spinless_matvec(
            norb,
            nocc,
            self.iter().map(|t| (t.coeff, t.actions, t.modes)),
            vec,
        )
    }

    /// Applies this operator to a spinful FCI state vector: returns `self @ vec`.
    ///
    /// The operator's `2 * norb` modes are spin-orbitals under the block-spin convention (mode
    /// `m < norb` is alpha orbital `m`; mode `m >= norb` is beta orbital `m - norb`). `vec` is a state
    /// vector of the `(n_alpha, n_beta)` sector, of length `C(norb, n_alpha) * C(norb, n_beta)`, with
    /// flat index `addr_a * C(norb, n_beta) + addr_b`. See [`crate::linalg::fci`] for the addressing
    /// and sign conventions.
    pub fn fci_matvec_spinful(
        &self,
        norb: u32,
        n_alpha: u32,
        n_beta: u32,
        vec: &[Complex64],
    ) -> Result<Vec<Complex64>, crate::linalg::fci::FciMatvecError> {
        crate::linalg::fci::spinful_matvec(
            norb,
            n_alpha,
            n_beta,
            self.iter().map(|t| (t.coeff, t.actions, t.modes)),
            vec,
        )
    }

    /// Compiles this operator into a reusable spinless scatter map for a precomputed sector.
    ///
    /// The returned [`crate::linalg::fci::CompiledSector`] captures the operator's vector-independent
    /// action once; its `apply`/`apply_conj` then serve the many matvecs/rmatvecs of an evolution
    /// (`expm_multiply`) without re-walking the ladder operators, re-checking conservation, or
    /// re-ranking destinations on every call. See [`crate::linalg::fci::SpinlessSector::compile`].
    pub fn compile_fci_spinless(
        &self,
        sector: &crate::linalg::fci::SpinlessSector,
    ) -> Result<crate::linalg::fci::CompiledSector, crate::linalg::fci::FciMatvecError> {
        sector.compile(self.iter().map(|t| (t.coeff, t.actions, t.modes)))
    }

    /// Compiles this operator into a reusable spinful scatter map for a precomputed sector.
    ///
    /// The spinful counterpart of [`Self::compile_fci_spinless`]; see
    /// [`crate::linalg::fci::SpinfulSector::compile`].
    pub fn compile_fci_spinful(
        &self,
        sector: &crate::linalg::fci::SpinfulSector,
    ) -> Result<crate::linalg::fci::CompiledSector, crate::linalg::fci::FciMatvecError> {
        sector.compile(self.iter().map(|t| (t.coeff, t.actions, t.modes)))
    }
}

fn _normal_ordered_term(
    term_view: FermionOperatorTermView,
    sandwich: Option<bool>,
) -> FermionOperator {
    let mut coeffs = vec![];
    let mut actions = vec![];
    let mut modes = vec![];
    let mut boundaries = vec![0];

    let mut stack = vec![(term_view.to_vec(), term_view.coeff)];
    while let Some((mut term, coeff)) = stack.pop() {
        let mut parity = false;
        let mut zero = false;
        for i in 1..term.len() {
            // shift the operator at index i to the left until it's in the correct location
            for j in (1..=i).rev() {
                let (action_right, index_right) = term[j];
                let (action_left, index_left) = term[j - 1];
                if *action_right == *action_left {
                    // both create or both destroy
                    match ((index_right).cmp(index_left), sandwich, *action_left) {
                        (Ordering::Equal, _, _) => {
                            // operators are the same, so product is zero
                            zero = true;
                            break;
                        }
                        (Ordering::Greater, None, _)
                        | (Ordering::Less, Some(true), true)
                        | (Ordering::Less, Some(false), false)
                        | (Ordering::Greater, Some(true), false)
                        | (Ordering::Greater, Some(false), true) => {
                            // swap operators and update sign
                            term.swap(j - 1, j);
                            parity = !parity;
                        }
                        (Ordering::Less, None, _)
                        | (Ordering::Greater, Some(true), true)
                        | (Ordering::Greater, Some(false), false)
                        | (Ordering::Less, Some(true), false)
                        | (Ordering::Less, Some(false), true) => {}
                    }
                } else if *action_right && !*action_left {
                    // create on right and destroy on left
                    if index_right == index_left {
                        // add new term
                        let mut new_term: Vec<FermionAction> = Vec::new();
                        new_term.extend(&term[..j - 1]);
                        new_term.extend(&term[j + 1..]);
                        let signed_coeff = if parity { -coeff } else { coeff };
                        stack.push((new_term, signed_coeff))
                    }
                    // swap operators and update sign
                    term.swap(j - 1, j);
                    parity = !parity;
                }
            }
        }
        if zero {
            continue;
        }
        let signed_coeff = if parity { -coeff } else { coeff };
        coeffs.push(signed_coeff);
        term.iter().for_each(|&(&a, &i)| {
            actions.push(a);
            modes.push(i);
        });
        boundaries.push(modes.len())
    }
    FermionOperator {
        coeffs,
        actions,
        modes,
        boundaries,
        groups: None,
    }
}

fn _compose(
    a: &FermionOperator,
    b: &FermionOperator,
) -> (Vec<Complex64>, Vec<bool>, Vec<u32>, Vec<usize>) {
    let mut coeffs = vec![];
    let mut actions = vec![];
    let mut modes = vec![];
    let mut boundaries = vec![0];

    for left in a.iter() {
        for right in b.iter() {
            coeffs.push(left.coeff * right.coeff);
            actions.extend_from_slice(right.actions);
            actions.extend_from_slice(left.actions);
            modes.extend_from_slice(right.modes);
            modes.extend_from_slice(left.modes);
            boundaries.push(modes.len());
        }
    }
    (coeffs, actions, modes, boundaries)
}

impl OperatorTrait for FermionOperator {
    type TermView<'a> = FermionOperatorTermView<'a>;
    type GroupTermView<'a> = FermionOperatorGroupTermView<'a>;

    fn zero() -> Self {
        Self {
            coeffs: vec![],
            actions: vec![],
            modes: vec![],
            boundaries: vec![0],
            groups: None,
        }
    }

    fn one() -> Self {
        Self {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            actions: vec![],
            modes: vec![],
            boundaries: vec![0, 0],
            groups: None,
        }
    }

    fn equiv(&self, other: &Self, atol: f64) -> bool {
        let mut diff = self.__sub__(other);
        diff = diff.simplify(atol);
        for c in diff.coeffs {
            if c.abs() > atol {
                return false;
            }
        }
        true
    }

    fn simplify(&self, atol: f64) -> Self {
        let mut terms = HashMap::new();
        for term in self.iter() {
            terms
                .entry((term.modes, term.actions))
                .and_modify(|c| *c += term.coeff)
                .or_insert(term.coeff);
        }
        let mut out = Self::zero();
        terms
            .iter()
            .filter(|((_, _), coeff)| coeff.abs() > atol)
            .for_each(|((modes, actions), coeff)| out._append_term(*coeff, actions, modes));
        out
    }

    fn adjoint(&self) -> Self {
        let mut coeffs = vec![];
        let mut actions = vec![];
        let mut modes = vec![];

        self.iter().for_each(|term| {
            coeffs.push(term.coeff.conj());
            actions.extend(term.actions.iter().rev().map(|a| !a));
            modes.extend(term.modes.iter().rev());
        });

        Self {
            coeffs,
            actions,
            modes,
            boundaries: self.boundaries.to_vec(),
            groups: self.groups.clone(),
        }
    }

    fn __iadd__(&mut self, other: &Self) {
        self.coeffs.extend_from_slice(&other.coeffs);
        self.actions.extend_from_slice(&other.actions);
        self.modes.extend_from_slice(&other.modes);
        let offset = self.boundaries[self.boundaries.len() - 1];
        self.boundaries
            .extend(other.boundaries[1..].iter().map(|b| b + offset));
        self.groups = None;
    }

    fn __imul__(&mut self, other: Complex64) {
        self.coeffs.iter_mut().for_each(|c| *c *= other);
    }

    fn __iand__(&mut self, other: &Self) {
        (self.coeffs, self.actions, self.modes, self.boundaries) = _compose(self, other);
        self.groups = None;
    }

    fn __imatmul__(&mut self, other: &Self) {
        (self.coeffs, self.actions, self.modes, self.boundaries) = _compose(other, self);
        self.groups = None;
    }

    fn ichop(&mut self, atol: f64) {
        let mut coeffs = vec![];
        let mut actions = vec![];
        let mut modes = vec![];
        let mut boundaries = vec![0];

        self.iter()
            .filter(|term| term.coeff.abs() > atol)
            .for_each(|term| {
                coeffs.push(term.coeff.conj());
                actions.extend_from_slice(term.actions);
                modes.extend_from_slice(term.modes);
                boundaries.push(modes.len());
            });

        self.coeffs = coeffs;
        self.actions = actions;
        self.modes = modes;
        self.boundaries = boundaries;
        self.groups = None;
    }

    fn iter(&self) -> impl ExactSizeIterator<Item = Self::TermView<'_>> {
        self.coeffs.iter().enumerate().map(|(i, coeff)| {
            let start = self.boundaries[i];
            let end = self.boundaries[i + 1];
            FermionOperatorTermView {
                coeff: *coeff,
                actions: &self.actions[start..end],
                modes: &self.modes[start..end],
            }
        })
    }

    fn has_groups(&self) -> bool {
        self.groups.is_some()
    }

    fn iter_with_groups(&self) -> impl ExactSizeIterator<Item = Self::GroupTermView<'_>> {
        if let Some(groups) = &self.groups {
            return self
                .coeffs
                .iter()
                .zip(groups)
                .enumerate()
                .map(|(i, (coeff, gidx))| {
                    let start = self.boundaries[i];
                    let end = self.boundaries[i + 1];
                    FermionOperatorGroupTermView {
                        coeff: *coeff,
                        actions: &self.actions[start..end],
                        modes: &self.modes[start..end],
                        group: *gidx,
                    }
                });
        }
        panic!("This method can only be called when groups are present!");
    }

    fn from_terms<'a, I>(terms: I) -> Self
    where
        Self: 'a,
        I: IntoIterator<Item = Self::TermView<'a>>,
    {
        let mut out = Self::zero();
        for term in terms {
            out._append_term(term.coeff, term.actions, term.modes);
        }
        out
    }

    fn from_terms_with_groups<'a, I>(terms: I) -> Self
    where
        Self: 'a,
        I: IntoIterator<Item = Self::GroupTermView<'a>>,
    {
        let mut out = Self::zero();
        let mut groups = Vec::new();
        for term in terms {
            out._append_term(term.coeff, term.actions, term.modes);
            groups.push(term.group);
        }
        out.groups = Some(groups);
        out
    }

    fn get_support(&self) -> HashSet<u32> {
        HashSet::from_iter(self.modes.clone())
    }

    fn relabel_modes(&self, permutation: Vec<u32>) -> Result<Self, CoherenceError> {
        if permutation.iter().collect::<HashSet<_>>().len() != permutation.len() {
            return Err(CoherenceError::DuplicateIndices);
        }
        let mut out = self.clone();
        let new_modes: Result<Vec<u32>, CoherenceError> = self
            .modes
            .iter()
            .map(|&idx| {
                permutation
                    .get(idx as usize)
                    .cloned()
                    .ok_or(CoherenceError::IndexMapTooSmall)
            })
            .collect();
        out.modes = new_modes?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero() {
        let zero = FermionOperator::zero();
        assert_eq!(
            zero,
            FermionOperator {
                coeffs: vec![],
                actions: vec![],
                modes: vec![],
                boundaries: vec![0],
                groups: None,
            }
        );
    }

    #[test]
    fn test_one() {
        let one = FermionOperator::one();
        assert_eq!(
            one,
            FermionOperator {
                coeffs: vec![Complex64::new(1.0, 0.0)],
                actions: vec![],
                modes: vec![],
                boundaries: vec![0, 0],
                groups: None,
            }
        );
    }

    #[test]
    fn test_add() {
        let one = FermionOperator::one();
        let two = FermionOperator {
            coeffs: vec![Complex64::new(2.0, 0.0)],
            actions: vec![],
            modes: vec![],
            boundaries: vec![0, 0],
            groups: None,
        };
        let three = one + two;
        assert_eq!(
            three,
            FermionOperator {
                coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
                actions: vec![],
                modes: vec![],
                boundaries: vec![0, 0, 0],
                groups: None,
            }
        );
    }

    #[test]
    fn test_add_assign() {
        let mut op = FermionOperator::one();
        let two = FermionOperator {
            coeffs: vec![Complex64::new(2.0, 0.0)],
            actions: vec![],
            modes: vec![],
            boundaries: vec![0, 0],
            groups: None,
        };
        op += two;
        assert_eq!(
            op,
            FermionOperator {
                coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
                actions: vec![],
                modes: vec![],
                boundaries: vec![0, 0, 0],
                groups: None,
            }
        );
    }

    #[test]
    fn test_sub() {
        let one = FermionOperator::one();
        let two = FermionOperator {
            coeffs: vec![Complex64::new(2.0, 0.0)],
            actions: vec![],
            modes: vec![],
            boundaries: vec![0, 0],
            groups: None,
        };
        let new_one = two - one;
        assert_eq!(
            new_one,
            FermionOperator {
                coeffs: vec![Complex64::new(2.0, 0.0), Complex64::new(-1.0, 0.0)],
                actions: vec![],
                modes: vec![],
                boundaries: vec![0, 0, 0],
                groups: None,
            }
        );
    }

    #[test]
    fn test_sub_assign() {
        let mut op = FermionOperator::one();
        let two = FermionOperator {
            coeffs: vec![Complex64::new(2.0, 0.0)],
            actions: vec![],
            modes: vec![],
            boundaries: vec![0, 0],
            groups: None,
        };
        op -= two;
        assert_eq!(
            op,
            FermionOperator {
                coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(-2.0, 0.0)],
                actions: vec![],
                modes: vec![],
                boundaries: vec![0, 0, 0],
                groups: None,
            }
        );
    }

    #[test]
    fn test_mul() {
        let one = FermionOperator::one();
        let three = one * Complex64::new(3.0, 0.0);
        assert_eq!(
            three,
            FermionOperator {
                coeffs: vec![Complex64::new(3.0, 0.0)],
                actions: vec![],
                modes: vec![],
                boundaries: vec![0, 0],
                groups: None,
            }
        );
    }

    #[test]
    fn test_rmul() {
        let one = FermionOperator::one();
        let three = Complex64::new(3.0, 0.0) * one;
        assert_eq!(
            three,
            FermionOperator {
                coeffs: vec![Complex64::new(3.0, 0.0)],
                actions: vec![],
                modes: vec![],
                boundaries: vec![0, 0],
                groups: None,
            }
        );
    }

    #[test]
    fn test_mul_assign() {
        let mut op = FermionOperator::one();
        op *= Complex64::new(3.0, 0.0);
        assert_eq!(
            op,
            FermionOperator {
                coeffs: vec![Complex64::new(3.0, 0.0)],
                actions: vec![],
                modes: vec![],
                boundaries: vec![0, 0],
                groups: None,
            }
        );
    }

    #[test]
    fn test_div() {
        let three = FermionOperator {
            coeffs: vec![Complex64::new(3.0, 0.0)],
            actions: vec![],
            modes: vec![],
            boundaries: vec![0, 0],
            groups: None,
        };
        let one_half = three / Complex64::new(2.0, 0.0);
        assert_eq!(
            one_half,
            FermionOperator {
                coeffs: vec![Complex64::new(1.5, 0.0)],
                actions: vec![],
                modes: vec![],
                boundaries: vec![0, 0],
                groups: None,
            }
        );
    }

    #[test]
    fn test_idiv() {
        let mut op = FermionOperator {
            coeffs: vec![Complex64::new(3.0, 0.0)],
            actions: vec![],
            modes: vec![],
            boundaries: vec![0, 0],
            groups: None,
        };
        op /= Complex64::new(2.0, 0.0);
        assert_eq!(
            op,
            FermionOperator {
                coeffs: vec![Complex64::new(1.5, 0.0)],
                actions: vec![],
                modes: vec![],
                boundaries: vec![0, 0],
                groups: None,
            }
        );
    }

    #[test]
    fn test_neg() {
        let one = FermionOperator::one();
        assert_eq!(
            -one,
            FermionOperator {
                coeffs: vec![Complex64::new(-1.0, 0.0)],
                actions: vec![],
                modes: vec![],
                boundaries: vec![0, 0],
                groups: None,
            }
        );
    }

    #[test]
    fn test_and() {
        let op1 = FermionOperator {
            coeffs: vec![Complex64::new(2.0, 0.0), Complex64::new(3.0, 0.0)],
            actions: vec![true, false],
            modes: vec![0, 1],
            boundaries: vec![0, 0, 2],
            groups: None,
        };
        let op2 = FermionOperator {
            coeffs: vec![Complex64::new(1.5, 0.0), Complex64::new(4.0, 0.0)],
            actions: vec![true, false],
            modes: vec![1, 0],
            boundaries: vec![0, 0, 2],
            groups: None,
        };
        let result = op1 & op2;
        assert_eq!(
            result,
            FermionOperator {
                coeffs: vec![
                    Complex64::new(3.0, 0.0),
                    Complex64::new(8.0, 0.0),
                    Complex64::new(4.5, 0.0),
                    Complex64::new(12.0, 0.0),
                ],
                actions: vec![true, false, true, false, true, false, true, false],
                modes: vec![1, 0, 0, 1, 1, 0, 0, 1],
                boundaries: vec![0, 0, 2, 4, 8],
                groups: None,
            }
        );
    }

    #[test]
    fn test_and_assign() {
        let mut op1 = FermionOperator {
            coeffs: vec![Complex64::new(2.0, 0.0), Complex64::new(3.0, 0.0)],
            actions: vec![true, false],
            modes: vec![0, 1],
            boundaries: vec![0, 0, 2],
            groups: None,
        };
        let op2 = FermionOperator {
            coeffs: vec![Complex64::new(1.5, 0.0), Complex64::new(4.0, 0.0)],
            actions: vec![true, false],
            modes: vec![1, 0],
            boundaries: vec![0, 0, 2],
            groups: None,
        };
        op1 &= op2;
        assert_eq!(
            op1,
            FermionOperator {
                coeffs: vec![
                    Complex64::new(3.0, 0.0),
                    Complex64::new(8.0, 0.0),
                    Complex64::new(4.5, 0.0),
                    Complex64::new(12.0, 0.0),
                ],
                actions: vec![true, false, true, false, true, false, true, false],
                modes: vec![1, 0, 0, 1, 1, 0, 0, 1],
                boundaries: vec![0, 0, 2, 4, 8],
                groups: None,
            }
        );
    }

    #[test]
    fn test_matmul() {
        let op1 = FermionOperator {
            coeffs: vec![Complex64::new(2.0, 0.0), Complex64::new(3.0, 0.0)],
            actions: vec![true, false],
            modes: vec![0, 1],
            boundaries: vec![0, 0, 2],
            groups: None,
        };
        let op2 = FermionOperator {
            coeffs: vec![Complex64::new(1.5, 0.0), Complex64::new(4.0, 0.0)],
            actions: vec![true, false],
            modes: vec![1, 0],
            boundaries: vec![0, 0, 2],
            groups: None,
        };
        let result = op2.__matmul__(&op1);
        assert_eq!(
            result,
            FermionOperator {
                coeffs: vec![
                    Complex64::new(3.0, 0.0),
                    Complex64::new(8.0, 0.0),
                    Complex64::new(4.5, 0.0),
                    Complex64::new(12.0, 0.0),
                ],
                actions: vec![true, false, true, false, true, false, true, false],
                modes: vec![1, 0, 0, 1, 1, 0, 0, 1],
                boundaries: vec![0, 0, 2, 4, 8],
                groups: None,
            }
        );
    }

    #[test]
    fn test_matmul_assign() {
        let op1 = FermionOperator {
            coeffs: vec![Complex64::new(2.0, 0.0), Complex64::new(3.0, 0.0)],
            actions: vec![true, false],
            modes: vec![0, 1],
            boundaries: vec![0, 0, 2],
            groups: None,
        };
        let mut op2 = FermionOperator {
            coeffs: vec![Complex64::new(1.5, 0.0), Complex64::new(4.0, 0.0)],
            actions: vec![true, false],
            modes: vec![1, 0],
            boundaries: vec![0, 0, 2],
            groups: None,
        };
        op2.__imatmul__(&op1);
        assert_eq!(
            op2,
            FermionOperator {
                coeffs: vec![
                    Complex64::new(3.0, 0.0),
                    Complex64::new(8.0, 0.0),
                    Complex64::new(4.5, 0.0),
                    Complex64::new(12.0, 0.0),
                ],
                actions: vec![true, false, true, false, true, false, true, false],
                modes: vec![1, 0, 0, 1, 1, 0, 0, 1],
                boundaries: vec![0, 0, 2, 4, 8],
                groups: None,
            }
        );
    }

    #[test]
    fn test_pow() {
        let op = FermionOperator {
            coeffs: vec![Complex64::new(2.0, 0.0)],
            actions: vec![true],
            modes: vec![0],
            boundaries: vec![0, 1],
            groups: None,
        };
        // exponent=0
        let one = FermionOperator::one();
        assert_eq!(op.__pow__(0), one);

        // exponent=1
        assert_eq!(op.__pow__(1), op);

        // exponent=2
        let squared = op.__pow__(2);
        assert_eq!(
            squared,
            FermionOperator {
                coeffs: vec![Complex64::new(4.0, 0.0)],
                actions: vec![true, true],
                modes: vec![0, 0],
                boundaries: vec![0, 2],
                groups: None,
            }
        );
    }

    #[test]
    fn test_ichop() {
        let mut op = FermionOperator {
            coeffs: vec![
                Complex64::new(1e-4, 0.0),
                Complex64::new(1e-6, 0.0),
                Complex64::new(1e-8, 0.0),
            ],
            actions: vec![true, false],
            modes: vec![0, 0],
            boundaries: vec![0, 0, 1, 2],
            groups: None,
        };

        op.ichop(1e-7);

        let expected1 = FermionOperator {
            coeffs: vec![Complex64::new(1e-4, 0.0), Complex64::new(1e-6, 0.0)],
            actions: vec![true],
            modes: vec![0],
            boundaries: vec![0, 0, 1],
            groups: None,
        };

        assert_eq!(op, expected1);

        op.ichop(1e-5);

        let expected2 = FermionOperator {
            coeffs: vec![Complex64::new(1e-4, 0.0)],
            actions: vec![],
            modes: vec![],
            boundaries: vec![0, 0],
            groups: None,
        };

        assert_eq!(op, expected2);
    }

    #[test]
    fn test_adjoint() {
        let op1 = FermionOperator {
            coeffs: vec![Complex64::new(0.0, 2.0), Complex64::new(3.0, 0.0)],
            actions: vec![true, false],
            modes: vec![0, 1],
            boundaries: vec![0, 0, 2],
            groups: None,
        };
        let adj = op1.adjoint();
        assert_eq!(
            adj,
            FermionOperator {
                coeffs: vec![Complex64::new(0.0, -2.0), Complex64::new(3.0, 0.0)],
                actions: vec![true, false],
                modes: vec![1, 0],
                boundaries: vec![0, 0, 2],
                groups: None,
            }
        );
    }

    #[test]
    fn test_equiv() {
        let zero = FermionOperator::zero();
        let op = Complex64::new(1e-8, 0.0) * FermionOperator::one();
        assert!(op.equiv(&zero, 1e-6));
        assert!(!op.equiv(&zero, 1e-10));
    }

    #[test]
    fn test_normal_ordered_1() {
        let op = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            actions: vec![true, false],
            modes: vec![0, 1],
            boundaries: vec![0, 2],
            groups: None,
        };

        assert_eq!(op.normal_ordered(None), op);
    }

    #[test]
    fn test_normal_ordered_2() {
        let op = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            actions: vec![true, true],
            modes: vec![0, 1],
            boundaries: vec![0, 2],
            groups: None,
        };

        let expected = FermionOperator {
            coeffs: vec![Complex64::new(-1.0, 0.0)],
            actions: vec![true, true],
            modes: vec![1, 0],
            boundaries: vec![0, 2],
            groups: None,
        };

        assert_eq!(op.normal_ordered(None), expected);
    }

    #[test]
    fn test_normal_ordered_3() {
        let op = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            actions: vec![false, true],
            modes: vec![0, 0],
            boundaries: vec![0, 2],
            groups: None,
        };

        let expected = FermionOperator {
            coeffs: vec![Complex64::new(-1.0, 0.0), Complex64::new(1.0, 0.0)],
            actions: vec![true, false],
            modes: vec![0, 0],
            boundaries: vec![0, 2, 2],
            groups: None,
        };

        assert_eq!(op.normal_ordered(None), expected);
    }

    #[test]
    fn test_normal_ordered_sandwich_none() {
        let op = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            actions: vec![false, false, true, true],
            modes: vec![0, 1, 0, 1],
            boundaries: vec![0, 4],
            groups: None,
        };

        let expected = FermionOperator {
            coeffs: vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(-1.0, 0.0),
            ],
            actions: vec![true, true, false, false, true, false, true, false],
            modes: vec![1, 0, 1, 0, 0, 0, 1, 1],
            boundaries: vec![0, 4, 6, 8, 8],
            groups: None,
        };

        assert_eq!(op.normal_ordered(None), expected);
    }

    #[test]
    fn test_normal_ordered_sandwich_true() {
        let op = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            actions: vec![false, false, true, true],
            modes: vec![1, 0, 0, 1],
            boundaries: vec![0, 4],
            groups: None,
        };

        let expected = FermionOperator {
            coeffs: vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(-1.0, 0.0),
                Complex64::new(-1.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
            actions: vec![true, true, false, false, true, false, true, false],
            modes: vec![0, 1, 1, 0, 0, 0, 1, 1],
            boundaries: vec![0, 4, 6, 8, 8],
            groups: None,
        };

        assert_eq!(op.normal_ordered(Some(true)), expected);
    }

    #[test]
    fn test_normal_ordered_sandwich_false() {
        let op = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            actions: vec![false, false, true, true],
            modes: vec![0, 1, 1, 0],
            boundaries: vec![0, 4],
            groups: None,
        };

        let expected = FermionOperator {
            coeffs: vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(-1.0, 0.0),
                Complex64::new(-1.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
            actions: vec![true, true, false, false, true, false, true, false],
            modes: vec![1, 0, 0, 1, 1, 1, 0, 0],
            boundaries: vec![0, 4, 6, 8, 8],
            groups: None,
        };

        assert_eq!(op.normal_ordered(Some(false)), expected);
    }

    #[test]
    fn test_is_hermitian() {
        let op = FermionOperator {
            coeffs: vec![Complex64::new(0.0, 1.00001), Complex64::new(0.0, -1.0)],
            actions: vec![true, false, true, false],
            modes: vec![0, 1, 1, 0],
            boundaries: vec![0, 2, 4],
            groups: None,
        };
        assert!(op.is_hermitian(1e-4));
        assert!(!op.is_hermitian(1e-6));
    }

    #[test]
    fn test_max_rank() {
        assert_eq!(FermionOperator::zero().max_rank(), 0);

        assert_eq!(FermionOperator::one().max_rank(), 0);

        assert_eq!(
            FermionOperator {
                coeffs: vec![Complex64::new(1.0, 0.0)],
                actions: vec![true],
                modes: vec![0],
                boundaries: vec![0, 1],
                groups: None,
            }
            .max_rank(),
            1
        );

        assert_eq!(
            FermionOperator {
                coeffs: vec![Complex64::new(1.0, 0.0)],
                actions: vec![true, false],
                modes: vec![0, 1],
                boundaries: vec![0, 2],
                groups: None,
            }
            .max_rank(),
            2
        );
    }

    #[test]
    fn test_conserves_particle_number() {
        let op1 = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            actions: vec![true, false],
            modes: vec![0, 1],
            boundaries: vec![0, 2],
            groups: None,
        };

        assert!(op1.conserves_particle_number());

        let op2 = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            actions: vec![true],
            modes: vec![0],
            boundaries: vec![0, 1],
            groups: None,
        };

        assert!(!op2.conserves_particle_number());
    }

    #[test]
    fn test_conserves_sector() {
        // a†_0 a_1: hopping within a single 2-orbital spinless block -> conserves.
        let hop = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            actions: vec![true, false],
            modes: vec![0, 1],
            boundaries: vec![0, 2],
            groups: None,
        };
        assert!(hop.conserves_sector(&[2]));
        // Empty block_sizes reduces to plain particle-number conservation.
        assert!(hop.conserves_sector(&[]));

        // a†_0 a_2 with a spinful split [2, 2]: moves a particle from the alpha block [0, 2) to the
        // beta block [2, 4) -> conserves total number but NOT the per-block (Sz) counts.
        let spin_flip = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            actions: vec![true, false],
            modes: vec![0, 2],
            boundaries: vec![0, 2],
            groups: None,
        };
        assert!(spin_flip.conserves_particle_number());
        assert!(!spin_flip.conserves_sector(&[2, 2]));
        // Treated as one 4-orbital spinless block, the same term conserves.
        assert!(spin_flip.conserves_sector(&[4]));

        // a†_2 (bare creation): does not conserve any sector.
        let raise = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            actions: vec![true],
            modes: vec![2],
            boundaries: vec![0, 1],
            groups: None,
        };
        assert!(!raise.conserves_sector(&[4]));

        // A mode beyond the last block (mode 4 with blocks summing to 4) cannot conserve.
        let out_of_range = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            actions: vec![true, false],
            modes: vec![0, 4],
            boundaries: vec![0, 2],
            groups: None,
        };
        assert!(!out_of_range.conserves_sector(&[2, 2]));
    }

    #[test]
    fn test_get_support() {
        let op = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
            actions: vec![true, false, true, true, false, false],
            modes: vec![0, 4, 1, 3, 4, 7],
            boundaries: vec![0, 2, 4],
            groups: None,
        };

        assert_eq!(op.get_support(), HashSet::from([0, 1, 3, 4, 7]));
    }

    #[test]
    fn test_relabel_modes() {
        let op = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
            actions: vec![true, false, true, false, true, false],
            modes: vec![0, 1, 0, 0, 2, 3],
            boundaries: vec![0, 2, 6],
            groups: None,
        };

        let permutation = vec![4, 2, 5, 3];

        let relabeled = op.relabel_modes(permutation).ok();

        let expected = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
            actions: vec![true, false, true, false, true, false],
            modes: vec![4, 2, 4, 4, 5, 3],
            boundaries: vec![0, 2, 6],
            groups: None,
        };

        assert_eq!(relabeled, Some(expected));
    }

    #[test]
    fn test_relabel_modes_duplicate_err() {
        let op = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
            actions: vec![true, false, true, false, true, false],
            modes: vec![0, 1, 0, 0, 2, 3],
            boundaries: vec![0, 2, 6],
            groups: None,
        };

        let permutation = vec![4, 4, 2, 3];

        let relabeled = op.relabel_modes(permutation);

        assert!(matches!(relabeled, Err(CoherenceError::DuplicateIndices)));
    }

    #[test]
    fn test_relabel_modes_index_too_small_err() {
        let op = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
            actions: vec![true, false, true, false, true, false],
            modes: vec![0, 1, 0, 0, 2, 3],
            boundaries: vec![0, 2, 6],
            groups: None,
        };

        let permutation = vec![4, 2, 5];

        let relabeled = op.relabel_modes(permutation);

        assert!(matches!(relabeled, Err(CoherenceError::IndexMapTooSmall)));
    }

    #[test]
    fn test_num_groups() {
        let mut zero = FermionOperator::zero();

        assert!(zero.num_groups().is_none());

        zero.groups = Some(vec![]);

        assert_eq!(zero.num_groups(), Some(0));

        let mut one = FermionOperator::one();
        one.groups = Some(vec![0]);

        assert_eq!(one.num_groups(), Some(1));
    }

    #[test]
    fn test_split_out_groups() {
        let op = FermionOperator {
            coeffs: vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
            ],
            actions: vec![true, false, true, false, true, true, false, false],
            modes: vec![0, 1, 1, 0, 0, 0, 1, 1],
            boundaries: vec![0, 2, 4, 8],
            groups: Some(vec![0, 0, 1]),
        };

        let expected = vec![
            FermionOperator {
                coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
                actions: vec![true, false, true, false],
                modes: vec![0, 1, 1, 0],
                boundaries: vec![0, 2, 4],
                groups: None,
            },
            FermionOperator {
                coeffs: vec![Complex64::new(2.0, 0.0)],
                actions: vec![true, true, false, false],
                modes: vec![0, 0, 1, 1],
                boundaries: vec![0, 4],
                groups: None,
            },
        ];

        let groups = op.split_out_groups();
        assert_eq!(groups, Some(expected));
    }

    #[test]
    fn test_split_out_groups_err() {
        let op = FermionOperator {
            coeffs: vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
            ],
            actions: vec![true, false, true, false, true, true, false, false],
            modes: vec![0, 1, 1, 0, 0, 0, 1, 1],
            boundaries: vec![0, 2, 4, 8],
            groups: None,
        };

        let groups = op.split_out_groups();
        assert!(groups.is_none());
    }

    #[test]
    fn test_iter_with_groups() {
        let op = FermionOperator {
            coeffs: vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
            ],
            actions: vec![true, false, true, false, true, true, false, false],
            modes: vec![0, 1, 1, 0, 0, 0, 1, 1],
            boundaries: vec![0, 2, 4, 8],
            groups: Some(vec![0, 0, 1]),
        };

        let terms: Vec<FermionOperatorGroupTermView> = op.iter_with_groups().collect();

        let expected = vec![
            FermionOperatorGroupTermView {
                coeff: Complex64::new(1.0, 0.0),
                actions: &[true, false],
                modes: &[0, 1],
                group: 0,
            },
            FermionOperatorGroupTermView {
                coeff: Complex64::new(1.0, 0.0),
                actions: &[true, false],
                modes: &[1, 0],
                group: 0,
            },
            FermionOperatorGroupTermView {
                coeff: Complex64::new(2.0, 0.0),
                actions: &[true, true, false, false],
                modes: &[0, 0, 1, 1],
                group: 1,
            },
        ];

        assert_eq!(terms, expected);
    }

    #[test]
    fn test_iter_from_terms_round_trip() {
        let op = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            actions: vec![true, false, true, true, false, false],
            modes: vec![0, 1, 0, 0, 1, 1],
            boundaries: vec![0, 2, 6],
            groups: None,
        };

        let round_trip = FermionOperator::from_terms(op.iter());

        assert_eq!(round_trip, op);
    }

    #[test]
    fn test_iter_with_groups_from_terms_with_groups_round_trip() {
        let op = FermionOperator {
            coeffs: vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
            ],
            actions: vec![true, false, true, false, true, true, false, false],
            modes: vec![0, 1, 1, 0, 0, 0, 1, 1],
            boundaries: vec![0, 2, 4, 8],
            groups: Some(vec![0, 0, 1]),
        };

        let round_trip = FermionOperator::from_terms_with_groups(op.iter_with_groups());

        assert_eq!(round_trip, op);
    }

    #[test]
    fn fci_matvec_spinless_delegates_to_kernel() {
        // a†_0 a_1 + a†_1 a_0 + 0.5 n_2 on norb=4 spinless orbitals, nocc=2.
        let op = FermionOperator {
            coeffs: vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.5, 0.0),
            ],
            actions: vec![true, false, true, false, true, false],
            modes: vec![0, 1, 1, 0, 2, 2],
            boundaries: vec![0, 2, 4, 6],
            groups: None,
        };
        let norb = 4u32;
        let nocc = 2u32;
        let dim = crate::linalg::fci::BinomialTable::new(norb).num_strings(norb, nocc);
        let vec: Vec<Complex64> = (0..dim)
            .map(|i| Complex64::new(i as f64 + 1.0, 0.0))
            .collect();

        // The method must equal a direct kernel call fed from the same term views.
        let expected = crate::linalg::fci::spinless_matvec(
            norb,
            nocc,
            op.iter().map(|t| (t.coeff, t.actions, t.modes)),
            &vec,
        )
        .unwrap();
        let got = op.fci_matvec_spinless(norb, nocc, &vec).unwrap();
        assert_eq!(got, expected);
        assert_eq!(got.len(), dim);
    }

    #[test]
    fn fci_matvec_spinful_delegates_to_kernel() {
        // Cross-spin density-density a†_0 a_0 a†_{norb} a_{norb} plus an alpha hop, norb=3.
        let norb = 3u32;
        let op = FermionOperator {
            coeffs: vec![Complex64::new(1.1, 0.0), Complex64::new(0.7, 0.0)],
            actions: vec![true, false, true, false, true, false],
            modes: vec![0, 0, norb, norb, 0, 1],
            boundaries: vec![0, 4, 6],
            groups: None,
        };
        let (n_a, n_b) = (2u32, 1u32);
        let table = crate::linalg::fci::BinomialTable::new(norb);
        let dim = table.num_strings(norb, n_a) * table.num_strings(norb, n_b);
        let vec: Vec<Complex64> = (0..dim)
            .map(|i| Complex64::new(i as f64 + 0.5, i as f64 * 0.2))
            .collect();

        let expected = crate::linalg::fci::spinful_matvec(
            norb,
            n_a,
            n_b,
            op.iter().map(|t| (t.coeff, t.actions, t.modes)),
            &vec,
        )
        .unwrap();
        let got = op.fci_matvec_spinful(norb, n_a, n_b, &vec).unwrap();
        assert_eq!(got, expected);
        assert_eq!(got.len(), dim);
    }
}
