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

use crate::operators::{CoherenceError, OperatorMacro, OperatorTrait, ScaledTerm, TermSortKey};
use num_complex::{Complex64, ComplexFloat};
use std::collections::{HashMap, HashSet};
use std::iter::zip;
use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign,
};

pub type EdgeAction<'a> = (&'a u32, &'a u32);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EdgeVertexOperatorTermView<'a> {
    pub coeff: Complex64,
    pub left_indices: &'a [u32],
    pub right_indices: &'a [u32],
}

impl EdgeVertexOperatorTermView<'_> {
    pub fn iter(&'_ self) -> impl ExactSizeIterator<Item = EdgeAction<'_>> + '_ {
        zip(self.left_indices, self.right_indices)
    }

    pub fn to_vec(&'_ self) -> Vec<EdgeAction<'_>> {
        zip(self.left_indices, self.right_indices).collect()
    }

    pub fn into_vec(&'_ self) -> Vec<(u32, u32)> {
        zip(
            self.left_indices.iter().copied(),
            self.right_indices.iter().copied(),
        )
        .collect()
    }
}

impl TermSortKey for EdgeVertexOperatorTermView<'_> {
    fn sort_key(&self) -> impl Ord {
        // Compare the operator string position-by-position, each factor being a (left, right) vertex
        // pair. This matches `into_vec` (and hence the sorted display order), so the canonical order
        // agrees with how terms are printed.
        self.into_vec()
    }
}

impl ScaledTerm for EdgeVertexOperatorTermView<'_> {
    fn scaled(mut self, factor: Complex64) -> Self {
        self.coeff *= factor;
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EdgeVertexOperatorGroupTermView<'a> {
    pub coeff: Complex64,
    pub left_indices: &'a [u32],
    pub right_indices: &'a [u32],
    pub group: u32,
}

impl EdgeVertexOperatorGroupTermView<'_> {
    pub fn iter(&'_ self) -> impl ExactSizeIterator<Item = EdgeAction<'_>> + '_ {
        zip(self.left_indices, self.right_indices)
    }

    pub fn to_vec(&'_ self) -> Vec<EdgeAction<'_>> {
        zip(self.left_indices, self.right_indices).collect()
    }

    pub fn into_vec(&'_ self) -> Vec<(u32, u32)> {
        zip(
            self.left_indices.iter().copied(),
            self.right_indices.iter().copied(),
        )
        .collect()
    }
}

impl TermSortKey for EdgeVertexOperatorGroupTermView<'_> {
    fn sort_key(&self) -> impl Ord {
        // Match the ungrouped `TermView` key exactly (ignoring `group`), so ordering a grouped
        // operator agrees with ordering the same terms ungrouped.
        self.into_vec()
    }
}

impl ScaledTerm for EdgeVertexOperatorGroupTermView<'_> {
    fn scaled(mut self, factor: Complex64) -> Self {
        self.coeff *= factor;
        self
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct EdgeVertexOperator {
    pub coeffs: Vec<Complex64>,
    pub left_indices: Vec<u32>,
    pub right_indices: Vec<u32>,
    pub boundaries: Vec<usize>,
    pub groups: Option<Vec<u32>>,
}

crate::impl_operator_macro!(EdgeVertexOperator);

impl EdgeVertexOperator {
    #[inline]
    pub fn left_indices(&self) -> &[u32] {
        &self.left_indices
    }

    #[inline]
    pub fn right_indices(&self) -> &[u32] {
        &self.right_indices
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
    pub fn _append_term(&mut self, coeff: Complex64, left_indices: &[u32], right_indices: &[u32]) {
        self.coeffs.push(coeff);
        self.left_indices.extend_from_slice(left_indices);
        self.right_indices.extend_from_slice(right_indices);
        self.boundaries.push(self.left_indices.len());
    }

    /// Splits this operator into new operators based on its [`groups`](Self::groups).
    ///
    /// If `group_indices` is `None`, every group is materialized once, in index order (`0` to
    /// [`num_groups`](Self::num_groups) `- 1`). If `group_indices` is `Some`, only the requested
    /// indices are materialized, in the given order; a duplicate index is materialized once
    /// internally and cloned once per occurrence in the output. This is significantly cheaper than
    /// requesting all indices when only a small number of groups out of a much larger total are
    /// needed, e.g. when subsampling groups for a randomized product formula, since terms
    /// belonging to a group that is not requested are skipped rather than appended anywhere.
    ///
    /// Returns `None` if `self.groups` is `None`, regardless of `group_indices`.
    pub fn split_out_groups(&self, group_indices: Option<&[u32]>) -> Option<Vec<Self>> {
        match group_indices {
            None => {
                let mut groups = vec![Self::zero(); self.num_groups()? as usize];
                for term in self.iter_with_groups() {
                    groups[term.group as usize]._append_term(
                        term.coeff,
                        term.left_indices,
                        term.right_indices,
                    );
                }
                Some(groups)
            }
            Some(group_indices) => {
                self.groups.as_ref()?;
                let mut wanted: HashMap<u32, Self> = group_indices
                    .iter()
                    .map(|&idx| (idx, Self::zero()))
                    .collect();
                for term in self.iter_with_groups() {
                    if let Some(acc) = wanted.get_mut(&term.group) {
                        acc._append_term(term.coeff, term.left_indices, term.right_indices);
                    }
                }
                Some(
                    group_indices
                        .iter()
                        .map(|idx| {
                            wanted
                                .get(idx)
                                .expect("every requested index was inserted above")
                                .clone()
                        })
                        .collect(),
                )
            }
        }
    }

    /// Returns an equivalent operator with normal-ordered terms.
    ///
    /// `ascending` fixes the orientation convention of the generalized edge operators: because
    /// `E_kj = -E_jk`, every edge operator has two representations, and this picks `j < k`
    /// (`ascending = true`) or `j > k` (`ascending = false`), absorbing the sign into the
    /// coefficient. Vertex operators `V_j = E_jj` are unaffected.
    ///
    /// `reduce` additionally contracts adjacent generators via the identities in
    /// [`_reduce_once`], which is what makes the result a genuine canonical form. Pass `false` to
    /// get the reorder-only behaviour.
    pub fn normal_ordered(&self, ascending: bool, reduce: bool) -> Self {
        let mut result = Self::zero();
        self.iter()
            .for_each(|term| result.__iadd__(&_normal_ordered_term(term, ascending, reduce)));
        result
    }
}

/// Rewrites `(left, right)` into the orientation selected by `ascending`, returning the rewritten
/// pair and the sign picked up from `E_kj = -E_jk`.
///
/// Vertex operators (`left == right`) are returned unchanged with a `+1` sign.
fn _canon_orientation(pair: (u32, u32), ascending: bool) -> ((u32, u32), bool) {
    let (left, right) = pair;
    if left == right {
        return (pair, false);
    }
    let wanted = if ascending {
        (left.min(right), left.max(right))
    } else {
        (left.max(right), left.min(right))
    };
    (wanted, wanted != pair)
}

/// Sorts `term` into normal order in place, returning whether the reordering picked up a sign.
///
/// This only ever *permutes* factors; contraction is handled separately by [`_reduce_once`].
fn _sort_term(term: &mut [(u32, u32)]) -> bool {
    let mut parity = false;
    for i in 1..term.len() {
        // shift the operator at index i to the left until it's in the correct location
        for j in (1..=i).rev() {
            let (right_1, right_2) = term[j];
            let (left_1, left_2) = term[j - 1];

            let left_is_vertex_op = left_1 == left_2;
            let right_is_vertex_op = right_1 == right_2;

            match (left_is_vertex_op, right_is_vertex_op) {
                (true, false) => {
                    // vertex op is left of edge op -> nothing to do
                }
                (true, true) => {
                    // two vertex ops; must check their indices
                    if left_1 > right_1 {
                        // -> this is a commuting operation
                        term.swap(j - 1, j);
                    }
                }
                (false, true) => {
                    // vertex op is right of edge op -> must _always_ swap
                    term.swap(j - 1, j);
                    // parity depends on whether the operator supports overlap
                    if left_1 == right_1 || left_2 == right_1 {
                        // -> anti-commuting operation when they do not!
                        parity = !parity;
                    }
                }
                (false, false) => {
                    // two edge ops
                    // whether we swap depends on the actual indices:
                    if left_1 > right_1 || (left_1 == right_1 && left_2 > right_2) {
                        term.swap(j - 1, j);
                        // Two edge operators anticommute iff they share *exactly one* mode:
                        // `{E_jk, E_kl} = 0`. Sharing *both* modes is the commuting case, since
                        // `E_jk` and `E_kj = -E_jk` are collinear and every operator commutes
                        // with itself. Disjoint supports commute too.
                        let overlap = HashSet::from([left_1, left_2])
                            .intersection(&HashSet::from([right_1, right_2]))
                            .count();
                        if overlap == 1 {
                            parity = !parity;
                        }
                    }
                }
            }
        }
    }
    parity
}

/// Applies a single contraction to the first reducible adjacent pair in `term`, if any.
///
/// Returns the scalar factor that must be multiplied into the term's coefficient, or `None` when
/// no rule applies. The rules, all of which strictly shorten the term:
///
/// - `V_j V_j = 1`, `E_jk E_jk = 1` (identical factors square to the identity),
/// - `E_jk E_kj = -1` (anti-parallel edge operators),
/// - `E_ab E_bc = -i E_ac` for distinct `a`, `b`, `c` (*fusion*: two edge operators sharing exactly
///   one mode collapse into a single one).
///
/// Together with `E_kj = -E_jk` the fusion rule generates every product of two edge operators that
/// share exactly one mode, so no further cases are needed.
fn _reduce_once(term: &mut Vec<(u32, u32)>, ascending: bool) -> Option<Complex64> {
    for j in 0..term.len().saturating_sub(1) {
        let (a, b) = term[j];
        let (c, d) = term[j + 1];

        // Identical support: contracts to a scalar and both factors disappear.
        if (a == c && b == d) || (a == d && b == c) {
            // `E_jk E_jk = +1` (and `V_j V_j = 1`), whereas `E_jk E_kj = -1`.
            let sign = if a == c && b == d { 1.0 } else { -1.0 };
            term.drain(j..=j + 1);
            return Some(Complex64::new(sign, 0.0));
        }

        // Vertex operators only contract against themselves, which the branch above covers.
        if a == b || c == d {
            continue;
        }

        // Exactly one shared mode: fuse into a single edge operator.
        let shared = HashSet::from([a, b])
            .intersection(&HashSet::from([c, d]))
            .count();
        if shared != 1 {
            continue;
        }

        // Rewrite both factors into the `E_xy E_yz` shape that the fusion rule is stated for,
        // tracking the sign each orientation flip contributes.
        let mut parity = false;
        let (mut a, mut b) = (a, b);
        let (mut c, mut d) = (c, d);
        if b != c {
            if a == c {
                (a, b) = (b, a);
                parity = !parity;
            } else if b == d {
                (c, d) = (d, c);
                parity = !parity;
            } else {
                // a == d: both need flipping
                (a, b) = (b, a);
                (c, d) = (d, c);
            }
        }
        debug_assert_eq!(
            b, c,
            "fusion requires the shared mode in the inner position"
        );

        // `E_ab E_bc = -i E_ac`, then re-canonicalize the surviving operator's orientation.
        let (fused, flipped) = _canon_orientation((a, d), ascending);
        if flipped {
            parity = !parity;
        }
        term[j] = fused;
        term.remove(j + 1);

        let factor = Complex64::new(0.0, -1.0);
        return Some(if parity { -factor } else { factor });
    }
    None
}

fn _normal_ordered_term(
    term_view: EdgeVertexOperatorTermView,
    ascending: bool,
    reduce: bool,
) -> EdgeVertexOperator {
    let mut coeffs = vec![];
    let mut left_indices = vec![];
    let mut right_indices = vec![];
    let mut boundaries = vec![0];

    let mut stack = vec![(term_view.into_vec(), term_view.coeff)];
    while let Some((mut term, coeff)) = stack.pop() {
        let mut coeff = coeff;

        if reduce {
            // Canonicalize orientations up front so the contraction rules can match on indices
            // without worrying about which of `E_jk`/`E_kj` happens to be stored.
            for factor in term.iter_mut() {
                let (canon, flipped) = _canon_orientation(*factor, ascending);
                *factor = canon;
                if flipped {
                    coeff = -coeff;
                }
            }
        }

        // Reorder, then contract, then reorder again: a contraction can bring two factors next to
        // each other that were not adjacent before (and fusion introduces a brand-new operator),
        // so this has to run to a fixed point. Every contraction removes at least one factor, so
        // the loop terminates after at most `term.len()` iterations.
        loop {
            if _sort_term(&mut term) {
                coeff = -coeff;
            }
            if !reduce {
                break;
            }
            match _reduce_once(&mut term, ascending) {
                Some(factor) => coeff *= factor,
                None => break,
            }
        }

        coeffs.push(coeff);
        term.iter().for_each(|&(a, i)| {
            left_indices.push(a);
            right_indices.push(i);
        });
        boundaries.push(right_indices.len())
    }
    EdgeVertexOperator {
        coeffs,
        left_indices,
        right_indices,
        boundaries,
        groups: None,
    }
}

fn _compose(
    a: &EdgeVertexOperator,
    b: &EdgeVertexOperator,
) -> (Vec<Complex64>, Vec<u32>, Vec<u32>, Vec<usize>) {
    // The output size is known exactly: one term per (left, right) pair, each holding the factors
    // of both inputs. Reserving up front turns the inner loop's repeated `extend_from_slice` into
    // plain memcpy without the reallocation-and-copy rounds an unreserved vector would incur.
    let num_terms = a.coeffs.len() * b.coeffs.len();
    let num_factors = a.left_indices.len() * b.coeffs.len() + b.left_indices.len() * a.coeffs.len();
    let mut coeffs = Vec::with_capacity(num_terms);
    let mut left_indices = Vec::with_capacity(num_factors);
    let mut right_indices = Vec::with_capacity(num_factors);
    let mut boundaries = Vec::with_capacity(num_terms + 1);
    boundaries.push(0);

    for left in a.iter() {
        for right in b.iter() {
            coeffs.push(left.coeff * right.coeff);
            left_indices.extend_from_slice(right.left_indices);
            left_indices.extend_from_slice(left.left_indices);
            right_indices.extend_from_slice(right.right_indices);
            right_indices.extend_from_slice(left.right_indices);
            boundaries.push(right_indices.len());
        }
    }
    (coeffs, left_indices, right_indices, boundaries)
}

impl OperatorTrait for EdgeVertexOperator {
    type TermView<'a> = EdgeVertexOperatorTermView<'a>;
    type GroupTermView<'a> = EdgeVertexOperatorGroupTermView<'a>;

    fn zero() -> Self {
        Self {
            coeffs: vec![],
            left_indices: vec![],
            right_indices: vec![],
            boundaries: vec![0],
            groups: None,
        }
    }

    fn one() -> Self {
        Self {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            left_indices: vec![],
            right_indices: vec![],
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

    /// Exact: [`normal_ordered`](Self::normal_ordered) with `reduce` contracts every reducible pair
    /// of adjacent generators, *including* fusing two edge operators that share one index via
    /// `E_ab E_bc = -i E_ac`, so a term that only cancels against its adjoint after such a
    /// contraction is still recognized as zero.
    fn is_hermitian(&self, atol: f64) -> bool {
        let mut diff = self.__sub__(&self.adjoint()).normal_ordered(true, true);
        diff.ichop(atol);
        diff.equiv(&Self::zero(), atol)
    }

    fn simplify(&self, atol: f64) -> Self {
        let mut terms = HashMap::new();
        for term in self.iter() {
            terms
                .entry((term.left_indices, term.right_indices))
                .and_modify(|c| *c += term.coeff)
                .or_insert(term.coeff);
        }
        let mut out = Self::zero();
        terms
            .iter()
            .filter(|((_, _), coeff)| coeff.abs() > atol)
            .for_each(|((left_indices, right_indices), coeff)| {
                out._append_term(*coeff, left_indices, right_indices)
            });
        out
    }

    fn adjoint(&self) -> Self {
        let mut coeffs = Vec::with_capacity(self.coeffs.len());
        let mut left_indices = Vec::with_capacity(self.left_indices.len());
        let mut right_indices = Vec::with_capacity(self.right_indices.len());

        // Besides conjugating the coefficients, the generators within each term must be reversed:
        // `(AB)† = B†A†` and the edge/vertex generators anticommute when they share exactly one
        // index, so `BA != AB` in general. Both index arrays are reversed over the same span, which
        // keeps each `(left, right)` pair intact while reversing the order of the pairs.
        self.iter().for_each(|term| {
            coeffs.push(term.coeff.conj());
            left_indices.extend(term.left_indices.iter().rev());
            right_indices.extend(term.right_indices.iter().rev());
        });

        Self {
            coeffs,
            left_indices,
            right_indices,
            boundaries: self.boundaries.to_vec(),
            groups: self.groups.clone(),
        }
    }

    fn __iadd__(&mut self, other: &Self) {
        self.coeffs.extend_from_slice(&other.coeffs);
        self.left_indices.extend_from_slice(&other.left_indices);
        self.right_indices.extend_from_slice(&other.right_indices);
        let offset = self.boundaries[self.boundaries.len() - 1];
        self.boundaries
            .extend(other.boundaries[1..].iter().map(|b| b + offset));
        self.groups = None;
    }

    fn __imul__(&mut self, other: Complex64) {
        self.coeffs.iter_mut().for_each(|c| *c *= other);
    }

    fn __isub__(&mut self, other: &Self) {
        self.coeffs.extend(other.coeffs.iter().map(|c| -c));
        self.left_indices.extend_from_slice(&other.left_indices);
        self.right_indices.extend_from_slice(&other.right_indices);
        let offset = self.boundaries[self.boundaries.len() - 1];
        self.boundaries
            .extend(other.boundaries[1..].iter().map(|b| b + offset));
        self.groups = None;
    }

    fn composed(&self, other: &Self) -> Self {
        let (coeffs, left_indices, right_indices, boundaries) = _compose(self, other);
        Self {
            coeffs,
            left_indices,
            right_indices,
            boundaries,
            groups: None,
        }
    }

    fn matmul(&self, other: &Self) -> Self {
        let (coeffs, left_indices, right_indices, boundaries) = _compose(other, self);
        Self {
            coeffs,
            left_indices,
            right_indices,
            boundaries,
            groups: None,
        }
    }

    fn __iand__(&mut self, other: &Self) {
        *self = self.composed(other);
    }

    fn __imatmul__(&mut self, other: &Self) {
        *self = self.matmul(other);
    }

    fn ichop(&mut self, atol: f64) {
        let mut coeffs = vec![];
        let mut left_indices = vec![];
        let mut right_indices = vec![];
        let mut boundaries = vec![0];

        self.iter()
            .filter(|term| term.coeff.abs() > atol)
            .for_each(|term| {
                coeffs.push(term.coeff);
                left_indices.extend_from_slice(term.left_indices);
                right_indices.extend_from_slice(term.right_indices);
                boundaries.push(right_indices.len());
            });

        self.coeffs = coeffs;
        self.left_indices = left_indices;
        self.right_indices = right_indices;
        self.boundaries = boundaries;
        self.groups = None;
    }

    fn iter(&self) -> impl ExactSizeIterator<Item = Self::TermView<'_>> {
        self.coeffs.iter().enumerate().map(|(i, coeff)| {
            let start = self.boundaries[i];
            let end = self.boundaries[i + 1];
            EdgeVertexOperatorTermView {
                coeff: *coeff,
                left_indices: &self.left_indices[start..end],
                right_indices: &self.right_indices[start..end],
            }
        })
    }

    #[inline]
    fn coeffs(&self) -> &[Complex64] {
        &self.coeffs
    }

    fn groups(&self) -> Option<&[u32]> {
        self.groups.as_deref()
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
                    EdgeVertexOperatorGroupTermView {
                        coeff: *coeff,
                        left_indices: &self.left_indices[start..end],
                        right_indices: &self.right_indices[start..end],
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
            out._append_term(term.coeff, term.left_indices, term.right_indices);
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
            out._append_term(term.coeff, term.left_indices, term.right_indices);
            groups.push(term.group);
        }
        out.groups = Some(groups);
        out
    }

    fn get_support(&self) -> HashSet<u32> {
        // Both index buffers feed a single set, rather than each building its own set to be unioned
        // afterwards: the union of two sets of mode indices is the set of all of them.
        self.left_indices
            .iter()
            .chain(&self.right_indices)
            .copied()
            .collect()
    }

    fn relabel_modes(&self, permutation: Vec<u32>) -> Result<Self, CoherenceError> {
        if permutation.iter().collect::<HashSet<_>>().len() != permutation.len() {
            return Err(CoherenceError::DuplicateIndices);
        }
        let relabel = |indices: &[u32]| -> Result<Vec<u32>, CoherenceError> {
            indices
                .iter()
                .map(|&idx| {
                    permutation
                        .get(idx as usize)
                        .copied()
                        .ok_or(CoherenceError::IndexMapTooSmall)
                })
                .collect()
        };
        // Both index buffers are relabelled before anything is copied, so that the error path does
        // not first clone the entire operator only to discard it. The left buffer is still mapped
        // first, keeping the error it reports ahead of the right buffer's.
        let left_indices = relabel(&self.left_indices)?;
        let right_indices = relabel(&self.right_indices)?;
        Ok(Self {
            coeffs: self.coeffs.clone(),
            left_indices,
            right_indices,
            boundaries: self.boundaries.clone(),
            // Relabelling permutes mode indices without reordering, splitting or merging terms, so
            // any grouping of those terms carries over unchanged. This is the one operation that
            // preserves `groups` rather than dropping it.
            groups: self.groups.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero() {
        let zero = EdgeVertexOperator::zero();
        assert_eq!(
            zero,
            EdgeVertexOperator {
                coeffs: vec![],
                left_indices: vec![],
                right_indices: vec![],
                boundaries: vec![0],
                groups: None,
            }
        );
    }

    #[test]
    fn test_one() {
        let one = EdgeVertexOperator::one();
        assert_eq!(
            one,
            EdgeVertexOperator {
                coeffs: vec![Complex64::new(1.0, 0.0)],
                left_indices: vec![],
                right_indices: vec![],
                boundaries: vec![0, 0],
                groups: None,
            }
        );
    }

    #[test]
    fn test_add() {
        let one = EdgeVertexOperator::one();
        let two = EdgeVertexOperator {
            coeffs: vec![Complex64::new(2.0, 0.0)],
            left_indices: vec![],
            right_indices: vec![],
            boundaries: vec![0, 0],
            groups: None,
        };
        let three = one + two;
        assert_eq!(
            three,
            EdgeVertexOperator {
                coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
                left_indices: vec![],
                right_indices: vec![],
                boundaries: vec![0, 0, 0],
                groups: None,
            }
        );
    }

    #[test]
    fn test_add_assign() {
        let mut op = EdgeVertexOperator::one();
        let two = EdgeVertexOperator {
            coeffs: vec![Complex64::new(2.0, 0.0)],
            left_indices: vec![],
            right_indices: vec![],
            boundaries: vec![0, 0],
            groups: None,
        };
        op += two;
        assert_eq!(
            op,
            EdgeVertexOperator {
                coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
                left_indices: vec![],
                right_indices: vec![],
                boundaries: vec![0, 0, 0],
                groups: None,
            }
        );
    }

    #[test]
    fn test_sub() {
        let one = EdgeVertexOperator::one();
        let two = EdgeVertexOperator {
            coeffs: vec![Complex64::new(2.0, 0.0)],
            left_indices: vec![],
            right_indices: vec![],
            boundaries: vec![0, 0],
            groups: None,
        };
        let new_one = two - one;
        assert_eq!(
            new_one,
            EdgeVertexOperator {
                coeffs: vec![Complex64::new(2.0, 0.0), Complex64::new(-1.0, 0.0)],
                left_indices: vec![],
                right_indices: vec![],
                boundaries: vec![0, 0, 0],
                groups: None,
            }
        );
    }

    #[test]
    fn test_sub_assign() {
        let mut op = EdgeVertexOperator::one();
        let two = EdgeVertexOperator {
            coeffs: vec![Complex64::new(2.0, 0.0)],
            left_indices: vec![],
            right_indices: vec![],
            boundaries: vec![0, 0],
            groups: None,
        };
        op -= two;
        assert_eq!(
            op,
            EdgeVertexOperator {
                coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(-2.0, 0.0)],
                left_indices: vec![],
                right_indices: vec![],
                boundaries: vec![0, 0, 0],
                groups: None,
            }
        );
    }

    #[test]
    fn test_mul() {
        let one = EdgeVertexOperator::one();
        let three = one * Complex64::new(3.0, 0.0);
        assert_eq!(
            three,
            EdgeVertexOperator {
                coeffs: vec![Complex64::new(3.0, 0.0)],
                left_indices: vec![],
                right_indices: vec![],
                boundaries: vec![0, 0],
                groups: None,
            }
        );
    }

    #[test]
    fn test_rmul() {
        let one = EdgeVertexOperator::one();
        let three = Complex64::new(3.0, 0.0) * one;
        assert_eq!(
            three,
            EdgeVertexOperator {
                coeffs: vec![Complex64::new(3.0, 0.0)],
                left_indices: vec![],
                right_indices: vec![],
                boundaries: vec![0, 0],
                groups: None,
            }
        );
    }

    #[test]
    fn test_mul_assign() {
        let mut op = EdgeVertexOperator::one();
        op *= Complex64::new(3.0, 0.0);
        assert_eq!(
            op,
            EdgeVertexOperator {
                coeffs: vec![Complex64::new(3.0, 0.0)],
                left_indices: vec![],
                right_indices: vec![],
                boundaries: vec![0, 0],
                groups: None,
            }
        );
    }

    #[test]
    fn test_div() {
        let three = EdgeVertexOperator {
            coeffs: vec![Complex64::new(3.0, 0.0)],
            left_indices: vec![],
            right_indices: vec![],
            boundaries: vec![0, 0],
            groups: None,
        };
        let one_half = three / Complex64::new(2.0, 0.0);
        assert_eq!(
            one_half,
            EdgeVertexOperator {
                coeffs: vec![Complex64::new(1.5, 0.0)],
                left_indices: vec![],
                right_indices: vec![],
                boundaries: vec![0, 0],
                groups: None,
            }
        );
    }

    #[test]
    fn test_idiv() {
        let mut op = EdgeVertexOperator {
            coeffs: vec![Complex64::new(3.0, 0.0)],
            left_indices: vec![],
            right_indices: vec![],
            boundaries: vec![0, 0],
            groups: None,
        };
        op /= Complex64::new(2.0, 0.0);
        assert_eq!(
            op,
            EdgeVertexOperator {
                coeffs: vec![Complex64::new(1.5, 0.0)],
                left_indices: vec![],
                right_indices: vec![],
                boundaries: vec![0, 0],
                groups: None,
            }
        );
    }

    #[test]
    fn test_neg() {
        let one = EdgeVertexOperator::one();
        assert_eq!(
            -one,
            EdgeVertexOperator {
                coeffs: vec![Complex64::new(-1.0, 0.0)],
                left_indices: vec![],
                right_indices: vec![],
                boundaries: vec![0, 0],
                groups: None,
            }
        );
    }

    #[test]
    fn test_and() {
        let op1 = EdgeVertexOperator {
            coeffs: vec![Complex64::new(2.0, 0.0), Complex64::new(3.0, 0.0)],
            left_indices: vec![0],
            right_indices: vec![1],
            boundaries: vec![0, 0, 1],
            groups: None,
        };
        let op2 = EdgeVertexOperator {
            coeffs: vec![Complex64::new(1.5, 0.0), Complex64::new(4.0, 0.0)],
            left_indices: vec![2],
            right_indices: vec![3],
            boundaries: vec![0, 0, 1],
            groups: None,
        };
        let result = op1 & op2;
        assert_eq!(
            result,
            EdgeVertexOperator {
                coeffs: vec![
                    Complex64::new(3.0, 0.0),
                    Complex64::new(8.0, 0.0),
                    Complex64::new(4.5, 0.0),
                    Complex64::new(12.0, 0.0),
                ],
                left_indices: vec![2, 0, 2, 0],
                right_indices: vec![3, 1, 3, 1],
                boundaries: vec![0, 0, 1, 2, 4],
                groups: None,
            }
        );
    }

    #[test]
    fn test_and_assign() {
        let mut op1 = EdgeVertexOperator {
            coeffs: vec![Complex64::new(2.0, 0.0), Complex64::new(3.0, 0.0)],
            left_indices: vec![0],
            right_indices: vec![1],
            boundaries: vec![0, 0, 1],
            groups: None,
        };
        let op2 = EdgeVertexOperator {
            coeffs: vec![Complex64::new(1.5, 0.0), Complex64::new(4.0, 0.0)],
            left_indices: vec![2],
            right_indices: vec![3],
            boundaries: vec![0, 0, 1],
            groups: None,
        };
        op1 &= op2;
        assert_eq!(
            op1,
            EdgeVertexOperator {
                coeffs: vec![
                    Complex64::new(3.0, 0.0),
                    Complex64::new(8.0, 0.0),
                    Complex64::new(4.5, 0.0),
                    Complex64::new(12.0, 0.0),
                ],
                left_indices: vec![2, 0, 2, 0],
                right_indices: vec![3, 1, 3, 1],
                boundaries: vec![0, 0, 1, 2, 4],
                groups: None,
            }
        );
    }

    #[test]
    fn test_matmul() {
        let op1 = EdgeVertexOperator {
            coeffs: vec![Complex64::new(2.0, 0.0), Complex64::new(3.0, 0.0)],
            left_indices: vec![0],
            right_indices: vec![1],
            boundaries: vec![0, 0, 1],
            groups: None,
        };
        let op2 = EdgeVertexOperator {
            coeffs: vec![Complex64::new(1.5, 0.0), Complex64::new(4.0, 0.0)],
            left_indices: vec![2],
            right_indices: vec![3],
            boundaries: vec![0, 0, 1],
            groups: None,
        };
        let result = op2.__matmul__(&op1);
        assert_eq!(
            result,
            EdgeVertexOperator {
                coeffs: vec![
                    Complex64::new(3.0, 0.0),
                    Complex64::new(8.0, 0.0),
                    Complex64::new(4.5, 0.0),
                    Complex64::new(12.0, 0.0),
                ],
                left_indices: vec![2, 0, 2, 0],
                right_indices: vec![3, 1, 3, 1],
                boundaries: vec![0, 0, 1, 2, 4],
                groups: None,
            }
        );
    }

    #[test]
    fn test_matmul_assign() {
        let op1 = EdgeVertexOperator {
            coeffs: vec![Complex64::new(2.0, 0.0), Complex64::new(3.0, 0.0)],
            left_indices: vec![0],
            right_indices: vec![1],
            boundaries: vec![0, 0, 1],
            groups: None,
        };
        let mut op2 = EdgeVertexOperator {
            coeffs: vec![Complex64::new(1.5, 0.0), Complex64::new(4.0, 0.0)],
            left_indices: vec![2],
            right_indices: vec![3],
            boundaries: vec![0, 0, 1],
            groups: None,
        };
        op2.__imatmul__(&op1);
        assert_eq!(
            op2,
            EdgeVertexOperator {
                coeffs: vec![
                    Complex64::new(3.0, 0.0),
                    Complex64::new(8.0, 0.0),
                    Complex64::new(4.5, 0.0),
                    Complex64::new(12.0, 0.0),
                ],
                left_indices: vec![2, 0, 2, 0],
                right_indices: vec![3, 1, 3, 1],
                boundaries: vec![0, 0, 1, 2, 4],
                groups: None,
            }
        );
    }

    #[test]
    fn test_pow() {
        let op = EdgeVertexOperator {
            coeffs: vec![Complex64::new(2.0, 0.0)],
            left_indices: vec![0],
            right_indices: vec![1],
            boundaries: vec![0, 1],
            groups: None,
        };
        // exponent=0
        let one = EdgeVertexOperator::one();
        assert_eq!(op.__pow__(0), one);

        // exponent=1
        assert_eq!(op.__pow__(1), op);

        // exponent=2
        let squared = op.__pow__(2);
        assert_eq!(
            squared,
            EdgeVertexOperator {
                coeffs: vec![Complex64::new(4.0, 0.0)],
                left_indices: vec![0, 0],
                right_indices: vec![1, 1],
                boundaries: vec![0, 2],
                groups: None,
            }
        );
    }

    #[test]
    fn test_ichop() {
        let mut op = EdgeVertexOperator {
            coeffs: vec![
                Complex64::new(1e-4, 0.0),
                Complex64::new(1e-6, 0.0),
                Complex64::new(1e-8, 0.0),
            ],
            left_indices: vec![0, 1],
            right_indices: vec![1, 2],
            boundaries: vec![0, 0, 1, 2],
            groups: None,
        };

        op.ichop(1e-7);

        let expected1 = EdgeVertexOperator {
            coeffs: vec![Complex64::new(1e-4, 0.0), Complex64::new(1e-6, 0.0)],
            left_indices: vec![0],
            right_indices: vec![1],
            boundaries: vec![0, 0, 1],
            groups: None,
        };

        assert_eq!(op, expected1);

        op.ichop(1e-5);

        let expected2 = EdgeVertexOperator {
            coeffs: vec![Complex64::new(1e-4, 0.0)],
            left_indices: vec![],
            right_indices: vec![],
            boundaries: vec![0, 0],
            groups: None,
        };

        assert_eq!(op, expected2);
    }

    #[test]
    fn test_ichop_preserves_complex_coeffs() {
        let mut op = EdgeVertexOperator {
            coeffs: vec![
                Complex64::new(1.0, 2.0),
                Complex64::new(0.0, -3.0),
                Complex64::new(1e-8, 1e-8),
            ],
            left_indices: vec![0, 1],
            right_indices: vec![1, 2],
            boundaries: vec![0, 0, 1, 2],
            groups: None,
        };

        op.ichop(1e-7);

        let expected = EdgeVertexOperator {
            coeffs: vec![Complex64::new(1.0, 2.0), Complex64::new(0.0, -3.0)],
            left_indices: vec![0],
            right_indices: vec![1],
            boundaries: vec![0, 0, 1],
            groups: None,
        };

        assert_eq!(op, expected);
    }

    #[test]
    fn test_adjoint() {
        // The second term is `V(0) E(0,1) E(1,2)`, i.e. multi-factor, so that the reversal of the
        // operator string is actually observable. A single-factor term would make it a no-op.
        let op1 = EdgeVertexOperator {
            coeffs: vec![Complex64::new(0.0, 2.0), Complex64::new(3.0, -4.0)],
            left_indices: vec![0, 0, 1],
            right_indices: vec![0, 1, 2],
            boundaries: vec![0, 0, 3],
            groups: None,
        };
        let adj = op1.adjoint();
        let expected = EdgeVertexOperator {
            coeffs: vec![Complex64::new(0.0, -2.0), Complex64::new(3.0, 4.0)],
            left_indices: vec![1, 0, 0],
            right_indices: vec![2, 1, 0],
            boundaries: vec![0, 0, 3],
            groups: None,
        };
        assert_eq!(adj, expected);
    }

    #[test]
    fn test_adjoint_is_antihomomorphism() {
        // `(A B)† == B† A†`. This identity only holds if `adjoint` reverses the operator string, so
        // it directly guards against dropping that reversal.
        let op1 = EdgeVertexOperator {
            coeffs: vec![Complex64::new(2.0, 1.0)],
            left_indices: vec![0, 0],
            right_indices: vec![0, 1],
            boundaries: vec![0, 2],
            groups: None,
        };
        let op2 = EdgeVertexOperator {
            coeffs: vec![Complex64::new(-1.0, 3.0)],
            left_indices: vec![1, 2],
            right_indices: vec![2, 2],
            boundaries: vec![0, 2],
            groups: None,
        };

        let lhs = op1.__matmul__(&op2).adjoint();
        let rhs = op2.adjoint().__matmul__(&op1.adjoint());
        assert!(lhs.equiv(&rhs, 1e-12));
    }

    #[test]
    fn test_is_hermitian_requires_reversal() {
        // `V(0) E(0,1)` is *not* Hermitian: the two generators share the index 0 and therefore
        // anticommute, so `(V(0) E(0,1))† = E(0,1) V(0) = -V(0) E(0,1)`.
        let op = EdgeVertexOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            left_indices: vec![0, 0],
            right_indices: vec![0, 1],
            boundaries: vec![0, 2],
            groups: None,
        };
        assert!(!op.is_hermitian(1e-10));

        // Symmetrizing it does give a Hermitian operator.
        let mut sym = op.clone();
        sym.__iadd__(&op.adjoint());
        assert!(sym.is_hermitian(1e-10));
    }

    #[test]
    fn test_equiv() {
        let zero = EdgeVertexOperator::zero();
        let op = Complex64::new(1e-8, 0.0) * EdgeVertexOperator::one();
        assert!(op.equiv(&zero, 1e-6));
        assert!(!op.equiv(&zero, 1e-10));
    }

    #[test]
    fn test_normal_ordered_noop() {
        let op = EdgeVertexOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            left_indices: vec![0, 1, 1, 0],
            right_indices: vec![0, 1, 1, 1],
            boundaries: vec![0, 4],
            groups: None,
        };

        assert_eq!(op.normal_ordered(true, false), op);
    }

    /// Builds a single-term operator from `(left, right)` pairs.
    fn single_term(actions: &[(u32, u32)], coeff: Complex64) -> EdgeVertexOperator {
        EdgeVertexOperator {
            coeffs: vec![coeff],
            left_indices: actions.iter().map(|&(l, _)| l).collect(),
            right_indices: actions.iter().map(|&(_, r)| r).collect(),
            boundaries: vec![0, actions.len()],
            groups: None,
        }
    }

    #[test]
    fn test_normal_ordered_reduce_scalars() {
        let one = Complex64::new(1.0, 0.0);

        // `V_0 V_0 = 1`
        let op = single_term(&[(0, 0), (0, 0)], one);
        assert_eq!(op.normal_ordered(true, true), single_term(&[], one));

        // `E_01 E_01 = 1`
        let op = single_term(&[(0, 1), (0, 1)], one);
        assert_eq!(op.normal_ordered(true, true), single_term(&[], one));

        // `E_01 E_10 = -1`
        let op = single_term(&[(0, 1), (1, 0)], one);
        assert_eq!(op.normal_ordered(true, true), single_term(&[], -one));
    }

    #[test]
    fn test_normal_ordered_reduce_fusion() {
        let one = Complex64::new(1.0, 0.0);
        let minus_i = Complex64::new(0.0, -1.0);

        // `E_01 E_12 = -i E_02`
        let op = single_term(&[(0, 1), (1, 2)], one);
        assert_eq!(
            op.normal_ordered(true, true),
            single_term(&[(0, 2)], minus_i)
        );

        // The same fusion has to be found when neither factor stores the shared mode in the inner
        // position, which requires applying `E_kj = -E_jk` first.
        let op = single_term(&[(1, 0), (2, 1)], one);
        assert_eq!(
            op.normal_ordered(true, true),
            single_term(&[(0, 2)], minus_i)
        );
    }

    #[test]
    fn test_normal_ordered_ascending() {
        let one = Complex64::new(1.0, 0.0);

        // `E_10 = -E_01`, so the non-selected orientation is rewritten and the sign absorbed.
        let op = single_term(&[(1, 0)], one);
        assert_eq!(op.normal_ordered(true, true), single_term(&[(0, 1)], -one));
        assert_eq!(op.normal_ordered(false, true), single_term(&[(1, 0)], one));

        // Vertex operators have no orientation freedom.
        let op = single_term(&[(1, 1)], one);
        assert_eq!(op.normal_ordered(true, true), op);
        assert_eq!(op.normal_ordered(false, true), op);
    }

    #[test]
    fn test_normal_ordered_gandon_rel1() {
        // Tests the 1. relation of Eq. (5) from arXiv:2512.11418v1: {E_{jk}, V_{k}} = 0
        let op = EdgeVertexOperator {
            coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
            left_indices: vec![0, 1, 0, 0],
            right_indices: vec![1, 1, 1, 0],
            boundaries: vec![0, 2, 4],
            groups: None,
        };

        let expected = EdgeVertexOperator {
            coeffs: vec![Complex64::new(-1.0, 0.0), Complex64::new(-1.0, 0.0)],
            left_indices: vec![1, 0, 0, 0],
            right_indices: vec![1, 1, 0, 1],
            boundaries: vec![0, 2, 4],
            groups: None,
        };

        assert_eq!(op.normal_ordered(true, false), expected);
    }

    #[test]
    fn test_normal_ordered_gandon_rel2() {
        // Tests the 2. relation of Eq. (5) from arXiv:2512.11418v1: {E_{jk}, E_{kl}} = 0
        //
        // Eq. (5) holds for `j != k != l`, so the two edge operators must share *exactly one*
        // mode for this relation to apply. Here that is `E_{1,2} E_{0,1}` (sharing only mode 1),
        // which anticommutes and hence picks up a sign when reordered. See
        // `test_normal_ordered_shared_pair_commutes` for the excluded share-both-modes case.
        let op = EdgeVertexOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            left_indices: vec![1, 0],
            right_indices: vec![2, 1],
            boundaries: vec![0, 2],
            groups: None,
        };

        let expected = EdgeVertexOperator {
            coeffs: vec![Complex64::new(-1.0, 0.0)],
            left_indices: vec![0, 1],
            right_indices: vec![1, 2],
            boundaries: vec![0, 2],
            groups: None,
        };

        assert_eq!(op.normal_ordered(true, false), expected);
    }

    /// Two edge operators spanning the *same* pair of modes commute, so reordering them must not
    /// introduce a sign.
    ///
    /// This is the case Eq. (5) of arXiv:2512.11418v1 does not cover: its relations are stated for
    /// `j != k != l != m`, so neither `{E_{jk}, E_{kl}} = 0` (exactly one shared mode) nor
    /// `[E_{jk}, E_{lm}] = 0` (disjoint modes) says anything about `E_{jk}` versus `E_{jk}` or
    /// `E_{kj}`. Since `E_{kj} = -E_{jk}`, those are collinear and commute.
    #[test]
    fn test_normal_ordered_shared_pair_commutes() {
        // E_{1,0} E_{0,1} -> E_{0,1} E_{1,0} with the coefficient *unchanged*.
        let op = EdgeVertexOperator {
            coeffs: vec![Complex64::new(1.0, 2.0)],
            left_indices: vec![1, 0],
            right_indices: vec![0, 1],
            boundaries: vec![0, 2],
            groups: None,
        };

        let expected = EdgeVertexOperator {
            coeffs: vec![Complex64::new(1.0, 2.0)],
            left_indices: vec![0, 1],
            right_indices: vec![1, 0],
            boundaries: vec![0, 2],
            groups: None,
        };

        assert_eq!(op.normal_ordered(true, false), expected);

        // Same for two *identical* edge operators, where the sort is a no-op but the parity rule is
        // still consulted.
        let op = EdgeVertexOperator {
            coeffs: vec![Complex64::new(1.0, 2.0)],
            left_indices: vec![1, 1],
            right_indices: vec![0, 0],
            boundaries: vec![0, 2],
            groups: None,
        };

        assert_eq!(op.normal_ordered(true, false), op);
    }

    #[test]
    fn test_normal_ordered_gandon_rel3() {
        // Tests the 3. relation of Eq. (5) from arXiv:2512.11418v1: [V_{k}, V_{l}] = 0
        let op = EdgeVertexOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            left_indices: vec![1, 0],
            right_indices: vec![1, 0],
            boundaries: vec![0, 2],
            groups: None,
        };

        let expected = EdgeVertexOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            left_indices: vec![0, 1],
            right_indices: vec![0, 1],
            boundaries: vec![0, 2],
            groups: None,
        };

        assert_eq!(op.normal_ordered(true, false), expected);
    }

    #[test]
    fn test_normal_ordered_gandon_rel4() {
        // Tests the 4. relation of Eq. (5) from arXiv:2512.11418v1: [E_{jk}, V_{l}] = 0
        let op = EdgeVertexOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            left_indices: vec![0, 2],
            right_indices: vec![1, 2],
            boundaries: vec![0, 2],
            groups: None,
        };

        let expected = EdgeVertexOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            left_indices: vec![2, 0],
            right_indices: vec![2, 1],
            boundaries: vec![0, 2],
            groups: None,
        };

        assert_eq!(op.normal_ordered(true, false), expected);
    }

    #[test]
    fn test_normal_ordered_gandon_rel5() {
        // Tests the 5. relation of Eq. (5) from arXiv:2512.11418v1: [E_{jk}, E_{lm}] = 0
        let op = EdgeVertexOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            left_indices: vec![2, 0],
            right_indices: vec![3, 1],
            boundaries: vec![0, 2],
            groups: None,
        };

        let expected = EdgeVertexOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            left_indices: vec![0, 2],
            right_indices: vec![1, 3],
            boundaries: vec![0, 2],
            groups: None,
        };

        assert_eq!(op.normal_ordered(true, false), expected);
    }

    /// Exercises the `atol` boundary of `is_hermitian`.
    ///
    /// Note that these are single-factor terms, so this test is insensitive to whether `adjoint`
    /// reverses the operator string; see `test_is_hermitian_requires_reversal` for that.
    #[test]
    fn test_is_hermitian() {
        let op = EdgeVertexOperator {
            coeffs: vec![Complex64::new(0.0, 1.00001), Complex64::new(0.0, -1.0)],
            left_indices: vec![0, 0],
            right_indices: vec![1, 1],
            boundaries: vec![0, 1, 2],
            groups: None,
        };
        assert!(op.is_hermitian(1e-4));
        assert!(!op.is_hermitian(1e-6));
    }

    #[test]
    fn test_get_support() {
        let op = EdgeVertexOperator {
            coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
            left_indices: vec![0, 3, 7],
            right_indices: vec![1, 4, 7],
            boundaries: vec![0, 2, 3],
            groups: None,
        };

        assert_eq!(op.get_support(), HashSet::from([0, 1, 3, 4, 7]));
    }

    #[test]
    fn test_relabel_modes() {
        let op = EdgeVertexOperator {
            coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
            left_indices: vec![0, 2, 1, 3],
            right_indices: vec![1, 3, 2, 0],
            boundaries: vec![0, 2, 6],
            groups: None,
        };

        let permutation = vec![4, 2, 5, 3];

        let relabeled = op.relabel_modes(permutation).ok();

        let expected = EdgeVertexOperator {
            coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
            left_indices: vec![4, 5, 2, 3],
            right_indices: vec![2, 3, 5, 4],
            boundaries: vec![0, 2, 6],
            groups: None,
        };

        assert_eq!(relabeled, Some(expected));
    }

    #[test]
    fn test_relabel_modes_duplicate_err() {
        let op = EdgeVertexOperator {
            coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
            left_indices: vec![0, 2, 1, 3],
            right_indices: vec![1, 3, 2, 0],
            boundaries: vec![0, 2, 6],
            groups: None,
        };

        let permutation = vec![4, 4, 2, 3];

        let relabeled = op.relabel_modes(permutation);

        assert!(matches!(relabeled, Err(CoherenceError::DuplicateIndices)));
    }

    #[test]
    fn test_relabel_modes_index_too_small_err() {
        let op = EdgeVertexOperator {
            coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
            left_indices: vec![0, 2, 1, 3],
            right_indices: vec![1, 3, 2, 0],
            boundaries: vec![0, 2, 6],
            groups: None,
        };

        let permutation = vec![4, 2, 5];

        let relabeled = op.relabel_modes(permutation);

        assert!(matches!(relabeled, Err(CoherenceError::IndexMapTooSmall)));
    }

    #[test]
    fn test_has_and_num_groups() {
        let mut zero = EdgeVertexOperator::zero();

        assert!(!zero.has_groups());
        assert!(zero.num_groups().is_none());

        zero.groups = Some(vec![]);

        // an operator may track groups while holding no terms at all
        assert!(zero.has_groups());
        assert_eq!(zero.num_groups(), Some(0));

        let mut one = EdgeVertexOperator::one();
        one.groups = Some(vec![0]);

        assert!(one.has_groups());
        assert_eq!(one.num_groups(), Some(1));
    }

    #[test]
    fn test_split_out_groups() {
        let op = EdgeVertexOperator {
            coeffs: vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
            ],
            left_indices: vec![0, 2, 1, 3],
            right_indices: vec![1, 3, 0, 2],
            boundaries: vec![0, 1, 2, 4],
            groups: Some(vec![0, 0, 1]),
        };

        let expected = vec![
            EdgeVertexOperator {
                coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
                left_indices: vec![0, 2],
                right_indices: vec![1, 3],
                boundaries: vec![0, 1, 2],
                groups: None,
            },
            EdgeVertexOperator {
                coeffs: vec![Complex64::new(2.0, 0.0)],
                left_indices: vec![1, 3],
                right_indices: vec![0, 2],
                boundaries: vec![0, 2],
                groups: None,
            },
        ];

        let groups = op.split_out_groups(None);
        assert_eq!(groups, Some(expected.clone()));

        // reversed and non-exhaustive: asserts order-preservation and partial selection.
        let selected = op.split_out_groups(Some(&[1, 0]));
        assert_eq!(
            selected,
            Some(vec![expected[1].clone(), expected[0].clone()])
        );

        // a duplicate index must be returned once per occurrence, not deduplicated.
        let duplicated = op.split_out_groups(Some(&[0, 0]));
        assert_eq!(
            duplicated,
            Some(vec![expected[0].clone(), expected[0].clone()])
        );

        // an empty request returns an empty (not `None`) result, since `groups` is present.
        let empty = op.split_out_groups(Some(&[]));
        assert_eq!(empty, Some(vec![]));
    }

    #[test]
    fn test_split_out_groups_err() {
        let op = EdgeVertexOperator {
            coeffs: vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
            ],
            left_indices: vec![0, 2, 1, 3],
            right_indices: vec![1, 3, 0, 2],
            boundaries: vec![0, 1, 2, 4],
            groups: None,
        };

        assert!(op.split_out_groups(None).is_none());
        assert!(op.split_out_groups(Some(&[0])).is_none());
    }

    #[test]
    #[should_panic(expected = "This method can only be called when groups are present!")]
    fn test_iter_with_groups_panics_without_groups() {
        // The documented contract is that `iter_with_groups` panics on an operator that carries no
        // group indices (see `has_groups`). `zero()` produces exactly such an operator.
        let op = EdgeVertexOperator::zero();
        let _ = op.iter_with_groups();
    }

    #[test]
    fn test_iter_with_groups() {
        let op = EdgeVertexOperator {
            coeffs: vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
            ],
            left_indices: vec![0, 2, 1, 3],
            right_indices: vec![1, 3, 0, 2],
            boundaries: vec![0, 1, 2, 4],
            groups: Some(vec![0, 0, 1]),
        };

        let terms: Vec<EdgeVertexOperatorGroupTermView> = op.iter_with_groups().collect();

        let expected = vec![
            EdgeVertexOperatorGroupTermView {
                coeff: Complex64::new(1.0, 0.0),
                left_indices: &[0],
                right_indices: &[1],
                group: 0,
            },
            EdgeVertexOperatorGroupTermView {
                coeff: Complex64::new(1.0, 0.0),
                left_indices: &[2],
                right_indices: &[3],
                group: 0,
            },
            EdgeVertexOperatorGroupTermView {
                coeff: Complex64::new(2.0, 0.0),
                left_indices: &[1, 3],
                right_indices: &[0, 2],
                group: 1,
            },
        ];

        assert_eq!(terms, expected);
    }

    #[test]
    fn test_iter_from_terms_round_trip() {
        let op = EdgeVertexOperator {
            coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            left_indices: vec![0, 1, 3],
            right_indices: vec![1, 0, 2],
            boundaries: vec![0, 1, 3],
            groups: None,
        };

        let round_trip = EdgeVertexOperator::from_terms(op.iter());

        assert_eq!(round_trip, op);
    }

    #[test]
    fn test_iter_with_groups_from_terms_with_groups_round_trip() {
        let op = EdgeVertexOperator {
            coeffs: vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
            ],
            left_indices: vec![0, 2, 1, 3],
            right_indices: vec![1, 3, 0, 2],
            boundaries: vec![0, 1, 2, 4],
            groups: Some(vec![0, 0, 1]),
        };

        let round_trip = EdgeVertexOperator::from_terms_with_groups(op.iter_with_groups());

        assert_eq!(round_trip, op);
    }

    /// Two grouped, non-commuting operands for the allocation-free rewrites below.
    ///
    /// Grouped so that the tests can also assert that the out-of-place operations drop `groups`
    /// exactly where their in-place counterparts do, and non-commuting so that an operand swap
    /// cannot pass unnoticed.
    fn operand_pair() -> (EdgeVertexOperator, EdgeVertexOperator) {
        let op1 = EdgeVertexOperator {
            coeffs: vec![Complex64::new(2.0, -1.0), Complex64::new(3.0, 0.5)],
            left_indices: vec![0, 2],
            right_indices: vec![1, 3],
            boundaries: vec![0, 1, 2],
            groups: Some(vec![0, 1]),
        };
        let op2 = EdgeVertexOperator {
            coeffs: vec![Complex64::new(1.5, 2.0), Complex64::new(0.0, 4.0)],
            left_indices: vec![1, 3],
            right_indices: vec![2, 0],
            boundaries: vec![0, 1, 2],
            groups: Some(vec![0, 0]),
        };
        (op1, op2)
    }

    #[test]
    fn test_and_matches_clone_then_and_assign() {
        let (op1, op2) = operand_pair();

        let mut expected = op1.clone();
        expected.__iand__(&op2);

        assert_eq!(op1.__and__(&op2), expected);
        assert!(!op1.__and__(&op2).has_groups());
    }

    #[test]
    fn test_matmul_matches_clone_then_matmul_assign() {
        let (op1, op2) = operand_pair();

        let mut expected = op1.clone();
        expected.__imatmul__(&op2);

        assert_eq!(op1.__matmul__(&op2), expected);
        assert!(!op1.__matmul__(&op2).has_groups());
    }

    #[test]
    fn test_sub_matches_add_of_negation() {
        let (op1, op2) = operand_pair();

        // The formulation `__sub__` used before it was fused, spelled out here so that the fused
        // version is pinned to it. Complex coefficients matter: a sign error in the real part alone
        // would survive real-only operands.
        let mut expected = op1.clone();
        expected.__iadd__(&op2.__neg__());

        assert_eq!(op1.__sub__(&op2), expected);

        let mut in_place = op1.clone();
        in_place.__isub__(&op2);
        assert_eq!(in_place, expected);
    }

    /// The out-of-place operations must not write through to either operand.
    ///
    /// `composed`/`matmul` take `&self`, so this cannot regress without a signature change, but the
    /// in-place counterparts they now back (`*self = self.composed(other)`) make the property worth
    /// stating outright.
    #[test]
    fn test_out_of_place_operations_leave_operands_untouched() {
        let (op1, op2) = operand_pair();
        let (op1_before, op2_before) = (op1.clone(), op2.clone());

        let _ = op1.__and__(&op2);
        let _ = op1.__matmul__(&op2);
        let _ = op1.__sub__(&op2);
        let _ = op1.__add__(&op2);
        let _ = op1.__neg__();
        let _ = op1.__pow__(2);
        let _ = op1.adjoint();
        let _ = op1.relabel_modes(vec![3, 2, 1, 0]);

        assert_eq!(op1, op1_before);
        assert_eq!(op2, op2_before);
    }

    /// The support of both index buffers, where the two only partially overlap.
    ///
    /// A single set now collects both buffers instead of unioning a set per buffer; overlapping but
    /// unequal sides are what distinguishes that from taking only one side, or their intersection.
    #[test]
    fn test_get_support_overlapping_sides() {
        let op = EdgeVertexOperator {
            coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
            left_indices: vec![0, 2, 4],
            right_indices: vec![2, 4, 7],
            boundaries: vec![0, 2, 3],
            groups: None,
        };

        assert_eq!(op.get_support(), HashSet::from([0, 2, 4, 7]));
    }

    /// A right-side index outside the permutation must still be reported.
    ///
    /// Both buffers are relabelled through one closure now, so the right side needs its own case to
    /// show it is mapped at all rather than copied through.
    #[test]
    fn test_relabel_modes_index_too_small_err_from_right() {
        let op = EdgeVertexOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            left_indices: vec![0],
            right_indices: vec![3],
            boundaries: vec![0, 1],
            groups: None,
        };

        let relabeled = op.relabel_modes(vec![1, 0, 2]);

        assert!(matches!(relabeled, Err(CoherenceError::IndexMapTooSmall)));
    }

    #[test]
    fn test_relabel_modes_preserves_groups() {
        let op = EdgeVertexOperator {
            coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            left_indices: vec![0, 2],
            right_indices: vec![1, 3],
            boundaries: vec![0, 1, 2],
            groups: Some(vec![0, 1]),
        };

        let relabeled = op.relabel_modes(vec![3, 2, 1, 0]).unwrap();

        assert_eq!(
            relabeled,
            EdgeVertexOperator {
                coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
                left_indices: vec![3, 1],
                right_indices: vec![2, 0],
                boundaries: vec![0, 1, 2],
                groups: Some(vec![0, 1]),
            }
        );
    }
}
