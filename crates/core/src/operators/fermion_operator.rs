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

use crate::operators::{CoherenceError, OperatorMacro, OperatorTrait};
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

    pub fn _append_term(&mut self, coeff: Complex64, actions: &[bool], modes: &[u32]) {
        // WARNING: this does not handle `groups` by design!
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
        for term in self.iter() {
            let (create_count, destroy_count) =
                term.iter()
                    .fold((0, 0), |(create_acc, destroy_acc), (action, _)| {
                        if *action {
                            (create_acc + 1, destroy_acc)
                        } else {
                            (create_acc, destroy_acc + 1)
                        }
                    });
            if create_count != destroy_count {
                return false;
            }
        }
        true
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
}
