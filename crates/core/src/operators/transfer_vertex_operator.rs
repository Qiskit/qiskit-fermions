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
use std::collections::{HashMap, HashSet};
use std::iter::zip;
use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign,
};

pub type TransferAction<'a> = (&'a u32, &'a u32);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransferVertexOperatorTermView<'a> {
    pub coeff: Complex64,
    pub left_indices: &'a [u32],
    pub right_indices: &'a [u32],
}

impl TransferVertexOperatorTermView<'_> {
    pub fn iter(&'_ self) -> impl ExactSizeIterator<Item = TransferAction<'_>> + '_ {
        zip(self.left_indices, self.right_indices)
    }

    pub fn to_vec(&'_ self) -> Vec<TransferAction<'_>> {
        zip(self.left_indices, self.right_indices).collect()
    }

    pub fn into_vec(&'_ self) -> Vec<(u32, u32)> {
        zip(self.left_indices.to_vec(), self.right_indices.to_vec()).collect()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TransferVertexOperator {
    pub coeffs: Vec<Complex64>,
    pub left_indices: Vec<u32>,
    pub right_indices: Vec<u32>,
    pub boundaries: Vec<usize>,
    pub groups: Option<Vec<u32>>,
}

crate::impl_operator_macro!(TransferVertexOperator);

impl TransferVertexOperator {
    #[inline]
    pub fn coeffs(&self) -> &[Complex64] {
        &self.coeffs
    }

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

    fn _append_term(&mut self, coeff: Complex64, left_indices: &[u32], right_indices: &[u32]) {
        // WARNING: this does not handle `groups` by design!
        self.coeffs.push(coeff);
        self.left_indices.extend_from_slice(left_indices);
        self.right_indices.extend_from_slice(right_indices);
        self.boundaries.push(self.left_indices.len());
    }

    pub fn iter(
        &'_ self,
    ) -> impl ExactSizeIterator<Item = TransferVertexOperatorTermView<'_>> + '_ {
        self.coeffs.iter().enumerate().map(|(i, coeff)| {
            let start = self.boundaries[i];
            let end = self.boundaries[i + 1];
            TransferVertexOperatorTermView {
                coeff: *coeff,
                left_indices: &self.left_indices[start..end],
                right_indices: &self.right_indices[start..end],
            }
        })
    }

    pub fn num_groups(&self) -> Option<u32> {
        let self_groups = self.groups.as_ref()?;
        if self_groups.len() == 0 {
            Some(0)
        } else {
            Some(self_groups.iter().max().unwrap() + 1)
        }
    }

    pub fn split_out_groups(&self) -> Option<Vec<Self>> {
        let self_groups = self.groups.as_ref()?;
        let num_groups = self.num_groups()?;
        let mut groups = vec![Self::zero(); num_groups as usize];
        for (group_idx, term) in zip(self_groups.iter(), self.iter()) {
            groups[*group_idx as usize]._append_term(
                term.coeff,
                term.left_indices,
                term.right_indices,
            );
        }
        Some(groups)
    }

    pub fn normal_ordered(&self) -> Self {
        let mut result = Self::zero();
        self.iter()
            .for_each(|term| result.__iadd__(&_normal_ordered_term(term)));
        result
    }

    pub fn is_hermitian(&self, atol: f64) -> bool {
        let mut diff = (self.__sub__(&self.adjoint())).normal_ordered();
        diff.ichop(atol);
        diff.equiv(&Self::zero(), atol)
    }
}

fn _normal_ordered_term(term_view: TransferVertexOperatorTermView) -> TransferVertexOperator {
    let mut coeffs = vec![];
    let mut left_indices = vec![];
    let mut right_indices = vec![];
    let mut boundaries = vec![0];

    let mut stack = vec![(term_view.to_vec(), term_view.coeff)];
    while let Some((mut term, coeff)) = stack.pop() {
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
                        // vertex op is left of transfer op -> nothing to do
                    }
                    (true, true) => {
                        // two vertex ops; must check their indices
                        if left_1 > right_1 {
                            // -> this is a commuting operation
                            term.swap(j - 1, j);
                        }
                    }
                    (false, true) => {
                        // vertex op is right of transfer op -> must _always_ swap
                        term.swap(j - 1, j);
                        // parity depends on whether the operator supports overlap
                        if left_1 == right_1 || left_2 == right_1 {
                            // -> anti-commuting operation when they do not!
                            parity = !parity;
                        }
                    }
                    (false, false) => {
                        // two transfer ops
                        // whether we swap depends on the actual indices:
                        if left_1 > right_1 || (left_1 == right_1 && left_2 > right_2) {
                            term.swap(j - 1, j);
                            // swap will commute unless either sided index pairs equal
                            if left_1 == right_1 || left_2 == right_2 {
                                parity = !parity;
                            }
                        }
                    }
                }
            }
        }
        let signed_coeff = if parity { -coeff } else { coeff };
        coeffs.push(signed_coeff);
        term.iter().for_each(|&(&a, &i)| {
            left_indices.push(a);
            right_indices.push(i);
        });
        boundaries.push(right_indices.len())
    }
    TransferVertexOperator {
        coeffs,
        left_indices,
        right_indices,
        boundaries,
        groups: None,
    }
}

fn _compose(
    a: &TransferVertexOperator,
    b: &TransferVertexOperator,
) -> (Vec<Complex64>, Vec<u32>, Vec<u32>, Vec<usize>) {
    let mut coeffs = vec![];
    let mut left_indices = vec![];
    let mut right_indices = vec![];
    let mut boundaries = vec![0];

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

impl OperatorTrait for TransferVertexOperator {
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
        Self {
            coeffs: self.coeffs.iter().map(|c| c.conj()).collect(),
            left_indices: self.left_indices.clone(),
            right_indices: self.right_indices.clone(),
            boundaries: self.boundaries.clone(),
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

    fn __iand__(&mut self, other: &Self) {
        (
            self.coeffs,
            self.left_indices,
            self.right_indices,
            self.boundaries,
        ) = _compose(self, other);
        self.groups = None;
    }

    fn __imatmul__(&mut self, other: &Self) {
        (
            self.coeffs,
            self.left_indices,
            self.right_indices,
            self.boundaries,
        ) = _compose(other, self);
        self.groups = None;
    }

    fn ichop(&mut self, atol: f64) {
        let mut coeffs = vec![];
        let mut left_indices = vec![];
        let mut right_indices = vec![];
        let mut boundaries = vec![0];

        self.iter()
            .filter(|term| term.coeff.abs() > atol)
            .for_each(|term| {
                coeffs.push(term.coeff.conj());
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

    fn get_support(&self) -> HashSet<u32> {
        let support_left: HashSet<u32> = HashSet::from_iter(self.left_indices.clone());
        let support_right: HashSet<u32> = HashSet::from_iter(self.right_indices.clone());
        support_left.union(&support_right).copied().collect()
    }

    fn relabel_modes(&self, permutation: Vec<u32>) -> Result<Self, CoherenceError> {
        if permutation.iter().collect::<HashSet<_>>().len() != permutation.len() {
            return Err(CoherenceError::DuplicateIndices);
        }
        let mut out = self.clone();
        let new_left: Result<Vec<u32>, CoherenceError> = self
            .left_indices
            .iter()
            .map(|&idx| {
                permutation
                    .get(idx as usize)
                    .cloned()
                    .ok_or(CoherenceError::IndexMapTooSmall)
            })
            .collect();
        out.left_indices = new_left?;
        let new_right: Result<Vec<u32>, CoherenceError> = self
            .right_indices
            .iter()
            .map(|&idx| {
                permutation
                    .get(idx as usize)
                    .cloned()
                    .ok_or(CoherenceError::IndexMapTooSmall)
            })
            .collect();
        out.right_indices = new_right?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero() {
        let zero = TransferVertexOperator::zero();
        assert_eq!(
            zero,
            TransferVertexOperator {
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
        let one = TransferVertexOperator::one();
        assert_eq!(
            one,
            TransferVertexOperator {
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
        let one = TransferVertexOperator::one();
        let two = TransferVertexOperator {
            coeffs: vec![Complex64::new(2.0, 0.0)],
            left_indices: vec![],
            right_indices: vec![],
            boundaries: vec![0, 0],
            groups: None,
        };
        let three = one + two;
        assert_eq!(
            three,
            TransferVertexOperator {
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
        let mut op = TransferVertexOperator::one();
        let two = TransferVertexOperator {
            coeffs: vec![Complex64::new(2.0, 0.0)],
            left_indices: vec![],
            right_indices: vec![],
            boundaries: vec![0, 0],
            groups: None,
        };
        op += two;
        assert_eq!(
            op,
            TransferVertexOperator {
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
        let one = TransferVertexOperator::one();
        let two = TransferVertexOperator {
            coeffs: vec![Complex64::new(2.0, 0.0)],
            left_indices: vec![],
            right_indices: vec![],
            boundaries: vec![0, 0],
            groups: None,
        };
        let new_one = two - one;
        assert_eq!(
            new_one,
            TransferVertexOperator {
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
        let mut op = TransferVertexOperator::one();
        let two = TransferVertexOperator {
            coeffs: vec![Complex64::new(2.0, 0.0)],
            left_indices: vec![],
            right_indices: vec![],
            boundaries: vec![0, 0],
            groups: None,
        };
        op -= two;
        assert_eq!(
            op,
            TransferVertexOperator {
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
        let one = TransferVertexOperator::one();
        let three = one * Complex64::new(3.0, 0.0);
        assert_eq!(
            three,
            TransferVertexOperator {
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
        let one = TransferVertexOperator::one();
        let three = Complex64::new(3.0, 0.0) * one;
        assert_eq!(
            three,
            TransferVertexOperator {
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
        let mut op = TransferVertexOperator::one();
        op *= Complex64::new(3.0, 0.0);
        assert_eq!(
            op,
            TransferVertexOperator {
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
        let three = TransferVertexOperator {
            coeffs: vec![Complex64::new(3.0, 0.0)],
            left_indices: vec![],
            right_indices: vec![],
            boundaries: vec![0, 0],
            groups: None,
        };
        let one_half = three / Complex64::new(2.0, 0.0);
        assert_eq!(
            one_half,
            TransferVertexOperator {
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
        let mut op = TransferVertexOperator {
            coeffs: vec![Complex64::new(3.0, 0.0)],
            left_indices: vec![],
            right_indices: vec![],
            boundaries: vec![0, 0],
            groups: None,
        };
        op /= Complex64::new(2.0, 0.0);
        assert_eq!(
            op,
            TransferVertexOperator {
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
        let one = TransferVertexOperator::one();
        assert_eq!(
            -one,
            TransferVertexOperator {
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
        let op1 = TransferVertexOperator {
            coeffs: vec![Complex64::new(2.0, 0.0), Complex64::new(3.0, 0.0)],
            left_indices: vec![0],
            right_indices: vec![1],
            boundaries: vec![0, 0, 1],
            groups: None,
        };
        let op2 = TransferVertexOperator {
            coeffs: vec![Complex64::new(1.5, 0.0), Complex64::new(4.0, 0.0)],
            left_indices: vec![2],
            right_indices: vec![3],
            boundaries: vec![0, 0, 1],
            groups: None,
        };
        let result = op1 & op2;
        assert_eq!(
            result,
            TransferVertexOperator {
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
        let mut op1 = TransferVertexOperator {
            coeffs: vec![Complex64::new(2.0, 0.0), Complex64::new(3.0, 0.0)],
            left_indices: vec![0],
            right_indices: vec![1],
            boundaries: vec![0, 0, 1],
            groups: None,
        };
        let op2 = TransferVertexOperator {
            coeffs: vec![Complex64::new(1.5, 0.0), Complex64::new(4.0, 0.0)],
            left_indices: vec![2],
            right_indices: vec![3],
            boundaries: vec![0, 0, 1],
            groups: None,
        };
        op1 &= op2;
        assert_eq!(
            op1,
            TransferVertexOperator {
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
        let op1 = TransferVertexOperator {
            coeffs: vec![Complex64::new(2.0, 0.0), Complex64::new(3.0, 0.0)],
            left_indices: vec![0],
            right_indices: vec![1],
            boundaries: vec![0, 0, 1],
            groups: None,
        };
        let op2 = TransferVertexOperator {
            coeffs: vec![Complex64::new(1.5, 0.0), Complex64::new(4.0, 0.0)],
            left_indices: vec![2],
            right_indices: vec![3],
            boundaries: vec![0, 0, 1],
            groups: None,
        };
        let result = op2.__matmul__(&op1);
        assert_eq!(
            result,
            TransferVertexOperator {
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
        let op1 = TransferVertexOperator {
            coeffs: vec![Complex64::new(2.0, 0.0), Complex64::new(3.0, 0.0)],
            left_indices: vec![0],
            right_indices: vec![1],
            boundaries: vec![0, 0, 1],
            groups: None,
        };
        let mut op2 = TransferVertexOperator {
            coeffs: vec![Complex64::new(1.5, 0.0), Complex64::new(4.0, 0.0)],
            left_indices: vec![2],
            right_indices: vec![3],
            boundaries: vec![0, 0, 1],
            groups: None,
        };
        op2.__imatmul__(&op1);
        assert_eq!(
            op2,
            TransferVertexOperator {
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
        let op = TransferVertexOperator {
            coeffs: vec![Complex64::new(2.0, 0.0)],
            left_indices: vec![0],
            right_indices: vec![1],
            boundaries: vec![0, 1],
            groups: None,
        };
        // exponent=0
        let one = TransferVertexOperator::one();
        assert_eq!(op.__pow__(0), one);

        // exponent=1
        assert_eq!(op.__pow__(1), op);

        // exponent=2
        let squared = op.__pow__(2);
        assert_eq!(
            squared,
            TransferVertexOperator {
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
        let mut op = TransferVertexOperator {
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

        let expected1 = TransferVertexOperator {
            coeffs: vec![Complex64::new(1e-4, 0.0), Complex64::new(1e-6, 0.0)],
            left_indices: vec![0],
            right_indices: vec![1],
            boundaries: vec![0, 0, 1],
            groups: None,
        };

        assert_eq!(op, expected1);

        op.ichop(1e-5);

        let expected2 = TransferVertexOperator {
            coeffs: vec![Complex64::new(1e-4, 0.0)],
            left_indices: vec![],
            right_indices: vec![],
            boundaries: vec![0, 0],
            groups: None,
        };

        assert_eq!(op, expected2);
    }

    #[test]
    fn test_adjoint() {
        let op1 = TransferVertexOperator {
            coeffs: vec![Complex64::new(0.0, 2.0), Complex64::new(3.0, 0.0)],
            left_indices: vec![0],
            right_indices: vec![1],
            boundaries: vec![0, 0, 1],
            groups: None,
        };
        let adj = op1.adjoint();
        let expected = TransferVertexOperator {
            coeffs: vec![Complex64::new(0.0, -2.0), Complex64::new(3.0, 0.0)],
            left_indices: vec![0],
            right_indices: vec![1],
            boundaries: vec![0, 0, 1],
            groups: None,
        };
        assert_eq!(adj, expected);
    }

    #[test]
    fn test_equiv() {
        let zero = TransferVertexOperator::zero();
        let op = Complex64::new(1e-8, 0.0) * TransferVertexOperator::one();
        assert!(op.equiv(&zero, 1e-6));
        assert!(!op.equiv(&zero, 1e-10));
    }

    #[test]
    fn test_normal_ordered_noop() {
        let op = TransferVertexOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            left_indices: vec![0, 1, 1, 0],
            right_indices: vec![0, 1, 1, 1],
            boundaries: vec![0, 4],
            groups: None,
        };

        assert_eq!(op.normal_ordered(), op);
    }

    #[test]
    fn test_normal_ordered_gandon_rel1() {
        // Tests the 1. relation of Eq. (7) from arXiv:2512.11418v1: {T_{jk}, V_{k}} = 0
        let op = TransferVertexOperator {
            coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
            left_indices: vec![0, 1, 0, 0],
            right_indices: vec![1, 1, 1, 0],
            boundaries: vec![0, 2, 4],
            groups: None,
        };

        let expected = TransferVertexOperator {
            coeffs: vec![Complex64::new(-1.0, 0.0), Complex64::new(-1.0, 0.0)],
            left_indices: vec![1, 0, 0, 0],
            right_indices: vec![1, 1, 0, 1],
            boundaries: vec![0, 2, 4],
            groups: None,
        };

        assert_eq!(op.normal_ordered(), expected);
    }

    #[test]
    fn test_normal_ordered_gandon_rel2() {
        // Tests the 2. relation of Eq. (7) from arXiv:2512.11418v1: {T_{jk}, T_{lk}} = 0
        let op = TransferVertexOperator {
            coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
            left_indices: vec![2, 0, 1, 1],
            right_indices: vec![1, 1, 2, 0],
            boundaries: vec![0, 2, 4],
            groups: None,
        };

        let expected = TransferVertexOperator {
            coeffs: vec![Complex64::new(-1.0, 0.0), Complex64::new(-1.0, 0.0)],
            left_indices: vec![0, 2, 1, 1],
            right_indices: vec![1, 1, 0, 2],
            boundaries: vec![0, 2, 4],
            groups: None,
        };

        assert_eq!(op.normal_ordered(), expected);
    }

    #[test]
    fn test_normal_ordered_gandon_rel3() {
        // Tests the 3. relation of Eq. (7) from arXiv:2512.11418v1: [V_{k}, V_{l}] = 0
        let op = TransferVertexOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            left_indices: vec![1, 0],
            right_indices: vec![1, 0],
            boundaries: vec![0, 2],
            groups: None,
        };

        let expected = TransferVertexOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            left_indices: vec![0, 1],
            right_indices: vec![0, 1],
            boundaries: vec![0, 2],
            groups: None,
        };

        assert_eq!(op.normal_ordered(), expected);
    }

    #[test]
    fn test_normal_ordered_gandon_rel4() {
        // Tests the 4. relation of Eq. (7) from arXiv:2512.11418v1: [T_{jk}, V_{l}] = 0
        let op = TransferVertexOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            left_indices: vec![0, 2],
            right_indices: vec![1, 2],
            boundaries: vec![0, 2],
            groups: None,
        };

        let expected = TransferVertexOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            left_indices: vec![2, 0],
            right_indices: vec![2, 1],
            boundaries: vec![0, 2],
            groups: None,
        };

        assert_eq!(op.normal_ordered(), expected);
    }

    #[test]
    fn test_normal_ordered_gandon_rel5() {
        // Tests the 5. relation of Eq. (7) from arXiv:2512.11418v1: [T_{jk}, T_{lm}] = 0
        let op = TransferVertexOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            left_indices: vec![2, 0],
            right_indices: vec![3, 1],
            boundaries: vec![0, 2],
            groups: None,
        };

        let expected = TransferVertexOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            left_indices: vec![0, 2],
            right_indices: vec![1, 3],
            boundaries: vec![0, 2],
            groups: None,
        };

        assert_eq!(op.normal_ordered(), expected);
    }

    #[test]
    fn test_normal_ordered_gandon_rel6() {
        // Tests the 6. relation of Eq. (7) from arXiv:2512.11418v1: [T_{jk}, T_{kj}] = 0
        let op = TransferVertexOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            left_indices: vec![1, 0],
            right_indices: vec![0, 1],
            boundaries: vec![0, 2],
            groups: None,
        };

        let expected = TransferVertexOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            left_indices: vec![0, 1],
            right_indices: vec![1, 0],
            boundaries: vec![0, 2],
            groups: None,
        };

        assert_eq!(op.normal_ordered(), expected);
    }

    #[test]
    fn test_normal_ordered_gandon_rel7() {
        // Tests the 7. relation of Eq. (7) from arXiv:2512.11418v1: [T_{jk}, T_{km}] = 0
        let op = TransferVertexOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            left_indices: vec![2, 0],
            right_indices: vec![0, 1],
            boundaries: vec![0, 2],
            groups: None,
        };

        let expected = TransferVertexOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            left_indices: vec![0, 2],
            right_indices: vec![1, 0],
            boundaries: vec![0, 2],
            groups: None,
        };

        assert_eq!(op.normal_ordered(), expected);
    }

    #[test]
    fn test_is_hermitian() {
        let op = TransferVertexOperator {
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
        let op = TransferVertexOperator {
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
        let op = TransferVertexOperator {
            coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
            left_indices: vec![0, 2, 1, 3],
            right_indices: vec![1, 3, 2, 0],
            boundaries: vec![0, 2, 6],
            groups: None,
        };

        let permutation = vec![4, 2, 5, 3];

        let relabeled = op.relabel_modes(permutation).ok();

        let expected = TransferVertexOperator {
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
        let op = TransferVertexOperator {
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
        let op = TransferVertexOperator {
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
    fn test_num_groups() {
        let mut zero = TransferVertexOperator::zero();

        assert!(zero.num_groups().is_none());

        zero.groups = Some(vec![]);

        assert_eq!(zero.num_groups(), Some(0));

        let mut one = TransferVertexOperator::one();
        one.groups = Some(vec![0]);

        assert_eq!(one.num_groups(), Some(1));
    }

    #[test]
    fn test_split_out_groups() {
        let op = TransferVertexOperator {
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
            TransferVertexOperator {
                coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
                left_indices: vec![0, 2],
                right_indices: vec![1, 3],
                boundaries: vec![0, 1, 2],
                groups: None,
            },
            TransferVertexOperator {
                coeffs: vec![Complex64::new(2.0, 0.0)],
                left_indices: vec![1, 3],
                right_indices: vec![0, 2],
                boundaries: vec![0, 2],
                groups: None,
            },
        ];

        let groups = op.split_out_groups();
        assert_eq!(groups, Some(expected));
    }

    #[test]
    fn test_split_out_groups_err() {
        let op = TransferVertexOperator {
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

        let groups = op.split_out_groups();
        assert!(groups.is_none());
    }
}
