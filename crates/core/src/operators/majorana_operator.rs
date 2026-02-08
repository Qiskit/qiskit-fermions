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

use crate::operators::{OperatorMacro, OperatorTrait};
use num_complex::{Complex64, ComplexFloat};
use std::collections::HashMap;
use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign,
};

pub type MajoranaAction<'a> = &'a u32;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MajoranaOperatorTermView<'a> {
    pub coeff: Complex64,
    pub modes: &'a [u32],
}

impl MajoranaOperatorTermView<'_> {
    pub fn iter(&'_ self) -> impl ExactSizeIterator<Item = MajoranaAction<'_>> + '_ {
        self.modes.iter()
    }

    // TODO: refactor these following methods

    pub fn to_vec(&'_ self) -> Vec<MajoranaAction<'_>> {
        self.modes.iter().collect()
    }

    pub fn into_vec(&'_ self) -> Vec<u32> {
        self.modes.to_vec()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MajoranaOperator {
    pub coeffs: Vec<Complex64>,
    pub modes: Vec<u32>,
    pub boundaries: Vec<usize>,
}

crate::impl_operator_macro!(MajoranaOperator);

impl MajoranaOperator {
    pub fn simplify(&self, atol: f64) -> Self {
        let mut terms = HashMap::new();
        for term in self.iter() {
            terms
                .entry(term.modes)
                .and_modify(|c| *c += term.coeff)
                .or_insert(term.coeff);
        }
        let mut out = Self::zero();
        terms
            .iter()
            .filter(|(_, coeff)| coeff.abs() > atol)
            .for_each(|(modes, coeff)| {
                out.coeffs.push(*coeff);
                out.modes.extend_from_slice(modes);
                out.boundaries.push(out.modes.len());
            });
        out
    }

    pub fn iter(&'_ self) -> impl ExactSizeIterator<Item = MajoranaOperatorTermView<'_>> + '_ {
        self.coeffs.iter().enumerate().map(|(i, coeff)| {
            let start = self.boundaries[i];
            let end = self.boundaries[i + 1];
            MajoranaOperatorTermView {
                coeff: *coeff,
                modes: &self.modes[start..end],
            }
        })
    }

    pub fn normal_ordered(&self, reduce: bool) -> Self {
        let mut coeffs = vec![];
        let mut modes = vec![];
        let mut boundaries = vec![0];

        for term in self.iter() {
            let (mut sorted_term, sign) = sort_and_parity(term.modes);
            if reduce {
                sorted_term = reduce_pairs(&sorted_term);
            }
            coeffs.push(Complex64::new(sign as f64, 0.0) * term.coeff);
            modes.extend_from_slice(&sorted_term);
            boundaries.push(modes.len());
        }
        Self {
            coeffs,
            modes,
            boundaries,
        }
    }

    pub fn is_hermitian(&self, atol: f64) -> bool {
        let mut diff = (self.__sub__(&self.adjoint())).normal_ordered(true);
        diff.ichop(atol);
        diff.equiv(&Self::zero(), atol)
    }

    pub fn many_body_order(&self) -> u32 {
        let mut max = 0;
        let mut prev_b = 0;
        // TODO: refactor this
        self.boundaries[1..].iter().for_each(|b| {
            let d = b - prev_b;
            if d > max {
                max = d;
            }
            prev_b = *b;
        });
        max as u32
    }

    pub fn is_even(&self) -> bool {
        let mut prev_b = 0;
        for b in self.boundaries[1..].iter() {
            if (b - prev_b) % 2 != 0 {
                return false;
            }
            prev_b = *b;
        }
        true
    }
}

/// FIXME: follow rustdoc standards
///
/// Compute the parity of a permutation using cycle decomposition.
///
/// Args:
///     perm: array of modes representing the permutation.
///
/// Returns:
///     +1 if the permutation is even, -1 if it is odd.
fn permutation_parity(perm: &[usize]) -> i32 {
    let n = perm.len();
    let mut visited = vec![false; n];
    let mut sign = 1;

    for i in 0..n {
        if !visited[i] {
            let mut cycle_len = 0;
            let mut j = i;
            while !visited[j] {
                visited[j] = true;
                j = perm[j];
                cycle_len += 1;
            }
            if cycle_len > 0 && cycle_len % 2 == 0 {
                sign = -sign;
            }
        }
    }

    sign
}

/// FIXME: follow rustdoc standards
///
/// Sort a list and compute the parity of the sorting permutation.
///
/// Args:
///     tpl: tuple of integers to be sorted.
///
/// Returns:
///     A tuple (sorted_tpl, sign):
///     - sorted_tpl: tuple containing the sorted integers.
///     - sign: +1 if the sorting permutation is even, -1 if it is odd.
fn sort_and_parity(tpl: &[u32]) -> (Vec<u32>, i32) {
    let mut indexed: Vec<(usize, u32)> = tpl.iter().cloned().enumerate().collect();
    // we need stable sort to ensure that the parity is correctly computed
    indexed.sort_by_key(|&(_, val)| -(val as i32));

    let perm: Vec<usize> = indexed.iter().map(|&(i, _)| i).collect();
    let sorted_tpl: Vec<u32> = indexed.iter().map(|&(_, val)| val).collect();

    (sorted_tpl, permutation_parity(&perm))
}

/// FIXME: follow rustdoc standards
///
/// Remove pairs of (consecutive) equal integers from a list.
///
/// Args:
///     tpl: tuple of integers.
///
/// Returns:
///     A tuple of integers after removing pairs.
fn reduce_pairs(tpl: &[u32]) -> Vec<u32> {
    let mut reduced = Vec::new();
    let mut i = 0;
    let n = tpl.len();

    while i < n {
        let mut count = 1;
        while i + count < n && tpl[i + count] == tpl[i] {
            count += 1;
        }
        if count % 2 == 1 {
            reduced.push(tpl[i]);
        }
        i += count;
    }

    reduced
}

impl OperatorTrait for MajoranaOperator {
    fn zero() -> Self {
        Self {
            coeffs: vec![],
            modes: vec![],
            boundaries: vec![0],
        }
    }

    fn one() -> Self {
        Self {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            modes: vec![],
            boundaries: vec![0, 0],
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

    fn adjoint(&self) -> Self {
        let mut coeffs = vec![];
        let mut modes = vec![];

        self.iter().for_each(|term| {
            coeffs.push(term.coeff.conj());
            modes.extend(term.modes.iter().rev());
        });

        Self {
            coeffs,
            modes,
            boundaries: self.boundaries.to_vec(),
        }
    }

    fn __iadd__(&mut self, other: &Self) {
        self.coeffs.extend_from_slice(&other.coeffs);
        self.modes.extend_from_slice(&other.modes);
        let offset = self.boundaries[self.boundaries.len() - 1];
        self.boundaries
            .extend(other.boundaries[1..].iter().map(|b| b + offset));
    }

    fn __imul__(&mut self, other: Complex64) {
        self.coeffs.iter_mut().for_each(|c| *c *= other);
    }

    fn __iand__(&mut self, other: &Self) {
        let mut coeffs = vec![];
        let mut modes = vec![];
        let mut boundaries = vec![0];

        for left in self.iter() {
            for right in other.iter() {
                coeffs.push(left.coeff * right.coeff);
                modes.extend_from_slice(right.modes);
                modes.extend_from_slice(left.modes);
                boundaries.push(modes.len());
            }
        }

        self.coeffs = coeffs;
        self.modes = modes;
        self.boundaries = boundaries;
    }

    fn ichop(&mut self, atol: f64) {
        let mut coeffs = vec![];
        let mut modes = vec![];
        let mut boundaries = vec![0];

        self.iter()
            .filter(|term| term.coeff.abs() > atol)
            .for_each(|term| {
                coeffs.push(term.coeff.conj());
                modes.extend_from_slice(term.modes);
                boundaries.push(modes.len());
            });

        self.coeffs = coeffs;
        self.modes = modes;
        self.boundaries = boundaries;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero() {
        let zero = MajoranaOperator::zero();
        assert_eq!(
            zero,
            MajoranaOperator {
                coeffs: vec![],
                modes: vec![],
                boundaries: vec![0],
            }
        );
    }

    #[test]
    fn test_one() {
        let one = MajoranaOperator::one();
        assert_eq!(
            one,
            MajoranaOperator {
                coeffs: vec![Complex64::new(1.0, 0.0)],
                modes: vec![],
                boundaries: vec![0, 0],
            }
        );
    }

    #[test]
    fn test_add() {
        let one = MajoranaOperator::one();
        let two = MajoranaOperator {
            coeffs: vec![Complex64::new(2.0, 0.0)],
            modes: vec![],
            boundaries: vec![0, 0],
        };
        let three = one + two;
        assert_eq!(
            three,
            MajoranaOperator {
                coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
                modes: vec![],
                boundaries: vec![0, 0, 0],
            }
        );
    }

    #[test]
    fn test_add_assign() {
        let mut op = MajoranaOperator::one();
        let two = MajoranaOperator {
            coeffs: vec![Complex64::new(2.0, 0.0)],
            modes: vec![],
            boundaries: vec![0, 0],
        };
        op += two;
        assert_eq!(
            op,
            MajoranaOperator {
                coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
                modes: vec![],
                boundaries: vec![0, 0, 0],
            }
        );
    }

    #[test]
    fn test_sub() {
        let one = MajoranaOperator::one();
        let two = MajoranaOperator {
            coeffs: vec![Complex64::new(2.0, 0.0)],
            modes: vec![],
            boundaries: vec![0, 0],
        };
        let new_one = two - one;
        assert_eq!(
            new_one,
            MajoranaOperator {
                coeffs: vec![Complex64::new(2.0, 0.0), Complex64::new(-1.0, 0.0)],
                modes: vec![],
                boundaries: vec![0, 0, 0],
            }
        );
    }

    #[test]
    fn test_sub_assign() {
        let mut op = MajoranaOperator::one();
        let two = MajoranaOperator {
            coeffs: vec![Complex64::new(2.0, 0.0)],
            modes: vec![],
            boundaries: vec![0, 0],
        };
        op -= two;
        assert_eq!(
            op,
            MajoranaOperator {
                coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(-2.0, 0.0)],
                modes: vec![],
                boundaries: vec![0, 0, 0],
            }
        );
    }

    #[test]
    fn test_mul() {
        let one = MajoranaOperator::one();
        let three = one * Complex64::new(3.0, 0.0);
        assert_eq!(
            three,
            MajoranaOperator {
                coeffs: vec![Complex64::new(3.0, 0.0)],
                modes: vec![],
                boundaries: vec![0, 0],
            }
        );
    }

    #[test]
    fn test_rmul() {
        let one = MajoranaOperator::one();
        let three = Complex64::new(3.0, 0.0) * one;
        assert_eq!(
            three,
            MajoranaOperator {
                coeffs: vec![Complex64::new(3.0, 0.0)],
                modes: vec![],
                boundaries: vec![0, 0],
            }
        );
    }

    #[test]
    fn test_mul_assign() {
        let mut op = MajoranaOperator::one();
        op *= Complex64::new(3.0, 0.0);
        assert_eq!(
            op,
            MajoranaOperator {
                coeffs: vec![Complex64::new(3.0, 0.0)],
                modes: vec![],
                boundaries: vec![0, 0],
            }
        );
    }

    #[test]
    fn test_div() {
        let three = MajoranaOperator {
            coeffs: vec![Complex64::new(3.0, 0.0)],
            modes: vec![],
            boundaries: vec![0, 0],
        };
        let one_half = three / Complex64::new(2.0, 0.0);
        assert_eq!(
            one_half,
            MajoranaOperator {
                coeffs: vec![Complex64::new(1.5, 0.0)],
                modes: vec![],
                boundaries: vec![0, 0],
            }
        );
    }

    #[test]
    fn test_idiv() {
        let mut op = MajoranaOperator {
            coeffs: vec![Complex64::new(3.0, 0.0)],
            modes: vec![],
            boundaries: vec![0, 0],
        };
        op /= Complex64::new(2.0, 0.0);
        assert_eq!(
            op,
            MajoranaOperator {
                coeffs: vec![Complex64::new(1.5, 0.0)],
                modes: vec![],
                boundaries: vec![0, 0],
            }
        );
    }

    #[test]
    fn test_neg() {
        let one = MajoranaOperator::one();
        assert_eq!(
            -one,
            MajoranaOperator {
                coeffs: vec![Complex64::new(-1.0, 0.0)],
                modes: vec![],
                boundaries: vec![0, 0],
            }
        );
    }

    #[test]
    fn test_and() {
        let op1 = MajoranaOperator {
            coeffs: vec![Complex64::new(2.0, 0.0), Complex64::new(3.0, 0.0)],
            modes: vec![0, 1],
            boundaries: vec![0, 0, 2],
        };
        let op2 = MajoranaOperator {
            coeffs: vec![Complex64::new(1.5, 0.0), Complex64::new(4.0, 0.0)],
            modes: vec![1, 0],
            boundaries: vec![0, 0, 2],
        };
        let result = op1 & op2;
        assert_eq!(
            result,
            MajoranaOperator {
                coeffs: vec![
                    Complex64::new(3.0, 0.0),
                    Complex64::new(8.0, 0.0),
                    Complex64::new(4.5, 0.0),
                    Complex64::new(12.0, 0.0),
                ],
                modes: vec![1, 0, 0, 1, 1, 0, 0, 1],
                boundaries: vec![0, 0, 2, 4, 8],
            }
        );
    }

    #[test]
    fn test_and_assign() {
        let mut op1 = MajoranaOperator {
            coeffs: vec![Complex64::new(2.0, 0.0), Complex64::new(3.0, 0.0)],
            modes: vec![0, 1],
            boundaries: vec![0, 0, 2],
        };
        let op2 = MajoranaOperator {
            coeffs: vec![Complex64::new(1.5, 0.0), Complex64::new(4.0, 0.0)],
            modes: vec![1, 0],
            boundaries: vec![0, 0, 2],
        };
        op1 &= op2;
        assert_eq!(
            op1,
            MajoranaOperator {
                coeffs: vec![
                    Complex64::new(3.0, 0.0),
                    Complex64::new(8.0, 0.0),
                    Complex64::new(4.5, 0.0),
                    Complex64::new(12.0, 0.0),
                ],
                modes: vec![1, 0, 0, 1, 1, 0, 0, 1],
                boundaries: vec![0, 0, 2, 4, 8],
            }
        );
    }

    #[test]
    fn test_pow() {
        let op = MajoranaOperator {
            coeffs: vec![Complex64::new(2.0, 0.0)],
            modes: vec![0],
            boundaries: vec![0, 1],
        };
        // exponent=0
        let one = MajoranaOperator::one();
        assert_eq!(op.__pow__(0), one);

        // exponent=1
        assert_eq!(op.__pow__(1), op);

        // exponent=2
        let squared = op.__pow__(2);
        assert_eq!(
            squared,
            MajoranaOperator {
                coeffs: vec![Complex64::new(4.0, 0.0),],
                modes: vec![0, 0],
                boundaries: vec![0, 2],
            }
        );
    }

    #[test]
    fn test_ichop() {
        let mut op = MajoranaOperator {
            coeffs: vec![
                Complex64::new(1e-4, 0.0),
                Complex64::new(1e-6, 0.0),
                Complex64::new(1e-8, 0.0),
            ],
            modes: vec![0, 1],
            boundaries: vec![0, 0, 1, 2],
        };

        op.ichop(1e-7);

        let expected1 = MajoranaOperator {
            coeffs: vec![Complex64::new(1e-4, 0.0), Complex64::new(1e-6, 0.0)],
            modes: vec![0],
            boundaries: vec![0, 0, 1],
        };

        assert_eq!(op, expected1);

        op.ichop(1e-5);

        let expected2 = MajoranaOperator {
            coeffs: vec![Complex64::new(1e-4, 0.0)],
            modes: vec![],
            boundaries: vec![0, 0],
        };

        assert_eq!(op, expected2);
    }

    #[test]
    fn test_adjoint() {
        let op1 = MajoranaOperator {
            coeffs: vec![Complex64::new(0.0, 2.0), Complex64::new(3.0, 0.0)],
            modes: vec![0, 1],
            boundaries: vec![0, 0, 2],
        };
        let adj = op1.adjoint();
        assert_eq!(
            adj,
            MajoranaOperator {
                coeffs: vec![Complex64::new(0.0, -2.0), Complex64::new(3.0, 0.0)],
                modes: vec![1, 0],
                boundaries: vec![0, 0, 2],
            }
        );
    }

    #[test]
    fn test_equiv() {
        let zero = MajoranaOperator::zero();
        let op = MajoranaOperator {
            coeffs: vec![Complex64::new(1e-8, 0.0)],
            modes: vec![],
            boundaries: vec![0, 0],
        };
        assert!(op.equiv(&zero, 1e-6));
        assert!(!op.equiv(&zero, 1e-10));
    }

    #[test]
    fn test_normal_ordered_1() {
        let op = MajoranaOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            modes: vec![1, 0],
            boundaries: vec![0, 2],
        };

        assert_eq!(op.normal_ordered(false), op);
    }

    #[test]
    fn test_normal_ordered_2() {
        let op = MajoranaOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            modes: vec![0, 1],
            boundaries: vec![0, 2],
        };

        let expected = MajoranaOperator {
            coeffs: vec![Complex64::new(-1.0, 0.0)],
            modes: vec![1, 0],
            boundaries: vec![0, 2],
        };

        assert_eq!(op.normal_ordered(false), expected);
    }

    #[test]
    fn test_normal_ordered_3() {
        let op = MajoranaOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            modes: vec![0, 0],
            boundaries: vec![0, 2],
        };

        let expected = MajoranaOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            modes: vec![],
            boundaries: vec![0, 0],
        };

        assert_eq!(op.normal_ordered(true), expected);
    }

    #[test]
    fn test_is_hermitian() {
        let op = MajoranaOperator {
            coeffs: vec![Complex64::new(0.0, 1.00001), Complex64::new(0.0, -1.0)],
            modes: vec![0, 1, 2, 3, 3, 2, 1, 0],
            boundaries: vec![0, 4, 8],
        };
        assert!(op.is_hermitian(1e-4));
        assert!(!op.is_hermitian(1e-6));
    }

    #[test]
    fn test_many_body_order() {
        assert_eq!(MajoranaOperator::one().many_body_order(), 0);

        assert_eq!(
            MajoranaOperator {
                coeffs: vec![Complex64::new(1.0, 0.0)],
                modes: vec![0],
                boundaries: vec![0, 1],
            }
            .many_body_order(),
            1
        );

        assert_eq!(
            MajoranaOperator {
                coeffs: vec![Complex64::new(1.0, 0.0)],
                modes: vec![0, 1],
                boundaries: vec![0, 2],
            }
            .many_body_order(),
            2
        );
    }

    #[test]
    fn test_is_even() {
        assert!(MajoranaOperator::zero().is_even());
        assert!(MajoranaOperator::one().is_even());
        assert!(
            !MajoranaOperator {
                coeffs: vec![Complex64::new(1.0, 0.0)],
                modes: vec![0],
                boundaries: vec![0, 1],
            }
            .is_even()
        );
        assert!(
            MajoranaOperator {
                coeffs: vec![Complex64::new(1.0, 0.0)],
                modes: vec![0, 1, 2, 3],
                boundaries: vec![0, 4],
            }
            .is_even()
        );
    }
}
