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
use std::cmp::Ordering;
use std::collections::HashMap;
use std::iter::zip;
use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign,
};

pub type FermionAction<'a> = (&'a bool, &'a u32);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FermionOperatorTermView<'a> {
    pub coeff: Complex64,
    pub actions: &'a [bool],
    pub indices: &'a [u32],
}

impl FermionOperatorTermView<'_> {
    pub fn iter(&'_ self) -> impl ExactSizeIterator<Item = FermionAction<'_>> + '_ {
        zip(self.actions, self.indices)
    }

    // TODO: refactor these following methods

    pub fn to_vec(&'_ self) -> Vec<FermionAction<'_>> {
        zip(self.actions, self.indices).collect()
    }

    pub fn into_vec(&'_ self) -> Vec<(bool, u32)> {
        zip(self.actions.to_vec(), self.indices.to_vec()).collect()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FermionOperator {
    pub coeffs: Vec<Complex64>,
    pub actions: Vec<bool>,
    pub indices: Vec<u32>,
    pub boundaries: Vec<usize>,
}

crate::impl_operator_macro!(FermionOperator);

impl FermionOperator {
    pub fn simplify(&self, atol: f64) -> Self {
        let mut terms = HashMap::new();
        for term in self.iter() {
            terms
                .entry((term.indices, term.actions))
                .and_modify(|c| *c += term.coeff)
                .or_insert(term.coeff);
        }
        let mut out = Self::zero();
        terms
            .iter()
            .filter(|((_, _), coeff)| coeff.abs() > atol)
            .for_each(|((indices, actions), coeff)| {
                out.coeffs.push(*coeff);
                out.actions.extend_from_slice(actions);
                out.indices.extend_from_slice(indices);
                out.boundaries.push(out.indices.len());
            });
        out
    }

    pub fn iter(&'_ self) -> impl ExactSizeIterator<Item = FermionOperatorTermView<'_>> + '_ {
        self.coeffs.iter().enumerate().map(|(i, coeff)| {
            let start = self.boundaries[i];
            let end = self.boundaries[i + 1];
            FermionOperatorTermView {
                coeff: *coeff,
                actions: &self.actions[start..end],
                indices: &self.indices[start..end],
            }
        })
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

fn _normal_ordered_term(term_view: FermionOperatorTermView) -> FermionOperator {
    let mut coeffs = vec![];
    let mut actions = vec![];
    let mut indices = vec![];
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
                    match (index_right).cmp(index_left) {
                        Ordering::Equal => {
                            // operators are the same, so product is zero
                            zero = true;
                            break;
                        }
                        Ordering::Greater => {
                            // swap operators and update sign
                            term.swap(j - 1, j);
                            parity = !parity;
                        }
                        Ordering::Less => {}
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
            indices.push(i);
        });
        boundaries.push(indices.len())
    }
    FermionOperator {
        coeffs,
        actions,
        indices,
        boundaries,
    }
}

impl OperatorTrait for FermionOperator {
    fn zero() -> Self {
        Self {
            coeffs: vec![],
            actions: vec![],
            indices: vec![],
            boundaries: vec![0],
        }
    }

    fn one() -> Self {
        Self {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            actions: vec![],
            indices: vec![],
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
        let mut actions = vec![];
        let mut indices = vec![];

        self.iter().for_each(|term| {
            coeffs.push(term.coeff.conj());
            actions.extend(term.actions.iter().rev().map(|a| !a));
            indices.extend(term.indices.iter().rev());
        });

        Self {
            coeffs,
            actions,
            indices,
            boundaries: self.boundaries.to_vec(),
        }
    }

    fn __iadd__(&mut self, other: &Self) {
        self.coeffs.extend_from_slice(&other.coeffs);
        self.actions.extend_from_slice(&other.actions);
        self.indices.extend_from_slice(&other.indices);
        let offset = self.boundaries[self.boundaries.len() - 1];
        self.boundaries
            .extend(other.boundaries[1..].iter().map(|b| b + offset));
    }

    fn __imul__(&mut self, other: Complex64) {
        self.coeffs.iter_mut().for_each(|c| *c *= other);
    }

    fn __iand__(&mut self, other: &Self) {
        let mut coeffs = vec![];
        let mut actions = vec![];
        let mut indices = vec![];
        let mut boundaries = vec![0];

        for left in self.iter() {
            for right in other.iter() {
                coeffs.push(left.coeff * right.coeff);
                actions.extend_from_slice(right.actions);
                actions.extend_from_slice(left.actions);
                indices.extend_from_slice(right.indices);
                indices.extend_from_slice(left.indices);
                boundaries.push(indices.len());
            }
        }

        self.coeffs = coeffs;
        self.actions = actions;
        self.indices = indices;
        self.boundaries = boundaries;
    }

    fn ichop(&mut self, atol: f64) {
        let mut coeffs = vec![];
        let mut actions = vec![];
        let mut indices = vec![];
        let mut boundaries = vec![0];

        self.iter()
            .filter(|term| term.coeff.abs() > atol)
            .for_each(|term| {
                coeffs.push(term.coeff.conj());
                actions.extend_from_slice(term.actions);
                indices.extend_from_slice(term.indices);
                boundaries.push(indices.len());
            });

        self.coeffs = coeffs;
        self.actions = actions;
        self.indices = indices;
        self.boundaries = boundaries;
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
                indices: vec![],
                boundaries: vec![0],
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
                indices: vec![],
                boundaries: vec![0, 0],
            }
        );
    }

    #[test]
    fn test_add() {
        let one = FermionOperator::one();
        let two = FermionOperator {
            coeffs: vec![Complex64::new(2.0, 0.0)],
            actions: vec![],
            indices: vec![],
            boundaries: vec![0, 0],
        };
        let three = one + two;
        assert_eq!(
            three,
            FermionOperator {
                coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
                actions: vec![],
                indices: vec![],
                boundaries: vec![0, 0, 0],
            }
        );
    }

    #[test]
    fn test_add_assign() {
        let mut op = FermionOperator::one();
        let two = FermionOperator {
            coeffs: vec![Complex64::new(2.0, 0.0)],
            actions: vec![],
            indices: vec![],
            boundaries: vec![0, 0],
        };
        op += two;
        assert_eq!(
            op,
            FermionOperator {
                coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
                actions: vec![],
                indices: vec![],
                boundaries: vec![0, 0, 0],
            }
        );
    }

    #[test]
    fn test_sub() {
        let one = FermionOperator::one();
        let two = FermionOperator {
            coeffs: vec![Complex64::new(2.0, 0.0)],
            actions: vec![],
            indices: vec![],
            boundaries: vec![0, 0],
        };
        let new_one = two - one;
        assert_eq!(
            new_one,
            FermionOperator {
                coeffs: vec![Complex64::new(2.0, 0.0), Complex64::new(-1.0, 0.0)],
                actions: vec![],
                indices: vec![],
                boundaries: vec![0, 0, 0],
            }
        );
    }

    #[test]
    fn test_sub_assign() {
        let mut op = FermionOperator::one();
        let two = FermionOperator {
            coeffs: vec![Complex64::new(2.0, 0.0)],
            actions: vec![],
            indices: vec![],
            boundaries: vec![0, 0],
        };
        op -= two;
        assert_eq!(
            op,
            FermionOperator {
                coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(-2.0, 0.0)],
                actions: vec![],
                indices: vec![],
                boundaries: vec![0, 0, 0],
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
                indices: vec![],
                boundaries: vec![0, 0],
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
                indices: vec![],
                boundaries: vec![0, 0],
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
                indices: vec![],
                boundaries: vec![0, 0],
            }
        );
    }

    #[test]
    fn test_div() {
        let three = FermionOperator {
            coeffs: vec![Complex64::new(3.0, 0.0)],
            actions: vec![],
            indices: vec![],
            boundaries: vec![0, 0],
        };
        let one_half = three / Complex64::new(2.0, 0.0);
        assert_eq!(
            one_half,
            FermionOperator {
                coeffs: vec![Complex64::new(1.5, 0.0)],
                actions: vec![],
                indices: vec![],
                boundaries: vec![0, 0],
            }
        );
    }

    #[test]
    fn test_idiv() {
        let mut op = FermionOperator {
            coeffs: vec![Complex64::new(3.0, 0.0)],
            actions: vec![],
            indices: vec![],
            boundaries: vec![0, 0],
        };
        op /= Complex64::new(2.0, 0.0);
        assert_eq!(
            op,
            FermionOperator {
                coeffs: vec![Complex64::new(1.5, 0.0)],
                actions: vec![],
                indices: vec![],
                boundaries: vec![0, 0],
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
                indices: vec![],
                boundaries: vec![0, 0],
            }
        );
    }

    #[test]
    fn test_and() {
        let op1 = FermionOperator {
            coeffs: vec![Complex64::new(2.0, 0.0), Complex64::new(3.0, 0.0)],
            actions: vec![true, false],
            indices: vec![0, 1],
            boundaries: vec![0, 0, 2],
        };
        let op2 = FermionOperator {
            coeffs: vec![Complex64::new(1.5, 0.0), Complex64::new(4.0, 0.0)],
            actions: vec![true, false],
            indices: vec![1, 0],
            boundaries: vec![0, 0, 2],
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
                indices: vec![1, 0, 0, 1, 1, 0, 0, 1],
                boundaries: vec![0, 0, 2, 4, 8],
            }
        );
    }

    #[test]
    fn test_and_assign() {
        let mut op1 = FermionOperator {
            coeffs: vec![Complex64::new(2.0, 0.0), Complex64::new(3.0, 0.0)],
            actions: vec![true, false],
            indices: vec![0, 1],
            boundaries: vec![0, 0, 2],
        };
        let op2 = FermionOperator {
            coeffs: vec![Complex64::new(1.5, 0.0), Complex64::new(4.0, 0.0)],
            actions: vec![true, false],
            indices: vec![1, 0],
            boundaries: vec![0, 0, 2],
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
                indices: vec![1, 0, 0, 1, 1, 0, 0, 1],
                boundaries: vec![0, 0, 2, 4, 8],
            }
        );
    }

    #[test]
    fn test_pow() {
        let op = FermionOperator {
            coeffs: vec![Complex64::new(2.0, 0.0)],
            actions: vec![true],
            indices: vec![0],
            boundaries: vec![0, 1],
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
                indices: vec![0, 0],
                boundaries: vec![0, 2],
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
            indices: vec![0, 0],
            boundaries: vec![0, 0, 1, 2],
        };

        op.ichop(1e-7);

        let expected1 = FermionOperator {
            coeffs: vec![Complex64::new(1e-4, 0.0), Complex64::new(1e-6, 0.0)],
            actions: vec![true],
            indices: vec![0],
            boundaries: vec![0, 0, 1],
        };

        assert_eq!(op, expected1);

        op.ichop(1e-5);

        let expected2 = FermionOperator {
            coeffs: vec![Complex64::new(1e-4, 0.0)],
            actions: vec![],
            indices: vec![],
            boundaries: vec![0, 0],
        };

        assert_eq!(op, expected2);
    }

    #[test]
    fn test_adjoint() {
        let op1 = FermionOperator {
            coeffs: vec![Complex64::new(0.0, 2.0), Complex64::new(3.0, 0.0)],
            actions: vec![true, false],
            indices: vec![0, 1],
            boundaries: vec![0, 0, 2],
        };
        let adj = op1.adjoint();
        assert_eq!(
            adj,
            FermionOperator {
                coeffs: vec![Complex64::new(0.0, -2.0), Complex64::new(3.0, 0.0)],
                actions: vec![true, false],
                indices: vec![1, 0],
                boundaries: vec![0, 0, 2],
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
            indices: vec![0, 1],
            boundaries: vec![0, 2],
        };

        assert_eq!(op.normal_ordered(), op);
    }

    #[test]
    fn test_normal_ordered_2() {
        let op = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            actions: vec![true, true],
            indices: vec![0, 1],
            boundaries: vec![0, 2],
        };

        let expected = FermionOperator {
            coeffs: vec![Complex64::new(-1.0, 0.0)],
            actions: vec![true, true],
            indices: vec![1, 0],
            boundaries: vec![0, 2],
        };

        assert_eq!(op.normal_ordered(), expected);
    }

    #[test]
    fn test_normal_ordered_3() {
        let op = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            actions: vec![false, true],
            indices: vec![0, 0],
            boundaries: vec![0, 2],
        };

        let expected = FermionOperator {
            coeffs: vec![Complex64::new(-1.0, 0.0), Complex64::new(1.0, 0.0)],
            actions: vec![true, false],
            indices: vec![0, 0],
            boundaries: vec![0, 2, 2],
        };

        assert_eq!(op.normal_ordered(), expected);
    }

    #[test]
    fn test_is_hermitian() {
        let op = FermionOperator {
            coeffs: vec![Complex64::new(0.0, 1.00001), Complex64::new(0.0, -1.0)],
            actions: vec![true, false, true, false],
            indices: vec![0, 1, 1, 0],
            boundaries: vec![0, 2, 4],
        };
        assert!(op.is_hermitian(1e-4));
        assert!(!op.is_hermitian(1e-6));
    }

    #[test]
    fn test_many_body_order() {
        assert_eq!(FermionOperator::one().many_body_order(), 0);

        assert_eq!(
            FermionOperator {
                coeffs: vec![Complex64::new(1.0, 0.0)],
                actions: vec![true],
                indices: vec![0],
                boundaries: vec![0, 1],
            }
            .many_body_order(),
            1
        );

        assert_eq!(
            FermionOperator {
                coeffs: vec![Complex64::new(1.0, 0.0)],
                actions: vec![true, false],
                indices: vec![0, 1],
                boundaries: vec![0, 2],
            }
            .many_body_order(),
            2
        );
    }

    #[test]
    fn test_conserves_particle_number() {
        let op1 = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            actions: vec![true, false],
            indices: vec![0, 1],
            boundaries: vec![0, 2],
        };

        assert!(op1.conserves_particle_number());

        let op2 = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            actions: vec![true],
            indices: vec![0],
            boundaries: vec![0, 1],
        };

        assert!(!op2.conserves_particle_number());
    }
}
