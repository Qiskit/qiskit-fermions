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

use num_complex::Complex64;
use std::cmp::Ordering;

use crate::operators::OperatorTrait;
use crate::operators::directed_interaction_operator::{
    DirectedInteraction, DirectedInteractionOperator,
};
use crate::operators::fermion_operator::FermionOperator;
use crate::operators::majorana_operator::MajoranaOperator;
use crate::operators::undirected_interaction_operator::UndirectedInteractionOperator;

fn map_directed_to_fermion(action: DirectedInteraction) -> FermionOperator {
    match (*action.0).cmp(action.1) {
        Ordering::Equal => FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(-2.0, 0.0)],
            actions: vec![true, false],
            modes: vec![*action.0, *action.0],
            boundaries: vec![0, 0, 2],
            groups: None,
        },
        Ordering::Less => FermionOperator {
            coeffs: vec![
                Complex64::new(0.5, 0.0),
                Complex64::new(0.5, 0.0),
                Complex64::new(-0.5, 0.0),
                Complex64::new(-0.5, 0.0),
            ],
            actions: vec![false, false, false, true, true, false, true, true],
            modes: vec![
                *action.0, *action.1, *action.0, *action.1, *action.0, *action.1, *action.0,
                *action.1,
            ],
            boundaries: vec![0, 2, 4, 6, 8],
            groups: None,
        },
        Ordering::Greater => FermionOperator {
            coeffs: vec![
                Complex64::new(-0.5, 0.0),
                Complex64::new(0.5, 0.0),
                Complex64::new(-0.5, 0.0),
                Complex64::new(0.5, 0.0),
            ],
            actions: vec![false, false, false, true, true, false, true, true],
            modes: vec![
                *action.1, *action.0, *action.1, *action.0, *action.1, *action.0, *action.1,
                *action.0,
            ],
            boundaries: vec![0, 2, 4, 6, 8],
            groups: None,
        },
    }
}

pub fn directed_interaction_to_fermion(inter_op: &DirectedInteractionOperator) -> FermionOperator {
    let mut mapped_operator = FermionOperator::zero();

    inter_op.iter().for_each(|term| {
        let mut mapped_term = FermionOperator::one();

        term.iter()
            .for_each(|action| mapped_term.__imatmul__(&map_directed_to_fermion(action)));

        mapped_term.__imul__(term.coeff);

        mapped_operator.__iadd__(&mapped_term);
    });

    mapped_operator
}

fn map_directed_to_majorana(action: DirectedInteraction) -> MajoranaOperator {
    match (*action.0).cmp(action.1) {
        Ordering::Equal => MajoranaOperator {
            coeffs: vec![Complex64::new(0.0, -1.0)],
            modes: vec![2 * *action.0, 2 * *action.0 + 1],
            boundaries: vec![0, 2],
            groups: None,
        },
        Ordering::Less => MajoranaOperator {
            coeffs: vec![Complex64::new(0.0, 0.5)],
            modes: vec![2 * *action.0 + 1, 2 * *action.1],
            boundaries: vec![0, 2],
            groups: None,
        },
        Ordering::Greater => MajoranaOperator {
            coeffs: vec![Complex64::new(0.0, -0.5)],
            modes: vec![2 * *action.1, 2 * *action.0 + 1],
            boundaries: vec![0, 2],
            groups: None,
        },
    }
}

pub fn directed_interaction_to_majorana(
    inter_op: &DirectedInteractionOperator,
) -> MajoranaOperator {
    let mut mapped_operator = MajoranaOperator::zero();

    inter_op.iter().for_each(|term| {
        let mut mapped_term = MajoranaOperator::one();

        term.iter()
            .for_each(|action| mapped_term.__imatmul__(&map_directed_to_majorana(action)));

        mapped_term.__imul__(term.coeff);

        mapped_operator.__iadd__(&mapped_term);
    });

    mapped_operator
}

fn map_directed_to_undirected(action: DirectedInteraction) -> UndirectedInteractionOperator {
    match (*action.0).cmp(action.1) {
        Ordering::Equal => UndirectedInteractionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            left_indices: vec![*action.0],
            right_indices: vec![*action.1],
            boundaries: vec![0, 1],
            groups: None,
        },
        Ordering::Less => UndirectedInteractionOperator {
            coeffs: vec![Complex64::new(0.0, 0.5)],
            left_indices: vec![*action.0, *action.0],
            right_indices: vec![*action.0, *action.1],
            boundaries: vec![0, 2],
            groups: None,
        },
        Ordering::Greater => UndirectedInteractionOperator {
            coeffs: vec![Complex64::new(0.0, 0.5)],
            left_indices: vec![*action.1, *action.0],
            right_indices: vec![*action.0, *action.0],
            boundaries: vec![0, 2],
            groups: None,
        },
    }
}

pub fn directed_interaction_to_undirected(
    inter_op: &DirectedInteractionOperator,
) -> UndirectedInteractionOperator {
    let mut mapped_operator = UndirectedInteractionOperator::zero();

    inter_op.iter().for_each(|term| {
        let mut mapped_term = UndirectedInteractionOperator::one();

        term.iter()
            .for_each(|action| mapped_term.__imatmul__(&map_directed_to_undirected(action)));

        mapped_term.__imul__(term.coeff);

        mapped_operator.__iadd__(&mapped_term);
    });

    mapped_operator
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mappers::library::majorana_fermion::majorana_to_fermion;

    #[test]
    fn test_directed_to_fermion() {
        let inter_op = DirectedInteractionOperator {
            coeffs: vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(-2.0, 0.0),
            ],
            left_indices: vec![0, 1, 2],
            right_indices: vec![0, 2, 1],
            boundaries: vec![0, 1, 2, 3],
            groups: None,
        };

        let fer_op = directed_interaction_to_fermion(&inter_op).simplify(1e-10);

        let expected = FermionOperator {
            coeffs: vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(-2.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(-2.0, 0.0),
            ],
            actions: vec![true, false, false, false, true, true],
            modes: vec![0, 0, 1, 2, 1, 2],
            boundaries: vec![0, 0, 2, 4, 6],
            groups: None,
        };

        assert!(fer_op.equiv(&expected, 1e-10));
    }

    #[test]
    fn test_directed_to_majorana() {
        let inter_op = DirectedInteractionOperator {
            coeffs: vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(-2.0, 0.0),
            ],
            left_indices: vec![0, 1, 2],
            right_indices: vec![0, 2, 1],
            boundaries: vec![0, 1, 2, 3],
            groups: None,
        };

        let maj_op = directed_interaction_to_majorana(&inter_op);

        let expected = MajoranaOperator {
            coeffs: vec![
                Complex64::new(0.0, -1.0),
                Complex64::new(0.0, 1.0),
                Complex64::new(0.0, 1.0),
            ],
            modes: vec![0, 1, 3, 4, 2, 5],
            boundaries: vec![0, 2, 4, 6],
            groups: None,
        };

        assert_eq!(maj_op, expected);
    }

    #[test]
    fn test_directed_to_undirected() {
        let inter_op = DirectedInteractionOperator {
            coeffs: vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(-2.0, 0.0),
            ],
            left_indices: vec![0, 1, 2],
            right_indices: vec![0, 2, 1],
            boundaries: vec![0, 1, 2, 3],
            groups: None,
        };

        let undirected_op = directed_interaction_to_undirected(&inter_op);

        let expected = UndirectedInteractionOperator {
            coeffs: vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 1.0),
                Complex64::new(0.0, -1.0),
            ],
            left_indices: vec![0, 1, 1, 1, 2],
            right_indices: vec![0, 1, 2, 2, 2],
            boundaries: vec![0, 1, 3, 5],
            groups: None,
        };

        assert_eq!(undirected_op, expected);
    }
}
