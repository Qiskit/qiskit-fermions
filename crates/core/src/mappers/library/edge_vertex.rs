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

use crate::operators::OperatorTrait;
use crate::operators::edge_vertex_operator::{EdgeAction, EdgeVertexOperator};
use crate::operators::fermion_operator::FermionOperator;
use crate::operators::majorana_operator::MajoranaOperator;

fn map_edge_vertex_to_fermion(action: EdgeAction) -> FermionOperator {
    if *action.0 == *action.1 {
        FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(-2.0, 0.0)],
            actions: vec![true, false],
            modes: vec![*action.0, *action.0],
            boundaries: vec![0, 0, 2],
            groups: None,
        }
    } else {
        FermionOperator {
            coeffs: vec![
                Complex64::new(0.0, -1.0),
                Complex64::new(0.0, -1.0),
                Complex64::new(0.0, -1.0),
                Complex64::new(0.0, -1.0),
            ],
            actions: vec![false, false, false, true, true, false, true, true],
            modes: vec![
                *action.0, *action.1, *action.0, *action.1, *action.0, *action.1, *action.0,
                *action.1,
            ],
            boundaries: vec![0, 2, 4, 6, 8],
            groups: None,
        }
    }
}

pub fn edge_vertex_to_fermion(inter_op: &EdgeVertexOperator) -> FermionOperator {
    let mut mapped_operator = FermionOperator::zero();

    inter_op.iter().for_each(|term| {
        let mut mapped_term = FermionOperator::one();

        term.iter()
            .for_each(|action| mapped_term.__imatmul__(&map_edge_vertex_to_fermion(action)));

        mapped_term.__imul__(term.coeff);

        mapped_operator.__iadd__(&mapped_term);
    });

    mapped_operator
}

fn map_edge_vertex_to_majorana(action: EdgeAction) -> MajoranaOperator {
    if *action.0 == *action.1 {
        MajoranaOperator {
            coeffs: vec![Complex64::new(0.0, -1.0)],
            modes: vec![2 * *action.0, 2 * *action.0 + 1],
            boundaries: vec![0, 2],
            groups: None,
        }
    } else {
        MajoranaOperator {
            coeffs: vec![Complex64::new(0.0, -1.0)],
            modes: vec![2 * *action.0, 2 * *action.1],
            boundaries: vec![0, 2],
            groups: None,
        }
    }
}

pub fn edge_vertex_to_majorana(inter_op: &EdgeVertexOperator) -> MajoranaOperator {
    let mut mapped_operator = MajoranaOperator::zero();

    inter_op.iter().for_each(|term| {
        let mut mapped_term = MajoranaOperator::one();

        term.iter()
            .for_each(|action| mapped_term.__imatmul__(&map_edge_vertex_to_majorana(action)));

        mapped_term.__imul__(term.coeff);

        mapped_operator.__iadd__(&mapped_term);
    });

    mapped_operator
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_vertex_to_fermion() {
        let inter_op = EdgeVertexOperator {
            coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            left_indices: vec![0, 1],
            right_indices: vec![0, 2],
            boundaries: vec![0, 1, 2],
            groups: None,
        };

        let fer_op = edge_vertex_to_fermion(&inter_op);

        let expected = FermionOperator {
            coeffs: vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(-2.0, 0.0),
                Complex64::new(0.0, -2.0),
                Complex64::new(0.0, -2.0),
                Complex64::new(0.0, -2.0),
                Complex64::new(0.0, -2.0),
            ],
            actions: vec![
                true, false, false, false, false, true, true, false, true, true,
            ],
            modes: vec![0, 0, 1, 2, 1, 2, 1, 2, 1, 2],
            boundaries: vec![0, 0, 2, 4, 6, 8, 10],
            groups: None,
        };

        assert_eq!(fer_op, expected);
    }

    #[test]
    fn test_edge_vertex_to_majorana() {
        let inter_op = EdgeVertexOperator {
            coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            left_indices: vec![0, 1],
            right_indices: vec![0, 2],
            boundaries: vec![0, 1, 2],
            groups: None,
        };

        let maj_op = edge_vertex_to_majorana(&inter_op);

        let expected = MajoranaOperator {
            coeffs: vec![Complex64::new(0.0, -1.0), Complex64::new(0.0, -2.0)],
            modes: vec![0, 1, 2, 4],
            boundaries: vec![0, 2, 4],
            groups: None,
        };

        assert_eq!(maj_op, expected);
    }
}
