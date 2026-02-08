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
use crate::operators::fermion_operator::{FermionAction, FermionOperator};
use crate::operators::majorana_operator::{MajoranaAction, MajoranaOperator};

fn map_fermion_action(action: FermionAction) -> MajoranaOperator {
    let im = if *action.0 { -0.5 } else { 0.5 };

    MajoranaOperator {
        coeffs: vec![Complex64::new(0.5, 0.0), Complex64::new(0.0, im)],
        modes: vec![*action.1 * 2, *action.1 * 2 + 1],
        boundaries: vec![0, 1, 2],
    }
}

pub fn fermion_to_majorana(fer_op: &FermionOperator) -> MajoranaOperator {
    let mut mapped_operator = MajoranaOperator::zero();

    fer_op.iter().for_each(|term| {
        let mut mapped_term = MajoranaOperator::one();

        term.iter()
            .for_each(|action| mapped_term.__iand__(&map_fermion_action(action)));

        mapped_term.__imul__(term.coeff);

        mapped_operator.__iadd__(&mapped_term);
    });

    mapped_operator
}

fn map_majorana_action(mode: MajoranaAction) -> FermionOperator {
    let idx = mode / 2;
    let im = if mode.is_multiple_of(2) { 1.0 } else { -1.0 };

    FermionOperator {
        coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, im)],
        actions: vec![true, false],
        indices: vec![idx, idx],
        boundaries: vec![0, 1, 2],
    }
}

pub fn majorana_to_fermion(maj_op: &MajoranaOperator) -> FermionOperator {
    let mut mapped_operator = FermionOperator::zero();

    maj_op.iter().for_each(|term| {
        let mut mapped_term = FermionOperator::one();

        term.iter()
            .for_each(|action| mapped_term.__iand__(&map_majorana_action(action)));

        mapped_term.__imul__(term.coeff);

        mapped_operator.__iadd__(&mapped_term);
    });

    mapped_operator
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fermion_to_majorana_1() {
        let fer_op = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            actions: vec![true],
            indices: vec![0],
            boundaries: vec![0, 1],
        };

        let maj_op = fermion_to_majorana(&fer_op);

        let expected = MajoranaOperator {
            coeffs: vec![Complex64::new(0.5, 0.0), Complex64::new(0.0, -0.5)],
            modes: vec![0, 1],
            boundaries: vec![0, 1, 2],
        };

        assert_eq!(maj_op, expected);
    }

    #[test]
    fn test_fermion_to_majorana_2() {
        let fer_op = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            actions: vec![false],
            indices: vec![0],
            boundaries: vec![0, 1],
        };

        let maj_op = fermion_to_majorana(&fer_op);

        let expected = MajoranaOperator {
            coeffs: vec![Complex64::new(0.5, 0.0), Complex64::new(0.0, 0.5)],
            modes: vec![0, 1],
            boundaries: vec![0, 1, 2],
        };

        assert_eq!(maj_op, expected);
    }

    #[test]
    fn test_fermion_to_majorana_3() {
        let fer_op = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            actions: vec![true, false],
            indices: vec![0, 0],
            boundaries: vec![0, 2],
        };

        let maj_op = fermion_to_majorana(&fer_op);
        let normal = maj_op.normal_ordered(true);
        let canon = normal.simplify(1e-8);

        let expected = MajoranaOperator {
            coeffs: vec![Complex64::new(0.5, 0.0), Complex64::new(0.0, 0.5)],
            modes: vec![1, 0],
            boundaries: vec![0, 0, 2],
        };

        assert!(canon.equiv(&expected, 1e-10));
    }

    #[test]
    fn test_fermion_to_majorana_4() {
        let fer_op = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
            actions: vec![true, false, true, false],
            indices: vec![0, 1, 1, 0],
            boundaries: vec![0, 2, 4],
        };

        let maj_op = fermion_to_majorana(&fer_op);
        let normal = maj_op.normal_ordered(true);
        let canon = normal.simplify(1e-8);

        let expected = MajoranaOperator {
            coeffs: vec![Complex64::new(0.0, 0.5), Complex64::new(0.0, -0.5)],
            modes: vec![3, 0, 2, 1],
            boundaries: vec![0, 2, 4],
        };

        assert!(canon.equiv(&expected, 1e-10));
    }

    #[test]
    fn test_majorana_to_fermion_1() {
        let maj_op = MajoranaOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            modes: vec![0],
            boundaries: vec![0, 1],
        };

        let fer_op = majorana_to_fermion(&maj_op);

        let expected = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)],
            actions: vec![true, false],
            indices: vec![0, 0],
            boundaries: vec![0, 1, 2],
        };

        assert_eq!(fer_op, expected);
    }

    #[test]
    fn test_majorana_to_fermion_2() {
        let maj_op = MajoranaOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            modes: vec![1],
            boundaries: vec![0, 1],
        };

        let fer_op = majorana_to_fermion(&maj_op);

        let expected = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, -1.0)],
            actions: vec![true, false],
            indices: vec![0, 0],
            boundaries: vec![0, 1, 2],
        };

        assert_eq!(fer_op, expected);
    }

    #[test]
    fn test_majorana_to_fermion_3() {
        let maj_op = MajoranaOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            modes: vec![0, 1],
            boundaries: vec![0, 2],
        };

        let fer_op = majorana_to_fermion(&maj_op);
        let normal = fer_op.normal_ordered();
        let canon = normal.simplify(1e-8);

        let expected = FermionOperator {
            coeffs: vec![Complex64::new(0.0, -1.0), Complex64::new(0.0, 2.0)],
            actions: vec![true, false],
            indices: vec![0, 0],
            boundaries: vec![0, 0, 2],
        };

        assert!(canon.equiv(&expected, 1e-10));
    }

    #[test]
    fn test_majorana_to_fermion_4() {
        let maj_op = MajoranaOperator {
            coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
            modes: vec![0, 3, 2, 1],
            boundaries: vec![0, 2, 4],
        };

        let fer_op = majorana_to_fermion(&maj_op);
        let normal = fer_op.normal_ordered();
        let canon = normal.simplify(1e-8);

        let expected = FermionOperator {
            coeffs: vec![Complex64::new(0.0, 2.0), Complex64::new(0.0, 2.0)],
            actions: vec![true, false, true, false],
            indices: vec![0, 1, 1, 0],
            boundaries: vec![0, 2, 4],
        };

        assert!(canon.equiv(&expected, 1e-10));
    }
}
