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

use crate::operators::OperatorMacro;
use num_complex::Complex64;

// PERF: we should be able to improve the efficiency of all three functions below by writing one
// function that computes the pair of BitAnd products (A&B, B&A) during a single loop rather than
// two separate loops (in the respective BitAnd implementations). Doing so for a generic type T
// will likely require a custom iterator to be implemented by the OperatorMacro. But this should
// directly benefit the commutator, anti_commutator, and various computations inside the
// double_commutator functions.

pub fn commutator<T>(op_a: &T, op_b: &T) -> T
where
    T: OperatorMacro,
{
    op_a.__matmul__(op_b).__sub__(&op_b.__matmul__(op_a))
}

pub fn anti_commutator<T>(op_a: &T, op_b: &T) -> T
where
    T: OperatorMacro,
{
    op_a.__matmul__(op_b).__add__(&op_b.__matmul__(op_a))
}

pub fn double_commutator<T>(op_a: &T, op_b: &T, op_c: &T, sign: bool) -> T
where
    T: OperatorMacro,
{
    let sign_num = if sign {
        Complex64::new(1.0, 0.0)
    } else {
        Complex64::new(-1.0, 0.0)
    };

    let op_ab = op_a.__matmul__(op_b);
    let op_ba = op_b.__matmul__(op_a);
    let op_ac = op_a.__matmul__(op_c);
    let op_ca = op_c.__matmul__(op_a);

    let op_abc = op_ab.__matmul__(op_c);
    let op_cab = op_c.__matmul__(&op_ab);
    let op_bac = op_ba.__matmul__(op_c);
    let op_cba = op_c.__matmul__(&op_ba);
    let op_acb = op_ac.__matmul__(op_b);
    let op_bca = op_b.__matmul__(&op_ca);

    // The bracketed half-weighted part is a single four-term sum: splitting it into two halves and
    // subtracting them would flip the sign of the `op_bca` contribution.
    let inner = op_bac
        .__neg__()
        .__add__(&op_cab.__mul__(sign_num))
        .__sub__(&op_acb)
        .__add__(&op_bca.__mul__(sign_num));

    op_abc
        .__sub__(&op_cba.__mul__(sign_num))
        .__add__(&inner.__mul__(Complex64::new(0.5, 0.0)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::OperatorTrait;
    use crate::operators::fermion_operator::FermionOperator;

    #[test]
    fn test_commutators() {
        let op1 = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            actions: vec![true, false, true, false],
            modes: vec![0, 1, 2, 3],
            boundaries: vec![0, 2, 4],
            groups: None,
        };
        let op2 = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            actions: vec![true, false, true, false],
            modes: vec![1, 0, 3, 2],
            boundaries: vec![0, 2, 4],
            groups: None,
        };
        let comm = commutator(&op1, &op2);

        let expected = FermionOperator {
            coeffs: [1.0, 2.0, 2.0, 4.0, -1.0, -2.0, -2.0, -4.0]
                .iter()
                .map(|c| Complex64::new(*c, 0.0))
                .collect(),
            actions: [true, false].iter().cloned().cycle().take(32).collect(),
            modes: vec![
                0, 1, 1, 0, 2, 3, 1, 0, 0, 1, 3, 2, 2, 3, 3, 2, 1, 0, 0, 1, 3, 2, 0, 1, 1, 0, 2, 3,
                3, 2, 2, 3,
            ],
            boundaries: vec![0, 4, 8, 12, 16, 20, 24, 28, 32],
            groups: None,
        };

        assert_eq!(comm, expected);
    }

    /// Pins down the orientation of the commutator, i.e. that it is `AB - BA` and not `BA - AB`.
    ///
    /// The number operator obeys `[n_0, a_0] = -a_0`, whose sign flips with the operand order. The
    /// other tests here compare against operators that either vanish or are symmetric under that
    /// swap, so they cannot distinguish the two.
    #[test]
    fn test_commutator_orientation() {
        // n_0 = a^\dagger_0 a_0
        let num_op = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            actions: vec![true, false],
            modes: vec![0, 0],
            boundaries: vec![0, 2],
            groups: None,
        };
        // a_0
        let ann_op = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            actions: vec![false],
            modes: vec![0],
            boundaries: vec![0, 1],
            groups: None,
        };

        let comm = commutator(&num_op, &ann_op)
            .normal_ordered(None)
            .simplify(1e-8);

        // -a_0
        let expected = FermionOperator {
            coeffs: vec![Complex64::new(-1.0, 0.0)],
            actions: vec![false],
            modes: vec![0],
            boundaries: vec![0, 1],
            groups: None,
        };

        assert!(comm.equiv(&expected, 1e-8));
    }

    #[test]
    fn test_anti_commutators() {
        let op1 = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            actions: vec![true, false, true, false],
            modes: vec![0, 1, 2, 3],
            boundaries: vec![0, 2, 4],
            groups: None,
        };
        let op2 = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            actions: vec![true, false, true, false],
            modes: vec![1, 0, 3, 2],
            boundaries: vec![0, 2, 4],
            groups: None,
        };
        let comm = anti_commutator(&op1, &op2);

        let expected = FermionOperator {
            coeffs: [1.0, 2.0, 2.0, 4.0, 1.0, 2.0, 2.0, 4.0]
                .iter()
                .map(|c| Complex64::new(*c, 0.0))
                .collect(),
            actions: [true, false].iter().cloned().cycle().take(32).collect(),
            modes: vec![
                0, 1, 1, 0, 2, 3, 1, 0, 0, 1, 3, 2, 2, 3, 3, 2, 1, 0, 0, 1, 3, 2, 0, 1, 1, 0, 2, 3,
                3, 2, 2, 3,
            ],
            boundaries: vec![0, 4, 8, 12, 16, 20, 24, 28, 32],
            groups: None,
        };

        assert_eq!(comm, expected);
    }

    #[test]
    fn test_double_commutators() {
        let op1 = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            actions: vec![true, false],
            modes: vec![0, 0],
            boundaries: vec![0, 2],
            groups: None,
        };
        let op2 = FermionOperator {
            coeffs: vec![Complex64::new(2.0, 0.0)],
            actions: vec![false, true],
            modes: vec![0, 0],
            boundaries: vec![0, 2],
            groups: None,
        };
        let op3 = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.5)],
            actions: vec![true, false, false, true],
            modes: vec![0, 0, 0, 0],
            boundaries: vec![0, 2, 4],
            groups: None,
        };
        let comm = double_commutator(&op1, &op2, &op3, false);
        let normal_ordered = comm.normal_ordered(None);
        let canon = normal_ordered.simplify(1e-8);
        assert_eq!(canon, FermionOperator::zero());
    }
}
