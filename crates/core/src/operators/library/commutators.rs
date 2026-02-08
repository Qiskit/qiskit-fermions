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
    op_a.__and__(op_b).__sub__(&op_b.__and__(op_a))
}

pub fn anti_commutator<T>(op_a: &T, op_b: &T) -> T
where
    T: OperatorMacro,
{
    op_a.__and__(op_b).__add__(&op_b.__and__(op_a))
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

    let op_ab = op_a.__and__(op_b);
    let op_ba = op_b.__and__(op_a);
    let op_ac = op_a.__and__(op_c);
    let op_ca = op_c.__and__(op_a);

    let op_abc = op_ab.__and__(op_c);
    let op_cab = op_c.__and__(&op_ab);
    let op_bac = op_ba.__and__(op_c);
    let op_cba = op_c.__and__(&op_ba);
    let op_acb = op_ac.__and__(op_b);
    let op_bca = op_b.__and__(&op_ca);

    let term1 = op_abc.__sub__(&op_cba.__mul__(sign_num));
    let term2 = op_bac.__neg__().__add__(&op_cab.__mul__(sign_num));
    let term3 = op_acb.__add__(&op_bca.__mul__(sign_num));
    let diff = term2.__sub__(&term3);

    term1.__add__(&diff.__mul__(Complex64::new(0.5, 0.0)))
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
            indices: vec![0, 1, 2, 3],
            boundaries: vec![0, 2, 4],
        };
        let op2 = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            actions: vec![true, false, true, false],
            indices: vec![1, 0, 3, 2],
            boundaries: vec![0, 2, 4],
        };
        let comm = commutator(&op1, &op2);

        let expected = FermionOperator {
            coeffs: vec![1.0, 2.0, 2.0, 4.0, -1.0, -2.0, -2.0, -4.0]
                .iter()
                .map(|c| Complex64::new(*c, 0.0))
                .collect(),
            actions: vec![true, false].iter().cloned().cycle().take(32).collect(),
            indices: vec![
                1, 0, 0, 1, 3, 2, 0, 1, 1, 0, 2, 3, 3, 2, 2, 3, 0, 1, 1, 0, 2, 3, 1, 0, 0, 1, 3, 2,
                2, 3, 3, 2,
            ],
            boundaries: vec![0, 4, 8, 12, 16, 20, 24, 28, 32],
        };

        assert_eq!(comm, expected);
    }

    #[test]
    fn test_anti_commutators() {
        let op1 = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            actions: vec![true, false, true, false],
            indices: vec![0, 1, 2, 3],
            boundaries: vec![0, 2, 4],
        };
        let op2 = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            actions: vec![true, false, true, false],
            indices: vec![1, 0, 3, 2],
            boundaries: vec![0, 2, 4],
        };
        let comm = anti_commutator(&op1, &op2);

        let expected = FermionOperator {
            coeffs: vec![1.0, 2.0, 2.0, 4.0, 1.0, 2.0, 2.0, 4.0]
                .iter()
                .map(|c| Complex64::new(*c, 0.0))
                .collect(),
            actions: vec![true, false].iter().cloned().cycle().take(32).collect(),
            indices: vec![
                1, 0, 0, 1, 3, 2, 0, 1, 1, 0, 2, 3, 3, 2, 2, 3, 0, 1, 1, 0, 2, 3, 1, 0, 0, 1, 3, 2,
                2, 3, 3, 2,
            ],
            boundaries: vec![0, 4, 8, 12, 16, 20, 24, 28, 32],
        };

        assert_eq!(comm, expected);
    }

    #[test]
    fn test_double_commutators() {
        let op1 = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            actions: vec![true, false],
            indices: vec![0, 0],
            boundaries: vec![0, 2],
        };
        let op2 = FermionOperator {
            coeffs: vec![Complex64::new(2.0, 0.0)],
            actions: vec![false, true],
            indices: vec![0, 0],
            boundaries: vec![0, 2],
        };
        let op3 = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.5)],
            actions: vec![true, false, false, true],
            indices: vec![0, 0, 0, 0],
            boundaries: vec![0, 2, 4],
        };
        let comm = double_commutator(&op1, &op2, &op3, false);
        let normal_ordered = comm.normal_ordered();
        let canon = normal_ordered.simplify(1e-8);
        assert_eq!(canon, FermionOperator::zero());
    }
}
