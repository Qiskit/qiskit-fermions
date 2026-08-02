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

use crate::operators::{OperatorMacro, OperatorTrait, ScaledTerm};
use num_complex::Complex64;

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
    T: OperatorMacro + OperatorTrait,
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

    // Rescales every term of `op` by `weight`, allocating nothing: a term view owns its coefficient
    // and only borrows the index data. A local `fn` rather than a closure because the borrow of
    // `op` outlives the call, and a closure's return type cannot be generic over that lifetime.
    fn weighted<'a, T: OperatorTrait>(
        op: &'a T,
        weight: Complex64,
    ) -> impl Iterator<Item = T::TermView<'a>> {
        op.iter().map(move |term| term.scaled(weight))
    }

    let half = Complex64::new(0.5, 0.0);

    // The result is the six triple products, each carrying a fixed weight:
    //
    //     ABC - s CBA - BAC/2 + s CAB/2 - ACB/2 + s BCA/2,    s = sign_num = +-1
    //
    // which reproduces both documented forms: `sign = false` gives
    // `(2 ABC + 2 CBA - BAC - CAB - ACB - BCA)/2` and `sign = true` gives
    // `(2 ABC - 2 CBA - BAC + CAB - ACB + BCA)/2`. Mind the `+s BCA/2`: the four half-weighted
    // products form a single sum, so grouping them into two halves in order to subtract one from
    // the other would flip that sign. Folding each weight into its own block removes that hazard
    // rather than merely warning about it.
    //
    // Appending all six blocks in a single pass replaces a chain of nine
    // `__mul__`/`__add__`/`__sub__` calls, each of which copied its left operand - an operand that
    // grew with every step, so early terms were copied over and over. Here every term is copied
    // exactly once. All weights are +-1 or +-1/2, so folding them into the coefficients is exact.
    //
    // The append order is the order the old chain emitted, which is *not* the order in which the
    // products are computed above: `op_cba` comes second, before `op_bac`. Keeping it makes the
    // result's term order - and hence its buffers - identical to the previous formulation rather
    // than merely equivalent. `test_double_commutator_layout` pins this down, and
    // `test_double_commutator_weights_*` pin the weights independently of that order.
    T::from_terms(
        weighted(&op_abc, Complex64::new(1.0, 0.0))
            .chain(weighted(&op_cba, -sign_num))
            .chain(weighted(&op_bac, -half))
            .chain(weighted(&op_cab, sign_num * half))
            .chain(weighted(&op_acb, -half))
            .chain(weighted(&op_bca, sign_num * half)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::edge_vertex_operator::EdgeVertexOperator;
    use crate::operators::fermion_operator::FermionOperator;
    use crate::operators::majorana_operator::MajoranaOperator;
    use crate::operators::transfer_vertex_operator::TransferVertexOperator;

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

    /// Pins down the exact memory layout of the double commutator's result.
    ///
    /// This is a layout pin, not an algebraic claim: it guards the order in which the six triple
    /// products are appended, and the exact order in which their weights are multiplied into the
    /// coefficients. Every other double-commutator test routes through `normal_ordered`,
    /// `simplify` or `equiv`, all of which are insensitive to term order, so none of them can
    /// detect a reordering. Two of the coefficients below differ from their partners in the last
    /// bit, which is real floating-point structure of the weighting order.
    ///
    /// The operands are deliberately non-commuting, non-symmetric, of differing lengths, with
    /// differing creation/annihilation patterns and complex coefficients, so that every one of the
    /// five buffers is actually constrained.
    #[test]
    fn test_double_commutator_layout() {
        // (1 + 0.25i) a^\dagger_0 a_1
        let op_a = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.25)],
            actions: vec![true, false],
            modes: vec![0, 1],
            boundaries: vec![0, 2],
            groups: None,
        };
        // (0.5 - 0.75i) a^\dagger_1 a^\dagger_2
        let op_b = FermionOperator {
            coeffs: vec![Complex64::new(0.5, -0.75)],
            actions: vec![true, true],
            modes: vec![1, 2],
            boundaries: vec![0, 2],
            groups: None,
        };
        // (-0.3 + 1.5i) a_2
        let op_c = FermionOperator {
            coeffs: vec![Complex64::new(-0.3, 1.5)],
            actions: vec![false],
            modes: vec![2],
            boundaries: vec![0, 1],
            groups: None,
        };

        // The index buffers do not depend on `sign`, only the coefficients do.
        let actions = vec![
            true, false, true, true, false, // ABC
            false, true, true, true, false, // CBA
            true, true, true, false, false, // BAC
            false, true, false, true, true, // CAB
            true, false, false, true, true, // ACB
            true, true, false, true, false, // BCA
        ];
        let modes = vec![
            0, 1, 1, 2, 2, // ABC
            2, 1, 2, 0, 1, // CBA
            1, 2, 0, 1, 2, // BAC
            2, 0, 1, 1, 2, // CAB
            0, 1, 2, 1, 2, // ACB
            1, 2, 2, 0, 1, // BCA
        ];
        let boundaries = vec![0, 5, 10, 15, 20, 25, 30];

        let comm = double_commutator(&op_a, &op_b, &op_c, false);
        assert_eq!(
            comm,
            FermionOperator {
                coeffs: vec![
                    Complex64::new(0.73125, 1.21875),
                    Complex64::new(0.73125, 1.21875),
                    Complex64::new(-0.365625, -0.609375),
                    Complex64::new(-0.365625, -0.609375),
                    Complex64::new(-0.36562500000000003, -0.609375),
                    Complex64::new(-0.36562500000000003, -0.609375),
                ],
                actions: actions.clone(),
                modes: modes.clone(),
                boundaries: boundaries.clone(),
                groups: None,
            }
        );

        let comm = double_commutator(&op_a, &op_b, &op_c, true);
        assert_eq!(
            comm,
            FermionOperator {
                coeffs: vec![
                    Complex64::new(0.73125, 1.21875),
                    Complex64::new(-0.73125, -1.21875),
                    Complex64::new(-0.365625, -0.609375),
                    Complex64::new(0.365625, 0.609375),
                    Complex64::new(-0.36562500000000003, -0.609375),
                    Complex64::new(0.36562500000000003, 0.609375),
                ],
                actions,
                modes,
                boundaries,
                groups: None,
            }
        );
    }

    /// Checks the weighted sum term-order-blindly, but sensitively to every weight's sign.
    ///
    /// Builds `ABC - s CBA - BAC/2 + s CAB/2 - ACB/2 + s BCA/2` explicitly from the ten products
    /// and compares with `equiv`, which is insensitive to term order. This is the check that
    /// catches a wrong sign or a wrong weight, complementing the layout pin (which catches a
    /// reordering but would also flag a harmless one). It is generic so that it exercises the
    /// `ScaledTerm` impls of all four operator types.
    fn assert_double_commutator_weights<T>(op_a: &T, op_b: &T, op_c: &T)
    where
        T: OperatorMacro + OperatorTrait + std::fmt::Debug,
    {
        let half = Complex64::new(0.5, 0.0);
        for sign in [false, true] {
            let sign_num = if sign {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(-1.0, 0.0)
            };

            let op_ab = op_a.__matmul__(op_b);
            let op_ba = op_b.__matmul__(op_a);
            let op_ac = op_a.__matmul__(op_c);
            let op_ca = op_c.__matmul__(op_a);

            let expected = op_ab
                .__matmul__(op_c)
                .__sub__(&op_c.__matmul__(&op_ba).__mul__(sign_num))
                .__sub__(&op_ba.__matmul__(op_c).__mul__(half))
                .__add__(&op_c.__matmul__(&op_ab).__mul__(sign_num * half))
                .__sub__(&op_ac.__matmul__(op_b).__mul__(half))
                .__add__(&op_b.__matmul__(&op_ca).__mul__(sign_num * half));

            let actual = double_commutator(op_a, op_b, op_c, sign);
            assert!(
                actual.equiv(&expected, 1e-12),
                "weights disagree for sign={sign}"
            );
            // Guard against a slip to `from_terms_with_groups`: every product comes from `matmul`,
            // which tracks no groups, so neither may the sum.
            assert!(actual.groups().is_none());
        }
    }

    #[test]
    fn test_double_commutator_weights_fermion() {
        let op_a = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.25), Complex64::new(-0.4, 0.0)],
            actions: vec![true, false, true],
            modes: vec![0, 1, 2],
            boundaries: vec![0, 2, 3],
            groups: None,
        };
        let op_b = FermionOperator {
            coeffs: vec![Complex64::new(0.5, -0.75)],
            actions: vec![true, true],
            modes: vec![1, 2],
            boundaries: vec![0, 2],
            groups: None,
        };
        let op_c = FermionOperator {
            coeffs: vec![Complex64::new(-0.3, 1.5), Complex64::new(0.0, 0.6)],
            actions: vec![false, false, true],
            modes: vec![2, 0, 1],
            boundaries: vec![0, 1, 3],
            groups: None,
        };
        assert_double_commutator_weights(&op_a, &op_b, &op_c);
    }

    #[test]
    fn test_double_commutator_weights_majorana() {
        let op_a = MajoranaOperator {
            coeffs: vec![Complex64::new(1.0, 0.25), Complex64::new(-0.4, 0.0)],
            modes: vec![0, 1, 2],
            boundaries: vec![0, 2, 3],
            groups: None,
        };
        let op_b = MajoranaOperator {
            coeffs: vec![Complex64::new(0.5, -0.75)],
            modes: vec![1, 3],
            boundaries: vec![0, 2],
            groups: None,
        };
        let op_c = MajoranaOperator {
            coeffs: vec![Complex64::new(-0.3, 1.5), Complex64::new(0.0, 0.6)],
            modes: vec![2, 0, 1],
            boundaries: vec![0, 1, 3],
            groups: None,
        };
        assert_double_commutator_weights(&op_a, &op_b, &op_c);
    }

    #[test]
    fn test_double_commutator_weights_edge_vertex() {
        let op_a = EdgeVertexOperator {
            coeffs: vec![Complex64::new(1.0, 0.25), Complex64::new(-0.4, 0.0)],
            left_indices: vec![0, 1, 2],
            right_indices: vec![1, 2, 2],
            boundaries: vec![0, 2, 3],
            groups: None,
        };
        let op_b = EdgeVertexOperator {
            coeffs: vec![Complex64::new(0.5, -0.75)],
            left_indices: vec![1, 3],
            right_indices: vec![2, 3],
            boundaries: vec![0, 2],
            groups: None,
        };
        let op_c = EdgeVertexOperator {
            coeffs: vec![Complex64::new(-0.3, 1.5), Complex64::new(0.0, 0.6)],
            left_indices: vec![2, 0, 1],
            right_indices: vec![2, 1, 1],
            boundaries: vec![0, 1, 3],
            groups: None,
        };
        assert_double_commutator_weights(&op_a, &op_b, &op_c);
    }

    #[test]
    fn test_double_commutator_weights_transfer_vertex() {
        let op_a = TransferVertexOperator {
            coeffs: vec![Complex64::new(1.0, 0.25), Complex64::new(-0.4, 0.0)],
            left_indices: vec![0, 1, 2],
            right_indices: vec![1, 2, 2],
            boundaries: vec![0, 2, 3],
            groups: None,
        };
        let op_b = TransferVertexOperator {
            coeffs: vec![Complex64::new(0.5, -0.75)],
            left_indices: vec![1, 3],
            right_indices: vec![2, 3],
            boundaries: vec![0, 2],
            groups: None,
        };
        let op_c = TransferVertexOperator {
            coeffs: vec![Complex64::new(-0.3, 1.5), Complex64::new(0.0, 0.6)],
            left_indices: vec![2, 0, 1],
            right_indices: vec![2, 1, 1],
            boundaries: vec![0, 1, 3],
            groups: None,
        };
        assert_double_commutator_weights(&op_a, &op_b, &op_c);
    }

    /// The result tracks no groups even when the inputs do.
    ///
    /// `matmul` drops groups, so all ten products are already ungrouped; this pins that the
    /// single-pass sum does not reintroduce them by reaching for `from_terms_with_groups`.
    #[test]
    fn test_double_commutator_drops_groups() {
        let grouped = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(0.5, 0.0)],
            actions: vec![true, false, true, false],
            modes: vec![0, 1, 1, 2],
            boundaries: vec![0, 2, 4],
            groups: Some(vec![0, 1]),
        };
        assert!(grouped.groups().is_some());
        for sign in [false, true] {
            let comm = double_commutator(&grouped, &grouped, &grouped, sign);
            assert!(comm.groups().is_none());
        }
    }

    /// A zero operand annihilates every one of the six products, so the sum is empty.
    ///
    /// Worth pinning because an empty result must still carry the leading `0` boundary that every
    /// operator has - a single-pass build that forgot to start from `zero()` would produce an
    /// operator with an empty `boundaries` vector instead.
    #[test]
    fn test_double_commutator_zero_operand() {
        let op = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.5)],
            actions: vec![true, false],
            modes: vec![0, 1],
            boundaries: vec![0, 2],
            groups: None,
        };
        for sign in [false, true] {
            let comm = double_commutator(&op, &FermionOperator::zero(), &op, sign);
            assert_eq!(comm, FermionOperator::zero());
            assert_eq!(comm.boundaries, vec![0]);
        }
    }
}
