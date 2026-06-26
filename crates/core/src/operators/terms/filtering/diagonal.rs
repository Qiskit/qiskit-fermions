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

use crate::operators::fermion_operator::{FermionOperator, FermionOperatorTermView};
use crate::operators::terms::filtering::retain_terms;
use std::iter::zip;

/// Returns whether a *normal-ordered* term is diagonal in the occupation-number basis.
///
/// A term is diagonal exactly when it is a product of number operators, i.e. every acted-upon mode
/// appears once as a creation and once as an annihilation. For a normal-ordered term (all creations
/// preceding all annihilations) this is equivalent to the multiset of creation modes being equal to
/// the multiset of annihilation modes.
fn _is_diagonal(term: FermionOperatorTermView) -> bool {
    let mut creations: Vec<u32> = zip(term.actions, term.modes)
        .filter(|(action, _)| **action)
        .map(|(_, mode)| *mode)
        .collect();
    let mut annihilations: Vec<u32> = zip(term.actions, term.modes)
        .filter(|(action, _)| !**action)
        .map(|(_, mode)| *mode)
        .collect();
    creations.sort_unstable();
    annihilations.sort_unstable();
    creations == annihilations
}

/// Filters out the terms of an operator that are diagonal in the occupation-number basis.
///
/// Every term that is a product of number operators (`a^\dagger_i a_i`) is dropped from `op`. This
/// includes the constant term (a product of zero number operators), single number operators, as
/// well as higher-order products such as `n_i n_j = a^\dagger_i a^\dagger_j a_j a_i`. Such terms
/// are diagonal in the occupation-number basis, so their time evolution does not affect the
/// sampled bitstrings (it only introduces global phases and single-qubit Z rotations).
///
/// # Assumptions
///
/// The terms of `op` are assumed to be *normal-ordered*! This is *not* being verified. See
/// [`FermionOperator::normal_ordered`] for how to obtain an operator of that form.
///
/// This is mutating `op` in place. If `op` tracks group indices (see
/// [`FermionOperator::groups`]), surviving terms retain their relative grouping but the group
/// indices are reassigned to a contiguous range starting from 0; callers must therefore *not* rely
/// on the specific group index of any term being preserved across a call to this function.
pub fn filter_diagonal_terms(op: &mut FermionOperator) {
    retain_terms(op, |term| !_is_diagonal(term));
}

#[cfg(test)]
mod tests {
    use num_complex::Complex64;

    use super::*;

    use crate::operators::OperatorTrait;
    use crate::operators::library::fcidump::FCIDump;

    #[test]
    fn test_filter_drops_constant_number_and_products() {
        // Terms (in order):
        //   - constant                              (diagonal -> drop)
        //   - n_0 = a†_0 a_0                         (diagonal -> drop)
        //   - n_0 n_1 = a†_0 a†_1 a_1 a_0            (diagonal -> drop)
        //   - a†_0 a_1                               (off-diagonal -> keep)
        let mut op = FermionOperator {
            coeffs: vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
            ],
            actions: vec![true, false, true, true, false, false, true, false],
            modes: vec![0, 0, 0, 1, 1, 0, 0, 1],
            boundaries: vec![0, 0, 2, 6, 8],
            groups: None,
        };

        filter_diagonal_terms(&mut op);

        let expected = FermionOperator {
            coeffs: vec![Complex64::new(4.0, 0.0)],
            actions: vec![true, false],
            modes: vec![0, 1],
            boundaries: vec![0, 2],
            groups: None,
        };
        assert_eq!(op, expected);
    }

    #[test]
    fn test_filter_keeps_off_diagonal_hopping() {
        // A single off-diagonal hopping term must be kept untouched.
        let mut op = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            actions: vec![true, false],
            modes: vec![0, 1],
            boundaries: vec![0, 2],
            groups: None,
        };
        let expected = op.clone();
        filter_diagonal_terms(&mut op);
        assert_eq!(op, expected);
    }

    #[test]
    fn test_filter_none_groups_stay_none() {
        let mut op = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            actions: vec![true, false, true, false],
            modes: vec![0, 0, 0, 1],
            boundaries: vec![0, 2, 4],
            groups: None,
        };
        filter_diagonal_terms(&mut op);
        assert!(op.groups.is_none());
    }

    #[test]
    fn test_filter_reindexes_groups() {
        // Two off-diagonal hopping terms sharing group 5, with a diagonal number operator in
        // group 2 in between. The number operator is dropped; the survivors must remain grouped
        // together and be re-indexed to a contiguous index (0).
        let mut op = FermionOperator {
            coeffs: vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
            ],
            actions: vec![true, false, true, false, true, false],
            modes: vec![0, 1, 0, 0, 1, 0],
            boundaries: vec![0, 2, 4, 6],
            groups: Some(vec![5, 2, 5]),
        };

        filter_diagonal_terms(&mut op);

        assert_eq!(op.groups, Some(vec![0, 0]));
        assert_eq!(op.num_groups(), Some(1));
    }

    #[test]
    fn test_filter_electronic_structure_hamiltonian() {
        let file_path = String::from("../../tests/h2.fcidump");
        let fcidump = FCIDump::from_file(file_path);

        let mut op = FermionOperator::from(&fcidump)
            .normal_ordered(None)
            .simplify(1e-16);
        let num_terms_before = op.coeffs.len();

        filter_diagonal_terms(&mut op);

        // every surviving term must be off-diagonal
        for term in op.iter() {
            assert!(!_is_diagonal(term), "a diagonal term survived filtering");
        }
        // the constant offset and the number operators must have been removed
        assert!(
            op.coeffs.len() < num_terms_before,
            "filtering should have removed at least the constant and number-operator terms"
        );
    }
}
