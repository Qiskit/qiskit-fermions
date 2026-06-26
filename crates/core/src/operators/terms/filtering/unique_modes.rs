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

use crate::operators::fermion_operator::FermionOperator;
use std::collections::HashSet;

/// Filters out the terms of an operator that act on too few unique modes.
///
/// Every term that acts on fewer than `min_unique_modes` *unique* fermionic modes is dropped from
/// `op`. The number of unique modes is the cardinality of the set of acted-upon mode indices,
/// irrespective of how many actions a term has. For example, the number operator
/// `a^\dagger_i a_i` acts on a single unique mode (`i`), while the constant term acts on none.
///
/// This is mutating `op` in place. If `op` tracks group indices (see
/// [`FermionOperator::groups`]), the entries of the surviving terms are kept as-is and are *not*
/// re-indexed; group indices may therefore become non-contiguous if all terms of a group are
/// removed.
pub fn filter_terms_by_num_unique_modes(op: &mut FermionOperator, min_unique_modes: u32) {
    // We compute the new flat storage for the surviving terms into fresh vectors and only assign
    // them back at the end, because `op.iter()` borrows `op` immutably for the duration of the
    // loop.
    let mut coeffs = Vec::with_capacity(op.coeffs.len());
    let mut actions = Vec::with_capacity(op.actions.len());
    let mut modes = Vec::with_capacity(op.modes.len());
    let mut boundaries = vec![0usize];
    let mut groups = op
        .groups
        .as_ref()
        .map(|_| Vec::with_capacity(op.coeffs.len()));

    for (idx, term) in op.iter().enumerate() {
        let num_unique = term.modes.iter().collect::<HashSet<_>>().len() as u32;
        if num_unique < min_unique_modes {
            continue;
        }
        coeffs.push(term.coeff);
        actions.extend_from_slice(term.actions);
        modes.extend_from_slice(term.modes);
        boundaries.push(modes.len());
        if let (Some(dst), Some(src)) = (groups.as_mut(), op.groups.as_ref()) {
            dst.push(src[idx]);
        }
    }

    op.coeffs = coeffs;
    op.actions = actions;
    op.modes = modes;
    op.boundaries = boundaries;
    op.groups = groups;
}

#[cfg(test)]
mod tests {
    use num_complex::Complex64;

    use super::*;

    use crate::operators::OperatorTrait;
    use crate::operators::library::fcidump::FCIDump;

    /// Builds an operator holding (in order): a constant term, a number operator `a^\dagger_0 a_0`,
    /// and a genuine 2-unique-mode hopping term `a^\dagger_0 a_1`.
    fn build_mixed_op() -> FermionOperator {
        FermionOperator {
            coeffs: vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
            ],
            actions: vec![true, false, true, false],
            modes: vec![0, 0, 0, 1],
            boundaries: vec![0, 0, 2, 4],
            groups: None,
        }
    }

    #[test]
    fn test_filter_drops_constant_and_number_operators() {
        let mut op = build_mixed_op();
        filter_terms_by_num_unique_modes(&mut op, 2);

        let expected = FermionOperator {
            coeffs: vec![Complex64::new(3.0, 0.0)],
            actions: vec![true, false],
            modes: vec![0, 1],
            boundaries: vec![0, 2],
            groups: None,
        };
        assert_eq!(op, expected);
    }

    #[test]
    fn test_filter_zero_threshold_keeps_everything() {
        let mut op = build_mixed_op();
        let expected = build_mixed_op();
        filter_terms_by_num_unique_modes(&mut op, 0);
        assert_eq!(op, expected);
    }

    #[test]
    fn test_filter_large_threshold_empties() {
        let mut op = build_mixed_op();
        filter_terms_by_num_unique_modes(&mut op, 3);
        assert_eq!(op, FermionOperator::zero());
    }

    #[test]
    fn test_filter_keeps_group_indices_without_reindexing() {
        // Three terms in groups 0, 1, 2. Filtering with threshold 2 drops the first two terms
        // (constant + number operator), which removes groups 0 and 1 entirely. The surviving term
        // must keep its original group index (2) rather than being re-indexed to 0.
        let mut op = build_mixed_op();
        op.groups = Some(vec![0, 1, 2]);

        filter_terms_by_num_unique_modes(&mut op, 2);

        assert_eq!(op.groups, Some(vec![2]));
        // The non-contiguous group indices are the explicitly chosen behavior: `num_groups` is
        // still derived as `max + 1`.
        assert_eq!(op.num_groups(), Some(3));
    }

    #[test]
    fn test_filter_none_groups_stay_none() {
        let mut op = build_mixed_op();
        filter_terms_by_num_unique_modes(&mut op, 2);
        assert!(op.groups.is_none());
    }

    #[test]
    fn test_filter_electronic_structure_hamiltonian() {
        let file_path = String::from("../../tests/h2.fcidump");
        let fcidump = FCIDump::from_file(file_path);

        let mut op = FermionOperator::from(&fcidump)
            .normal_ordered(None)
            .simplify(1e-16);
        let num_terms_before = op.coeffs.len();

        filter_terms_by_num_unique_modes(&mut op, 2);

        // every surviving term must act on at least two unique modes
        for term in op.iter() {
            let num_unique = term.modes.iter().collect::<HashSet<_>>().len();
            assert!(
                num_unique >= 2,
                "a term acting on fewer than 2 unique modes survived filtering"
            );
        }
        // the constant offset and the number operators must have been removed
        assert!(
            op.coeffs.len() < num_terms_before,
            "filtering should have removed at least the constant and number-operator terms"
        );
    }
}
