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

use crate::operators::{GroupedTerm, OperatorTrait};

/// Rebuilds `operator` with its terms ordered by group index.
///
/// The terms are sorted by [`GroupedTerm::group`] alone, which makes each group one contiguous run of
/// terms and the group indices non-decreasing. The sort is *stable*, so terms within a group keep
/// their relative order; ordering canonically first and by group second therefore yields a
/// well-defined refinement of the canonical order rather than an arbitrary permutation of it.
///
/// Group indices carry no meaning beyond saying which terms belong together, so this changes only the
/// layout, never the value: the result is [`equiv`](OperatorTrait::equiv) to the input and holds the
/// same terms with the same group index each. What the layout buys is lookup cost. `split_out_groups`
/// scans every term to find the requested groups in general, but on a group-ordered operator each
/// group's terms form one range that can be located by binary search, so the cost of a lookup scales
/// with the number of groups *requested* rather than the number of terms *held*. Hoisting this call
/// out of a loop that repeatedly samples a few groups from a large operator (a randomized product
/// formula, say) is what makes those lookups affordable.
///
/// An operator tracking no groups has nothing to order by; it is returned as an unchanged copy, so
/// that this composes in a pipeline without the caller having to guard. The result tracks no groups
/// either.
pub fn group_order<OpType>(operator: &OpType) -> OpType
where
    OpType: OperatorTrait,
{
    // We only read `operator` through borrowed term views and build a fresh owned result, so a
    // shared borrow suffices — the caller keeps ownership and need not clone just to reorder.
    if operator.has_groups() {
        let mut terms: Vec<_> = operator.iter_with_groups().collect();
        // `sort_by_key` is stable, which is what preserves the within-group order documented above.
        terms.sort_by_key(|term| term.group());
        OpType::from_terms_with_groups(terms)
    } else {
        OpType::from_terms(operator.iter())
    }
}

#[cfg(test)]
mod tests {
    use num_complex::Complex64;

    use super::*;
    use crate::operators::fermion_operator::FermionOperator;
    use crate::operators::majorana_operator::MajoranaOperator;

    #[test]
    fn test_group_order_makes_groups_contiguous() {
        // Groups deliberately interleaved: this is the shape
        // `group_terms_by_electronic_structure` produces, since it assigns indices in order of first
        // encounter, so a group's second term can appear far after its first.
        let op = FermionOperator {
            coeffs: vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
            ],
            actions: vec![true, false, true, false, true, false, true, false],
            modes: vec![0, 1, 2, 3, 1, 0, 3, 2],
            boundaries: vec![0, 2, 4, 6, 8],
            groups: Some(vec![0, 1, 0, 1]),
        };

        let ordered = group_order(&op);

        // The value is untouched ...
        assert!(op.equiv(&ordered, 1e-12));
        // ... while each group becomes one contiguous run.
        assert_eq!(ordered.groups, Some(vec![0, 0, 1, 1]));
    }

    #[test]
    fn test_group_order_is_stable_within_a_group() {
        // Two of the three terms sit in group 0, so nothing may reorder them relative to one another.
        let op = FermionOperator {
            coeffs: vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
            ],
            actions: vec![true, false, true, false, true, false],
            modes: vec![1, 0, 0, 1, 2, 3],
            boundaries: vec![0, 2, 4, 6],
            groups: Some(vec![1, 0, 0]),
        };

        let ordered = group_order(&op);

        assert!(op.equiv(&ordered, 1e-12));
        assert_eq!(ordered.groups, Some(vec![0, 0, 1]));
        let terms: Vec<_> = ordered.iter_with_groups().collect();
        // The two group-0 terms keep the order they were stored in: a†_0 a_1 (coeff 2) before
        // a†_2 a_3 (coeff 3). A non-stable sort would be free to swap them.
        assert_eq!(terms[0].modes, &[0, 1]);
        assert_eq!(terms[0].coeff, Complex64::new(2.0, 0.0));
        assert_eq!(terms[1].modes, &[2, 3]);
        assert_eq!(terms[1].coeff, Complex64::new(3.0, 0.0));
        // ... and the group-1 term lands last, carrying its own coefficient.
        assert_eq!(terms[2].modes, &[1, 0]);
        assert_eq!(terms[2].coeff, Complex64::new(1.0, 0.0));
    }

    #[test]
    fn test_group_order_preserves_the_term_group_association() {
        let op = FermionOperator {
            coeffs: vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
            ],
            actions: vec![true, false, true, false, true, false],
            modes: vec![0, 1, 2, 3, 4, 5],
            boundaries: vec![0, 2, 4, 6],
            groups: Some(vec![2, 0, 1]),
        };

        // Record which group index each term carries before reordering.
        let mut before: Vec<_> = op
            .iter_with_groups()
            .map(|t| (t.modes.to_vec(), t.group))
            .collect();

        let ordered = group_order(&op);

        // Every term still carries the group index it started with; only the positions changed.
        let mut after: Vec<_> = ordered
            .iter_with_groups()
            .map(|t| (t.modes.to_vec(), t.group))
            .collect();
        before.sort();
        after.sort();
        assert_eq!(after, before);
        assert_eq!(ordered.groups, Some(vec![0, 1, 2]));
    }

    #[test]
    fn test_group_order_majorana() {
        let op = MajoranaOperator {
            coeffs: vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
            ],
            modes: vec![0, 1, 2, 3, 4, 5],
            boundaries: vec![0, 2, 4, 6],
            groups: Some(vec![1, 0, 1]),
        };

        let ordered = group_order(&op);

        assert!(op.equiv(&ordered, 1e-12));
        assert_eq!(ordered.groups, Some(vec![0, 1, 1]));
        let terms: Vec<_> = ordered.iter_with_groups().collect();
        assert_eq!(terms[0].modes, &[2, 3]);
        assert_eq!(terms[0].coeff, Complex64::new(2.0, 0.0));
    }

    #[test]
    fn test_group_order_already_ordered_is_a_no_op() {
        let op = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            actions: vec![true, false, true, false],
            modes: vec![0, 1, 2, 3],
            boundaries: vec![0, 2, 4],
            groups: Some(vec![0, 1]),
        };

        let ordered = group_order(&op);

        // Stability makes this bit-identical, not merely equivalent.
        assert_eq!(op, ordered);
        assert_eq!(op.groups, ordered.groups);
    }

    #[test]
    fn test_group_order_ungrouped_returns_unchanged_copy() {
        let op = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            actions: vec![true, false, true, true, false, false],
            modes: vec![0, 1, 0, 1, 1, 0],
            boundaries: vec![0, 2, 6],
            groups: None,
        };

        let ordered = group_order(&op);

        // An ungrouped operator has nothing to order by, so it comes back untouched rather than
        // raising; the result tracks no groups either.
        assert_eq!(op, ordered);
        assert_eq!(ordered.groups, None);
    }
}
