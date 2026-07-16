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

use crate::operators::{OperatorTrait, TermSortKey};

/// Rebuilds `operator` with its terms in a canonical order.
///
/// The terms are sorted lexicographically by [`TermSortKey::sort_key`], which depends only on each
/// term's structure and not on its coefficient. The order is therefore deterministic for a given
/// set of terms regardless of how the operator was assembled. The terms themselves are left
/// untouched — this only reorders them, it does not simplify or normal-order the operator. This is
/// generic over every operator type.
///
/// Any group indices are **not** preserved: the returned operator has `groups` set to `None`, since
/// a canonical reordering does not respect group boundaries.
pub fn canonical_order<OpType>(operator: &OpType) -> OpType
where
    OpType: OperatorTrait,
{
    // We only read `operator` through borrowed term views and build a fresh owned result, so a
    // shared borrow suffices — the caller keeps ownership and need not clone just to reorder.
    let mut terms: Vec<_> = operator.iter().collect();
    terms.sort_by(|a, b| a.sort_key().cmp(&b.sort_key()));
    OpType::from_terms(terms)
}

#[cfg(test)]
mod tests {
    use num_complex::Complex64;

    use super::*;
    use crate::operators::fermion_operator::FermionOperator;
    use crate::operators::majorana_operator::MajoranaOperator;

    #[test]
    fn test_canonical_order_round_trip_fermion() {
        let op = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            actions: vec![true, false, true, true, false, false],
            modes: vec![0, 1, 0, 1, 1, 0],
            boundaries: vec![0, 2, 6],
            groups: None,
        };

        let ordered = canonical_order(&op);

        // The round-trip preserves the operator up to simplification.
        assert!(op.equiv(&ordered, 1e-12));
    }

    #[test]
    fn test_canonical_order_round_trip_majorana() {
        let op = MajoranaOperator {
            coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(3.0, 0.0)],
            modes: vec![0, 1, 2, 3],
            boundaries: vec![0, 2, 4],
            groups: None,
        };

        let ordered = canonical_order(&op);

        assert!(op.equiv(&ordered, 1e-12));
    }

    #[test]
    fn test_canonical_order_sorts_fermion_terms() {
        // Two terms deliberately stored out of canonical order:
        //   term 0: a†_1 a_0  -> key [(true, 1), (false, 0)]
        //   term 1: a†_0 a_1  -> key [(true, 0), (false, 1)]
        // The key is (action, mode) per factor, matching `into_vec`. Both terms lead with a
        // creation operator, so the tie breaks on mode: term 1 (mode 0) precedes term 0 (mode 1).
        let op = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            actions: vec![true, false, true, false],
            modes: vec![1, 0, 0, 1],
            boundaries: vec![0, 2, 4],
            groups: None,
        };

        let ordered = canonical_order(&op);

        // The canonical order preserves the operator's value ...
        assert!(op.equiv(&ordered, 1e-12));
        // ... while reordering the terms so the mode-0-leading term is first, carrying its own
        // coefficient with it.
        let terms: Vec<_> = ordered.iter().collect();
        assert_eq!(terms[0].modes, &[0, 1]);
        assert_eq!(terms[0].actions, &[true, false]);
        assert_eq!(terms[0].coeff, Complex64::new(2.0, 0.0));
        assert_eq!(terms[1].modes, &[1, 0]);
        assert_eq!(terms[1].coeff, Complex64::new(1.0, 0.0));
    }

    #[test]
    fn test_canonical_order_sorts_majorana_terms() {
        // Terms stored as [ (2,3), (0,1) ]; canonical order sorts them to [ (0,1), (2,3) ].
        let op = MajoranaOperator {
            coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            modes: vec![2, 3, 0, 1],
            boundaries: vec![0, 2, 4],
            groups: None,
        };

        let ordered = canonical_order(&op);

        assert!(op.equiv(&ordered, 1e-12));
        let terms: Vec<_> = ordered.iter().collect();
        assert_eq!(terms[0].modes, &[0, 1]);
        assert_eq!(terms[0].coeff, Complex64::new(2.0, 0.0));
        assert_eq!(terms[1].modes, &[2, 3]);
        assert_eq!(terms[1].coeff, Complex64::new(1.0, 0.0));
    }

    #[test]
    fn test_canonical_order_drops_groups() {
        // A canonical reordering does not respect group boundaries, so `groups` is cleared.
        let op = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            actions: vec![true, false, true, false],
            modes: vec![1, 0, 0, 1],
            boundaries: vec![0, 2, 4],
            groups: Some(vec![0, 1]),
        };

        let ordered = canonical_order(&op);

        assert!(ordered.groups.is_none());
    }

    #[test]
    fn test_from_terms_round_trip_fermion() {
        let op = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            actions: vec![true, false, true, true, false, false],
            modes: vec![0, 1, 0, 1, 1, 0],
            boundaries: vec![0, 2, 6],
            groups: None,
        };

        // `from_terms(op.iter())` is a faithful inverse of `iter`.
        let rebuilt = FermionOperator::from_terms(op.iter());
        assert_eq!(op, rebuilt);
    }

    #[test]
    fn test_from_terms_with_groups_round_trip_fermion() {
        let op = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            actions: vec![true, false, true, true, false, false],
            modes: vec![0, 1, 0, 1, 1, 0],
            boundaries: vec![0, 2, 6],
            groups: Some(vec![0, 1]),
        };

        let rebuilt = FermionOperator::from_terms_with_groups(op.iter_with_groups());
        assert_eq!(op, rebuilt);
        assert_eq!(op.groups, rebuilt.groups);
    }
}
