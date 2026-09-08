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

//! Analysis of an existing grouping.
//!
//! Where the sibling modules *assign* group indices, these routines only *read* them. Group indices
//! carry no intrinsic meaning (the array merely says which terms belong together, never why), so
//! nothing here reports whether a grouping is "correct". Each routine answers one narrow, stated
//! question about a grouping, leaving the interpretation to the caller.

use num_complex::ComplexFloat;

use crate::operators::{GroupedTerm, OperatorTrait};

/// Returns the mean absolute coefficient magnitude of each group, or `None` if the operator tracks
/// no groups.
///
/// The `i`-th entry is the sum of `|coeff|` over the terms in group `i`, divided by the number of
/// terms in that group. This is the sampling weight of a randomized product formula (for example,
/// qDRIFT) that draws whole groups rather than individual terms: it is the magnitude of one *atomic*
/// group, which is the relevant scale because grouping is what makes each sampled piece Hermitian
/// (and hence its time evolution unitary) in the first place.
///
/// Computing this natively reduces the per-term coefficients and group indices down to one value per
/// group in a single pass, so a caller across an FFI boundary receives only
/// [`num_groups`](OperatorTrait::num_groups) values instead of two arrays of one value per
/// (ungrouped) term that it would have to reduce itself.
///
/// A group index that no term carries yields a weight of `0.0`: the index range is dense by
/// construction (see [`num_groups`](OperatorTrait::num_groups)), but nothing enforces that, and a
/// `0.0` weight keeps such a group out of the sample rather than poisoning every weight with a
/// `NaN`.
pub fn group_coeff_means<OpType>(op: &OpType) -> Option<Vec<f64>>
where
    OpType: OperatorTrait,
{
    let groups = op.groups()?;
    let mut sums = vec![0.0; op.num_groups()? as usize];
    let mut counts = vec![0_u32; sums.len()];
    for (&group, coeff) in groups.iter().zip(op.coeffs()) {
        sums[group as usize] += coeff.norm();
        counts[group as usize] += 1;
    }
    for (sum, &count) in sums.iter_mut().zip(counts.iter()) {
        if count > 0 {
            *sum /= f64::from(count);
        }
    }
    Some(sums)
}

/// Returns, for each group, whether the operator formed by that group's terms is Hermitian, or
/// `None` if the operator tracks no groups.
///
/// Groups prescribe no meaning of their own; this checks one *common* convention, namely that each
/// group is separately Hermitian. That is the property a randomized product formula relies on when
/// it samples whole groups, since only a Hermitian group has a unitary time evolution.
///
/// An empty group counts as Hermitian, because the zero operator is.
///
/// # Guarantee
///
/// This inherits the one-sided guarantee of [`is_hermitian`](OperatorTrait::is_hermitian): a `true`
/// entry is always reliable, while a `false` entry is reliable only for operator types whose normal
/// form is a genuine canonical form. For the others the check is *conservative* and can report
/// `false` for a group that is in fact Hermitian.
pub fn groups_are_hermitian<OpType>(op: &OpType, atol: f64) -> Option<Vec<bool>>
where
    OpType: OperatorTrait,
{
    let num_groups = op.num_groups()? as usize;
    // `(0..n).map(...)` rather than `vec![Vec::new(); n]`: the latter requires
    // `GroupTermView: Clone`, which this generic body cannot see even though every concrete view is
    // `Copy`.
    let mut buckets: Vec<Vec<OpType::GroupTermView<'_>>> =
        (0..num_groups).map(|_| Vec::new()).collect();
    // Bucketing in a single pass keeps this linear in the number of terms. Filtering the terms once
    // per group would instead be O(num_groups * num_terms), which degrades to quadratic for an
    // operator whose groups are mostly singletons.
    for term in op.iter_with_groups() {
        buckets[term.group() as usize].push(term);
    }
    Some(
        buckets
            .into_iter()
            .map(|bucket| OpType::from_terms_with_groups(bucket).is_hermitian(atol))
            .collect(),
    )
}

/// Returns, for each group, whether all of its coefficients are numerically equal, or `None` if the
/// operator tracks no groups.
///
/// With `abs`, the coefficient *magnitudes* are compared, which is the assumption that
/// [`group_coeff_means`] makes when it averages `|coeff|`. Without it, the coefficients must match
/// exactly. Note that a Hermitian group may legitimately fail the stricter form, since a conjugate
/// pair with complex coefficients has equal magnitudes but unequal coefficients.
///
/// An empty or single-term group is trivially uniform.
///
/// This is *not* a weaker form of [`groups_are_hermitian`]: neither implies the other. Uniform
/// coefficients do not make a group Hermitian, and a Hermitian group may mix magnitudes.
pub fn groups_have_uniform_coeffs<OpType>(op: &OpType, atol: f64, abs: bool) -> Option<Vec<bool>>
where
    OpType: OperatorTrait,
{
    let groups = op.groups()?;
    let num_groups = op.num_groups()? as usize;
    let mut uniform = vec![true; num_groups];
    // The first coefficient seen for a group is the reference every later one is compared against.
    let mut references = vec![None; num_groups];
    for (&group, &coeff) in groups.iter().zip(op.coeffs()) {
        let idx = group as usize;
        match references[idx] {
            None => references[idx] = Some(coeff),
            Some(reference) => {
                let deviates = if abs {
                    (coeff.abs() - reference.abs()).abs() > atol
                } else {
                    (coeff - reference).abs() > atol
                };
                if deviates {
                    uniform[idx] = false;
                }
            }
        }
    }
    Some(uniform)
}

#[cfg(test)]
mod tests {
    use num_complex::Complex64;

    use super::*;

    use crate::operators::ScaledTerm;
    use crate::operators::edge_vertex_operator::EdgeVertexOperator;
    use crate::operators::fermion_operator::FermionOperator;
    use crate::operators::majorana_operator::MajoranaOperator;
    use crate::operators::transfer_vertex_operator::TransferVertexOperator;

    /// Builds `coeffs[0] * a†_0 a_1 + coeffs[1] * a†_1 a_0 + coeffs[2] * a†_2 a_3`.
    ///
    /// The first two terms are each other's adjoint, so they form a Hermitian pair when they share a
    /// group and carry equal coefficients. The third is unpaired and is never Hermitian on its own.
    fn build_op(coeffs: Vec<Complex64>, groups: Option<Vec<u32>>) -> FermionOperator {
        FermionOperator {
            coeffs,
            actions: vec![true, false, true, false, true, false],
            modes: vec![0, 1, 1, 0, 2, 3],
            boundaries: vec![0, 2, 4, 6],
            groups,
        }
    }

    #[test]
    fn test_group_coeff_means() {
        let mut op = build_op(
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(-3.0, 4.0),
                Complex64::new(2.0, 0.0),
            ],
            None,
        );

        // an ungrouped operator has no per-group weights to report
        assert!(group_coeff_means(&op).is_none());

        op.groups = Some(vec![0, 0, 1]);

        // group 0 averages |1.0| and |-3.0 + 4.0j| = 5.0, that is (1.0 + 5.0) / 2; group 1 holds
        // the single term |2.0|. Note that the magnitude, not the real part, is what is averaged.
        assert_eq!(group_coeff_means(&op), Some(vec![3.0, 2.0]));

        // a grouped operator holding no terms has no groups to weight either
        let mut zero = FermionOperator::zero();
        zero.groups = Some(vec![]);
        assert_eq!(group_coeff_means(&zero), Some(vec![]));

        // group 1 is carried by no term at all, so it weighs 0.0 rather than NaN-ing the sample
        op.groups = Some(vec![0, 0, 2]);
        assert_eq!(group_coeff_means(&op), Some(vec![3.0, 0.0, 2.0]));
    }

    #[test]
    fn test_groups_are_hermitian() {
        let one = Complex64::new(1.0, 0.0);

        // an ungrouped operator has no groups to check
        assert!(groups_are_hermitian(&build_op(vec![one; 3], None), 1e-8).is_none());

        // a†_0 a_1 and a†_1 a_0 with equal coefficients form a Hermitian pair, while the third term
        // is on its own and is not Hermitian.
        let op = build_op(vec![one, one, one], Some(vec![0, 0, 1]));
        assert_eq!(groups_are_hermitian(&op, 1e-8), Some(vec![true, false]));

        // a group index carried by no term holds the zero operator, which is Hermitian
        let sparse = build_op(vec![one, one, one], Some(vec![0, 0, 2]));
        assert_eq!(
            groups_are_hermitian(&sparse, 1e-8),
            Some(vec![true, true, false])
        );
    }

    #[test]
    fn test_hermiticity_and_uniformity_are_independent() {
        // Neither check implies the other, in either direction.
        let one = Complex64::new(1.0, 0.0);
        let five = Complex64::new(5.0, 0.0);
        let imag = Complex64::new(0.0, 1.0);

        // Uniform coefficients, but NOT Hermitian: the two terms are not each other's adjoint.
        let uniform_not_hermitian = FermionOperator {
            coeffs: vec![one, one],
            actions: vec![true, false, true, false],
            modes: vec![0, 1, 2, 3],
            boundaries: vec![0, 2, 4],
            groups: Some(vec![0, 0]),
        };
        assert_eq!(
            groups_are_hermitian(&uniform_not_hermitian, 1e-8),
            Some(vec![false])
        );
        assert_eq!(
            groups_have_uniform_coeffs(&uniform_not_hermitian, 1e-8, true),
            Some(vec![true])
        );

        // Hermitian, but with mixed magnitudes: two conjugate pairs at different scales.
        let hermitian_mixed = FermionOperator {
            coeffs: vec![one, one, five, five],
            actions: vec![true, false, true, false, true, false, true, false],
            modes: vec![0, 1, 1, 0, 2, 3, 3, 2],
            boundaries: vec![0, 2, 4, 6, 8],
            groups: Some(vec![0, 0, 0, 0]),
        };
        assert_eq!(
            groups_are_hermitian(&hermitian_mixed, 1e-8),
            Some(vec![true])
        );
        assert_eq!(
            groups_have_uniform_coeffs(&hermitian_mixed, 1e-8, true),
            Some(vec![false])
        );

        // A Hermitian conjugate pair with imaginary coefficients has equal magnitudes but unequal
        // coefficients, so only the `abs` form recognises it as uniform.
        let conjugate_pair = FermionOperator {
            coeffs: vec![imag, -imag],
            actions: vec![true, false, true, false],
            modes: vec![0, 1, 1, 0],
            boundaries: vec![0, 2, 4],
            groups: Some(vec![0, 0]),
        };
        assert_eq!(
            groups_are_hermitian(&conjugate_pair, 1e-8),
            Some(vec![true])
        );
        assert_eq!(
            groups_have_uniform_coeffs(&conjugate_pair, 1e-8, true),
            Some(vec![true])
        );
        assert_eq!(
            groups_have_uniform_coeffs(&conjugate_pair, 1e-8, false),
            Some(vec![false])
        );
    }

    #[test]
    fn test_groups_have_uniform_coeffs() {
        let one = Complex64::new(1.0, 0.0);

        // an ungrouped operator has no groups to check
        assert!(groups_have_uniform_coeffs(&build_op(vec![one; 3], None), 1e-8, true).is_none());

        let op = build_op(
            vec![one, Complex64::new(-1.0, 0.0), Complex64::new(2.0, 0.0)],
            Some(vec![0, 0, 1]),
        );

        // group 0 mixes +1 and -1: equal in magnitude, but not equal as coefficients. Group 1 holds
        // a single term and is trivially uniform either way.
        assert_eq!(
            groups_have_uniform_coeffs(&op, 1e-8, true),
            Some(vec![true, true])
        );
        assert_eq!(
            groups_have_uniform_coeffs(&op, 1e-8, false),
            Some(vec![false, true])
        );

        // the tolerance is honoured
        let nearly = build_op(
            vec![one, Complex64::new(1.00001, 0.0), one],
            Some(vec![0, 0, 1]),
        );
        assert_eq!(
            groups_have_uniform_coeffs(&nearly, 1e-8, false),
            Some(vec![false, true])
        );
        assert_eq!(
            groups_have_uniform_coeffs(&nearly, 1e-4, false),
            Some(vec![true, true])
        );

        // an empty group is trivially uniform
        let mut zero = FermionOperator::zero();
        zero.groups = Some(vec![]);
        assert_eq!(groups_have_uniform_coeffs(&zero, 1e-8, true), Some(vec![]));
    }

    /// Asserts the contract that holds for *every* operator type, through the generic bound.
    ///
    /// The bound is the point of this helper: these routines are generic over `OperatorTrait`, so
    /// they can only compile if the group index is reachable through
    /// [`GroupedTerm`](crate::operators::GroupedTerm) rather than only on the concrete view structs.
    /// `grouped` must be the multiplicative identity carrying a single group index, which each
    /// operator type spells out itself because the trait exposes no generic way to assign groups.
    fn assert_analysis_contract<OpType>(grouped: OpType)
    where
        OpType: OperatorTrait,
    {
        // An operator that tracks no groups reports nothing at all.
        let ungrouped = OpType::one();
        assert!(group_coeff_means(&ungrouped).is_none());
        assert!(groups_are_hermitian(&ungrouped, 1e-8).is_none());
        assert!(groups_have_uniform_coeffs(&ungrouped, 1e-8, true).is_none());

        // The multiplicative identity in a single group is Hermitian and uniform, which is
        // expressible without knowing the term vocabulary of any specific operator type.
        assert_eq!(group_coeff_means(&grouped), Some(vec![1.0]));
        assert_eq!(groups_are_hermitian(&grouped, 1e-8), Some(vec![true]));
        assert_eq!(
            groups_have_uniform_coeffs(&grouped, 1e-8, true),
            Some(vec![true])
        );

        // `i` times the identity is anti-Hermitian. Without this case the Hermiticity check would
        // pass even if it unconditionally reported `true`.
        let imaginary = OpType::from_terms_with_groups(
            grouped
                .iter_with_groups()
                .map(|term| term.scaled(Complex64::new(0.0, 1.0)))
                .collect::<Vec<_>>(),
        );
        assert_eq!(groups_are_hermitian(&imaginary, 1e-8), Some(vec![false]));
    }

    #[test]
    fn test_analysis_contract_through_trait_bound() {
        let one = Complex64::new(1.0, 0.0);
        assert_analysis_contract(FermionOperator {
            coeffs: vec![one],
            actions: vec![],
            modes: vec![],
            boundaries: vec![0, 0],
            groups: Some(vec![0]),
        });
        assert_analysis_contract(MajoranaOperator {
            coeffs: vec![one],
            modes: vec![],
            boundaries: vec![0, 0],
            groups: Some(vec![0]),
        });
        assert_analysis_contract(EdgeVertexOperator {
            coeffs: vec![one],
            left_indices: vec![],
            right_indices: vec![],
            boundaries: vec![0, 0],
            groups: Some(vec![0]),
        });
        assert_analysis_contract(TransferVertexOperator {
            coeffs: vec![one],
            left_indices: vec![],
            right_indices: vec![],
            boundaries: vec![0, 0],
            groups: Some(vec![0]),
        });
    }
}
