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

//! Rewriting a single operator term as a sum of signed Majorana monomials.
//!
//! Every operator representation in this crate is built, directly or indirectly, from Majorana
//! generators. Expressing a term in that common language gives a representation-independent handle on
//! it, which is what lets a routine ask structural questions (such as whether two terms commute)
//! without a `match` over the four operator types.
//!
//! The expansion is not uniformly cheap. A term of a Majorana, edge-vertex or transfer-vertex operator
//! *is* a single monomial. A fermionic term is not: each `a` or `a^dagger` is itself a sum of two
//! Majorana generators, so a term of degree `d` expands into `2^d` monomials.

use num_complex::Complex64;

use crate::operators::terms::grouping::clash::ClashError;

/// Majorana generators per fermionic mode.
///
/// Mode `j` owns generators `2j` and `2j + 1`.
pub(crate) const GENERATORS_PER_MODE: u64 = 2;

/// A set of Majorana generators, as an ascending list of distinct generator indices.
///
/// A sorted list rather than a bitset, because a monomial holds only as many generators as its term
/// has factors, while an operator can use very many. Every question asked of a support (how many
/// generators it holds, how many it shares with another, what their product is) is then a walk over
/// a handful of entries, whatever the indices, where a bitset walks one word per 64 generators the
/// whole operator uses.
///
/// Note that *composing* two supports is a symmetric difference, not a union: a term is a *product*
/// of generators and `gamma^2 = 1`, so a generator occurring twice cancels out.
pub(crate) type Support = Vec<u64>;

/// One Majorana monomial: a product of distinct Majorana generators, with its signed amplitude.
///
/// The support identifies the monomial; the coefficient carries the sign picked up by ordering the
/// generators canonically, along with any numeric prefactor the expansion introduced.
pub(crate) type Monomial = (Support, Complex64);

/// Reduces a list of Majorana generators, sorted ascending, to the support of their product.
///
/// Sorted, copies of a generator sit next to each other, and since `gamma^2 = 1` each adjacent pair
/// cancels: a generator equal to the last one kept removes it instead of being kept. A generator
/// occurring an odd number of times therefore survives once, and one occurring an even number of
/// times not at all. This runs in place, in one pass.
pub(crate) fn cancel_repeated_generators(generators: &mut Support) {
    let mut kept = 0;
    for read in 0..generators.len() {
        let generator = generators[read];
        if kept > 0 && generators[kept - 1] == generator {
            kept -= 1;
        } else {
            generators[kept] = generator;
            kept += 1;
        }
    }
    generators.truncate(kept);
}

/// Returns the support of a product of Majorana generators given in any order.
///
/// Only the support is wanted, not the sign of reordering, so a library sort serves. Sorting first
/// and cancelling after keeps a term of `k` generators at `O(k log k)`, where inserting them one by
/// one into a sorted list would cost `O(k^2)`.
pub(crate) fn product_support(mut generators: Support) -> Support {
    generators.sort_unstable();
    cancel_repeated_generators(&mut generators);
    generators
}

/// Returns the number of generators shared by two supports.
///
/// Both lists are sorted, so one simultaneous walk over them finds every common entry.
#[inline]
pub(crate) fn count_shared_generators(left: &[u64], right: &[u64]) -> usize {
    let (mut i, mut j, mut shared) = (0, 0, 0);
    while i < left.len() && j < right.len() {
        match left[i].cmp(&right[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                shared += 1;
                i += 1;
                j += 1;
            }
        }
    }
    shared
}

/// Composes two monomial supports into `product`, returning whether the sign flipped.
///
/// `product` is overwritten rather than returned, so that a caller composing many pairs reuses one
/// allocation for all of them.
///
/// Writing the product means concatenating the two generator lists and sorting the result back into
/// canonical order; since `gamma_i gamma_j = -gamma_j gamma_i`, each transposition flips the sign.
/// Rather than materializing the concatenation, a single merge walk counts the transpositions
/// directly: moving each generator of `right` left past every generator of `left` that exceeds it
/// costs one each, and those are exactly the entries of `left` not yet consumed by the walk. A
/// generator common to both moves past the ones strictly above it and then annihilates with its
/// partner, so it is left out of the product.
pub(crate) fn compose_into(left: &[u64], right: &[u64], product: &mut Support) -> bool {
    product.clear();
    let mut transpositions: usize = 0;
    let (mut i, mut j) = (0, 0);
    while i < left.len() || j < right.len() {
        if j == right.len() || (i < left.len() && left[i] < right[j]) {
            product.push(left[i]);
            i += 1;
        } else if i == left.len() || right[j] < left[i] {
            // Every remaining generator of `left` exceeds this one.
            transpositions += left.len() - i;
            product.push(right[j]);
            j += 1;
        } else {
            // Shared: it moves past the generators of `left` strictly above it, then annihilates.
            transpositions += left.len() - i - 1;
            i += 1;
            j += 1;
        }
    }
    transpositions % 2 == 1
}

/// An operator whose terms can be rewritten as sums of signed Majorana monomials.
///
/// Expressing a term in the common Majorana language is what lets representation-independent routines
/// avoid a `match` over the four operator types.
pub trait MajoranaExpansion {
    /// Replaces `out` with the Majorana expansion of term `index`.
    ///
    /// The supports hold raw generator indices (mode `j` owning `2j` and `2j + 1`), so the supports
    /// of different terms are directly comparable.
    fn majorana_monomials_into(&self, index: usize, out: &mut Vec<Monomial>);

    /// Checks that every term can be expanded, before any expansion is attempted.
    ///
    /// Representations whose terms are single monomials always can, which is the default.
    fn check_expandable(&self) -> Result<(), ClashError> {
        Ok(())
    }
}

/// Expands a term built from Majorana generators: exactly one monomial, with unit coefficient.
///
/// The coefficient is dropped deliberately. These expansions feed structural questions about a term,
/// and a non-zero scalar cannot change the answer to any of them: writing `A` and `B` for the
/// monomials and `a`, `b` for their coefficients, `(aA)(bB) = ab AB` and `(bB)(aA) = ab BA`, so the
/// scalars factor out of both orderings identically. Contrast [`FermionOperator`], whose expansion is
/// a genuine *sum*: there the relative coefficients decide whether contributions cancel, and dropping
/// them would be wrong.
pub(crate) fn single_monomial(support: Support, out: &mut Vec<Monomial>) {
    out.clear();
    out.push((support, Complex64::ONE));
}

/// Expands one term of an edge- or transfer-vertex operator.
///
/// `generator` maps a single `(left, right)` factor onto the pair of Majorana generators it acts on.
pub(crate) fn expand_vertex_term(
    left_indices: &[u32],
    right_indices: &[u32],
    out: &mut Vec<Monomial>,
    generator: impl Fn(u64, u64) -> (u64, u64),
) {
    let mut generators = Support::with_capacity(2 * left_indices.len());
    for (&left, &right) in left_indices.iter().zip(right_indices) {
        let (first, second) = generator(u64::from(left), u64::from(right));
        generators.push(first);
        generators.push(second);
    }
    single_monomial(product_support(generators), out);
}

/// Sorts Majorana generators ascending, returning whether an odd number of swaps was needed.
///
/// Insertion sort rather than a library sort because the transposition *parity* is the point, and a
/// comparison sort does not expose it. Degrees are single digits in practice, so the quadratic cost is
/// irrelevant.
pub(crate) fn sort_sign(generators: &mut [u64]) -> bool {
    let mut swaps: u64 = 0;
    for i in 1..generators.len() {
        let mut j = i;
        while j > 0 && generators[j - 1] > generators[j] {
            generators.swap(j - 1, j);
            swaps += 1;
            j -= 1;
        }
    }
    swaps % 2 == 1
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::operators::edge_vertex_operator::EdgeVertexOperator;
    use crate::operators::fermion_operator::FermionOperator;
    use crate::operators::majorana_operator::MajoranaOperator;
    use crate::operators::transfer_vertex_operator::TransferVertexOperator;

    #[test]
    fn test_product_support_cancels_repeated_generators() {
        // `gamma^2 = 1`, so a generator occurring twice drops out.
        assert_eq!(product_support(vec![5, 5]), Vec::<u64>::new());
        // An odd count survives once, and the result is sorted whatever the input order.
        assert_eq!(product_support(vec![70, 3, 70, 9, 70]), vec![3, 9, 70]);
        assert_eq!(product_support(vec![3, 70, 3, 70]), Vec::<u64>::new());
        assert_eq!(product_support(Vec::new()), Vec::<u64>::new());
    }

    #[test]
    fn test_compose_sign_matches_explicit_reordering() {
        // Composing `gamma_1` with `gamma_0` needs one transposition to reach canonical order.
        let left = vec![1];
        let right = vec![0];
        let mut product = Vec::new();
        let negated = compose_into(&left, &right, &mut product);
        assert_eq!(product.len(), 2);
        assert!(negated, "gamma_1 gamma_0 = -gamma_0 gamma_1");

        // The other order is already canonical.
        let negated = compose_into(&right, &left, &mut product);
        assert_eq!(product.len(), 2);
        assert!(!negated);
    }

    #[test]
    fn test_compose_annihilates_shared_generators() {
        let support = vec![3];
        let mut product = vec![7];
        compose_into(&support, &support, &mut product);
        assert!(
            product.is_empty(),
            "a generator composed with itself squares to the identity"
        );
    }

    #[test]
    fn test_largest_index_expands_cheaply() {
        // A support lists its generators, so its size follows the term's degree and never the index:
        // even the largest index a `u32` can hold costs one entry per generator.
        let last = u32::MAX;
        let mut out = Vec::new();

        let fermion = FermionOperator {
            coeffs: vec![Complex64::ONE],
            actions: vec![true],
            modes: vec![last],
            boundaries: vec![0, 1],
            groups: None,
        };
        fermion.majorana_monomials_into(0, &mut out);
        assert!(out.iter().all(|(support, _)| support.len() == 1));

        let edge = EdgeVertexOperator {
            coeffs: vec![Complex64::ONE],
            left_indices: vec![0],
            right_indices: vec![last],
            boundaries: vec![0, 1],
            groups: None,
        };
        edge.majorana_monomials_into(0, &mut out);
        assert_eq!(out[0].0.len(), 2);

        let transfer = TransferVertexOperator {
            coeffs: vec![Complex64::ONE],
            left_indices: vec![last],
            right_indices: vec![0],
            boundaries: vec![0, 1],
            groups: None,
        };
        transfer.majorana_monomials_into(0, &mut out);
        assert_eq!(out[0].0.len(), 2);

        let majorana = MajoranaOperator {
            coeffs: vec![Complex64::ONE],
            modes: vec![last],
            boundaries: vec![0, 1],
            groups: None,
        };
        majorana.majorana_monomials_into(0, &mut out);
        assert_eq!(out[0].0, vec![u64::from(last)]);
    }

    #[test]
    fn test_fermion_expansion_counts_and_identity_cancellation() {
        // `a^dagger_0 a_1` has degree 2, hence four monomials, none of which cancels.
        let op = FermionOperator {
            coeffs: vec![Complex64::ONE],
            actions: vec![true, false],
            modes: vec![0, 1],
            boundaries: vec![0, 2],
            groups: None,
        };
        let mut out = Vec::new();
        op.majorana_monomials_into(0, &mut out);
        assert_eq!(out.len(), 4);
        assert!(out.iter().all(|(support, _)| support.len() == 2));

        // `a^dagger_0 a_0` also yields four monomials, but two of them pick the *same* generator
        // twice and so collapse onto the identity.
        let number = FermionOperator {
            coeffs: vec![Complex64::ONE],
            actions: vec![true, false],
            modes: vec![0, 0],
            boundaries: vec![0, 2],
            groups: None,
        };
        number.majorana_monomials_into(0, &mut out);
        assert_eq!(out.len(), 4);
        let identities = out.iter().filter(|(support, _)| support.is_empty()).count();
        assert_eq!(identities, 2, "two choices repeat a generator and cancel");

        // In `a^dagger_0 a_0 a^dagger_0` every choice picks one of mode 0's two generators three
        // times. Exactly one of them is then picked an odd number of times, and survives once.
        let triple = FermionOperator {
            coeffs: vec![Complex64::ONE],
            actions: vec![true, false, true],
            modes: vec![0, 0, 0],
            boundaries: vec![0, 3],
            groups: None,
        };
        triple.majorana_monomials_into(0, &mut out);
        assert_eq!(out.len(), 8);
        assert!(out.iter().all(|(support, _)| support.len() == 1));
    }

    #[test]
    fn test_majorana_family_expands_to_one_monomial() {
        let maj = MajoranaOperator {
            coeffs: vec![Complex64::ONE],
            modes: vec![0, 3],
            boundaries: vec![0, 2],
            groups: None,
        };
        let edge = EdgeVertexOperator {
            coeffs: vec![Complex64::ONE],
            left_indices: vec![0],
            right_indices: vec![1],
            boundaries: vec![0, 1],
            groups: None,
        };
        let transfer = TransferVertexOperator {
            coeffs: vec![Complex64::ONE],
            left_indices: vec![0],
            right_indices: vec![1],
            boundaries: vec![0, 1],
            groups: None,
        };
        let mut out = Vec::new();
        for (label, len) in [
            ("majorana", {
                maj.majorana_monomials_into(0, &mut out);
                out.len()
            }),
            ("edge", {
                edge.majorana_monomials_into(0, &mut out);
                out.len()
            }),
            ("transfer", {
                transfer.majorana_monomials_into(0, &mut out);
                out.len()
            }),
        ] {
            assert_eq!(len, 1, "{label} terms are single monomials");
        }
    }

    #[test]
    fn test_edge_orientations_share_a_support_but_transfer_ones_do_not() {
        let build_edge = |l: u32, r: u32| EdgeVertexOperator {
            coeffs: vec![Complex64::ONE],
            left_indices: vec![l],
            right_indices: vec![r],
            boundaries: vec![0, 1],
            groups: None,
        };
        let mut forward = Vec::new();
        let mut reverse = Vec::new();
        build_edge(0, 1).majorana_monomials_into(0, &mut forward);
        build_edge(1, 0).majorana_monomials_into(0, &mut reverse);
        assert_eq!(
            forward[0].0, reverse[0].0,
            "E_jk and E_kj differ only by a sign, which is not part of the support"
        );

        let build_transfer = |l: u32, r: u32| TransferVertexOperator {
            coeffs: vec![Complex64::ONE],
            left_indices: vec![l],
            right_indices: vec![r],
            boundaries: vec![0, 1],
            groups: None,
        };
        build_transfer(0, 1).majorana_monomials_into(0, &mut forward);
        build_transfer(1, 0).majorana_monomials_into(0, &mut reverse);
        assert_ne!(
            forward[0].0, reverse[0].0,
            "the two orientations of a transfer are distinguishable"
        );
    }

    #[test]
    fn test_support_size_is_independent_of_index_magnitude() {
        // Two operators that differ only by how far apart their indices are hold supports of the
        // same size: a support lists generators, so their values never enter its cost.
        let build = |modes: Vec<u32>| MajoranaOperator {
            coeffs: vec![Complex64::ONE],
            modes,
            boundaries: vec![0, 2],
            groups: None,
        };
        let mut near = Vec::new();
        let mut far = Vec::new();
        build(vec![0, 3]).majorana_monomials_into(0, &mut near);
        build(vec![0, 3_000_000]).majorana_monomials_into(0, &mut far);
        assert_eq!(near[0].0, vec![0, 3]);
        assert_eq!(far[0].0, vec![0, 3_000_000]);
    }
}
