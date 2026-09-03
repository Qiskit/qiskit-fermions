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

use crate::mappers::qk_obs::{
    QkComplex64, as_bit_terms, compose_actions, ffi_backend as ffi, map_operator, max_paired_index,
    one_pauli_string, pauli_sum, qk_coeff,
};
use crate::mappers::ternary_tree::PauliLabel;
use crate::mappers::ternary_tree::encoding::{TernaryTreeEncoding, TernaryTreeEncodingError};
use crate::operators::OperatorTrait;
use crate::operators::edge_vertex_operator::{EdgeAction, EdgeVertexOperator};
use crate::operators::fermion_operator::{FermionAction, FermionOperator};
use crate::operators::majorana_operator::{MajoranaAction, MajoranaOperator};
use crate::operators::transfer_vertex_operator::{TransferAction, TransferVertexOperator};

/// Maps a single Majorana operator onto its Pauli image under a ternary-tree encoding.
///
/// The image is a *single* Pauli string with coefficient exactly `1`: a leg's string is a product of
/// Paulis on distinct qubits, so it is Hermitian and squares to the identity with no phase to correct.
/// Every factor of `i` that a term produces arises when two such strings are composed, and is
/// accumulated there.
///
/// The encoding's slices are handed to `qk_obs_new` as they are, so this allocates nothing beyond
/// the observable itself.
fn map_majorana_action(
    enc: &TernaryTreeEncoding,
    action: MajoranaAction,
    num_qubits: u32,
) -> *mut ffi::QkObs {
    let (labels, indices) = enc.majorana_image(*action);
    one_pauli_string(
        num_qubits,
        QkComplex64 { re: 1.0, im: 0.0 },
        as_bit_terms(labels),
        indices,
    )
}

/// Maps a single fermionic action onto its Pauli image under a ternary-tree encoding.
///
/// With `gamma_2j = a^dagger_j + a_j` and `gamma_2j+1 = i(a^dagger_j - a_j)`,
///
/// ```text
///     a^dagger_j  ->  (gamma_2j - i gamma_2j+1) / 2
///     a_j         ->  (gamma_2j + i gamma_2j+1) / 2
/// ```
///
/// so the image is a two-term sum of the mode's two Majorana strings. This is the direct
/// generalisation of the Jordan-Wigner mapper's `map_fermion_action`: on a chain along `Z` the two
/// strings are `Z...Z X_j` and `Z...Z Y_j`, and the observable built here is the one that mapper
/// builds.
fn map_fermion_action(
    enc: &TernaryTreeEncoding,
    action: FermionAction,
    num_qubits: u32,
) -> *mut ffi::QkObs {
    let im = if *action.0 { -0.5 } else { 0.5 };
    // The encoding stores a mode's two strings next to each other, so the two-term sum is built
    // straight from that span.
    let (labels, indices, split) = enc.mode_images(*action.1);
    pauli_sum(
        num_qubits,
        &[
            QkComplex64 { re: 0.5, im: 0.0 },
            QkComplex64 { re: 0.0, im },
        ],
        as_bit_terms(labels),
        indices,
        &[0, split, labels.len()],
    )
}

/// Maps the vertex operator `V_j = -i gamma_2j gamma_2j+1` of `mode` onto its Pauli image.
///
/// The image is precomputed in the encoding, since it depends on nothing else, and is a `Z` string
/// with coefficient `+1` on every tree. Unlike under Jordan-Wigner it is not in general the weight-1
/// `Z_j`: that is a property of the chain, where the two Majorana strings' `Z` chains below `j` are
/// empty (on a balanced tree a vertex operator is not weight 1).
fn map_vertex(enc: &TernaryTreeEncoding, mode: u32, num_qubits: u32) -> *mut ffi::QkObs {
    let (labels, indices) = enc.vertex_image(mode);
    one_pauli_string(
        num_qubits,
        QkComplex64 { re: 1.0, im: 0.0 },
        as_bit_terms(labels),
        indices,
    )
}

/// Multiplies two single-qubit Paulis, returning the product's label (`None` for the identity) and
/// its phase as a power of `i`.
///
/// The product of two distinct Paulis is `+i` times the third when they appear in cyclic order
/// `X, Y, Z`, and `-i` times it otherwise.
#[inline]
fn pauli_product(left: PauliLabel, right: PauliLabel) -> (Option<PauliLabel>, u32) {
    use PauliLabel::{X, Y, Z};
    match (left, right) {
        (X, Y) => (Some(Z), 1),
        (Y, Z) => (Some(X), 1),
        (Z, X) => (Some(Y), 1),
        (Y, X) => (Some(Z), 3),
        (Z, Y) => (Some(X), 3),
        (X, Z) => (Some(Y), 3),
        _ => (None, 0),
    }
}

/// Multiplies two Majorana images and scales the product by `factor`.
///
/// The off-diagonal generators of the edge and transfer algebras are each defined as a product of
/// exactly two Majorana operators on different modes, so every one of their images is a single Pauli
/// string -- whatever the tree. The relative phase of that string depends on how the two paths
/// overlap, which is tree-dependent, so it is accumulated qubit by qubit rather than written down.
///
/// This is where the ternary-tree mappers genuinely diverge from their Jordan-Wigner counterparts,
/// which can shortcut the same result with a closed form: on a chain the two `Z` chains cancel below
/// the lower endpoint, leaving `Z` strictly between the two and a sign fixed by the index order. That
/// cancellation is a property of the chain, not of ternary trees (an edge operator's coefficient on a
/// balanced tree is not the chain's), so the product is taken explicitly here.
///
/// It is taken as a merge of the two ascending strings rather than through `qk_obs_compose`: both are
/// single Pauli strings, so their product factorizes over qubits, and the merge costs one pass over
/// their supports where the compose would cost three observables. This runs once per off-diagonal
/// action in the mappers' innermost loop.
///
/// Unlike the vertex operators these are not precomputed: there are `O(N^2)` of them, against the
/// `N` vertex operators, and a Hamiltonian typically touches only a few.
fn majorana_product(
    enc: &TernaryTreeEncoding,
    left: u32,
    right: u32,
    factor: QkComplex64,
    num_qubits: u32,
) -> *mut ffi::QkObs {
    let (left_labels, left_indices) = enc.majorana_image(left);
    let (right_labels, right_indices) = enc.majorana_image(right);
    let capacity = left_indices.len() + right_indices.len();
    let mut indices = Vec::with_capacity(capacity);
    let mut labels = Vec::with_capacity(capacity);
    // The power of `i` accumulated so far. Paulis on different qubits commute, so only the qubits
    // both strings act on contribute, and each contributes its own single-qubit product.
    let mut power = 0;
    let (mut l, mut r) = (0, 0);
    while l < left_indices.len() || r < right_indices.len() {
        let next_left = left_indices.get(l).copied().unwrap_or(u32::MAX);
        let next_right = right_indices.get(r).copied().unwrap_or(u32::MAX);
        if next_left < next_right {
            indices.push(next_left);
            labels.push(left_labels[l]);
            l += 1;
        } else if next_right < next_left {
            indices.push(next_right);
            labels.push(right_labels[r]);
            r += 1;
        } else {
            let (label, phase) = pauli_product(left_labels[l], right_labels[r]);
            if let Some(label) = label {
                indices.push(next_left);
                labels.push(label);
            }
            power += phase;
            l += 1;
            r += 1;
        }
    }
    // `factor * i^power`.
    let coeff = match power % 4 {
        0 => factor,
        1 => QkComplex64 {
            re: -factor.im,
            im: factor.re,
        },
        2 => QkComplex64 {
            re: -factor.re,
            im: -factor.im,
        },
        _ => QkComplex64 {
            re: factor.im,
            im: -factor.re,
        },
    };
    one_pauli_string(num_qubits, coeff, as_bit_terms(&labels), &indices)
}

/// Maps a single generalized edge operator onto its Pauli image.
///
/// With the definitions `V_j = E_jj = -i gamma_2j gamma_2j+1` and `E_jk = -i gamma_2j gamma_2k`.
fn map_edge_action(
    enc: &TernaryTreeEncoding,
    action: EdgeAction,
    num_qubits: u32,
) -> *mut ffi::QkObs {
    let (left, right) = (*action.0, *action.1);
    if left == right {
        return map_vertex(enc, left, num_qubits);
    }
    majorana_product(
        enc,
        2 * left,
        2 * right,
        QkComplex64 { re: 0.0, im: -1.0 },
        num_qubits,
    )
}

/// Maps a single generalized transfer operator onto its Pauli image.
///
/// With the definitions `V_j = T_jj = -i gamma_2j gamma_2j+1` and `T_jk = (i/2) gamma_2j+1 gamma_2k`.
/// Unlike an edge operator there is no antisymmetry to exploit: `T_jk` and `T_kj` are different
/// operators, and both directions are stored.
fn map_transfer_action(
    enc: &TernaryTreeEncoding,
    action: TransferAction,
    num_qubits: u32,
) -> *mut ffi::QkObs {
    let (left, right) = (*action.0, *action.1);
    if left == right {
        return map_vertex(enc, left, num_qubits);
    }
    majorana_product(
        enc,
        2 * left + 1,
        2 * right,
        QkComplex64 { re: 0.0, im: 0.5 },
        num_qubits,
    )
}

/// Checks that both `enc` and `num_qubits` cover every mode up to `max_mode`.
///
/// Two independent bounds, both of which must hold:
///
/// * The encoding only spans its tree's nodes, so a mode beyond them has no Pauli image at all.
/// * `num_qubits` must cover the encoding's *whole* register, not merely the modes the operator
///   touches. A mode's Pauli image spans every node on its path to the root, so even mode `0` can
///   act on a high qubit index -- on the balanced tree at four modes its image is `X_0 Z_1`. Taking
///   the minimum of the two bounds would therefore let an out-of-range qubit index reach
///   `qk_obs_new`, whose non-unwinding panic aborts the process rather than returning an error.
///
/// `num_qubits` may exceed the tree's node count, in which case the extra qubits carry the identity.
fn check_support(
    enc: &TernaryTreeEncoding,
    num_qubits: u32,
    max_mode: Option<u32>,
) -> Result<(), TernaryTreeEncodingError> {
    // The encoding must have a Pauli image for every mode the operator acts on. Reported as a
    // shortfall of the encoding rather than of `num_qubits`, because raising `num_qubits` cannot
    // widen it.
    if let Some(max_mode) = max_mode
        && max_mode >= enc.num_modes()
    {
        return Err(TernaryTreeEncodingError::ModeOutsideEncoding {
            num_modes: enc.num_modes(),
            max_mode,
        });
    }
    // The register must hold every qubit the encoding can address, independently of `max_mode`.
    if num_qubits < enc.num_modes() {
        return Err(TernaryTreeEncodingError::NumQubitsTooSmall {
            num_qubits,
            num_modes: enc.num_modes(),
        });
    }
    Ok(())
}

/// The body every ternary-tree mapper shares, given the operator's largest mode and its per-action
/// image builder.
///
/// A macro rather than a generic function for the reason [`map_operator`] takes a closure: each
/// operator's term view exposes `iter()` as an inherent method, so a term's actions are only
/// reachable from code that names the concrete view type, which this expands into.
///
/// The support check comes first and must not be skipped: it is what keeps the image builders'
/// lookups by mode in range, and so what keeps `map_operator`'s closure from unwinding.
macro_rules! map_through_encoding {
    ($op:expr, $enc:expr, $num_qubits:expr, $max_mode:expr, $image:path) => {{
        let enc: &TernaryTreeEncoding = $enc;
        check_support(enc, $num_qubits, $max_mode)?;
        Ok(map_operator($op.iter(), $num_qubits, |term, num_qubits| {
            (
                compose_actions(term.iter(), num_qubits, |action, num_qubits| {
                    $image(enc, action, num_qubits)
                }),
                qk_coeff(term.coeff),
            )
        }))
    }};
}

/// Maps a [`FermionOperator`] onto a qubit operator under a ternary-tree encoding.
///
/// On a Jordan-Wigner tree
/// ([`TernaryTree::chain(n, PauliLabel::Z)`](crate::mappers::ternary_tree::TernaryTree::chain)) this
/// produces the same observable as
/// [`fermion_jordan_wigner`](super::jordan_wigner::fermion_jordan_wigner), which is asserted by the
/// tests. Other trees trade that encoding's linear Pauli weight for a shallower one.
pub fn fermion_ternary_tree(
    fer_op: &FermionOperator,
    enc: &TernaryTreeEncoding,
    num_qubits: u32,
) -> Result<*mut ffi::QkObs, TernaryTreeEncodingError> {
    let max_mode = fer_op.modes.iter().max().copied();
    map_through_encoding!(fer_op, enc, num_qubits, max_mode, map_fermion_action)
}

/// Maps a [`MajoranaOperator`] onto a qubit operator under a ternary-tree encoding.
///
/// Each Majorana operator's image is a single Pauli string, so a term of `L` of them composes `L`
/// strings rather than inflating into a sum.
pub fn majorana_ternary_tree(
    maj_op: &MajoranaOperator,
    enc: &TernaryTreeEncoding,
    num_qubits: u32,
) -> Result<*mut ffi::QkObs, TernaryTreeEncodingError> {
    // Majorana index `m` acts on mode `m / 2`, so it is the quotient that must be covered.
    let max_mode = maj_op.modes.iter().max().map(|&m| m / 2);
    map_through_encoding!(maj_op, enc, num_qubits, max_mode, map_majorana_action)
}

/// Maps an [`EdgeVertexOperator`] onto a qubit operator under a ternary-tree encoding.
pub fn edge_vertex_ternary_tree(
    inter_op: &EdgeVertexOperator,
    enc: &TernaryTreeEncoding,
    num_qubits: u32,
) -> Result<*mut ffi::QkObs, TernaryTreeEncodingError> {
    let max_mode = max_paired_index(&inter_op.left_indices, &inter_op.right_indices);
    map_through_encoding!(inter_op, enc, num_qubits, max_mode, map_edge_action)
}

/// Maps a [`TransferVertexOperator`] onto a qubit operator under a ternary-tree encoding.
pub fn transfer_vertex_ternary_tree(
    inter_op: &TransferVertexOperator,
    enc: &TernaryTreeEncoding,
    num_qubits: u32,
) -> Result<*mut ffi::QkObs, TernaryTreeEncodingError> {
    let max_mode = max_paired_index(&inter_op.left_indices, &inter_op.right_indices);
    map_through_encoding!(inter_op, enc, num_qubits, max_mode, map_transfer_action)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::mappers::library::jordan_wigner::{
        edge_vertex_jordan_wigner, fermion_jordan_wigner, majorana_jordan_wigner,
        transfer_vertex_jordan_wigner,
    };
    use crate::mappers::qk_obs::{assert_obs_equal, compact};
    use crate::mappers::ternary_tree::TernaryTree;
    use crate::mappers::ternary_tree::encoding::ModeMap;
    use num_complex::Complex64;

    /// The Jordan-Wigner encoding, as a ternary tree.
    fn jw_encoding(n: u32) -> TernaryTreeEncoding {
        let tree = TernaryTree::chain(n, PauliLabel::Z).unwrap();
        TernaryTreeEncoding::new(&tree, &ModeMap::Identity).unwrap()
    }

    /// The balanced ternary-tree encoding.
    fn balanced_encoding(n: u32) -> TernaryTreeEncoding {
        let tree = TernaryTree::breadth_first(n, 3, [PauliLabel::X, PauliLabel::Y, PauliLabel::Z])
            .unwrap();
        TernaryTreeEncoding::new(&tree, &ModeMap::Identity).unwrap()
    }

    /// A four-mode electronic-structure-shaped fermionic operator.
    fn fermion_op() -> FermionOperator {
        FermionOperator {
            coeffs: vec![
                Complex64::new(2.0, 0.0),
                Complex64::new(0.1, 0.0),
                Complex64::new(0.0, -1.0),
                Complex64::new(0.25, 0.5),
            ],
            actions: vec![
                true, false, true, false, true, false, true, true, false, false,
            ],
            modes: vec![0, 0, 1, 2, 2, 1, 0, 3, 3, 1],
            boundaries: vec![0, 0, 2, 6, 10],
            groups: None,
        }
    }

    fn majorana_op() -> MajoranaOperator {
        MajoranaOperator {
            coeffs: vec![
                Complex64::new(0.5, 0.0),
                Complex64::new(0.0, 0.25),
                Complex64::new(-1.5, 0.0),
            ],
            modes: vec![0, 3, 2, 5, 4, 1, 6, 7],
            boundaries: vec![0, 2, 4, 8],
            groups: None,
        }
    }

    fn edge_op() -> EdgeVertexOperator {
        EdgeVertexOperator {
            coeffs: vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.5),
                Complex64::new(-0.75, 0.0),
            ],
            left_indices: vec![0, 1, 2, 0, 3],
            right_indices: vec![0, 2, 1, 3, 1],
            boundaries: vec![0, 1, 3, 5],
            groups: None,
        }
    }

    fn transfer_op() -> TransferVertexOperator {
        TransferVertexOperator {
            coeffs: vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.5),
                Complex64::new(-0.75, 0.0),
            ],
            left_indices: vec![1, 0, 2, 3, 1],
            right_indices: vec![1, 2, 0, 1, 3],
            boundaries: vec![0, 1, 3, 5],
            groups: None,
        }
    }

    /// The four regressions that tie this framework to an independently verified mapper.
    ///
    /// On the Jordan-Wigner tree every generator's image is the one the direct mapper builds, so the
    /// two must agree term for term. These are the tests that would make collapsing the Jordan-Wigner
    /// mappers onto this engine reviewable.
    #[test]
    fn test_fermion_matches_jordan_wigner_on_the_chain() {
        let op = fermion_op();
        let enc = jw_encoding(4);
        assert_obs_equal(
            fermion_ternary_tree(&op, &enc, enc.num_modes()).unwrap(),
            fermion_jordan_wigner(&op, 4).unwrap(),
            4,
        );
    }

    #[test]
    fn test_majorana_matches_jordan_wigner_on_the_chain() {
        let op = majorana_op();
        let enc = jw_encoding(4);
        assert_obs_equal(
            majorana_ternary_tree(&op, &enc, enc.num_modes()).unwrap(),
            majorana_jordan_wigner(&op, 4).unwrap(),
            4,
        );
    }

    #[test]
    fn test_edge_vertex_matches_jordan_wigner_on_the_chain() {
        let op = edge_op();
        let enc = jw_encoding(4);
        assert_obs_equal(
            edge_vertex_ternary_tree(&op, &enc, enc.num_modes()).unwrap(),
            edge_vertex_jordan_wigner(&op, 4).unwrap(),
            4,
        );
    }

    #[test]
    fn test_transfer_vertex_matches_jordan_wigner_on_the_chain() {
        let op = transfer_op();
        let enc = jw_encoding(4);
        assert_obs_equal(
            transfer_vertex_ternary_tree(&op, &enc, enc.num_modes()).unwrap(),
            transfer_vertex_jordan_wigner(&op, 4).unwrap(),
            4,
        );
    }

    /// On a non-Jordan-Wigner tree the four mappers must still agree with each other.
    ///
    /// The Majorana route is the reference: the other three algebras are *defined* as products of
    /// Majorana operators, so mapping an operator directly must equal mapping its Majorana form. This
    /// is what catches a tree-general image being written as a Jordan-Wigner closed form, which the
    /// regressions above cannot see.
    #[test]
    fn test_edge_vertex_matches_its_majorana_form_on_a_balanced_tree() {
        use crate::mappers::library::edge_vertex::edge_vertex_to_majorana;
        let op = edge_op();
        let enc = balanced_encoding(4);
        let via_majorana = edge_vertex_to_majorana(&op);
        assert_obs_equal(
            edge_vertex_ternary_tree(&op, &enc, enc.num_modes()).unwrap(),
            majorana_ternary_tree(&via_majorana, &enc, enc.num_modes()).unwrap(),
            4,
        );
    }

    #[test]
    fn test_transfer_vertex_matches_its_majorana_form_on_a_balanced_tree() {
        use crate::mappers::library::transfer_vertex::transfer_vertex_to_majorana;
        let op = transfer_op();
        let enc = balanced_encoding(4);
        let via_majorana = transfer_vertex_to_majorana(&op);
        assert_obs_equal(
            transfer_vertex_ternary_tree(&op, &enc, enc.num_modes()).unwrap(),
            majorana_ternary_tree(&via_majorana, &enc, enc.num_modes()).unwrap(),
            4,
        );
    }

    #[test]
    fn test_fermion_matches_its_majorana_form_on_a_balanced_tree() {
        use crate::mappers::library::majorana_fermion::fermion_to_majorana;
        let op = fermion_op();
        let enc = balanced_encoding(4);
        let via_majorana = fermion_to_majorana(&op);
        assert_obs_equal(
            fermion_ternary_tree(&op, &enc, enc.num_modes()).unwrap(),
            majorana_ternary_tree(&via_majorana, &enc, enc.num_modes()).unwrap(),
            4,
        );
    }

    /// A single off-diagonal action carries exactly its definition's prefactor.
    ///
    /// The fixtures above only have terms with two off-diagonal actions, in which the prefactor
    /// appears squared and a wrong sign cancels. Here every ordered pair is mapped as a term of its
    /// own and compared with the operator's Majorana form. They are separate operators because
    /// summing them would cancel for the edge operators (`E_kj = -E_jk`).
    #[test]
    fn test_a_single_off_diagonal_action_carries_its_prefactor() {
        use crate::mappers::library::edge_vertex::edge_vertex_to_majorana;
        use crate::mappers::library::transfer_vertex::transfer_vertex_to_majorana;
        // Not 1, so that the coefficient is exercised alongside the prefactor.
        let coeff = Complex64::new(0.3, -0.7);
        let permuted = TernaryTreeEncoding::new(
            &TernaryTree::chain(4, PauliLabel::X).unwrap(),
            &ModeMap::Permutation(vec![2, 0, 3, 1]),
        )
        .unwrap();
        for enc in [jw_encoding(4), balanced_encoding(4), permuted] {
            let n = enc.num_modes();
            for left in 0..n {
                for right in (0..n).filter(|&right| right != left) {
                    let edge = EdgeVertexOperator {
                        coeffs: vec![coeff],
                        left_indices: vec![left],
                        right_indices: vec![right],
                        boundaries: vec![0, 1],
                        groups: None,
                    };
                    assert_obs_equal(
                        edge_vertex_ternary_tree(&edge, &enc, n).unwrap(),
                        majorana_ternary_tree(&edge_vertex_to_majorana(&edge), &enc, n).unwrap(),
                        n,
                    );
                    let transfer = TransferVertexOperator {
                        coeffs: vec![coeff],
                        left_indices: vec![left],
                        right_indices: vec![right],
                        boundaries: vec![0, 1],
                        groups: None,
                    };
                    assert_obs_equal(
                        transfer_vertex_ternary_tree(&transfer, &enc, n).unwrap(),
                        majorana_ternary_tree(&transfer_vertex_to_majorana(&transfer), &enc, n)
                            .unwrap(),
                        n,
                    );
                }
            }
        }
    }

    /// The vertex operator is diagonal on every tree, but only weight 1 on the chain.
    ///
    /// This is the concrete form of the warning on [`map_vertex`]: the Jordan-Wigner closed form
    /// `V_l -> Z_l` is a property of the chain, and a mapper that hardcoded it would be wrong here.
    #[test]
    fn test_vertex_weight_is_tree_dependent() {
        let vertex = EdgeVertexOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            left_indices: vec![0],
            right_indices: vec![0],
            boundaries: vec![0, 1],
            groups: None,
        };

        // On the chain: the weight-1 Pauli `Z_0`.
        let chain = fermion_weight(edge_vertex_ternary_tree(&vertex, &jw_encoding(4), 4).unwrap());
        assert_eq!(chain, vec![1], "V_0 should be weight 1 on the chain");

        // On the balanced tree the two Majorana paths overlap differently, and it is not.
        let balanced =
            fermion_weight(edge_vertex_ternary_tree(&vertex, &balanced_encoding(4), 4).unwrap());
        assert_eq!(
            balanced,
            vec![3],
            "V_0 should be weight 3 on the balanced tree at n=4"
        );
    }

    /// Returns the Pauli weight of each term of an observable, consuming it.
    fn fermion_weight(obs: *mut ffi::QkObs) -> Vec<usize> {
        let obs = unsafe { compact(obs) };
        let num_terms = unsafe { ffi::qk_obs_num_terms(obs) } as usize;
        let boundaries =
            unsafe { std::slice::from_raw_parts(ffi::qk_obs_boundaries(obs), num_terms + 1) };
        (0..num_terms)
            .map(|t| boundaries[t + 1] - boundaries[t])
            .collect()
    }

    /// The merged product of two Majorana images is the one `qk_obs_compose` builds.
    ///
    /// Taken over every ordered pair, including a string with itself (the identity) and the two
    /// strings of one mode (`i` times its vertex), on trees whose strings overlap in different ways.
    #[test]
    fn test_majorana_product_matches_compose() {
        let factor = QkComplex64 { re: 0.3, im: -0.7 };
        let one = QkComplex64 { re: 1.0, im: 0.0 };
        for enc in [
            jw_encoding(5),
            balanced_encoding(7),
            TernaryTreeEncoding::new(
                &TernaryTree::chain(4, PauliLabel::X).unwrap(),
                &ModeMap::Permutation(vec![2, 0, 3, 1]),
            )
            .unwrap(),
        ] {
            // A padded register, so that the identity on the spare qubit is exercised too.
            let num_qubits = enc.num_modes() + 1;
            for left in 0..2 * enc.num_modes() {
                for right in 0..2 * enc.num_modes() {
                    let (labels, indices) = enc.majorana_image(left);
                    let left_obs =
                        one_pauli_string(num_qubits, factor, as_bit_terms(labels), indices);
                    let (labels, indices) = enc.majorana_image(right);
                    let right_obs =
                        one_pauli_string(num_qubits, one, as_bit_terms(labels), indices);
                    // `qk_obs_compose(a, b)` returns `b @ a`, so this is `left @ right`.
                    let expected = unsafe { ffi::qk_obs_compose(right_obs, left_obs) };
                    unsafe { ffi::qk_obs_free(left_obs) };
                    unsafe { ffi::qk_obs_free(right_obs) };
                    assert_obs_equal(
                        majorana_product(&enc, left, right, factor, num_qubits),
                        expected,
                        num_qubits,
                    );
                }
            }
        }
    }

    /// Total parity must be the product of `1 - 2 n_j` over all modes, on any tree.
    ///
    /// `total_parity` emits the leftover leg with coefficient `+1` from a derivation rather than a
    /// product, so this is what checks the derivation: the reference is built from the Majorana
    /// strings alone. The **odd** mode counts are the load-bearing cases. Taking the product of `+i gamma gamma'`
    /// rather than `-i gamma gamma'` yields `prod_j (2 n_j - 1)`, which differs by `(-1)^N` and so
    /// agrees for every even `N`; an even-only test passes against the wrong sign.
    #[test]
    fn test_total_parity_matches_the_product_of_vertices() {
        for enc in [
            jw_encoding(3),
            balanced_encoding(3),
            jw_encoding(4),
            balanced_encoding(4),
            jw_encoding(5),
            balanced_encoding(5),
            TernaryTreeEncoding::new(
                &TernaryTree::chain(5, PauliLabel::X).unwrap(),
                &ModeMap::Permutation(vec![3, 0, 4, 1, 2]),
            )
            .unwrap(),
        ] {
            let n = enc.num_modes();
            // `V_j = 1 - 2 a^dagger_j a_j` is exactly the per-mode parity, so their product is the
            // total. Written out as the single Majorana term `(-i)^N gamma_0 gamma_1 ... gamma_2N-1`,
            // so that it is checked through the Majorana strings alone.
            let phase = Complex64::new(0.0, -1.0).powu(n);
            let all_majoranas = MajoranaOperator {
                coeffs: vec![phase],
                modes: (0..2 * n).collect(),
                boundaries: vec![0, 2 * n as usize],
                groups: None,
            };
            assert_obs_equal(
                enc.total_parity(enc.num_modes()).unwrap(),
                majorana_ternary_tree(&all_majoranas, &enc, enc.num_modes()).unwrap(),
                enc.num_modes(),
            );
        }
    }

    /// A permuted mode map must agree with relabelling the operator instead.
    #[test]
    fn test_mode_map_agrees_with_relabelling() {
        let op = fermion_op();
        let tree = TernaryTree::breadth_first(4, 3, [PauliLabel::X, PauliLabel::Y, PauliLabel::Z])
            .unwrap();
        // Node `u` serves mode `perm[u]`, so an operator on mode `j` lands where `perm[u] == j`.
        let perm = vec![2u32, 0, 3, 1];
        let permuted =
            TernaryTreeEncoding::new(&tree, &ModeMap::Permutation(perm.clone())).unwrap();

        // `relabel_modes` takes original -> new. Mode `j` must be mapped at the node serving it, so
        // the relabelling that reproduces `permuted` under the identity map sends `perm[u] -> u`.
        let mut relabel = vec![0u32; 4];
        for (node, &mode) in perm.iter().enumerate() {
            relabel[mode as usize] = node as u32;
        }
        let relabelled = op.relabel_modes(relabel).unwrap();
        let identity = TernaryTreeEncoding::new(&tree, &ModeMap::Identity).unwrap();

        assert_obs_equal(
            fermion_ternary_tree(&op, &permuted, permuted.num_modes()).unwrap(),
            fermion_ternary_tree(&relabelled, &identity, identity.num_modes()).unwrap(),
            4,
        );
    }

    #[test]
    fn test_rejects_operator_outside_the_encoding() {
        let op = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            actions: vec![true],
            modes: vec![7],
            boundaries: vec![0, 1],
            groups: None,
        };
        let err = fermion_ternary_tree(&op, &jw_encoding(4), 4).unwrap_err();
        assert!(matches!(
            err,
            TernaryTreeEncodingError::ModeOutsideEncoding {
                num_modes: 4,
                max_mode: 7
            }
        ));

        let maj = MajoranaOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            modes: vec![15],
            boundaries: vec![0, 1],
            groups: None,
        };
        // Majorana 15 acts on mode 7, so the bound is reported in modes.
        let err = majorana_ternary_tree(&maj, &jw_encoding(4), 4).unwrap_err();
        assert!(matches!(
            err,
            TernaryTreeEncodingError::ModeOutsideEncoding {
                num_modes: 4,
                max_mode: 7
            }
        ));
    }

    /// `num_qubits` below the encoding's mode count is rejected, whatever modes the operator touches.
    ///
    /// The bound is the encoding's extent, not the operator's: a mode's Pauli image spans its whole
    /// path to the root, so even an operator on mode `0` alone can address the highest qubit. A
    /// fixture whose largest mode happens to be `num_modes - 1` cannot tell the two bounds apart, so
    /// the operator here deliberately touches only the lowest mode.
    #[test]
    fn test_rejects_too_few_qubits() {
        let op = fermion_op();
        let enc = jw_encoding(4);
        // The operator needs 4 qubits and the encoding covers 4 modes, but only 3 were requested.
        let err = fermion_ternary_tree(&op, &enc, 3).unwrap_err();
        assert!(matches!(
            err,
            TernaryTreeEncodingError::NumQubitsTooSmall {
                num_qubits: 3,
                num_modes: 4
            }
        ));

        // On the balanced tree at four modes, `gamma_0`'s image is `X_0 Z_1`, so mapping a
        // number operator on mode 0 over a single qubit would build a term on qubit 1. Clamping the
        // bound to `num_qubits` used to let this through and abort the process in `qk_obs_new`.
        let mode_zero = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            actions: vec![true, false],
            modes: vec![0, 0],
            boundaries: vec![0, 2],
            groups: None,
        };
        for num_qubits in 1..4 {
            let err =
                fermion_ternary_tree(&mode_zero, &balanced_encoding(4), num_qubits).unwrap_err();
            assert!(matches!(
                err,
                TernaryTreeEncodingError::NumQubitsTooSmall {
                    num_qubits: reported,
                    num_modes: 4
                } if reported == num_qubits
            ));
        }
        // The full register is accepted, and the padded one too.
        for num_qubits in [4, 6] {
            assert!(fermion_ternary_tree(&mode_zero, &balanced_encoding(4), num_qubits).is_ok());
        }
    }

    /// An operator outside the encoding is reported against the encoding, not the register.
    ///
    /// Raising `num_qubits` cannot widen the encoding, so an error naming `num_qubits` would send a
    /// caller with plenty of qubits and an under-sized encoding after the wrong fix.
    #[test]
    fn test_reports_the_encoding_bound_when_the_register_is_ample() {
        let op = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            actions: vec![true],
            modes: vec![7],
            boundaries: vec![0, 1],
            groups: None,
        };
        let err = fermion_ternary_tree(&op, &jw_encoding(4), 10).unwrap_err();
        assert!(matches!(
            err,
            TernaryTreeEncodingError::ModeOutsideEncoding {
                num_modes: 4,
                max_mode: 7
            }
        ));
    }

    /// `total_parity` rejects an under-sized register rather than aborting the process.
    #[test]
    fn test_total_parity_rejects_too_few_qubits() {
        let enc = jw_encoding(4);
        let err = enc.total_parity(2).unwrap_err();
        assert!(matches!(
            err,
            TernaryTreeEncodingError::NumQubitsTooSmall {
                num_qubits: 2,
                num_modes: 4
            }
        ));
        // Exactly the mode count, and a padded register, are both fine.
        for num_qubits in [4, 6] {
            let parity = enc.total_parity(num_qubits).unwrap();
            assert_eq!(unsafe { ffi::qk_obs_num_qubits(parity) }, num_qubits);
            unsafe { ffi::qk_obs_free(parity) };
        }
    }

    /// Mapping against an encoding padded with spare qubits must agree with the unpadded one.
    #[test]
    fn test_padding_is_the_identity() {
        let op = fermion_op();
        let tree = TernaryTree::chain(4, PauliLabel::Z).unwrap();
        let enc = TernaryTreeEncoding::new(&tree, &ModeMap::Identity).unwrap();
        // The padding is requested at the mapping call, not baked into the encoding.
        let mapped = fermion_ternary_tree(&op, &enc, 6).unwrap();
        assert_eq!(unsafe { ffi::qk_obs_num_qubits(mapped) }, 6);
        // The same operator on 6 qubits under the direct mapper.
        assert_obs_equal(mapped, fermion_jordan_wigner(&op, 6).unwrap(), 6);
    }

    /// The balanced tree really is shallower, which is the point of the framework.
    #[test]
    fn test_balanced_tree_lowers_the_weight() {
        let n = 13;
        let single = MajoranaOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            // The last mode's gamma, whose Jordan-Wigner string spans the whole register.
            modes: vec![2 * (n - 1)],
            boundaries: vec![0, 1],
            groups: None,
        };
        let chain = fermion_weight(majorana_ternary_tree(&single, &jw_encoding(n), n).unwrap());
        let balanced =
            fermion_weight(majorana_ternary_tree(&single, &balanced_encoding(n), n).unwrap());
        assert_eq!(chain, vec![n as usize], "JW weight should be n");
        assert_eq!(
            balanced,
            vec![3],
            "balanced weight should be ceil(log3(2n+1))"
        );
    }
}
