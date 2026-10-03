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
    QkComplex64, bit_terms, compose_actions, ffi_backend as ffi, map_operator, one_pauli_string,
    qk_coeff,
};
use crate::mappers::ternary_tree::encoding::TernaryTreeEncoding;
use crate::operators::edge_vertex_operator::{EdgeAction, EdgeVertexOperator};
use crate::operators::fermion_operator::{FermionAction, FermionOperator};
use crate::operators::majorana_operator::{MajoranaAction, MajoranaOperator};
use crate::operators::transfer_vertex_operator::{TransferAction, TransferVertexOperator};
use crate::operators::{CoherenceError, OperatorTrait};

/// Maps a single Majorana operator onto its Pauli image under a ternary-tree encoding.
///
/// The image is a *single* Pauli string with coefficient exactly `1`: a leg's string is a product of
/// Paulis on distinct qubits, so it is Hermitian and squares to the identity with no phase to correct.
/// Every factor of `i` that a term produces arises when two such strings are composed, and
/// `qk_obs_compose` accumulates it there.
fn map_majorana_action(
    enc: &TernaryTreeEncoding,
    action: MajoranaAction,
    num_qubits: u32,
) -> *mut ffi::QkObs {
    let (qubits, labels) = enc.majorana_image(*action);
    one_pauli_string(
        num_qubits,
        QkComplex64 { re: 1.0, im: 0.0 },
        &mut bit_terms(labels),
        &mut qubits.to_vec(),
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
    let mode = *action.1;
    let im = if *action.0 { -0.5 } else { 0.5 };

    let (even_qubits, even_labels) = enc.majorana_image(2 * mode);
    let (odd_qubits, odd_labels) = enc.majorana_image(2 * mode + 1);

    let mut coeffs = [
        QkComplex64 { re: 0.5, im: 0.0 },
        QkComplex64 { re: 0.0, im },
    ];
    let mut terms = bit_terms(even_labels);
    terms.extend(bit_terms(odd_labels));
    let mut indices = even_qubits.to_vec();
    indices.extend_from_slice(odd_qubits);
    let mut boundaries = [
        0usize,
        even_qubits.len(),
        even_qubits.len() + odd_qubits.len(),
    ];

    unsafe {
        ffi::qk_obs_new(
            num_qubits,
            coeffs.len().try_into().unwrap(),
            terms.len().try_into().unwrap(),
            coeffs.as_mut_ptr(),
            terms.as_mut_ptr(),
            indices.as_mut_ptr(),
            boundaries.as_mut_ptr(),
        )
    }
}

/// Multiplies two Majorana images and scales the product by `factor`.
///
/// The generators of the vertex, edge and transfer algebras are each defined as a product of exactly
/// two Majorana operators, so every one of their images is a single Pauli string -- whatever the tree.
/// The relative phase of that string depends on how the two paths overlap, which is tree-dependent,
/// so it is obtained from `qk_obs_compose` rather than written down.
///
/// This is where the ternary-tree mappers genuinely diverge from their Jordan-Wigner counterparts,
/// which can shortcut the same result with a closed form: on a chain the two `Z` chains cancel below
/// the lower endpoint, leaving `Z` strictly between the two and a sign fixed by the index order. That
/// cancellation is a property of the chain, not of ternary trees (on a balanced tree a vertex
/// operator is not weight 1, and an edge operator's coefficient is not the chain's), so the product
/// is taken explicitly here.
fn majorana_product(
    enc: &TernaryTreeEncoding,
    left: u32,
    right: u32,
    factor: QkComplex64,
    num_qubits: u32,
) -> *mut ffi::QkObs {
    let left = map_majorana_action(enc, &left, num_qubits);
    let right = map_majorana_action(enc, &right, num_qubits);
    // `qk_obs_compose(second, first)` applies `first` and then `second`, so this is `left * right`.
    let product = unsafe { ffi::qk_obs_compose(right, left) };
    unsafe { ffi::qk_obs_free(left) };
    unsafe { ffi::qk_obs_free(right) };
    unsafe { ffi::qk_obs_multiply_inplace(product, &factor) };
    product
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
        return majorana_product(
            enc,
            2 * left,
            2 * left + 1,
            QkComplex64 { re: 0.0, im: -1.0 },
            num_qubits,
        );
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
        return majorana_product(
            enc,
            2 * left,
            2 * left + 1,
            QkComplex64 { re: 0.0, im: -1.0 },
            num_qubits,
        );
    }
    majorana_product(
        enc,
        2 * left + 1,
        2 * right,
        QkComplex64 { re: 0.0, im: 0.5 },
        num_qubits,
    )
}

/// Returns the largest mode index a pair of index buffers acts on.
fn max_paired_index(left_indices: &[u32], right_indices: &[u32]) -> Option<u32> {
    left_indices.iter().chain(right_indices).max().copied()
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
) -> Result<(), CoherenceError> {
    // The encoding must have a Pauli image for every mode the operator acts on. Reported against the
    // encoding's own extent, because raising `num_qubits` cannot widen it.
    if let Some(max_mode) = max_mode
        && max_mode >= enc.num_modes()
    {
        return Err(CoherenceError::NumQubitsTooSmall {
            num_qubits: enc.num_modes(),
            max_mode,
        });
    }
    // The register must hold every qubit the encoding can address, independently of `max_mode`.
    if num_qubits < enc.num_modes() {
        return Err(CoherenceError::NumQubitsTooSmallForEncoding {
            num_qubits,
            num_modes: enc.num_modes(),
        });
    }
    Ok(())
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
) -> Result<*mut ffi::QkObs, CoherenceError> {
    check_support(enc, num_qubits, fer_op.modes.iter().max().copied())?;
    Ok(map_operator(
        fer_op.iter(),
        num_qubits,
        |term, num_qubits| {
            (
                compose_actions(term.iter(), num_qubits, |action, num_qubits| {
                    map_fermion_action(enc, action, num_qubits)
                }),
                qk_coeff(term.coeff),
            )
        },
    ))
}

/// Maps a [`MajoranaOperator`] onto a qubit operator under a ternary-tree encoding.
///
/// Each Majorana operator's image is a single Pauli string, so a term of `L` of them composes `L`
/// strings rather than inflating into a sum.
pub fn majorana_ternary_tree(
    maj_op: &MajoranaOperator,
    enc: &TernaryTreeEncoding,
    num_qubits: u32,
) -> Result<*mut ffi::QkObs, CoherenceError> {
    // Majorana index `m` acts on mode `m / 2`, so it is the quotient that must be covered.
    check_support(enc, num_qubits, maj_op.modes.iter().max().map(|&m| m / 2))?;
    Ok(map_operator(
        maj_op.iter(),
        num_qubits,
        |term, num_qubits| {
            (
                compose_actions(term.iter(), num_qubits, |action, num_qubits| {
                    map_majorana_action(enc, action, num_qubits)
                }),
                qk_coeff(term.coeff),
            )
        },
    ))
}

/// Maps an [`EdgeVertexOperator`] onto a qubit operator under a ternary-tree encoding.
pub fn edge_vertex_ternary_tree(
    inter_op: &EdgeVertexOperator,
    enc: &TernaryTreeEncoding,
    num_qubits: u32,
) -> Result<*mut ffi::QkObs, CoherenceError> {
    check_support(
        enc,
        num_qubits,
        max_paired_index(&inter_op.left_indices, &inter_op.right_indices),
    )?;
    Ok(map_operator(
        inter_op.iter(),
        num_qubits,
        |term, num_qubits| {
            (
                compose_actions(term.iter(), num_qubits, |action, num_qubits| {
                    map_edge_action(enc, action, num_qubits)
                }),
                qk_coeff(term.coeff),
            )
        },
    ))
}

/// Maps a [`TransferVertexOperator`] onto a qubit operator under a ternary-tree encoding.
pub fn transfer_vertex_ternary_tree(
    inter_op: &TransferVertexOperator,
    enc: &TernaryTreeEncoding,
    num_qubits: u32,
) -> Result<*mut ffi::QkObs, CoherenceError> {
    check_support(
        enc,
        num_qubits,
        max_paired_index(&inter_op.left_indices, &inter_op.right_indices),
    )?;
    Ok(map_operator(
        inter_op.iter(),
        num_qubits,
        |term, num_qubits| {
            (
                compose_actions(term.iter(), num_qubits, |action, num_qubits| {
                    map_transfer_action(enc, action, num_qubits)
                }),
                qk_coeff(term.coeff),
            )
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::mappers::library::jordan_wigner::{
        edge_vertex_jordan_wigner, fermion_jordan_wigner, majorana_jordan_wigner,
        transfer_vertex_jordan_wigner,
    };
    use crate::mappers::qk_obs::compact;
    use crate::mappers::ternary_tree::PauliLabel;
    use crate::mappers::ternary_tree::TernaryTree;
    use crate::mappers::ternary_tree::encoding::ModeMap;
    use num_complex::Complex64;

    /// Asserts that two observables are equal, consuming both.
    fn assert_obs_equal(got: *mut ffi::QkObs, expected: *mut ffi::QkObs, num_qubits: u32) {
        let factor = QkComplex64 { re: -1.0, im: 0.0 };
        let negated = unsafe { ffi::qk_obs_multiply(expected, &factor) };
        let diff = unsafe { ffi::qk_obs_add(got, negated) };
        let diff = unsafe { ffi::qk_obs_canonicalize(diff, 1e-9) };
        let zero = unsafe { ffi::qk_obs_zero(num_qubits) };
        assert!(unsafe { ffi::qk_obs_equal(diff, zero) });
    }

    /// The Jordan-Wigner encoding, as a ternary tree.
    fn jw_encoding(n: u32) -> TernaryTreeEncoding {
        let tree = TernaryTree::chain(n, PauliLabel::Z).unwrap();
        TernaryTreeEncoding::new(&tree, &ModeMap::Identity).unwrap()
    }

    /// The balanced (JKMN) encoding.
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

    /// The vertex operator is diagonal on every tree, but only weight 1 on the chain.
    ///
    /// This is the concrete form of the warning on [`majorana_product`]: the Jordan-Wigner closed form
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

    /// Total parity must be the product of `1 - 2 n_j` over all modes, on any tree.
    ///
    /// The **odd** mode counts are the load-bearing cases. Taking the product of `+i gamma gamma'`
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
        ] {
            let n = enc.num_modes();
            // `V_j = 1 - 2 a^dagger_j a_j` is exactly the per-mode parity, so their product is the
            // total. Built as a single edge-vertex term so the composition happens in the mapper.
            let all_vertices = EdgeVertexOperator {
                coeffs: vec![Complex64::new(1.0, 0.0)],
                left_indices: (0..n).collect(),
                right_indices: (0..n).collect(),
                boundaries: vec![0, n as usize],
                groups: None,
            };
            assert_obs_equal(
                enc.total_parity(enc.num_modes()).unwrap(),
                edge_vertex_ternary_tree(&all_vertices, &enc, enc.num_modes()).unwrap(),
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
            CoherenceError::NumQubitsTooSmall {
                num_qubits: 4,
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
            CoherenceError::NumQubitsTooSmall {
                num_qubits: 4,
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
            CoherenceError::NumQubitsTooSmallForEncoding {
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
                CoherenceError::NumQubitsTooSmallForEncoding {
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
    /// Raising `num_qubits` cannot widen the encoding, so reporting the clamped bound would send a
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
            CoherenceError::NumQubitsTooSmall {
                num_qubits: 4,
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
            CoherenceError::NumQubitsTooSmallForEncoding {
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
