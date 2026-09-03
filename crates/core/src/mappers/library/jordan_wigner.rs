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
    QK_BIT_TERM_X, QK_BIT_TERM_Y, QK_BIT_TERM_Z, QkComplex64, compose_actions, ffi_backend as ffi,
    map_operator, one_pauli_string, qk_coeff,
};
use crate::operators::edge_vertex_operator::{EdgeAction, EdgeVertexOperator};
use crate::operators::fermion_operator::{FermionAction, FermionOperator};
use crate::operators::majorana_operator::{MajoranaAction, MajoranaOperator};
use crate::operators::transfer_vertex_operator::{TransferAction, TransferVertexOperator};
use crate::operators::{CoherenceError, OperatorTrait};

/// Maps a single fermionic action onto its Pauli image.
///
/// Unlike the generators of the other three algebras, a creation or annihilation operator maps onto
/// a *two*-term sum,
///
/// ```text
///     a^dagger_j  ->  Z_0 ... Z_{j-1} (X_j - i Y_j) / 2
///     a_j         ->  Z_0 ... Z_{j-1} (X_j + i Y_j) / 2
/// ```
///
/// which is why composing a term of `L` fermionic actions can produce up to `2^L` Pauli strings.
fn map_fermion_action(action: FermionAction, num_qubits: u32) -> *mut ffi::QkObs {
    let fer_idx = *action.1 as usize;
    let im = if *action.0 { -0.5 } else { 0.5 };
    let mut coeffs: Vec<QkComplex64> = vec![
        QkComplex64 { re: 0.5, im: 0.0 },
        QkComplex64 { re: 0.0, im },
    ];

    let mut bit_terms = Vec::<ffi::QkBitTerm>::new();
    let mut indices = Vec::<u32>::new();
    for qb_idx in 0..fer_idx {
        bit_terms.push(QK_BIT_TERM_Z);
        indices.push(qb_idx as u32);
    }
    bit_terms.push(QK_BIT_TERM_X);
    indices.push(fer_idx as u32);
    for qb_idx in 0..fer_idx {
        bit_terms.push(QK_BIT_TERM_Z);
        indices.push(qb_idx as u32);
    }
    bit_terms.push(QK_BIT_TERM_Y);
    indices.push(fer_idx as u32);

    let mut boundaries: Vec<usize> = vec![0, fer_idx + 1, 2 * fer_idx + 2];

    unsafe {
        ffi::qk_obs_new(
            num_qubits,
            coeffs.len().try_into().unwrap(),
            bit_terms.len().try_into().unwrap(),
            coeffs.as_mut_ptr(),
            bit_terms.as_mut_ptr(),
            indices.as_mut_ptr(),
            boundaries.as_mut_ptr(),
        )
    }
}

pub fn fermion_jordan_wigner(
    fer_op: &FermionOperator,
    num_qubits: u32,
) -> Result<*mut ffi::QkObs, CoherenceError> {
    // Each mode index `j` maps onto qubit `j`, so the operator's largest mode index must fit
    // within `num_qubits`. Without this check, the underlying `qk_obs_*` calls receive an
    // out-of-range qubit index and abort the process with a non-unwinding panic.
    if let Some(&max_mode) = fer_op.modes.iter().max()
        && max_mode >= num_qubits
    {
        return Err(CoherenceError::NumQubitsTooSmall {
            num_qubits,
            max_mode,
        });
    }

    Ok(map_operator(
        fer_op.iter(),
        num_qubits,
        |term, num_qubits| {
            (
                compose_actions(term.iter(), num_qubits, map_fermion_action),
                qk_coeff(term.coeff),
            )
        },
    ))
}

/// Appends the Jordan-Wigner Z-string on the qubits below `qubit` to `bit_terms`/`indices`.
fn push_z_string(bit_terms: &mut Vec<ffi::QkBitTerm>, indices: &mut Vec<u32>, qubit: u32) {
    for qb_idx in 0..qubit {
        bit_terms.push(QK_BIT_TERM_Z);
        indices.push(qb_idx);
    }
}

/// Returns the largest fermionic mode index the two index buffers of a vertex-type operator act on.
fn max_paired_index(left_indices: &[u32], right_indices: &[u32]) -> Option<u32> {
    // Both endpoints must be checked: an operator whose largest index only ever appears on the right
    // would otherwise pass a left-only check and abort inside the `qk_obs_*` calls.
    left_indices
        .iter()
        .max()
        .into_iter()
        .chain(right_indices.iter().max())
        .max()
        .copied()
}

/// Maps a single Majorana operator onto its Pauli image.
///
/// With the [`MajoranaOperator`] convention that even indices carry `gamma = a^dagger + a` and odd
/// ones `gamma' = i(a^dagger - a)`, Majorana index `m` acts on fermionic mode `m / 2` and
///
/// ```text
///     gamma_m  ->  Z_0 ... Z_{idx-1} (X_idx if m even else Y_idx),  idx = m / 2
/// ```
///
/// with coefficient exactly `1`. This is a *single* Pauli string, where a fermionic action maps onto
/// a two-term sum -- so routing a term of `L` Majorana operators through a [`FermionOperator`] first
/// would inflate one Pauli string into up to `4^L` terms before merging them back down. The saving
/// grows with the length of the terms; for single-operator terms the two routes cost about the same.
fn map_majorana_action(action: MajoranaAction, num_qubits: u32) -> *mut ffi::QkObs {
    let qubit = *action / 2;

    let mut bit_terms = Vec::<ffi::QkBitTerm>::new();
    let mut indices = Vec::<u32>::new();
    push_z_string(&mut bit_terms, &mut indices, qubit);
    bit_terms.push(if action.is_multiple_of(2) {
        QK_BIT_TERM_X
    } else {
        QK_BIT_TERM_Y
    });
    indices.push(qubit);

    one_pauli_string(
        num_qubits,
        QkComplex64 { re: 1.0, im: 0.0 },
        &mut bit_terms,
        &mut indices,
    )
}

pub fn majorana_jordan_wigner(
    maj_op: &MajoranaOperator,
    num_qubits: u32,
) -> Result<*mut ffi::QkObs, CoherenceError> {
    // Majorana index `m` acts on fermionic mode `m / 2`, so it is that quotient -- not the index
    // itself -- which has to fit within `num_qubits`. `max_mode` reports the quotient for the same
    // reason: the error speaks of a "mode index", and `gamma_7` needs 4 qubits, not 8.
    if let Some(&max_maj) = maj_op.modes.iter().max()
        && max_maj / 2 >= num_qubits
    {
        return Err(CoherenceError::NumQubitsTooSmall {
            num_qubits,
            max_mode: max_maj / 2,
        });
    }

    Ok(map_operator(
        maj_op.iter(),
        num_qubits,
        |term, num_qubits| {
            (
                compose_actions(term.iter(), num_qubits, map_majorana_action),
                qk_coeff(term.coeff),
            )
        },
    ))
}

/// Maps a single generalized edge operator onto its Pauli image.
///
/// Writing `lo`/`hi` for the smaller/larger of the two indices, the images are
///
/// ```text
///     V_l = E_ll  ->  Z_l                                          (coefficient  1)
///     E_lr        ->  Y_lo Z_{lo+1} ... Z_{hi-1} X_hi              (coefficient -1 if l < r, else +1)
/// ```
///
/// Note what does and does not change with the index order: the *string* is the same either way and
/// only the **sign** flips, which is the antisymmetry `E_lr = -E_rl`. Contrast
/// [`map_transfer_action`], where the sign is fixed and the Pauli letters change instead.
///
/// The `Z` chains of the two Majoranas cancel below `lo`, leaving `Z` only strictly between the
/// endpoints, so the image has weight `hi - lo + 1` rather than `O(hi)`.
///
/// # Convention
///
/// This differs from Eq. (10) of [1]_ (`arXiv:2512.11418v2`), which gives `E_{j,j+1} = X_j Y_{j+1}`
/// and `T_{j,j+1} = -Y_j Y_{j+1} / 2`, by an exchange of `X` and `Y` on the endpoints -- a
/// single-qubit basis choice that the paper itself notes ("the provided circuits prepare the
/// stabilizers in the X,Z basis instead of the X,Y basis"). Both conventions satisfy every defining
/// relation of the algebra; the one used here is the one consistent with this crate's
/// [`edge_vertex_to_fermion`](super::edge_vertex::edge_vertex_to_fermion) and its `gamma'` sign, so
/// that mapping an operator directly agrees with converting it to a [`FermionOperator`] first.
fn map_edge_action(action: EdgeAction, num_qubits: u32) -> *mut ffi::QkObs {
    let (left, right) = (*action.0, *action.1);

    if left == right {
        // A vertex, `V_l = 1 - 2 a^dagger_l a_l`, is diagonal: weight-1 with no Z-string at all.
        return one_pauli_string(
            num_qubits,
            QkComplex64 { re: 1.0, im: 0.0 },
            &mut [QK_BIT_TERM_Z],
            &mut [left],
        );
    }

    let (lo, hi) = (left.min(right), left.max(right));
    let re = if left < right { -1.0 } else { 1.0 };

    let mut bit_terms = vec![QK_BIT_TERM_Y];
    let mut indices = vec![lo];
    for qb_idx in (lo + 1)..hi {
        bit_terms.push(QK_BIT_TERM_Z);
        indices.push(qb_idx);
    }
    bit_terms.push(QK_BIT_TERM_X);
    indices.push(hi);

    one_pauli_string(
        num_qubits,
        QkComplex64 { re, im: 0.0 },
        &mut bit_terms,
        &mut indices,
    )
}

pub fn edge_vertex_jordan_wigner(
    inter_op: &EdgeVertexOperator,
    num_qubits: u32,
) -> Result<*mut ffi::QkObs, CoherenceError> {
    if let Some(max_mode) = max_paired_index(&inter_op.left_indices, &inter_op.right_indices)
        && max_mode >= num_qubits
    {
        return Err(CoherenceError::NumQubitsTooSmall {
            num_qubits,
            max_mode,
        });
    }

    Ok(map_operator(
        inter_op.iter(),
        num_qubits,
        |term, num_qubits| {
            (
                compose_actions(term.iter(), num_qubits, map_edge_action),
                qk_coeff(term.coeff),
            )
        },
    ))
}

/// Maps a single generalized transfer operator onto its Pauli image.
///
/// Writing `lo`/`hi` for the smaller/larger of the two indices, the images are
///
/// ```text
///     V_l = T_ll  ->  Z_l                                          (coefficient    1)
///     T_lr        ->  P_lo Z_{lo+1} ... Z_{hi-1} P_hi              (coefficient -1/2)
///                     with P = X if l < r, else Y
/// ```
///
/// The index order works the opposite way round to [`map_edge_action`]: the coefficient is `-1/2`
/// for **both** orientations and it is the Pauli *letters* that swap. `T_lr` and `T_rl` are
/// genuinely different operators (there is no antisymmetry to exploit), which is why both directions
/// are stored rather than canonicalized.
///
/// See [`map_edge_action`] for how this convention relates to the one in `arXiv:2512.11418v2`.
fn map_transfer_action(action: TransferAction, num_qubits: u32) -> *mut ffi::QkObs {
    let (left, right) = (*action.0, *action.1);

    if left == right {
        return one_pauli_string(
            num_qubits,
            QkComplex64 { re: 1.0, im: 0.0 },
            &mut [QK_BIT_TERM_Z],
            &mut [left],
        );
    }

    let (lo, hi) = (left.min(right), left.max(right));
    let endpoint = if left < right {
        QK_BIT_TERM_X
    } else {
        QK_BIT_TERM_Y
    };

    let mut bit_terms = vec![endpoint];
    let mut indices = vec![lo];
    for qb_idx in (lo + 1)..hi {
        bit_terms.push(QK_BIT_TERM_Z);
        indices.push(qb_idx);
    }
    bit_terms.push(endpoint);
    indices.push(hi);

    one_pauli_string(
        num_qubits,
        QkComplex64 { re: -0.5, im: 0.0 },
        &mut bit_terms,
        &mut indices,
    )
}

pub fn transfer_vertex_jordan_wigner(
    inter_op: &TransferVertexOperator,
    num_qubits: u32,
) -> Result<*mut ffi::QkObs, CoherenceError> {
    if let Some(max_mode) = max_paired_index(&inter_op.left_indices, &inter_op.right_indices)
        && max_mode >= num_qubits
    {
        return Err(CoherenceError::NumQubitsTooSmall {
            num_qubits,
            max_mode,
        });
    }

    Ok(map_operator(
        inter_op.iter(),
        num_qubits,
        |term, num_qubits| {
            (
                compose_actions(term.iter(), num_qubits, map_transfer_action),
                qk_coeff(term.coeff),
            )
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::mappers::library::edge_vertex::edge_vertex_to_fermion;
    use crate::mappers::library::majorana_fermion::majorana_to_fermion;
    use crate::mappers::library::transfer_vertex::transfer_vertex_to_fermion;
    use crate::mappers::qk_obs::compact;
    use num_complex::Complex64;

    /// Asserts that two observables are equal, consuming both.
    ///
    /// Compares by canonicalizing the difference against zero rather than structurally, since the
    /// mappers make no promise about the order or multiplicity of the terms they emit.
    fn assert_obs_equal(got: *mut ffi::QkObs, expected: *mut ffi::QkObs, num_qubits: u32) {
        let factor = QkComplex64 { re: -1.0, im: 0.0 };
        let negated = unsafe { ffi::qk_obs_multiply(expected, &factor) };
        let diff = unsafe { ffi::qk_obs_add(got, negated) };
        let diff = unsafe { ffi::qk_obs_canonicalize(diff, 1e-9) };
        let zero = unsafe { ffi::qk_obs_zero(num_qubits) };
        assert!(unsafe { ffi::qk_obs_equal(diff, zero) });
    }

    /// Builds the single-Pauli-string observable `coeff * (bit_terms @ indices)`.
    ///
    /// Every generator handled by the three mappers below has a one-string image, so the expectations
    /// are all built through this rather than through the multi-term arrays the fermionic tests need.
    fn expected_string(
        num_qubits: u32,
        re: f64,
        im: f64,
        bit_terms: &[ffi::QkBitTerm],
        indices: &[u32],
    ) -> *mut ffi::QkObs {
        one_pauli_string(
            num_qubits,
            QkComplex64 { re, im },
            &mut bit_terms.to_vec(),
            &mut indices.to_vec(),
        )
    }

    fn majorana_op(
        coeffs: Vec<Complex64>,
        modes: Vec<u32>,
        boundaries: Vec<usize>,
    ) -> MajoranaOperator {
        MajoranaOperator {
            coeffs,
            modes,
            boundaries,
            groups: None,
        }
    }

    /// Builds a single-term operator over one generator, for the image tests.
    fn one_majorana(mode: u32) -> MajoranaOperator {
        majorana_op(vec![Complex64::new(1.0, 0.0)], vec![mode], vec![0, 1])
    }

    fn edge_op(
        coeffs: Vec<Complex64>,
        left_indices: Vec<u32>,
        right_indices: Vec<u32>,
        boundaries: Vec<usize>,
    ) -> EdgeVertexOperator {
        EdgeVertexOperator {
            coeffs,
            left_indices,
            right_indices,
            boundaries,
            groups: None,
        }
    }

    fn one_edge(left: u32, right: u32) -> EdgeVertexOperator {
        edge_op(
            vec![Complex64::new(1.0, 0.0)],
            vec![left],
            vec![right],
            vec![0, 1],
        )
    }

    fn transfer_op(
        coeffs: Vec<Complex64>,
        left_indices: Vec<u32>,
        right_indices: Vec<u32>,
        boundaries: Vec<usize>,
    ) -> TransferVertexOperator {
        TransferVertexOperator {
            coeffs,
            left_indices,
            right_indices,
            boundaries,
            groups: None,
        }
    }

    fn one_transfer(left: u32, right: u32) -> TransferVertexOperator {
        transfer_op(
            vec![Complex64::new(1.0, 0.0)],
            vec![left],
            vec![right],
            vec![0, 1],
        )
    }

    #[test]
    fn test_majorana_jordan_wigner_images() {
        // `gamma_0` is the boundary case: mode 0 has an empty Z-string, so the image is a bare `X_0`.
        assert_obs_equal(
            majorana_jordan_wigner(&one_majorana(0), 3).unwrap(),
            expected_string(3, 1.0, 0.0, &[QK_BIT_TERM_X], &[0]),
            3,
        );
        // Even index -> X, odd -> Y, both on mode `m / 2` behind a full Z-string.
        assert_obs_equal(
            majorana_jordan_wigner(&one_majorana(4), 3).unwrap(),
            expected_string(
                3,
                1.0,
                0.0,
                &[QK_BIT_TERM_Z, QK_BIT_TERM_Z, QK_BIT_TERM_X],
                &[0, 1, 2],
            ),
            3,
        );
        assert_obs_equal(
            majorana_jordan_wigner(&one_majorana(5), 3).unwrap(),
            expected_string(
                3,
                1.0,
                0.0,
                &[QK_BIT_TERM_Z, QK_BIT_TERM_Z, QK_BIT_TERM_Y],
                &[0, 1, 2],
            ),
            3,
        );
    }

    #[test]
    fn test_edge_vertex_jordan_wigner_images() {
        // A vertex is diagonal: weight-1, no Z-string.
        assert_obs_equal(
            edge_vertex_jordan_wigner(&one_edge(2, 2), 4).unwrap(),
            expected_string(4, 1.0, 0.0, &[QK_BIT_TERM_Z], &[2]),
            4,
        );
        // The Z-string survives only strictly *between* the endpoints; the two Majorana chains cancel
        // below the lower one.
        assert_obs_equal(
            edge_vertex_jordan_wigner(&one_edge(0, 3), 4).unwrap(),
            expected_string(
                4,
                -1.0,
                0.0,
                &[QK_BIT_TERM_Y, QK_BIT_TERM_Z, QK_BIT_TERM_Z, QK_BIT_TERM_X],
                &[0, 1, 2, 3],
            ),
            4,
        );
        // Reversing the indices keeps the string and flips only the sign -- this is `E_lr = -E_rl`,
        // and it is the likeliest place for a sign slip.
        assert_obs_equal(
            edge_vertex_jordan_wigner(&one_edge(3, 0), 4).unwrap(),
            expected_string(
                4,
                1.0,
                0.0,
                &[QK_BIT_TERM_Y, QK_BIT_TERM_Z, QK_BIT_TERM_Z, QK_BIT_TERM_X],
                &[0, 1, 2, 3],
            ),
            4,
        );
        // Adjacent endpoints leave the interior Z-string empty.
        assert_obs_equal(
            edge_vertex_jordan_wigner(&one_edge(1, 2), 4).unwrap(),
            expected_string(4, -1.0, 0.0, &[QK_BIT_TERM_Y, QK_BIT_TERM_X], &[1, 2]),
            4,
        );
    }

    #[test]
    fn test_transfer_vertex_jordan_wigner_images() {
        assert_obs_equal(
            transfer_vertex_jordan_wigner(&one_transfer(1, 1), 3).unwrap(),
            expected_string(3, 1.0, 0.0, &[QK_BIT_TERM_Z], &[1]),
            3,
        );
        // Unlike the edge operator, reversing the indices leaves the coefficient at `-0.5` and swaps
        // the Pauli letters instead. Both orientations are `-0.5`; a reviewer's instinct to expect a
        // sign flip here is what these two assertions are guarding against.
        assert_obs_equal(
            transfer_vertex_jordan_wigner(&one_transfer(0, 2), 3).unwrap(),
            expected_string(
                3,
                -0.5,
                0.0,
                &[QK_BIT_TERM_X, QK_BIT_TERM_Z, QK_BIT_TERM_X],
                &[0, 1, 2],
            ),
            3,
        );
        assert_obs_equal(
            transfer_vertex_jordan_wigner(&one_transfer(2, 0), 3).unwrap(),
            expected_string(
                3,
                -0.5,
                0.0,
                &[QK_BIT_TERM_Y, QK_BIT_TERM_Z, QK_BIT_TERM_Y],
                &[0, 1, 2],
            ),
            3,
        );
    }

    #[test]
    fn test_edge_vertex_jordan_wigner_composition_order() {
        // `E_01 E_12` pins the operand order of the `qk_obs_compose` call: composing the two images
        // left to right gives `(-Y_0 X_1)(-Y_1 X_2) = Y_0 (X_1 Y_1) X_2 = +i (Y_0 Z_1 X_2)`, while the
        // reversed order would yield `-i` instead. A word of non-commuting factors is the only thing
        // that can catch this, so it cannot be folded into the single-generator tests above.
        let op = edge_op(
            vec![Complex64::new(1.0, 0.0)],
            vec![0, 1],
            vec![1, 2],
            vec![0, 2],
        );
        assert_obs_equal(
            edge_vertex_jordan_wigner(&op, 3).unwrap(),
            expected_string(
                3,
                0.0,
                1.0,
                &[QK_BIT_TERM_Y, QK_BIT_TERM_Z, QK_BIT_TERM_X],
                &[0, 1, 2],
            ),
            3,
        );
    }

    #[test]
    fn test_vertex_squared_maps_to_identity() {
        // `V_1 V_1 = 1`, so the image is the identity: a term with *no* bit terms at all. This is the
        // shape `WEIGHT_PER_TERM` exists to account for, and repeated indices make it arise naturally
        // here rather than only from an empty input term.
        let op = edge_op(
            vec![Complex64::new(1.0, 0.0)],
            vec![1, 1],
            vec![1, 1],
            vec![0, 2],
        );
        assert_obs_equal(
            edge_vertex_jordan_wigner(&op, 3).unwrap(),
            unsafe { ffi::qk_obs_identity(3) },
            3,
        );
    }

    #[test]
    fn test_majorana_jordan_wigner_num_qubits_too_small() {
        // `gamma_7` acts on fermionic mode 3, so it needs 4 qubits -- and the error reports the
        // *mode*, 3, not the Majorana index 7.
        let err = majorana_jordan_wigner(&one_majorana(7), 3).unwrap_err();
        assert!(matches!(
            err,
            CoherenceError::NumQubitsTooSmall {
                num_qubits: 3,
                max_mode: 3
            }
        ));
        assert!(majorana_jordan_wigner(&one_majorana(7), 4).is_ok());

        // The even index of the same mode truncates to the same bound.
        let err = majorana_jordan_wigner(&one_majorana(6), 3).unwrap_err();
        assert!(matches!(
            err,
            CoherenceError::NumQubitsTooSmall {
                num_qubits: 3,
                max_mode: 3
            }
        ));
    }

    #[test]
    fn test_vertex_jordan_wigner_num_qubits_too_small() {
        // The largest index sits in the *right* buffer, which a left-only bounds check would miss --
        // and missing it aborts the process inside `qk_obs_*` rather than returning an error.
        let err = edge_vertex_jordan_wigner(&one_edge(0, 3), 3).unwrap_err();
        assert!(matches!(
            err,
            CoherenceError::NumQubitsTooSmall {
                num_qubits: 3,
                max_mode: 3
            }
        ));
        assert!(edge_vertex_jordan_wigner(&one_edge(0, 3), 4).is_ok());

        let err = transfer_vertex_jordan_wigner(&one_transfer(0, 3), 3).unwrap_err();
        assert!(matches!(
            err,
            CoherenceError::NumQubitsTooSmall {
                num_qubits: 3,
                max_mode: 3
            }
        ));
        assert!(transfer_vertex_jordan_wigner(&one_transfer(0, 3), 4).is_ok());
    }

    /// Number of qubits used by the cross-validation tests below.
    const CROSS_NUM_QUBITS: u32 = 3;

    #[test]
    fn test_majorana_jordan_wigner_matches_route_via_fermion() {
        // Cross-validates the direct Pauli images against converting to a `FermionOperator` first and
        // mapping that. The two routes share no code, so a sign error in the image table cannot cancel
        // out -- which is what makes this the load-bearing test for the algebra.
        //
        // The converter route is *not* the production path precisely because it expands each generator
        // into a two-term fermionic sum, costing `4^L` Pauli terms for a length-`L` word where the
        // direct mapper emits one; the operators here are kept tiny so that blowup stays cheap.
        //
        // Cases worth their place: an odd-length word (not Hermitian, and where a stray factor of `i`
        // would hide), a repeated index (the Z-strings cancel to the identity), a complex coefficient
        // (catches a conjugation error), and a multi-term operator (catches scaling the accumulator
        // rather than the term).
        let ops = [
            one_majorana(0),
            one_majorana(5),
            majorana_op(vec![Complex64::new(1.0, 0.0)], vec![0, 3], vec![0, 2]),
            majorana_op(vec![Complex64::new(1.0, 0.0)], vec![2, 5, 1], vec![0, 3]),
            majorana_op(vec![Complex64::new(1.0, 0.0)], vec![1, 1], vec![0, 2]),
            majorana_op(vec![Complex64::new(-0.5, 2.0)], vec![0, 3], vec![0, 2]),
            majorana_op(
                vec![
                    Complex64::new(2.0, 0.0),
                    Complex64::new(0.0, -1.5),
                    Complex64::new(0.75, 0.0),
                ],
                vec![0, 4, 1, 3, 2],
                vec![0, 1, 3, 5],
            ),
            // A bare coefficient, i.e. an empty term, maps onto the identity.
            majorana_op(vec![Complex64::new(1.25, 0.0)], vec![], vec![0, 0]),
        ];

        for op in ops {
            let direct = majorana_jordan_wigner(&op, CROSS_NUM_QUBITS).unwrap();
            let via_fermion =
                fermion_jordan_wigner(&majorana_to_fermion(&op), CROSS_NUM_QUBITS).unwrap();
            assert_obs_equal(direct, via_fermion, CROSS_NUM_QUBITS);
        }
    }

    #[test]
    fn test_edge_vertex_jordan_wigner_matches_route_via_fermion() {
        // See `test_majorana_jordan_wigner_matches_route_via_fermion` for why this oracle is used.
        // Both index orderings of the same pair appear together in the last operator, so an
        // antisymmetry error cannot hide by being present on both sides of the comparison.
        let ops = [
            one_edge(0, 0),
            one_edge(0, 1),
            one_edge(1, 0),
            one_edge(0, 2),
            edge_op(
                vec![Complex64::new(1.0, 0.0)],
                vec![1, 1],
                vec![1, 2],
                vec![0, 2],
            ),
            edge_op(
                vec![Complex64::new(-0.5, 2.0)],
                vec![0, 1],
                vec![1, 2],
                vec![0, 2],
            ),
            edge_op(
                vec![
                    Complex64::new(2.0, 0.0),
                    Complex64::new(0.5, 0.0),
                    Complex64::new(0.0, -1.0),
                ],
                vec![0, 0, 2],
                vec![0, 2, 0],
                vec![0, 1, 2, 3],
            ),
        ];

        for op in ops {
            let direct = edge_vertex_jordan_wigner(&op, CROSS_NUM_QUBITS).unwrap();
            let via_fermion =
                fermion_jordan_wigner(&edge_vertex_to_fermion(&op), CROSS_NUM_QUBITS).unwrap();
            assert_obs_equal(direct, via_fermion, CROSS_NUM_QUBITS);
        }
    }

    #[test]
    fn test_transfer_vertex_jordan_wigner_matches_route_via_fermion() {
        // See `test_majorana_jordan_wigner_matches_route_via_fermion`. `T_lr` and `T_rl` are genuinely
        // different operators, so both orientations are checked individually and together.
        let ops = [
            one_transfer(0, 0),
            one_transfer(0, 1),
            one_transfer(1, 0),
            one_transfer(0, 2),
            one_transfer(2, 0),
            transfer_op(
                vec![Complex64::new(1.0, 0.0)],
                vec![2, 0],
                vec![2, 2],
                vec![0, 2],
            ),
            transfer_op(
                vec![Complex64::new(-0.5, 2.0)],
                vec![0, 1],
                vec![1, 2],
                vec![0, 2],
            ),
            transfer_op(
                vec![
                    Complex64::new(2.0, 0.0),
                    Complex64::new(0.5, 0.0),
                    Complex64::new(0.25, 0.0),
                    Complex64::new(0.0, -1.0),
                ],
                vec![0, 0, 1, 1],
                vec![0, 1, 0, 2],
                vec![0, 1, 2, 3, 4],
            ),
        ];

        for op in ops {
            let direct = transfer_vertex_jordan_wigner(&op, CROSS_NUM_QUBITS).unwrap();
            let via_fermion =
                fermion_jordan_wigner(&transfer_vertex_to_fermion(&op), CROSS_NUM_QUBITS).unwrap();
            assert_obs_equal(direct, via_fermion, CROSS_NUM_QUBITS);
        }
    }

    #[test]
    fn test_vertex_jordan_wigner_merges_duplicate_terms() {
        // The accumulators grow by concatenation, so without the periodic merge the result would scale
        // with the number of terms mapped rather than the number of *distinct* Pauli strings. These
        // operators make the point sharply: every vertex maps onto `Z_0`, so the whole input collapses
        // onto a single term however many copies there are.
        //
        // Sized as the fermionic sibling tests are: far enough past `MIN_COMPACTION_FLOOR` for the
        // growth trigger to fire repeatedly, and no further.
        let num_repeats = 1 << 17;
        let op = edge_op(
            vec![Complex64::new(1.0, 0.0); num_repeats],
            vec![0; num_repeats],
            vec![0; num_repeats],
            (0..=num_repeats).collect(),
        );

        let qb_op = edge_vertex_jordan_wigner(&op, 2).unwrap();
        let num_terms = unsafe { ffi::qk_obs_num_terms(qb_op) } as usize;
        assert!(
            num_terms < num_repeats / 8,
            "expected duplicates to be merged, got {num_terms} terms from {num_repeats} copies"
        );

        // ... and the operator must still be `num_repeats * Z_0`.
        assert_obs_equal(
            qb_op,
            expected_string(2, num_repeats as f64, 0.0, &[QK_BIT_TERM_Z], &[0]),
            2,
        );
    }

    #[test]
    fn test_jordan_wigner() {
        let fer_op = FermionOperator {
            coeffs: vec![
                -1.2563390730032502,
                -1.2563390730032502,
                -2.3575299028703285e-16,
                -2.3575299028703285e-16,
                -2.3575299028703285e-16,
                -2.3575299028703285e-16,
                -0.4718960072811406,
                -0.4718960072811406,
                0.33785507740175824,
                0.33785507740175824,
                0.33785507740175824,
                0.33785507740175824,
                0.09046559989211567,
                0.09046559989211567,
                0.09046559989211567,
                0.09046559989211567,
                0.09046559989211567,
                0.09046559989211567,
                0.09046559989211567,
                0.09046559989211567,
                0.09046559989211567,
                0.09046559989211567,
                0.09046559989211567,
                0.09046559989211567,
                0.09046559989211567,
                0.09046559989211567,
                0.09046559989211567,
                0.09046559989211567,
                0.3322908651276483,
                0.3322908651276483,
                0.3322908651276483,
                0.3322908651276483,
                0.3322908651276483,
                0.3322908651276483,
                0.3322908651276483,
                0.3322908651276483,
                0.34928686136600917,
                0.34928686136600917,
                0.34928686136600917,
                0.34928686136600917,
            ]
            .iter()
            .map(|c| Complex64::new(*c, 0.0))
            .collect(),
            actions: vec![
                true, false, true, false, true, false, true, false, true, false, true, false, true,
                false, true, false, true, true, false, false, true, true, false, false, true, true,
                false, false, true, true, false, false, true, true, false, false, true, true,
                false, false, true, true, false, false, true, true, false, false, true, true,
                false, false, true, true, false, false, true, true, false, false, true, true,
                false, false, true, true, false, false, true, true, false, false, true, true,
                false, false, true, true, false, false, true, true, false, false, true, true,
                false, false, true, true, false, false, true, true, false, false, true, true,
                false, false, true, true, false, false, true, true, false, false, true, true,
                false, false, true, true, false, false, true, true, false, false, true, true,
                false, false, true, true, false, false, true, true, false, false, true, true,
                false, false, true, true, false, false, true, true, false, false,
            ],
            modes: vec![
                0, 0, 2, 2, 1, 0, 0, 1, 3, 2, 2, 3, 1, 1, 3, 3, 0, 0, 0, 0, 2, 0, 0, 2, 0, 2, 2, 0,
                2, 2, 2, 2, 1, 1, 0, 0, 3, 1, 0, 2, 1, 3, 2, 0, 3, 3, 2, 2, 0, 1, 0, 1, 2, 1, 0, 3,
                0, 3, 2, 1, 2, 3, 2, 3, 1, 0, 1, 0, 3, 0, 1, 2, 1, 2, 3, 0, 3, 2, 3, 2, 0, 0, 1, 1,
                2, 0, 1, 3, 0, 2, 3, 1, 2, 2, 3, 3, 1, 0, 0, 1, 3, 0, 0, 3, 1, 2, 2, 1, 3, 2, 2, 3,
                0, 1, 1, 0, 2, 1, 1, 2, 0, 3, 3, 0, 2, 3, 3, 2, 1, 1, 1, 1, 3, 1, 1, 3, 1, 3, 3, 1,
                3, 3, 3, 3,
            ],
            boundaries: vec![
                0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68,
                72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140,
                144,
            ],
            groups: None,
        };
        let qb_op = fermion_jordan_wigner(&fer_op, 4).unwrap();

        let mut coeffs: Vec<QkComplex64> = vec![
            QkComplex64 {
                re: -0.8105479805373261,
                im: 0.0,
            },
            QkComplex64 {
                re: -0.22575349222402477,
                im: 0.0,
            },
            QkComplex64 {
                re: 0.17218393261915543,
                im: 0.0,
            },
            QkComplex64 {
                re: 0.12091263261776633,
                im: 0.0,
            },
            QkComplex64 {
                re: 0.17218393261915554,
                im: 0.0,
            },
            QkComplex64 {
                re: 0.16892753870087912,
                im: 0.0,
            },
            QkComplex64 {
                re: -0.22575349222402477,
                im: 0.0,
            },
            QkComplex64 {
                re: 0.16614543256382416,
                im: 0.0,
            },
            QkComplex64 {
                re: 0.04523279994605783,
                im: 0.0,
            },
            QkComplex64 {
                re: 0.04523279994605783,
                im: 0.0,
            },
            QkComplex64 {
                re: 0.04523279994605783,
                im: 0.0,
            },
            QkComplex64 {
                re: 0.04523279994605783,
                im: 0.0,
            },
            QkComplex64 {
                re: 0.16614543256382416,
                im: 0.0,
            },
            QkComplex64 {
                re: 0.17464343068300459,
                im: 0.0,
            },
            QkComplex64 {
                re: 0.12091263261776633,
                im: 0.0,
            },
        ];

        let mut bit_terms: Vec<ffi::QkBitTerm> = vec![
            QK_BIT_TERM_Z,
            QK_BIT_TERM_Z,
            QK_BIT_TERM_Z,
            QK_BIT_TERM_Z,
            QK_BIT_TERM_Z,
            QK_BIT_TERM_Z,
            QK_BIT_TERM_Z,
            QK_BIT_TERM_Z,
            QK_BIT_TERM_Z,
            QK_BIT_TERM_Z,
            QK_BIT_TERM_Y,
            QK_BIT_TERM_Y,
            QK_BIT_TERM_Y,
            QK_BIT_TERM_Y,
            QK_BIT_TERM_Y,
            QK_BIT_TERM_Y,
            QK_BIT_TERM_X,
            QK_BIT_TERM_X,
            QK_BIT_TERM_X,
            QK_BIT_TERM_X,
            QK_BIT_TERM_Y,
            QK_BIT_TERM_Y,
            QK_BIT_TERM_X,
            QK_BIT_TERM_X,
            QK_BIT_TERM_X,
            QK_BIT_TERM_X,
            QK_BIT_TERM_Z,
            QK_BIT_TERM_Z,
            QK_BIT_TERM_Z,
            QK_BIT_TERM_Z,
            QK_BIT_TERM_Z,
            QK_BIT_TERM_Z,
        ];

        let mut indices: Vec<u32> = vec![
            1, 0, 0, 1, 2, 0, 2, 3, 0, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 1, 2, 1,
            3, 2, 3,
        ];

        let mut boundaries: Vec<usize> =
            vec![0, 0, 1, 2, 4, 5, 7, 8, 10, 14, 18, 22, 26, 28, 30, 32];

        let mut expected = unsafe {
            ffi::qk_obs_new(
                4,
                coeffs.len().try_into().unwrap(),
                bit_terms.len().try_into().unwrap(),
                coeffs.as_mut_ptr(),
                bit_terms.as_mut_ptr(),
                indices.as_mut_ptr(),
                boundaries.as_mut_ptr(),
            )
        };

        let factor = QkComplex64 { re: -1.0, im: 0.0 };
        expected = unsafe { ffi::qk_obs_multiply(expected, &factor) };

        let mut diff = unsafe { ffi::qk_obs_add(qb_op, expected) };

        diff = unsafe { ffi::qk_obs_canonicalize(diff, 1e-6) };

        let zero = unsafe { ffi::qk_obs_zero(4) };

        let equal = unsafe { ffi::qk_obs_equal(diff, zero) };

        assert!(equal)
    }

    #[test]
    fn test_jordan_wigner_num_qubits_too_small() {
        // an operator acting on mode index 3 requires at least 4 qubits
        let fer_op = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            actions: vec![true],
            modes: vec![3],
            boundaries: vec![0, 1],
            groups: None,
        };

        // too few qubits must be reported instead of aborting the process
        let err = fermion_jordan_wigner(&fer_op, 3).unwrap_err();
        assert!(matches!(
            err,
            CoherenceError::NumQubitsTooSmall {
                num_qubits: 3,
                max_mode: 3
            }
        ));

        // exactly enough qubits succeeds
        assert!(fermion_jordan_wigner(&fer_op, 4).is_ok());
    }

    #[test]
    fn test_identity_only_terms_are_compacted() {
        // An identity Pauli term has *no* bit terms, so it costs nothing in `qk_obs_len` while still
        // occupying a coefficient and a boundary. Measuring the accumulators in `qk_obs_len` alone
        // therefore left identity-dominated operators growing without bound -- neither trigger could
        // ever fire -- which is precisely the regression compaction exists to prevent.
        //
        // Sized to clear `MIN_COMPACTION_FLOOR` by enough for the growth trigger to fire a handful of
        // times (measured: 7 merges here), which is what makes this exercise the incremental path
        // rather than only the final unconditional one. Going higher only costs time: at `1 << 21`
        // this ran 15x slower for 148 merges instead of 7, and caught nothing extra.
        let num_repeats = 1 << 17;
        let fer_op = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0); num_repeats],
            actions: vec![],
            modes: vec![],
            // Every term is empty, i.e. a bare coefficient, which maps onto the identity.
            boundaries: vec![0; num_repeats + 1],
            groups: None,
        };

        let qb_op = fermion_jordan_wigner(&fer_op, 2).unwrap();
        let num_terms = unsafe { ffi::qk_obs_num_terms(qb_op) } as usize;
        assert!(
            num_terms <= 2,
            "expected identity terms to be merged, got {num_terms} from {num_repeats} copies"
        );

        // ... and the operator must still be `num_repeats * I`.
        let mut coeffs: Vec<QkComplex64> = vec![QkComplex64 {
            re: -(num_repeats as f64),
            im: 0.0,
        }];
        let mut boundaries: Vec<usize> = vec![0, 0];
        // The single term is the identity, so it has no bit terms; the arrays are still real
        // allocations rather than null pointers, since `qk_obs_new` dereferences what it is handed.
        let mut bit_terms: Vec<ffi::QkBitTerm> = vec![QK_BIT_TERM_Z];
        let mut indices: Vec<u32> = vec![0];
        let negated = unsafe {
            ffi::qk_obs_new(
                2,
                1,
                0,
                coeffs.as_mut_ptr(),
                bit_terms.as_mut_ptr(),
                indices.as_mut_ptr(),
                boundaries.as_mut_ptr(),
            )
        };
        let diff = unsafe { ffi::qk_obs_add(qb_op, negated) };
        let diff = unsafe { ffi::qk_obs_canonicalize(diff, 1e-9) };
        let zero = unsafe { ffi::qk_obs_zero(2) };
        assert!(unsafe { ffi::qk_obs_equal(diff, zero) });
    }

    #[test]
    fn test_jordan_wigner_merges_duplicate_terms() {
        // The accumulators are grown with an addition that concatenates rather than merges, so
        // without periodic canonicalization the result grows with the number of terms mapped even
        // when they all map onto the same Pauli strings.
        //
        // Repeat one number operator many times: every copy maps to the same two Pauli terms, so a
        // compacting mapper returns a handful of terms regardless of how many copies there are.
        //
        // Sized as in `test_identity_only_terms_are_compacted`: far enough past
        // `MIN_COMPACTION_FLOOR` for the growth trigger to fire repeatedly (measured: 11 merges), and
        // no further, since the assertion below is satisfied by orders of magnitude either way.
        let num_repeats = 1 << 17;
        let fer_op = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0); num_repeats],
            actions: [true, false].repeat(num_repeats),
            modes: vec![0; 2 * num_repeats],
            boundaries: (0..=num_repeats).map(|i| 2 * i).collect(),
            groups: None,
        };

        let qb_op = fermion_jordan_wigner(&fer_op, 2).unwrap();
        let num_terms = unsafe { ffi::qk_obs_num_terms(qb_op) } as usize;

        // `a^dagger_0 a_0` maps onto the identity and `Z_0`, so the merged result is tiny. Allow
        // generous slack for whatever is left over between compactions, but assert the count does
        // not scale with `num_repeats` (which is what regressed): the growth rule keeps the
        // accumulators near the compaction floor, which is far below the number of copies here.
        assert!(
            num_terms < num_repeats / 8,
            "expected duplicates to be merged, got {num_terms} terms from {num_repeats} copies"
        );

        // The operator must still be `num_repeats * (I - Z_0) / 2`.
        let canon = unsafe { compact(qb_op) };
        let expected_coeff = QkComplex64 {
            re: 0.5 * num_repeats as f64,
            im: 0.0,
        };
        let mut coeffs: Vec<QkComplex64> = vec![
            expected_coeff,
            QkComplex64 {
                re: -expected_coeff.re,
                im: 0.0,
            },
        ];
        let mut bit_terms: Vec<ffi::QkBitTerm> = vec![QK_BIT_TERM_Z];
        let mut indices: Vec<u32> = vec![0];
        let mut boundaries: Vec<usize> = vec![0, 0, 1];
        let expected = unsafe {
            ffi::qk_obs_new(
                2,
                coeffs.len().try_into().unwrap(),
                bit_terms.len().try_into().unwrap(),
                coeffs.as_mut_ptr(),
                bit_terms.as_mut_ptr(),
                indices.as_mut_ptr(),
                boundaries.as_mut_ptr(),
            )
        };
        let factor = QkComplex64 { re: -1.0, im: 0.0 };
        let negated = unsafe { ffi::qk_obs_multiply(expected, &factor) };
        let diff = unsafe { ffi::qk_obs_add(canon, negated) };
        let diff = unsafe { ffi::qk_obs_canonicalize(diff, 1e-9) };
        let zero = unsafe { ffi::qk_obs_zero(2) };
        assert!(unsafe { ffi::qk_obs_equal(diff, zero) });
    }
}
