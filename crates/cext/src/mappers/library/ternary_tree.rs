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

use crate::exit_codes::ExitCode;
use crate::pointers::const_ptr_as_ref;

use qiskit_fermions_core::mappers::library::ternary_tree::{
    edge_vertex_ternary_tree, fermion_ternary_tree, majorana_ternary_tree,
    transfer_vertex_ternary_tree,
};
use qiskit_fermions_core::mappers::ternary_tree::encoding::TernaryTreeEncoding;
use qiskit_fermions_core::operators::edge_vertex_operator::EdgeVertexOperator;
use qiskit_fermions_core::operators::fermion_operator::FermionOperator;
use qiskit_fermions_core::operators::majorana_operator::MajoranaOperator;
use qiskit_fermions_core::operators::transfer_vertex_operator::TransferVertexOperator;

/// @ingroup qf_mapper_library
///
/// @brief Maps a fermionic operator through a ternary-tree encoding.
///
/// @param op A pointer to the fermionic operator to be mapped.
/// @param encoding A pointer to the compiled encoding to map through.
/// @param num_qubits The number of qubits for the mapped operator. This must be at least the
///        encoding's mode count; a larger value pads with the identity.
/// @param out A pointer to where the created qubit operator will be written on success. It is left
///        untouched if the transformation fails.
///
/// @return An exit code. This is ``>0`` if an error occurred. In particular, a
///         ``QfExitCode_ValueError`` is returned if ``op`` acts on a mode the encoding does not
///         cover.
///
/// @rst
///
/// On a chain along ``Z`` this produces the same observable as :c:func:`qf_ferm_op_jordan_wigner`;
/// other trees trade that encoding's Pauli weight, linear in the number of modes, for a shallower one.
///
/// Example
/// -------
///
/// .. code-block:: c
///
///     QfTernaryTree *tree;
///     QfPauliLabel order[3] = {QfPauliLabel_X, QfPauliLabel_Y, QfPauliLabel_Z};
///     qf_ternary_tree_breadth_first(4, 3, order, &tree);
///
///     QfTernaryTreeEncoding *encoding;
///     qf_ternary_tree_encoding_new(tree, NULL, &encoding);
///
///     QkObs *result;
///     QfExitCode exit = qf_ferm_op_ternary_tree(hamil, encoding, 4, &result);
///
///     assert(exit == QfExitCode_Success);
///
///     qf_ternary_tree_encoding_free(encoding);
///     qf_ternary_tree_free(tree);
///
/// @endrst
///
/// # Safety
///
/// `op` and `encoding` must be valid pointers to their respective types, and `out` must be non-null
/// and aligned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ferm_op_ternary_tree(
    op: *const FermionOperator,
    encoding: *const TernaryTreeEncoding,
    num_qubits: u32,
    out: *mut *mut qiskit_sys::QkObs,
) -> ExitCode {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let op = unsafe { const_ptr_as_ref(op) };
    let encoding = unsafe { const_ptr_as_ref(encoding) };

    match fermion_ternary_tree(op, encoding, num_qubits) {
        Ok(obs) => {
            // SAFETY: Per documentation, `out` is non-null and aligned.
            unsafe { out.write(obs) };
            ExitCode::Success
        }
        Err(_) => ExitCode::ValueError,
    }
}

/// @ingroup qf_mapper_library
///
/// @brief Maps a Majorana operator through a ternary-tree encoding.
///
/// @param op A pointer to the Majorana operator to be mapped.
/// @param encoding A pointer to the compiled encoding to map through.
/// @param num_qubits The number of qubits for the mapped operator. This must be at least the
///        encoding's mode count; a larger value pads with the identity.
/// @param out A pointer to where the created qubit operator will be written on success. It is left
///        untouched if the transformation fails.
///
/// @return An exit code. This is ``>0`` if an error occurred. In particular, a
///         ``QfExitCode_ValueError`` is returned if ``op`` acts on a mode the encoding does not
///         cover. Note that Majorana index :math:`m` acts on mode :math:`\lfloor m/2 \rfloor`.
///
/// @rst
///
/// Each Majorana operator's image is a single Pauli string, whatever the tree, so a term that is a
/// product of several of them composes those strings rather than inflating into a sum.
///
/// @endrst
///
/// # Safety
///
/// `op` and `encoding` must be valid pointers to their respective types, and `out` must be non-null
/// and aligned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_maj_op_ternary_tree(
    op: *const MajoranaOperator,
    encoding: *const TernaryTreeEncoding,
    num_qubits: u32,
    out: *mut *mut qiskit_sys::QkObs,
) -> ExitCode {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let op = unsafe { const_ptr_as_ref(op) };
    let encoding = unsafe { const_ptr_as_ref(encoding) };

    match majorana_ternary_tree(op, encoding, num_qubits) {
        Ok(obs) => {
            // SAFETY: Per documentation, `out` is non-null and aligned.
            unsafe { out.write(obs) };
            ExitCode::Success
        }
        Err(_) => ExitCode::ValueError,
    }
}

/// @ingroup qf_mapper_library
///
/// @brief Maps an edge-vertex operator through a ternary-tree encoding.
///
/// @param op A pointer to the edge-vertex operator to be mapped.
/// @param encoding A pointer to the compiled encoding to map through.
/// @param num_qubits The number of qubits for the mapped operator. This must be at least the
///        encoding's mode count; a larger value pads with the identity.
/// @param out A pointer to where the created qubit operator will be written on success. It is left
///        untouched if the transformation fails.
///
/// @return An exit code. This is ``>0`` if an error occurred. In particular, a
///         ``QfExitCode_ValueError`` is returned if ``op`` acts on a mode the encoding does not
///         cover.
///
/// @rst
///
/// .. note::
///    Unlike under Jordan-Wigner, a vertex operator is not in general the weight-1 Pauli :math:`Z_l`.
///    That is a property of the chain, where the two Majorana strings' :math:`Z` chains cancel; on the
///    balanced tree at four modes a vertex operator has weight 3.
///
/// @endrst
///
/// # Safety
///
/// `op` and `encoding` must be valid pointers to their respective types, and `out` must be non-null
/// and aligned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_edge_op_ternary_tree(
    op: *const EdgeVertexOperator,
    encoding: *const TernaryTreeEncoding,
    num_qubits: u32,
    out: *mut *mut qiskit_sys::QkObs,
) -> ExitCode {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let op = unsafe { const_ptr_as_ref(op) };
    let encoding = unsafe { const_ptr_as_ref(encoding) };

    match edge_vertex_ternary_tree(op, encoding, num_qubits) {
        Ok(obs) => {
            // SAFETY: Per documentation, `out` is non-null and aligned.
            unsafe { out.write(obs) };
            ExitCode::Success
        }
        Err(_) => ExitCode::ValueError,
    }
}

/// @ingroup qf_mapper_library
///
/// @brief Maps a transfer-vertex operator through a ternary-tree encoding.
///
/// @param op A pointer to the transfer-vertex operator to be mapped.
/// @param encoding A pointer to the compiled encoding to map through.
/// @param num_qubits The number of qubits for the mapped operator. This must be at least the
///        encoding's mode count; a larger value pads with the identity.
/// @param out A pointer to where the created qubit operator will be written on success. It is left
///        untouched if the transformation fails.
///
/// @return An exit code. This is ``>0`` if an error occurred. In particular, a
///         ``QfExitCode_ValueError`` is returned if ``op`` acts on a mode the encoding does not
///         cover.
///
/// # Safety
///
/// `op` and `encoding` must be valid pointers to their respective types, and `out` must be non-null
/// and aligned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_transfer_op_ternary_tree(
    op: *const TransferVertexOperator,
    encoding: *const TernaryTreeEncoding,
    num_qubits: u32,
    out: *mut *mut qiskit_sys::QkObs,
) -> ExitCode {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let op = unsafe { const_ptr_as_ref(op) };
    let encoding = unsafe { const_ptr_as_ref(encoding) };

    match transfer_vertex_ternary_tree(op, encoding, num_qubits) {
        Ok(obs) => {
            // SAFETY: Per documentation, `out` is non-null and aligned.
            unsafe { out.write(obs) };
            ExitCode::Success
        }
        Err(_) => ExitCode::ValueError,
    }
}
