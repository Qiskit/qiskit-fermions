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

use crate::pointers::const_ptr_as_ref;

use qiskit_fermions_core::mappers::library::transfer_vertex::{
    transfer_vertex_to_edge_vertex, transfer_vertex_to_fermion, transfer_vertex_to_majorana,
};
use qiskit_fermions_core::operators::edge_vertex_operator::EdgeVertexOperator;
use qiskit_fermions_core::operators::fermion_operator::FermionOperator;
use qiskit_fermions_core::operators::majorana_operator::MajoranaOperator;
use qiskit_fermions_core::operators::transfer_vertex_operator::TransferVertexOperator;

/// @ingroup qf_mapper_library
///
/// @brief Map a ``QfTransferVertexOperator`` to a ``QfFermionOperator``.
///
/// @param transfer_op A pointer to the transfer-vertex operator to be mapped.
///
/// @return A pointer to the mapped fermionic operator.
///
/// @rst
///
/// Definition
/// ----------
///
/// This function decomposes the transfer and vertex operators in terms of the fermionic creation
/// and annihilation operators, as defined :ref:`here <qf_transfer_op-definition>`.
///
/// .. note::
///    The mapped operator is not simplified. Because each generator expands into a sum of fermionic
///    terms, the result generally contains duplicate terms; call
///    :c:func:`qf_ferm_op_normal_ordered` followed by :c:func:`qf_ferm_op_simplify` to reduce it.
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     // define some kind of transfer-vertex operator
///     QfTransferVertexOperator *transfer_op = qf_transfer_op_one();
///
///     // and map it to a fermionic operator
///     QfFermionOperator *fer_op = qf_transfer_vertex_to_fermion(transfer_op);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_transfer_vertex_to_fermion(
    transfer_op: *const TransferVertexOperator,
) -> *mut FermionOperator {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let transfer_op = unsafe { const_ptr_as_ref(transfer_op) };

    let fer_op = transfer_vertex_to_fermion(transfer_op);
    Box::into_raw(Box::new(fer_op))
}

/// @ingroup qf_mapper_library
///
/// @brief Map a ``QfTransferVertexOperator`` to a ``QfMajoranaOperator``.
///
/// @param transfer_op A pointer to the transfer-vertex operator to be mapped.
///
/// @return A pointer to the mapped Majorana operator.
///
/// @rst
///
/// Definition
/// ----------
///
/// This function decomposes the transfer and vertex operators in terms of the Majorana operators,
/// as defined :ref:`here <qf_transfer_op-definition>`.
///
/// .. note::
///    The mapped operator is not simplified; see :c:func:`qf_maj_op_normal_ordered` and
///    :c:func:`qf_maj_op_simplify`.
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     // define some kind of transfer-vertex operator
///     QfTransferVertexOperator *transfer_op = qf_transfer_op_one();
///
///     // and map it to a Majorana operator
///     QfMajoranaOperator *maj_op = qf_transfer_vertex_to_majorana(transfer_op);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_transfer_vertex_to_majorana(
    transfer_op: *const TransferVertexOperator,
) -> *mut MajoranaOperator {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let transfer_op = unsafe { const_ptr_as_ref(transfer_op) };

    let maj_op = transfer_vertex_to_majorana(transfer_op);
    Box::into_raw(Box::new(maj_op))
}

/// @ingroup qf_mapper_library
///
/// @brief Map a ``QfTransferVertexOperator`` to a ``QfEdgeVertexOperator``.
///
/// @param transfer_op A pointer to the transfer-vertex operator to be mapped.
///
/// @return A pointer to the mapped edge-vertex operator.
///
/// @rst
///
/// Definition
/// ----------
///
/// This function rewrites each transfer operator in terms of the edge and vertex operators. A
/// vertex operator maps to itself, whereas a transfer operator :math:`T_{jk}` becomes a
/// length-two product of an edge and a vertex operator.
///
/// .. note::
///    This is the one mapper that stays within the interaction-operator representations, so it is
///    the natural route to compare the two: mapping a transfer operator to an edge-vertex operator
///    and then on to a fermionic one agrees with mapping it to a fermionic operator directly.
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     // define some kind of transfer-vertex operator
///     QfTransferVertexOperator *transfer_op = qf_transfer_op_one();
///
///     // and map it to an edge-vertex operator
///     QfEdgeVertexOperator *edge_op = qf_transfer_vertex_to_edge_vertex(transfer_op);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_transfer_vertex_to_edge_vertex(
    transfer_op: *const TransferVertexOperator,
) -> *mut EdgeVertexOperator {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let transfer_op = unsafe { const_ptr_as_ref(transfer_op) };

    let edge_op = transfer_vertex_to_edge_vertex(transfer_op);
    Box::into_raw(Box::new(edge_op))
}
