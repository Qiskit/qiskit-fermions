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

use qiskit_fermions_core::mappers::library::edge_vertex::{
    edge_vertex_to_fermion, edge_vertex_to_majorana,
};
use qiskit_fermions_core::operators::edge_vertex_operator::EdgeVertexOperator;
use qiskit_fermions_core::operators::fermion_operator::FermionOperator;
use qiskit_fermions_core::operators::majorana_operator::MajoranaOperator;

/// @ingroup qf_mapper_library
///
/// @brief Map a ``QfEdgeVertexOperator`` to a ``QfFermionOperator``.
///
/// @param edge_op A pointer to the edge-vertex operator to be mapped.
///
/// @return A pointer to the mapped fermionic operator.
///
/// @rst
///
/// Definition
/// ----------
///
/// This function decomposes the edge and vertex operators in terms of the fermionic creation and
/// annihilation operators, as defined :ref:`here <qf_edge_op-definition>`.
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
///     // define some kind of edge-vertex operator
///     QfEdgeVertexOperator *edge_op = qf_edge_op_one();
///
///     // and map it to a fermionic operator
///     QfFermionOperator *fer_op = qf_edge_vertex_to_fermion(edge_op);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_edge_vertex_to_fermion(
    edge_op: *const EdgeVertexOperator,
) -> *mut FermionOperator {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let edge_op = unsafe { const_ptr_as_ref(edge_op) };

    let fer_op = edge_vertex_to_fermion(edge_op);
    Box::into_raw(Box::new(fer_op))
}

/// @ingroup qf_mapper_library
///
/// @brief Map a ``QfEdgeVertexOperator`` to a ``QfMajoranaOperator``.
///
/// @param edge_op A pointer to the edge-vertex operator to be mapped.
///
/// @return A pointer to the mapped Majorana operator.
///
/// @rst
///
/// Definition
/// ----------
///
/// This function decomposes the edge and vertex operators in terms of the Majorana operators, as
/// defined :ref:`here <qf_edge_op-definition>`.
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
///     // define some kind of edge-vertex operator
///     QfEdgeVertexOperator *edge_op = qf_edge_op_one();
///
///     // and map it to a Majorana operator
///     QfMajoranaOperator *maj_op = qf_edge_vertex_to_majorana(edge_op);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_edge_vertex_to_majorana(
    edge_op: *const EdgeVertexOperator,
) -> *mut MajoranaOperator {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let edge_op = unsafe { const_ptr_as_ref(edge_op) };

    let maj_op = edge_vertex_to_majorana(edge_op);
    Box::into_raw(Box::new(maj_op))
}
