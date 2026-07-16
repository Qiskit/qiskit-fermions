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

use crate::operators::edge_vertex_operator::PyEdgeVertexOperator;
use crate::operators::fermion_operator::PyFermionOperator;
use crate::operators::majorana_operator::PyMajoranaOperator;
use crate::operators::transfer_vertex_operator::PyTransferVertexOperator;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use qiskit_fermions_core::mappers::library::transfer_vertex::{
    transfer_vertex_to_edge_vertex, transfer_vertex_to_fermion, transfer_vertex_to_majorana,
};

/// Map a :class:`.TransferVertexOperator` to a :class:`.FermionOperator`.
///
/// Args:
///     inter_op: the transfer-vertex operator to map.
///
/// Returns:
///     The mapped fermionic operator.
///
/// Definition
/// ==========
///
/// This function decomposes the transfer-vertex operators in terms of the fermionic action
/// operators as defined :ref:`here <TransferVertexOperator-definition>`.
///
/// Usage
/// =====
///
/// .. doctest::
///
///     >>> from qiskit_fermions.mappers.library import transfer_vertex_to_fermion
///     >>> from qiskit_fermions.operators import TransferVertexOperator
///     >>> inter_op = TransferVertexOperator.from_dict({((0, 0),): 1, ((1, 2),): 2, ((2, 1),): -2})
///     >>> fer_op = transfer_vertex_to_fermion(inter_op)
///     >>> print(format(fer_op.normal_ordered().simplify()))
///       1.000000e0 +0.000000e0j * ()
///      -2.000000e0 +0.000000e0j * (-2 -1)
///      -2.000000e0 +0.000000e0j * (+0 -0)
///       2.000000e0 -0.000000e0j * (+2 +1)
///
/// ..
#[gen_stub_pyfunction(module = "qiskit_fermions.mappers.library.transfer_vertex")]
#[pyfunction(name = "transfer_vertex_to_fermion")]
pub fn py_transfer_vertex_to_fermion(inter_op: PyTransferVertexOperator) -> PyFermionOperator {
    transfer_vertex_to_fermion(&inter_op.inner).into()
}

/// Map a :class:`.TransferVertexOperator` to a :class:`.MajoranaOperator`.
///
/// Args:
///     inter_op: the transfer-vertex operator to map.
///
/// Returns:
///     The mapped majorana operator.
///
/// Definition
/// ==========
///
/// This function decomposes the transfer-vertex operators in terms of the majorana operators as
/// defined :ref:`here <TransferVertexOperator-definition>`.
///
/// Usage
/// =====
///
/// .. doctest::
///
///     >>> from qiskit_fermions.mappers.library import transfer_vertex_to_majorana
///     >>> from qiskit_fermions.operators import TransferVertexOperator
///     >>> inter_op = TransferVertexOperator.from_dict({((0, 0),): 1, ((1, 2),): 2, ((2, 1),): -2})
///     >>> maj_op = transfer_vertex_to_majorana(inter_op)
///     >>> print(format(maj_op.normal_ordered().simplify()))
///       0.000000e0 +1.000000e0j * (γ'0 γ0)
///      -0.000000e0 -1.000000e0j * (γ2 γ'1)
///      -0.000000e0 -1.000000e0j * (γ'2 γ1)
///
/// ..
#[gen_stub_pyfunction(module = "qiskit_fermions.mappers.library.transfer_vertex")]
#[pyfunction(name = "transfer_vertex_to_majorana")]
pub fn py_transfer_vertex_to_majorana(inter_op: PyTransferVertexOperator) -> PyMajoranaOperator {
    transfer_vertex_to_majorana(&inter_op.inner).into()
}

/// Map a :class:`.TransferVertexOperator` to an :class:`.EdgeVertexOperator`.
///
/// Args:
///     inter_op: the transfer-vertex operator to map.
///
/// Returns:
///     The mapped edge-vertex operator.
///
/// Definition
/// ==========
///
/// This function decomposes the transfer-vertex operators in terms of the edge-vertex operators as
/// defined :ref:`here <TransferVertexOperator-definition>`.
///
/// Usage
/// =====
///
/// .. doctest::
///
///     >>> from qiskit_fermions.mappers.library import transfer_vertex_to_edge_vertex
///     >>> from qiskit_fermions.operators import TransferVertexOperator
///     >>> dir_op = TransferVertexOperator.from_dict({((0, 0),): 1, ((1, 2),): 2, ((2, 1),): -2})
///     >>> undir_op = transfer_vertex_to_edge_vertex(dir_op)
///     >>> print(format(undir_op.simplify()))
///       1.000000e0 +0.000000e0j * (V(0))
///       0.000000e0 +1.000000e0j * (V(1) E(1,2))
///      -0.000000e0 -1.000000e0j * (E(1,2) V(2))
///
/// ..
#[gen_stub_pyfunction(module = "qiskit_fermions.mappers.library.transfer_vertex")]
#[pyfunction(name = "transfer_vertex_to_edge_vertex")]
pub fn py_transfer_vertex_to_edge_vertex(
    inter_op: PyTransferVertexOperator,
) -> PyEdgeVertexOperator {
    transfer_vertex_to_edge_vertex(&inter_op.inner).into()
}

#[pymodule]
pub mod transfer_vertex {
    #[pymodule_export]
    use super::py_transfer_vertex_to_fermion;

    #[pymodule_export]
    use super::py_transfer_vertex_to_majorana;

    #[pymodule_export]
    use super::py_transfer_vertex_to_edge_vertex;
}
