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
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use qiskit_fermions_core::mappers::library::edge_vertex::{
    edge_vertex_to_fermion, edge_vertex_to_majorana,
};

/// Map an :class:`.EdgeVertexOperator` to a :class:`.FermionOperator`.
///
/// Args:
///     inter_op: the edge-vertex operator to map.
///
/// Returns:
///     The mapped fermionic operator.
///
/// Definition
/// ==========
///
/// This function decomposes the edge-vertex operators in terms of the fermionic action operators
/// as defined :ref:`here <EdgeVertexOperator-definition>`.
///
/// Usage
/// =====
///
/// .. doctest::
///
///     >>> from qiskit_fermions.mappers.library import edge_vertex_to_fermion
///     >>> from qiskit_fermions.operators import EdgeVertexOperator
///     >>> inter_op = EdgeVertexOperator.from_dict({((0, 0),): 1, ((1, 2),): 2})
///     >>> fer_op = edge_vertex_to_fermion(inter_op)
///     >>> print(format(fer_op.normal_ordered().simplify()))
///       1.000000e0 +0.000000e0j * ()
///      -0.000000e0 +2.000000e0j * (-2 -1)
///      -2.000000e0 +0.000000e0j * (+0 -0)
///       0.000000e0 -2.000000e0j * (+1 -2)
///      -0.000000e0 +2.000000e0j * (+2 -1)
///      -0.000000e0 +2.000000e0j * (+2 +1)
///
/// ..
#[gen_stub_pyfunction(module = "qiskit_fermions._lib.mappers.mappers_library.edge_vertex")]
#[pyfunction(name = "edge_vertex_to_fermion")]
pub fn py_edge_vertex_to_fermion(inter_op: PyEdgeVertexOperator) -> PyFermionOperator {
    edge_vertex_to_fermion(&inter_op.inner).into()
}

/// Map an :class:`.EdgeVertexOperator` to a :class:`.MajoranaOperator`.
///
/// Args:
///     inter_op: the edge-vertex operator to map.
///
/// Returns:
///     The mapped majorana operator.
///
/// Definition
/// ==========
///
/// This function decomposes the edge-vertex operators in terms of the majorana operators as
/// defined :ref:`here <EdgeVertexOperator-definition>`.
///
/// Usage
/// =====
///
/// .. doctest::
///
///     >>> from qiskit_fermions.mappers.library import edge_vertex_to_majorana
///     >>> from qiskit_fermions.operators import EdgeVertexOperator
///     >>> inter_op = EdgeVertexOperator.from_dict({((0, 0),): 1, ((1, 2),): 2})
///     >>> maj_op = edge_vertex_to_majorana(inter_op)
///     >>> print(format(maj_op.normal_ordered().simplify()))
///       0.000000e0 +1.000000e0j * (γ'0 γ0)
///       0.000000e0 +2.000000e0j * (γ2 γ1)
///
/// ..
#[gen_stub_pyfunction(module = "qiskit_fermions._lib.mappers.mappers_library.edge_vertex")]
#[pyfunction(name = "edge_vertex_to_majorana")]
pub fn py_edge_vertex_to_majorana(inter_op: PyEdgeVertexOperator) -> PyMajoranaOperator {
    edge_vertex_to_majorana(&inter_op.inner).into()
}

#[gen_stub_pymethods]
#[pymethods]
impl PyEdgeVertexOperator {
    /// Converts this operator into a :class:`.FermionOperator`.
    ///
    /// This implements the :class:`.SupportsFermionOperator` protocol by delegating to
    /// :func:`.edge_vertex_to_fermion`.
    fn _fermion_operator_(&self) -> PyFermionOperator {
        py_edge_vertex_to_fermion(self.clone())
    }

    /// Converts this operator into a :class:`.MajoranaOperator`.
    ///
    /// This implements the :class:`.SupportsMajoranaOperator` protocol by delegating to
    /// :func:`.edge_vertex_to_majorana`.
    fn _majorana_operator_(&self) -> PyMajoranaOperator {
        py_edge_vertex_to_majorana(self.clone())
    }
}

#[pymodule]
pub mod edge_vertex {
    #[pymodule_export]
    use super::py_edge_vertex_to_fermion;

    #[pymodule_export]
    use super::py_edge_vertex_to_majorana;
}
