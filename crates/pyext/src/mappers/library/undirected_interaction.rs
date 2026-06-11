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

use crate::operators::fermion_operator::PyFermionOperator;
use crate::operators::majorana_operator::PyMajoranaOperator;
use crate::operators::undirected_interaction_operator::PyUndirectedInteractionOperator;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use qiskit_fermions_core::mappers::library::undirected_interaction::{
    undirected_interaction_to_fermion, undirected_interaction_to_majorana,
};

/// Map an :class:`.UndirectedInteractionOperator` to a :class:`.FermionOperator`.
///
/// Args:
///     inter_op: the undirected interaction operator to map.
///
/// Returns:
///     The mapped fermionic operator.
///
/// ----
///
/// Definition
/// ==========
///
/// This function decomposes the edge-vertex operators in terms of the fermionic action operators
/// as defined :ref:`here <UndirectedInteractionOperator-definition>`.
///
/// Usage
/// =====
///
/// .. doctest::
///
///     >>> from qiskit_fermions.mappers.library import undirected_interaction_to_fermion
///     >>> from qiskit_fermions.operators import UndirectedInteractionOperator
///     >>> inter_op = UndirectedInteractionOperator.from_dict({((0, 0),): 1, ((1, 2),): 2})
///     >>> fer_op = undirected_interaction_to_fermion(inter_op)
///     >>> print(format(fer_op.normal_ordered().simplify()))
///       1.000000e0 +0.000000e0j * ()
///      -0.000000e0 +2.000000e0j * (-2 -1)
///      -2.000000e0 +0.000000e0j * (+0 -0)
///       0.000000e0 -2.000000e0j * (+1 -2)
///      -0.000000e0 +2.000000e0j * (+2 -1)
///      -0.000000e0 +2.000000e0j * (+2 +1)
///
/// ..
#[gen_stub_pyfunction(module = "qiskit_fermions.mappers.library.undirected_interaction")]
#[pyfunction(name = "undirected_interaction_to_fermion")]
pub fn py_undirected_interaction_to_fermion(
    inter_op: PyUndirectedInteractionOperator,
) -> PyFermionOperator {
    PyFermionOperator {
        inner: undirected_interaction_to_fermion(&inter_op.inner),
    }
}

/// Map an :class:`.UndirectedInteractionOperator` to a :class:`.MajoranaOperator`.
///
/// Args:
///     inter_op: the undirected interaction operator to map.
///
/// Returns:
///     The mapped majorana operator.
///
/// ----
///
/// Definition
/// ==========
///
/// This function decomposes the edge-vertex operators in terms of the majorana operators as
/// defined :ref:`here <UndirectedInteractionOperator-definition>`.
///
/// Usage
/// =====
///
/// .. doctest::
///
///     >>> from qiskit_fermions.mappers.library import undirected_interaction_to_majorana
///     >>> from qiskit_fermions.operators import UndirectedInteractionOperator
///     >>> inter_op = UndirectedInteractionOperator.from_dict({((0, 0),): 1, ((1, 2),): 2})
///     >>> maj_op = undirected_interaction_to_majorana(inter_op)
///     >>> print(format(maj_op.normal_ordered().simplify()))
///       0.000000e0 +1.000000e0j * (γ'0 γ0)
///       0.000000e0 +2.000000e0j * (γ2 γ1)
///
/// ..
#[gen_stub_pyfunction(module = "qiskit_fermions.mappers.library.undirected_interaction")]
#[pyfunction(name = "undirected_interaction_to_majorana")]
pub fn py_undirected_interaction_to_majorana(
    inter_op: PyUndirectedInteractionOperator,
) -> PyMajoranaOperator {
    PyMajoranaOperator {
        inner: undirected_interaction_to_majorana(&inter_op.inner),
    }
}

#[pymodule]
pub mod undirected_interaction {
    #[pymodule_export]
    use super::py_undirected_interaction_to_fermion;

    #[pymodule_export]
    use super::py_undirected_interaction_to_majorana;
}
