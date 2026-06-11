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

use crate::operators::directed_interaction_operator::PyDirectedInteractionOperator;
use crate::operators::fermion_operator::PyFermionOperator;
use crate::operators::majorana_operator::PyMajoranaOperator;
use crate::operators::undirected_interaction_operator::PyUndirectedInteractionOperator;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use qiskit_fermions_core::mappers::library::directed_interaction::{
    directed_interaction_to_fermion, directed_interaction_to_majorana,
    directed_interaction_to_undirected,
};

/// Map a :class:`.DirectedInteractionOperator` to a :class:`.FermionOperator`.
///
/// Args:
///     inter_op: the directed interaction operator to map.
///
/// Returns:
///     The mapped fermionic operator.
///
/// ----
///
/// Definition
/// ==========
///
/// This function decomposes the transfer-vertex operators in terms of the fermionic action
/// operators as defined :ref:`here <DirectedInteractionOperator-definition>`.
///
/// Usage
/// =====
///
/// .. doctest::
///
///     >>> from qiskit_fermions.mappers.library import directed_interaction_to_fermion
///     >>> from qiskit_fermions.operators import DirectedInteractionOperator
///     >>> inter_op = DirectedInteractionOperator.from_dict({((0, 0),): 1, ((1, 2),): 2, ((2, 1),): -2})
///     >>> fer_op = directed_interaction_to_fermion(inter_op)
///     >>> print(format(fer_op.normal_ordered().simplify()))
///       1.000000e0 +0.000000e0j * ()
///      -2.000000e0 +0.000000e0j * (-2 -1)
///      -2.000000e0 +0.000000e0j * (+0 -0)
///       2.000000e0 -0.000000e0j * (+2 +1)
///
/// ..
#[gen_stub_pyfunction(module = "qiskit_fermions.mappers.library.directed_interaction")]
#[pyfunction(name = "directed_interaction_to_fermion")]
pub fn py_directed_interaction_to_fermion(
    inter_op: PyDirectedInteractionOperator,
) -> PyFermionOperator {
    PyFermionOperator {
        inner: directed_interaction_to_fermion(&inter_op.inner),
    }
}

/// Map a :class:`.DirectedInteractionOperator` to a :class:`.MajoranaOperator`.
///
/// Args:
///     inter_op: the directed interaction operator to map.
///
/// Returns:
///     The mapped majorana operator.
///
/// ----
///
/// Definition
/// ==========
///
/// This function decomposes the transfer-vertex operators in terms of the majorana operators as
/// defined :ref:`here <DirectedInteractionOperator-definition>`.
///
/// Usage
/// =====
///
/// .. doctest::
///
///     >>> from qiskit_fermions.mappers.library import directed_interaction_to_majorana
///     >>> from qiskit_fermions.operators import DirectedInteractionOperator
///     >>> inter_op = DirectedInteractionOperator.from_dict({((0, 0),): 1, ((1, 2),): 2, ((2, 1),): -2})
///     >>> maj_op = directed_interaction_to_majorana(inter_op)
///     >>> print(format(maj_op.normal_ordered().simplify()))
///       0.000000e0 +1.000000e0j * (γ'0 γ0)
///      -0.000000e0 -1.000000e0j * (γ2 γ'1)
///      -0.000000e0 -1.000000e0j * (γ'2 γ1)
///
/// ..
#[gen_stub_pyfunction(module = "qiskit_fermions.mappers.library.directed_interaction")]
#[pyfunction(name = "directed_interaction_to_majorana")]
pub fn py_directed_interaction_to_majorana(
    inter_op: PyDirectedInteractionOperator,
) -> PyMajoranaOperator {
    PyMajoranaOperator {
        inner: directed_interaction_to_majorana(&inter_op.inner),
    }
}

/// Map a :class:`.DirectedInteractionOperator` to an :class:`.UndirectedInteractionOperator`.
///
/// Args:
///     inter_op: the directed interaction operator to map.
///
/// Returns:
///     The mapped undirected interaction operator.
///
/// ----
///
/// Definition
/// ==========
///
/// This function decomposes the transfer-vertex operators in terms of the edge-vertex operators as
/// defined :ref:`here <DirectedInteractionOperator-definition>`.
///
/// Usage
/// =====
///
/// .. doctest::
///
///     >>> from qiskit_fermions.mappers.library import directed_interaction_to_undirected
///     >>> from qiskit_fermions.operators import DirectedInteractionOperator
///     >>> dir_op = DirectedInteractionOperator.from_dict({((0, 0),): 1, ((1, 2),): 2, ((2, 1),): -2})
///     >>> undir_op = directed_interaction_to_undirected(dir_op)
///     >>> print(format(undir_op.simplify()))
///       1.000000e0 +0.000000e0j * (V(0))
///       0.000000e0 +1.000000e0j * (V(1) E(1,2))
///      -0.000000e0 -1.000000e0j * (E(1,2) V(2))
///
/// ..
#[gen_stub_pyfunction(module = "qiskit_fermions.mappers.library.directed_interaction")]
#[pyfunction(name = "directed_interaction_to_undirected")]
pub fn py_directed_interaction_to_undirected(
    inter_op: PyDirectedInteractionOperator,
) -> PyUndirectedInteractionOperator {
    PyUndirectedInteractionOperator {
        inner: directed_interaction_to_undirected(&inter_op.inner),
    }
}

#[pymodule]
pub mod directed_interaction {
    #[pymodule_export]
    use super::py_directed_interaction_to_fermion;

    #[pymodule_export]
    use super::py_directed_interaction_to_majorana;

    #[pymodule_export]
    use super::py_directed_interaction_to_undirected;
}
