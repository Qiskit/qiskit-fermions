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
#[gen_stub_pyfunction(module = "qiskit_fermions.mappers.library.directed_interaction")]
#[pyfunction(name = "directed_interaction_to_majorana")]
pub fn py_directed_interaction_to_majorana(
    inter_op: PyDirectedInteractionOperator,
) -> PyMajoranaOperator {
    PyMajoranaOperator {
        inner: directed_interaction_to_majorana(&inter_op.inner),
    }
}

/// Map a :class:`.DirectedInteractionOperator` to a :class:`.UndirectedInteractionOperator`.
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
