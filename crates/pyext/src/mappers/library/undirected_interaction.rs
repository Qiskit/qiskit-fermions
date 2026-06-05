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

/// Map a :class:`.UndirectedInteractionOperator` to a :class:`.FermionOperator`.
#[gen_stub_pyfunction(module = "qiskit_fermions.mappers.library.undirected_interaction")]
#[pyfunction(name = "undirected_interaction_to_fermion")]
pub fn py_undirected_interaction_to_fermion(
    inter_op: PyUndirectedInteractionOperator,
) -> PyFermionOperator {
    PyFermionOperator {
        inner: undirected_interaction_to_fermion(&inter_op.inner),
    }
}

/// Map a :class:`.UndirectedInteractionOperator` to a :class:`.MajoranaOperator`.
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
