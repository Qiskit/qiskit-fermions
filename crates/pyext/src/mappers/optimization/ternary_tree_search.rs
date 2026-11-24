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

use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;

use qiskit_fermions_core::mappers::optimization::ternary_tree_search::hatt_tree;

use crate::mappers::ternary_tree::PyTernaryTree;
use crate::operators::majorana_operator::PyMajoranaOperator;

/// Grow a ternary tree adapted to a Hamiltonian, following the HATT algorithm.
///
/// The algorithm is topology-blind: it shapes the tree around which Majorana operators actually
/// appear together in the Hamiltonian being mapped.
///
/// This is the native routine behind :func:`qiskit_fermions.mappers.optimization.hatt_tree`. It
/// takes a :class:`.MajoranaOperator` and a required ``num_modes``, where the public function also
/// accepts any operator implementing :class:`.SupportsMajoranaOperator`, can infer ``num_modes``, and
/// documents the algorithm, its cost, and its tie-breaking.
#[gen_stub_pyfunction(
    module = "qiskit_fermions._lib.mappers.mappers_optimization.ternary_tree_search"
)]
#[pyfunction(name = "hatt_tree")]
#[pyo3(signature = (hamiltonian, num_modes, *, weighted = false))]
pub fn py_hatt_tree(
    hamiltonian: &Bound<PyMajoranaOperator>,
    num_modes: u32,
    weighted: bool,
) -> PyResult<PyTernaryTree> {
    // Borrowed rather than taken by value: `PyMajoranaOperator` is `Clone`, so extracting it would
    // copy every term buffer purely to read it.
    let tree =
        hatt_tree(&hamiltonian.borrow().inner, num_modes, weighted).map_err(crate::value_err)?;
    Ok(PyTernaryTree { inner: tree })
}

#[pymodule]
pub mod ternary_tree_search {
    #[pymodule_export]
    use super::py_hatt_tree;
}
