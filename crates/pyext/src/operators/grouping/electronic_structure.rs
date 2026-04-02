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
use pyo3::{exceptions::PyValueError, prelude::*};
use pyo3_stub_gen::derive::*;
use qiskit_fermions_core::operators::grouping::electronic_structure::group_terms_by_electronic_structure;

/// Groups terms of an operator by their electronic structure.
#[gen_stub_pyfunction(module = "qiskit_fermions.operators.grouping.electronic_structure")]
#[pyfunction(name = "group_terms_by_electronic_structure")]
#[pyo3(signature = (op, num_modes, *, two_body_physicist_order=false))]
pub fn py_group_terms_by_electronic_structure(
    op: &Bound<PyFermionOperator>,
    num_modes: u32,
    two_body_physicist_order: bool,
) -> PyResult<()> {
    let res = group_terms_by_electronic_structure(
        &mut op.borrow_mut().inner,
        num_modes,
        two_body_physicist_order,
    );
    match res {
        Ok(_) => Ok(()),
        Err(e) => Err(PyValueError::new_err(e.to_string())),
    }
}

#[pymodule]
pub mod electronic_structure {
    #[pymodule_export]
    use super::py_group_terms_by_electronic_structure;
}
