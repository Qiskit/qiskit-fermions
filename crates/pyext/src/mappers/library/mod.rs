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
use qiskit_pyo3_ffi as ffi;

pub mod edge_vertex;
pub mod jordan_wigner;
pub mod majorana_fermion;
pub mod ternary_tree;
pub mod transfer_vertex;

/// Converts a mapped `QkObs` into the Python `SparseObservable` that owns it.
///
/// Shared by every mapper in this module tree: each one ends by handing its `QkObs` to Python, and
/// that hand-off is the same regardless of the encoding.
///
/// # Safety
///
/// `obs` must be a valid, uniquely-owned `QkObs` pointer; ownership transfers to Python.
pub(crate) unsafe fn into_py_obs(obs: *mut ffi::QkObs) -> Py<PyAny> {
    unsafe {
        let py = Python::assume_attached();
        let py_obs = ffi::qk_obs_to_python(obs);
        Bound::from_owned_ptr(py, py_obs).into()
    }
}

#[pymodule]
pub mod mappers_library {
    #[pymodule_export]
    use super::transfer_vertex::transfer_vertex;

    #[pymodule_export]
    use super::edge_vertex::edge_vertex;

    #[pymodule_export]
    use super::jordan_wigner::jordan_wigner;

    #[pymodule_export]
    use super::majorana_fermion::majorana_fermion;

    #[pymodule_export]
    use super::ternary_tree::ternary_tree;
}
