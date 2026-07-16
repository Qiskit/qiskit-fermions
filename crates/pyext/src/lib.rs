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

use std::fmt::Display;

use pyo3::prelude::*;
use pyo3_stub_gen::{Result, StubGenConfig, StubInfo};

pub mod linalg;
pub mod mappers;
pub mod operators;

/// Converts any displayable error into a [`PyValueError`](pyo3::exceptions::PyValueError).
///
/// Handy as a `.map_err(crate::value_err)` argument when surfacing a core error to Python.
pub(crate) fn value_err<E: Display>(e: E) -> PyErr {
    ::pyo3::exceptions::PyValueError::new_err(e.to_string())
}

#[pymodule]
mod _lib {
    use pyo3::prelude::*;
    use qiskit_pyo3_ffi as ffi;

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        ffi::qk_import(m.py())?;
        Ok(())
    }

    #[pymodule_export]
    use super::linalg::linalg;

    #[pymodule_export]
    use super::operators::operators;

    #[pymodule_export]
    use super::mappers::mappers;
}

pub fn stub_info() -> Result<StubInfo> {
    let mut config = StubGenConfig::default();
    config.use_type_statement = false;
    config.doc_gen = None;

    let manifest_dir: &::std::path::Path = env!("CARGO_MANIFEST_DIR").as_ref();
    StubInfo::from_project_root(
        "qiskit_fermions._lib".to_string(),
        manifest_dir.join("../../python"),
        true,
        config,
    )
}
