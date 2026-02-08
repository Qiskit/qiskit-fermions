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

pub mod jordan_wigner;
pub mod majorana_fermion;

#[pymodule]
pub mod mappers_library {
    #[pymodule_export]
    use super::jordan_wigner::jordan_wigner;

    #[pymodule_export]
    use super::majorana_fermion::majorana_fermion;
}
