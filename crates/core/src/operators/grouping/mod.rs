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

use thiserror::Error;

#[derive(Error, Debug)]
pub enum GroupingError {
    #[error("the operator does not conform to an electronic structure observable")]
    ElectronicStructureError,
}

pub mod electronic_structure;
