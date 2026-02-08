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

/// Errors related to C input.
#[derive(Error, Debug)]
pub enum CInputError {
    #[error("Unexpected null pointer.")]
    NullPointerError,
    #[error("Non-aligned memory.")]
    AlignmentError,
    #[error("Index out of bounds.")]
    IndexError,
}

/// @ingroup qf_exit_code
///
/// Integer exit codes returned to C.
#[repr(u32)]
#[derive(PartialEq, Eq, Debug)]
pub enum ExitCode {
    /// `0`: Success.
    Success = 0,
    /// `100`: Error related to data input.
    CInputError = 100,
    /// `101`: Unexpected null pointer.
    NullPointerError = 101,
    /// `102`: Pointer is not aligned to expected data.
    AlignmentError = 102,
    /// `103`: Index out of bounds.
    IndexError = 103,
}

impl From<CInputError> for ExitCode {
    fn from(value: CInputError) -> Self {
        match value {
            CInputError::AlignmentError => ExitCode::AlignmentError,
            CInputError::NullPointerError => ExitCode::NullPointerError,
            CInputError::IndexError => ExitCode::IndexError,
        }
    }
}
