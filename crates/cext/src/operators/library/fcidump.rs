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

use crate::pointers::const_ptr_as_ref;
use std::ffi::{CStr, c_char};

use qiskit_fermions_core::operators::fermion_operator::FermionOperator;
use qiskit_fermions_core::operators::library::fcidump::FCIDump;

/// @ingroup qf_fcidump
///
/// @brief Parses an FCIDump file.
///
/// @param file_path The path to the FCIDump file.
///
/// @return A pointer to the FCIDump data structure.
///
/// @rst
///
/// Example
/// -------
///
/// Assuming you have an FCIDump file called ``molecule.fcidump``, you use this function like so:
///
/// .. code-block:: c
///     :linenos:
///
///     QfFCIDump *fcidump = qf_fcidump_from_file("molecule.fcidump");
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_fcidump_from_file(file_path: *mut c_char) -> *mut FCIDump {
    let rust_file_path = unsafe { CStr::from_ptr(file_path).to_string_lossy().into_owned() };
    let fcidump = FCIDump::from_file(rust_file_path);
    Box::into_raw(Box::new(fcidump))
}

/// @ingroup qf_fcidump
///
/// @brief Frees an existing FCIDump data structure.
///
/// @param fcidump A pointer to the FCIDump data structure to be freed.
///
/// @rst
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     QfFCIDump *fcidump = qf_fcidump_from_file("molecule.fcidump");
///     qf_fcidump_free(fcidump);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_fcidump_free(fcidump: *mut FCIDump) {
    if !fcidump.is_null() {
        if !fcidump.is_aligned() {
            panic!("Attempted to free a non-aligned pointer.")
        }
        // SAFETY: We have verified the pointer is non-null and aligned, so it should be
        // readable by Box.
        unsafe {
            let _ = Box::from_raw(fcidump);
        }
    }
}

/// @ingroup qf_fcidump
///
/// @brief Gets the number of orbitals from an FCIDump.
///
/// @param fcidump A pointer to the FCIDump data structure to be freed.
///
/// @return The number of orbitals.
///
/// @rst
///
/// This number, :math:`n`, is extracted from the ``NORB=n`` field in the header of the FCIDump
/// file.
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     QfFCIDump *fcidump = qf_fcidump_from_file("molecule.fcidump");
///     uint32_t norb = qf_fcidump_norb(fcidump);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_fcidump_norb(fcidump: *const FCIDump) -> u32 {
    let fcidump = unsafe { const_ptr_as_ref(fcidump) };
    fcidump.norb
}

/// @ingroup qf_fcidump
///
/// @brief Gets the number of electrons from an FCIDump.
///
/// @param fcidump A pointer to the FCIDump data structure to be freed.
///
/// @return The number of electrons.
///
/// @rst
///
/// This number, :math:`n`, is extracted from the ``NELEC=n`` field in the header of the FCIDump
/// file.
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     QfFCIDump *fcidump = qf_fcidump_from_file("molecule.fcidump");
///     uint32_t nelec = qf_fcidump_nelec(fcidump);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_fcidump_nelec(fcidump: *const FCIDump) -> u32 {
    let fcidump = unsafe { const_ptr_as_ref(fcidump) };
    fcidump.nelec
}

/// @ingroup qf_fcidump
///
/// @brief Gets the spin quantum number from an FCIDump.
///
/// @param fcidump A pointer to the FCIDump data structure to be freed.
///
/// @return The spin quantum number (multiplied by 2 to ensure an integer value).
///
/// @rst
///
/// This number, :math:`S`, is extracted from the ``MS2=S`` field in the header of the FCIDump
/// file.
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     QfFCIDump *fcidump = qf_fcidump_from_file("molecule.fcidump");
///     uint32_t ms2 = qf_fcidump_ms2(fcidump);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_fcidump_ms2(fcidump: *const FCIDump) -> u32 {
    let fcidump = unsafe { const_ptr_as_ref(fcidump) };
    fcidump.ms2
}

/// @ingroup qf_fcidump_constructors
///
/// @brief Constructs an :c:struct:`QfFermionOperator` from a :c:struct:`QfFCIDump`.
///
/// @param fcidump A pointer to the FCIDump data structure.
///
/// @return A pointer to the created operator.
///
/// @rst
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     QfFCIDump *fcidump = qf_fcidump_from_file("molecule.fcidump");
///     QfFermionOperator *op = qf_ferm_op_from_fcidump(fcidump);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ferm_op_from_fcidump(fcidump: *const FCIDump) -> *mut FermionOperator {
    let fcidump = unsafe { const_ptr_as_ref(fcidump) };
    let op = FermionOperator::from(fcidump);
    Box::into_raw(Box::new(op))
}
