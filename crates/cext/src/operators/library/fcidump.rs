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

use crate::exit_codes::ExitCode;
use crate::pointers::const_ptr_as_ref;
use std::ffi::{CStr, c_char};

use qiskit_fermions_core::operators::fermion_operator::FermionOperator;
use qiskit_fermions_core::operators::library::fcidump::FCIDump;

/// @ingroup qf_fcidump
///
/// @brief Parses an FCIDump file.
///
/// @param file_path The path to the FCIDump file.
/// @param out A pointer to the pointer that will be set to the parsed FCIDump data structure. It is
///        only written to on success; on failure it is left untouched.
///
/// @return An exit code. This is ``QfExitCode_ValueError`` if the file cannot be opened or read, or
///         if it does not honour the FCIDump format (a missing header namelist, a missing ``NORB``
///         or ``NELEC`` field, a malformed ``MS2`` field, or an unsupported MO energy value).
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
///     QfFCIDump *fcidump = NULL;
///     QfExitCode exit = qf_fcidump_from_file("molecule.fcidump", &fcidump);
///
///     assert(exit == QfExitCode_Success);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_fcidump_from_file(
    file_path: *mut c_char,
    out: *mut *mut FCIDump,
) -> ExitCode {
    let rust_file_path = unsafe { CStr::from_ptr(file_path).to_string_lossy().into_owned() };

    // The core reports every parse failure as an `FCIDumpError`. Surfacing it as an exit code (and
    // leaving `out` alone) is what keeps a bad path from unwinding across the FFI boundary, which is
    // undefined behaviour in an `extern "C"` function.
    match FCIDump::from_file(rust_file_path) {
        Ok(fcidump) => {
            // SAFETY: Per documentation, `out` is non-null and aligned.
            unsafe { out.write(Box::into_raw(Box::new(fcidump))) };
            ExitCode::Success
        }
        Err(_) => ExitCode::ValueError,
    }
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
///     QfFCIDump *fcidump = NULL;
///     qf_fcidump_from_file("molecule.fcidump", &fcidump);
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
/// @param fcidump A pointer to the FCIDump data structure to query.
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
///     QfFCIDump *fcidump = NULL;
///     qf_fcidump_from_file("molecule.fcidump", &fcidump);
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
/// @param fcidump A pointer to the FCIDump data structure to query.
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
///     QfFCIDump *fcidump = NULL;
///     qf_fcidump_from_file("molecule.fcidump", &fcidump);
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
/// @param fcidump A pointer to the FCIDump data structure to query.
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
///     QfFCIDump *fcidump = NULL;
///     qf_fcidump_from_file("molecule.fcidump", &fcidump);
///     uint32_t ms2 = qf_fcidump_ms2(fcidump);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_fcidump_ms2(fcidump: *const FCIDump) -> u32 {
    let fcidump = unsafe { const_ptr_as_ref(fcidump) };
    fcidump.ms2
}

/// @ingroup qf_fcidump
///
/// @brief Checks whether an FCIDump carries unrestricted (spin-dependent) integrals.
///
/// @param fcidump A pointer to the FCIDump data structure to query.
///
/// @return ``true`` if the beta-spin integrals are present.
///
/// @rst
///
/// The beta-spin 1-body and the alpha-beta / beta-beta 2-body arrays are present exactly when this
/// returns ``true``; they are absent together. Use it to guard
/// :c:func:`qf_fcidump_get_one_body_tril_b`, :c:func:`qf_fcidump_get_two_body_tril_ab` and
/// :c:func:`qf_fcidump_get_two_body_tril_bb`.
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     QfFCIDump *fcidump = NULL;
///     qf_fcidump_from_file("molecule.fcidump", &fcidump);
///
///     bool unrestricted = qf_fcidump_is_unrestricted(fcidump);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_fcidump_is_unrestricted(fcidump: *const FCIDump) -> bool {
    let fcidump = unsafe { const_ptr_as_ref(fcidump) };
    fcidump.one_body_b.is_some()
}

/// @ingroup qf_fcidump
///
/// @brief Checks whether an FCIDump provides a constant energy offset.
///
/// @param fcidump A pointer to the FCIDump data structure to query.
///
/// @return ``true`` if the file carried an integral line whose four indices are all zero.
///
/// @rst
///
/// Use this to guard :c:func:`qf_fcidump_constant`. This is independent of
/// :c:func:`qf_fcidump_is_unrestricted`: a spin-restricted file may well carry a constant.
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     QfFCIDump *fcidump = NULL;
///     qf_fcidump_from_file("molecule.fcidump", &fcidump);
///
///     bool has_constant = qf_fcidump_has_constant(fcidump);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_fcidump_has_constant(fcidump: *const FCIDump) -> bool {
    let fcidump = unsafe { const_ptr_as_ref(fcidump) };
    fcidump.constant.is_some()
}

/// @ingroup qf_fcidump
///
/// @brief Gets the constant energy offset from an FCIDump.
///
/// @param fcidump A pointer to the FCIDump data structure to query.
///
/// @return The constant energy offset.
///
/// @rst
///
/// This is the value of the integral line whose four indices are all zero, which for an electronic
/// structure Hamiltonian is the nuclear-repulsion energy.
///
/// .. warning::
///    Only call this when :c:func:`qf_fcidump_has_constant` returns ``true``; calling it otherwise
///    panics.
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     QfFCIDump *fcidump = NULL;
///     qf_fcidump_from_file("molecule.fcidump", &fcidump);
///
///     double constant = 0.0;
///     if (qf_fcidump_has_constant(fcidump)) {
///         constant = qf_fcidump_constant(fcidump);
///     }
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_fcidump_constant(fcidump: *const FCIDump) -> f64 {
    let fcidump = unsafe { const_ptr_as_ref(fcidump) };
    fcidump.constant.expect(
        "Expected a constant to be present. It is the user's responsibility to check this via \
         qf_fcidump_has_constant before calling this function.",
    )
}

/// @ingroup qf_fcidump
///
/// @brief Provides read-only access to the alpha-spin 1-body integrals.
///
/// @param fcidump A pointer to the FCIDump data structure whose integrals to access.
/// @param out A pointer to the array of doubles into which to write the integrals.
/// @param out_len A pointer to the integer into which to write the length of the output array.
///
/// @rst
///
/// .. note::
///    This function borrows the FCIDump's internal buffer rather than copying it. The returned
///    pointer stays valid only until the data structure is freed; do not free it yourself.
///
/// .. seealso::
///    The explanation of the internal data layout, :ref:`here <qf_fcidump-implementation>`.
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     QfFCIDump *fcidump = NULL;
///     qf_fcidump_from_file("molecule.fcidump", &fcidump);
///
///     double *integrals;
///     uint64_t len;
///     qf_fcidump_get_one_body_tril_a(fcidump, &integrals, &len);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_fcidump_get_one_body_tril_a(
    fcidump: *const FCIDump,
    out: *mut *mut f64,
    out_len: *mut u64,
) {
    let fcidump = unsafe { const_ptr_as_ref(fcidump) };
    // The arrays are built by `Array1::zeros` and only ever indexed-assigned, never sliced, so they
    // are contiguous and `as_ptr` covers the whole buffer. A future refactor introducing a slice
    // here would break that and must copy instead.
    let arr = &fcidump.one_body_a;
    unsafe { out.write(arr.as_ptr().cast_mut()) };
    unsafe { out_len.write(arr.len().try_into().unwrap()) };
}

/// @ingroup qf_fcidump
///
/// @brief Provides read-only access to the beta-spin 1-body integrals.
///
/// @param fcidump A pointer to the FCIDump data structure whose integrals to access.
/// @param out A pointer to the array of doubles into which to write the integrals.
/// @param out_len A pointer to the integer into which to write the length of the output array.
///
/// @rst
///
/// .. warning::
///    Only call this on a data structure carrying unrestricted spin data. Check
///    :c:func:`qf_fcidump_is_unrestricted` first; calling it otherwise panics.
///
/// @endrst
///
/// @rst
///
/// .. note::
///    This function borrows the FCIDump's internal buffer rather than copying it. The returned
///    pointer stays valid only until the data structure is freed; do not free it yourself.
///
/// .. seealso::
///    The explanation of the internal data layout, :ref:`here <qf_fcidump-implementation>`.
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     QfFCIDump *fcidump = NULL;
///     qf_fcidump_from_file("molecule.fcidump", &fcidump);
///
///     double *integrals;
///     uint64_t len;
///     qf_fcidump_get_one_body_tril_b(fcidump, &integrals, &len);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_fcidump_get_one_body_tril_b(
    fcidump: *const FCIDump,
    out: *mut *mut f64,
    out_len: *mut u64,
) {
    let fcidump = unsafe { const_ptr_as_ref(fcidump) };
    // The arrays are built by `Array1::zeros` and only ever indexed-assigned, never sliced, so they
    // are contiguous and `as_ptr` covers the whole buffer. A future refactor introducing a slice
    // here would break that and must copy instead.
    let arr = fcidump.one_body_b.as_ref().expect(
        "Expected unrestricted spin data to be present. It is the user's responsibility to \
         check this via qf_fcidump_is_unrestricted before calling this function.",
    );
    unsafe { out.write(arr.as_ptr().cast_mut()) };
    unsafe { out_len.write(arr.len().try_into().unwrap()) };
}

/// @ingroup qf_fcidump
///
/// @brief Provides read-only access to the alpha-alpha-spin 2-body integrals.
///
/// @param fcidump A pointer to the FCIDump data structure whose integrals to access.
/// @param out A pointer to the array of doubles into which to write the integrals.
/// @param out_len A pointer to the integer into which to write the length of the output array.
///
/// @rst
///
/// .. note::
///    This function borrows the FCIDump's internal buffer rather than copying it. The returned
///    pointer stays valid only until the data structure is freed; do not free it yourself.
///
/// .. note::
///    The values are in `chemist` ordering, :math:`(ij|kl)`, and carry no factor of
///    :math:`\frac{1}{2}`.
///
/// .. seealso::
///    The explanation of the internal data layout, :ref:`here <qf_fcidump-implementation>`.
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     QfFCIDump *fcidump = NULL;
///     qf_fcidump_from_file("molecule.fcidump", &fcidump);
///
///     double *integrals;
///     uint64_t len;
///     qf_fcidump_get_two_body_tril_aa(fcidump, &integrals, &len);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_fcidump_get_two_body_tril_aa(
    fcidump: *const FCIDump,
    out: *mut *mut f64,
    out_len: *mut u64,
) {
    let fcidump = unsafe { const_ptr_as_ref(fcidump) };
    // The arrays are built by `Array1::zeros` and only ever indexed-assigned, never sliced, so they
    // are contiguous and `as_ptr` covers the whole buffer. A future refactor introducing a slice
    // here would break that and must copy instead.
    let arr = &fcidump.two_body_aa;
    unsafe { out.write(arr.as_ptr().cast_mut()) };
    unsafe { out_len.write(arr.len().try_into().unwrap()) };
}

/// @ingroup qf_fcidump
///
/// @brief Provides read-only access to the alpha-beta-spin 2-body integrals.
///
/// @param fcidump A pointer to the FCIDump data structure whose integrals to access.
/// @param out A pointer to the array of doubles into which to write the integrals.
/// @param out_len A pointer to the integer into which to write the length of the output array.
///
/// @rst
///
/// .. warning::
///    Only call this on a data structure carrying unrestricted spin data. Check
///    :c:func:`qf_fcidump_is_unrestricted` first; calling it otherwise panics.
///
/// @endrst
///
/// @rst
///
/// .. note::
///    This function borrows the FCIDump's internal buffer rather than copying it. The returned
///    pointer stays valid only until the data structure is freed; do not free it yourself.
///
/// .. note::
///    The values are in `chemist` ordering, :math:`(ij|kl)`, and carry no factor of
///    :math:`\frac{1}{2}`.
///
/// .. warning::
///    Unlike the other 2-body arrays, this one is only 4-fold (S4) symmetric: it holds the
///    **full** ``(npair, npair)`` matrix, whose row pair indexes the alpha-spin and whose
///    column pair indexes the beta-spin species. It is therefore *not* symmetric under
///    exchanging the two pairs.
///
/// .. seealso::
///    The explanation of the internal data layout, :ref:`here <qf_fcidump-implementation>`.
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     QfFCIDump *fcidump = NULL;
///     qf_fcidump_from_file("molecule.fcidump", &fcidump);
///
///     double *integrals;
///     uint64_t len;
///     qf_fcidump_get_two_body_tril_ab(fcidump, &integrals, &len);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_fcidump_get_two_body_tril_ab(
    fcidump: *const FCIDump,
    out: *mut *mut f64,
    out_len: *mut u64,
) {
    let fcidump = unsafe { const_ptr_as_ref(fcidump) };
    // The arrays are built by `Array1::zeros` and only ever indexed-assigned, never sliced, so they
    // are contiguous and `as_ptr` covers the whole buffer. A future refactor introducing a slice
    // here would break that and must copy instead.
    let arr = fcidump.two_body_ab.as_ref().expect(
        "Expected unrestricted spin data to be present. It is the user's responsibility to \
         check this via qf_fcidump_is_unrestricted before calling this function.",
    );
    unsafe { out.write(arr.as_ptr().cast_mut()) };
    unsafe { out_len.write(arr.len().try_into().unwrap()) };
}

/// @ingroup qf_fcidump
///
/// @brief Provides read-only access to the beta-beta-spin 2-body integrals.
///
/// @param fcidump A pointer to the FCIDump data structure whose integrals to access.
/// @param out A pointer to the array of doubles into which to write the integrals.
/// @param out_len A pointer to the integer into which to write the length of the output array.
///
/// @rst
///
/// .. warning::
///    Only call this on a data structure carrying unrestricted spin data. Check
///    :c:func:`qf_fcidump_is_unrestricted` first; calling it otherwise panics.
///
/// @endrst
///
/// @rst
///
/// .. note::
///    This function borrows the FCIDump's internal buffer rather than copying it. The returned
///    pointer stays valid only until the data structure is freed; do not free it yourself.
///
/// .. note::
///    The values are in `chemist` ordering, :math:`(ij|kl)`, and carry no factor of
///    :math:`\frac{1}{2}`.
///
/// .. seealso::
///    The explanation of the internal data layout, :ref:`here <qf_fcidump-implementation>`.
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     QfFCIDump *fcidump = NULL;
///     qf_fcidump_from_file("molecule.fcidump", &fcidump);
///
///     double *integrals;
///     uint64_t len;
///     qf_fcidump_get_two_body_tril_bb(fcidump, &integrals, &len);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_fcidump_get_two_body_tril_bb(
    fcidump: *const FCIDump,
    out: *mut *mut f64,
    out_len: *mut u64,
) {
    let fcidump = unsafe { const_ptr_as_ref(fcidump) };
    // The arrays are built by `Array1::zeros` and only ever indexed-assigned, never sliced, so they
    // are contiguous and `as_ptr` covers the whole buffer. A future refactor introducing a slice
    // here would break that and must copy instead.
    let arr = fcidump.two_body_bb.as_ref().expect(
        "Expected unrestricted spin data to be present. It is the user's responsibility to \
         check this via qf_fcidump_is_unrestricted before calling this function.",
    );
    unsafe { out.write(arr.as_ptr().cast_mut()) };
    unsafe { out_len.write(arr.len().try_into().unwrap()) };
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
///     QfFCIDump *fcidump = NULL;
///     qf_fcidump_from_file("molecule.fcidump", &fcidump);
///     QfFermionOperator *op = qf_ferm_op_from_fcidump(fcidump);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ferm_op_from_fcidump(fcidump: *const FCIDump) -> *mut FermionOperator {
    let fcidump = unsafe { const_ptr_as_ref(fcidump) };
    let op = FermionOperator::from(fcidump);
    Box::into_raw(Box::new(op))
}
