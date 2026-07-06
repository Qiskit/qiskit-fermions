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

use crate::pointers::mut_ptr_as_ref;

use qiskit_fermions_core::operators::fermion_operator::FermionOperator;
use qiskit_fermions_core::operators::terms::filtering::diagonal::filter_diagonal_terms;

/// @ingroup qf_operator_terms_filtering
///
/// @brief Filters out the terms of an operator that are diagonal in the occupation-number basis.
///
/// @param op A pointer to the normal-ordered fermionic operator whose terms are to be filtered
/// (modified in place).
///
/// @rst
///
/// This removes every term from the provided :c:struct:`QfFermionOperator` that is a product of
/// number operators (:math:`a^\dagger_i a_i`). This includes the constant term (a product of zero
/// number operators), single number operators, as well as higher-order products such as
/// :math:`n_i n_j = a^\dagger_i a^\dagger_j a_j a_i`.
///
/// .. hint::
///    A major motivation for filtering such terms arises during the time evolution of operators,
///    where such diagonal terms, do not affect the time evolution besides introducing a global
///    phase. Consequently, bitstrings sampled from a time evolved state remain unaffected.
///    Removing them is therefore recommended, for example when preparing an electronic-structure
///    Hamiltonian for the qDRIFT time evolution used by SqDRIFT, as it avoids an otherwise
///    unnecessary sampling overhead.
///
/// .. caution::
///    The provided operator *must* be normal-ordered! This is an underlying assumption of the
///    implementation that is *not* being verified! See :c:func:`qf_ferm_op_normal_ordered` for how
///    to get an operator of that form.
///
/// .. note::
///    The operator is modified *in place*. If it tracks group indices (see
///    :c:func:`qf_ferm_op_get_groups`), surviving terms retain their relative grouping but the
///    group indices are reassigned to a contiguous range starting from 0. You must therefore *not*
///    rely on the specific group index of any term being preserved across a call to this function.
///
/// .. code-block:: c
///     :linenos:
///
///     QfFCIDump *fcidump = qf_fcidump_from_file("molecule.fcidump");
///     QfFermionOperator *op = qf_ferm_op_from_fcidump(fcidump);
///
///     QfFermionOperator *normal = qf_ferm_op_normal_ordered(op);
///
///     qf_filter_diagonal_terms(normal);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_filter_diagonal_terms(op: *mut FermionOperator) {
    let op = unsafe { mut_ptr_as_ref(op) };

    filter_diagonal_terms(op);
}
