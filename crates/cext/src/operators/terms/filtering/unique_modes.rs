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
use qiskit_fermions_core::operators::terms::filtering::unique_modes::filter_terms_by_num_unique_modes;

/// @ingroup qf_operator_terms_filtering
///
/// @brief Filters out the terms of an operator that act on too few unique modes.
///
/// @param op A pointer to the fermionic operator whose terms are to be filtered (modified in
/// place).
/// @param min_unique_modes The minimum number of unique modes a term must act on to be kept.
///
/// @rst
///
/// This removes every term from the provided :c:struct:`QfFermionOperator` that acts on fewer than
/// ``min_unique_modes`` *unique* fermionic modes. The number of unique modes is the number of
/// distinct mode indices a term acts on, irrespective of how many actions it has.
///
/// This is most useful when preparing an electronic-structure Hamiltonian for time evolution: such
/// a Hamiltonian contains terms whose inclusion in a time-evolution circuit has no impact on the
/// sampled bitstrings and, thus, only results in an increased sampling overhead. Filtering these
/// out with ``min_unique_modes=2`` removes
///
/// - the constant energy offset (0 unique modes), whose time evolution only introduces a global
///   phase, and
/// - the number operators :math:`a^\dagger_i a_i` (1 unique mode), whose time evolution amounts to
///   single-qubit Z rotations that do not affect the sampled bitstrings.
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
///     QfFermionOperator *hamil = qf_ferm_op_from_fcidump(fcidump);
///
///     // remove the constant offset and all number operators
///     qf_filter_terms_by_num_unique_modes(hamil, 2);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_filter_terms_by_num_unique_modes(
    op: *mut FermionOperator,
    min_unique_modes: u32,
) {
    let op = unsafe { mut_ptr_as_ref(op) };

    filter_terms_by_num_unique_modes(op, min_unique_modes);
}
