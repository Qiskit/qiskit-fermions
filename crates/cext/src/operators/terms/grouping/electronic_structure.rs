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
use crate::pointers::mut_ptr_as_ref;

use qiskit_fermions_core::operators::fermion_operator::FermionOperator;
use qiskit_fermions_core::operators::terms::grouping::electronic_structure::group_terms_by_electronic_structure;

/// @ingroup qf_operator_terms_grouping
///
/// @brief Groups the terms of an operator by their electronic structure.
///
/// @param op A pointer to the normal-ordered fermionic operator whose terms are to be grouped.
/// @param num_modes The number of fermionic modes in the operator.
/// @param two_body_physicist_order whether the 2-body terms are stored in physicists order.
///
/// @return An exit code.
/// * ``QfExitCode_Success`` upon success
/// * ``QfExitCode_ValueError`` if an invalid term is encountered during the grouping.
///
/// @rst
///
/// This function automatically populates the ``groups`` attribute (see also
/// :c:func:`qf_ferm_op_get_groups`) of the provided :c:struct:`QfFermionOperator` such that terms
/// satisfying a symmetric perturbation present in electronic-structure Hamiltonians are grouped.
///
/// .. caution::
///    The provided operator *must* be normal-ordered! This is an underlying assumption of the
///    implementation that is *not* being verified! See :c:func:`qf_ferm_op_normal_ordered` for how
///    to get an operator of that form.
///
/// More concretely, given an electronic-structure Hamiltonian of the form
///
/// .. math::
///
///    \mathcal{H} = \sum_{ij} c_{ij} a^\dagger_i a_j
///         + \sum_{ijkl} c_{ijkl} a^\dagger_i a^\dagger_j a_k a_l \, ,
///
/// this function will group 1-body terms with permutational symmetry of ``(i, j)`` as well as the
/// 2-body terms with permutational symmetry of ``(i, j, k, l)``. For the 2-body terms, not all
/// permutations will be grouped. Instead, the ``two_body_physicist_order`` determines how the four
/// indices get grouped into pairs of two within which permutational symmetries exist:
///
/// - ``two_body_physicist_order=False`` (`default`): ``(i, l)`` and ``(j, k)``
/// - ``two_body_physicist_order=True``: ``(i, k)`` and ``(j, l)``
///
/// .. code-block:: c
///     :linenos:
///
///     QfFCIDump *fcidump = qf_fcidump_from_file("molecule.fcidump");
///     QfFermionOperator *op = qf_ferm_op_from_fcidump(fcidump);
///
///     uint32_t num_modes = 2 * qf_fcidump_norb(fcidump);
///
///     QfFermionOperator *normal = qf_ferm_op_normal_ordered(op, NULL);
///
///     QfExitCode exit = qf_ferm_op_group_terms_by_electronic_structure(normal, num_modes, false);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ferm_op_group_terms_by_electronic_structure(
    op: *mut FermionOperator,
    num_modes: u32,
    two_body_physicist_order: bool,
) -> ExitCode {
    let op = unsafe { mut_ptr_as_ref(op) };

    let res = group_terms_by_electronic_structure(op, num_modes, two_body_physicist_order);
    match res {
        Ok(_) => ExitCode::Success,
        Err(_) => ExitCode::ValueError,
    }
}
