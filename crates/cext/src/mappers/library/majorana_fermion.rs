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

use qiskit_fermions_core::mappers::library::majorana_fermion::{
    fermion_to_majorana, majorana_to_fermion,
};
use qiskit_fermions_core::operators::fermion_operator::FermionOperator;
use qiskit_fermions_core::operators::majorana_operator::MajoranaOperator;

/// @ingroup qf_mapper_library
///
/// @brief Map a ``QfFermionOperator`` to a ``QfMajoranaOperator``.
///
/// @param fer_op A pointer to the fermionic operator to be mapped.
///
/// @return A pointer to the mapped majorana operator.
///
/// @rst
///
/// Definition
/// ----------
///
/// This function implements the simple transformation:
///
/// .. math::
///
///    a^\dagger_j \rightarrow \frac{1}{2} (\gamma_j - i \gamma'_j) ~~\text{and}~~
///    a_j \rightarrow \frac{1}{2} (\gamma_j + i \gamma'_j)
///
/// where :math:`a^\dagger_j` (:math:`a_j`) is the fermionic creation (annihilation) operator
/// acting on the :math:`j`-th spin-less fermionic mode, and :math:`\gamma_j`/:math:`\gamma'_j` are
/// the two Majorana fermion operators. In the case of the :c:struct:`QfMajoranaOperator` these
/// will be stored on the even and odd Majorana modes, respectively.
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     // define some kind of fermionic operator
///     QfFermionOperator *fer_op = qf_ferm_op_one();
///
///     // and map it to a majorana operator
///     QfMajoranaOperator *maj_op = qf_fermion_to_majorana(fer_op);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_fermion_to_majorana(
    fer_op: *const FermionOperator,
) -> *mut MajoranaOperator {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let fer_op = unsafe { const_ptr_as_ref(fer_op) };

    let maj_op = fermion_to_majorana(fer_op);
    Box::into_raw(Box::new(maj_op))
}

/// @ingroup qf_mapper_library
///
/// @brief Map a ``QfMajoranaOperator`` to a ``QfFermionOperator``.
///
/// @param maj_op A pointer to the majorana operator to be mapped.
///
/// @return A pointer to the mapped fermion operator.
///
/// @rst
///
/// Definition
/// ----------
///
/// This function implements the simple transformation:
///
/// .. math::
///
///    \gamma_j \rightarrow a^\dagger_j + a_j ~~\text{and}~~
///    \gamma'_j \rightarrow i (a^\dagger_j - a_j)
///
/// where :math:`\gamma_j`/:math:`\gamma'_j` are the two Majorana fermion operators (stored on the
/// even and odd modes, respectively), and :math:`a^\dagger_j` (:math:`a_j`) is the fermionic
/// creation (annihilation) operator acting on the :math:`j`-th spin-less fermionic mode.
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     // define some kind of majorana operator
///     QfMajoranaOperator *maj_op = qf_maj_op_one();
///
///     // and map it to a fermion operator
///     QfFermionOperator *fer_op = qf_majorana_to_fermion(maj_op);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_majorana_to_fermion(
    maj_op: *const MajoranaOperator,
) -> *mut FermionOperator {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let maj_op = unsafe { const_ptr_as_ref(maj_op) };

    let fer_op = majorana_to_fermion(maj_op);
    Box::into_raw(Box::new(fer_op))
}
