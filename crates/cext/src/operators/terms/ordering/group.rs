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

use qiskit_fermions_core::operators::edge_vertex_operator::EdgeVertexOperator;
use qiskit_fermions_core::operators::fermion_operator::FermionOperator;
use qiskit_fermions_core::operators::majorana_operator::MajoranaOperator;
use qiskit_fermions_core::operators::terms::ordering::group::group_order;
use qiskit_fermions_core::operators::transfer_vertex_operator::TransferVertexOperator;

/// @ingroup qf_operator_terms_ordering
///
/// @brief Returns a copy of a fermionic operator with its terms ordered by group index.
///
/// @param op A pointer to the operator whose terms are to be reordered.
///
/// @return A pointer to the created operator.
///
/// @rst
///
/// The terms are sorted by their group index alone, which makes each group one contiguous run of
/// terms and the group indices non-decreasing. The sort is stable, so terms within a group keep their
/// relative order. The terms themselves are left untouched — this only reorders them, it does not
/// simplify or normal-order the operator.
///
/// Group indices say only which terms belong together, so this changes the operator's *layout*, not
/// its value. What the layout buys is lookup cost:
/// :c:func:`qf_ferm_op_split_out_groups` has to scan every term to find the requested groups in
/// general, but on a group-ordered operator each group's terms form one range it can locate by binary
/// search, so a lookup costs what the *requested* groups cost rather than what the *held* terms cost.
///
/// .. note::
///    An operator tracking no group indices (see :c:func:`qf_ferm_op_get_groups`) has nothing to
///    order by and is returned as an unchanged copy. The result tracks no groups either.
///
/// .. code-block:: c
///     :linenos:
///
///     QfFermionOperator *op = qf_ferm_op_zero();
///     bool actions[2] = {true, false};
///     uint32_t modes_a[2] = {0, 1};
///     QkComplex64 coeff_a = {1.0, 0.0};
///     qf_ferm_op_add_term(op, 2, actions, modes_a, &coeff_a);
///     uint32_t modes_b[2] = {1, 0};
///     QkComplex64 coeff_b = {2.0, 0.0};
///     qf_ferm_op_add_term(op, 2, actions, modes_b, &coeff_b);
///     uint32_t groups[2] = {1, 0};
///     qf_ferm_op_set_groups(op, groups, 2);
///
///     QfFermionOperator *ordered = qf_ferm_op_group_order(op);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ferm_op_group_order(
    op: *const FermionOperator,
) -> *mut FermionOperator {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let op = unsafe { const_ptr_as_ref(op) };

    let result = group_order(op);
    Box::into_raw(Box::new(result))
}

/// @ingroup qf_operator_terms_ordering
///
/// @brief Returns a copy of a Majorana operator with its terms ordered by group index.
///
/// @param op A pointer to the operator whose terms are to be reordered.
///
/// @return A pointer to the created operator.
///
/// @rst
///
/// The terms are sorted by their group index alone, which makes each group one contiguous run of
/// terms and the group indices non-decreasing. The sort is stable, so terms within a group keep their
/// relative order. The terms themselves are left untouched — this only reorders them, it does not
/// simplify or normal-order the operator.
///
/// Group indices say only which terms belong together, so this changes the operator's *layout*, not
/// its value. What the layout buys is lookup cost:
/// :c:func:`qf_maj_op_split_out_groups` has to scan every term to find the requested groups in
/// general, but on a group-ordered operator each group's terms form one range it can locate by binary
/// search, so a lookup costs what the *requested* groups cost rather than what the *held* terms cost.
///
/// .. note::
///    An operator tracking no group indices (see :c:func:`qf_maj_op_get_groups`) has nothing to
///    order by and is returned as an unchanged copy. The result tracks no groups either.
///
/// .. code-block:: c
///     :linenos:
///
///     QfMajoranaOperator *op = qf_maj_op_zero();
///     uint32_t modes_a[2] = {0, 1};
///     QkComplex64 coeff_a = {1.0, 0.0};
///     qf_maj_op_add_term(op, 2, modes_a, &coeff_a);
///     uint32_t modes_b[2] = {2, 3};
///     QkComplex64 coeff_b = {2.0, 0.0};
///     qf_maj_op_add_term(op, 2, modes_b, &coeff_b);
///     uint32_t groups[2] = {1, 0};
///     qf_maj_op_set_groups(op, groups, 2);
///
///     QfMajoranaOperator *ordered = qf_maj_op_group_order(op);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_maj_op_group_order(
    op: *const MajoranaOperator,
) -> *mut MajoranaOperator {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let op = unsafe { const_ptr_as_ref(op) };

    let result = group_order(op);
    Box::into_raw(Box::new(result))
}

/// @ingroup qf_operator_terms_ordering
///
/// @brief Returns a copy of an edge-vertex operator with its terms ordered by group index.
///
/// @param op A pointer to the operator whose terms are to be reordered.
///
/// @return A pointer to the created operator.
///
/// @rst
///
/// The terms are sorted by their group index alone, which makes each group one contiguous run of
/// terms and the group indices non-decreasing. The sort is stable, so terms within a group keep their
/// relative order. The terms themselves are left untouched — this only reorders them, it does not
/// simplify or normal-order the operator.
///
/// Group indices say only which terms belong together, so this changes the operator's *layout*, not
/// its value. What the layout buys is lookup cost:
/// :c:func:`qf_edge_op_split_out_groups` has to scan every term to find the requested groups in
/// general, but on a group-ordered operator each group's terms form one range it can locate by binary
/// search, so a lookup costs what the *requested* groups cost rather than what the *held* terms cost.
///
/// .. note::
///    An operator tracking no group indices (see :c:func:`qf_edge_op_get_groups`) has nothing to
///    order by and is returned as an unchanged copy. The result tracks no groups either.
///
/// .. code-block:: c
///     :linenos:
///
///     QfEdgeVertexOperator *op = qf_edge_op_zero();
///     uint32_t left_a[1] = {0};
///     uint32_t right_a[1] = {1};
///     QkComplex64 coeff_a = {1.0, 0.0};
///     qf_edge_op_add_term(op, 1, left_a, right_a, &coeff_a);
///     uint32_t left_b[1] = {1};
///     uint32_t right_b[1] = {2};
///     QkComplex64 coeff_b = {2.0, 0.0};
///     qf_edge_op_add_term(op, 1, left_b, right_b, &coeff_b);
///     uint32_t groups[2] = {1, 0};
///     qf_edge_op_set_groups(op, groups, 2);
///
///     QfEdgeVertexOperator *ordered = qf_edge_op_group_order(op);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_edge_op_group_order(
    op: *const EdgeVertexOperator,
) -> *mut EdgeVertexOperator {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let op = unsafe { const_ptr_as_ref(op) };

    let result = group_order(op);
    Box::into_raw(Box::new(result))
}

/// @ingroup qf_operator_terms_ordering
///
/// @brief Returns a copy of a transfer-vertex operator with its terms ordered by group index.
///
/// @param op A pointer to the operator whose terms are to be reordered.
///
/// @return A pointer to the created operator.
///
/// @rst
///
/// The terms are sorted by their group index alone, which makes each group one contiguous run of
/// terms and the group indices non-decreasing. The sort is stable, so terms within a group keep their
/// relative order. The terms themselves are left untouched — this only reorders them, it does not
/// simplify or normal-order the operator.
///
/// Group indices say only which terms belong together, so this changes the operator's *layout*, not
/// its value. What the layout buys is lookup cost:
/// :c:func:`qf_transfer_op_split_out_groups` has to scan every term to find the requested groups in
/// general, but on a group-ordered operator each group's terms form one range it can locate by binary
/// search, so a lookup costs what the *requested* groups cost rather than what the *held* terms cost.
///
/// .. note::
///    An operator tracking no group indices (see :c:func:`qf_transfer_op_get_groups`) has nothing to
///    order by and is returned as an unchanged copy. The result tracks no groups either.
///
/// .. code-block:: c
///     :linenos:
///
///     QfTransferVertexOperator *op = qf_transfer_op_zero();
///     uint32_t left_a[1] = {0};
///     uint32_t right_a[1] = {1};
///     QkComplex64 coeff_a = {1.0, 0.0};
///     qf_transfer_op_add_term(op, 1, left_a, right_a, &coeff_a);
///     uint32_t left_b[1] = {1};
///     uint32_t right_b[1] = {2};
///     QkComplex64 coeff_b = {2.0, 0.0};
///     qf_transfer_op_add_term(op, 1, left_b, right_b, &coeff_b);
///     uint32_t groups[2] = {1, 0};
///     qf_transfer_op_set_groups(op, groups, 2);
///
///     QfTransferVertexOperator *ordered = qf_transfer_op_group_order(op);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_transfer_op_group_order(
    op: *const TransferVertexOperator,
) -> *mut TransferVertexOperator {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let op = unsafe { const_ptr_as_ref(op) };

    let result = group_order(op);
    Box::into_raw(Box::new(result))
}
