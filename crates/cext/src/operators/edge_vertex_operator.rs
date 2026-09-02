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
use crate::pointers::{const_ptr_as_ref, mut_ptr_as_ref, slice_from_ptr};

use num_complex::Complex64;
use qiskit_fermions_core::operators::edge_vertex_operator::EdgeVertexOperator;
use qiskit_fermions_core::operators::{CoherenceError, OperatorMacro, OperatorTrait};

/// @ingroup qf_edge_op
///
/// @brief Constructs a new operator.
///
/// @param num_terms The number of terms in the operator.
/// @param num_indices The number of generators summed over all terms. Both index arrays have this
///     same length, since every generator is identified by exactly one ``(left, right)`` pair.
/// @param coeffs A pointer to an array of term coefficients. The length of this array should be
///     ``num_terms``.
/// @param left_indices A pointer to an array of left-hand mode indices over all terms. The length
///     of this array should be ``num_indices``.
/// @param right_indices A pointer to an array of right-hand mode indices over all terms. The
///     length of this array should be ``num_indices``.
/// @param boundaries A pointer to an array of the boundaries between terms. The length of this
///     array should be ``num_terms + 1``.
///
/// @return A pointer to the created operator.
///
/// @rst
///
/// Any of the pointer arguments can be ``NULL`` if and only if their corresponding length is zero.
///
/// A generator with equal indices is a vertex operator, :math:`V_j = E_{jj}`; one with differing
/// indices is a (generalized) edge operator, :math:`E_{jk}`.
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     // this builds `1.0 + (-1.0) * V(0) E(0,1)`
///     uint64_t num_terms = 2;
///     uint64_t num_indices = 2;
///     uint32_t left_indices[2] = {0, 0};
///     uint32_t right_indices[2] = {0, 1};
///     QkComplex64 coeffs[2] = {{1.0, 0.0}, {-1.0, 0.0}};
///     uint32_t boundaries[3] = {0, 0, 2};
///     QfEdgeVertexOperator *op = qf_edge_op_new(num_terms, num_indices, coeffs,
///                                               left_indices, right_indices, boundaries);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_edge_op_new(
    num_terms: u64,
    num_indices: u64,
    coeffs: *const Complex64,
    left_indices: *const u32,
    right_indices: *const u32,
    boundaries: *const u32,
) -> *mut EdgeVertexOperator {
    let num_terms = num_terms as usize;
    let num_indices = num_indices as usize;

    let op = EdgeVertexOperator {
        coeffs: unsafe { slice_from_ptr(coeffs, num_terms).to_vec() },
        left_indices: unsafe { slice_from_ptr(left_indices, num_indices).to_vec() },
        right_indices: unsafe { slice_from_ptr(right_indices, num_indices).to_vec() },
        boundaries: unsafe {
            slice_from_ptr(boundaries, num_terms + 1)
                .iter()
                .map(|b| *b as usize)
                .collect()
        },
        groups: None,
    };
    Box::into_raw(Box::new(op))
}

/// @ingroup qf_edge_op
///
/// @brief Frees an existing operator.
///
/// @param op A pointer to the edge-vertex operator to be freed.
///
/// @rst
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     QfEdgeVertexOperator *op = qf_edge_op_one();
///     qf_edge_op_free(op);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_edge_op_free(op: *mut EdgeVertexOperator) {
    if !op.is_null() {
        if !op.is_aligned() {
            panic!("Attempted to free a non-aligned pointer.")
        }
        // SAFETY: We have verified the pointer is non-null and aligned, so it should be
        // readable by Box.
        unsafe {
            let _ = Box::from_raw(op);
        }
    }
}

/// @ingroup qf_edge_op
///
/// @brief Provides read-only access to the operator's coefficients.
///
/// @param op A pointer to the edge-vertex operator whose coefficients to access.
/// @param coeffs_out A pointer to the array of complex values into which to write the coefficients.
/// @param coeffs_len A pointer to the integer into which to write the length of the output array.
///
/// @rst
///
/// .. note::
///    This function borrows the operator's internal buffer rather than copying it. The returned
///    pointer stays valid only until the operator is modified or freed; do not free it yourself.
///
/// .. seealso::
///    The explanation of the internal data structure, :ref:`here <qf_edge_op-implementation>`.
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     uint64_t num_terms = 2;
///     QkComplex64 coeffs[2] = {{1.0, 0.0}, {0.0, -1.0}};
///     uint32_t boundaries[3] = {0, 0, 0};
///     QfEdgeVertexOperator *op =
///         qf_edge_op_new(num_terms, 0, coeffs, NULL, NULL, boundaries);
///
///     QkComplex64 *coeffs_out;
///     uint64_t coeffs_len;
///
///     qf_edge_op_get_coeffs(op, &coeffs_out, &coeffs_len);
///
///     assert(coeffs_len == 2);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_edge_op_get_coeffs(
    op: *const EdgeVertexOperator,
    coeffs_out: *mut *mut Complex64,
    coeffs_len: *mut u64,
) {
    let op = unsafe { const_ptr_as_ref(op) };
    unsafe { coeffs_out.write(op.coeffs.as_ptr().cast_mut()) };
    unsafe { coeffs_len.write(op.coeffs.len().try_into().unwrap()) };
}

/// @ingroup qf_edge_op
///
/// @brief Provides read-only access to the operator's left-hand mode indices.
///
/// @param op A pointer to the edge-vertex operator whose left indices to access.
/// @param left_indices_out A pointer to the array of integers into which to write the indices.
/// @param left_indices_len A pointer to the integer into which to write the length of the output
///     array.
///
/// @rst
///
/// .. note::
///    This function borrows the operator's internal buffer rather than copying it. The returned
///    pointer stays valid only until the operator is modified or freed; do not free it yourself.
///
/// .. seealso::
///    The explanation of the internal data structure, :ref:`here <qf_edge_op-implementation>`.
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     uint64_t num_terms = 1;
///     uint64_t num_indices = 2;
///     uint32_t left_indices[2] = {0, 0};
///     uint32_t right_indices[2] = {0, 1};
///     QkComplex64 coeffs[1] = {{1.0, 0.0}};
///     uint32_t boundaries[2] = {0, 2};
///     QfEdgeVertexOperator *op = qf_edge_op_new(num_terms, num_indices, coeffs,
///                                               left_indices, right_indices, boundaries);
///
///     uint32_t *left_out;
///     uint64_t left_len;
///
///     qf_edge_op_get_left_indices(op, &left_out, &left_len);
///
///     assert(left_len == 2);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_edge_op_get_left_indices(
    op: *const EdgeVertexOperator,
    left_indices_out: *mut *mut u32,
    left_indices_len: *mut u64,
) {
    let op = unsafe { const_ptr_as_ref(op) };
    unsafe { left_indices_out.write(op.left_indices.as_ptr().cast_mut()) };
    unsafe { left_indices_len.write(op.left_indices.len().try_into().unwrap()) };
}

/// @ingroup qf_edge_op
///
/// @brief Provides read-only access to the operator's right-hand mode indices.
///
/// @param op A pointer to the edge-vertex operator whose right indices to access.
/// @param right_indices_out A pointer to the array of integers into which to write the indices.
/// @param right_indices_len A pointer to the integer into which to write the length of the output
///     array.
///
/// @rst
///
/// .. note::
///    This function borrows the operator's internal buffer rather than copying it. The returned
///    pointer stays valid only until the operator is modified or freed; do not free it yourself.
///
/// .. seealso::
///    The explanation of the internal data structure, :ref:`here <qf_edge_op-implementation>`.
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     uint64_t num_terms = 1;
///     uint64_t num_indices = 2;
///     uint32_t left_indices[2] = {0, 0};
///     uint32_t right_indices[2] = {0, 1};
///     QkComplex64 coeffs[1] = {{1.0, 0.0}};
///     uint32_t boundaries[2] = {0, 2};
///     QfEdgeVertexOperator *op = qf_edge_op_new(num_terms, num_indices, coeffs,
///                                               left_indices, right_indices, boundaries);
///
///     uint32_t *right_out;
///     uint64_t right_len;
///
///     qf_edge_op_get_right_indices(op, &right_out, &right_len);
///
///     assert(right_len == 2);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_edge_op_get_right_indices(
    op: *const EdgeVertexOperator,
    right_indices_out: *mut *mut u32,
    right_indices_len: *mut u64,
) {
    let op = unsafe { const_ptr_as_ref(op) };
    unsafe { right_indices_out.write(op.right_indices.as_ptr().cast_mut()) };
    unsafe { right_indices_len.write(op.right_indices.len().try_into().unwrap()) };
}

/// @ingroup qf_edge_op
///
/// @brief Provides read-only access to the indices indicating the boundaries between operator terms.
///
/// @param op A pointer to the edge-vertex operator whose boundaries to access.
/// @param boundaries_out A pointer to the array of integers into which to write the boundaries.
/// @param boundaries_len A pointer to the integer into which to write the length of the output
///     array.
///
/// @rst
///
/// .. note::
///    This function borrows the operator's internal buffer rather than copying it. The returned
///    pointer stays valid only until the operator is modified or freed; do not free it yourself.
///
/// .. seealso::
///    The explanation of the internal data structure, :ref:`here <qf_edge_op-implementation>`.
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     uint64_t num_terms = 1;
///     uint64_t num_indices = 2;
///     uint32_t left_indices[2] = {0, 0};
///     uint32_t right_indices[2] = {0, 1};
///     QkComplex64 coeffs[1] = {{1.0, 0.0}};
///     uint32_t boundaries[2] = {0, 2};
///     QfEdgeVertexOperator *op = qf_edge_op_new(num_terms, num_indices, coeffs,
///                                               left_indices, right_indices, boundaries);
///
///     size_t *boundaries_out;
///     uint64_t boundaries_len;
///
///     qf_edge_op_get_boundaries(op, &boundaries_out, &boundaries_len);
///
///     assert(boundaries_len == 2);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_edge_op_get_boundaries(
    op: *const EdgeVertexOperator,
    boundaries_out: *mut *mut usize,
    boundaries_len: *mut u64,
) {
    let op = unsafe { const_ptr_as_ref(op) };
    unsafe { boundaries_out.write(op.boundaries.as_ptr().cast_mut()) };
    unsafe { boundaries_len.write(op.boundaries.len().try_into().unwrap()) };
}

/// @ingroup qf_edge_op
///
/// @brief Constructs the additive identity operator.
///
/// @return A pointer to the created operator.
///
/// @rst
///
/// Adding the operator that is constructed by this method to another one has no effect.
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     QfEdgeVertexOperator *zero = qf_edge_op_zero();
///
///     QfEdgeVertexOperator *op_plus_zero = qf_edge_op_add(op, zero);
///
///     assert(qf_edge_op_equal(op, op_plus_zero));
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_edge_op_zero() -> *mut EdgeVertexOperator {
    let op = EdgeVertexOperator::zero();
    Box::into_raw(Box::new(op))
}

/// @ingroup qf_edge_op
///
/// @brief Constructs the multiplicative identity operator.
///
/// @return A pointer to the created operator.
///
/// @rst
///
/// Composing the operator that is constructed by this method with another one has no effect.
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     QfEdgeVertexOperator *one = qf_edge_op_one();
///
///     QfEdgeVertexOperator *op_times_one = qf_edge_op_compose(op, one);
///
///     assert(qf_edge_op_equal(op, op_times_one));
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_edge_op_one() -> *mut EdgeVertexOperator {
    let op = EdgeVertexOperator::one();
    Box::into_raw(Box::new(op))
}

/// @ingroup qf_edge_op
///
/// @brief Checks whether an operator has its ``groups`` attribute set.
///
/// @param op A pointer to the edge-vertex operator to check.
///
/// @return Whether the operator tracks group indices.
///
/// @rst
///
/// .. seealso::
///    The explanation on :ref:`grouping_explanation`.
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     QfEdgeVertexOperator *op = ...;
///
///     bool has_groups = qf_edge_op_has_groups(op);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_edge_op_has_groups(op: *const EdgeVertexOperator) -> bool {
    let op = unsafe { const_ptr_as_ref(op) };
    op.has_groups()
}

/// @ingroup qf_edge_op
///
/// @brief Gets the number of groups from an operator.
///
/// @param op A pointer to the edge-vertex operator whose number of groups to get.
///
/// @return The number of group indices from the operator's ``groups`` attribute.
///
/// @rst
///
/// .. note::
///    The number of groups is evaluated lazily as the largest occurring group index plus 1.
///
/// .. seealso::
///    The explanation on :ref:`grouping_explanation`.
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     QfEdgeVertexOperator *op = ...;
///
///     uint32_t num_groups = qf_edge_op_num_groups(op);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_edge_op_num_groups(op: *const EdgeVertexOperator) -> u32 {
    let op = unsafe { const_ptr_as_ref(op) };
    op.num_groups().expect(
        "Expected groups to be present. It is the user's responsibility to check this via \
        qf_edge_op_has_groups before calling this function.",
    )
}

/// @ingroup qf_edge_op
///
/// @brief Gets the mean absolute coefficient magnitude of each group.
///
/// @param op A pointer to the edge-vertex operator whose group weights to compute.
/// @param weights_out A pointer to the array of doubles into which to write the weights. Must be
///     sized to :c:func:`qf_edge_op_num_groups`.
///
/// @rst
///
/// The ``i``-th entry is the sum of ``abs(coeff)`` over the terms in group ``i``, divided by the
/// number of terms in that group. This is the sampling weight of a randomized product formula
/// (e.g. qDRIFT) that draws whole groups rather than individual terms, and is computed in a single
/// pass over the operator rather than by reducing :c:func:`qf_edge_op_get_coeffs` and
/// :c:func:`qf_edge_op_get_groups` (one value per *ungrouped* term each) on the caller's side.
///
/// .. note::
///    A group index that no term carries weighs ``0.0``, which keeps it out of the sample.
///
/// .. seealso::
///    The explanation on :ref:`grouping_explanation`.
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     QfEdgeVertexOperator *op = ...;
///
///     uint32_t groups_in[4] = {0, 1, 0, 1};
///     qf_edge_op_set_groups(op, groups_in, 4);
///
///     double weights[2];
///     qf_edge_op_group_weights(op, weights);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_edge_op_group_weights(
    op: *const EdgeVertexOperator,
    weights_out: *mut f64,
) {
    let op = unsafe { const_ptr_as_ref(op) };
    let weights = op.group_weights().expect(
        "Expected groups to be present. It is the user's responsibility to check this via \
        qf_edge_op_has_groups before calling this function.",
    );
    for (i, weight) in weights.iter().enumerate() {
        unsafe { weights_out.add(i).write(*weight) };
    }
}

/// @ingroup qf_edge_op
///
/// @brief Gets the group indices for all operator terms.
///
/// @param op A pointer to the edge-vertex operator whose group indices to get.
/// @param groups_out A pointer to the integer array into which to write the group indices.
/// @param groups_len A pointer to the integer into which to write the length of the output array.
///
/// @rst
///
/// .. note::
///    This function borrows the operator's internal buffer rather than copying it. The returned
///    pointer stays valid only until the operator is modified or freed; do not free it yourself.
///
/// .. seealso::
///    The explanation on :ref:`grouping_explanation`.
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     QfEdgeVertexOperator *op = ...;
///     uint32_t *groups_out;
///     uint64_t groups_len;
///
///     qf_edge_op_get_groups(op, &groups_out, &groups_len);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_edge_op_get_groups(
    op: *const EdgeVertexOperator,
    groups_out: *mut *mut u32,
    groups_len: *mut u64,
) {
    let op = unsafe { const_ptr_as_ref(op) };
    let groups = &op.groups.as_ref().expect(
        "Expected groups to be present. It is the user's responsibility to check this via \
        qf_edge_op_has_groups before calling this function.",
    );
    unsafe { groups_out.write(groups.as_ptr().cast_mut()) };
    unsafe { groups_len.write(groups.len().try_into().unwrap()) };
}

/// @ingroup qf_edge_op
///
/// @brief Sets the ``groups`` attribute of the provided operator.
///
/// @param op A pointer to the edge-vertex operator whose ``groups`` attribute to write.
/// @param groups_in A pointer to the ``groups`` integer array to write into the operator.
/// @param groups_len The number of terms in the ``groups_in`` array.
///
/// @rst
///
/// .. seealso::
///    The explanation on :ref:`grouping_explanation`.
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     QfEdgeVertexOperator *op = ...;
///
///     uint32_t num_terms = 4;
///     uint32_t groups_in[4] = {0, 1, 0, 1};
///     qf_edge_op_set_groups(op, groups_in, num_terms);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_edge_op_set_groups(
    op: *mut EdgeVertexOperator,
    groups_in: *const u32,
    groups_len: u64,
) {
    let op = unsafe { mut_ptr_as_ref(op) };
    let groups_in = unsafe { const_ptr_as_ref(groups_in) };
    let mut groups = vec![0; groups_len as usize];
    groups.copy_from_slice(unsafe { slice_from_ptr(groups_in, groups_len as usize) });
    op.groups = Some(groups);
}

/// @ingroup qf_edge_op
///
/// @brief Deletes the ``groups`` attribute from the provided operator.
///
/// @param op A pointer to the edge-vertex operator whose ``groups`` attribute to delete.
///
/// @rst
///
/// .. seealso::
///    The explanation on :ref:`grouping_explanation`.
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     QfEdgeVertexOperator *op = ...;
///
///     qf_edge_op_del_groups(op);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_edge_op_del_groups(op: *mut EdgeVertexOperator) {
    let op = unsafe { mut_ptr_as_ref(op) };
    op.groups = None;
}

/// @ingroup qf_edge_op
///
/// @brief Splits this operator into a list of new operators based on its ``groups`` attribute.
///
/// @param op A pointer to the edge-vertex operator whose ``groups`` to split out.
/// @param group_indices A pointer to the array of group indices for which to build operators, in
///     the desired output order. May be ``NULL``, in which case every group is built, in index
///     order (equivalent to passing every index from ``0`` to
///     :c:func:`qf_edge_op_num_groups` ``- 1``).
/// @param num_indices The number of indices in the ``group_indices`` array. Ignored if
///     ``group_indices`` is ``NULL``.
/// @param group_ops_out A pointer to the array of :c:struct:`QfEdgeVertexOperator` into which to
///     write the operators for each requested group. Must be sized to ``num_indices`` when
///     ``group_indices`` is non-``NULL``, or to :c:func:`qf_edge_op_num_groups` when it is
///     ``NULL``.
///
/// @rst
///
/// A duplicate index in ``group_indices`` is written once per occurrence in ``group_ops_out``.
/// Requesting only a small number of groups out of a much larger total is significantly cheaper
/// than requesting all of them, since terms belonging to a group that is not requested are
/// skipped rather than appended anywhere.
///
/// .. seealso::
///    The explanation on :ref:`grouping_explanation`.
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     QfEdgeVertexOperator *op = ...;
///
///     uint32_t groups_in[4] = {0, 1, 0, 1};
///     qf_edge_op_set_groups(op, groups_in, 4);
///
///     // build every group, in index order
///     QfEdgeVertexOperator *group_ops[2];
///     qf_edge_op_split_out_groups(op, NULL, 0, group_ops);
///
///     // build only group 1
///     uint32_t group_indices[1] = {1};
///     QfEdgeVertexOperator *group_op[1];
///     qf_edge_op_split_out_groups(op, group_indices, 1, group_op);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_edge_op_split_out_groups(
    op: *const EdgeVertexOperator,
    group_indices: *const u32,
    num_indices: u64,
    group_ops_out: *mut *mut EdgeVertexOperator,
) {
    let op = unsafe { const_ptr_as_ref(op) };
    let group_indices = if group_indices.is_null() {
        None
    } else {
        Some(unsafe { slice_from_ptr(group_indices, num_indices as usize) })
    };
    let groups = op.split_out_groups(group_indices).expect(
        "Expected groups to be present. It is the user's responsibility to check this via \
        qf_edge_op_has_groups before calling this function.",
    );
    let cgroups: Vec<*mut EdgeVertexOperator> = groups
        .into_iter()
        .map(|g| Box::into_raw(Box::new(g)))
        .collect();
    for (i, ptr) in cgroups.iter().enumerate() {
        unsafe { group_ops_out.add(i).write(*ptr) };
    }
}

/// @ingroup qf_edge_op
///
/// @brief Adds a term to an existing operator.
///
/// @param op A pointer to the edge-vertex operator to be modified.
/// @param num_indices The length of both index arrays.
/// @param left_indices A pointer to an array of left-hand mode indices. The length of this array
///     should be ``num_indices``.
/// @param right_indices A pointer to an array of right-hand mode indices. The length of this array
///     should be ``num_indices``.
/// @param coeff A pointer to the complex coefficient.
///
/// @rst
///
/// Any of the pointer arguments can be ``NULL`` if and only if their corresponding length is zero.
///
/// .. caution::
///    This function resets the operator's ``groups`` attribute to ``NULL``.
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     QfEdgeVertexOperator *one = qf_edge_op_one();
///
///     QfEdgeVertexOperator *op = qf_edge_op_zero();
///     QkComplex64 coeff = {1.0, 0.0};
///
///     qf_edge_op_add_term(op, 0, NULL, NULL, &coeff);
///
///     assert(qf_edge_op_equal(op, one));
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_edge_op_add_term(
    op: *mut EdgeVertexOperator,
    num_indices: u64,
    left_indices: *const u32,
    right_indices: *const u32,
    coeff: *const Complex64,
) {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let op = unsafe { mut_ptr_as_ref(op) };
    let coeff = unsafe { const_ptr_as_ref(coeff) };

    let num_indices = num_indices as usize;

    op._append_term(
        *coeff,
        unsafe { slice_from_ptr(left_indices, num_indices) },
        unsafe { slice_from_ptr(right_indices, num_indices) },
    );

    op.groups = None;
}

/// @ingroup qf_edge_op
///
/// @brief Adds two operators together.
///
/// @param left A pointer to the left operator.
/// @param right A pointer to the right operator.
///
/// @return A pointer to the resulting operator.
///
/// @rst
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     QfEdgeVertexOperator *one = qf_edge_op_one();
///     QfEdgeVertexOperator *zero = qf_edge_op_zero();
///
///     QfEdgeVertexOperator *result = qf_edge_op_add(one, zero);
///
///     assert(qf_edge_op_equal(result, one));
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_edge_op_add(
    left: *const EdgeVertexOperator,
    right: *const EdgeVertexOperator,
) -> *mut EdgeVertexOperator {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let left = unsafe { const_ptr_as_ref(left) };
    let right = unsafe { const_ptr_as_ref(right) };

    let result = left.__add__(right);
    Box::into_raw(Box::new(result))
}

/// @ingroup qf_edge_op
///
/// @brief Multiplies an operator by a scalar.
///
/// @param op A pointer to the operator.
/// @param scalar A pointer to the scalar.
///
/// @return A pointer to the resulting operator.
///
/// @rst
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     QfEdgeVertexOperator *one = qf_edge_op_one();
///     QkComplex64 coeff = {2.0, 0.0};
///     QfEdgeVertexOperator *result = qf_edge_op_mul(one, &coeff);
///
///     QfEdgeVertexOperator *expected = qf_edge_op_zero();
///     qf_edge_op_add_term(expected, 0, NULL, NULL, &coeff);
///
///     assert(qf_edge_op_equal(result, expected));
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_edge_op_mul(
    op: *const EdgeVertexOperator,
    scalar: *const Complex64,
) -> *mut EdgeVertexOperator {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let op = unsafe { const_ptr_as_ref(op) };
    let scalar = unsafe { const_ptr_as_ref(scalar) };

    let result = op.__mul__(*scalar);
    Box::into_raw(Box::new(result))
}

/// @ingroup qf_edge_op
///
/// @brief Composes two operators with each other.
///
/// @param left A pointer to the left operator.
/// @param right A pointer to the right operator.
///
/// @return A pointer to the resulting operator.
///
/// @rst
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     QfEdgeVertexOperator *one = qf_edge_op_one();
///     QfEdgeVertexOperator *zero = qf_edge_op_zero();
///
///     QfEdgeVertexOperator *result = qf_edge_op_compose(one, zero);
///
///     assert(qf_edge_op_equal(result, zero));
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_edge_op_compose(
    left: *const EdgeVertexOperator,
    right: *const EdgeVertexOperator,
) -> *mut EdgeVertexOperator {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let left = unsafe { const_ptr_as_ref(left) };
    let right = unsafe { const_ptr_as_ref(right) };

    let result = left.__and__(right);
    Box::into_raw(Box::new(result))
}

/// @ingroup qf_edge_op
///
/// @brief Returns the Hermitian conjugate (or adjoint) of an operator.
///
/// This affects the terms and coefficients as follows:
///
/// - the generators in each term reverse their order
/// - the coefficients are complex conjugated
///
/// @param op A pointer to the operator.
///
/// @return A pointer to the created operator.
///
/// @rst
///
/// .. note::
///    Reversing the operator string is essential: the edge and vertex generators anticommute when
///    they share exactly one index, so ``BA != AB`` in general.
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     QfEdgeVertexOperator *op = qf_edge_op_zero();
///     QkComplex64 coeff = {0.0, 1.0};
///     qf_edge_op_add_term(op, 0, NULL, NULL, &coeff);
///
///     QfEdgeVertexOperator *adjoint = qf_edge_op_adjoint(op);
///
///     QfEdgeVertexOperator *expected = qf_edge_op_zero();
///     QkComplex64 coeff_adj = {0.0, -1.0};
///     qf_edge_op_add_term(expected, 0, NULL, NULL, &coeff_adj);
///
///     assert(qf_edge_op_equal(adjoint, expected));
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_edge_op_adjoint(
    op: *const EdgeVertexOperator,
) -> *mut EdgeVertexOperator {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let op = unsafe { const_ptr_as_ref(op) };

    let result = op.adjoint();
    Box::into_raw(Box::new(result))
}

/// @ingroup qf_edge_op
///
/// @brief Removes terms whose coefficient magnitude lies below the provided threshold.
///
/// @param op A pointer to the operator.
/// @param atol The absolute tolerance for coefficient truncation.
///
/// @rst
///
/// .. caution::
///    This functions truncates coefficients greedily! If the acted upon operator might contain
///    separate coefficients for duplicate terms consider calling :c:func:`qf_edge_op_simplify`
///    instead!
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     QfEdgeVertexOperator *op = qf_edge_op_zero();
///     QkComplex64 coeff = {1e-8, 0.0};
///     qf_edge_op_add_term(op, 0, NULL, NULL, &coeff);
///
///     qf_edge_op_ichop(op, 1e-6);
///
///     QfEdgeVertexOperator *expected = qf_edge_op_zero();
///
///     assert(qf_edge_op_equal(op, expected));
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_edge_op_ichop(op: *mut EdgeVertexOperator, atol: f64) {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let op = unsafe { mut_ptr_as_ref(op) };

    op.ichop(atol);
}

/// @ingroup qf_edge_op
///
/// @brief Returns an equivalent but simplified operator.
///
/// @param op A pointer to the edge-vertex operator to be simplified.
/// @param atol The absolute tolerance for coefficient truncation.
///
/// @return An equivalent but simplified operator.
///
/// @rst
/// The simplification process first sums all coefficients that belong to equal terms and then
/// only retains those whose total coefficient exceeds the specified tolerance (just like
/// :c:func:`qf_edge_op_ichop`).
///
/// When an operator has been arithmetically manipulated or constructed in a way that does not
/// guarantee unique terms, this method should be called before applying any method that
/// filters numerically small coefficients to avoid loss of information.
///
/// .. note::
///    This groups terms by their exact stored index arrays. Two terms that are mathematically
///    equal but stored with different generator orientations (recall ``E_kj = -E_jk``) are not
///    recognized as duplicates; call :c:func:`qf_edge_op_normal_ordered` first to bring them into
///    a common form.
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     QfEdgeVertexOperator *op = qf_edge_op_zero();
///     QkComplex64 coeff = {1e-5, 0.0};
///     for (int i = 0; i < 100; i++) {
///       qf_edge_op_add_term(op, 0, NULL, NULL, &coeff);
///     }
///
///     QfEdgeVertexOperator *canon = qf_edge_op_simplify(op, 1e-4);
///
///     assert(qf_edge_op_len(canon) == 1);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_edge_op_simplify(
    op: *const EdgeVertexOperator,
    atol: f64,
) -> *mut EdgeVertexOperator {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let op = unsafe { const_ptr_as_ref(op) };

    let result = op.simplify(atol);
    Box::into_raw(Box::new(result))
}

/// @ingroup qf_edge_op
///
/// @brief Returns an equivalent operator with normal ordered terms.
///
/// @param op A pointer to the operator.
/// @param ascending Whether the indices of each edge operator should ascend or descend.
/// @param reduce Whether to contract adjacent generators via the algebra's identities.
///
/// @return A pointer to the created operator.
///
/// @rst
///
/// The ``ascending`` flag fixes the orientation convention of the generalized edge operators:
/// because :math:`E_{kj} = -E_{jk}`, every edge operator has two representations, and this picks
/// :math:`j < k` (``true``) or :math:`j > k` (``false``), absorbing the sign into the coefficient.
/// Vertex operators :math:`V_j = E_{jj}` are unaffected.
///
/// The ``reduce`` flag additionally contracts adjacent generators, which is what makes the result
/// a genuine canonical form. The rules are :math:`V_j V_j = 1`, :math:`E_{jk} E_{jk} = 1`,
/// :math:`E_{jk} E_{kj} = -1`, and the *fusion* rule
/// :math:`E_{ab} E_{bc} = -i E_{ac}` for distinct :math:`a, b, c`.
///
/// .. seealso::
///    :c:func:`qf_transfer_op_normal_ordered`, which takes no ``ascending`` flag because the
///    transfer operators :math:`T_{jk}` and :math:`T_{kj}` are genuinely *different* operators
///    rather than two representations of one.
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     // `E(1,0)` normal-orders to `-E(0,1)` when ascending
///     QfEdgeVertexOperator *op = qf_edge_op_zero();
///     uint32_t left[1] = {1};
///     uint32_t right[1] = {0};
///     QkComplex64 coeff = {1.0, 0.0};
///     qf_edge_op_add_term(op, 1, left, right, &coeff);
///
///     QfEdgeVertexOperator *ordered = qf_edge_op_normal_ordered(op, true, true);
///
///     QfEdgeVertexOperator *expected = qf_edge_op_zero();
///     uint32_t left_exp[1] = {0};
///     uint32_t right_exp[1] = {1};
///     QkComplex64 coeff_exp = {-1.0, 0.0};
///     qf_edge_op_add_term(expected, 1, left_exp, right_exp, &coeff_exp);
///
///     assert(qf_edge_op_equal(ordered, expected));
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_edge_op_normal_ordered(
    op: *const EdgeVertexOperator,
    ascending: bool,
    reduce: bool,
) -> *mut EdgeVertexOperator {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let op = unsafe { const_ptr_as_ref(op) };

    let result = op.normal_ordered(ascending, reduce);
    Box::into_raw(Box::new(result))
}

/// @ingroup qf_edge_op
///
/// @brief Checks whether an operator is Hermitian.
///
/// @param op A pointer to the edge-vertex operator to be checked.
/// @param atol The absolute tolerance upto which coefficients are considered equal.
///
/// @return Whether the provided operator is Hermitian.
///
/// @rst
///
/// .. note::
///    This check is implemented using :c:func:`qf_edge_op_equiv` on the
///    :c:func:`qf_edge_op_normal_ordered` difference of ``op`` and its
///    :c:func:`qf_edge_op_adjoint` and :c:func:`qf_edge_op_zero`.
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     // `V(0) E(0,1)` is *not* Hermitian: the two generators share the index 0 and therefore
///     // anticommute.
///     QfEdgeVertexOperator *op = qf_edge_op_zero();
///     uint32_t left[2] = {0, 0};
///     uint32_t right[2] = {0, 1};
///     QkComplex64 coeff = {1.0, 0.0};
///     qf_edge_op_add_term(op, 2, left, right, &coeff);
///
///     assert(!qf_edge_op_is_hermitian(op, 1e-10));
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_edge_op_is_hermitian(op: *const EdgeVertexOperator, atol: f64) -> bool {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let op = unsafe { const_ptr_as_ref(op) };

    op.is_hermitian(atol)
}

/// @ingroup qf_edge_op
///
/// @brief Compare two operators for equality.
///
/// Equality in this context means an exact match of the internal data arrays.
///
/// @param left A pointer to the left operator.
/// @param right A pointer to the right operator.
///
/// @return Whether the two operators are equal.
///
/// @rst
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     QfEdgeVertexOperator *one = qf_edge_op_one();
///     QfEdgeVertexOperator *zero = qf_edge_op_zero();
///
///     assert(qf_edge_op_equal(one, one));
///     assert(!qf_edge_op_equal(one, zero));
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_edge_op_equal(
    left: *const EdgeVertexOperator,
    right: *const EdgeVertexOperator,
) -> bool {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let left = unsafe { const_ptr_as_ref(left) };
    let right = unsafe { const_ptr_as_ref(right) };

    left.eq(right)
}

/// @ingroup qf_edge_op
///
/// @brief Compare two operators for equivalence.
///
/// Equivalence in this context means approximate equality up to the specified absolute tolerance.
/// To be more precise, this method returns ``True``, when all the absolute values of the
/// coefficients in the difference ``other - self`` are below the specified threshold ``atol``.
///
/// @param left A pointer to the left operator.
/// @param right A pointer to the right operator.
/// @param atol The absolute tolerance for coefficient equivalence.
///
/// @return Whether the two operators are equivalent.
///
/// @rst
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     QfEdgeVertexOperator *zero = qf_edge_op_zero();
///
///     QfEdgeVertexOperator *op = qf_edge_op_zero();
///     QkComplex64 coeff = {1e-7, 0.0};
///     qf_edge_op_add_term(op, 0, NULL, NULL, &coeff);
///
///     assert(qf_edge_op_equiv(op, zero, 1e-6));
///     assert(!qf_edge_op_equiv(op, zero, 1e-8));
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_edge_op_equiv(
    left: *const EdgeVertexOperator,
    right: *const EdgeVertexOperator,
    atol: f64,
) -> bool {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let left = unsafe { const_ptr_as_ref(left) };
    let right = unsafe { const_ptr_as_ref(right) };

    left.equiv(right, atol)
}

/// @ingroup qf_edge_op
///
/// @brief Returns the length (or number of terms) of the provided operator.
///
/// @param op A pointer to the edge-vertex operator.
///
/// @return The length (or number of terms) of the operator.
///
/// @rst
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     QfEdgeVertexOperator *op = qf_edge_op_zero();
///     uint32_t left[2] = {0, 1};
///     uint32_t right[2] = {1, 2};
///     QkComplex64 coeff = {1.0, 0.0};
///     qf_edge_op_add_term(op, 2, left, right, &coeff);
///
///     assert(qf_edge_op_len(op) == 1);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_edge_op_len(op: *const EdgeVertexOperator) -> usize {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let op = unsafe { const_ptr_as_ref(op) };

    op.boundaries.len() - 1
}

/// @ingroup qf_edge_op
///
/// @brief Relabels the modes of the provided operator.
///
/// @param op A pointer to the edge-vertex operator.
/// @param num_modes The number of mode indices in the provided permutation list.
/// @param permutation The index permutation list.
///
/// @return An exit code.
/// * ``QfExitCode_Success`` upon success
/// * ``QfExitCode_DuplicateIndexError`` if duplicate indices were found in the permutation
/// * ``QfExitCode_IndexError`` for any other index errors, such as invalid indices.
///
/// @rst
///
/// .. note::
///    Both index arrays are relabelled. Unlike most operations, this preserves the operator's
///    ``groups`` attribute, since relabelling permutes mode indices without reordering, splitting
///    or merging terms.
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     QfEdgeVertexOperator *op = qf_edge_op_zero();
///     uint32_t left[2] = {0, 2};
///     uint32_t right[2] = {1, 3};
///     QkComplex64 coeff = {1.0, 0.0};
///     qf_edge_op_add_term(op, 2, left, right, &coeff);
///
///     uint32_t permutation[4] = {3, 2, 1, 0};
///
///     QfExitCode exit = qf_edge_op_relabel_modes(op, 4, permutation);
///
///     assert(exit == QfExitCode_Success);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_edge_op_relabel_modes(
    op: *mut EdgeVertexOperator,
    num_modes: u64,
    permutation: *const u32,
) -> ExitCode {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let op = unsafe { mut_ptr_as_ref(op) };

    let permutation = unsafe { slice_from_ptr(permutation, num_modes as usize).to_vec() };

    let relabeled_op = match op.relabel_modes(permutation) {
        Ok(relabeled) => relabeled,
        Err(e) => {
            return match e {
                CoherenceError::DuplicateIndices => ExitCode::DuplicateIndexError,
                _ => ExitCode::IndexError,
            };
        }
    };

    *op = relabeled_op;
    ExitCode::Success
}
