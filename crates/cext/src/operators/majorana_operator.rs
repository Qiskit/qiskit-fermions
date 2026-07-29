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
use qiskit_fermions_core::operators::majorana_operator::MajoranaOperator;
use qiskit_fermions_core::operators::{CoherenceError, OperatorMacro, OperatorTrait};

/// @ingroup qf_maj_op
///
/// @brief Constructs a new operator.
///
/// @param num_terms The number of terms in the operator.
/// @param num_modes The number of modes summed over all terms.
/// @param coeffs A pointer to an array of term coefficients. The length of this array should be
///     ``num_terms``.
/// @param modes A pointer to an array of modes over all terms. The length of this array should
///     be ``num_modes``.
/// @param boundaries A pointer to an array of the boundaries between terms. The length of this
///     array should be ``num_terms + 1``.
///
/// @rst
///
/// Any of the pointer arguments may be ``NULL`` if and only if their corresponding length is zero.
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     uint64_t num_terms = 3;
///     uint64_t num_modes = 4;
///     uint32_t modes[4] = {0, 1, 2, 3};
///     QkComplex64 coeffs[3] = {{1.0, 0.0}, {-1.0, 0.0}, {0.0, -1.0}};
///     uint32_t boundaries[4] = {0, 0, 2, 4};
///     QfMajoranaOperator *op = qf_maj_op_new(num_terms, num_modes, coeffs,
///                                            modes, boundaries);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_maj_op_new(
    num_terms: u64,
    num_modes: u64,
    coeffs: *const Complex64,
    modes: *const u32,
    boundaries: *const u32,
) -> *mut MajoranaOperator {
    let num_terms = num_terms as usize;
    let num_modes = num_modes as usize;

    let op = MajoranaOperator {
        coeffs: unsafe { slice_from_ptr(coeffs, num_terms).to_vec() },
        modes: unsafe { slice_from_ptr(modes, num_modes).to_vec() },
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

/// @ingroup qf_maj_op
///
/// @brief Frees an existing operator.
///
/// @param op A pointer to the Majorana operator to be freed.
///
/// @rst
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     QfMajoranaOperator *op = qf_maj_op_one();
///     qf_maj_op_free(op);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_maj_op_free(op: *mut MajoranaOperator) {
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

/// @ingroup qf_maj_op
///
/// @brief Provides read-only access to the operator's coefficients.
///
/// @param op A pointer to the majorana operator whose coefficients to access.
/// @param coeffs_out A pointer to the array of complex values into which to write the coefficients.
/// @param coeffs_len A pointer to the integer into which to write the length of the output array.
///
/// @rst
///
/// .. note::
///    This function returns a **copy** of the internal data.
///
/// .. seealso::
///    The explanation of the internal data structure, :ref:`here <qf_maj_op-implementation>`.
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     uint64_t num_terms = 2;
///     uint64_t num_modes = 0;
///     QkComplex64 coeffs[2] = {{1.0, 0.0}, {0.0, -1.0}};
///     uint32_t boundaries[3] = {0, 0, 0};
///     QfMajoranaOperator *op =
///         qf_maj_op_new(num_terms, num_modes, coeffs, NULL, boundaries);
///
///     QkComplex64 *coeffs_out;
///     uint64_t *coeffs_len;
///
///     qf_maj_op_get_coeffs(op, &coeffs_out, &coeffs_len);
///
///     assert(coeffs_len == 2);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_maj_op_get_coeffs(
    op: *const MajoranaOperator,
    coeffs_out: *mut *mut Complex64,
    coeffs_len: *mut u64,
) {
    let op = unsafe { const_ptr_as_ref(op) };
    unsafe { coeffs_out.write(op.coeffs.as_ptr().cast_mut()) };
    unsafe { coeffs_len.write(op.coeffs.len().try_into().unwrap()) };
}

/// @ingroup qf_maj_op
///
/// @brief Provides read-only access to the operator's acted-upon mode indices.
///
/// @param op A pointer to the majorana operator whose modes to access.
/// @param modes_out A pointer to the array of boolean values into which to write the modes.
/// @param modes_len A pointer to the integer into which to write the length of the output array.
///
/// @rst
///
/// .. note::
///    This function returns a **copy** of the internal data.
///
/// .. seealso::
///    The explanation of the internal data structure, :ref:`here <qf_maj_op-implementation>`.
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     uint64_t num_terms = 2;
///     bool actions[2] = {true, false};
///     uint32_t modes[2] = {0, 1};
///     QkComplex64 coeffs[2] = {{1.0, 0.0}, {0.0, -1.0}};
///     uint32_t boundaries[3] = {0, 0, 2};
///     QfMajoranaOperator *op =
///         qf_maj_op_new(num_terms, num_actions, coeffs, modes, boundaries);
///
///     QkComplex64 *modes_out;
///     uint64_t *modes_len;
///
///     qf_maj_op_get_modes(op, &modes_out, &modes_len);
///
///     assert(modes_len == 2);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_maj_op_get_modes(
    op: *const MajoranaOperator,
    modes_out: *mut *mut u32,
    modes_len: *mut u64,
) {
    let op = unsafe { const_ptr_as_ref(op) };
    unsafe { modes_out.write(op.modes.as_ptr().cast_mut()) };
    unsafe { modes_len.write(op.modes.len().try_into().unwrap()) };
}

/// @ingroup qf_maj_op
///
/// @brief Provides read-only access to the indices indicating the boundaries between operator terms.
///
/// @param op A pointer to the majorana operator whose boundaries to access.
/// @param boundaries_out A pointer to the array of boolean values into which to write the boundaries.
/// @param boundaries_len A pointer to the integer into which to write the length of the output array.
///
/// @rst
///
/// .. note::
///    This function returns a **copy** of the internal data.
///
/// .. seealso::
///    The explanation of the internal data structure, :ref:`here <qf_maj_op-implementation>`.
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     uint64_t num_terms = 2;
///     uint64_t num_modes = 2;
///     uint32_t modes[2] = {0, 1};
///     QkComplex64 coeffs[2] = {{1.0, 0.0}, {0.0, -1.0}};
///     uint32_t boundaries[3] = {0, 0, 2};
///     QfMajoranaOperator *op =
///         qf_maj_op_new(num_terms, num_modes, coeffs, modes, boundaries);
///
///     QkComplex64 *boundaries_out;
///     uint64_t *boundaries_len;
///
///     qf_maj_op_get_boundaries(op, &boundaries_out, &boundaries_len);
///
///     assert(boundaries_len == 3);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_maj_op_get_boundaries(
    op: *const MajoranaOperator,
    boundaries_out: *mut *mut usize,
    boundaries_len: *mut u64,
) {
    let op = unsafe { const_ptr_as_ref(op) };
    unsafe { boundaries_out.write(op.boundaries.as_ptr().cast_mut()) };
    unsafe { boundaries_len.write(op.boundaries.len().try_into().unwrap()) };
}

/// @ingroup qf_maj_op
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
///     QfMajoranaOperator *zero = qf_maj_op_zero();
///
///     QfMajoranaOperator *op_plus_zero = qf_maj_op_add(op, zero);
///
///     assert(qf_maj_op_equal(op, op_plus_zero));
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_maj_op_zero() -> *mut MajoranaOperator {
    let op = MajoranaOperator::zero();
    Box::into_raw(Box::new(op))
}

/// @ingroup qf_maj_op
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
///     QfMajoranaOperator *one = qf_maj_op_one();
///
///     QfMajoranaOperator *op_times_one = qf_maj_op_compose(op, one);
///
///     assert(qf_maj_op_equal(op, op_times_one));
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_maj_op_one() -> *mut MajoranaOperator {
    let op = MajoranaOperator::one();
    Box::into_raw(Box::new(op))
}

/// @ingroup qf_maj_op
///
/// @brief Checks whether this operator has a ``groups`` attribute that is not empty.
///
/// @param op A pointer to the majorana operator to be checked.
///
/// @return Whether the provided operator has a non-empty ``groups`` attribute.
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
///     QfMajoranaOperator *op = ...;
///
///     bool has_groups = qf_maj_op_has_groups(op);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_maj_op_has_groups(op: *const MajoranaOperator) -> bool {
    let op = unsafe { const_ptr_as_ref(op) };
    op.groups.is_some()
}

/// @ingroup qf_maj_op
///
/// @brief Gets the number of groups from an operator.
///
/// @param op A pointer to the majorana operator whose number of groups to get.
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
///     QfMajoranaOperator *op = ...;
///
///     uint32_t num_groups = qf_maj_op_num_groups(op);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_maj_op_num_groups(op: *const MajoranaOperator) -> u32 {
    let op = unsafe { const_ptr_as_ref(op) };
    op.num_groups().expect(
        "Expected groups to be present. It is the user's responsibility to check this via \
        qf_maj_op_has_groups before calling this function.",
    )
}

/// @ingroup qf_maj_op
///
/// @brief Gets the group indices for all operator terms.
///
/// @param op A pointer to the majorana operator whose group indices to get.
/// @param groups_out A pointer to the integer array into which to write the group indices.
/// @param groups_len A pointer to the integer into which to write the length of the output array.
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
///     QfMajoranaOperator *op = ...;
///     uint32_t *groups_out;
///     uint32_t groups_len;
///
///     qf_maj_op_get_groups(op, &groups_out, &groups_len);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_maj_op_get_groups(
    op: *const MajoranaOperator,
    groups_out: *mut *mut u32,
    groups_len: *mut u64,
) {
    let op = unsafe { const_ptr_as_ref(op) };
    let groups = &op.groups.as_ref().expect(
        "Expected groups to be present. It is the user's responsibility to check this via \
        qf_maj_op_has_groups before calling this function.",
    );
    unsafe { groups_out.write(groups.as_ptr().cast_mut()) };
    unsafe { groups_len.write(groups.len().try_into().unwrap()) };
}

/// @ingroup qf_maj_op
///
/// @brief Sets the ``groups`` attribute of the provided operator.
///
/// @param op A pointer to the majorana operator whose ``groups`` attribute to write.
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
///     QfMajoranaOperator *op = ...;
///
///     uint32_t num_terms = 4;
///     uint32_t groups_in[4] = {0, 1, 0, 1};
///     qf_maj_op_set_groups(op, groups_in, num_terms);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_maj_op_set_groups(
    op: *mut MajoranaOperator,
    groups_in: *const u32,
    groups_len: u64,
) {
    let op = unsafe { mut_ptr_as_ref(op) };
    let groups_in = unsafe { const_ptr_as_ref(groups_in) };
    let mut groups = vec![0; groups_len as usize];
    groups.copy_from_slice(unsafe { slice_from_ptr(groups_in, groups_len as usize) });
    op.groups = Some(groups);
}

/// @ingroup qf_maj_op
///
/// @brief Deletes the ``groups`` attribute from the provided operator.
///
/// @param op A pointer to the majorana operator whose ``groups`` attribute to delete.
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
///     QfMajoranaOperator *op = ...;
///
///     qf_maj_op_del_groups(op);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_maj_op_del_groups(op: *mut MajoranaOperator) {
    let op = unsafe { mut_ptr_as_ref(op) };
    op.groups = None;
}

/// @ingroup qf_maj_op
///
/// @brief Splits this operator into a list of new operators based on its ``groups`` attribute.
///
/// @param op A pointer to the majorana operator whose ``groups`` to split out.
/// @param group_indices A pointer to the array of group indices for which to build operators, in
///     the desired output order. May be ``NULL``, in which case every group is built, in index
///     order (equivalent to passing every index from ``0`` to
///     :c:func:`qf_maj_op_num_groups` ``- 1``).
/// @param num_indices The number of indices in the ``group_indices`` array. Ignored if
///     ``group_indices`` is ``NULL``.
/// @param group_ops_out A pointer to the array of :c:struct:`QfMajoranaOperator` into which to
///     write the operators for each requested group. Must be sized to ``num_indices`` when
///     ``group_indices`` is non-``NULL``, or to :c:func:`qf_maj_op_num_groups` when it is
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
///     uint64_t num_terms = 4;
///     uint64_t num_modes = 8;
///     uint32_t modes[8] = {0, 1, 2, 3, 1, 0, 3, 2};
///     QkComplex64 coeffs[4] = {{1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}};
///     uint32_t boundaries[5] = {0, 2, 4, 6, 8};
///     QfMajoranaOperator *op = qf_maj_op_new(num_terms, num_modes, coeffs, modes, boundaries);
///
///     uint32_t groups_in[4] = {0, 1, 0, 1};
///     qf_maj_op_set_groups(op, groups_in, num_terms);
///
///     // build every group, in index order
///     QfMajoranaOperator *group_ops[2];
///     qf_maj_op_split_out_groups(op, NULL, 0, group_ops);
///
///     // build only group 1
///     uint32_t group_indices[1] = {1};
///     QfMajoranaOperator *group_op[1];
///     qf_maj_op_split_out_groups(op, group_indices, 1, group_op);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_maj_op_split_out_groups(
    op: *const MajoranaOperator,
    group_indices: *const u32,
    num_indices: u64,
    group_ops_out: *mut *mut MajoranaOperator,
) {
    let op = unsafe { const_ptr_as_ref(op) };
    let group_indices = if group_indices.is_null() {
        None
    } else {
        Some(unsafe { slice_from_ptr(group_indices, num_indices as usize) })
    };
    let groups = op.split_out_groups(group_indices).expect(
        "Expected groups to be present. It is the user's responsibility to check this via \
        qf_maj_op_has_groups before calling this function.",
    );
    let cgroups: Vec<*mut MajoranaOperator> = groups
        .into_iter()
        .map(|g| Box::into_raw(Box::new(g)))
        .collect();
    for (i, ptr) in cgroups.iter().enumerate() {
        unsafe { group_ops_out.add(i).write(*ptr) };
    }
}

/// @ingroup qf_maj_op
///
/// @brief Adds a term to an existing operator.
///
/// @param op A pointer to the Majorana operator to be modified.
/// @param num_modes The length of the modes array.
/// @param modes A pointer to an array of mode indices. The length of this array should be
///     ``num_modes``.
/// @param coeff A pointer to the complex coefficient.
///
/// @rst
///
/// Any of the pointer arguments may be ``NULL`` if and only if their corresponding length is zero.
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
///     QfMajoranaOperator *one = qf_maj_op_one();
///
///     QfMajoranaOperator *op = qf_maj_op_zero();
///     uint32_t modes[0] = {};
///     QkComplex64 coeff = {1.0, 0.0};
///
///     qf_maj_op_add_term(op, 0, modes, &coeff);
///
///     assert(qf_maj_op_equal(op, one));
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_maj_op_add_term(
    op: *mut MajoranaOperator,
    num_modes: u64,
    modes: *const u32,
    coeff: *const Complex64,
) {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let op = unsafe { mut_ptr_as_ref(op) };
    let coeff = unsafe { const_ptr_as_ref(coeff) };

    let num_modes = num_modes as usize;

    op.coeffs.push(*coeff);
    op.modes
        .extend_from_slice(unsafe { slice_from_ptr(modes, num_modes) });
    op.boundaries.push(op.modes.len());

    op.groups = None;
}

/// @ingroup qf_maj_op
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
///     QfMajoranaOperator *one = qf_maj_op_one();
///     QfMajoranaOperator *zero = qf_maj_op_zero();
///
///     QfMajoranaOperator *result = qf_maj_op_add(one, zero);
///
///     assert(qf_maj_op_equal(result, one));
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_maj_op_add(
    left: *const MajoranaOperator,
    right: *const MajoranaOperator,
) -> *mut MajoranaOperator {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let left = unsafe { const_ptr_as_ref(left) };
    let right = unsafe { const_ptr_as_ref(right) };

    let result = left.__add__(right);
    Box::into_raw(Box::new(result))
}

/// @ingroup qf_maj_op
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
///     QfMajoranaOperator *one = qf_maj_op_one();
///     QkComplex64 coeff = {2.0, 0.0};
///     QfMajoranaOperator *result = qf_maj_op_mul(one, &coeff);
///
///     QfMajoranaOperator *expected = qf_maj_op_zero();
///     uint32_t modes[0] = {};
///     qf_maj_op_add_term(expected, 0, modes, &coeff);
///
///     assert(qf_maj_op_equal(result, expected));
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_maj_op_mul(
    op: *const MajoranaOperator,
    scalar: *const Complex64,
) -> *mut MajoranaOperator {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let op = unsafe { const_ptr_as_ref(op) };
    let scalar = unsafe { const_ptr_as_ref(scalar) };

    let result = op.__mul__(*scalar);
    Box::into_raw(Box::new(result))
}

/// @ingroup qf_maj_op
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
///     QfMajoranaOperator *one = qf_maj_op_one();
///     QfMajoranaOperator *zero = qf_maj_op_zero();
///
///     QfMajoranaOperator *result = qf_maj_op_compose(one, zero);
///
///     assert(qf_maj_op_equal(result, zero));
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_maj_op_compose(
    left: *const MajoranaOperator,
    right: *const MajoranaOperator,
) -> *mut MajoranaOperator {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let left = unsafe { const_ptr_as_ref(left) };
    let right = unsafe { const_ptr_as_ref(right) };

    let result = left.__and__(right);
    Box::into_raw(Box::new(result))
}

/// @ingroup qf_maj_op
///
/// @brief Returns the Hermitian conjugate (or adjoint) of an operator.
///
/// This affects the terms and coefficients as follows:
///
/// - the actions in each term reverse their order
/// - the coefficients are complex conjugated
///
/// @param op A pointer to the operator.
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
///     QfMajoranaOperator *op = qf_maj_op_zero();
///     uint32_t modes[0] = {};
///     QkComplex64 coeff = {0.0, 1.0};
///     qf_maj_op_add_term(op, 0, modes, &coeff);
///
///     QfMajoranaOperator *adjoint = qf_maj_op_adjoint(op);
///
///     QfMajoranaOperator *expected = qf_maj_op_zero();
///     QkComplex64 coeff_adj = {0.0, -1.0};
///     qf_maj_op_add_term(expected, 0, modes, &coeff_adj);
///
///     assert(qf_maj_op_equal(adjoint, expected));
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_maj_op_adjoint(op: *const MajoranaOperator) -> *mut MajoranaOperator {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let op = unsafe { const_ptr_as_ref(op) };

    let result = op.adjoint();
    Box::into_raw(Box::new(result))
}

/// @ingroup qf_maj_op
///
/// @brief Removes terms whose coefficient magnitude lies below the provided threshold.
///
/// @param op A pointer to the operator.
/// @param atol The absolute tolerance for coefficient truncation.
///
/// @rst
///
/// .. caution::
///    This functions truncates coefficients greedily! If the acted upon operator may contain
///    separate coefficients for duplicate terms consider calling :c:func:`qf_maj_op_simplify`
///    instead!
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     QfMajoranaOperator *op = qf_maj_op_zero();
///     uint32_t modes[0] = {};
///     QkComplex64 coeff = {1e-8};
///     qf_maj_op_add_term(op, 0, modes, &coeff);
///
///     qf_maj_op_ichop(op, 1e-6);
///
///     QfMajoranaOperator *expected = qf_maj_op_zero();
///
///     assert(qf_maj_op_equal(op, expected));
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_maj_op_ichop(op: *mut MajoranaOperator, atol: f64) {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let op = unsafe { mut_ptr_as_ref(op) };

    op.ichop(atol);
}

/// @ingroup qf_maj_op
///
/// @brief Returns an equivalent but simplified operator.
///
/// @param op A pointer to the Majorana operator to be simplified.
/// @param atol The absolute tolerance for coefficient truncation.
///
/// @return An equivalent but simplified operator.
///
/// @rst
/// The simplification process first sums all coefficients that belong to equal terms and then
/// only retains those whose total coefficient exceeds the specified tolerance (just like
/// :c:func:`qf_maj_op_ichop`).
///
/// When an operator has been arithmetically manipulated or constructed in a way that does not
/// guarantee unique terms, this method should be called before applying any method that
/// filters numerically small coefficients to avoid loss of information. See the example below
/// which showcases how :c:func:`qf_maj_op_ichop` can truncate terms that sum to a total
/// coefficient magnitude which should not be truncated:
///
/// .. code-block:: c
///     :linenos:
///
///     uint64_t num_terms = 100000;
///     uint64_t num_modes = 0;
///     uint32_t modes[0] = {};
///     QkComplex64 coeffs[100000];
///     uint32_t boundaries[100001];
///     for (int i = 0; i < 100000; i++) {
///       coeffs[i].re = 1e-5;
///       coeffs[i].im = 0.0;
///       boundaries[i] = 0;
///     }
///     boundaries[100000] = 0;
///     QfMajoranaOperator *op =
///         qf_maj_op_new(num_terms, num_modes, coeffs, modes, boundaries);
///
///     QfMajoranaOperator *canon = qf_maj_op_simplify(op, 1e-4);
///
///     QfMajoranaOperator *one = qf_maj_op_one();
///     bool canon_is_equal = qf_maj_op_equiv(canon, one, 1e-6);
///
///     qf_maj_op_ichop(op, 1e-4);
///
///     QfMajoranaOperator *zero = qf_maj_op_zero();
///     bool ichop_is_equal = qf_maj_op_equiv(op, zero, 1e-6);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_maj_op_simplify(
    op: *const MajoranaOperator,
    atol: f64,
) -> *mut MajoranaOperator {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let op = unsafe { const_ptr_as_ref(op) };

    let result = op.simplify(atol);
    Box::into_raw(Box::new(result))
}

/// @ingroup qf_maj_op
///
/// @brief Returns an equivalent operator with normal ordered terms.
///
/// The normal order of an operator term is defined such that all actions are ordered
/// lexicographically. Whether they ascend or descend depends on the value of the ``ascending``
/// parameter.
///
/// @param op A pointer to the operator.
/// @param ascending Whether indices should ascend or descend.
/// @param reduce Whether to reduce each term to its minimal form by removing actions that square
/// to the identity.
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
///     QfMajoranaOperator *op = qf_maj_op_zero();
///     uint32_t modes[4] = {0, 2, 1, 3};
///     QkComplex64 coeff = {1.0, 0.0};
///     qf_maj_op_add_term(op, 4, modes, &coeff);
///
///     QfMajoranaOperator *normal_ordered = qf_maj_op_normal_ordered(op, false, true);
///
///     QkComplex64 coeff_minus = {-1.0, 0.0};
///     QfMajoranaOperator *expected = qf_maj_op_zero();
///     uint32_t modes_exp[4] = {3, 2, 1, 0};
///     qf_maj_op_add_term(expected, 4, modes_exp, &coeff_minus);
///
///     assert(qf_maj_op_equal(normal_ordered, expected));
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_maj_op_normal_ordered(
    op: *const MajoranaOperator,
    ascending: bool,
    reduce: bool,
) -> *mut MajoranaOperator {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let op = unsafe { const_ptr_as_ref(op) };

    let result = op.normal_ordered(ascending, reduce);
    Box::into_raw(Box::new(result))
}

/// @ingroup qf_maj_op
///
/// @brief Checks whether an operator is Hermitian.
///
/// @param op A pointer to the Majorana operator to be checked.
/// @param atol The absolute tolerance upto which coefficients are considered equal.
///
/// @return Whether the provided operator is Hermitian.
///
/// @rst
///
/// .. note::
///    This check is implemented using :c:func:`qf_maj_op_equiv` on the
///    :c:func:`qf_maj_op_normal_ordered` difference of ``op`` and its
///    :c:func:`qf_maj_op_adjoint` and :c:func:`qf_maj_op_zero`.
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     QfMajoranaOperator *op = qf_maj_op_zero();
///     uint32_t modes1[2] = {0, 1};
///     QkComplex64 coeff1 = {0.0, 1.00001};
///     qf_maj_op_add_term(op, 2, modes1, &coeff1);
///     uint32_t modes2[2] = {0, 1};
///     QkComplex64 coeff2 = {0.0, -1};
///     qf_maj_op_add_term(op, 2, modes2, &coeff2);
///
///     assert(qf_maj_op_is_hermitian(op, 1e-4));
///     assert(!qf_maj_op_is_hermitian(op, 1e-8));
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_maj_op_is_hermitian(op: *const MajoranaOperator, atol: f64) -> bool {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let op = unsafe { const_ptr_as_ref(op) };

    op.is_hermitian(atol)
}

/// @ingroup qf_maj_op
///
/// @brief Checks the maximum rank of an operator.
///
/// @param op A pointer to the Majorana operator to be checked.
///
/// @return The maximum rank of the operator.
///
/// @rst
///
/// .. note::
///    The length of the longest term can depend on the operator's form which means that (for
///    example) operator simplification or normal-ordering can result in a different maximum rank.
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     QfMajoranaOperator *op = qf_maj_op_zero();
///     uint32_t modes[4] = {0, 1, 2, 3};
///     QkComplex64 coeff = {1.0, 0.0};
///     qf_maj_op_add_term(op, 4, modes, &coeff);
///
///     assert(qf_maj_op_max_rank(op), 4);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_maj_op_max_rank(op: *const MajoranaOperator) -> u32 {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let op = unsafe { const_ptr_as_ref(op) };

    op.max_rank()
}

/// @ingroup qf_maj_op
///
/// @brief Checks whether an operator is even.
///
/// @param op A pointer to the Majorana operator to be checked.
///
/// @return Whether the provided operator is even.
///
/// @rst
///
/// .. note::
///    An operator is considered even when all of its terms contain an even number of actions.
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     QfMajoranaOperator *op = qf_maj_op_zero();
///     QkComplex64 coeff = {1.0, 0.0};
///     uint32_t modes1[2] = {0, 1};
///     qf_maj_op_add_term(op, 2, modes1, &coeff);
///
///     assert(qf_maj_op_is_even(op));
///
///     uint32_t modes2[1] = {2};
///     qf_maj_op_add_term(op, 2, modes2, &coeff);
///
///     assert(!qf_maj_op_is_even(op));
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_maj_op_is_even(op: *const MajoranaOperator) -> bool {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let op = unsafe { const_ptr_as_ref(op) };

    op.is_even()
}

/// @ingroup qf_maj_op
///
/// @brief Compare two operators for equality.
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
///     QfMajoranaOperator *one = qf_maj_op_one();
///     QfMajoranaOperator *zero = qf_maj_op_zero();
///
///     assert(qf_maj_op_equal(one, one));
///     assert(!qf_maj_op_equal(one, zero));
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_maj_op_equal(
    left: *const MajoranaOperator,
    right: *const MajoranaOperator,
) -> bool {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let left = unsafe { const_ptr_as_ref(left) };
    let right = unsafe { const_ptr_as_ref(right) };

    left.eq(right)
}

/// @ingroup qf_maj_op
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
///     QfMajoranaOperator *zero = qf_maj_op_zero();
///
///     QfMajoranaOperator *op = qf_maj_op_zero();
///     uint32_t modes[0] = {};
///     QkComplex64 coeff = {1e-7, 0.0};
///     qf_maj_op_add_term(op, 0, modes, &coeff);
///
///     assert(qf_maj_op_equiv(op, zero, 1e-6));
///     assert(!qf_maj_op_equiv(op, zero, 1e-8));
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_maj_op_equiv(
    left: *const MajoranaOperator,
    right: *const MajoranaOperator,
    atol: f64,
) -> bool {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let left = unsafe { const_ptr_as_ref(left) };
    let right = unsafe { const_ptr_as_ref(right) };

    left.equiv(right, atol)
}

/// @ingroup qf_maj_op
///
/// @brief Returns the length (or number of terms) of the provided operator.
///
/// @param op A pointer to the Majorana operator.
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
///     QfMajoranaOperator *op = qf_maj_op_zero();
///     uint32_t modes[4] = {0, 1, 2, 3};
///     QkComplex64 coeff = {1.0, 0.0};
///     qf_maj_op_add_term(op, 4, modes, &coeff);
///
///     assert(qf_maj_op_len(op) == 1);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_maj_op_len(op: *const MajoranaOperator) -> usize {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let op = unsafe { const_ptr_as_ref(op) };

    op.boundaries.len() - 1
}

/// @ingroup qf_maj_op
///
/// @brief Relabels the modes of the provided operator.
///
/// @param op A pointer to the Majorana operator.
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
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     QfMajoranaOperator *op = qf_maj_op_zero();
///     uint32_t modes[4] = {0, 1, 2, 3};
///     QkComplex64 coeff = {1.0, 0.0};
///     qf_maj_op_add_term(op, 4, modes, &coeff);
///
///     uint32_t permutation[4] = {3, 2, 1, 0};
///
///     QfExitCode exit = qf_maj_op_relabel_modes(op, 4, permutation);
///
///     assert(exit == QfExitCode_Success);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_maj_op_relabel_modes(
    op: *mut MajoranaOperator,
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
