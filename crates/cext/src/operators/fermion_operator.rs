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
use qiskit_fermions_core::operators::fermion_operator::FermionOperator;
use qiskit_fermions_core::operators::{CoherenceError, OperatorMacro, OperatorTrait};

/// @ingroup qf_ferm_op
///
/// @brief Constructs a new operator.
///
/// @param num_terms The number of terms in the operator.
/// @param num_actions The number of actions summed over all terms.
/// @param coeffs A pointer to an array of term coefficients. The length of this array should be
///     ``num_terms``.
/// @param actions A pointer to an array of actions over all terms. The length of this array should
///     be ``num_actions``.
/// @param modes A pointer to an array of action modes over all terms. The length of this array
///     should be ``num_actions``.
/// @param boundaries A pointer to an array of the boundaries between terms. The length of this
///     array should be ``num_terms + 1``.
///
/// @rst
///
/// Any of the pointer arguments can be ``NULL`` if and only if their corresponding length is zero.
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     uint64_t num_terms = 3;
///     uint64_t num_actions = 4;
///     bool actions[4] = {true, false, true, false};
///     uint32_t modes[4] = {0, 1, 2, 3};
///     QkComplex64 coeffs[3] = {{1.0, 0.0}, {-1.0, 0.0}, {0.0, -1.0}};
///     uint32_t boundaries[4] = {0, 0, 2, 4};
///     QfFermionOperator *op = qf_ferm_op_new(num_terms, num_actions, coeffs,
///                                            actions, modes, boundaries);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ferm_op_new(
    num_terms: u64,
    num_actions: u64,
    coeffs: *const Complex64,
    actions: *const bool,
    modes: *const u32,
    boundaries: *const u32,
) -> *mut FermionOperator {
    let num_terms = num_terms as usize;
    let num_actions = num_actions as usize;

    let op = FermionOperator {
        coeffs: unsafe { slice_from_ptr(coeffs, num_terms).to_vec() },
        actions: unsafe { slice_from_ptr(actions, num_actions).to_vec() },
        modes: unsafe { slice_from_ptr(modes, num_actions).to_vec() },
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

/// @ingroup qf_ferm_op
///
/// @brief Frees an existing operator.
///
/// @param op A pointer to the fermionic operator to be freed.
///
/// @rst
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     QfFermionOperator *op = qf_ferm_op_one();
///     qf_ferm_op_free(op);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ferm_op_free(op: *mut FermionOperator) {
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

/// @ingroup qf_ferm_op
///
/// @brief Provides read-only access to the operator's coefficients.
///
/// @param op A pointer to the fermionic operator whose coefficients to access.
/// @param coeffs_out A pointer to the array of complex values into which to write the coefficients.
/// @param coeffs_len A pointer to the integer into which to write the length of the output array.
///
/// @rst
///
/// .. note::
///    This function returns a **copy** of the internal data.
///
/// .. seealso::
///    The explanation of the internal data structure, :ref:`here <qf_ferm_op-implementation>`.
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     uint64_t num_terms = 2;
///     uint64_t num_actions = 0;
///     QkComplex64 coeffs[2] = {{1.0, 0.0}, {0.0, -1.0}};
///     uint32_t boundaries[3] = {0, 0, 0};
///     QfFermionOperator *op =
///         qf_ferm_op_new(num_terms, num_actions, coeffs, NULL, NULL, boundaries);
///
///     QkComplex64 *coeffs_out;
///     uint64_t *coeffs_len;
///
///     qf_ferm_op_get_coeffs(op, &coeffs_out, &coeffs_len);
///
///     assert(coeffs_len == 2);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ferm_op_get_coeffs(
    op: *const FermionOperator,
    coeffs_out: *mut *mut Complex64,
    coeffs_len: *mut u64,
) {
    let op = unsafe { const_ptr_as_ref(op) };
    unsafe { coeffs_out.write(op.coeffs.as_ptr().cast_mut()) };
    unsafe { coeffs_len.write(op.coeffs.len().try_into().unwrap()) };
}

/// @ingroup qf_ferm_op
///
/// @brief Provides read-only access to the operator's actions.
///
/// @param op A pointer to the fermionic operator whose actions to access.
/// @param actions_out A pointer to the array of boolean values into which to write the actions.
/// @param actions_len A pointer to the integer into which to write the length of the output array.
///
/// @rst
///
/// .. note::
///    This function returns a **copy** of the internal data.
///
/// .. seealso::
///    The explanation of the internal data structure, :ref:`here <qf_ferm_op-implementation>`.
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     uint64_t num_terms = 2;
///     uint64_t num_actions = 2;
///     bool actions[2] = {true, false};
///     uint32_t modes[2] = {0, 1};
///     QkComplex64 coeffs[2] = {{1.0, 0.0}, {0.0, -1.0}};
///     uint32_t boundaries[3] = {0, 0, 2};
///     QfFermionOperator *op =
///         qf_ferm_op_new(num_terms, num_actions, coeffs, actions, modes, boundaries);
///
///     QkComplex64 *actions_out;
///     uint64_t *actions_len;
///
///     qf_ferm_op_get_actions(op, &actions_out, &actions_len);
///
///     assert(actions_len == 2);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ferm_op_get_actions(
    op: *const FermionOperator,
    actions_out: *mut *mut bool,
    actions_len: *mut u64,
) {
    let op = unsafe { const_ptr_as_ref(op) };
    unsafe { actions_out.write(op.actions.as_ptr().cast_mut()) };
    unsafe { actions_len.write(op.actions.len().try_into().unwrap()) };
}

/// @ingroup qf_ferm_op
///
/// @brief Provides read-only access to the operator's acted-upon mode indices.
///
/// @param op A pointer to the fermionic operator whose modes to access.
/// @param modes_out A pointer to the array of boolean values into which to write the modes.
/// @param modes_len A pointer to the integer into which to write the length of the output array.
///
/// @rst
///
/// .. note::
///    This function returns a **copy** of the internal data.
///
/// .. seealso::
///    The explanation of the internal data structure, :ref:`here <qf_ferm_op-implementation>`.
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     uint64_t num_terms = 2;
///     uint64_t num_actions = 2;
///     bool actions[2] = {true, false};
///     uint32_t modes[2] = {0, 1};
///     QkComplex64 coeffs[2] = {{1.0, 0.0}, {0.0, -1.0}};
///     uint32_t boundaries[3] = {0, 0, 2};
///     QfFermionOperator *op =
///         qf_ferm_op_new(num_terms, num_actions, coeffs, actions, modes, boundaries);
///
///     QkComplex64 *modes_out;
///     uint64_t *modes_len;
///
///     qf_ferm_op_get_modes(op, &modes_out, &modes_len);
///
///     assert(modes_len == 2);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ferm_op_get_modes(
    op: *const FermionOperator,
    modes_out: *mut *mut u32,
    modes_len: *mut u64,
) {
    let op = unsafe { const_ptr_as_ref(op) };
    unsafe { modes_out.write(op.modes.as_ptr().cast_mut()) };
    unsafe { modes_len.write(op.modes.len().try_into().unwrap()) };
}

/// @ingroup qf_ferm_op
///
/// @brief Provides read-only access to the indices indicating the boundaries between operator terms.
///
/// @param op A pointer to the fermionic operator whose boundaries to access.
/// @param boundaries_out A pointer to the array of boolean values into which to write the boundaries.
/// @param boundaries_len A pointer to the integer into which to write the length of the output array.
///
/// @rst
///
/// .. note::
///    This function returns a **copy** of the internal data.
///
/// .. seealso::
///    The explanation of the internal data structure, :ref:`here <qf_ferm_op-implementation>`.
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     uint64_t num_terms = 2;
///     uint64_t num_actions = 2;
///     bool actions[2] = {true, false};
///     uint32_t modes[2] = {0, 1};
///     QkComplex64 coeffs[2] = {{1.0, 0.0}, {0.0, -1.0}};
///     uint32_t boundaries[3] = {0, 0, 2};
///     QfFermionOperator *op =
///         qf_ferm_op_new(num_terms, num_actions, coeffs, actions, modes, boundaries);
///
///     QkComplex64 *boundaries_out;
///     uint64_t *boundaries_len;
///
///     qf_ferm_op_get_boundaries(op, &boundaries_out, &boundaries_len);
///
///     assert(boundaries_len == 3);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ferm_op_get_boundaries(
    op: *const FermionOperator,
    boundaries_out: *mut *mut usize,
    boundaries_len: *mut u64,
) {
    let op = unsafe { const_ptr_as_ref(op) };
    unsafe { boundaries_out.write(op.boundaries.as_ptr().cast_mut()) };
    unsafe { boundaries_len.write(op.boundaries.len().try_into().unwrap()) };
}

/// @ingroup qf_ferm_op
///
/// @brief Gets the size of the support of an operator.
///
/// @param op A pointer to the fermionic operator whose support size to get.
///
/// @return The number of distinct mode indices acted upon by the operator.
///
/// @rst
///
/// Use this to size the output buffer of :c:func:`qf_ferm_op_get_support`, which does not report a
/// length of its own.
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     bool actions[3] = {true, false, true};
///     uint32_t modes[3] = {2, 0, 2};
///     QkComplex64 coeffs[2] = {{1.0, 0.0}, {1.0, 0.0}};
///     uint32_t boundaries[3] = {0, 2, 3};
///     QfFermionOperator *op = qf_ferm_op_new(2, 3, coeffs, actions, modes, boundaries);
///
///     uint32_t num_support = qf_ferm_op_num_support(op);
///
///     assert(num_support == 2);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ferm_op_num_support(op: *const FermionOperator) -> u32 {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let op = unsafe { const_ptr_as_ref(op) };

    op.get_support()
        .len()
        .try_into()
        .expect("the number of distinct modes cannot exceed u32::MAX")
}

/// @ingroup qf_ferm_op
///
/// @brief Gets the support of an operator, i.e. the mode indices acted upon by the operator.
///
/// @param op A pointer to the fermionic operator whose support to get.
/// @param support_out A pointer to the integer array into which to write the support. It must be
///     sized to at least :c:func:`qf_ferm_op_num_support` elements.
///
/// @rst
///
/// The indices are written in **ascending** order, and every index appears exactly once no matter
/// how many terms act upon it.
///
/// .. note::
///    Unlike the other getters, this does not hand out a pointer into the operator: the support is
///    computed on demand rather than stored, so the caller provides the output buffer. Query its
///    required length with :c:func:`qf_ferm_op_num_support` first.
///
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     bool actions[3] = {true, false, true};
///     uint32_t modes[3] = {2, 0, 2};
///     QkComplex64 coeffs[2] = {{1.0, 0.0}, {1.0, 0.0}};
///     uint32_t boundaries[3] = {0, 2, 3};
///     QfFermionOperator *op = qf_ferm_op_new(2, 3, coeffs, actions, modes, boundaries);
///
///     uint32_t num_support = qf_ferm_op_num_support(op);
///     uint32_t *support_out = malloc(num_support * sizeof(uint32_t));
///
///     qf_ferm_op_get_support(op, support_out);
///
///     assert(support_out[0] == 0);
///     assert(support_out[1] == 2);
///
///     free(support_out);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ferm_op_get_support(op: *const FermionOperator, support_out: *mut u32) {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let op = unsafe { const_ptr_as_ref(op) };

    // The support is gathered into a `HashSet`, whose iteration order is not reproducible. Sorting
    // makes the output deterministic, which is what lets a caller index into it meaningfully.
    let mut support: Vec<u32> = op.get_support().into_iter().collect();
    support.sort_unstable();

    for (i, mode) in support.iter().enumerate() {
        // SAFETY: Per documentation, `support_out` is sized to hold the whole support.
        unsafe { support_out.add(i).write(*mode) };
    }
}

/// @ingroup qf_ferm_op
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
///     QfFermionOperator *zero = qf_ferm_op_zero();
///
///     QfFermionOperator *op_plus_zero = qf_ferm_op_add(op, zero);
///
///     assert(qf_ferm_op_equal(op, op_plus_zero));
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ferm_op_zero() -> *mut FermionOperator {
    let op = FermionOperator::zero();
    Box::into_raw(Box::new(op))
}

/// @ingroup qf_ferm_op
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
///     QfFermionOperator *one = qf_ferm_op_one();
///
///     QfFermionOperator *op_times_one = qf_ferm_op_compose(op, one);
///
///     assert(qf_ferm_op_equal(op, op_times_one));
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ferm_op_one() -> *mut FermionOperator {
    let op = FermionOperator::one();
    Box::into_raw(Box::new(op))
}

/// @ingroup qf_ferm_op
///
/// @brief Checks whether this operator tracks group indices.
///
/// @param op A pointer to the fermionic operator to be checked.
///
/// @return Whether the provided operator has a ``groups`` attribute.
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
///     QfFCIDump *fcidump = NULL;
///     qf_fcidump_from_file("molecule.fcidump", &fcidump);
///     QfFermionOperator *op = qf_ferm_op_from_fcidump(fcidump);
///
///     bool has_groups = qf_ferm_op_has_groups(op);
///
///     assert(!has_groups);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ferm_op_has_groups(op: *const FermionOperator) -> bool {
    let op = unsafe { const_ptr_as_ref(op) };
    op.has_groups()
}

/// @ingroup qf_ferm_op
///
/// @brief Gets the number of groups from an operator.
///
/// @param op A pointer to the fermionic operator whose number of groups to get.
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
///     QfFCIDump *fcidump = NULL;
///     qf_fcidump_from_file("molecule.fcidump", &fcidump);
///     QfFermionOperator *op = qf_ferm_op_from_fcidump(fcidump);
///
///     uint32_t num_groups = qf_ferm_op_num_groups(op);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ferm_op_num_groups(op: *const FermionOperator) -> u32 {
    let op = unsafe { const_ptr_as_ref(op) };
    op.num_groups().expect(
        "Expected groups to be present. It is the user's responsibility to check this via \
        qf_ferm_op_has_groups before calling this function.",
    )
}

/// @ingroup qf_ferm_op
///
/// @brief Gets the group indices for all operator terms.
///
/// @param op A pointer to the fermionic operator whose group indices to get.
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
///     QfFermionOperator *op = ...;
///     uint32_t *groups_out;
///     uint32_t groups_len;
///
///     qf_ferm_op_get_groups(op, &groups_out, &groups_len);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ferm_op_get_groups(
    op: *const FermionOperator,
    groups_out: *mut *mut u32,
    groups_len: *mut u64,
) {
    let op = unsafe { const_ptr_as_ref(op) };
    let groups = &op.groups.as_ref().expect(
        "Expected groups to be present. It is the user's responsibility to check this via \
        qf_ferm_op_has_groups before calling this function.",
    );
    unsafe { groups_out.write(groups.as_ptr().cast_mut()) };
    unsafe { groups_len.write(groups.len().try_into().unwrap()) };
}

/// @ingroup qf_ferm_op
///
/// @brief Sets the ``groups`` attribute of the provided operator.
///
/// @param op A pointer to the fermionic operator whose ``groups`` attribute to write.
/// @param groups_in A pointer to the ``groups`` integer array to write into the operator.
/// @param groups_len The number of terms in the ``groups_in`` array.
///
/// @return An exit code.
/// * ``QfExitCode_Success`` upon success
/// * ``QfExitCode_ValueError`` if ``groups_len`` differs from the number of terms in the operator
///
/// @rst
///
/// The length is validated here because nothing downstream re-checks it: too few indices would
/// silently drop the trailing terms wherever terms are iterated together with their groups, and too
/// many would make :c:func:`qf_ferm_op_num_groups` report groups that no term carries.
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
///     QfFermionOperator *op = ...;
///
///     uint32_t num_terms = 4;
///     uint32_t groups_in[4] = {0, 1, 0, 1};
///     qf_ferm_op_set_groups(op, groups_in, num_terms);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ferm_op_set_groups(
    op: *mut FermionOperator,
    groups_in: *const u32,
    groups_len: u64,
) -> ExitCode {
    let op = unsafe { mut_ptr_as_ref(op) };
    let groups_in = unsafe { const_ptr_as_ref(groups_in) };
    let mut groups = vec![0; groups_len as usize];
    groups.copy_from_slice(unsafe { slice_from_ptr(groups_in, groups_len as usize) });
    match op.set_groups(Some(groups)) {
        Ok(()) => ExitCode::Success,
        Err(_) => ExitCode::ValueError,
    }
}

/// @ingroup qf_ferm_op
///
/// @brief Deletes the ``groups`` attribute from the provided operator.
///
/// @param op A pointer to the fermionic operator whose ``groups`` attribute to delete.
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
///     QfFermionOperator *op = ...;
///
///     qf_ferm_op_del_groups(op);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ferm_op_del_groups(op: *mut FermionOperator) {
    let op = unsafe { mut_ptr_as_ref(op) };
    op.clear_groups();
}

/// @ingroup qf_ferm_op
///
/// @brief Splits this operator into a list of new operators based on its ``groups`` attribute.
///
/// @param op A pointer to the fermionic operator whose ``groups`` to split out.
/// @param group_indices A pointer to the array of group indices for which to build operators, in
///     the desired output order. May be ``NULL``, in which case every group is built, in index
///     order (equivalent to passing every index from ``0`` to
///     :c:func:`qf_ferm_op_num_groups` ``- 1``).
/// @param num_indices The number of indices in the ``group_indices`` array. Ignored if
///     ``group_indices`` is ``NULL``.
/// @param group_ops_out A pointer to the array of :c:struct:`QfFermionOperator` into which to
///     write the operators for each requested group. Must be sized to ``num_indices`` when
///     ``group_indices`` is non-``NULL``, or to :c:func:`qf_ferm_op_num_groups` when it is
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
///     uint64_t num_actions = 8;
///     bool actions[8] = {true, false, true, false, true, false, true, false};
///     uint32_t modes[8] = {0, 1, 2, 3, 1, 0, 3, 2};
///     QkComplex64 coeffs[4] = {{1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}};
///     uint32_t boundaries[5] = {0, 2, 4, 6, 8};
///     QfFermionOperator *op =
///         qf_ferm_op_new(num_terms, num_actions, coeffs, actions, modes, boundaries);
///
///     uint32_t groups_in[4] = {0, 1, 0, 1};
///     qf_ferm_op_set_groups(op, groups_in, num_terms);
///
///     // build every group, in index order
///     QfFermionOperator *group_ops[2];
///     qf_ferm_op_split_out_groups(op, NULL, 0, group_ops);
///
///     // build only group 1
///     uint32_t group_indices[1] = {1};
///     QfFermionOperator *group_op[1];
///     qf_ferm_op_split_out_groups(op, group_indices, 1, group_op);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ferm_op_split_out_groups(
    op: *const FermionOperator,
    group_indices: *const u32,
    num_indices: u64,
    group_ops_out: *mut *mut FermionOperator,
) {
    let op = unsafe { const_ptr_as_ref(op) };
    let group_indices = if group_indices.is_null() {
        None
    } else {
        Some(unsafe { slice_from_ptr(group_indices, num_indices as usize) })
    };
    let groups = op.split_out_groups(group_indices).expect(
        "Expected groups to be present. It is the user's responsibility to check this via \
        qf_ferm_op_has_groups before calling this function.",
    );
    let cgroups: Vec<*mut FermionOperator> = groups
        .into_iter()
        .map(|g| Box::into_raw(Box::new(g)))
        .collect();
    for (i, ptr) in cgroups.iter().enumerate() {
        unsafe { group_ops_out.add(i).write(*ptr) };
    }
}

/// @ingroup qf_ferm_op
///
/// @brief Adds a term to an existing operator.
///
/// @param op A pointer to the fermionic operator to be modified.
/// @param num_actions The length of the actions array.
/// @param actions A pointer to an array of actions. The length of this array should be
///     ``num_actions``.
/// @param modes A pointer to an array of action modes. The length of this array should be
///     ``num_actions``.
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
///     QfFermionOperator *one = qf_ferm_op_one();
///
///     QfFermionOperator *op = qf_ferm_op_zero();
///     bool actions[0] = {};
///     uint32_t modes[0] = {};
///     QkComplex64 coeff = {1.0, 0.0};
///
///     qf_ferm_op_add_term(op, 0, actions, modes, &coeff);
///
///     assert(qf_ferm_op_equal(op, one));
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ferm_op_add_term(
    op: *mut FermionOperator,
    num_actions: u64,
    actions: *const bool,
    modes: *const u32,
    coeff: *const Complex64,
) {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let op = unsafe { mut_ptr_as_ref(op) };
    let coeff = unsafe { const_ptr_as_ref(coeff) };

    let num_actions = num_actions as usize;

    op.coeffs.push(*coeff);
    op.actions
        .extend_from_slice(unsafe { slice_from_ptr(actions, num_actions) });
    op.modes
        .extend_from_slice(unsafe { slice_from_ptr(modes, num_actions) });
    op.boundaries.push(op.modes.len());

    op.clear_groups();
}

/// @ingroup qf_ferm_op
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
///     QfFermionOperator *one = qf_ferm_op_one();
///     QfFermionOperator *zero = qf_ferm_op_zero();
///
///     QfFermionOperator *result = qf_ferm_op_add(one, zero);
///
///     assert(qf_ferm_op_equal(result, one));
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ferm_op_add(
    left: *const FermionOperator,
    right: *const FermionOperator,
) -> *mut FermionOperator {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let left = unsafe { const_ptr_as_ref(left) };
    let right = unsafe { const_ptr_as_ref(right) };

    let result = left.__add__(right);
    Box::into_raw(Box::new(result))
}

/// @ingroup qf_ferm_op
///
/// @brief Adds two operators together, scaling the coefficients of the right one.
///
/// @param left A pointer to the left operator.
/// @param right A pointer to the right operator.
/// @param factor A pointer to the factor to scale the right operator's coefficients with.
///
/// @return A pointer to the resulting operator, ``left + factor * right``.
///
/// @rst
///
/// This fuses the scaling into the addition, which avoids building a fully scaled copy of ``right``
/// just to append it. Passing a factor of ``-1`` therefore subtracts, which is why no separate
/// subtraction function is provided.
///
/// .. note::
///    Like :c:func:`qf_ferm_op_add`, this appends the terms of ``right`` without combining them with
///    those of ``left``, so the result tracks no group indices. Call :c:func:`qf_ferm_op_simplify` to
///    collect equal terms afterwards.
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     QfFermionOperator *one = qf_ferm_op_one();
///     QfFermionOperator *other = qf_ferm_op_one();
///
///     // Subtracting via a factor of -1 leaves the two terms cancelling each other out.
///     QkComplex64 factor = {-1.0, 0.0};
///     QfFermionOperator *result = qf_ferm_op_scaled_add(one, other, &factor);
///
///     QfFermionOperator *simplified = qf_ferm_op_simplify(result, 1e-10);
///     assert(qf_ferm_op_len(simplified) == 0);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ferm_op_scaled_add(
    left: *const FermionOperator,
    right: *const FermionOperator,
    factor: *const Complex64,
) -> *mut FermionOperator {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let left = unsafe { const_ptr_as_ref(left) };
    let right = unsafe { const_ptr_as_ref(right) };
    let factor = unsafe { const_ptr_as_ref(factor) };

    let result = left.__scaled_add__(right, *factor);
    Box::into_raw(Box::new(result))
}

/// @ingroup qf_ferm_op
///
/// @brief Adds an operator into another one, in place.
///
/// @param left A pointer to the operator to add into.
/// @param right A pointer to the operator to add.
///
/// @rst
///
/// The in-place counterpart of :c:func:`qf_ferm_op_add`, which saves copying ``left`` into a freshly
/// allocated result.
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
///     QfFermionOperator *left = qf_ferm_op_one();
///     QfFermionOperator *right = qf_ferm_op_one();
///
///     qf_ferm_op_add_inplace(left, right);
///
///     assert(qf_ferm_op_len(left) == 2);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ferm_op_add_inplace(
    left: *mut FermionOperator,
    right: *const FermionOperator,
) {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let left = unsafe { mut_ptr_as_ref(left) };
    let right = unsafe { const_ptr_as_ref(right) };

    left.__iadd__(right);
}

/// @ingroup qf_ferm_op
///
/// @brief Adds an operator into another one in place, scaling the coefficients of the right one.
///
/// @param left A pointer to the operator to add into.
/// @param right A pointer to the operator to add.
/// @param factor A pointer to the factor to scale the right operator's coefficients with.
///
/// @rst
///
/// The in-place counterpart of :c:func:`qf_ferm_op_scaled_add`, computing ``left + factor * right``
/// without allocating a result. As there, a factor of ``-1`` subtracts.
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
///     QfFermionOperator *left = qf_ferm_op_one();
///     QfFermionOperator *right = qf_ferm_op_one();
///
///     // Subtract, leaving two terms that cancel each other out.
///     QkComplex64 factor = {-1.0, 0.0};
///     qf_ferm_op_scaled_add_inplace(left, right, &factor);
///
///     QfFermionOperator *simplified = qf_ferm_op_simplify(left, 1e-10);
///     assert(qf_ferm_op_len(simplified) == 0);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ferm_op_scaled_add_inplace(
    left: *mut FermionOperator,
    right: *const FermionOperator,
    factor: *const Complex64,
) {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let left = unsafe { mut_ptr_as_ref(left) };
    let right = unsafe { const_ptr_as_ref(right) };
    let factor = unsafe { const_ptr_as_ref(factor) };

    left.__iscaled_add__(right, *factor);
}

/// @ingroup qf_ferm_op
///
/// @brief Multiplies an operator by a scalar, in place.
///
/// @param op A pointer to the operator to scale.
/// @param scalar A pointer to the scalar.
///
/// @rst
///
/// The in-place counterpart of :c:func:`qf_ferm_op_mul`, which saves copying the operator into a
/// freshly allocated result.
///
/// .. note::
///    Unlike the two in-place additions above, this **preserves** the ``groups`` attribute: scaling
///    the coefficients leaves the number of terms, and hence the one-index-per-term invariant,
///    untouched.
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     QfFermionOperator *op = qf_ferm_op_one();
///     QkComplex64 scalar = {2.0, 0.0};
///
///     qf_ferm_op_mul_inplace(op, &scalar);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ferm_op_mul_inplace(
    op: *mut FermionOperator,
    scalar: *const Complex64,
) {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let op = unsafe { mut_ptr_as_ref(op) };
    let scalar = unsafe { const_ptr_as_ref(scalar) };

    op.__imul__(*scalar);
}

/// @ingroup qf_ferm_op
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
///     QfFermionOperator *one = qf_ferm_op_one();
///     QkComplex64 coeff = {2.0, 0.0};
///     QfFermionOperator *result = qf_ferm_op_mul(one, &coeff);
///
///     QfFermionOperator *expected = qf_ferm_op_zero();
///     bool actions[0] = {};
///     uint32_t modes[0] = {};
///     qf_ferm_op_add_term(expected, 0, actions, modes, &coeff);
///
///     assert(qf_ferm_op_equal(result, expected));
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ferm_op_mul(
    op: *const FermionOperator,
    scalar: *const Complex64,
) -> *mut FermionOperator {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let op = unsafe { const_ptr_as_ref(op) };
    let scalar = unsafe { const_ptr_as_ref(scalar) };

    let result = op.__mul__(*scalar);
    Box::into_raw(Box::new(result))
}

/// @ingroup qf_ferm_op
///
/// @brief Composes two operators with each other.
///
/// @param left A pointer to the left operator.
/// @param right A pointer to the right operator.
///
/// @return A pointer to ``left.compose(right)``, which equals the operator
///     ``result = right @ left`` in terms of the matrix multiplication ``@``. In other words,
///     ``right`` is applied first. To obtain ``left @ right``, swap the arguments.
///
/// @rst
///
/// .. note::
///    The composition of two operators tracks no groups, even when both operands do.
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     QfFermionOperator *one = qf_ferm_op_one();
///     QfFermionOperator *zero = qf_ferm_op_zero();
///
///     QfFermionOperator *result = qf_ferm_op_compose(one, zero);
///
///     assert(qf_ferm_op_equal(result, zero));
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ferm_op_compose(
    left: *const FermionOperator,
    right: *const FermionOperator,
) -> *mut FermionOperator {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let left = unsafe { const_ptr_as_ref(left) };
    let right = unsafe { const_ptr_as_ref(right) };

    let result = left.__and__(right);
    Box::into_raw(Box::new(result))
}

/// @ingroup qf_ferm_op
///
/// @brief Returns the Hermitian conjugate (or adjoint) of an operator.
///
/// This affects the terms and coefficients as follows:
///
/// - the actions in each term reverse their order and flip between creation and annihilation
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
///     QfFermionOperator *op = qf_ferm_op_zero();
///     bool actions[0] = {};
///     uint32_t modes[0] = {};
///     QkComplex64 coeff = {0.0, 1.0};
///     qf_ferm_op_add_term(op, 0, actions, modes, &coeff);
///
///     QfFermionOperator *adjoint = qf_ferm_op_adjoint(op);
///
///     QfFermionOperator *expected = qf_ferm_op_zero();
///     QkComplex64 coeff_adj = {0.0, -1.0};
///     qf_ferm_op_add_term(expected, 0, actions, modes, &coeff_adj);
///
///     assert(qf_ferm_op_equal(adjoint, expected));
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ferm_op_adjoint(op: *const FermionOperator) -> *mut FermionOperator {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let op = unsafe { const_ptr_as_ref(op) };

    let result = op.adjoint();
    Box::into_raw(Box::new(result))
}

/// @ingroup qf_ferm_op
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
///    separate coefficients for duplicate terms consider calling :c:func:`qf_ferm_op_simplify`
///    instead!
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     QfFermionOperator *op = qf_ferm_op_zero();
///     bool actions[0] = {};
///     uint32_t modes[0] = {};
///     QkComplex64 coeff = {1e-8};
///     qf_ferm_op_add_term(op, 0, actions, modes, &coeff);
///
///     qf_ferm_op_ichop(op, 1e-6);
///
///     QfFermionOperator *expected = qf_ferm_op_zero();
///
///     assert(qf_ferm_op_equal(op, expected));
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ferm_op_ichop(op: *mut FermionOperator, atol: f64) {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let op = unsafe { mut_ptr_as_ref(op) };

    op.ichop(atol);
}

/// @ingroup qf_ferm_op
///
/// @brief Returns an equivalent but simplified operator.
///
/// @param op A pointer to the fermionic operator to be simplified.
/// @param atol The absolute tolerance for coefficient truncation.
///
/// @return An equivalent but simplified operator.
///
/// @rst
/// The simplification process first sums all coefficients that belong to equal terms and then
/// only retains those whose total coefficient exceeds the specified tolerance (just like
/// :c:func:`qf_ferm_op_ichop`).
///
/// When an operator has been arithmetically manipulated or constructed in a way that does not
/// guarantee unique terms, this method should be called before applying any method that
/// filters numerically small coefficients to avoid loss of information. See the example below
/// which showcases how :c:func:`qf_ferm_op_ichop` can truncate terms that sum to a total
/// coefficient magnitude which should not be truncated:
///
/// .. code-block:: c
///     :linenos:
///
///     uint64_t num_terms = 100000;
///     uint64_t num_actions = 0;
///     bool actions[0] = {};
///     uint32_t modes[0] = {};
///     QkComplex64 coeffs[100000];
///     uint32_t boundaries[100001];
///     for (int i = 0; i < 100000; i++) {
///       coeffs[i].re = 1e-5;
///       coeffs[i].im = 0.0;
///       boundaries[i] = 0;
///     }
///     boundaries[100000] = 0;
///     QfFermionOperator *op = qf_ferm_op_new(num_terms, num_actions, coeffs,
///                                            actions, modes, boundaries);
///
///     QfFermionOperator *canon = qf_ferm_op_simplify(op, 1e-4);
///
///     QfFermionOperator *one = qf_ferm_op_one();
///     bool canon_is_equal = qf_ferm_op_equiv(canon, one, 1e-6);
///
///     qf_ferm_op_ichop(op, 1e-4);
///
///     QfFermionOperator *zero = qf_ferm_op_zero();
///     bool ichop_is_equal = qf_ferm_op_equiv(op, zero, 1e-6);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ferm_op_simplify(
    op: *const FermionOperator,
    atol: f64,
) -> *mut FermionOperator {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let op = unsafe { const_ptr_as_ref(op) };

    let result = op.simplify(atol);
    Box::into_raw(Box::new(result))
}

/// @ingroup qf_ferm_op
///
/// @brief Returns an equivalent operator with normal ordered terms.
///
/// The normal order of an operator term is defined such that all creation actions appear before
/// all annihilation actions.
/// Within each group, the acted-upon modes are ordered lexicographically. Whether their order
/// is ascending or descending depends upon the value of the ``sandwich`` argument:
///
/// - ``NULL``: both groups are ordered lexicographically descending (e.g. ``+_1 +_0 -_1 -_0``)
/// - ``True``: larger indices appear towards the middle, i.e. creation actions are
///   lexicographically ascending while annihilation ones are descending (e.g. ``+_0 +_1 -_1 -_0``)
/// - ``False``: smaller indices appear towards the middle, i.e. creation actions are
///   lexicographically descending while annihilation ones are ascending (e.g. ``+_1 +_0 -_0 -_1``)
///
/// @param op A pointer to the operator.
/// @param sandwich A pointer to a boolean value. This pointer can be ``NULL``.
///
/// @return A pointer to the created operator.
///
/// @rst
///
/// .. note::
///    When a term is being reordered, the anti-commutation relations have to be taken into
///    account, :math:`a_i a^\dagger_j = \delta_{ij} - a^\dagger_j a^i`, implying that the
///    number of terms might change.
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     QfFermionOperator *op = qf_ferm_op_zero();
///     bool actions[4] = {false, true, false, true};
///     uint32_t modes[4] = {1, 1, 0, 0};
///     QkComplex64 coeff = {1.0, 0.0};
///     qf_ferm_op_add_term(op, 4, actions, modes, &coeff);
///
///     QfFermionOperator *normal_ordered = qf_ferm_op_normal_ordered(op, NULL);
///
///     uint64_t num_terms = 4;
///     uint64_t num_actions = 8;
///     bool actions_exp[8] = {true, false, true, false, true, true, false, false};
///     uint32_t modes_exp[8] = {0, 0, 1, 1, 1, 0, 1, 0};
///     QkComplex64 coeffs_exp[4] = {
///         {1.0, 0.0}, {-1.0, 0.0}, {-1.0, 0.0}, {-1.0, 0.0}};
///     uint32_t boundaries_exp[5] = {0, 0, 2, 4, 8};
///     QfFermionOperator *expected =
///         qf_ferm_op_new(num_terms, num_actions, coeffs_exp, actions_exp,
///                        modes_exp, boundaries_exp);
///
///     assert(qf_ferm_op_equal(normal_ordered, expected));
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ferm_op_normal_ordered(
    op: *const FermionOperator,
    sandwich: *const bool,
) -> *mut FermionOperator {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let op = unsafe { const_ptr_as_ref(op) };

    let sandwich = if sandwich.is_null() {
        None
    } else {
        unsafe { Some(sandwich.as_ref().unwrap()) }
    };

    let result = op.normal_ordered(sandwich.copied());
    Box::into_raw(Box::new(result))
}

/// @ingroup qf_ferm_op
///
/// @brief Checks whether an operator is Hermitian.
///
/// @param op A pointer to the fermionic operator to be checked.
/// @param atol The absolute tolerance upto which coefficients are considered equal.
///
/// @return Whether the provided operator is Hermitian.
///
/// @rst
///
/// .. note::
///    This check is implemented using :c:func:`qf_ferm_op_equiv` on the
///    :c:func:`qf_ferm_op_normal_ordered` difference of ``op`` and its
///    :c:func:`qf_ferm_op_adjoint` and :c:func:`qf_ferm_op_zero`.
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     QfFermionOperator *op = qf_ferm_op_zero();
///     bool actions1[2] = {true, false};
///     uint32_t modes1[2] = {0, 1};
///     QkComplex64 coeff1 = {0.0, 1.00001};
///     qf_ferm_op_add_term(op, 2, actions1, modes1, &coeff1);
///     bool actions2[2] = {true, false};
///     uint32_t modes2[2] = {1, 0};
///     QkComplex64 coeff2 = {0.0, -1};
///     qf_ferm_op_add_term(op, 2, actions2, modes2, &coeff2);
///
///     assert(qf_ferm_op_is_hermitian(op, 1e-4));
///     assert(!qf_ferm_op_is_hermitian(op, 1e-8));
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ferm_op_is_hermitian(op: *const FermionOperator, atol: f64) -> bool {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let op = unsafe { const_ptr_as_ref(op) };

    op.is_hermitian(atol)
}

/// @ingroup qf_ferm_op
///
/// @brief Checks the maximum rank of an operator.
///
/// @param op A pointer to the fermionic operator to be checked.
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
///     QfFermionOperator *op = qf_ferm_op_zero();
///     bool actions[4] = {true, false, true, false};
///     uint32_t modes[4] = {0, 1, 2, 3};
///     QkComplex64 coeff = {1.0, 0.0};
///     qf_ferm_op_add_term(op, 4, actions, modes, &coeff);
///
///     assert(qf_ferm_op_max_rank(op), 4);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ferm_op_max_rank(op: *const FermionOperator) -> u32 {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let op = unsafe { const_ptr_as_ref(op) };

    op.max_rank()
}

/// @ingroup qf_ferm_op
///
/// @brief Checks whether an operator is particle-number conserving.
///
/// @param op A pointer to the fermionic operator to be checked.
///
/// @return Whether the provided operator is particle-number conserving.
///
/// @rst
///
/// This is the special case of :c:func:`qf_ferm_op_conserves_sector` that treats all modes as a
/// single block. Use that function to additionally require conservation *within* mode blocks, for
/// example to check conservation of the spin z-component alongside the particle number.
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     // A hopping term conserves the particle number, ...
///     QfFermionOperator *op = qf_ferm_op_zero();
///     bool actions[2] = {true, false};
///     uint32_t modes[2] = {0, 2};
///     QkComplex64 coeff = {1.0, 0.0};
///     qf_ferm_op_add_term(op, 2, actions, modes, &coeff);
///
///     assert(qf_ferm_op_conserves_particle_number(op));
///
///     // ... whereas a bare creation operator does not.
///     QfFermionOperator *cre = qf_ferm_op_zero();
///     bool cre_actions[1] = {true};
///     uint32_t cre_modes[1] = {0};
///     qf_ferm_op_add_term(cre, 1, cre_actions, cre_modes, &coeff);
///
///     assert(!qf_ferm_op_conserves_particle_number(cre));
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ferm_op_conserves_particle_number(op: *const FermionOperator) -> bool {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let op = unsafe { const_ptr_as_ref(op) };

    op.conserves_particle_number()
}

/// @ingroup qf_ferm_op
///
/// @brief Checks whether an operator conserves the particle number within each mode block.
///
/// @param op A pointer to the fermionic operator to be checked.
/// @param block_sizes A pointer to the array of block sizes partitioning the mode range. May be
///     ``NULL`` to treat all modes as a single block.
/// @param num_blocks The number of entries in the ``block_sizes`` array.
///
/// @return Whether the provided operator conserves the particle number within every block.
///
/// @rst
///
/// The ``block_sizes`` partition the mode range into consecutive, non-overlapping blocks: block
/// ``b`` spans the modes ``[start_b, start_b + block_sizes[b])`` where ``start_b`` is the sum of all
/// preceding block sizes. A term conserves the sector if and only if, in *every* block, its number
/// of creation operators equals its number of annihilation operators. A term acting on a mode beyond
/// the last block never conserves the sector, because there is no block whose count it could
/// balance.
///
/// For a spin-orbital layout of ``norb`` spatial orbitals, ``[norb]`` checks the particle number of
/// a spinless system, while ``[norb, norb]`` requires the alpha block ``[0, norb)`` and the beta
/// block ``[norb, 2 * norb)`` to be conserved *separately*, i.e. strict conservation of both the
/// particle number and the z-component of the spin.
///
/// .. note::
///    Passing ``NULL`` (with ``num_blocks`` set to ``0``) treats all modes as one unbounded block,
///    making this equivalent to :c:func:`qf_ferm_op_conserves_particle_number`.
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     // This term hops an electron from the beta block into the alpha block.
///     QfFermionOperator *op = qf_ferm_op_zero();
///     bool actions[2] = {true, false};
///     uint32_t modes[2] = {0, 2};
///     QkComplex64 coeff = {1.0, 0.0};
///     qf_ferm_op_add_term(op, 2, actions, modes, &coeff);
///
///     // The total particle number is conserved, ...
///     uint32_t one_block[1] = {4};
///     assert(qf_ferm_op_conserves_sector(op, one_block, 1));
///     assert(qf_ferm_op_conserves_sector(op, NULL, 0));
///
///     // ... but the two spin sectors are not conserved individually.
///     uint32_t spin_blocks[2] = {2, 2};
///     assert(!qf_ferm_op_conserves_sector(op, spin_blocks, 2));
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ferm_op_conserves_sector(
    op: *const FermionOperator,
    block_sizes: *const u32,
    num_blocks: u64,
) -> bool {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let op = unsafe { const_ptr_as_ref(op) };

    // A null pointer stands for "no blocks", which the core implementation reads as one unbounded
    // block. Guarding here keeps `slice_from_ptr` away from a null pointer.
    let block_sizes = if block_sizes.is_null() {
        &[][..]
    } else {
        // SAFETY: Per documentation, `block_sizes` holds `num_blocks` readable elements.
        unsafe { slice_from_ptr(block_sizes, num_blocks as usize) }
    };

    op.conserves_sector(block_sizes)
}

/// @ingroup qf_ferm_op
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
///     QfFermionOperator *one = qf_ferm_op_one();
///     QfFermionOperator *zero = qf_ferm_op_zero();
///
///     assert(qf_ferm_op_equal(one, one));
///     assert(!qf_ferm_op_equal(one, zero));
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ferm_op_equal(
    left: *const FermionOperator,
    right: *const FermionOperator,
) -> bool {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let left = unsafe { const_ptr_as_ref(left) };
    let right = unsafe { const_ptr_as_ref(right) };

    left.eq(right)
}

/// @ingroup qf_ferm_op
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
///     QfFermionOperator *zero = qf_ferm_op_zero();
///
///     QfFermionOperator *op = qf_ferm_op_zero();
///     bool actions[0] = {};
///     uint32_t modes[0] = {};
///     QkComplex64 coeff = {1e-7, 0.0};
///     qf_ferm_op_add_term(op, 0, actions, modes, &coeff);
///
///     assert(qf_ferm_op_equiv(op, zero, 1e-6));
///     assert(!qf_ferm_op_equiv(op, zero, 1e-8));
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ferm_op_equiv(
    left: *const FermionOperator,
    right: *const FermionOperator,
    atol: f64,
) -> bool {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let left = unsafe { const_ptr_as_ref(left) };
    let right = unsafe { const_ptr_as_ref(right) };

    left.equiv(right, atol)
}

/// @ingroup qf_ferm_op
///
/// @brief Returns the length (or number of terms) of the provided operator.
///
/// @param op A pointer to the fermionic operator.
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
///     QfFermionOperator *op = qf_ferm_op_zero();
///     bool actions[4] = {true, false, true, false};
///     uint32_t modes[4] = {0, 1, 2, 3};
///     QkComplex64 coeff = {1.0, 0.0};
///     qf_ferm_op_add_term(op, 4, actions, modes, &coeff);
///
///     assert(qf_ferm_op_len(op) == 1);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ferm_op_len(op: *const FermionOperator) -> usize {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let op = unsafe { const_ptr_as_ref(op) };

    op.boundaries.len() - 1
}

/// @ingroup qf_ferm_op
///
/// @brief Relabels the indices of the provided operator.
///
/// @param op A pointer to the fermionic operator.
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
///     QfFermionOperator *op = qf_ferm_op_zero();
///     bool actions[4] = {true, false, true, false};
///     uint32_t indices[4] = {0, 1, 2, 3};
///     QkComplex64 coeff = {1.0, 0.0};
///     qf_ferm_op_add_term(op, 4, actions, indices, &coeff);
///
///     uint32_t permutation[4] = {3, 2, 1, 0};
///
///     QfExitCode exit = qf_ferm_op_relabel_modes(op, 4, permutation);
///
///     assert(exit == QfExitCode_Success);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ferm_op_relabel_modes(
    op: *mut FermionOperator,
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
