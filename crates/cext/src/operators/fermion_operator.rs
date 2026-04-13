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
/// Any of the pointer arguments may be ``NULL`` if and only if their corresponding length is zero.
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
/// @brief Checks whether this operator has a ``groups`` attribute that is not empty.
///
/// @param op A pointer to the fermionic operator to be checked.
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
///     QfFCIDump *fcidump = qf_fcidump_from_file("molecule.fcidump");
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
    op.groups.is_some()
}

/// @ingroup qf_ferm_op
///
/// @brief Gets the number of unique group indices from an operator.
///
/// @param op A pointer to the fermionic operator whose number of group indices to get.
///
/// @return The number of unique group indices from the operator's ``groups`` attribute.
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
///     QfFCIDump *fcidump = qf_fcidump_from_file("molecule.fcidump");
///     QfFermionOperator *op = qf_ferm_op_from_fcidump(fcidump);
///
///     uint32_t num_groups = qf_ferm_op_num_groups(op);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ferm_op_num_groups(op: *const FermionOperator) -> u32 {
    let op = unsafe { const_ptr_as_ref(op) };
    let groups = &op.groups.as_ref().expect(
        "Expected groups to be present. It is the user's responsibility to check this via \
        qf_ferm_op_has_groups before calling this function.",
    );
    groups.iter().max().unwrap() + 1
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
    groups_len: *mut u32,
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
///     uint32_t num_terms = 4;
///     uint32_t groups_in[4] = {0, 1, 0, 1};
///     qf_ferm_op_set_groups(op, groups_in, num_terms);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ferm_op_set_groups(
    op: *mut FermionOperator,
    groups_in: *const u32,
    groups_len: u32,
) {
    let op = unsafe { mut_ptr_as_ref(op) };
    let groups_in = unsafe { const_ptr_as_ref(groups_in) };
    let mut groups = vec![0; groups_len as usize];
    groups.copy_from_slice(unsafe { slice_from_ptr(groups_in, groups_len as usize) });
    op.groups = Some(groups);
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
    op.groups = None;
}

/// @ingroup qf_ferm_op
///
/// @brief Splits this operator into a list of new operators based on its ``groups`` attribute.
///
/// @param op A pointer to the fermionic operator whose ``groups`` to split out.
/// @param group_ops_out A pointer to the array of :c:struct:`QfFermionOperator` into which to
///     write the operators for each group.
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
///     QfFermionOperator *group_ops[2];
///     qf_ferm_op_split_out_groups(op, group_ops);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ferm_op_split_out_groups(
    op: *const FermionOperator,
    group_ops_out: *mut *mut FermionOperator,
) {
    let op = unsafe { const_ptr_as_ref(op) };
    let groups = op.split_out_groups().expect(
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

    op.groups = None;
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
///    This functions truncates coefficients greedily! If the acted upon operator may contain
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
/// The normal order of an operator term is defined such that all creation actions before all
/// annihilation actions and the modes of actions within each group descend lexicographically
/// (e.g. ``+_1 +_0 -_1 -_0``).
///
/// @param op A pointer to the operator.
///
/// @return A pointer to the created operator.
///
/// @rst
///
/// .. note::
///    When a term is being reordered, the anti-commutation relations have to be taken into
///    account, :math:`a_i a^\dagger_j = \delta_{ij} - a^\dagger_j a^i`, implying that the
///    number of terms may change.
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
///     QfFermionOperator *normal_ordered = qf_ferm_op_normal_ordered(op);
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
) -> *mut FermionOperator {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let op = unsafe { const_ptr_as_ref(op) };

    let result = op.normal_ordered();
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
///     qf_ferm_op_add_term(op, 2, actions2, modes2, &coeff1);
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
/// @brief Checks the many-body order of an operator.
///
/// @param op A pointer to the fermionic operator to be checked.
///
/// @return The many-body order of the operator.
///
/// @rst
///
/// .. note::
///    The many-body order is defined as the length of the longest term contained in the operator.
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
///     assert(qf_ferm_op_many_body_order(op, 4));
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ferm_op_many_body_order(op: *const FermionOperator) -> u32 {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let op = unsafe { const_ptr_as_ref(op) };

    op.many_body_order()
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
pub unsafe extern "C" fn qf_ferm_op_conserves_particle_number(op: *const FermionOperator) -> bool {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let op = unsafe { const_ptr_as_ref(op) };

    op.conserves_particle_number()
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
