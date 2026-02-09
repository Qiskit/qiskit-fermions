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
use qiskit_fermions_core::operators::{OperatorMacro, OperatorTrait};

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
/// @param indices A pointer to an array of action indices over all terms. The length of this
///     array should be ``num_actions``.
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
///     uint32_t indices[4] = {0, 1, 2, 3};
///     QkComplex64 coeffs[3] = {{1.0, 0.0}, {-1.0, 0.0}, {0.0, -1.0}};
///     uint32_t boundaries[4] = {0, 0, 2, 4};
///     QfFermionOperator *op = qf_ferm_op_new(num_terms, num_actions, coeffs,
///                                            actions, indices, boundaries);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ferm_op_new(
    num_terms: u64,
    num_actions: u64,
    coeffs: *const Complex64,
    actions: *const bool,
    indices: *const u32,
    boundaries: *const u32,
) -> *mut FermionOperator {
    let num_terms = num_terms as usize;
    let num_actions = num_actions as usize;

    let op = FermionOperator {
        coeffs: unsafe { slice_from_ptr(coeffs, num_terms).to_vec() },
        actions: unsafe { slice_from_ptr(actions, num_actions).to_vec() },
        indices: unsafe { slice_from_ptr(indices, num_actions).to_vec() },
        boundaries: unsafe {
            slice_from_ptr(boundaries, num_terms + 1)
                .into_iter()
                .map(|b| *b as usize)
                .collect()
        },
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
/// @brief Adds a term to an existing operator.
///
/// @param op A pointer to the fermionic operator to be modified.
/// @param num_actions The length of the actions array.
/// @param actions A pointer to an array of actions. The length of this array should be
///     ``num_actions``.
/// @param indices A pointer to an array of action indices. The length of this array should be
///     ``num_actions``.
/// @param coeff A pointer to the complex coefficient.
///
/// @return An exit code.
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
///     QfFermionOperator *one = qf_ferm_op_one();
///
///     QfFermionOperator *op = qf_ferm_op_zero();
///     bool actions[0] = {};
///     uint32_t indices[0] = {};
///     QkComplex64 coeff = {1.0, 0.0};
///
///     qf_ferm_op_add_term(op, 0, actions, indices, &coeff);
///
///     assert(qf_ferm_op_equal(op, one));
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ferm_op_add_term(
    op: *mut FermionOperator,
    num_actions: u64,
    actions: *const bool,
    indices: *const u32,
    coeff: *const Complex64,
) -> ExitCode {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let op = unsafe { mut_ptr_as_ref(op) };
    let coeff = unsafe { const_ptr_as_ref(coeff) };

    let num_actions = num_actions as usize;

    op.coeffs.push(*coeff);
    op.actions
        .extend_from_slice(unsafe { slice_from_ptr(actions, num_actions) });
    op.indices
        .extend_from_slice(unsafe { slice_from_ptr(indices, num_actions) });
    op.boundaries.push(op.indices.len());

    ExitCode::Success
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
///     uint32_t indices[0] = {};
///     qf_ferm_op_add_term(expected, 0, actions, indices, &coeff);
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
///     uint32_t indices[0] = {};
///     QkComplex64 coeff = {0.0, 1.0};
///     qf_ferm_op_add_term(op, 0, actions, indices, &coeff);
///
///     QfFermionOperator *adjoint = qf_ferm_op_adjoint(op);
///
///     QfFermionOperator *expected = qf_ferm_op_zero();
///     QkComplex64 coeff_adj = {0.0, -1.0};
///     qf_ferm_op_add_term(expected, 0, actions, indices, &coeff_adj);
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
///     uint32_t indices[0] = {};
///     QkComplex64 coeff = {1e-8};
///     qf_ferm_op_add_term(op, 0, actions, indices, &coeff);
///
///     QfExitCode result = qf_ferm_op_ichop(op, 1e-6);
///
///     QfFermionOperator *expected = qf_ferm_op_zero();
///
///     assert(qf_ferm_op_equal(op, expected));
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ferm_op_ichop(op: *mut FermionOperator, atol: f64) -> ExitCode {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let op = unsafe { mut_ptr_as_ref(op) };

    op.ichop(atol);

    ExitCode::Success
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
///     uint32_t indices[0] = {};
///     QkComplex64 coeffs[100000];
///     uint32_t boundaries[100001];
///     for (int i = 0; i < 100000; i++) {
///       coeffs[i].re = 1e-5;
///       coeffs[i].im = 0.0;
///       boundaries[i] = 0;
///     }
///     boundaries[100000] = 0;
///     QfFermionOperator *op = qf_ferm_op_new(num_terms, num_actions, coeffs,
///                                            actions, indices, boundaries);
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
/// annihilation actions and the indices of actions within each group descend lexicographically
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
///     uint32_t indices[4] = {1, 1, 0, 0};
///     QkComplex64 coeff = {1.0, 0.0};
///     qf_ferm_op_add_term(op, 4, actions, indices, &coeff);
///
///     QfFermionOperator *normal_ordered = qf_ferm_op_normal_ordered(op);
///
///     uint64_t num_terms = 4;
///     uint64_t num_actions = 8;
///     bool actions_exp[8] = {true, false, true, false, true, true, false, false};
///     uint32_t indices_exp[8] = {0, 0, 1, 1, 1, 0, 1, 0};
///     QkComplex64 coeffs_exp[4] = {
///         {1.0, 0.0}, {-1.0, 0.0}, {-1.0, 0.0}, {-1.0, 0.0}};
///     uint32_t boundaries_exp[5] = {0, 0, 2, 4, 8};
///     QfFermionOperator *expected =
///         qf_ferm_op_new(num_terms, num_actions, coeffs_exp, actions_exp,
///                        indices_exp, boundaries_exp);
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
///     uint32_t indices1[2] = {0, 1};
///     QkComplex64 coeff1 = {0.0, 1.00001};
///     qf_ferm_op_add_term(op, 2, actions1, indices1, &coeff1);
///     bool actions2[2] = {true, false};
///     uint32_t indices2[2] = {1, 0};
///     QkComplex64 coeff2 = {0.0, -1};
///     qf_ferm_op_add_term(op, 2, actions2, indices2, &coeff1);
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
///     uint32_t indices[4] = {0, 1, 2, 3};
///     QkComplex64 coeff = {1.0, 0.0};
///     qf_ferm_op_add_term(op, 4, actions, indices, &coeff);
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
///     uint32_t indices1[2] = {0, 1};
///     QkComplex64 coeff1 = {0.0, 1.00001};
///     qf_ferm_op_add_term(op, 2, actions1, indices1, &coeff1);
///     bool actions2[2] = {true, false};
///     uint32_t indices2[2] = {1, 0};
///     QkComplex64 coeff2 = {0.0, -1};
///     qf_ferm_op_add_term(op, 2, actions2, indices2, &coeff2);
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
///     uint32_t indices[0] = {};
///     QkComplex64 coeff = {1e-7, 0.0};
///     qf_ferm_op_add_term(op, 0, actions, indices, &coeff);
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
///     uint32_t indices[4] = {0, 1, 2, 3};
///     QkComplex64 coeff = {1.0, 0.0};
///     qf_ferm_op_add_term(op, 4, actions, indices, &coeff);
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
