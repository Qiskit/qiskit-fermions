// This code is a Qiskit project.
//
// (C) Copyright IBM 2026.
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use crate::exit_codes::ExitCode;
use crate::pointers::{check_ptr, const_ptr_as_ref, mut_ptr_as_ref};

use num_complex::Complex64;
use qiskit_fermions_core::operators::majorana_operator::MajoranaOperator;
use qiskit_fermions_core::operators::{OperatorMacro, OperatorTrait};

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
    let coeffs = unsafe { const_ptr_as_ref(coeffs) };
    let modes = unsafe { const_ptr_as_ref(modes) };
    let boundaries = unsafe { const_ptr_as_ref(boundaries) };

    let num_terms = num_terms as usize;
    let num_modes = num_modes as usize;

    check_ptr(coeffs).unwrap();
    check_ptr(modes).unwrap();
    check_ptr(boundaries).unwrap();
    // SAFETY: At this point we know the pointers are non-null and aligned. We rely on C that
    // the pointers point to arrays of appropriate length, as specified in the function docs.
    let ccoeffs = unsafe { ::std::slice::from_raw_parts(coeffs, num_terms).to_vec() };
    let cmodes = unsafe { ::std::slice::from_raw_parts(modes, num_modes).to_vec() };
    let cboundaries = unsafe { ::std::slice::from_raw_parts(boundaries, num_terms + 1) };
    let cboundaries_usize = cboundaries.iter().map(|b| *b as usize).collect();

    let op = MajoranaOperator {
        coeffs: ccoeffs,
        modes: cmodes,
        boundaries: cboundaries_usize,
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
/// @brief Adds a term to an existing operator.
///
/// @param op A pointer to the Majorana operator to be modified.
/// @param num_modes The length of the modes array.
/// @param modes A pointer to an array of mode indices. The length of this array should be
///     ``num_modes``.
/// @param coeff A pointer to the complex coefficient.
///
/// @return An exit code.
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
) -> ExitCode {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let op = unsafe { mut_ptr_as_ref(op) };
    let coeff = unsafe { const_ptr_as_ref(coeff) };

    let num_modes = num_modes as usize;

    check_ptr(modes).unwrap();
    // SAFETY: At this point we know the pointers are non-null and aligned. We rely on C that
    // the pointers point to arrays of appropriate length, as specified in the function docs.
    let cmodes = unsafe { ::std::slice::from_raw_parts(modes, num_modes) };

    op.coeffs.push(*coeff);
    op.modes.extend_from_slice(cmodes);
    op.boundaries.push(op.modes.len());

    ExitCode::Success
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
///     QfExitCode result = qf_maj_op_ichop(op, 1e-6);
///
///     QfMajoranaOperator *expected = qf_maj_op_zero();
///
///     assert(qf_maj_op_equal(op, expected));
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_maj_op_ichop(op: *mut MajoranaOperator, atol: f64) -> ExitCode {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let op = unsafe { mut_ptr_as_ref(op) };

    op.ichop(atol);

    ExitCode::Success
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
/// The normal order of an operator term is defined such that all actions are ordered by
/// lexicographically descending indices.
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
///     uint32_t modes[4] = {0, 2, 1, 3};
///     QkComplex64 coeff = {1.0, 0.0};
///     qf_maj_op_add_term(op, 4, modes, &coeff);
///
///     QfMajoranaOperator *normal_ordered = qf_maj_op_normal_ordered(op);
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
    reduce: bool,
) -> *mut MajoranaOperator {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let op = unsafe { const_ptr_as_ref(op) };

    let result = op.normal_ordered(reduce);
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
/// @brief Checks the many-body order of an operator.
///
/// @param op A pointer to the Majorana operator to be checked.
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
///     QfMajoranaOperator *op = qf_maj_op_zero();
///     uint32_t modes[4] = {0, 1, 2, 3};
///     QkComplex64 coeff = {1.0, 0.0};
///     qf_maj_op_add_term(op, 4, modes, &coeff);
///
///     assert(qf_maj_op_many_body_order(op, 4));
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_maj_op_many_body_order(op: *const MajoranaOperator) -> u32 {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let op = unsafe { const_ptr_as_ref(op) };

    op.many_body_order()
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
