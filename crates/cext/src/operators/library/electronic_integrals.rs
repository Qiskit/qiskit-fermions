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

use crate::pointers::check_ptr;

use ndarray::{Array1, ArrayView1};
use qiskit_fermions_core::operators::fermion_operator::FermionOperator;
use qiskit_fermions_core::operators::library::electronic_integrals::{From1Body, From2Body};

/// @ingroup qf_electronic_integrals
///
/// @brief Constructs an operator from spin-symmetric triangular 1-body integrals.
///
/// @param one_body_a a 1-dimensional array of length ``norb * (norb + 1) / 2`` storing the 1-body
///                   electronic integral coefficients of the alpha-spin species, as a flattened
///                   triangular matrix.
/// @param norb the number of orbitals.
///
/// @return The 1-body component of the electronic structure Hamiltonian as defined above.
///
/// @rst
///
/// The resulting operator is defined by
///
/// .. math::
///
///     \sum_i c^\alpha_{ii} (a^\dagger_i a_i + a^\dagger_{i+n} a_{i+n}) +
///     \sum_{i \lt j} c^\alpha_{ij} (a^\dagger_i a_j + a^\dagger_j a_i +
///                                   a^\dagger_{i+n} a_{j+n} + a^\dagger_{j+n} a_{i+n})
///
/// where :math:`c^\alpha` are the integral coefficients stored in ``one_body_a``, :math:`i`
/// and :math:`j` are the indices expanded from the triangular index :math:`ij` which indexes
/// the array, and :math:`n` is the number of orbitals, ``norb``.
///
/// .. code-block:: c
///     :linenos:
///
///     int norb = 2;
///     double one_body_a[3] = {1.0, 2.0, 3.0};
///     QfFermionOperator *op = qf_ferm_op_from_1body_tril_spin_sym(one_body_a, norb);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ferm_op_from_1body_tril_spin_sym(
    one_body_a: *mut f64,
    norb: u32,
) -> *mut FermionOperator {
    let len_arr = ((norb * (norb + 1)) / 2) as usize;

    check_ptr(one_body_a).unwrap();
    // SAFETY: At this point we know the pointers are non-null and aligned. We rely on C that
    // the pointers point to arrays of appropriate length, as specified in the function docs.
    let carray = unsafe { ::std::slice::from_raw_parts(one_body_a, len_arr).to_vec() };
    let one_body_a_arr = Array1::from_vec(carray);

    let op = FermionOperator::from_1body_tril_spin_sym(ArrayView1::from(&one_body_a_arr), norb);
    Box::into_raw(Box::new(op))
}

/// @ingroup qf_electronic_integrals
///
/// @brief Constructs an operator from separate spin-species triangular 1-body integrals.
///
/// @param one_body_a a 1-dimensional array of length ``norb * (norb + 1) / 2`` storing the 1-body
///                   electronic integral coefficients of the alpha-spin species, as a flattened
///                   triangular matrix.
/// @param one_body_b a 1-dimensional array of length ``norb * (norb + 1) / 2`` storing the 1-body
///                   electronic integral coefficients of the beta-spin species, as a flattened
///                   triangular matrix.
/// @param norb the number of orbitals.
///
/// @return The 1-body component of the electronic structure Hamiltonian as defined above.
///
/// @rst
/// The resulting operator is defined by
///
/// .. math::
///
///     \sum_i c^\alpha_{ii} a^\dagger_i a_i + c^\beta_{ii} a^\dagger_{i+n} a_{i+n} +
///     \sum_{i \lt j} c^\alpha_{ij} (a^\dagger_i a_j + a^\dagger_j a_i) +
///                    c^\beta_{ij} (a^\dagger_{i+n} a_{j+n} + a^\dagger_{j+n} a_{i+n})
///
/// where :math:`c^\alpha` (:math:`c^\beta`) are the integral coefficients stored in
/// ``one_body_a`` (``one_body_b``, resp.), :math:`i` and :math:`j` are the indices expanded
/// from the triangular index :math:`ij` which indexes the arrays, and :math:`n` is the number
/// of orbitals, ``norb``.
///
/// .. code-block:: c
///     :linenos:
///
///     int norb = 2;
///     double one_body_a[3] = {1.0, 2.0, 3.0};
///     double one_body_b[3] = {-1.0, -2.0, -3.0};
///     QfFermionOperator *op = qf_ferm_op_from_1body_tril_spin(one_body_a, one_body_b, norb);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ferm_op_from_1body_tril_spin(
    one_body_a: *mut f64,
    one_body_b: *mut f64,
    norb: u32,
) -> *mut FermionOperator {
    let len_arr = ((norb * (norb + 1)) / 2) as usize;

    check_ptr(one_body_a).unwrap();
    // SAFETY: At this point we know the pointers are non-null and aligned. We rely on C that
    // the pointers point to arrays of appropriate length, as specified in the function docs.
    let carray = unsafe { ::std::slice::from_raw_parts(one_body_a, len_arr).to_vec() };
    let one_body_a_arr = Array1::from_vec(carray);

    check_ptr(one_body_b).unwrap();
    // SAFETY: At this point we know the pointers are non-null and aligned. We rely on C that
    // the pointers point to arrays of appropriate length, as specified in the function docs.
    let carray = unsafe { ::std::slice::from_raw_parts(one_body_b, len_arr).to_vec() };
    let one_body_b_arr = Array1::from_vec(carray);

    let op = FermionOperator::from_1body_tril_spin(
        ArrayView1::from(&one_body_a_arr),
        ArrayView1::from(&one_body_b_arr),
        norb,
    );
    Box::into_raw(Box::new(op))
}

/// @ingroup qf_electronic_integrals
///
/// @brief Constructs an operator from spin-symmetric triangular 2-body integrals.
///
/// @param two_body_aa a 1-dimensional array of the S8-fold symmetric 2-body electronic integral
///                    coefficients of the alpha/alpha-spin species, as a flattened array.
/// @param norb the number of orbitals.
///
/// @return The 2-body component of the electronic structure Hamiltonian as defined above.
///
/// @rst
///
///
/// The resulting operator is defined by
///
/// .. math::
///
///     \sum_{ijkl} \frac{1}{2} c^{\alpha\alpha}_{ijkl}
///         \sum_{(i,j,k,l) \in \mathcal{P}(ijkl)}
///             (a^\dagger_i a^\dagger_k a_l a_j +
///              a^\dagger_{i+n} a^\dagger_k a_l a_{j+n} +
///              a^\dagger_i a^\dagger_{k+n} a_{l+n} a_j +
///              a^\dagger_{i+n} a^\dagger_{k+n} a_{l+n} a_{j+n})
///
/// where :math:`c^{\alpha\alpha}` are the integral coefficients stored in ``two_body_aa``,
/// :math:`ijkl` is the running index of the array, :math:`\mathcal{P}` generates the unique
/// permutations of the 4-index :math:`(i,j,k,l)` (see below), and :math:`n` is the number of
/// orbitals, ``norb``.
///
/// .. note::
///     ``two_body_aa`` is an S8-fold symmetric array. That means, it is the flattened
///     lower-triangular data of a matrix of shape ``(npair, npair)``, where
///     ``npair = (norb * (norb + 1) // 2``. This in turn is the lower-triangular data of the
///     4-dimensional array of shape ``(norb, norb, norb, norb)``.
///     Therefore, :math:`\mathcal{P}` above expands the flattened index :math:`ijkl` into all
///     index permutations :math:`(i,j,k,l)` that index this 4-dimensional array.
///
/// .. code-block:: c
///     :linenos:
///
///     int norb = 2;
///     double two_body_aa[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
///     QfFermionOperator *op = qf_ferm_op_from_2body_tril_spin_sym(two_body_aa, norb);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ferm_op_from_2body_tril_spin_sym(
    two_body_aa: *mut f64,
    norb: u32,
) -> *mut FermionOperator {
    let npair = ((norb * (norb + 1)) / 2) as usize;
    let len_arr = (npair * (npair + 1)) / 2;

    check_ptr(two_body_aa).unwrap();
    // SAFETY: At this point we know the pointers are non-null and aligned. We rely on C that
    // the pointers point to arrays of appropriate length, as specified in the function docs.
    let carray = unsafe { ::std::slice::from_raw_parts(two_body_aa, len_arr).to_vec() };
    let two_body_aa_arr = Array1::from_vec(carray);

    let op = FermionOperator::from_2body_tril_spin_sym(ArrayView1::from(&two_body_aa_arr), norb);

    Box::into_raw(Box::new(op))
}

/// @ingroup qf_electronic_integrals
///
/// @brief Constructs an operator from separate spin-species triangular 2-body integrals.
///
/// @param two_body_aa a 1-dimensional array of the S8-fold symmetric 2-body electronic integral
///                    coefficients of the alpha/alpha-spin species, as a flattened array.
/// @param two_body_ab a 1-dimensional array of the S4-fold symmetric 2-body electronic integral
///                    coefficients of the alpha/beta-spin species, as a flattened array.
/// @param two_body_bb a 1-dimensional array of the S8-fold symmetric 2-body electronic integral
///                    coefficients of the beta/beta-spin species, as a flattened array.
/// @param norb the number of orbitals.
///
/// @return The 2-body component of the electronic structure Hamiltonian as defined above.
///
/// @rst
///
/// The resulting operator is defined by
///
/// .. math::
///
///     \sum_{ijkl} \frac{1}{2}
///         \sum_{(i,j,k,l) \in \mathcal{P}(ijkl)}
///             c^{\alpha\alpha}_{ijkl} a^\dagger_i a^\dagger_k a_l a_j +
///             c^{\beta\beta}_{ijkl} a^\dagger_{i+n} a^\dagger_{k+n} a_{l+n} a_{j+n}
///     + \sum_{ijkl} \frac{1}{2}
///         \sum_{(i,j,k,l) \in \mathcal{P'}(ijkl)}
///             c^{\alpha\beta}_{ijkl} a^\dagger_{i+n} a^\dagger_k a_l a_{j+n} +
///             c^{\alpha\beta}_{ijkl} a^\dagger_i a^\dagger_{k+n} a_{l+n} a_j +
///
/// where :math:`c^{\alpha\alpha}` (:math:`c^{\alpha\beta}`, :math:`c^{\beta\beta}`) are the
/// integral coefficients stored in ``two_body_aa`` (``two_body_ab``, ``two_body_bb``, resp.),
/// :math:`ijkl` is the running index of the array, :math:`\mathcal{P}` (:math:`\mathcal{P'}`)
/// generates the unique permutations of the 4-index :math:`(i,j,k,l)` (see below), and
/// :math:`n` is the number of orbitals, ``norb``.
///
/// .. note::
///     ``two_body_aa`` and ``two_body_bb`` are a S8-fold symmetric arrays. That means, they
///     are the flattened lower-triangular data of matrices of shape ``(npair, npair)``, where
///     ``npair = (norb * (norb + 1) // 2``. These in turn are the lower-triangular data of the
///     4-dimensional arrays of shape ``(norb, norb, norb, norb)``.
///     Therefore, :math:`\mathcal{P}` above expands the flattened index :math:`ijkl` into all
///     index permutations :math:`(i,j,k,l)` that index these 4-dimensional arrays.
///
///     However, ``two_body_ab`` is only S4-fold symmetric. Thus, it contains the full data of
///     the ``(npair, npair)`` matrix (but still in flattened form). :math:`\mathcal{P'}`
///     performs the corresponding index expansion. (In the definition above, we reused the
///     index :math:`ijkl` as an abuse of notation.)
///
/// .. code-block:: c
///     :linenos:
///
///     int norb = 2;
///     double two_body_aa[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
///     double two_body_ab[9] = {11.0, 12.0, 13.0, 14.0, 15.0,
///                              16.0, 17.0, 18.0, 19.0};
///     double two_body_bb[6] = {-1.0, -2.0, -3.0, -4.0, -5.0, -6.0};
///     QfFermionOperator *op = qf_ferm_op_from_2body_tril_spin(
///         two_body_aa, two_body_ab, two_body_bb, norb);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ferm_op_from_2body_tril_spin(
    two_body_aa: *mut f64,
    two_body_ab: *mut f64,
    two_body_bb: *mut f64,
    norb: u32,
) -> *mut FermionOperator {
    let npair = ((norb * (norb + 1)) / 2) as usize;
    let len_arr_s4 = npair * npair;
    let len_arr_s8 = (npair * (npair + 1)) / 2;

    check_ptr(two_body_aa).unwrap();
    // SAFETY: At this point we know the pointers are non-null and aligned. We rely on C that
    // the pointers point to arrays of appropriate length, as specified in the function docs.
    let carray = unsafe { ::std::slice::from_raw_parts(two_body_aa, len_arr_s8).to_vec() };
    let two_body_aa_arr = Array1::from_vec(carray);

    check_ptr(two_body_ab).unwrap();
    // SAFETY: At this point we know the pointers are non-null and aligned. We rely on C that
    // the pointers point to arrays of appropriate length, as specified in the function docs.
    let carray = unsafe { ::std::slice::from_raw_parts(two_body_ab, len_arr_s4).to_vec() };
    let two_body_ab_arr = Array1::from_vec(carray);

    check_ptr(two_body_bb).unwrap();
    // SAFETY: At this point we know the pointers are non-null and aligned. We rely on C that
    // the pointers point to arrays of appropriate length, as specified in the function docs.
    let carray = unsafe { ::std::slice::from_raw_parts(two_body_bb, len_arr_s8).to_vec() };
    let two_body_bb_arr = Array1::from_vec(carray);

    let op = FermionOperator::from_2body_tril_spin(
        ArrayView1::from(&two_body_aa_arr),
        ArrayView1::from(&two_body_ab_arr),
        ArrayView1::from(&two_body_bb_arr),
        norb,
    );

    Box::into_raw(Box::new(op))
}
