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

#include "common.h"
#include <qiskit_fermions.h>
#include <stdint.h>
#include <stdnoreturn.h>

static int test_filter_drops_constant_number_and_products(void) {
    // Terms (in order):
    //   - constant                              (diagonal -> drop)
    //   - n_0 = a†_0 a_0                         (diagonal -> drop)
    //   - n_0 n_1 = a†_0 a†_1 a_1 a_0            (diagonal -> drop)
    //   - a†_0 a_1                               (off-diagonal -> keep)
    uint64_t num_terms = 4;
    uint64_t num_actions = 8;
    bool actions[8] = {true, false, true, true, false, false, true, false};
    uint32_t modes[8] = {0, 0, 0, 1, 1, 0, 0, 1};
    QkComplex64 coeffs[4] = {{1.0, 0.0}, {2.0, 0.0}, {3.0, 0.0}, {4.0, 0.0}};
    uint32_t boundaries[5] = {0, 0, 2, 6, 8};
    QfFermionOperator *op =
        qf_ferm_op_new(num_terms, num_actions, coeffs, actions, modes, boundaries);

    qf_ferm_op_filter_diagonal_terms(op);

    // Only the off-diagonal hopping term a†_0 a_1 must survive.
    QfFermionOperator *expected = qf_ferm_op_zero();
    bool action_keep[2] = {true, false};
    uint32_t modes_keep[2] = {0, 1};
    QkComplex64 coeff_keep = {4.0, 0.0};
    qf_ferm_op_add_term(expected, 2, action_keep, modes_keep, &coeff_keep);

    bool is_equal = qf_ferm_op_equal(op, expected);

    qf_ferm_op_free(op);
    qf_ferm_op_free(expected);

    if (!is_equal) {
        return EqualityError;
    }
    return Ok;
}

static int test_filter_keeps_off_diagonal_hopping(void) {
    // A single off-diagonal hopping term must be kept untouched.
    uint64_t num_terms = 1;
    uint64_t num_actions = 2;
    bool actions[2] = {true, false};
    uint32_t modes[2] = {0, 1};
    QkComplex64 coeffs[1] = {{1.0, 0.0}};
    uint32_t boundaries[2] = {0, 2};
    QfFermionOperator *op =
        qf_ferm_op_new(num_terms, num_actions, coeffs, actions, modes, boundaries);

    QfFermionOperator *expected =
        qf_ferm_op_new(num_terms, num_actions, coeffs, actions, modes, boundaries);

    qf_ferm_op_filter_diagonal_terms(op);

    bool is_equal = qf_ferm_op_equal(op, expected);

    qf_ferm_op_free(op);
    qf_ferm_op_free(expected);

    if (!is_equal) {
        return EqualityError;
    }
    return Ok;
}

static int test_filter_electronic_structure_hamiltonian(void) {
    QfFCIDump *fcidump = qf_fcidump_from_file("../../h2.fcidump");
    QfFermionOperator *op = qf_ferm_op_from_fcidump(fcidump);
    qf_fcidump_free(fcidump);

    QfFermionOperator *normal = qf_ferm_op_normal_ordered(op, NULL);
    qf_ferm_op_free(op);

    QkComplex64 *coeffs_before;
    uint64_t num_terms_before;
    qf_ferm_op_get_coeffs(normal, &coeffs_before, &num_terms_before);

    qf_ferm_op_filter_diagonal_terms(normal);

    QkComplex64 *coeffs_after;
    uint64_t num_terms_after;
    qf_ferm_op_get_coeffs(normal, &coeffs_after, &num_terms_after);

    qf_ferm_op_free(normal);

    // The constant offset and the number-operator terms must have been removed.
    if (num_terms_after >= num_terms_before) {
        return EqualityError;
    }
    return Ok;
}

int test_filtering(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_filter_drops_constant_number_and_products);
    num_failed += RUN_TEST(test_filter_keeps_off_diagonal_hopping);
    num_failed += RUN_TEST(test_filter_electronic_structure_hamiltonian);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
