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
#include <qiskit.h>
#include <qiskit_fermions.h>
#include <stdint.h>
#include <stdio.h>

static int test_fermion_to_majorana(void) {
    QfFermionOperator *fer_op = qf_ferm_op_zero();
    QkComplex64 coeff = {1.0, 0.0};
    bool action_fer[2] = {true, false};
    uint32_t indices_fer[2] = {0, 0};
    qf_ferm_op_add_term(fer_op, 2, action_fer, indices_fer, &coeff);

    QfMajoranaOperator *result = qf_fermion_to_majorana(fer_op);
    QfMajoranaOperator *canon = qf_maj_op_normal_ordered(result, true);

    uint64_t num_terms = 2;
    uint64_t num_modes = 2;
    uint32_t modes[2] = {1, 0};
    QkComplex64 exp_coeffs[2] = {{0.5, 0.0}, {0.0, 0.5}};
    uint32_t boundaries[3] = {0, 0, 2};
    QfMajoranaOperator *expected =
        qf_maj_op_new(num_terms, num_modes, exp_coeffs, modes, boundaries);

    bool is_equal = qf_maj_op_equiv(canon, expected, 1e-8);

    qf_ferm_op_free(fer_op);
    qf_maj_op_free(result);
    qf_maj_op_free(canon);
    qf_maj_op_free(expected);

    if (!is_equal) {
        return EqualityError;
    }
    return Ok;
}

static int test_majorana_to_fermion(void) {
    QfMajoranaOperator *maj_op = qf_maj_op_zero();
    QkComplex64 coeff = {1.0, 0.0};
    uint32_t modes_maj[2] = {0, 1};
    qf_maj_op_add_term(maj_op, 2, modes_maj, &coeff);

    QfFermionOperator *result = qf_majorana_to_fermion(maj_op);
    QfFermionOperator *canon = qf_ferm_op_normal_ordered(result);

    uint64_t num_terms = 2;
    uint64_t num_actions = 2;
    bool actions[2] = {true, false};
    uint32_t modes[2] = {0, 0};
    QkComplex64 exp_coeffs[2] = {{0.0, -1.0}, {0.0, 2.0}};
    uint32_t boundaries[3] = {0, 0, 2};
    QfFermionOperator *expected =
        qf_ferm_op_new(num_terms, num_actions, exp_coeffs, actions, modes, boundaries);

    bool is_equal = qf_ferm_op_equiv(canon, expected, 1e-8);

    qf_maj_op_free(maj_op);
    qf_ferm_op_free(result);
    qf_ferm_op_free(canon);
    qf_ferm_op_free(expected);

    if (!is_equal) {
        return EqualityError;
    }
    return Ok;
}

int test_majorana_fermion(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_fermion_to_majorana);
    num_failed += RUN_TEST(test_majorana_to_fermion);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
