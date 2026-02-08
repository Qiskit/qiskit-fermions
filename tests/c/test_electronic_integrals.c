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

static int test_ferm_op_from_1body_tril_spin_sym(void) {
    int norb = 2;
    double one_body_a[3] = {1.0, 2.0, 3.0};
    QfFermionOperator *op = qf_ferm_op_from_1body_tril_spin_sym(one_body_a, norb);

    uint64_t num_terms = 8;
    uint64_t num_actions = 16;
    bool actions_exp[16] = {true, false, true, false, true, false, true, false,
                            true, false, true, false, true, false, true, false};
    uint32_t indices_exp[16] = {0, 0, 2, 2, 1, 0, 0, 1, 3, 2, 2, 3, 1, 1, 3, 3};
    QkComplex64 coeffs_exp[8] = {{1.0, 0.0}, {1.0, 0.0}, {2.0, 0.0}, {2.0, 0.0},
                                 {2.0, 0.0}, {2.0, 0.0}, {3.0, 0.0}, {3.0, 0.0}};
    uint32_t boundaries_exp[9] = {0, 2, 4, 6, 8, 10, 12, 14, 16};
    QfFermionOperator *expected = qf_ferm_op_new(num_terms, num_actions, coeffs_exp, actions_exp,
                                                 indices_exp, boundaries_exp);

    bool is_equal = qf_ferm_op_equal(op, expected);

    qf_ferm_op_free(op);
    qf_ferm_op_free(expected);

    if (!is_equal) {
        return EqualityError;
    }
    return Ok;
}

static int test_ferm_op_from_1body_tril_spin(void) {
    int norb = 2;
    double one_body_a[3] = {1.0, 2.0, 3.0};
    double one_body_b[3] = {-1.0, -2.0, -3.0};
    QfFermionOperator *op = qf_ferm_op_from_1body_tril_spin(one_body_a, one_body_b, norb);

    uint64_t num_terms = 8;
    uint64_t num_actions = 16;
    bool actions_exp[16] = {true, false, true, false, true, false, true, false,
                            true, false, true, false, true, false, true, false};
    uint32_t indices_exp[16] = {0, 0, 1, 0, 0, 1, 1, 1, 2, 2, 3, 2, 2, 3, 3, 3};
    QkComplex64 coeffs_exp[8] = {{1.0, 0.0},  {2.0, 0.0},  {2.0, 0.0},  {3.0, 0.0},
                                 {-1.0, 0.0}, {-2.0, 0.0}, {-2.0, 0.0}, {-3.0, 0.0}};
    uint32_t boundaries_exp[9] = {0, 2, 4, 6, 8, 10, 12, 14, 16};
    QfFermionOperator *expected = qf_ferm_op_new(num_terms, num_actions, coeffs_exp, actions_exp,
                                                 indices_exp, boundaries_exp);

    bool is_equal = qf_ferm_op_equal(op, expected);

    qf_ferm_op_free(op);
    qf_ferm_op_free(expected);

    if (!is_equal) {
        return EqualityError;
    }
    return Ok;
}

static int test_ferm_op_from_2body_tril_spin_sym(void) {
    int norb = 2;
    double two_body_aa[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    QfFermionOperator *op = qf_ferm_op_from_2body_tril_spin_sym(two_body_aa, norb);

    uint64_t num_terms = 64;
    uint64_t num_actions = 256;
    bool actions_exp[256] = {
        true,  true,  false, false, true,  true,  false, false, true,  true,  false, false, true,
        true,  false, false, true,  true,  false, false, true,  true,  false, false, true,  true,
        false, false, true,  true,  false, false, true,  true,  false, false, true,  true,  false,
        false, true,  true,  false, false, true,  true,  false, false, true,  true,  false, false,
        true,  true,  false, false, true,  true,  false, false, true,  true,  false, false, true,
        true,  false, false, true,  true,  false, false, true,  true,  false, false, true,  true,
        false, false, true,  true,  false, false, true,  true,  false, false, true,  true,  false,
        false, true,  true,  false, false, true,  true,  false, false, true,  true,  false, false,
        true,  true,  false, false, true,  true,  false, false, true,  true,  false, false, true,
        true,  false, false, true,  true,  false, false, true,  true,  false, false, true,  true,
        false, false, true,  true,  false, false, true,  true,  false, false, true,  true,  false,
        false, true,  true,  false, false, true,  true,  false, false, true,  true,  false, false,
        true,  true,  false, false, true,  true,  false, false, true,  true,  false, false, true,
        true,  false, false, true,  true,  false, false, true,  true,  false, false, true,  true,
        false, false, true,  true,  false, false, true,  true,  false, false, true,  true,  false,
        false, true,  true,  false, false, true,  true,  false, false, true,  true,  false, false,
        true,  true,  false, false, true,  true,  false, false, true,  true,  false, false, true,
        true,  false, false, true,  true,  false, false, true,  true,  false, false, true,  true,
        false, false, true,  true,  false, false, true,  true,  false, false, true,  true,  false,
        false, true,  true,  false, false, true,  true,  false, false};
    uint32_t indices_exp[256] = {
        0, 0, 0, 0, 2, 0, 0, 2, 0, 2, 2, 0, 2, 2, 2, 2, 1, 0, 0, 0, 3, 0, 0, 2, 1, 2, 2, 0, 3,
        2, 2, 2, 0, 0, 0, 1, 2, 0, 0, 3, 0, 2, 2, 1, 2, 2, 2, 3, 0, 1, 0, 0, 2, 1, 0, 2, 0, 3,
        2, 0, 2, 3, 2, 2, 0, 0, 1, 0, 2, 0, 1, 2, 0, 2, 3, 0, 2, 2, 3, 2, 1, 1, 0, 0, 3, 1, 0,
        2, 1, 3, 2, 0, 3, 3, 2, 2, 0, 1, 0, 1, 2, 1, 0, 3, 0, 3, 2, 1, 2, 3, 2, 3, 1, 0, 1, 0,
        3, 0, 1, 2, 1, 2, 3, 0, 3, 2, 3, 2, 0, 0, 1, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 2, 3, 3, 1,
        0, 0, 1, 3, 0, 0, 3, 1, 2, 2, 1, 3, 2, 2, 3, 0, 1, 1, 0, 2, 1, 1, 2, 0, 3, 3, 0, 2, 3,
        3, 2, 1, 1, 0, 1, 3, 1, 0, 3, 1, 3, 2, 1, 3, 3, 2, 3, 1, 0, 1, 1, 3, 0, 1, 3, 1, 2, 3,
        1, 3, 2, 3, 3, 1, 1, 1, 0, 3, 1, 1, 2, 1, 3, 3, 0, 3, 3, 3, 2, 0, 1, 1, 1, 2, 1, 1, 3,
        0, 3, 3, 1, 2, 3, 3, 3, 1, 1, 1, 1, 3, 1, 1, 3, 1, 3, 3, 1, 3, 3, 3, 3};
    QkComplex64 coeffs_exp[64] = {
        {0.5, 0.0}, {0.5, 0.0}, {0.5, 0.0}, {0.5, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0},
        {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0},
        {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {1.5, 0.0},
        {1.5, 0.0}, {1.5, 0.0}, {1.5, 0.0}, {1.5, 0.0}, {1.5, 0.0}, {1.5, 0.0}, {1.5, 0.0},
        {1.5, 0.0}, {1.5, 0.0}, {1.5, 0.0}, {1.5, 0.0}, {1.5, 0.0}, {1.5, 0.0}, {1.5, 0.0},
        {1.5, 0.0}, {2.0, 0.0}, {2.0, 0.0}, {2.0, 0.0}, {2.0, 0.0}, {2.0, 0.0}, {2.0, 0.0},
        {2.0, 0.0}, {2.0, 0.0}, {2.5, 0.0}, {2.5, 0.0}, {2.5, 0.0}, {2.5, 0.0}, {2.5, 0.0},
        {2.5, 0.0}, {2.5, 0.0}, {2.5, 0.0}, {2.5, 0.0}, {2.5, 0.0}, {2.5, 0.0}, {2.5, 0.0},
        {2.5, 0.0}, {2.5, 0.0}, {2.5, 0.0}, {2.5, 0.0}, {3.0, 0.0}, {3.0, 0.0}, {3.0, 0.0},
        {3.0, 0.0}};
    uint32_t boundaries_exp[65] = {0,   4,   8,   12,  16,  20,  24,  28,  32,  36,  40,  44,  48,
                                   52,  56,  60,  64,  68,  72,  76,  80,  84,  88,  92,  96,  100,
                                   104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144, 148, 152,
                                   156, 160, 164, 168, 172, 176, 180, 184, 188, 192, 196, 200, 204,
                                   208, 212, 216, 220, 224, 228, 232, 236, 240, 244, 248, 252, 256};

    QfFermionOperator *expected = qf_ferm_op_new(num_terms, num_actions, coeffs_exp, actions_exp,
                                                 indices_exp, boundaries_exp);

    bool is_equal = qf_ferm_op_equal(op, expected);

    qf_ferm_op_free(op);
    qf_ferm_op_free(expected);

    if (!is_equal) {
        return EqualityError;
    }
    return Ok;
}

static int test_ferm_op_from_2body_tril_spin(void) {
    int norb = 2;
    double two_body_aa[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double two_body_ab[9] = {11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0};
    double two_body_bb[6] = {-1.0, -2.0, -3.0, -4.0, -5.0, -6.0};
    QfFermionOperator *op =
        qf_ferm_op_from_2body_tril_spin(two_body_aa, two_body_ab, two_body_bb, norb);

    uint64_t num_terms = 64;
    uint64_t num_actions = 256;
    bool actions_exp[256] = {
        true,  true,  false, false, true,  true,  false, false, true,  true,  false, false, true,
        true,  false, false, true,  true,  false, false, true,  true,  false, false, true,  true,
        false, false, true,  true,  false, false, true,  true,  false, false, true,  true,  false,
        false, true,  true,  false, false, true,  true,  false, false, true,  true,  false, false,
        true,  true,  false, false, true,  true,  false, false, true,  true,  false, false, true,
        true,  false, false, true,  true,  false, false, true,  true,  false, false, true,  true,
        false, false, true,  true,  false, false, true,  true,  false, false, true,  true,  false,
        false, true,  true,  false, false, true,  true,  false, false, true,  true,  false, false,
        true,  true,  false, false, true,  true,  false, false, true,  true,  false, false, true,
        true,  false, false, true,  true,  false, false, true,  true,  false, false, true,  true,
        false, false, true,  true,  false, false, true,  true,  false, false, true,  true,  false,
        false, true,  true,  false, false, true,  true,  false, false, true,  true,  false, false,
        true,  true,  false, false, true,  true,  false, false, true,  true,  false, false, true,
        true,  false, false, true,  true,  false, false, true,  true,  false, false, true,  true,
        false, false, true,  true,  false, false, true,  true,  false, false, true,  true,  false,
        false, true,  true,  false, false, true,  true,  false, false, true,  true,  false, false,
        true,  true,  false, false, true,  true,  false, false, true,  true,  false, false, true,
        true,  false, false, true,  true,  false, false, true,  true,  false, false, true,  true,
        false, false, true,  true,  false, false, true,  true,  false, false, true,  true,  false,
        false, true,  true,  false, false, true,  true,  false, false};
    uint32_t indices_exp[256] = {
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1,
        0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1,
        1, 1, 1, 1, 1, 1, 0, 2, 2, 0, 2, 0, 0, 2, 0, 3, 2, 0, 3, 0, 0, 2, 0, 2, 3, 0, 2, 0, 0,
        3, 0, 3, 3, 0, 3, 0, 0, 3, 1, 2, 2, 0, 2, 1, 0, 2, 0, 2, 2, 1, 2, 0, 1, 2, 1, 3, 2, 0,
        3, 1, 0, 2, 0, 3, 2, 1, 3, 0, 1, 2, 1, 2, 3, 0, 2, 1, 0, 3, 0, 2, 3, 1, 2, 0, 1, 3, 1,
        3, 3, 0, 3, 1, 0, 3, 0, 3, 3, 1, 3, 0, 1, 3, 1, 2, 2, 1, 2, 1, 1, 2, 1, 3, 2, 1, 3, 1,
        1, 2, 1, 2, 3, 1, 2, 1, 1, 3, 1, 3, 3, 1, 3, 1, 1, 3, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2,
        3, 2, 3, 2, 2, 2, 2, 3, 2, 3, 3, 2, 2, 2, 3, 2, 3, 3, 2, 3, 2, 2, 2, 3, 3, 3, 2, 2, 3,
        2, 3, 3, 2, 3, 3, 2, 3, 3, 2, 3, 3, 3, 3, 3, 2, 2, 3, 3, 3, 3, 3, 3, 3};
    QkComplex64 coeffs_exp[64] = {
        {0.5, 0.0},  {1.0, 0.0},  {1.0, 0.0},  {1.0, 0.0},  {1.0, 0.0},  {1.5, 0.0},  {1.5, 0.0},
        {1.5, 0.0},  {1.5, 0.0},  {2.0, 0.0},  {2.0, 0.0},  {2.5, 0.0},  {2.5, 0.0},  {2.5, 0.0},
        {2.5, 0.0},  {3.0, 0.0},  {5.5, 0.0},  {5.5, 0.0},  {6.0, 0.0},  {6.0, 0.0},  {6.0, 0.0},
        {6.0, 0.0},  {6.5, 0.0},  {6.5, 0.0},  {7.0, 0.0},  {7.0, 0.0},  {7.0, 0.0},  {7.0, 0.0},
        {7.5, 0.0},  {7.5, 0.0},  {7.5, 0.0},  {7.5, 0.0},  {7.5, 0.0},  {7.5, 0.0},  {7.5, 0.0},
        {7.5, 0.0},  {8.0, 0.0},  {8.0, 0.0},  {8.0, 0.0},  {8.0, 0.0},  {8.5, 0.0},  {8.5, 0.0},
        {9.0, 0.0},  {9.0, 0.0},  {9.0, 0.0},  {9.0, 0.0},  {9.5, 0.0},  {9.5, 0.0},  {-0.5, 0.0},
        {-1.0, 0.0}, {-1.0, 0.0}, {-1.0, 0.0}, {-1.0, 0.0}, {-1.5, 0.0}, {-1.5, 0.0}, {-1.5, 0.0},
        {-1.5, 0.0}, {-2.0, 0.0}, {-2.0, 0.0}, {-2.5, 0.0}, {-2.5, 0.0}, {-2.5, 0.0}, {-2.5, 0.0},
        {-3.0, 0.0}};
    uint32_t boundaries_exp[65] = {0,   4,   8,   12,  16,  20,  24,  28,  32,  36,  40,  44,  48,
                                   52,  56,  60,  64,  68,  72,  76,  80,  84,  88,  92,  96,  100,
                                   104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144, 148, 152,
                                   156, 160, 164, 168, 172, 176, 180, 184, 188, 192, 196, 200, 204,
                                   208, 212, 216, 220, 224, 228, 232, 236, 240, 244, 248, 252, 256};
    QfFermionOperator *expected = qf_ferm_op_new(num_terms, num_actions, coeffs_exp, actions_exp,
                                                 indices_exp, boundaries_exp);

    bool is_equal = qf_ferm_op_equal(op, expected);

    qf_ferm_op_free(op);
    qf_ferm_op_free(expected);

    if (!is_equal) {
        return EqualityError;
    }
    return Ok;
}

int test_electronic_integrals(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_ferm_op_from_1body_tril_spin_sym);
    num_failed += RUN_TEST(test_ferm_op_from_1body_tril_spin);
    num_failed += RUN_TEST(test_ferm_op_from_2body_tril_spin_sym);
    num_failed += RUN_TEST(test_ferm_op_from_2body_tril_spin);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
