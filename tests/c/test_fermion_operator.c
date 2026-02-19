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
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdnoreturn.h>

static int test_new(void) {
    uint64_t num_terms = 3;
    uint64_t num_actions = 4;
    bool actions[4] = {true, false, true, false};
    uint32_t indices[4] = {0, 1, 2, 3};
    QkComplex64 coeffs[3] = {{1.0, 0.0}, {-1.0, 0.0}, {0.0, -1.0}};
    uint32_t boundaries[4] = {0, 0, 2, 4};
    QfFermionOperator *op =
        qf_ferm_op_new(num_terms, num_actions, coeffs, actions, indices, boundaries);

    QfFermionOperator *expected = qf_ferm_op_zero();
    QkComplex64 coeff0 = {1.0, 0.0};
    qf_ferm_op_add_term(expected, 0, NULL, NULL, &coeff0);
    bool action1[2] = {true, false};
    uint32_t indices1[2] = {0, 1};
    QkComplex64 coeff1 = {-1.0, 0.0};
    qf_ferm_op_add_term(expected, 2, action1, indices1, &coeff1);
    bool action2[2] = {true, false};
    uint32_t indices2[2] = {2, 3};
    QkComplex64 coeff2 = {0.0, -1.0};
    qf_ferm_op_add_term(expected, 2, action2, indices2, &coeff2);

    bool is_equal = qf_ferm_op_equal(op, expected);

    qf_ferm_op_free(op);
    qf_ferm_op_free(expected);

    if (!is_equal) {
        return EqualityError;
    }
    return Ok;
}

static int test_add(void) {
    QfFermionOperator *zero = qf_ferm_op_zero();
    QfFermionOperator *one = qf_ferm_op_one();

    QfFermionOperator *op = qf_ferm_op_add(zero, one);

    bool is_equal = qf_ferm_op_equal(op, one);

    qf_ferm_op_free(op);

    if (!is_equal) {
        return EqualityError;
    }
    return Ok;
}

static int test_add_term(void) {
    QfFermionOperator *one = qf_ferm_op_one();

    QfFermionOperator *op = qf_ferm_op_zero();
    QkComplex64 coeff = {1.0, 0.0};

    qf_ferm_op_add_term(op, 0, NULL, NULL, &coeff);

    bool is_equal = qf_ferm_op_equal(op, one);

    qf_ferm_op_free(op);

    if (!is_equal) {
        return EqualityError;
    }
    return Ok;
}

static int test_equiv_pos(void) {
    QfFermionOperator *op = qf_ferm_op_zero();
    QkComplex64 coeff = {1e-7, 0.0};

    qf_ferm_op_add_term(op, 0, NULL, NULL, &coeff);

    QfFermionOperator *zero = qf_ferm_op_zero();

    bool is_equiv = qf_ferm_op_equiv(op, zero, 1e-6);

    qf_ferm_op_free(op);

    if (!is_equiv) {
        return EqualityError;
    }
    return Ok;
}

static int test_equiv_neg(void) {
    QfFermionOperator *op = qf_ferm_op_zero();
    QkComplex64 coeff = {1e-7, 0.0};

    qf_ferm_op_add_term(op, 0, NULL, NULL, &coeff);

    QfFermionOperator *zero = qf_ferm_op_zero();

    bool is_not_equiv = !qf_ferm_op_equiv(op, zero, 1e-8);

    qf_ferm_op_free(op);

    if (!is_not_equiv) {
        return EqualityError;
    }
    return Ok;
}

static int test_mul(void) {
    QfFermionOperator *one = qf_ferm_op_one();

    QkComplex64 coeff = {2.0, 0.0};

    QfFermionOperator *op = qf_ferm_op_mul(one, &coeff);

    QfFermionOperator *expected = qf_ferm_op_zero();
    qf_ferm_op_add_term(expected, 0, NULL, NULL, &coeff);

    bool is_equal = qf_ferm_op_equal(op, expected);

    qf_ferm_op_free(op);

    if (!is_equal) {
        return EqualityError;
    }
    return Ok;
}

static int test_compose(void) {
    uint64_t num_terms = 2;
    uint64_t num_actions = 2;
    bool actions[2] = {true, false};
    uint32_t boundaries[3] = {0, 0, 2};
    uint32_t indices1[2] = {0, 1};
    QkComplex64 coeffs1[2] = {{2.0, 0.0}, {3.0, 0.0}};
    QfFermionOperator *op1 =
        qf_ferm_op_new(num_terms, num_actions, coeffs1, actions, indices1, boundaries);

    uint32_t indices2[2] = {1, 0};
    QkComplex64 coeffs2[2] = {{1.5, 0.0}, {4.0, 0.0}};
    QfFermionOperator *op2 =
        qf_ferm_op_new(num_terms, num_actions, coeffs2, actions, indices2, boundaries);

    QfFermionOperator *result = qf_ferm_op_compose(op1, op2);

    num_terms = 4;
    num_actions = 8;
    bool actions_exp[8] = {true, false, true, false, true, false, true, false};
    uint32_t indices_exp[8] = {1, 0, 0, 1, 1, 0, 0, 1};
    QkComplex64 coeffs_exp[4] = {{3.0, 0.0}, {8.0, 0.0}, {4.5, 0.0}, {12.0, 0.0}};
    uint32_t boundaries_exp[5] = {0, 0, 2, 4, 8};
    QfFermionOperator *expected = qf_ferm_op_new(num_terms, num_actions, coeffs_exp, actions_exp,
                                                 indices_exp, boundaries_exp);

    bool is_equal = qf_ferm_op_equal(result, expected);

    qf_ferm_op_free(op1);
    qf_ferm_op_free(op2);
    qf_ferm_op_free(result);
    qf_ferm_op_free(expected);

    if (!is_equal) {
        return EqualityError;
    }
    return Ok;
}

static int test_ichop(void) {
    QfFermionOperator *op = qf_ferm_op_zero();
    QkComplex64 coeff = {1e-8, 0.0};
    qf_ferm_op_add_term(op, 0, NULL, NULL, &coeff);

    qf_ferm_op_ichop(op, 1e-6);

    QfFermionOperator *expected = qf_ferm_op_zero();

    bool is_equal = qf_ferm_op_equal(op, expected);

    qf_ferm_op_free(op);
    qf_ferm_op_free(expected);

    if (!is_equal) {
        return EqualityError;
    }
    return Ok;
}

static int test_simplify(void) {
    uint64_t num_terms = 5;
    uint64_t num_actions = 4;
    bool actions[4] = {true, true, false, false};
    uint32_t indices[4] = {0, 0, 1, 1};
    QkComplex64 coeffs[5] = {{1e-10, 0.0}, {2.0, 0.0}, {3.0, 0.0}, {4.0, 0.0}, {-4.0, 0.0}};
    uint32_t boundaries[6] = {0, 0, 1, 2, 3, 4};
    QfFermionOperator *op =
        qf_ferm_op_new(num_terms, num_actions, coeffs, actions, indices, boundaries);

    QfFermionOperator *canon = qf_ferm_op_simplify(op, 1e-8);

    QfFermionOperator *expected = qf_ferm_op_zero();
    bool actions_exp[1] = {true};
    uint32_t indices_exp[1] = {0};
    QkComplex64 coeff = {5.0, 0.0};
    qf_ferm_op_add_term(expected, 1, actions_exp, indices_exp, &coeff);

    bool is_equal = qf_ferm_op_equiv(canon, expected, 1e-10);

    qf_ferm_op_free(op);
    qf_ferm_op_free(canon);
    qf_ferm_op_free(expected);

    if (!is_equal) {
        return EqualityError;
    }
    return Ok;
}

static int test_simplify_vs_ichop(void) {
    uint64_t num_terms = 100000;
    uint64_t num_actions = 0;
    QkComplex64 coeffs[100000];
    uint32_t boundaries[100001];
    for (int i = 0; i < 100000; i++) {
        coeffs[i].re = 1e-5;
        coeffs[i].im = 0.0;
        boundaries[i] = 0;
    }
    boundaries[100000] = 0;
    QfFermionOperator *op = qf_ferm_op_new(num_terms, num_actions, coeffs, NULL, NULL, boundaries);

    QfFermionOperator *canon = qf_ferm_op_simplify(op, 1e-4);

    QfFermionOperator *one = qf_ferm_op_one();
    bool canon_is_equal = qf_ferm_op_equiv(canon, one, 1e-6);

    qf_ferm_op_ichop(op, 1e-4);

    QfFermionOperator *zero = qf_ferm_op_zero();
    bool ichop_is_equal = qf_ferm_op_equiv(op, zero, 1e-6);

    qf_ferm_op_free(op);
    qf_ferm_op_free(canon);
    qf_ferm_op_free(one);
    qf_ferm_op_free(zero);

    bool is_equal = canon_is_equal && ichop_is_equal;

    if (!is_equal) {
        return EqualityError;
    }
    return Ok;
}

static int test_adjoint(void) {
    QfFermionOperator *op = qf_ferm_op_zero();
    QkComplex64 coeff = {0.0, 1.0};
    qf_ferm_op_add_term(op, 0, NULL, NULL, &coeff);

    QfFermionOperator *adjoint = qf_ferm_op_adjoint(op);

    QfFermionOperator *expected = qf_ferm_op_zero();
    QkComplex64 coeff_adj = {0.0, -1.0};
    qf_ferm_op_add_term(expected, 0, NULL, NULL, &coeff_adj);

    bool is_equal = qf_ferm_op_equal(adjoint, expected);

    qf_ferm_op_free(op);
    qf_ferm_op_free(adjoint);
    qf_ferm_op_free(expected);

    if (!is_equal) {
        return EqualityError;
    }
    return Ok;
}

static int test_normal_ordered(void) {
    QfFermionOperator *op = qf_ferm_op_zero();
    bool action[4] = {false, true, false, true};
    uint32_t indices[4] = {1, 1, 0, 0};
    QkComplex64 coeff = {1.0, 0.0};
    qf_ferm_op_add_term(op, 4, action, indices, &coeff);

    QfFermionOperator *normal_ordered = qf_ferm_op_normal_ordered(op);

    uint64_t num_terms = 4;
    uint64_t num_actions = 8;
    bool actions_exp[8] = {true, false, true, false, true, true, false, false};
    uint32_t indices_exp[8] = {0, 0, 1, 1, 1, 0, 1, 0};
    QkComplex64 coeffs_exp[4] = {{1.0, 0.0}, {-1.0, 0.0}, {-1.0, 0.0}, {-1.0, 0.0}};
    uint32_t boundaries_exp[5] = {0, 0, 2, 4, 8};
    QfFermionOperator *expected = qf_ferm_op_new(num_terms, num_actions, coeffs_exp, actions_exp,
                                                 indices_exp, boundaries_exp);

    bool is_equal = qf_ferm_op_equiv(normal_ordered, expected, 1e-10);

    qf_ferm_op_free(op);
    qf_ferm_op_free(normal_ordered);
    qf_ferm_op_free(expected);

    if (!is_equal) {
        return EqualityError;
    }
    return Ok;
}

static int test_is_hermitian(void) {
    QfFermionOperator *op = qf_ferm_op_zero();
    bool action[2] = {true, false};
    uint32_t indices1[2] = {0, 1};
    QkComplex64 coeff1 = {0.0, 1.00001};
    qf_ferm_op_add_term(op, 2, action, indices1, &coeff1);
    uint32_t indices2[2] = {1, 0};
    QkComplex64 coeff2 = {0.0, -1};
    qf_ferm_op_add_term(op, 2, action, indices2, &coeff2);

    bool is_hermitian = qf_ferm_op_is_hermitian(op, 1e-4);

    bool is_not_hermitian = qf_ferm_op_is_hermitian(op, 1e-8);

    bool correct = is_hermitian && !is_not_hermitian;

    qf_ferm_op_free(op);

    if (!correct) {
        return EqualityError;
    }
    return Ok;
}

static int test_many_body_order(void) {
    QfFermionOperator *op = qf_ferm_op_zero();
    bool action[4] = {true, false, true, false};
    uint32_t indices[4] = {0, 1, 2, 3};
    QkComplex64 coeff = {1.0, 0.0};
    qf_ferm_op_add_term(op, 4, action, indices, &coeff);

    uint32_t many_body_order = qf_ferm_op_many_body_order(op);

    bool correct = many_body_order == 4;

    qf_ferm_op_free(op);

    if (!correct) {
        return EqualityError;
    }
    return Ok;
}

static int test_conserves_particle_number(void) {
    QfFermionOperator *op1 = qf_ferm_op_zero();
    bool action1[2] = {true, false};
    uint32_t indices1[2] = {0, 1};
    QkComplex64 coeff = {1.0, 0.0};
    qf_ferm_op_add_term(op1, 2, action1, indices1, &coeff);

    bool conserves = qf_ferm_op_conserves_particle_number(op1);

    QfFermionOperator *op2 = qf_ferm_op_zero();
    bool action2[1] = {true};
    uint32_t indices2[1] = {0};
    qf_ferm_op_add_term(op2, 1, action2, indices2, &coeff);

    bool not_conserves = qf_ferm_op_conserves_particle_number(op2);

    bool correct = conserves && !not_conserves;

    qf_ferm_op_free(op1);
    qf_ferm_op_free(op2);

    if (!correct) {
        return EqualityError;
    }
    return Ok;
}

static int test_len(void) {
    QfFermionOperator *op = qf_ferm_op_zero();
    bool action[4] = {true, false, true, false};
    uint32_t indices[4] = {0, 1, 2, 3};
    QkComplex64 coeff = {1.0, 0.0};
    qf_ferm_op_add_term(op, 4, action, indices, &coeff);

    size_t len = qf_ferm_op_len(op);

    bool correct = len == 1;

    qf_ferm_op_free(op);

    if (!correct) {
        return EqualityError;
    }
    return Ok;
}

int test_fermion_operator(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_new);
    num_failed += RUN_TEST(test_add);
    num_failed += RUN_TEST(test_add_term);
    num_failed += RUN_TEST(test_equiv_pos);
    num_failed += RUN_TEST(test_equiv_neg);
    num_failed += RUN_TEST(test_mul);
    num_failed += RUN_TEST(test_compose);
    num_failed += RUN_TEST(test_ichop);
    num_failed += RUN_TEST(test_simplify);
    num_failed += RUN_TEST(test_simplify_vs_ichop);
    num_failed += RUN_TEST(test_adjoint);
    num_failed += RUN_TEST(test_normal_ordered);
    num_failed += RUN_TEST(test_is_hermitian);
    num_failed += RUN_TEST(test_many_body_order);
    num_failed += RUN_TEST(test_conserves_particle_number);
    num_failed += RUN_TEST(test_len);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
