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
#include <stdio.h>
#include <stdnoreturn.h>

static int test_new(void) {
    uint64_t num_terms = 3;
    uint64_t num_modes = 4;
    uint32_t modes[4] = {0, 1, 2, 3};
    QkComplex64 coeffs[3] = {{1.0, 0.0}, {-1.0, 0.0}, {0.0, -1.0}};
    uint32_t boundaries[4] = {0, 0, 2, 4};
    QfMajoranaOperator *op = qf_maj_op_new(num_terms, num_modes, coeffs, modes, boundaries);

    QfMajoranaOperator *expected = qf_maj_op_zero();
    QkComplex64 coeff0 = {1.0, 0.0};
    qf_maj_op_add_term(expected, 0, NULL, &coeff0);
    uint32_t modes1[2] = {0, 1};
    QkComplex64 coeff1 = {-1.0, 0.0};
    qf_maj_op_add_term(expected, 2, modes1, &coeff1);
    uint32_t modes2[2] = {2, 3};
    QkComplex64 coeff2 = {0.0, -1.0};
    qf_maj_op_add_term(expected, 2, modes2, &coeff2);

    bool is_equal = qf_maj_op_equal(op, expected);

    qf_maj_op_free(op);
    qf_maj_op_free(expected);

    if (!is_equal) {
        return EqualityError;
    }
    return Ok;
}

static int test_add(void) {
    QfMajoranaOperator *zero = qf_maj_op_zero();
    QfMajoranaOperator *one = qf_maj_op_one();

    QfMajoranaOperator *op = qf_maj_op_add(zero, one);

    bool is_equal = qf_maj_op_equal(op, one);

    qf_maj_op_free(op);

    if (!is_equal) {
        return EqualityError;
    }
    return Ok;
}

static int test_add_term(void) {
    QfMajoranaOperator *one = qf_maj_op_one();

    QfMajoranaOperator *op = qf_maj_op_zero();
    QkComplex64 coeff = {1.0, 0.0};

    qf_maj_op_add_term(op, 0, NULL, &coeff);

    bool is_equal = qf_maj_op_equal(op, one);

    qf_maj_op_free(op);

    if (!is_equal) {
        return EqualityError;
    }
    return Ok;
}

static int test_equiv_pos(void) {
    QfMajoranaOperator *op = qf_maj_op_zero();
    QkComplex64 coeff = {1e-7, 0.0};

    qf_maj_op_add_term(op, 0, NULL, &coeff);

    QfMajoranaOperator *zero = qf_maj_op_zero();

    bool is_equiv = qf_maj_op_equiv(op, zero, 1e-6);

    qf_maj_op_free(op);

    if (!is_equiv) {
        return EqualityError;
    }
    return Ok;
}

static int test_equiv_neg(void) {
    QfMajoranaOperator *op = qf_maj_op_zero();
    QkComplex64 coeff = {1e-7, 0.0};

    qf_maj_op_add_term(op, 0, NULL, &coeff);

    QfMajoranaOperator *zero = qf_maj_op_zero();

    bool is_not_equiv = !qf_maj_op_equiv(op, zero, 1e-8);

    qf_maj_op_free(op);

    if (!is_not_equiv) {
        return EqualityError;
    }
    return Ok;
}

static int test_mul(void) {
    QfMajoranaOperator *one = qf_maj_op_one();

    QkComplex64 coeff = {2.0, 0.0};

    QfMajoranaOperator *op = qf_maj_op_mul(one, &coeff);

    QfMajoranaOperator *expected = qf_maj_op_zero();
    qf_maj_op_add_term(expected, 0, NULL, &coeff);

    bool is_equal = qf_maj_op_equal(op, expected);

    qf_maj_op_free(op);

    if (!is_equal) {
        return EqualityError;
    }
    return Ok;
}

static int test_compose(void) {
    QfMajoranaOperator *op1 = qf_maj_op_zero();
    QkComplex64 coeff1a = {2.0, 0.0};
    qf_maj_op_add_term(op1, 0, NULL, &coeff1a);
    QkComplex64 coeff1b = {3.0, 0.0};
    uint32_t modes1b[2] = {0, 1};
    qf_maj_op_add_term(op1, 2, modes1b, &coeff1b);

    QfMajoranaOperator *op2 = qf_maj_op_zero();
    QkComplex64 coeff2a = {1.5, 0.0};
    qf_maj_op_add_term(op2, 0, NULL, &coeff2a);
    QkComplex64 coeff2b = {4.0, 0.0};
    uint32_t modes2b[2] = {1, 0};
    qf_maj_op_add_term(op2, 2, modes2b, &coeff2b);

    QfMajoranaOperator *result = qf_maj_op_compose(op1, op2);

    QfMajoranaOperator *expected = qf_maj_op_zero();
    QkComplex64 coeff_exp1 = {3.0, 0.0};
    qf_maj_op_add_term(expected, 0, NULL, &coeff_exp1);
    QkComplex64 coeff_exp2 = {8.0, 0.0};
    uint32_t modes_exp2[2] = {1, 0};
    qf_maj_op_add_term(expected, 2, modes_exp2, &coeff_exp2);
    QkComplex64 coeff_exp3 = {4.5, 0.0};
    uint32_t modes_exp3[2] = {0, 1};
    qf_maj_op_add_term(expected, 2, modes_exp3, &coeff_exp3);
    QkComplex64 coeff_exp4 = {12.0, 0.0};
    uint32_t modes_exp4[4] = {1, 0, 0, 1};
    qf_maj_op_add_term(expected, 4, modes_exp4, &coeff_exp4);

    bool is_equal = qf_maj_op_equal(result, expected);

    qf_maj_op_free(op1);
    qf_maj_op_free(op2);
    qf_maj_op_free(result);
    qf_maj_op_free(expected);

    if (!is_equal) {
        return EqualityError;
    }
    return Ok;
}

static int test_ichop(void) {
    QfMajoranaOperator *op = qf_maj_op_zero();
    QkComplex64 coeff = {1e-8, 0.0};
    qf_maj_op_add_term(op, 0, NULL, &coeff);

    qf_maj_op_ichop(op, 1e-6);

    QfMajoranaOperator *expected = qf_maj_op_zero();

    bool is_equal = qf_maj_op_equal(op, expected);

    qf_maj_op_free(op);
    qf_maj_op_free(expected);

    if (!is_equal) {
        return EqualityError;
    }
    return Ok;
}

static int test_simplify(void) {
    uint64_t num_terms = 5;
    uint64_t num_modes = 4;
    uint32_t modes[4] = {0, 0, 1, 1};
    QkComplex64 coeffs[5] = {{1e-10, 0.0}, {2.0, 0.0}, {3.0, 0.0}, {4.0, 0.0}, {-4.0, 0.0}};
    uint32_t boundaries[6] = {0, 0, 1, 2, 3, 4};
    QfMajoranaOperator *op = qf_maj_op_new(num_terms, num_modes, coeffs, modes, boundaries);

    QfMajoranaOperator *canon = qf_maj_op_simplify(op, 1e-8);

    QfMajoranaOperator *expected = qf_maj_op_zero();
    uint32_t modes_exp[1] = {0};
    QkComplex64 coeff = {5.0, 0.0};
    qf_maj_op_add_term(expected, 1, modes_exp, &coeff);

    bool is_equal = qf_maj_op_equiv(canon, expected, 1e-10);

    qf_maj_op_free(op);
    qf_maj_op_free(canon);
    qf_maj_op_free(expected);

    if (!is_equal) {
        return EqualityError;
    }
    return Ok;
}

static int test_simplify_vs_ichop(void) {
    uint64_t num_terms = 100000;
    uint64_t num_modes = 0;
    QkComplex64 coeffs[100000];
    uint32_t boundaries[100001];
    for (int i = 0; i < 100000; i++) {
        coeffs[i].re = 1e-5;
        coeffs[i].im = 0.0;
        boundaries[i] = 0;
    }
    boundaries[100000] = 0;
    QfMajoranaOperator *op = qf_maj_op_new(num_terms, num_modes, coeffs, NULL, boundaries);

    QfMajoranaOperator *canon = qf_maj_op_simplify(op, 1e-4);

    QfMajoranaOperator *one = qf_maj_op_one();
    bool canon_is_equal = qf_maj_op_equiv(canon, one, 1e-6);

    qf_maj_op_ichop(op, 1e-4);

    QfMajoranaOperator *zero = qf_maj_op_zero();
    bool ichop_is_equal = qf_maj_op_equiv(op, zero, 1e-6);

    qf_maj_op_free(op);
    qf_maj_op_free(canon);
    qf_maj_op_free(one);
    qf_maj_op_free(zero);

    bool is_equal = canon_is_equal && ichop_is_equal;

    if (!is_equal) {
        return EqualityError;
    }
    return Ok;
}

static int test_adjoint(void) {
    QfMajoranaOperator *op = qf_maj_op_zero();
    QkComplex64 coeff = {0.0, 1.0};
    qf_maj_op_add_term(op, 0, NULL, &coeff);

    QfMajoranaOperator *adjoint = qf_maj_op_adjoint(op);

    QfMajoranaOperator *expected = qf_maj_op_zero();
    QkComplex64 coeff_adj = {0.0, -1.0};
    qf_maj_op_add_term(expected, 0, NULL, &coeff_adj);

    bool is_equal = qf_maj_op_equal(adjoint, expected);

    qf_maj_op_free(op);
    qf_maj_op_free(adjoint);
    qf_maj_op_free(expected);

    if (!is_equal) {
        return EqualityError;
    }
    return Ok;
}

static int test_normal_ordered(void) {
    QfMajoranaOperator *op = qf_maj_op_zero();
    uint32_t modes[4] = {0, 2, 1, 3};
    QkComplex64 coeff = {1.0, 0.0};
    qf_maj_op_add_term(op, 4, modes, &coeff);

    QfMajoranaOperator *normal_ordered = qf_maj_op_normal_ordered(op, false);

    QkComplex64 coeff_minus = {-1.0, 0.0};
    QfMajoranaOperator *expected = qf_maj_op_zero();
    uint32_t modes_exp[4] = {3, 2, 1, 0};
    qf_maj_op_add_term(expected, 4, modes_exp, &coeff_minus);

    bool is_equal = qf_maj_op_equal(normal_ordered, expected);

    qf_maj_op_free(op);
    qf_maj_op_free(normal_ordered);
    qf_maj_op_free(expected);

    if (!is_equal) {
        return EqualityError;
    }
    return Ok;
}

static int test_is_hermitian(void) {
    QfMajoranaOperator *op = qf_maj_op_zero();
    uint32_t modes1[4] = {0, 1, 2, 3};
    QkComplex64 coeff1 = {0.0, 1.00001};
    qf_maj_op_add_term(op, 4, modes1, &coeff1);
    uint32_t modes2[4] = {3, 2, 1, 0};
    QkComplex64 coeff2 = {0.0, -1};
    qf_maj_op_add_term(op, 4, modes2, &coeff2);

    bool is_hermitian = qf_maj_op_is_hermitian(op, 1e-4);

    bool is_not_hermitian = qf_maj_op_is_hermitian(op, 1e-8);

    bool correct = is_hermitian && !is_not_hermitian;

    qf_maj_op_free(op);

    if (!correct) {
        return EqualityError;
    }
    return Ok;
}

static int test_many_body_order(void) {
    QfMajoranaOperator *op = qf_maj_op_zero();
    uint32_t modes[4] = {0, 1, 2, 3};
    QkComplex64 coeff = {1.0, 0.0};
    qf_maj_op_add_term(op, 4, modes, &coeff);

    uint32_t many_body_order = qf_maj_op_many_body_order(op);

    bool correct = many_body_order == 4;

    qf_maj_op_free(op);

    if (!correct) {
        return EqualityError;
    }
    return Ok;
}

static int test_is_even(void) {
    QfMajoranaOperator *op1 = qf_maj_op_zero();
    uint32_t modes1[2] = {0, 1};
    QkComplex64 coeff = {1.0, 0.0};
    qf_maj_op_add_term(op1, 2, modes1, &coeff);

    bool conserves = qf_maj_op_is_even(op1);

    QfMajoranaOperator *op2 = qf_maj_op_zero();
    uint32_t modes2[1] = {0};
    qf_maj_op_add_term(op2, 1, modes2, &coeff);

    bool not_conserves = qf_maj_op_is_even(op2);

    bool correct = conserves && !not_conserves;

    qf_maj_op_free(op1);
    qf_maj_op_free(op2);

    if (!correct) {
        return EqualityError;
    }
    return Ok;
}

static int test_len(void) {
    QfMajoranaOperator *op = qf_maj_op_zero();
    uint32_t modes[4] = {0, 1, 2, 3};
    QkComplex64 coeff = {1.0, 0.0};
    qf_maj_op_add_term(op, 4, modes, &coeff);

    size_t len = qf_maj_op_len(op);

    bool correct = len == 1;

    qf_maj_op_free(op);

    if (!correct) {
        return EqualityError;
    }
    return Ok;
}

int test_majorana_operator(void) {
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
    num_failed += RUN_TEST(test_is_even);
    num_failed += RUN_TEST(test_len);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
