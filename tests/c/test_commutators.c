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

static int test_ferm_op_commutator(void) {
    QfFermionOperator *op1 = qf_ferm_op_zero();
    QkComplex64 coeff1 = {1.0, 0.0};
    bool action1[2] = {true, false};
    uint32_t indices1[2] = {0, 0};
    qf_ferm_op_add_term(op1, 2, action1, indices1, &coeff1);
    QfFermionOperator *op2 = qf_ferm_op_zero();
    QkComplex64 coeff2 = {2.0, 0.0};
    bool action2[2] = {false, true};
    uint32_t indices2[2] = {0, 0};
    qf_ferm_op_add_term(op2, 2, action2, indices2, &coeff2);

    QfFermionOperator *comm = qf_ferm_op_commutator(op1, op2);

    QfFermionOperator *normal = qf_ferm_op_normal_ordered(comm);
    QfFermionOperator *canon = qf_ferm_op_simplify(normal, 1e-8);

    qf_ferm_op_ichop(canon, 1e-8);

    QfFermionOperator *zero = qf_ferm_op_zero();
    bool is_equal = qf_ferm_op_equal(canon, zero);

    qf_ferm_op_free(op1);
    qf_ferm_op_free(op2);
    qf_ferm_op_free(zero);

    if (!is_equal) {
        return EqualityError;
    }
    return Ok;
}

static int test_ferm_op_anti_commutator(void) {
    QfFermionOperator *op1 = qf_ferm_op_zero();
    QkComplex64 coeff1 = {1.0, 0.0};
    bool action1[2] = {true, false};
    uint32_t indices1[2] = {0, 0};
    qf_ferm_op_add_term(op1, 2, action1, indices1, &coeff1);
    QfFermionOperator *op2 = qf_ferm_op_zero();
    QkComplex64 coeff2 = {2.0, 0.0};
    bool action2[2] = {false, true};
    uint32_t indices2[2] = {0, 0};
    qf_ferm_op_add_term(op2, 2, action2, indices2, &coeff2);

    QfFermionOperator *anti_comm = qf_ferm_op_anti_commutator(op1, op2);

    QfFermionOperator *normal = qf_ferm_op_normal_ordered(anti_comm);
    QfFermionOperator *canon = qf_ferm_op_simplify(normal, 1e-8);

    qf_ferm_op_ichop(canon, 1e-8);

    QfFermionOperator *zero = qf_ferm_op_zero();
    bool is_equal = qf_ferm_op_equal(canon, zero);

    qf_ferm_op_free(op1);
    qf_ferm_op_free(op2);
    qf_ferm_op_free(zero);

    if (!is_equal) {
        return EqualityError;
    }
    return Ok;
}

static int test_ferm_op_double_commutator(void) {
    QfFermionOperator *op1 = qf_ferm_op_zero();
    QkComplex64 coeff1 = {1.0, 0.0};
    bool action1[2] = {true, false};
    uint32_t indices1[2] = {0, 0};
    qf_ferm_op_add_term(op1, 2, action1, indices1, &coeff1);
    QfFermionOperator *op2 = qf_ferm_op_zero();
    QkComplex64 coeff2 = {2.0, 0.0};
    bool action2[2] = {false, true};
    uint32_t indices2[2] = {0, 0};
    qf_ferm_op_add_term(op2, 2, action2, indices2, &coeff2);
    QfFermionOperator *op3 = qf_ferm_op_zero();
    qf_ferm_op_add_term(op3, 2, action1, indices1, &coeff1);
    QkComplex64 coeff3 = {2.0, 0.5};
    qf_ferm_op_add_term(op3, 2, action2, indices2, &coeff3);

    QfFermionOperator *double_comm = qf_ferm_op_double_commutator(op1, op2, op3, false);

    QfFermionOperator *normal = qf_ferm_op_normal_ordered(double_comm);
    QfFermionOperator *canon = qf_ferm_op_simplify(normal, 1e-8);

    qf_ferm_op_ichop(canon, 1e-8);

    QfFermionOperator *zero = qf_ferm_op_zero();
    bool is_equal = qf_ferm_op_equal(canon, zero);

    qf_ferm_op_free(op1);
    qf_ferm_op_free(op2);
    qf_ferm_op_free(zero);

    if (!is_equal) {
        return EqualityError;
    }
    return Ok;
}

int test_commutators(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_ferm_op_commutator);
    num_failed += RUN_TEST(test_ferm_op_anti_commutator);
    num_failed += RUN_TEST(test_ferm_op_double_commutator);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
