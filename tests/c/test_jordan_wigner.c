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

static int test_mapping(void) {
    QfFermionOperator *hamil = qf_ferm_op_zero();

    QkComplex64 coeff_1body[4] = {
        {-1.2563390730032502, 0.0},
        {-0.4718960072811406, 0.0},
        {-1.2563390730032502, 0.0},
        {-0.4718960072811406, 0.0},
    };
    bool action_1body[8] = {true, false, true, false, true, false, true, false};
    uint32_t indices_1body[8] = {0, 0, 1, 1, 2, 2, 3, 3};
    for (int i = 0; i < 4; i++) {
        qf_ferm_op_add_term(hamil, 2, action_1body + 2 * i, indices_1body + 2 * i, &coeff_1body[i]);
    }

    QkComplex64 coeff_2body[10] = {
        {-0.4836505304710653, 0.0},  {-0.6757101548035165, 0.0},  {-0.6645817302552967, 0.0},
        {-0.18093119978423133, 0.0}, {-0.18093119978423133, 0.0}, {-0.18093119978423133, 0.0},
        {-0.18093119978423133, 0.0}, {-0.6645817302552967, 0.0},  {-0.6985737227320183, 0.0},
        {-0.4836505304710653, 0.0},
    };
    bool action_2body[40] = {true,  true,  false, false, true,  true,  false, false, true,  true,
                             false, false, true,  true,  false, false, true,  true,  false, false,
                             true,  true,  false, false, true,  true,  false, false, true,  true,
                             false, false, true,  true,  false, false, true,  true,  false, false};
    uint32_t indices_2body[40] = {0, 1, 0, 1, 0, 2, 0, 2, 0, 3, 0, 3, 0, 2, 1, 3, 0, 3, 1, 2,
                                  1, 2, 0, 3, 1, 3, 0, 2, 1, 2, 1, 2, 1, 3, 1, 3, 2, 3, 2, 3};
    for (int i = 0; i < 10; i++) {
        qf_ferm_op_add_term(hamil, 4, action_2body + 4 * i, indices_2body + 4 * i, &coeff_2body[i]);
    }

    QkObs *result = qf_jordan_wigner(hamil, 4);

    QkComplex64 coeffs[15] = {
        {-0.8105479805373266, 0.0}, {0.1721839326191555, 0.0},   {-0.22575349222402474, 0.0},
        {0.17218393261915543, 0.0}, {-0.22575349222402474, 0.0}, {0.12091263261776633, 0.0},
        {0.16892753870087912, 0.0}, {0.16614543256382416, 0.0},  {0.04523279994605783, 0.0},
        {0.04523279994605783, 0.0}, {0.04523279994605783, 0.0},  {0.04523279994605783, 0.0},
        {0.16614543256382416, 0.0}, {0.17464343068300459, 0.0},  {0.12091263261776633, 0.0},
    };
    QkBitTerm bits[32] = {
        QkBitTerm_Z, QkBitTerm_Z, QkBitTerm_Z, QkBitTerm_Z, QkBitTerm_Z, QkBitTerm_Z, QkBitTerm_Z,
        QkBitTerm_Z, QkBitTerm_Z, QkBitTerm_Z, QkBitTerm_Y, QkBitTerm_Y, QkBitTerm_Y, QkBitTerm_Y,
        QkBitTerm_Y, QkBitTerm_Y, QkBitTerm_X, QkBitTerm_X, QkBitTerm_X, QkBitTerm_X, QkBitTerm_Y,
        QkBitTerm_Y, QkBitTerm_X, QkBitTerm_X, QkBitTerm_X, QkBitTerm_X, QkBitTerm_Z, QkBitTerm_Z,
        QkBitTerm_Z, QkBitTerm_Z, QkBitTerm_Z, QkBitTerm_Z,
    };
    uint32_t indices[32] = {
        0, 1, 2, 3, 0, 1, 0, 2, 0, 3, 0, 1, 2, 3, 0, 1,
        2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 1, 2, 1, 3, 2, 3,
    };
    size_t boundaries[16] = {0, 0, 1, 2, 3, 4, 6, 8, 10, 14, 18, 22, 26, 28, 30, 32};

    QkObs *expected = qk_obs_new(4, 15, 32, coeffs, bits, indices, boundaries);

    QkComplex64 factor = {-1.0, 0.0};
    QkObs *diff = qk_obs_add(result, qk_obs_multiply(expected, &factor));
    QkObs *canon = qk_obs_canonicalize(diff, 1e-6);

    QkObs *zero = qk_obs_zero(4);

    bool is_equal = qk_obs_equal(canon, zero);

    qf_ferm_op_free(hamil);
    qk_obs_free(result);
    qk_obs_free(expected);

    if (!is_equal) {
        return EqualityError;
    }
    return Ok;
}

int test_jordan_wigner(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_mapping);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
