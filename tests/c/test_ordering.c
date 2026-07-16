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

static int test_ferm_op_canonical_order_sorts_terms(void) {
    // Two terms stored out of canonical order:
    //   term 0: a†_1 a_0  -> key [(1, true), (0, false)]
    //   term 1: a†_0 a_1  -> key [(0, true), (1, false)]
    // The mode-0-leading term must come first after reordering.
    uint64_t num_terms = 2;
    uint64_t num_actions = 4;
    bool actions[4] = {true, false, true, false};
    uint32_t modes[4] = {1, 0, 0, 1};
    QkComplex64 coeffs[2] = {{1.0, 0.0}, {2.0, 0.0}};
    uint32_t boundaries[3] = {0, 2, 4};
    QfFermionOperator *op =
        qf_ferm_op_new(num_terms, num_actions, coeffs, actions, modes, boundaries);

    // Expected: a†_0 a_1 (coeff 2) first, then a†_1 a_0 (coeff 1).
    QfFermionOperator *expected = qf_ferm_op_zero();
    bool actions_ab[2] = {true, false};
    uint32_t modes_a[2] = {0, 1};
    QkComplex64 coeff_a = {2.0, 0.0};
    qf_ferm_op_add_term(expected, 2, actions_ab, modes_a, &coeff_a);
    uint32_t modes_b[2] = {1, 0};
    QkComplex64 coeff_b = {1.0, 0.0};
    qf_ferm_op_add_term(expected, 2, actions_ab, modes_b, &coeff_b);

    QfFermionOperator *ordered = qf_ferm_op_canonical_order(op);

    bool is_equal = qf_ferm_op_equal(ordered, expected);

    qf_ferm_op_free(op);
    qf_ferm_op_free(expected);
    qf_ferm_op_free(ordered);

    if (!is_equal) {
        return EqualityError;
    }
    return Ok;
}

static int test_maj_op_canonical_order_sorts_terms(void) {
    // Terms stored as [ (2, 3), (0, 1) ]; canonical order sorts them to [ (0, 1), (2, 3) ].
    uint64_t num_terms = 2;
    uint64_t num_modes = 4;
    uint32_t modes[4] = {2, 3, 0, 1};
    QkComplex64 coeffs[2] = {{1.0, 0.0}, {2.0, 0.0}};
    uint32_t boundaries[3] = {0, 2, 4};
    QfMajoranaOperator *op = qf_maj_op_new(num_terms, num_modes, coeffs, modes, boundaries);

    QfMajoranaOperator *expected = qf_maj_op_zero();
    uint32_t modes_a[2] = {0, 1};
    QkComplex64 coeff_a = {2.0, 0.0};
    qf_maj_op_add_term(expected, 2, modes_a, &coeff_a);
    uint32_t modes_b[2] = {2, 3};
    QkComplex64 coeff_b = {1.0, 0.0};
    qf_maj_op_add_term(expected, 2, modes_b, &coeff_b);

    QfMajoranaOperator *ordered = qf_maj_op_canonical_order(op);

    bool is_equal = qf_maj_op_equal(ordered, expected);

    qf_maj_op_free(op);
    qf_maj_op_free(expected);
    qf_maj_op_free(ordered);

    if (!is_equal) {
        return EqualityError;
    }
    return Ok;
}

static int test_ferm_op_canonical_order_preserves_groups(void) {
    // term 0: a†_1 a_0 in group 0; term 1: a†_0 a_1 in group 1.
    uint64_t num_terms = 2;
    uint64_t num_actions = 4;
    bool actions[4] = {true, false, true, false};
    uint32_t modes[4] = {1, 0, 0, 1};
    QkComplex64 coeffs[2] = {{1.0, 0.0}, {2.0, 0.0}};
    uint32_t boundaries[3] = {0, 2, 4};
    QfFermionOperator *op =
        qf_ferm_op_new(num_terms, num_actions, coeffs, actions, modes, boundaries);
    uint32_t groups_in[2] = {0, 1};
    qf_ferm_op_set_groups(op, groups_in, num_terms);

    QfFermionOperator *ordered = qf_ferm_op_canonical_order(op);

    // a†_0 a_1 (group 1) sorts first and a†_1 a_0 (group 0) second, so the group tags travel along
    // to become {1, 0}.
    int result = Ok;
    if (!qf_ferm_op_has_groups(ordered)) {
        result = EqualityError;
    } else {
        uint32_t *groups_out;
        uint64_t groups_len;
        qf_ferm_op_get_groups(ordered, &groups_out, &groups_len);
        if (groups_len != 2 || groups_out[0] != 1 || groups_out[1] != 0) {
            result = EqualityError;
        }
    }

    qf_ferm_op_free(op);
    qf_ferm_op_free(ordered);

    return result;
}

int test_ordering(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_ferm_op_canonical_order_sorts_terms);
    num_failed += RUN_TEST(test_maj_op_canonical_order_sorts_terms);
    num_failed += RUN_TEST(test_ferm_op_canonical_order_preserves_groups);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
