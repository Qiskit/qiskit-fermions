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
    uint32_t modes[4] = {0, 1, 2, 3};
    QkComplex64 coeffs[3] = {{1.0, 0.0}, {-1.0, 0.0}, {0.0, -1.0}};
    uint32_t boundaries[4] = {0, 0, 2, 4};
    QfFermionOperator *op =
        qf_ferm_op_new(num_terms, num_actions, coeffs, actions, modes, boundaries);

    QfFermionOperator *expected = qf_ferm_op_zero();
    QkComplex64 coeff0 = {1.0, 0.0};
    qf_ferm_op_add_term(expected, 0, NULL, NULL, &coeff0);
    bool action1[2] = {true, false};
    uint32_t modes1[2] = {0, 1};
    QkComplex64 coeff1 = {-1.0, 0.0};
    qf_ferm_op_add_term(expected, 2, action1, modes1, &coeff1);
    bool action2[2] = {true, false};
    uint32_t modes2[2] = {2, 3};
    QkComplex64 coeff2 = {0.0, -1.0};
    qf_ferm_op_add_term(expected, 2, action2, modes2, &coeff2);

    bool is_equal = qf_ferm_op_equal(op, expected);

    qf_ferm_op_free(op);
    qf_ferm_op_free(expected);

    if (!is_equal) {
        return EqualityError;
    }
    return Ok;
}

static int test_getters(void) {
    uint64_t num_terms = 3;
    uint64_t num_actions = 4;
    bool actions[4] = {true, false, true, false};
    uint32_t modes[4] = {0, 1, 2, 3};
    QkComplex64 coeffs[3] = {{1.0, 0.0}, {-1.0, 0.0}, {0.0, -1.0}};
    uint32_t boundaries[4] = {0, 0, 2, 4};
    QfFermionOperator *op =
        qf_ferm_op_new(num_terms, num_actions, coeffs, actions, modes, boundaries);

    bool passed_all = true;

    QkComplex64 *coeffs_out;
    uint64_t coeffs_len;
    qf_ferm_op_get_coeffs(op, &coeffs_out, &coeffs_len);
    passed_all = passed_all && (coeffs_len == num_terms);

    for (uint64_t i = 0; i < num_terms; i++) {
        passed_all = passed_all && (coeffs_out[i].re == coeffs[i].re);
        passed_all = passed_all && (coeffs_out[i].im == coeffs[i].im);
    }

    bool *actions_out;
    uint64_t actions_len;
    qf_ferm_op_get_actions(op, &actions_out, &actions_len);
    passed_all = passed_all && (actions_len == num_actions);

    uint32_t *modes_out;
    uint64_t modes_len;
    qf_ferm_op_get_modes(op, &modes_out, &modes_len);
    passed_all = passed_all && (modes_len == num_actions);

    for (uint64_t i = 0; i < num_actions; i++) {
        passed_all = passed_all && (actions_out[i] == actions[i]);
        passed_all = passed_all && (modes_out[i] == modes[i]);
    }

    size_t *boundaries_out;
    uint64_t boundaries_len;
    qf_ferm_op_get_boundaries(op, &boundaries_out, &boundaries_len);
    passed_all = passed_all && (boundaries_len == 4);

    for (uint64_t i = 0; i < 4; i++) {
        passed_all = passed_all && (boundaries_out[i] == boundaries[i]);
    }

    qf_ferm_op_free(op);

    if (!passed_all) {
        return EqualityError;
    }
    return Ok;
}

static int test_get_support(void) {
    // Two terms deliberately share mode 2, and the modes are supplied out of order, so the
    // test covers both de-duplication and the ascending-order guarantee.
    bool actions[3] = {true, false, true};
    uint32_t modes[3] = {2, 0, 2};
    QkComplex64 coeffs[2] = {{1.0, 0.0}, {1.0, 0.0}};
    uint32_t boundaries[3] = {0, 2, 3};
    QfFermionOperator *op = qf_ferm_op_new(2, 3, coeffs, actions, modes, boundaries);

    bool passed_all = true;

    uint32_t num_support = qf_ferm_op_num_support(op);
    passed_all = passed_all && (num_support == 2);

    uint32_t support_out[2];
    qf_ferm_op_get_support(op, support_out);

    uint32_t expected[2] = {0, 2};
    for (uint32_t i = 0; i < num_support; i++) {
        passed_all = passed_all && (support_out[i] == expected[i]);
    }

    // An operator without terms has an empty support, so the buffer is never written to.
    QfFermionOperator *empty = qf_ferm_op_zero();
    passed_all = passed_all && (qf_ferm_op_num_support(empty) == 0);

    qf_ferm_op_free(op);
    qf_ferm_op_free(empty);

    if (!passed_all) {
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
    qf_ferm_op_free(zero);
    qf_ferm_op_free(one);

    if (!is_equal) {
        return EqualityError;
    }
    return Ok;
}

// `scaled_add` is the only subtraction the C API offers, so the -1 factor is the load-bearing
// case: it must agree with a hand-built difference.
static int test_scaled_add(void) {
    // A = a^dag_0 a_1
    bool actions[2] = {true, false};
    uint32_t modes_a[2] = {0, 1};
    uint32_t boundaries[2] = {0, 2};
    QkComplex64 coeff = {1.0, 0.0};
    QfFermionOperator *op_a = qf_ferm_op_new(1, 2, &coeff, actions, modes_a, boundaries);
    // B = a^dag_1 a_0
    uint32_t modes_b[2] = {1, 0};
    QfFermionOperator *op_b = qf_ferm_op_new(1, 2, &coeff, actions, modes_b, boundaries);

    bool passed_all = true;

    // A factor of -1 subtracts, so adding the result back to `op_b` recovers `op_a`.
    QkComplex64 minus_one = {-1.0, 0.0};
    QkComplex64 plus_one_early = {1.0, 0.0};
    QfFermionOperator *difference = qf_ferm_op_scaled_add(op_a, op_b, &minus_one);
    QfFermionOperator *restored = qf_ferm_op_scaled_add(difference, op_b, &plus_one_early);
    QfFermionOperator *simplified_restored = qf_ferm_op_simplify(restored, 1e-10);
    QfFermionOperator *simplified_a = qf_ferm_op_simplify(op_a, 1e-10);
    passed_all = passed_all && qf_ferm_op_equiv(simplified_restored, simplified_a, 1e-10);
    qf_ferm_op_free(simplified_restored);
    qf_ferm_op_free(simplified_a);

    // A factor of +1 is plain addition.
    QkComplex64 plus_one = {1.0, 0.0};
    QfFermionOperator *summed = qf_ferm_op_scaled_add(op_a, op_b, &plus_one);
    QfFermionOperator *added = qf_ferm_op_add(op_a, op_b);
    passed_all = passed_all && qf_ferm_op_equal(summed, added);

    // Scaling by zero still appends the terms; only their coefficients vanish.
    QkComplex64 zero_factor = {0.0, 0.0};
    QfFermionOperator *scaled_zero = qf_ferm_op_scaled_add(op_a, op_b, &zero_factor);
    passed_all = passed_all && (qf_ferm_op_len(scaled_zero) == qf_ferm_op_len(added));
    QfFermionOperator *chopped = qf_ferm_op_simplify(scaled_zero, 1e-10);
    passed_all = passed_all && (qf_ferm_op_len(chopped) == qf_ferm_op_len(op_a));

    // Appending terms drops the grouping, exactly as `add` does.
    uint32_t groups_in[1] = {0};
    qf_ferm_op_set_groups(op_a, groups_in, 1);
    QfFermionOperator *from_grouped = qf_ferm_op_scaled_add(op_a, op_b, &plus_one);
    passed_all = passed_all && !qf_ferm_op_has_groups(from_grouped);
    qf_ferm_op_del_groups(op_a);

    qf_ferm_op_free(op_a);
    qf_ferm_op_free(op_b);
    qf_ferm_op_free(difference);
    qf_ferm_op_free(restored);
    qf_ferm_op_free(summed);
    qf_ferm_op_free(added);
    qf_ferm_op_free(scaled_zero);
    qf_ferm_op_free(chopped);
    qf_ferm_op_free(from_grouped);

    if (!passed_all) {
        return EqualityError;
    }
    return Ok;
}

// Each in-place operation must agree with its out-of-place counterpart. The group handling is
// asserted in both directions, because the three do *not* behave alike: the two additions change
// the number of terms and therefore drop the grouping, whereas scaling the coefficients keeps it.
static int test_inplace_arithmetic(void) {
    bool actions[2] = {true, false};
    uint32_t modes_a[2] = {0, 1};
    uint32_t boundaries[2] = {0, 2};
    QkComplex64 coeff = {1.0, 0.0};
    QfFermionOperator *op_a = qf_ferm_op_new(1, 2, &coeff, actions, modes_a, boundaries);
    uint32_t modes_b[2] = {1, 0};
    QfFermionOperator *op_b = qf_ferm_op_new(1, 2, &coeff, actions, modes_b, boundaries);
    QfFermionOperator *scratch = qf_ferm_op_new(1, 2, &coeff, actions, modes_a, boundaries);

    bool passed_all = true;

    // add_inplace == add
    QfFermionOperator *expected_sum = qf_ferm_op_add(op_a, op_b);
    qf_ferm_op_add_inplace(scratch, op_b);
    passed_all = passed_all && qf_ferm_op_equal(scratch, expected_sum);

    // scaled_add_inplace == scaled_add
    QkComplex64 factor = {-2.0, 0.5};
    QfFermionOperator *expected_scaled = qf_ferm_op_scaled_add(op_a, op_b, &factor);
    QfFermionOperator *scaled = qf_ferm_op_new(1, 2, &coeff, actions, modes_a, boundaries);
    qf_ferm_op_scaled_add_inplace(scaled, op_b, &factor);
    passed_all = passed_all && qf_ferm_op_equal(scaled, expected_scaled);

    // mul_inplace == mul
    QkComplex64 scalar = {0.0, 3.0};
    QfFermionOperator *expected_scalar = qf_ferm_op_mul(op_a, &scalar);
    QfFermionOperator *multiplied = qf_ferm_op_new(1, 2, &coeff, actions, modes_a, boundaries);
    qf_ferm_op_mul_inplace(multiplied, &scalar);
    passed_all = passed_all && qf_ferm_op_equal(multiplied, expected_scalar);

    // The two appending operations drop the grouping, ...
    QfFermionOperator *grouped_add = qf_ferm_op_new(1, 2, &coeff, actions, modes_a, boundaries);
    uint32_t groups_in[1] = {0};
    qf_ferm_op_set_groups(grouped_add, groups_in, 1);
    qf_ferm_op_add_inplace(grouped_add, op_b);
    passed_all = passed_all && !qf_ferm_op_has_groups(grouped_add);

    QfFermionOperator *grouped_scaled = qf_ferm_op_new(1, 2, &coeff, actions, modes_a, boundaries);
    qf_ferm_op_set_groups(grouped_scaled, groups_in, 1);
    qf_ferm_op_scaled_add_inplace(grouped_scaled, op_b, &factor);
    passed_all = passed_all && !qf_ferm_op_has_groups(grouped_scaled);

    // ... while scaling the coefficients keeps it.
    QfFermionOperator *grouped_mul = qf_ferm_op_new(1, 2, &coeff, actions, modes_a, boundaries);
    qf_ferm_op_set_groups(grouped_mul, groups_in, 1);
    qf_ferm_op_mul_inplace(grouped_mul, &scalar);
    passed_all = passed_all && qf_ferm_op_has_groups(grouped_mul);

    qf_ferm_op_free(op_a);
    qf_ferm_op_free(op_b);
    qf_ferm_op_free(scratch);
    qf_ferm_op_free(expected_sum);
    qf_ferm_op_free(expected_scaled);
    qf_ferm_op_free(scaled);
    qf_ferm_op_free(expected_scalar);
    qf_ferm_op_free(multiplied);
    qf_ferm_op_free(grouped_add);
    qf_ferm_op_free(grouped_scaled);
    qf_ferm_op_free(grouped_mul);

    if (!passed_all) {
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
    qf_ferm_op_free(one);

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
    qf_ferm_op_free(zero);

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
    qf_ferm_op_free(zero);

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
    qf_ferm_op_free(one);
    qf_ferm_op_free(expected);

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
    uint32_t modes1[2] = {0, 1};
    QkComplex64 coeffs1[2] = {{2.0, 0.0}, {3.0, 0.0}};
    QfFermionOperator *op1 =
        qf_ferm_op_new(num_terms, num_actions, coeffs1, actions, modes1, boundaries);

    uint32_t modes2[2] = {1, 0};
    QkComplex64 coeffs2[2] = {{1.5, 0.0}, {4.0, 0.0}};
    QfFermionOperator *op2 =
        qf_ferm_op_new(num_terms, num_actions, coeffs2, actions, modes2, boundaries);

    QfFermionOperator *result = qf_ferm_op_compose(op1, op2);

    num_terms = 4;
    num_actions = 8;
    bool actions_exp[8] = {true, false, true, false, true, false, true, false};
    uint32_t modes_exp[8] = {1, 0, 0, 1, 1, 0, 0, 1};
    QkComplex64 coeffs_exp[4] = {{3.0, 0.0}, {8.0, 0.0}, {4.5, 0.0}, {12.0, 0.0}};
    uint32_t boundaries_exp[5] = {0, 0, 2, 4, 8};
    QfFermionOperator *expected =
        qf_ferm_op_new(num_terms, num_actions, coeffs_exp, actions_exp, modes_exp, boundaries_exp);

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

// Pins the operand order of `qf_ferm_op_compose`, which its docstring documents as
// `left.compose(right) == right @ left`, i.e. `right` is applied first. The operands are chosen so
// that the two orders give *different* results: with `A = a^dag_0 a_1` and `B = a^dag_1 a_0`,
// normal ordering turns `A.compose(B)` into `n_1 + ...` but the opposite order into `n_0 + ...`. A
// commuting pair (such as the `one`/`zero` operators used elsewhere) would pass under either
// convention and prove nothing.
static int test_compose_operand_order(void) {
    bool actions[2] = {true, false};
    uint32_t boundaries[2] = {0, 2};
    QkComplex64 coeff = {1.0, 0.0};

    // A = a^dag_0 a_1
    uint32_t modes_a[2] = {0, 1};
    QfFermionOperator *op_a = qf_ferm_op_new(1, 2, &coeff, actions, modes_a, boundaries);
    // B = a^dag_1 a_0
    uint32_t modes_b[2] = {1, 0};
    QfFermionOperator *op_b = qf_ferm_op_new(1, 2, &coeff, actions, modes_b, boundaries);

    QfFermionOperator *composed = qf_ferm_op_compose(op_a, op_b);
    QfFermionOperator *ordered = qf_ferm_op_normal_ordered(composed, NULL);
    QfFermionOperator *actual = qf_ferm_op_simplify(ordered, 1e-10);

    // The expected result of `A.compose(B)`: n_1 + a^dag_1 a^dag_0 a_1 a_0.
    bool actions_exp[6] = {true, false, true, true, false, false};
    uint32_t modes_exp[6] = {1, 1, 1, 0, 1, 0};
    QkComplex64 coeffs_exp[2] = {{1.0, 0.0}, {1.0, 0.0}};
    uint32_t boundaries_exp[3] = {0, 2, 6};
    QfFermionOperator *expected =
        qf_ferm_op_new(2, 6, coeffs_exp, actions_exp, modes_exp, boundaries_exp);

    bool correct_order = qf_ferm_op_equiv(actual, expected, 1e-10);

    // The swapped order must *not* agree, otherwise this test could not tell them apart.
    QfFermionOperator *swapped = qf_ferm_op_compose(op_b, op_a);
    QfFermionOperator *swapped_ordered = qf_ferm_op_normal_ordered(swapped, NULL);
    QfFermionOperator *swapped_actual = qf_ferm_op_simplify(swapped_ordered, 1e-10);
    bool orders_differ = !qf_ferm_op_equiv(swapped_actual, expected, 1e-10);

    qf_ferm_op_free(op_a);
    qf_ferm_op_free(op_b);
    qf_ferm_op_free(composed);
    qf_ferm_op_free(ordered);
    qf_ferm_op_free(actual);
    qf_ferm_op_free(expected);
    qf_ferm_op_free(swapped);
    qf_ferm_op_free(swapped_ordered);
    qf_ferm_op_free(swapped_actual);

    if (!correct_order || !orders_differ) {
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
    uint32_t modes[4] = {0, 0, 1, 1};
    QkComplex64 coeffs[5] = {{1e-10, 0.0}, {2.0, 0.0}, {3.0, 0.0}, {4.0, 0.0}, {-4.0, 0.0}};
    uint32_t boundaries[6] = {0, 0, 1, 2, 3, 4};
    QfFermionOperator *op =
        qf_ferm_op_new(num_terms, num_actions, coeffs, actions, modes, boundaries);

    QfFermionOperator *canon = qf_ferm_op_simplify(op, 1e-8);

    QfFermionOperator *expected = qf_ferm_op_zero();
    bool actions_exp[1] = {true};
    uint32_t modes_exp[1] = {0};
    QkComplex64 coeff = {5.0, 0.0};
    qf_ferm_op_add_term(expected, 1, actions_exp, modes_exp, &coeff);

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
    // Both arrays are heap-allocated: `boundaries` holds `num_terms + 1` entries, which at this
    // size is ~400 KB and overflows the default 1 MB stack on some platforms (notably MSVC) if
    // declared as a local array. The term count itself is load-bearing and cannot simply be
    // reduced: the test needs 100000 coefficients of 1e-5 so that they sum to exactly 1.0 while
    // each one stays below the 1e-4 chop tolerance, which is what distinguishes `simplify` from
    // `ichop` here.
    QkComplex64 *coeffs = (QkComplex64 *)malloc(num_terms * sizeof(QkComplex64));
    uint32_t *boundaries = (uint32_t *)malloc((num_terms + 1) * sizeof(uint32_t));
    for (uint64_t i = 0; i < num_terms; i++) {
        coeffs[i].re = 1e-5;
        coeffs[i].im = 0.0;
        boundaries[i] = 0;
    }
    boundaries[num_terms] = 0;
    QfFermionOperator *op = qf_ferm_op_new(num_terms, num_actions, coeffs, NULL, NULL, boundaries);

    QfFermionOperator *canon = qf_ferm_op_simplify(op, 1e-4);

    QfFermionOperator *one = qf_ferm_op_one();
    bool canon_is_equal = qf_ferm_op_equiv(canon, one, 1e-6);

    qf_ferm_op_ichop(op, 1e-4);

    QfFermionOperator *zero = qf_ferm_op_zero();
    bool ichop_is_equal = qf_ferm_op_equiv(op, zero, 1e-6);

    free(coeffs);
    free(boundaries);
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
    bool action[4] = {false, false, true, true};
    uint32_t modes[4] = {0, 1, 0, 1};
    QkComplex64 coeff = {1.0, 0.0};
    qf_ferm_op_add_term(op, 4, action, modes, &coeff);

    QfFermionOperator *normal_ordered = qf_ferm_op_normal_ordered(op, NULL);

    uint64_t num_terms = 4;
    uint64_t num_actions = 8;
    bool actions_exp[8] = {true, true, false, false, true, false, true, false};
    uint32_t modes_exp[8] = {1, 0, 1, 0, 0, 0, 1, 1};
    QkComplex64 coeffs_exp[4] = {{1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {-1.0, 0.0}};
    uint32_t boundaries_exp[5] = {0, 4, 6, 8, 8};
    QfFermionOperator *expected =
        qf_ferm_op_new(num_terms, num_actions, coeffs_exp, actions_exp, modes_exp, boundaries_exp);

    bool is_equal = qf_ferm_op_equiv(normal_ordered, expected, 1e-10);

    qf_ferm_op_free(op);
    qf_ferm_op_free(normal_ordered);
    qf_ferm_op_free(expected);

    if (!is_equal) {
        return EqualityError;
    }
    return Ok;
}

static int test_sandwich_ordered_true(void) {
    QfFermionOperator *op = qf_ferm_op_zero();
    bool action[4] = {false, false, true, true};
    uint32_t modes[4] = {1, 0, 0, 1};
    QkComplex64 coeff = {1.0, 0.0};
    qf_ferm_op_add_term(op, 4, action, modes, &coeff);

    bool sandwich = true;
    QfFermionOperator *sandwich_ordered = qf_ferm_op_normal_ordered(op, &sandwich);

    uint64_t num_terms = 4;
    uint64_t num_actions = 8;
    bool actions_exp[8] = {true, true, false, false, true, false, true, false};
    uint32_t modes_exp[8] = {0, 1, 1, 0, 0, 0, 1, 1};
    QkComplex64 coeffs_exp[4] = {{1.0, 0.0}, {-1.0, 0.0}, {-1.0, 0.0}, {1.0, 0.0}};
    uint32_t boundaries_exp[5] = {0, 4, 6, 8, 8};
    QfFermionOperator *expected =
        qf_ferm_op_new(num_terms, num_actions, coeffs_exp, actions_exp, modes_exp, boundaries_exp);

    bool is_equal = qf_ferm_op_equiv(sandwich_ordered, expected, 1e-10);

    qf_ferm_op_free(op);
    qf_ferm_op_free(sandwich_ordered);
    qf_ferm_op_free(expected);

    if (!is_equal) {
        return EqualityError;
    }
    return Ok;
}

static int test_sandwich_ordered_false(void) {
    QfFermionOperator *op = qf_ferm_op_zero();
    bool action[4] = {false, false, true, true};
    uint32_t modes[4] = {0, 1, 1, 0};
    QkComplex64 coeff = {1.0, 0.0};
    qf_ferm_op_add_term(op, 4, action, modes, &coeff);

    bool sandwich = false;
    QfFermionOperator *sandwich_ordered = qf_ferm_op_normal_ordered(op, &sandwich);

    uint64_t num_terms = 4;
    uint64_t num_actions = 8;
    bool actions_exp[8] = {true, true, false, false, true, false, true, false};
    uint32_t modes_exp[8] = {1, 0, 0, 1, 1, 1, 0, 0};
    QkComplex64 coeffs_exp[4] = {{1.0, 0.0}, {-1.0, 0.0}, {-1.0, 0.0}, {1.0, 0.0}};
    uint32_t boundaries_exp[5] = {0, 4, 6, 8, 8};
    QfFermionOperator *expected =
        qf_ferm_op_new(num_terms, num_actions, coeffs_exp, actions_exp, modes_exp, boundaries_exp);

    bool is_equal = qf_ferm_op_equiv(sandwich_ordered, expected, 1e-10);

    qf_ferm_op_free(op);
    qf_ferm_op_free(sandwich_ordered);
    qf_ferm_op_free(expected);

    if (!is_equal) {
        return EqualityError;
    }
    return Ok;
}

static int test_is_hermitian(void) {
    QfFermionOperator *op = qf_ferm_op_zero();
    bool action[2] = {true, false};
    uint32_t modes1[2] = {0, 1};
    QkComplex64 coeff1 = {0.0, 1.00001};
    qf_ferm_op_add_term(op, 2, action, modes1, &coeff1);
    uint32_t modes2[2] = {1, 0};
    QkComplex64 coeff2 = {0.0, -1};
    qf_ferm_op_add_term(op, 2, action, modes2, &coeff2);

    bool is_hermitian = qf_ferm_op_is_hermitian(op, 1e-4);

    bool is_not_hermitian = qf_ferm_op_is_hermitian(op, 1e-8);

    bool correct = is_hermitian && !is_not_hermitian;

    qf_ferm_op_free(op);

    if (!correct) {
        return EqualityError;
    }
    return Ok;
}

static int test_max_rank(void) {
    QfFermionOperator *op = qf_ferm_op_zero();
    bool action[4] = {true, false, true, false};
    uint32_t modes[4] = {0, 1, 2, 3};
    QkComplex64 coeff = {1.0, 0.0};
    qf_ferm_op_add_term(op, 4, action, modes, &coeff);

    uint32_t max_rank = qf_ferm_op_max_rank(op);

    bool correct = max_rank == 4;

    qf_ferm_op_free(op);

    if (!correct) {
        return EqualityError;
    }
    return Ok;
}

static int test_conserves_particle_number(void) {
    QfFermionOperator *op1 = qf_ferm_op_zero();
    bool action1[2] = {true, false};
    uint32_t modes1[2] = {0, 1};
    QkComplex64 coeff = {1.0, 0.0};
    qf_ferm_op_add_term(op1, 2, action1, modes1, &coeff);

    bool conserves = qf_ferm_op_conserves_particle_number(op1);

    QfFermionOperator *op2 = qf_ferm_op_zero();
    bool action2[1] = {true};
    uint32_t modes2[1] = {0};
    qf_ferm_op_add_term(op2, 1, action2, modes2, &coeff);

    bool not_conserves = qf_ferm_op_conserves_particle_number(op2);

    bool correct = conserves && !not_conserves;

    qf_ferm_op_free(op1);
    qf_ferm_op_free(op2);

    if (!correct) {
        return EqualityError;
    }
    return Ok;
}

// The interesting case for `conserves_sector` is the one that distinguishes it from
// `conserves_particle_number`: a term that balances globally but not within the individual blocks.
static int test_conserves_sector(void) {
    // Hops an electron from the beta block [2, 4) into the alpha block [0, 2).
    QfFermionOperator *op = qf_ferm_op_zero();
    bool actions[2] = {true, false};
    uint32_t modes[2] = {0, 2};
    QkComplex64 coeff = {1.0, 0.0};
    qf_ferm_op_add_term(op, 2, actions, modes, &coeff);

    bool passed_all = true;

    // Treating all four modes as one block is just particle-number conservation.
    uint32_t one_block[1] = {4};
    passed_all = passed_all && qf_ferm_op_conserves_sector(op, one_block, 1);

    // A NULL block array means the same thing, and must agree with the dedicated function.
    passed_all = passed_all && qf_ferm_op_conserves_sector(op, NULL, 0);
    passed_all = passed_all && (qf_ferm_op_conserves_sector(op, NULL, 0) ==
                                qf_ferm_op_conserves_particle_number(op));

    // Splitting into two spin blocks is what this term violates.
    uint32_t spin_blocks[2] = {2, 2};
    passed_all = passed_all && !qf_ferm_op_conserves_sector(op, spin_blocks, 2);

    // A number operator stays within its block, so it conserves both.
    QfFermionOperator *num_op = qf_ferm_op_zero();
    bool num_actions[2] = {true, false};
    uint32_t num_modes[2] = {0, 0};
    qf_ferm_op_add_term(num_op, 2, num_actions, num_modes, &coeff);

    passed_all = passed_all && qf_ferm_op_conserves_sector(num_op, spin_blocks, 2);

    // A term acting beyond the last block can never balance, since no block covers that mode.
    uint32_t short_block[1] = {2};
    passed_all = passed_all && !qf_ferm_op_conserves_sector(op, short_block, 1);

    qf_ferm_op_free(op);
    qf_ferm_op_free(num_op);

    if (!passed_all) {
        return EqualityError;
    }
    return Ok;
}

static int test_len(void) {
    QfFermionOperator *op = qf_ferm_op_zero();
    bool action[4] = {true, false, true, false};
    uint32_t modes[4] = {0, 1, 2, 3};
    QkComplex64 coeff = {1.0, 0.0};
    qf_ferm_op_add_term(op, 4, action, modes, &coeff);

    size_t len = qf_ferm_op_len(op);

    bool correct = len == 1;

    qf_ferm_op_free(op);

    if (!correct) {
        return EqualityError;
    }
    return Ok;
}

static int test_relabel_modes(void) {
    QfFermionOperator *op = qf_ferm_op_zero();
    QkComplex64 coeff = {1.0, 0.0};
    bool action1[2] = {true, false};
    uint32_t modes1[2] = {0, 1};
    qf_ferm_op_add_term(op, 2, action1, modes1, &coeff);
    bool action2[4] = {true, false, true, false};
    uint32_t modes2[4] = {0, 0, 2, 3};
    qf_ferm_op_add_term(op, 4, action2, modes2, &coeff);

    uint32_t permutation[4] = {4, 2, 5, 3};

    QfExitCode exit = qf_ferm_op_relabel_modes(op, 4, permutation);

    if (exit != QfExitCode_Success) {
        qf_ferm_op_free(op);
        return RuntimeError;
    }

    uint64_t num_terms = 2;
    uint64_t num_actions = 6;
    bool actions_exp[6] = {true, false, true, false, true, false};
    uint32_t modes_exp[6] = {4, 2, 4, 4, 5, 3};
    QkComplex64 coeffs_exp[2] = {{1.0, 0.0}, {1.0, 0.0}};
    uint32_t boundaries_exp[3] = {0, 2, 6};
    QfFermionOperator *expected =
        qf_ferm_op_new(num_terms, num_actions, coeffs_exp, actions_exp, modes_exp, boundaries_exp);

    bool is_equal = qf_ferm_op_equal(op, expected);

    qf_ferm_op_free(op);
    qf_ferm_op_free(expected);

    if (!is_equal) {
        return EqualityError;
    }
    return Ok;
}

static int test_relabel_modes_duplicate_err(void) {
    QfFermionOperator *op = qf_ferm_op_zero();
    QkComplex64 coeff = {1.0, 0.0};
    bool action1[2] = {true, false};
    uint32_t modes1[2] = {0, 1};
    qf_ferm_op_add_term(op, 2, action1, modes1, &coeff);
    bool action2[4] = {true, false, true, false};
    uint32_t modes2[4] = {0, 0, 2, 3};
    qf_ferm_op_add_term(op, 4, action2, modes2, &coeff);

    uint32_t permutation[4] = {4, 4, 5, 3};

    QfExitCode exit = qf_ferm_op_relabel_modes(op, 4, permutation);

    qf_ferm_op_free(op);

    return exit == QfExitCode_DuplicateIndexError ? Ok : EqualityError;
}

static int test_relabel_modes_too_small_err(void) {
    QfFermionOperator *op = qf_ferm_op_zero();
    QkComplex64 coeff = {1.0, 0.0};
    bool action1[2] = {true, false};
    uint32_t modes1[2] = {0, 1};
    qf_ferm_op_add_term(op, 2, action1, modes1, &coeff);
    bool action2[4] = {true, false, true, false};
    uint32_t modes2[4] = {0, 0, 2, 3};
    qf_ferm_op_add_term(op, 4, action2, modes2, &coeff);

    uint32_t permutation[4] = {4, 2, 5};

    QfExitCode exit = qf_ferm_op_relabel_modes(op, 3, permutation);

    qf_ferm_op_free(op);

    return exit == QfExitCode_IndexError ? Ok : EqualityError;
}

static int test_groups(void) {
    uint64_t num_terms = 4;
    uint64_t num_actions = 8;
    bool actions[8] = {true, false, true, false, true, false, true, false};
    uint32_t modes[8] = {0, 1, 2, 3, 1, 0, 3, 2};
    QkComplex64 coeffs[4] = {{1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}};
    uint32_t boundaries[5] = {0, 2, 4, 6, 8};
    QfFermionOperator *op =
        qf_ferm_op_new(num_terms, num_actions, coeffs, actions, modes, boundaries);

    bool has_no_groups = !qf_ferm_op_has_groups(op);

    uint32_t groups_in[4] = {0, 1, 0, 1};

    qf_ferm_op_set_groups(op, groups_in, num_terms);

    bool has_some_groups = qf_ferm_op_has_groups(op);

    uint32_t num_groups = qf_ferm_op_num_groups(op);

    bool correct_num_groups = num_groups == 2;

    QfFermionOperator *group_ops[2];

    qf_ferm_op_split_out_groups(op, NULL, 0, group_ops);

    uint32_t boundaries_group[3] = {0, 2, 4};
    QkComplex64 coeffs_group[2] = {{1.0, 0.0}, {1.0, 0.0}};
    bool actions_group[4] = {true, false, true, false};
    uint32_t modes_g0[4] = {0, 1, 1, 0};
    QfFermionOperator *group0 =
        qf_ferm_op_new(2, 4, coeffs_group, actions_group, modes_g0, boundaries_group);
    uint32_t modes_g1[4] = {2, 3, 3, 2};
    QfFermionOperator *group1 =
        qf_ferm_op_new(2, 4, coeffs_group, actions_group, modes_g1, boundaries_group);

    bool correct_group0 = qf_ferm_op_equiv(group_ops[0], group0, 1e-10);
    bool correct_group1 = qf_ferm_op_equiv(group_ops[1], group1, 1e-10);

    uint32_t group_indices[2] = {1, 1};
    QfFermionOperator *group_ops_indexed[2];
    qf_ferm_op_split_out_groups(op, group_indices, 2, group_ops_indexed);

    bool correct_indexed0 = qf_ferm_op_equiv(group_ops_indexed[0], group1, 1e-10);
    bool correct_indexed1 = qf_ferm_op_equiv(group_ops_indexed[1], group1, 1e-10);

    qf_ferm_op_free(group_ops_indexed[0]);
    qf_ferm_op_free(group_ops_indexed[1]);

    uint32_t *groups_out;
    uint64_t groups_len;

    qf_ferm_op_get_groups(op, &groups_out, &groups_len);

    bool correct_groups_len = groups_len == num_terms;
    bool correct_groups_out0 = groups_out[0] == 0;
    bool correct_groups_out1 = groups_out[1] == 1;
    bool correct_groups_out2 = groups_out[2] == 0;
    bool correct_groups_out3 = groups_out[3] == 1;

    qf_ferm_op_del_groups(op);

    bool deleted_groups = !qf_ferm_op_has_groups(op);

    bool passed_all = has_no_groups && has_some_groups && correct_num_groups && correct_group0 &&
                      correct_group1 && correct_indexed0 && correct_indexed1 &&
                      correct_groups_len && correct_groups_out0 && correct_groups_out1 &&
                      correct_groups_out2 && correct_groups_out3 && deleted_groups;

    qf_ferm_op_free(op);
    qf_ferm_op_free(group0);
    qf_ferm_op_free(group1);
    qf_ferm_op_free(group_ops[0]);
    qf_ferm_op_free(group_ops[1]);

    if (!passed_all) {
        return EqualityError;
    }
    return Ok;
}

int test_fermion_operator(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_new);
    num_failed += RUN_TEST(test_getters);
    num_failed += RUN_TEST(test_get_support);
    num_failed += RUN_TEST(test_add);
    num_failed += RUN_TEST(test_scaled_add);
    num_failed += RUN_TEST(test_inplace_arithmetic);
    num_failed += RUN_TEST(test_add_term);
    num_failed += RUN_TEST(test_equiv_pos);
    num_failed += RUN_TEST(test_equiv_neg);
    num_failed += RUN_TEST(test_mul);
    num_failed += RUN_TEST(test_compose);
    num_failed += RUN_TEST(test_compose_operand_order);
    num_failed += RUN_TEST(test_ichop);
    num_failed += RUN_TEST(test_simplify);
    num_failed += RUN_TEST(test_simplify_vs_ichop);
    num_failed += RUN_TEST(test_adjoint);
    num_failed += RUN_TEST(test_normal_ordered);
    num_failed += RUN_TEST(test_sandwich_ordered_true);
    num_failed += RUN_TEST(test_sandwich_ordered_false);
    num_failed += RUN_TEST(test_is_hermitian);
    num_failed += RUN_TEST(test_max_rank);
    num_failed += RUN_TEST(test_conserves_particle_number);
    num_failed += RUN_TEST(test_conserves_sector);
    num_failed += RUN_TEST(test_len);
    num_failed += RUN_TEST(test_relabel_modes);
    num_failed += RUN_TEST(test_relabel_modes_duplicate_err);
    num_failed += RUN_TEST(test_relabel_modes_too_small_err);
    num_failed += RUN_TEST(test_groups);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
