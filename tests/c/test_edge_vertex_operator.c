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

static int test_new(void) {
    // `1.0 + (-1.0) * V(0) E(0,1)`
    uint64_t num_terms = 2;
    uint64_t num_indices = 2;
    uint32_t left_indices[2] = {0, 0};
    uint32_t right_indices[2] = {0, 1};
    QkComplex64 coeffs[2] = {{1.0, 0.0}, {-1.0, 0.0}};
    uint32_t boundaries[3] = {0, 0, 2};
    QfEdgeVertexOperator *op =
        qf_edge_op_new(num_terms, num_indices, coeffs, left_indices, right_indices, boundaries);

    QfEdgeVertexOperator *expected = qf_edge_op_zero();
    QkComplex64 coeff0 = {1.0, 0.0};
    qf_edge_op_add_term(expected, 0, NULL, NULL, &coeff0);
    uint32_t left1[2] = {0, 0};
    uint32_t right1[2] = {0, 1};
    QkComplex64 coeff1 = {-1.0, 0.0};
    qf_edge_op_add_term(expected, 2, left1, right1, &coeff1);

    bool is_equal = qf_edge_op_equal(op, expected);

    qf_edge_op_free(op);
    qf_edge_op_free(expected);

    if (!is_equal) {
        return EqualityError;
    }
    return Ok;
}

static int test_getters(void) {
    uint64_t num_terms = 2;
    uint64_t num_indices = 3;
    uint32_t left_indices[3] = {0, 1, 2};
    uint32_t right_indices[3] = {0, 2, 3};
    QkComplex64 coeffs[2] = {{1.0, 0.0}, {0.0, -1.0}};
    uint32_t boundaries[3] = {0, 1, 3};
    QfEdgeVertexOperator *op =
        qf_edge_op_new(num_terms, num_indices, coeffs, left_indices, right_indices, boundaries);

    bool passed_all = true;

    QkComplex64 *coeffs_out;
    uint64_t coeffs_len;
    qf_edge_op_get_coeffs(op, &coeffs_out, &coeffs_len);
    passed_all = passed_all && (coeffs_len == num_terms);
    for (uint64_t i = 0; i < num_terms; i++) {
        passed_all = passed_all && (coeffs_out[i].re == coeffs[i].re);
        passed_all = passed_all && (coeffs_out[i].im == coeffs[i].im);
    }

    // Both index buffers must be exposed independently; a copy-paste slip that returned the same
    // buffer twice would survive a test that only checked one of them.
    uint32_t *left_out;
    uint64_t left_len;
    qf_edge_op_get_left_indices(op, &left_out, &left_len);
    passed_all = passed_all && (left_len == num_indices);
    for (uint64_t i = 0; i < num_indices; i++) {
        passed_all = passed_all && (left_out[i] == left_indices[i]);
    }

    uint32_t *right_out;
    uint64_t right_len;
    qf_edge_op_get_right_indices(op, &right_out, &right_len);
    passed_all = passed_all && (right_len == num_indices);
    for (uint64_t i = 0; i < num_indices; i++) {
        passed_all = passed_all && (right_out[i] == right_indices[i]);
    }

    size_t *boundaries_out;
    uint64_t boundaries_len;
    qf_edge_op_get_boundaries(op, &boundaries_out, &boundaries_len);
    passed_all = passed_all && (boundaries_len == 3);
    for (uint64_t i = 0; i < 3; i++) {
        passed_all = passed_all && (boundaries_out[i] == boundaries[i]);
    }

    qf_edge_op_free(op);

    if (!passed_all) {
        return EqualityError;
    }
    return Ok;
}

static int test_add(void) {
    QfEdgeVertexOperator *one = qf_edge_op_one();
    QfEdgeVertexOperator *zero = qf_edge_op_zero();

    QfEdgeVertexOperator *result = qf_edge_op_add(one, zero);

    bool is_equal = qf_edge_op_equal(result, one);

    qf_edge_op_free(one);
    qf_edge_op_free(zero);
    qf_edge_op_free(result);

    if (!is_equal) {
        return EqualityError;
    }
    return Ok;
}

static int test_add_term(void) {
    QfEdgeVertexOperator *one = qf_edge_op_one();

    QfEdgeVertexOperator *op = qf_edge_op_zero();
    QkComplex64 coeff = {1.0, 0.0};
    qf_edge_op_add_term(op, 0, NULL, NULL, &coeff);

    bool is_equal = qf_edge_op_equal(op, one);

    qf_edge_op_free(one);
    qf_edge_op_free(op);

    if (!is_equal) {
        return EqualityError;
    }
    return Ok;
}

static int test_mul(void) {
    QfEdgeVertexOperator *one = qf_edge_op_one();
    QkComplex64 coeff = {2.0, 0.0};
    QfEdgeVertexOperator *result = qf_edge_op_mul(one, &coeff);

    QfEdgeVertexOperator *expected = qf_edge_op_zero();
    qf_edge_op_add_term(expected, 0, NULL, NULL, &coeff);

    bool is_equal = qf_edge_op_equal(result, expected);

    qf_edge_op_free(one);
    qf_edge_op_free(result);
    qf_edge_op_free(expected);

    if (!is_equal) {
        return EqualityError;
    }
    return Ok;
}

static int test_compose(void) {
    QfEdgeVertexOperator *one = qf_edge_op_one();
    QfEdgeVertexOperator *zero = qf_edge_op_zero();

    QfEdgeVertexOperator *result = qf_edge_op_compose(one, zero);

    bool is_equal = qf_edge_op_equal(result, zero);

    qf_edge_op_free(one);
    qf_edge_op_free(zero);
    qf_edge_op_free(result);

    if (!is_equal) {
        return EqualityError;
    }
    return Ok;
}

static int test_equiv(void) {
    QfEdgeVertexOperator *zero = qf_edge_op_zero();

    QfEdgeVertexOperator *op = qf_edge_op_zero();
    QkComplex64 coeff = {1e-7, 0.0};
    qf_edge_op_add_term(op, 0, NULL, NULL, &coeff);

    bool loose = qf_edge_op_equiv(op, zero, 1e-6);
    bool tight = !qf_edge_op_equiv(op, zero, 1e-8);

    qf_edge_op_free(zero);
    qf_edge_op_free(op);

    if (!(loose && tight)) {
        return EqualityError;
    }
    return Ok;
}

static int test_ichop(void) {
    QfEdgeVertexOperator *op = qf_edge_op_zero();
    QkComplex64 coeff = {1e-8, 0.0};
    qf_edge_op_add_term(op, 0, NULL, NULL, &coeff);

    qf_edge_op_ichop(op, 1e-6);

    QfEdgeVertexOperator *expected = qf_edge_op_zero();

    bool is_equal = qf_edge_op_equal(op, expected);

    qf_edge_op_free(op);
    qf_edge_op_free(expected);

    if (!is_equal) {
        return EqualityError;
    }
    return Ok;
}

static int test_simplify(void) {
    // 100 copies of a 1e-5 identity term sum to 1e-3, which must survive a 1e-4 threshold. A
    // greedy `ichop` would instead discard every one of them.
    QfEdgeVertexOperator *op = qf_edge_op_zero();
    QkComplex64 coeff = {1e-5, 0.0};
    for (int i = 0; i < 100; i++) {
        qf_edge_op_add_term(op, 0, NULL, NULL, &coeff);
    }

    QfEdgeVertexOperator *canon = qf_edge_op_simplify(op, 1e-4);

    bool correct_len = qf_edge_op_len(canon) == 1;

    QkComplex64 *coeffs_out;
    uint64_t coeffs_len;
    qf_edge_op_get_coeffs(canon, &coeffs_out, &coeffs_len);
    bool correct_coeff = (coeffs_len == 1) && (coeffs_out[0].re > 9.9e-4);

    qf_edge_op_ichop(op, 1e-4);
    bool ichop_dropped_all = qf_edge_op_len(op) == 0;

    qf_edge_op_free(op);
    qf_edge_op_free(canon);

    if (!(correct_len && correct_coeff && ichop_dropped_all)) {
        return EqualityError;
    }
    return Ok;
}

static int test_adjoint(void) {
    // `V(0) E(0,1)` is multi-factor, so the reversal of the operator string is observable. A
    // single-factor term would make it a no-op.
    QfEdgeVertexOperator *op = qf_edge_op_zero();
    uint32_t left[2] = {0, 0};
    uint32_t right[2] = {0, 1};
    QkComplex64 coeff = {3.0, -4.0};
    qf_edge_op_add_term(op, 2, left, right, &coeff);

    QfEdgeVertexOperator *adj = qf_edge_op_adjoint(op);

    QfEdgeVertexOperator *expected = qf_edge_op_zero();
    uint32_t left_exp[2] = {0, 0};
    uint32_t right_exp[2] = {1, 0};
    QkComplex64 coeff_exp = {3.0, 4.0};
    qf_edge_op_add_term(expected, 2, left_exp, right_exp, &coeff_exp);

    bool is_equal = qf_edge_op_equal(adj, expected);

    qf_edge_op_free(op);
    qf_edge_op_free(adj);
    qf_edge_op_free(expected);

    if (!is_equal) {
        return EqualityError;
    }
    return Ok;
}

static int test_normal_ordered(void) {
    // `E(1,0) = -E(0,1)`, so the two orientation conventions must give different results. This is
    // what distinguishes the `ascending` flag from a no-op.
    QfEdgeVertexOperator *op = qf_edge_op_zero();
    uint32_t left[1] = {1};
    uint32_t right[1] = {0};
    QkComplex64 coeff = {1.0, 0.0};
    qf_edge_op_add_term(op, 1, left, right, &coeff);

    QfEdgeVertexOperator *asc = qf_edge_op_normal_ordered(op, true, true);
    QfEdgeVertexOperator *desc = qf_edge_op_normal_ordered(op, false, true);

    QfEdgeVertexOperator *expected_asc = qf_edge_op_zero();
    uint32_t left_asc[1] = {0};
    uint32_t right_asc[1] = {1};
    QkComplex64 coeff_asc = {-1.0, 0.0};
    qf_edge_op_add_term(expected_asc, 1, left_asc, right_asc, &coeff_asc);

    bool correct_asc = qf_edge_op_equal(asc, expected_asc);
    // descending keeps the stored orientation, hence the original coefficient
    bool correct_desc = qf_edge_op_equal(desc, op);

    qf_edge_op_free(op);
    qf_edge_op_free(asc);
    qf_edge_op_free(desc);
    qf_edge_op_free(expected_asc);

    if (!(correct_asc && correct_desc)) {
        return EqualityError;
    }
    return Ok;
}

static int test_normal_ordered_reduce(void) {
    // The contraction rules: `E(0,1) E(0,1) = 1` and the fusion `E(0,1) E(1,2) = -i E(0,2)`.
    QkComplex64 one_coeff = {1.0, 0.0};

    QfEdgeVertexOperator *squared = qf_edge_op_zero();
    uint32_t left_sq[2] = {0, 0};
    uint32_t right_sq[2] = {1, 1};
    qf_edge_op_add_term(squared, 2, left_sq, right_sq, &one_coeff);
    QfEdgeVertexOperator *reduced_sq = qf_edge_op_normal_ordered(squared, true, true);
    QfEdgeVertexOperator *expected_sq = qf_edge_op_one();
    bool correct_sq = qf_edge_op_equiv(reduced_sq, expected_sq, 1e-12);

    QfEdgeVertexOperator *fused = qf_edge_op_zero();
    uint32_t left_fu[2] = {0, 1};
    uint32_t right_fu[2] = {1, 2};
    qf_edge_op_add_term(fused, 2, left_fu, right_fu, &one_coeff);
    QfEdgeVertexOperator *reduced_fu = qf_edge_op_normal_ordered(fused, true, true);
    QfEdgeVertexOperator *expected_fu = qf_edge_op_zero();
    uint32_t left_fe[1] = {0};
    uint32_t right_fe[1] = {2};
    QkComplex64 minus_i = {0.0, -1.0};
    qf_edge_op_add_term(expected_fu, 1, left_fe, right_fe, &minus_i);
    bool correct_fu = qf_edge_op_equiv(reduced_fu, expected_fu, 1e-12);

    qf_edge_op_free(squared);
    qf_edge_op_free(reduced_sq);
    qf_edge_op_free(expected_sq);
    qf_edge_op_free(fused);
    qf_edge_op_free(reduced_fu);
    qf_edge_op_free(expected_fu);

    if (!(correct_sq && correct_fu)) {
        return EqualityError;
    }
    return Ok;
}

static int test_is_hermitian(void) {
    // `V(0) E(0,1)` is *not* Hermitian: the generators share index 0 and therefore anticommute,
    // so `(V(0) E(0,1))^dag = -V(0) E(0,1)`. Symmetrizing it does give a Hermitian operator.
    QfEdgeVertexOperator *op = qf_edge_op_zero();
    uint32_t left[2] = {0, 0};
    uint32_t right[2] = {0, 1};
    QkComplex64 coeff = {1.0, 0.0};
    qf_edge_op_add_term(op, 2, left, right, &coeff);

    bool not_hermitian = !qf_edge_op_is_hermitian(op, 1e-10);

    QfEdgeVertexOperator *adj = qf_edge_op_adjoint(op);
    QfEdgeVertexOperator *sym = qf_edge_op_add(op, adj);

    bool sym_is_hermitian = qf_edge_op_is_hermitian(sym, 1e-10);

    qf_edge_op_free(op);
    qf_edge_op_free(adj);
    qf_edge_op_free(sym);

    if (!(not_hermitian && sym_is_hermitian)) {
        return EqualityError;
    }
    return Ok;
}

static int test_len(void) {
    QfEdgeVertexOperator *op = qf_edge_op_zero();
    bool empty = qf_edge_op_len(op) == 0;

    uint32_t left[2] = {0, 1};
    uint32_t right[2] = {1, 2};
    QkComplex64 coeff = {1.0, 0.0};
    qf_edge_op_add_term(op, 2, left, right, &coeff);

    bool one_term = qf_edge_op_len(op) == 1;

    qf_edge_op_free(op);

    if (!(empty && one_term)) {
        return EqualityError;
    }
    return Ok;
}

static int test_relabel_modes(void) {
    QfEdgeVertexOperator *op = qf_edge_op_zero();
    uint32_t left[2] = {0, 2};
    uint32_t right[2] = {1, 3};
    QkComplex64 coeff = {1.0, 0.0};
    qf_edge_op_add_term(op, 2, left, right, &coeff);

    uint32_t permutation[4] = {3, 2, 1, 0};

    QfExitCode exit = qf_edge_op_relabel_modes(op, 4, permutation);
    if (exit != QfExitCode_Success) {
        qf_edge_op_free(op);
        return RuntimeError;
    }

    QfEdgeVertexOperator *expected = qf_edge_op_zero();
    uint32_t left_exp[2] = {3, 1};
    uint32_t right_exp[2] = {2, 0};
    qf_edge_op_add_term(expected, 2, left_exp, right_exp, &coeff);

    bool is_equal = qf_edge_op_equal(op, expected);

    qf_edge_op_free(op);
    qf_edge_op_free(expected);

    if (!is_equal) {
        return EqualityError;
    }
    return Ok;
}

static int test_relabel_modes_duplicate_err(void) {
    QfEdgeVertexOperator *op = qf_edge_op_zero();
    uint32_t left[1] = {0};
    uint32_t right[1] = {1};
    QkComplex64 coeff = {1.0, 0.0};
    qf_edge_op_add_term(op, 1, left, right, &coeff);

    uint32_t permutation[4] = {0, 0, 2, 3};

    QfExitCode exit = qf_edge_op_relabel_modes(op, 4, permutation);

    qf_edge_op_free(op);

    return exit == QfExitCode_DuplicateIndexError ? Ok : EqualityError;
}

static int test_relabel_modes_too_small_err(void) {
    // The right-hand index 3 falls outside a 3-entry permutation. Exercising it from the *right*
    // buffer shows that side is mapped at all, rather than copied through.
    QfEdgeVertexOperator *op = qf_edge_op_zero();
    uint32_t left[1] = {0};
    uint32_t right[1] = {3};
    QkComplex64 coeff = {1.0, 0.0};
    qf_edge_op_add_term(op, 1, left, right, &coeff);

    uint32_t permutation[3] = {1, 0, 2};

    QfExitCode exit = qf_edge_op_relabel_modes(op, 3, permutation);

    qf_edge_op_free(op);

    return exit == QfExitCode_IndexError ? Ok : EqualityError;
}

static int test_groups(void) {
    uint64_t num_terms = 4;
    uint64_t num_indices = 4;
    uint32_t left_indices[4] = {0, 1, 2, 3};
    uint32_t right_indices[4] = {1, 2, 3, 0};
    QkComplex64 coeffs[4] = {{1.0, 0.0}, {3.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}};
    uint32_t boundaries[5] = {0, 1, 2, 3, 4};
    QfEdgeVertexOperator *op =
        qf_edge_op_new(num_terms, num_indices, coeffs, left_indices, right_indices, boundaries);

    bool has_no_groups = !qf_edge_op_has_groups(op);

    uint32_t groups_in[4] = {0, 1, 0, 1};
    qf_edge_op_set_groups(op, groups_in, num_terms);

    bool has_some_groups = qf_edge_op_has_groups(op);
    bool correct_num_groups = qf_edge_op_num_groups(op) == 2;

    uint32_t *groups_out;
    uint64_t groups_len;
    qf_edge_op_get_groups(op, &groups_out, &groups_len);
    bool correct_groups = groups_len == 4;
    for (uint64_t i = 0; i < 4; i++) {
        correct_groups = correct_groups && (groups_out[i] == groups_in[i]);
    }

    // group 0 holds coeffs 1.0 and 1.0 (mean 1.0); group 1 holds 3.0 and 1.0 (mean 2.0). Both
    // means are exactly representable, so an exact comparison is safe here.
    double weights[2];
    qf_edge_op_group_weights(op, weights);
    bool correct_weights = (weights[0] == 1.0) && (weights[1] == 2.0);

    QfEdgeVertexOperator *group_ops[2];
    qf_edge_op_split_out_groups(op, NULL, 0, group_ops);
    bool correct_split_len =
        (qf_edge_op_len(group_ops[0]) == 2) && (qf_edge_op_len(group_ops[1]) == 2);

    // a duplicate index must be materialized once per occurrence, not deduplicated
    uint32_t group_indices[2] = {1, 1};
    QfEdgeVertexOperator *group_ops_indexed[2];
    qf_edge_op_split_out_groups(op, group_indices, 2, group_ops_indexed);
    bool correct_indexed = qf_edge_op_equiv(group_ops_indexed[0], group_ops[1], 1e-10) &&
                           qf_edge_op_equiv(group_ops_indexed[1], group_ops[1], 1e-10);

    qf_edge_op_del_groups(op);
    bool groups_deleted = !qf_edge_op_has_groups(op);

    qf_edge_op_free(group_ops[0]);
    qf_edge_op_free(group_ops[1]);
    qf_edge_op_free(group_ops_indexed[0]);
    qf_edge_op_free(group_ops_indexed[1]);
    qf_edge_op_free(op);

    bool passed_all = has_no_groups && has_some_groups && correct_num_groups && correct_groups &&
                      correct_weights && correct_split_len && correct_indexed && groups_deleted;

    if (!passed_all) {
        return EqualityError;
    }
    return Ok;
}

static int test_canonical_order(void) {
    QfEdgeVertexOperator *op = qf_edge_op_zero();
    uint32_t left_a[1] = {1};
    uint32_t right_a[1] = {2};
    QkComplex64 coeff_a = {1.0, 0.0};
    qf_edge_op_add_term(op, 1, left_a, right_a, &coeff_a);
    uint32_t left_b[1] = {0};
    uint32_t right_b[1] = {1};
    QkComplex64 coeff_b = {2.0, 0.0};
    qf_edge_op_add_term(op, 1, left_b, right_b, &coeff_b);

    QfEdgeVertexOperator *ordered = qf_edge_op_canonical_order(op);

    // `E(0,1)` sorts ahead of `E(1,2)`, so the terms swap and the coefficients travel with them.
    QkComplex64 *coeffs_out;
    uint64_t coeffs_len;
    qf_edge_op_get_coeffs(ordered, &coeffs_out, &coeffs_len);
    bool correct = (coeffs_len == 2) && (coeffs_out[0].re == 2.0) && (coeffs_out[1].re == 1.0);

    qf_edge_op_free(op);
    qf_edge_op_free(ordered);

    if (!correct) {
        return EqualityError;
    }
    return Ok;
}

static int test_commutators(void) {
    // `V(0)` and `E(0,1)` share index 0, so they anticommute: the commutator is non-zero while
    // the anti-commutator vanishes.
    QfEdgeVertexOperator *v0 = qf_edge_op_zero();
    uint32_t left_v[1] = {0};
    uint32_t right_v[1] = {0};
    QkComplex64 coeff = {1.0, 0.0};
    qf_edge_op_add_term(v0, 1, left_v, right_v, &coeff);

    QfEdgeVertexOperator *e01 = qf_edge_op_zero();
    uint32_t left_e[1] = {0};
    uint32_t right_e[1] = {1};
    qf_edge_op_add_term(e01, 1, left_e, right_e, &coeff);

    QfEdgeVertexOperator *comm = qf_edge_op_commutator(v0, e01);
    QfEdgeVertexOperator *anti = qf_edge_op_anti_commutator(v0, e01);

    QfEdgeVertexOperator *comm_no = qf_edge_op_normal_ordered(comm, true, true);
    QfEdgeVertexOperator *anti_no = qf_edge_op_normal_ordered(anti, true, true);

    QfEdgeVertexOperator *zero = qf_edge_op_zero();
    bool comm_nonzero = !qf_edge_op_equiv(comm_no, zero, 1e-10);
    bool anti_vanishes = qf_edge_op_equiv(anti_no, zero, 1e-10);

    qf_edge_op_free(v0);
    qf_edge_op_free(e01);
    qf_edge_op_free(comm);
    qf_edge_op_free(anti);
    qf_edge_op_free(comm_no);
    qf_edge_op_free(anti_no);
    qf_edge_op_free(zero);

    if (!(comm_nonzero && anti_vanishes)) {
        return EqualityError;
    }
    return Ok;
}

static int test_map_to_fermion(void) {
    QfEdgeVertexOperator *op = qf_edge_op_zero();
    uint32_t left[1] = {0};
    uint32_t right[1] = {0};
    QkComplex64 coeff = {1.0, 0.0};
    qf_edge_op_add_term(op, 1, left, right, &coeff);

    QfFermionOperator *fer_op = qf_edge_vertex_to_fermion(op);

    // `V(0) = 1 - 2 a^dag_0 a_0` is Hermitian, and so is its fermionic image.
    bool is_hermitian = qf_ferm_op_is_hermitian(fer_op, 1e-10);
    bool non_empty = qf_ferm_op_len(fer_op) > 0;

    qf_edge_op_free(op);
    qf_ferm_op_free(fer_op);

    if (!(is_hermitian && non_empty)) {
        return EqualityError;
    }
    return Ok;
}

static int test_map_to_majorana(void) {
    QfEdgeVertexOperator *op = qf_edge_op_zero();
    uint32_t left[1] = {0};
    uint32_t right[1] = {1};
    QkComplex64 coeff = {1.0, 0.0};
    qf_edge_op_add_term(op, 1, left, right, &coeff);

    QfMajoranaOperator *maj_op = qf_edge_vertex_to_majorana(op);

    bool is_hermitian = qf_maj_op_is_hermitian(maj_op, 1e-10);
    bool non_empty = qf_maj_op_len(maj_op) > 0;

    qf_edge_op_free(op);
    qf_maj_op_free(maj_op);

    if (!(is_hermitian && non_empty)) {
        return EqualityError;
    }
    return Ok;
}

int test_edge_vertex_operator(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_new);
    num_failed += RUN_TEST(test_getters);
    num_failed += RUN_TEST(test_add);
    num_failed += RUN_TEST(test_add_term);
    num_failed += RUN_TEST(test_mul);
    num_failed += RUN_TEST(test_compose);
    num_failed += RUN_TEST(test_equiv);
    num_failed += RUN_TEST(test_ichop);
    num_failed += RUN_TEST(test_simplify);
    num_failed += RUN_TEST(test_adjoint);
    num_failed += RUN_TEST(test_normal_ordered);
    num_failed += RUN_TEST(test_normal_ordered_reduce);
    num_failed += RUN_TEST(test_is_hermitian);
    num_failed += RUN_TEST(test_len);
    num_failed += RUN_TEST(test_relabel_modes);
    num_failed += RUN_TEST(test_relabel_modes_duplicate_err);
    num_failed += RUN_TEST(test_relabel_modes_too_small_err);
    num_failed += RUN_TEST(test_groups);
    num_failed += RUN_TEST(test_canonical_order);
    num_failed += RUN_TEST(test_commutators);
    num_failed += RUN_TEST(test_map_to_fermion);
    num_failed += RUN_TEST(test_map_to_majorana);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
