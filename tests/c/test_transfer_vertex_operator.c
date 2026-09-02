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
    // `1.0 + (-1.0) * V(0) T(0,1)`
    uint64_t num_terms = 2;
    uint64_t num_indices = 2;
    uint32_t left_indices[2] = {0, 0};
    uint32_t right_indices[2] = {0, 1};
    QkComplex64 coeffs[2] = {{1.0, 0.0}, {-1.0, 0.0}};
    uint32_t boundaries[3] = {0, 0, 2};
    QfTransferVertexOperator *op =
        qf_transfer_op_new(num_terms, num_indices, coeffs, left_indices, right_indices, boundaries);

    QfTransferVertexOperator *expected = qf_transfer_op_zero();
    QkComplex64 coeff0 = {1.0, 0.0};
    qf_transfer_op_add_term(expected, 0, NULL, NULL, &coeff0);
    uint32_t left1[2] = {0, 0};
    uint32_t right1[2] = {0, 1};
    QkComplex64 coeff1 = {-1.0, 0.0};
    qf_transfer_op_add_term(expected, 2, left1, right1, &coeff1);

    bool is_equal = qf_transfer_op_equal(op, expected);

    qf_transfer_op_free(op);
    qf_transfer_op_free(expected);

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
    QfTransferVertexOperator *op =
        qf_transfer_op_new(num_terms, num_indices, coeffs, left_indices, right_indices, boundaries);

    bool passed_all = true;

    QkComplex64 *coeffs_out;
    uint64_t coeffs_len;
    qf_transfer_op_get_coeffs(op, &coeffs_out, &coeffs_len);
    passed_all = passed_all && (coeffs_len == num_terms);
    for (uint64_t i = 0; i < num_terms; i++) {
        passed_all = passed_all && (coeffs_out[i].re == coeffs[i].re);
        passed_all = passed_all && (coeffs_out[i].im == coeffs[i].im);
    }

    // Both index buffers must be exposed independently; a copy-paste slip that returned the same
    // buffer twice would survive a test that only checked one of them.
    uint32_t *left_out;
    uint64_t left_len;
    qf_transfer_op_get_left_indices(op, &left_out, &left_len);
    passed_all = passed_all && (left_len == num_indices);
    for (uint64_t i = 0; i < num_indices; i++) {
        passed_all = passed_all && (left_out[i] == left_indices[i]);
    }

    uint32_t *right_out;
    uint64_t right_len;
    qf_transfer_op_get_right_indices(op, &right_out, &right_len);
    passed_all = passed_all && (right_len == num_indices);
    for (uint64_t i = 0; i < num_indices; i++) {
        passed_all = passed_all && (right_out[i] == right_indices[i]);
    }

    size_t *boundaries_out;
    uint64_t boundaries_len;
    qf_transfer_op_get_boundaries(op, &boundaries_out, &boundaries_len);
    passed_all = passed_all && (boundaries_len == 3);
    for (uint64_t i = 0; i < 3; i++) {
        passed_all = passed_all && (boundaries_out[i] == boundaries[i]);
    }

    qf_transfer_op_free(op);

    if (!passed_all) {
        return EqualityError;
    }
    return Ok;
}

static int test_add(void) {
    QfTransferVertexOperator *one = qf_transfer_op_one();
    QfTransferVertexOperator *zero = qf_transfer_op_zero();

    QfTransferVertexOperator *result = qf_transfer_op_add(one, zero);

    bool is_equal = qf_transfer_op_equal(result, one);

    qf_transfer_op_free(one);
    qf_transfer_op_free(zero);
    qf_transfer_op_free(result);

    if (!is_equal) {
        return EqualityError;
    }
    return Ok;
}

static int test_add_term(void) {
    QfTransferVertexOperator *one = qf_transfer_op_one();

    QfTransferVertexOperator *op = qf_transfer_op_zero();
    QkComplex64 coeff = {1.0, 0.0};
    qf_transfer_op_add_term(op, 0, NULL, NULL, &coeff);

    bool is_equal = qf_transfer_op_equal(op, one);

    qf_transfer_op_free(one);
    qf_transfer_op_free(op);

    if (!is_equal) {
        return EqualityError;
    }
    return Ok;
}

static int test_mul(void) {
    QfTransferVertexOperator *one = qf_transfer_op_one();
    QkComplex64 coeff = {2.0, 0.0};
    QfTransferVertexOperator *result = qf_transfer_op_mul(one, &coeff);

    QfTransferVertexOperator *expected = qf_transfer_op_zero();
    qf_transfer_op_add_term(expected, 0, NULL, NULL, &coeff);

    bool is_equal = qf_transfer_op_equal(result, expected);

    qf_transfer_op_free(one);
    qf_transfer_op_free(result);
    qf_transfer_op_free(expected);

    if (!is_equal) {
        return EqualityError;
    }
    return Ok;
}

static int test_compose(void) {
    QfTransferVertexOperator *one = qf_transfer_op_one();
    QfTransferVertexOperator *zero = qf_transfer_op_zero();

    QfTransferVertexOperator *result = qf_transfer_op_compose(one, zero);

    bool is_equal = qf_transfer_op_equal(result, zero);

    qf_transfer_op_free(one);
    qf_transfer_op_free(zero);
    qf_transfer_op_free(result);

    if (!is_equal) {
        return EqualityError;
    }
    return Ok;
}

static int test_equiv(void) {
    QfTransferVertexOperator *zero = qf_transfer_op_zero();

    QfTransferVertexOperator *op = qf_transfer_op_zero();
    QkComplex64 coeff = {1e-7, 0.0};
    qf_transfer_op_add_term(op, 0, NULL, NULL, &coeff);

    bool loose = qf_transfer_op_equiv(op, zero, 1e-6);
    bool tight = !qf_transfer_op_equiv(op, zero, 1e-8);

    qf_transfer_op_free(zero);
    qf_transfer_op_free(op);

    if (!(loose && tight)) {
        return EqualityError;
    }
    return Ok;
}

static int test_ichop(void) {
    QfTransferVertexOperator *op = qf_transfer_op_zero();
    QkComplex64 coeff = {1e-8, 0.0};
    qf_transfer_op_add_term(op, 0, NULL, NULL, &coeff);

    qf_transfer_op_ichop(op, 1e-6);

    QfTransferVertexOperator *expected = qf_transfer_op_zero();

    bool is_equal = qf_transfer_op_equal(op, expected);

    qf_transfer_op_free(op);
    qf_transfer_op_free(expected);

    if (!is_equal) {
        return EqualityError;
    }
    return Ok;
}

static int test_simplify(void) {
    // 100 copies of a 1e-5 identity term sum to 1e-3, which must survive a 1e-4 threshold. A
    // greedy `ichop` would instead discard every one of them.
    QfTransferVertexOperator *op = qf_transfer_op_zero();
    QkComplex64 coeff = {1e-5, 0.0};
    for (int i = 0; i < 100; i++) {
        qf_transfer_op_add_term(op, 0, NULL, NULL, &coeff);
    }

    QfTransferVertexOperator *canon = qf_transfer_op_simplify(op, 1e-4);

    bool correct_len = qf_transfer_op_len(canon) == 1;

    QkComplex64 *coeffs_out;
    uint64_t coeffs_len;
    qf_transfer_op_get_coeffs(canon, &coeffs_out, &coeffs_len);
    bool correct_coeff = (coeffs_len == 1) && (coeffs_out[0].re > 9.9e-4);

    qf_transfer_op_ichop(op, 1e-4);
    bool ichop_dropped_all = qf_transfer_op_len(op) == 0;

    qf_transfer_op_free(op);
    qf_transfer_op_free(canon);

    if (!(correct_len && correct_coeff && ichop_dropped_all)) {
        return EqualityError;
    }
    return Ok;
}

static int test_adjoint(void) {
    // `V(0) T(0,1)` is multi-factor, so the reversal of the operator string is observable.
    QfTransferVertexOperator *op = qf_transfer_op_zero();
    uint32_t left[2] = {0, 0};
    uint32_t right[2] = {0, 1};
    QkComplex64 coeff = {3.0, -4.0};
    qf_transfer_op_add_term(op, 2, left, right, &coeff);

    QfTransferVertexOperator *adj = qf_transfer_op_adjoint(op);

    QfTransferVertexOperator *expected = qf_transfer_op_zero();
    uint32_t left_exp[2] = {0, 0};
    uint32_t right_exp[2] = {1, 0};
    QkComplex64 coeff_exp = {3.0, 4.0};
    qf_transfer_op_add_term(expected, 2, left_exp, right_exp, &coeff_exp);

    bool is_equal = qf_transfer_op_equal(adj, expected);

    qf_transfer_op_free(op);
    qf_transfer_op_free(adj);
    qf_transfer_op_free(expected);

    if (!is_equal) {
        return EqualityError;
    }
    return Ok;
}

static int test_normal_ordered_reduce(void) {
    // `T(0,1) T(0,1) = 1/4`, so a coefficient of 4.0 reduces to the identity.
    QfTransferVertexOperator *squared = qf_transfer_op_zero();
    uint32_t left_sq[2] = {0, 0};
    uint32_t right_sq[2] = {1, 1};
    QkComplex64 four = {4.0, 0.0};
    qf_transfer_op_add_term(squared, 2, left_sq, right_sq, &four);

    QfTransferVertexOperator *reduced_sq = qf_transfer_op_normal_ordered(squared, true);
    QfTransferVertexOperator *one = qf_transfer_op_one();
    bool correct_sq = qf_transfer_op_equiv(reduced_sq, one, 1e-12);

    // `T(0,1) T(1,0) = -1/4 V(0) V(1)`, so a coefficient of 4.0 gives `-V(0) V(1)`.
    QfTransferVertexOperator *anti = qf_transfer_op_zero();
    uint32_t left_an[2] = {0, 1};
    uint32_t right_an[2] = {1, 0};
    qf_transfer_op_add_term(anti, 2, left_an, right_an, &four);

    QfTransferVertexOperator *reduced_an = qf_transfer_op_normal_ordered(anti, true);

    QfTransferVertexOperator *expected_an = qf_transfer_op_zero();
    uint32_t left_ae[2] = {0, 1};
    uint32_t right_ae[2] = {0, 1};
    QkComplex64 minus_one = {-1.0, 0.0};
    qf_transfer_op_add_term(expected_an, 2, left_ae, right_ae, &minus_one);
    bool correct_an = qf_transfer_op_equiv(reduced_an, expected_an, 1e-12);

    qf_transfer_op_free(squared);
    qf_transfer_op_free(reduced_sq);
    qf_transfer_op_free(one);
    qf_transfer_op_free(anti);
    qf_transfer_op_free(reduced_an);
    qf_transfer_op_free(expected_an);

    if (!(correct_sq && correct_an)) {
        return EqualityError;
    }
    return Ok;
}

static int test_normal_ordered_no_fusion(void) {
    // Two transfer operators sharing a single mode cannot collapse into one, unlike two edge
    // operators. Reducing must leave the term length alone rather than invent a shorter form.
    QfTransferVertexOperator *op = qf_transfer_op_zero();
    uint32_t left[2] = {0, 0};
    uint32_t right[2] = {1, 2};
    QkComplex64 coeff = {1.0, 0.0};
    qf_transfer_op_add_term(op, 2, left, right, &coeff);

    QfTransferVertexOperator *reduced = qf_transfer_op_normal_ordered(op, true);

    size_t *boundaries_out;
    uint64_t boundaries_len;
    qf_transfer_op_get_boundaries(reduced, &boundaries_out, &boundaries_len);

    bool still_length_two = true;
    for (uint64_t i = 0; i + 1 < boundaries_len; i++) {
        still_length_two = still_length_two && ((boundaries_out[i + 1] - boundaries_out[i]) == 2);
    }

    qf_transfer_op_free(op);
    qf_transfer_op_free(reduced);

    if (!still_length_two) {
        return EqualityError;
    }
    return Ok;
}

static int test_is_hermitian(void) {
    // `V(0) T(0,1)` is *not* Hermitian: the generators share index 0 and therefore anticommute.
    QfTransferVertexOperator *op = qf_transfer_op_zero();
    uint32_t left[2] = {0, 0};
    uint32_t right[2] = {0, 1};
    QkComplex64 coeff = {1.0, 0.0};
    qf_transfer_op_add_term(op, 2, left, right, &coeff);

    bool not_hermitian = !qf_transfer_op_is_hermitian(op, 1e-10);

    QfTransferVertexOperator *adj = qf_transfer_op_adjoint(op);
    QfTransferVertexOperator *sym = qf_transfer_op_add(op, adj);

    bool sym_is_hermitian = qf_transfer_op_is_hermitian(sym, 1e-10);

    qf_transfer_op_free(op);
    qf_transfer_op_free(adj);
    qf_transfer_op_free(sym);

    if (!(not_hermitian && sym_is_hermitian)) {
        return EqualityError;
    }
    return Ok;
}

static int test_len(void) {
    QfTransferVertexOperator *op = qf_transfer_op_zero();
    bool empty = qf_transfer_op_len(op) == 0;

    uint32_t left[2] = {0, 1};
    uint32_t right[2] = {1, 2};
    QkComplex64 coeff = {1.0, 0.0};
    qf_transfer_op_add_term(op, 2, left, right, &coeff);

    bool one_term = qf_transfer_op_len(op) == 1;

    qf_transfer_op_free(op);

    if (!(empty && one_term)) {
        return EqualityError;
    }
    return Ok;
}

static int test_relabel_modes(void) {
    QfTransferVertexOperator *op = qf_transfer_op_zero();
    uint32_t left[2] = {0, 2};
    uint32_t right[2] = {1, 3};
    QkComplex64 coeff = {1.0, 0.0};
    qf_transfer_op_add_term(op, 2, left, right, &coeff);

    uint32_t permutation[4] = {3, 2, 1, 0};

    QfExitCode exit = qf_transfer_op_relabel_modes(op, 4, permutation);
    if (exit != QfExitCode_Success) {
        qf_transfer_op_free(op);
        return RuntimeError;
    }

    QfTransferVertexOperator *expected = qf_transfer_op_zero();
    uint32_t left_exp[2] = {3, 1};
    uint32_t right_exp[2] = {2, 0};
    qf_transfer_op_add_term(expected, 2, left_exp, right_exp, &coeff);

    bool is_equal = qf_transfer_op_equal(op, expected);

    qf_transfer_op_free(op);
    qf_transfer_op_free(expected);

    if (!is_equal) {
        return EqualityError;
    }
    return Ok;
}

static int test_relabel_modes_duplicate_err(void) {
    QfTransferVertexOperator *op = qf_transfer_op_zero();
    uint32_t left[1] = {0};
    uint32_t right[1] = {1};
    QkComplex64 coeff = {1.0, 0.0};
    qf_transfer_op_add_term(op, 1, left, right, &coeff);

    uint32_t permutation[4] = {0, 0, 2, 3};

    QfExitCode exit = qf_transfer_op_relabel_modes(op, 4, permutation);

    qf_transfer_op_free(op);

    return exit == QfExitCode_DuplicateIndexError ? Ok : EqualityError;
}

static int test_relabel_modes_too_small_err(void) {
    // The right-hand index 3 falls outside a 3-entry permutation.
    QfTransferVertexOperator *op = qf_transfer_op_zero();
    uint32_t left[1] = {0};
    uint32_t right[1] = {3};
    QkComplex64 coeff = {1.0, 0.0};
    qf_transfer_op_add_term(op, 1, left, right, &coeff);

    uint32_t permutation[3] = {1, 0, 2};

    QfExitCode exit = qf_transfer_op_relabel_modes(op, 3, permutation);

    qf_transfer_op_free(op);

    return exit == QfExitCode_IndexError ? Ok : EqualityError;
}

static int test_groups(void) {
    uint64_t num_terms = 4;
    uint64_t num_indices = 4;
    uint32_t left_indices[4] = {0, 1, 2, 3};
    uint32_t right_indices[4] = {1, 2, 3, 0};
    QkComplex64 coeffs[4] = {{1.0, 0.0}, {3.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}};
    uint32_t boundaries[5] = {0, 1, 2, 3, 4};
    QfTransferVertexOperator *op =
        qf_transfer_op_new(num_terms, num_indices, coeffs, left_indices, right_indices, boundaries);

    bool has_no_groups = !qf_transfer_op_has_groups(op);

    uint32_t groups_in[4] = {0, 1, 0, 1};
    qf_transfer_op_set_groups(op, groups_in, num_terms);

    bool has_some_groups = qf_transfer_op_has_groups(op);
    bool correct_num_groups = qf_transfer_op_num_groups(op) == 2;

    uint32_t *groups_out;
    uint64_t groups_len;
    qf_transfer_op_get_groups(op, &groups_out, &groups_len);
    bool correct_groups = groups_len == 4;
    for (uint64_t i = 0; i < 4; i++) {
        correct_groups = correct_groups && (groups_out[i] == groups_in[i]);
    }

    // group 0 holds coeffs 1.0 and 1.0 (mean 1.0); group 1 holds 3.0 and 1.0 (mean 2.0). Both
    // means are exactly representable, so an exact comparison is safe here.
    double weights[2];
    qf_transfer_op_group_weights(op, weights);
    bool correct_weights = (weights[0] == 1.0) && (weights[1] == 2.0);

    QfTransferVertexOperator *group_ops[2];
    qf_transfer_op_split_out_groups(op, NULL, 0, group_ops);
    bool correct_split_len =
        (qf_transfer_op_len(group_ops[0]) == 2) && (qf_transfer_op_len(group_ops[1]) == 2);

    // a duplicate index must be materialized once per occurrence, not deduplicated
    uint32_t group_indices[2] = {1, 1};
    QfTransferVertexOperator *group_ops_indexed[2];
    qf_transfer_op_split_out_groups(op, group_indices, 2, group_ops_indexed);
    bool correct_indexed = qf_transfer_op_equiv(group_ops_indexed[0], group_ops[1], 1e-10) &&
                           qf_transfer_op_equiv(group_ops_indexed[1], group_ops[1], 1e-10);

    qf_transfer_op_del_groups(op);
    bool groups_deleted = !qf_transfer_op_has_groups(op);

    qf_transfer_op_free(group_ops[0]);
    qf_transfer_op_free(group_ops[1]);
    qf_transfer_op_free(group_ops_indexed[0]);
    qf_transfer_op_free(group_ops_indexed[1]);
    qf_transfer_op_free(op);

    bool passed_all = has_no_groups && has_some_groups && correct_num_groups && correct_groups &&
                      correct_weights && correct_split_len && correct_indexed && groups_deleted;

    if (!passed_all) {
        return EqualityError;
    }
    return Ok;
}

static int test_canonical_order(void) {
    QfTransferVertexOperator *op = qf_transfer_op_zero();
    uint32_t left_a[1] = {1};
    uint32_t right_a[1] = {2};
    QkComplex64 coeff_a = {1.0, 0.0};
    qf_transfer_op_add_term(op, 1, left_a, right_a, &coeff_a);
    uint32_t left_b[1] = {0};
    uint32_t right_b[1] = {1};
    QkComplex64 coeff_b = {2.0, 0.0};
    qf_transfer_op_add_term(op, 1, left_b, right_b, &coeff_b);

    QfTransferVertexOperator *ordered = qf_transfer_op_canonical_order(op);

    // `T(0,1)` sorts ahead of `T(1,2)`, so the terms swap and the coefficients travel with them.
    QkComplex64 *coeffs_out;
    uint64_t coeffs_len;
    qf_transfer_op_get_coeffs(ordered, &coeffs_out, &coeffs_len);
    bool correct = (coeffs_len == 2) && (coeffs_out[0].re == 2.0) && (coeffs_out[1].re == 1.0);

    qf_transfer_op_free(op);
    qf_transfer_op_free(ordered);

    if (!correct) {
        return EqualityError;
    }
    return Ok;
}

static int test_commutators(void) {
    // `V(0)` and `T(0,1)` share index 0, so they anticommute: the commutator is non-zero while
    // the anti-commutator vanishes.
    QfTransferVertexOperator *v0 = qf_transfer_op_zero();
    uint32_t left_v[1] = {0};
    uint32_t right_v[1] = {0};
    QkComplex64 coeff = {1.0, 0.0};
    qf_transfer_op_add_term(v0, 1, left_v, right_v, &coeff);

    QfTransferVertexOperator *t01 = qf_transfer_op_zero();
    uint32_t left_t[1] = {0};
    uint32_t right_t[1] = {1};
    qf_transfer_op_add_term(t01, 1, left_t, right_t, &coeff);

    QfTransferVertexOperator *comm = qf_transfer_op_commutator(v0, t01);
    QfTransferVertexOperator *anti = qf_transfer_op_anti_commutator(v0, t01);

    QfTransferVertexOperator *comm_no = qf_transfer_op_normal_ordered(comm, true);
    QfTransferVertexOperator *anti_no = qf_transfer_op_normal_ordered(anti, true);

    QfTransferVertexOperator *zero = qf_transfer_op_zero();
    bool comm_nonzero = !qf_transfer_op_equiv(comm_no, zero, 1e-10);
    bool anti_vanishes = qf_transfer_op_equiv(anti_no, zero, 1e-10);

    qf_transfer_op_free(v0);
    qf_transfer_op_free(t01);
    qf_transfer_op_free(comm);
    qf_transfer_op_free(anti);
    qf_transfer_op_free(comm_no);
    qf_transfer_op_free(anti_no);
    qf_transfer_op_free(zero);

    if (!(comm_nonzero && anti_vanishes)) {
        return EqualityError;
    }
    return Ok;
}

static int test_map_to_edge_vertex_round_trip(void) {
    // The two routes to a fermionic operator must agree: going through the edge-vertex
    // representation first is equivalent to mapping directly. This is the strongest available
    // cross-check of the three transfer mappers against each other.
    QfTransferVertexOperator *op = qf_transfer_op_zero();
    uint32_t left_a[1] = {0};
    uint32_t right_a[1] = {0};
    QkComplex64 coeff_a = {1.0, 0.0};
    qf_transfer_op_add_term(op, 1, left_a, right_a, &coeff_a);
    uint32_t left_b[1] = {1};
    uint32_t right_b[1] = {2};
    QkComplex64 coeff_b = {2.0, 0.0};
    qf_transfer_op_add_term(op, 1, left_b, right_b, &coeff_b);

    QfFermionOperator *direct = qf_transfer_vertex_to_fermion(op);

    QfEdgeVertexOperator *via_edge = qf_transfer_vertex_to_edge_vertex(op);
    QfFermionOperator *indirect = qf_edge_vertex_to_fermion(via_edge);

    // Both routes emit unsimplified sums, so normal-order and simplify before comparing.
    QfFermionOperator *direct_no = qf_ferm_op_normal_ordered(direct, NULL);
    QfFermionOperator *indirect_no = qf_ferm_op_normal_ordered(indirect, NULL);
    QfFermionOperator *direct_s = qf_ferm_op_simplify(direct_no, 1e-12);
    QfFermionOperator *indirect_s = qf_ferm_op_simplify(indirect_no, 1e-12);

    bool agree = qf_ferm_op_equiv(direct_s, indirect_s, 1e-10);

    qf_transfer_op_free(op);
    qf_ferm_op_free(direct);
    qf_edge_op_free(via_edge);
    qf_ferm_op_free(indirect);
    qf_ferm_op_free(direct_no);
    qf_ferm_op_free(indirect_no);
    qf_ferm_op_free(direct_s);
    qf_ferm_op_free(indirect_s);

    if (!agree) {
        return EqualityError;
    }
    return Ok;
}

static int test_map_to_majorana(void) {
    QfTransferVertexOperator *op = qf_transfer_op_zero();
    uint32_t left[1] = {0};
    uint32_t right[1] = {0};
    QkComplex64 coeff = {1.0, 0.0};
    qf_transfer_op_add_term(op, 1, left, right, &coeff);

    QfMajoranaOperator *maj_op = qf_transfer_vertex_to_majorana(op);

    bool is_hermitian = qf_maj_op_is_hermitian(maj_op, 1e-10);
    bool non_empty = qf_maj_op_len(maj_op) > 0;

    qf_transfer_op_free(op);
    qf_maj_op_free(maj_op);

    if (!(is_hermitian && non_empty)) {
        return EqualityError;
    }
    return Ok;
}

int test_transfer_vertex_operator(void) {
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
    num_failed += RUN_TEST(test_normal_ordered_reduce);
    num_failed += RUN_TEST(test_normal_ordered_no_fusion);
    num_failed += RUN_TEST(test_is_hermitian);
    num_failed += RUN_TEST(test_len);
    num_failed += RUN_TEST(test_relabel_modes);
    num_failed += RUN_TEST(test_relabel_modes_duplicate_err);
    num_failed += RUN_TEST(test_relabel_modes_too_small_err);
    num_failed += RUN_TEST(test_groups);
    num_failed += RUN_TEST(test_canonical_order);
    num_failed += RUN_TEST(test_commutators);
    num_failed += RUN_TEST(test_map_to_edge_vertex_round_trip);
    num_failed += RUN_TEST(test_map_to_majorana);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
