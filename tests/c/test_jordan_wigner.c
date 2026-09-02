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

    QkObs *result;
    if (qf_ferm_op_jordan_wigner(hamil, 4, &result) != QfExitCode_Success) {
        qf_ferm_op_free(hamil);
        return RuntimeError;
    }

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

static int test_num_qubits_too_small(void) {
    // an operator acting on mode index 3 requires at least 4 qubits
    QfFermionOperator *op = qf_ferm_op_zero();
    QkComplex64 coeff = {1.0, 0.0};
    bool action[1] = {true};
    uint32_t index[1] = {3};
    qf_ferm_op_add_term(op, 1, action, index, &coeff);

    // too few qubits must be reported instead of aborting the process
    QkObs *result = NULL;
    QfExitCode exit = qf_ferm_op_jordan_wigner(op, 3, &result);

    qf_ferm_op_free(op);

    if (exit != QfExitCode_ValueError) {
        return RuntimeError;
    }
    if (result != NULL) {
        return NullptrError;
    }
    return Ok;
}

/// Compares `result` against a single Pauli string, freeing both.
///
/// The image of every generator mapped below is one Pauli string, so the expectations are all built
/// through this rather than through the multi-term arrays the fermionic case needs.
static int check_one_string(QkObs *result, uint32_t num_qubits, QkComplex64 coeff, QkBitTerm *bits,
                            uint32_t *indices, size_t num_bits) {
    size_t boundaries[2] = {0, num_bits};
    QkObs *expected = qk_obs_new(num_qubits, 1, num_bits, &coeff, bits, indices, boundaries);

    QkComplex64 factor = {-1.0, 0.0};
    QkObs *negated = qk_obs_multiply(expected, &factor);
    QkObs *diff = qk_obs_add(result, negated);
    QkObs *canon = qk_obs_canonicalize(diff, 1e-9);
    QkObs *zero = qk_obs_zero(num_qubits);

    bool is_equal = qk_obs_equal(canon, zero);

    qk_obs_free(result);
    qk_obs_free(expected);

    return is_equal ? Ok : EqualityError;
}

static int test_majorana_mapping(void) {
    // gamma_4 sits on fermionic mode 2, behind a Z-string covering the modes below it.
    QfMajoranaOperator *op = qf_maj_op_zero();
    QkComplex64 coeff = {1.0, 0.0};
    uint32_t modes[1] = {4};
    qf_maj_op_add_term(op, 1, modes, &coeff);

    QkObs *result;
    QfExitCode exit = qf_maj_op_jordan_wigner(op, 3, &result);
    qf_maj_op_free(op);
    if (exit != QfExitCode_Success) {
        return RuntimeError;
    }

    QkBitTerm bits[3] = {QkBitTerm_Z, QkBitTerm_Z, QkBitTerm_X};
    uint32_t indices[3] = {0, 1, 2};
    return check_one_string(result, 3, coeff, bits, indices, 3);
}

static int test_edge_vertex_mapping(void) {
    // An edge operator maps onto Y..X with the Z-string spanning only the modes strictly between
    // its endpoints, and its sign flips with the index order (E_lr = -E_rl).
    QfEdgeVertexOperator *op = qf_edge_op_zero();
    QkComplex64 coeff = {1.0, 0.0};
    uint32_t left[1] = {0};
    uint32_t right[1] = {3};
    qf_edge_op_add_term(op, 1, left, right, &coeff);

    QkObs *result;
    QfExitCode exit = qf_edge_op_jordan_wigner(op, 4, &result);
    qf_edge_op_free(op);
    if (exit != QfExitCode_Success) {
        return RuntimeError;
    }

    QkComplex64 expected_coeff = {-1.0, 0.0};
    QkBitTerm bits[4] = {QkBitTerm_Y, QkBitTerm_Z, QkBitTerm_Z, QkBitTerm_X};
    uint32_t indices[4] = {0, 1, 2, 3};
    return check_one_string(result, 4, expected_coeff, bits, indices, 4);
}

static int test_transfer_vertex_mapping(void) {
    // Unlike the edge operator, reversing the indices of a transfer operator leaves the coefficient
    // at -1/2 and swaps the Pauli letters instead. This checks the reversed orientation, which is
    // the one a sign-flip assumption would get wrong.
    QfTransferVertexOperator *op = qf_transfer_op_zero();
    QkComplex64 coeff = {1.0, 0.0};
    uint32_t left[1] = {2};
    uint32_t right[1] = {0};
    qf_transfer_op_add_term(op, 1, left, right, &coeff);

    QkObs *result;
    QfExitCode exit = qf_transfer_op_jordan_wigner(op, 3, &result);
    qf_transfer_op_free(op);
    if (exit != QfExitCode_Success) {
        return RuntimeError;
    }

    QkComplex64 expected_coeff = {-0.5, 0.0};
    QkBitTerm bits[3] = {QkBitTerm_Y, QkBitTerm_Z, QkBitTerm_Y};
    uint32_t indices[3] = {0, 1, 2};
    return check_one_string(result, 3, expected_coeff, bits, indices, 3);
}

static int test_majorana_num_qubits_too_small(void) {
    // gamma_7 acts on fermionic mode 3, so it needs 4 qubits rather than 8: the bound is counted in
    // fermionic modes, not Majorana indices.
    QfMajoranaOperator *op = qf_maj_op_zero();
    QkComplex64 coeff = {1.0, 0.0};
    uint32_t modes[1] = {7};
    qf_maj_op_add_term(op, 1, modes, &coeff);

    QkObs *result = NULL;
    QfExitCode exit = qf_maj_op_jordan_wigner(op, 3, &result);
    if (exit != QfExitCode_ValueError || result != NULL) {
        qf_maj_op_free(op);
        return exit != QfExitCode_ValueError ? RuntimeError : NullptrError;
    }

    // ... and exactly enough qubits succeeds
    exit = qf_maj_op_jordan_wigner(op, 4, &result);
    qf_maj_op_free(op);
    if (exit != QfExitCode_Success) {
        return RuntimeError;
    }
    qk_obs_free(result);
    return Ok;
}

static int test_vertex_num_qubits_too_small(void) {
    // The largest index sits in the *right* buffer, which a left-only bounds check would miss.
    QfEdgeVertexOperator *edge_op = qf_edge_op_zero();
    QkComplex64 coeff = {1.0, 0.0};
    uint32_t left[1] = {0};
    uint32_t right[1] = {3};
    qf_edge_op_add_term(edge_op, 1, left, right, &coeff);

    QkObs *result = NULL;
    QfExitCode exit = qf_edge_op_jordan_wigner(edge_op, 3, &result);
    qf_edge_op_free(edge_op);
    if (exit != QfExitCode_ValueError) {
        return RuntimeError;
    }
    if (result != NULL) {
        return NullptrError;
    }

    QfTransferVertexOperator *transfer_op = qf_transfer_op_zero();
    qf_transfer_op_add_term(transfer_op, 1, left, right, &coeff);

    exit = qf_transfer_op_jordan_wigner(transfer_op, 3, &result);
    qf_transfer_op_free(transfer_op);
    if (exit != QfExitCode_ValueError) {
        return RuntimeError;
    }
    if (result != NULL) {
        return NullptrError;
    }
    return Ok;
}

int test_jordan_wigner(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_mapping);
    num_failed += RUN_TEST(test_num_qubits_too_small);
    num_failed += RUN_TEST(test_majorana_mapping);
    num_failed += RUN_TEST(test_edge_vertex_mapping);
    num_failed += RUN_TEST(test_transfer_vertex_mapping);
    num_failed += RUN_TEST(test_majorana_num_qubits_too_small);
    num_failed += RUN_TEST(test_vertex_num_qubits_too_small);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
