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

// Returns whether two observables represent the same operator, consuming neither.
//
// Compares by canonicalizing the difference against zero rather than structurally, since the
// mappers make no promise about the order or multiplicity of the terms they emit.
static bool obs_equal(const QkObs *left, const QkObs *right, uint32_t num_qubits) {
    QkComplex64 minus_one = {-1.0, 0.0};
    QkObs *negated = qk_obs_multiply(right, &minus_one);
    QkObs *diff = qk_obs_add(left, negated);
    QkObs *canonical = qk_obs_canonicalize(diff, 1e-9);
    QkObs *zero = qk_obs_zero(num_qubits);

    bool equal = qk_obs_equal(canonical, zero);

    qk_obs_free(negated);
    qk_obs_free(diff);
    qk_obs_free(canonical);
    qk_obs_free(zero);
    return equal;
}

// Builds a small hopping Hamiltonian on four modes.
static QfFermionOperator *hopping_hamiltonian(void) {
    QfFermionOperator *hamil = qf_ferm_op_zero();
    QkComplex64 coeffs[2] = {{1.0, 0.0}, {1.0, 0.0}};
    bool actions[4] = {true, false, true, false};
    uint32_t indices[4] = {0, 3, 3, 0};
    for (int i = 0; i < 2; i++) {
        qf_ferm_op_add_term(hamil, 2, actions + 2 * i, indices + 2 * i, &coeffs[i]);
    }
    return hamil;
}

// A chain along Z is the Jordan-Wigner encoding, so mapping through it must reproduce
// `qf_ferm_op_jordan_wigner` exactly. This is the regression that ties the generic framework to a
// mapper verified independently against Qiskit.
static int test_chain_matches_jordan_wigner(void) {
    QfFermionOperator *hamil = hopping_hamiltonian();

    QfTernaryTree *tree;
    if (qf_ternary_tree_chain(4, QfPauliLabel_Z, &tree) != QfExitCode_Success) {
        qf_ferm_op_free(hamil);
        return RuntimeError;
    }

    QfTernaryTreeEncoding *encoding;
    if (qf_ternary_tree_encoding_new(tree, NULL, &encoding) != QfExitCode_Success) {
        qf_ternary_tree_free(tree);
        qf_ferm_op_free(hamil);
        return RuntimeError;
    }

    QkObs *via_tree;
    QkObs *via_jw;
    int result = Ok;
    if (qf_ferm_op_ternary_tree(hamil, encoding, 4, &via_tree) != QfExitCode_Success) {
        result = RuntimeError;
    } else {
        if (qf_ferm_op_jordan_wigner(hamil, 4, &via_jw) != QfExitCode_Success) {
            result = RuntimeError;
        } else {
            if (!obs_equal(via_tree, via_jw, 4)) {
                result = EqualityError;
            }
            qk_obs_free(via_jw);
        }
        qk_obs_free(via_tree);
    }

    qf_ternary_tree_encoding_free(encoding);
    qf_ternary_tree_free(tree);
    qf_ferm_op_free(hamil);
    return result;
}

// The balanced tree's Pauli weight is logarithmic where a chain's is linear. This is the whole
// point of choosing a tree, so it is asserted on the structure rather than inferred from a mapped
// operator.
static int test_balanced_tree_is_shallower(void) {
    QfPauliLabel order[3] = {QfPauliLabel_X, QfPauliLabel_Y, QfPauliLabel_Z};

    QfTernaryTree *balanced;
    if (qf_ternary_tree_breadth_first(13, 3, order, &balanced) != QfExitCode_Success) {
        return RuntimeError;
    }
    QfTernaryTree *chain;
    if (qf_ternary_tree_chain(13, QfPauliLabel_Z, &chain) != QfExitCode_Success) {
        qf_ternary_tree_free(balanced);
        return RuntimeError;
    }

    int result = Ok;
    // ceil(log3(2 * 13 + 1)) == 3, against 13 for the chain.
    if (qf_ternary_tree_max_weight(balanced) != 3) {
        result = EqualityError;
    }
    if (qf_ternary_tree_max_weight(chain) != 13) {
        result = EqualityError;
    }
    // Both use one qubit per mode, and both have 2N+1 legs.
    if (qf_ternary_tree_num_nodes(balanced) != 13 || qf_ternary_tree_num_nodes(chain) != 13) {
        result = EqualityError;
    }
    if (qf_ternary_tree_num_legs(balanced) != 27 || qf_ternary_tree_num_legs(chain) != 27) {
        result = EqualityError;
    }

    qf_ternary_tree_free(chain);
    qf_ternary_tree_free(balanced);
    return result;
}

// An arbitrary, mixed-branching tree built from a parent-and-label specification -- the general
// interface, and the shape neither convenience constructor produces.
static int test_arbitrary_tree(void) {
    // Node 0 branches three ways, node 1 twice, node 4 carries one child.
    QfTernaryTreeNode spec[7] = {
        {0, QfPauliLabel_Z, true}, // the root; parent and label are ignored
        {0, QfPauliLabel_X, false}, {0, QfPauliLabel_Y, false}, {0, QfPauliLabel_Z, false},
        {1, QfPauliLabel_X, false}, {1, QfPauliLabel_Z, false}, {4, QfPauliLabel_Y, false},
    };

    QfTernaryTree *tree;
    if (qf_ternary_tree_new(7, spec, &tree) != QfExitCode_Success) {
        return RuntimeError;
    }

    int result = Ok;
    if (qf_ternary_tree_num_nodes(tree) != 7) {
        result = EqualityError;
    }
    // 2N+1 holds whatever the shape.
    if (qf_ternary_tree_num_legs(tree) != 15) {
        result = EqualityError;
    }
    if (qf_ternary_tree_max_weight(tree) != 4) {
        result = EqualityError;
    }

    qf_ternary_tree_free(tree);
    return result;
}

// Every operator type has a ternary-tree mapper, and mapping a Majorana operator directly must
// agree with the Jordan-Wigner mapper on the chain.
static int test_all_operator_types(void) {
    QfTernaryTree *tree;
    if (qf_ternary_tree_chain(4, QfPauliLabel_Z, &tree) != QfExitCode_Success) {
        return RuntimeError;
    }
    QfTernaryTreeEncoding *encoding;
    if (qf_ternary_tree_encoding_new(tree, NULL, &encoding) != QfExitCode_Success) {
        qf_ternary_tree_free(tree);
        return RuntimeError;
    }

    int result = Ok;

    // A Majorana operator.
    QfMajoranaOperator *maj_op = qf_maj_op_zero();
    QkComplex64 maj_coeff = {0.5, 0.0};
    uint32_t maj_modes[2] = {0, 3};
    qf_maj_op_add_term(maj_op, 2, maj_modes, &maj_coeff);

    QkObs *maj_tree;
    QkObs *maj_jw;
    if (qf_maj_op_ternary_tree(maj_op, encoding, 4, &maj_tree) != QfExitCode_Success) {
        result = RuntimeError;
    } else {
        if (qf_maj_op_jordan_wigner(maj_op, 4, &maj_jw) != QfExitCode_Success) {
            result = RuntimeError;
        } else {
            if (!obs_equal(maj_tree, maj_jw, 4)) {
                result = EqualityError;
            }
            qk_obs_free(maj_jw);
        }
        qk_obs_free(maj_tree);
    }
    qf_maj_op_free(maj_op);

    // An edge-vertex operator.
    QfEdgeVertexOperator *edge_op = qf_edge_op_zero();
    QkComplex64 edge_coeff = {1.0, 0.0};
    uint32_t edge_left[1] = {0};
    uint32_t edge_right[1] = {2};
    qf_edge_op_add_term(edge_op, 1, edge_left, edge_right, &edge_coeff);

    QkObs *edge_tree;
    QkObs *edge_jw;
    if (qf_edge_op_ternary_tree(edge_op, encoding, 4, &edge_tree) != QfExitCode_Success) {
        result = RuntimeError;
    } else {
        if (qf_edge_op_jordan_wigner(edge_op, 4, &edge_jw) != QfExitCode_Success) {
            result = RuntimeError;
        } else {
            if (!obs_equal(edge_tree, edge_jw, 4)) {
                result = EqualityError;
            }
            qk_obs_free(edge_jw);
        }
        qk_obs_free(edge_tree);
    }
    qf_edge_op_free(edge_op);

    // A transfer-vertex operator.
    QfTransferVertexOperator *transfer_op = qf_transfer_op_zero();
    QkComplex64 transfer_coeff = {1.0, 0.0};
    uint32_t transfer_left[1] = {1};
    uint32_t transfer_right[1] = {3};
    qf_transfer_op_add_term(transfer_op, 1, transfer_left, transfer_right, &transfer_coeff);

    QkObs *transfer_tree;
    QkObs *transfer_jw;
    if (qf_transfer_op_ternary_tree(transfer_op, encoding, 4, &transfer_tree) !=
        QfExitCode_Success) {
        result = RuntimeError;
    } else {
        if (qf_transfer_op_jordan_wigner(transfer_op, 4, &transfer_jw) != QfExitCode_Success) {
            result = RuntimeError;
        } else {
            if (!obs_equal(transfer_tree, transfer_jw, 4)) {
                result = EqualityError;
            }
            qk_obs_free(transfer_jw);
        }
        qk_obs_free(transfer_tree);
    }
    qf_transfer_op_free(transfer_op);

    qf_ternary_tree_encoding_free(encoding);
    qf_ternary_tree_free(tree);
    return result;
}

// The leftover leg is the total fermionic parity. On the chain it is the full Z string; on the
// balanced tree it is not, which is the trap this pins down.
static int test_total_parity(void) {
    int result = Ok;

    QfTernaryTree *chain;
    if (qf_ternary_tree_chain(4, QfPauliLabel_Z, &chain) != QfExitCode_Success) {
        return RuntimeError;
    }
    QfTernaryTreeEncoding *chain_enc;
    if (qf_ternary_tree_encoding_new(chain, NULL, &chain_enc) != QfExitCode_Success) {
        qf_ternary_tree_free(chain);
        return RuntimeError;
    }

    QkObs *parity = qf_ternary_tree_encoding_total_parity(chain_enc, 4);
    if (parity == NULL) {
        result = NullptrError;
    } else {
        // Z_0 Z_1 Z_2 Z_3 with coefficient +1.
        QkComplex64 coeffs[1] = {{1.0, 0.0}};
        QkBitTerm bit_terms[4] = {QkBitTerm_Z, QkBitTerm_Z, QkBitTerm_Z, QkBitTerm_Z};
        uint32_t indices[4] = {0, 1, 2, 3};
        size_t boundaries[2] = {0, 4};
        QkObs *expected = qk_obs_new(4, 1, 4, coeffs, bit_terms, indices, boundaries);

        if (!obs_equal(parity, expected, 4)) {
            result = EqualityError;
        }
        qk_obs_free(expected);
        qk_obs_free(parity);
    }

    qf_ternary_tree_encoding_free(chain_enc);
    qf_ternary_tree_free(chain);
    if (result != Ok) {
        return result;
    }

    // On the balanced tree at four modes the parity string is Z_0 Z_3, not the all-Z string: the
    // leftover leg is the all-Z *path*, which acts as the identity on nodes not on that path.
    QfPauliLabel order[3] = {QfPauliLabel_X, QfPauliLabel_Y, QfPauliLabel_Z};
    QfTernaryTree *balanced;
    if (qf_ternary_tree_breadth_first(4, 3, order, &balanced) != QfExitCode_Success) {
        return RuntimeError;
    }
    QfTernaryTreeEncoding *balanced_enc;
    if (qf_ternary_tree_encoding_new(balanced, NULL, &balanced_enc) != QfExitCode_Success) {
        qf_ternary_tree_free(balanced);
        return RuntimeError;
    }

    parity = qf_ternary_tree_encoding_total_parity(balanced_enc, 4);
    if (parity == NULL) {
        result = NullptrError;
    } else {
        QkComplex64 coeffs[1] = {{1.0, 0.0}};
        QkBitTerm bit_terms[2] = {QkBitTerm_Z, QkBitTerm_Z};
        uint32_t indices[2] = {0, 3};
        size_t boundaries[2] = {0, 2};
        QkObs *expected = qk_obs_new(4, 1, 2, coeffs, bit_terms, indices, boundaries);

        if (!obs_equal(parity, expected, 4)) {
            result = EqualityError;
        }
        qk_obs_free(expected);
        qk_obs_free(parity);
    }

    qf_ternary_tree_encoding_free(balanced_enc);
    qf_ternary_tree_free(balanced);
    return result;
}

// A permuted mode map changes which mode each node serves.
static int test_mode_map(void) {
    QfPauliLabel order[3] = {QfPauliLabel_X, QfPauliLabel_Y, QfPauliLabel_Z};
    QfTernaryTree *tree;
    if (qf_ternary_tree_breadth_first(4, 3, order, &tree) != QfExitCode_Success) {
        return RuntimeError;
    }

    uint32_t mode_map[4] = {3, 2, 1, 0};
    QfTernaryTreeEncoding *encoding;
    int result = Ok;
    if (qf_ternary_tree_encoding_new(tree, mode_map, &encoding) != QfExitCode_Success) {
        result = RuntimeError;
    } else {
        if (qf_ternary_tree_encoding_num_modes(encoding) != 4) {
            result = EqualityError;
        }
        qf_ternary_tree_encoding_free(encoding);
    }

    qf_ternary_tree_free(tree);
    return result;
}

// Extra qubits are padded with the identity, matching the Jordan-Wigner mappers.
static int test_padding(void) {
    QfFermionOperator *hamil = hopping_hamiltonian();

    QfTernaryTree *tree;
    if (qf_ternary_tree_chain(4, QfPauliLabel_Z, &tree) != QfExitCode_Success) {
        qf_ferm_op_free(hamil);
        return RuntimeError;
    }
    QfTernaryTreeEncoding *encoding;
    if (qf_ternary_tree_encoding_new(tree, NULL, &encoding) != QfExitCode_Success) {
        qf_ternary_tree_free(tree);
        qf_ferm_op_free(hamil);
        return RuntimeError;
    }

    int result = Ok;
    if (qf_ternary_tree_encoding_num_modes(encoding) != 4) {
        result = EqualityError;
    }

    QkObs *via_tree;
    QkObs *via_jw;
    if (qf_ferm_op_ternary_tree(hamil, encoding, 6, &via_tree) != QfExitCode_Success) {
        result = RuntimeError;
    } else {
        if (qf_ferm_op_jordan_wigner(hamil, 6, &via_jw) != QfExitCode_Success) {
            result = RuntimeError;
        } else {
            if (!obs_equal(via_tree, via_jw, 6)) {
                result = EqualityError;
            }
            qk_obs_free(via_jw);
        }
        qk_obs_free(via_tree);
    }

    qf_ternary_tree_encoding_free(encoding);
    qf_ternary_tree_free(tree);
    qf_ferm_op_free(hamil);
    return result;
}

// Malformed specifications and out-of-range arguments must be reported rather than aborting.
static int test_errors(void) {
    QfTernaryTree *tree;

    // No nodes at all.
    if (qf_ternary_tree_chain(0, QfPauliLabel_Z, &tree) != QfExitCode_ValueError) {
        return EqualityError;
    }

    // Two roots.
    QfTernaryTreeNode two_roots[2] = {
        {0, QfPauliLabel_Z, true},
        {0, QfPauliLabel_Z, true},
    };
    if (qf_ternary_tree_new(2, two_roots, &tree) != QfExitCode_ValueError) {
        return EqualityError;
    }

    // The same label used twice on one node.
    QfTernaryTreeNode duplicate[3] = {
        {0, QfPauliLabel_Z, true},
        {0, QfPauliLabel_X, false},
        {0, QfPauliLabel_X, false},
    };
    if (qf_ternary_tree_new(3, duplicate, &tree) != QfExitCode_ValueError) {
        return EqualityError;
    }

    // A parent out of range.
    QfTernaryTreeNode out_of_range[2] = {
        {0, QfPauliLabel_Z, true},
        {7, QfPauliLabel_X, false},
    };
    if (qf_ternary_tree_new(2, out_of_range, &tree) != QfExitCode_ValueError) {
        return EqualityError;
    }

    // A null specification.
    if (qf_ternary_tree_new(2, NULL, &tree) != QfExitCode_NullPointerError) {
        return EqualityError;
    }

    // A branching above 3 is an error, not silently the branching=3 tree.
    QfPauliLabel all_labels[3] = {QfPauliLabel_X, QfPauliLabel_Y, QfPauliLabel_Z};
    if (qf_ternary_tree_breadth_first(7, 4, all_labels, &tree) != QfExitCode_ValueError) {
        return EqualityError;
    }

    if (qf_ternary_tree_chain(4, QfPauliLabel_Z, &tree) != QfExitCode_Success) {
        return RuntimeError;
    }
    QfTernaryTreeEncoding *encoding;
    int result = Ok;

    // A mode map that is not a permutation.
    uint32_t bad_map[4] = {0, 1, 1, 2};
    if (qf_ternary_tree_encoding_new(tree, bad_map, &encoding) != QfExitCode_ValueError) {
        result = EqualityError;
    }

    // An operator acting outside the encoding's modes.
    if (qf_ternary_tree_encoding_new(tree, NULL, &encoding) != QfExitCode_Success) {
        qf_ternary_tree_free(tree);
        return RuntimeError;
    }
    QfFermionOperator *too_wide = qf_ferm_op_zero();
    QkComplex64 coeff = {1.0, 0.0};
    bool action[1] = {true};
    uint32_t index[1] = {7};
    qf_ferm_op_add_term(too_wide, 1, action, index, &coeff);

    QkObs *unused;
    if (qf_ferm_op_ternary_tree(too_wide, encoding, 4, &unused) != QfExitCode_ValueError) {
        result = EqualityError;
    }

    // Too few qubits for the encoding: now a mapper-level error rather than a compile-time one.
    QfFermionOperator *in_range = qf_ferm_op_zero();
    uint32_t low_index[1] = {3};
    qf_ferm_op_add_term(in_range, 1, action, low_index, &coeff);
    if (qf_ferm_op_ternary_tree(in_range, encoding, 3, &unused) != QfExitCode_ValueError) {
        result = EqualityError;
    }
    qf_ferm_op_free(in_range);

    // An operator on mode 0 alone still needs the encoding's whole register: a mode's Pauli image
    // spans its path to the root. Bounding only by the operator's modes used to abort the process.
    QfFermionOperator *lowest = qf_ferm_op_zero();
    uint32_t zero_index[1] = {0};
    qf_ferm_op_add_term(lowest, 1, action, zero_index, &coeff);
    for (uint32_t num_qubits = 1; num_qubits < 4; num_qubits++) {
        if (qf_ferm_op_ternary_tree(lowest, encoding, num_qubits, &unused) !=
            QfExitCode_ValueError) {
            result = EqualityError;
        }
    }
    qf_ferm_op_free(lowest);

    // An under-sized register is reported as NULL rather than aborting in qk_obs_new.
    if (qf_ternary_tree_encoding_total_parity(encoding, 2) != NULL) {
        result = EqualityError;
    }

    // The encoding's own weight, restricted to the paired legs.
    if (qf_ternary_tree_encoding_max_weight(encoding) != 4) {
        result = EqualityError;
    }

    qf_ferm_op_free(too_wide);
    qf_ternary_tree_encoding_free(encoding);
    qf_ternary_tree_free(tree);

    // Freeing a null pointer is a no-op.
    qf_ternary_tree_free(NULL);
    qf_ternary_tree_encoding_free(NULL);

    return result;
}

int test_ternary_tree(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_chain_matches_jordan_wigner);
    num_failed += RUN_TEST(test_balanced_tree_is_shallower);
    num_failed += RUN_TEST(test_arbitrary_tree);
    num_failed += RUN_TEST(test_all_operator_types);
    num_failed += RUN_TEST(test_total_parity);
    num_failed += RUN_TEST(test_mode_map);
    num_failed += RUN_TEST(test_padding);
    num_failed += RUN_TEST(test_errors);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
