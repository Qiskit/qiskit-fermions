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
    // Node 0, the root, branches three ways, node 1 twice, node 4 carries one child. The root's
    // entries are ignored.
    uint32_t parents[7] = {0, 0, 0, 0, 1, 1, 4};
    QfPauliLabel labels[7] = {
        QfPauliLabel_Z, QfPauliLabel_X, QfPauliLabel_Y, QfPauliLabel_Z,
        QfPauliLabel_X, QfPauliLabel_Z, QfPauliLabel_Y,
    };

    QfTernaryTree *tree;
    if (qf_ternary_tree_new(7, 0, parents, labels, &tree) != QfExitCode_Success) {
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

    QkObs *parity;
    if (qf_ternary_tree_encoding_total_parity(chain_enc, 4, &parity) != QfExitCode_Success) {
        result = RuntimeError;
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

    if (qf_ternary_tree_encoding_total_parity(balanced_enc, 4, &parity) != QfExitCode_Success) {
        result = RuntimeError;
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

// Builds the number operator of a single mode.
static QfFermionOperator *number_operator(uint32_t mode) {
    QfFermionOperator *op = qf_ferm_op_zero();
    QkComplex64 coeff = {1.0, 0.0};
    bool actions[2] = {true, false};
    uint32_t indices[2] = {mode, mode};
    qf_ferm_op_add_term(op, 2, actions, indices, &coeff);
    return op;
}

// Maps the number operator of `mode` through `encoding` onto four qubits.
static QkObs *map_number_operator(const QfTernaryTreeEncoding *encoding, uint32_t mode) {
    QfFermionOperator *op = number_operator(mode);
    QkObs *mapped = NULL;
    qf_ferm_op_ternary_tree(op, encoding, 4, &mapped);
    qf_ferm_op_free(op);
    return mapped;
}

// A permuted mode map changes which mode each node serves. Under the reversed map node `u` serves
// mode `3 - u`, so the image of mode 0 through it is the image of mode 3 through the identity map,
// and vice versa. Neither equals the unpermuted image of the same mode.
static int test_mode_map(void) {
    QfPauliLabel order[3] = {QfPauliLabel_X, QfPauliLabel_Y, QfPauliLabel_Z};
    QfTernaryTree *tree;
    if (qf_ternary_tree_breadth_first(4, 3, order, &tree) != QfExitCode_Success) {
        return RuntimeError;
    }

    QfTernaryTreeEncoding *identity;
    uint32_t mode_map[4] = {3, 2, 1, 0};
    QfTernaryTreeEncoding *reversed;
    if (qf_ternary_tree_encoding_new(tree, NULL, &identity) != QfExitCode_Success) {
        qf_ternary_tree_free(tree);
        return RuntimeError;
    }
    if (qf_ternary_tree_encoding_new(tree, mode_map, &reversed) != QfExitCode_Success) {
        qf_ternary_tree_encoding_free(identity);
        qf_ternary_tree_free(tree);
        return RuntimeError;
    }

    int result = Ok;
    if (qf_ternary_tree_encoding_num_modes(reversed) != 4) {
        result = EqualityError;
    }

    QkObs *reversed_0 = map_number_operator(reversed, 0);
    QkObs *reversed_3 = map_number_operator(reversed, 3);
    QkObs *identity_0 = map_number_operator(identity, 0);
    QkObs *identity_3 = map_number_operator(identity, 3);
    if (!reversed_0 || !reversed_3 || !identity_0 || !identity_3) {
        result = RuntimeError;
    } else {
        if (!obs_equal(reversed_0, identity_3, 4) || !obs_equal(reversed_3, identity_0, 4)) {
            result = EqualityError;
        }
        if (obs_equal(reversed_0, identity_0, 4)) {
            result = EqualityError;
        }
    }
    qk_obs_free(reversed_0);
    qk_obs_free(reversed_3);
    qk_obs_free(identity_0);
    qk_obs_free(identity_3);

    qf_ternary_tree_encoding_free(identity);
    qf_ternary_tree_encoding_free(reversed);
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

    // A root that is not a node.
    uint32_t pair_parents[2] = {0, 0};
    QfPauliLabel pair_labels[2] = {QfPauliLabel_Z, QfPauliLabel_X};
    if (qf_ternary_tree_new(2, 2, pair_parents, pair_labels, &tree) != QfExitCode_ValueError) {
        return EqualityError;
    }

    // The same label used twice on one node.
    uint32_t duplicate_parents[3] = {0, 0, 0};
    QfPauliLabel duplicate_labels[3] = {QfPauliLabel_Z, QfPauliLabel_X, QfPauliLabel_X};
    if (qf_ternary_tree_new(3, 0, duplicate_parents, duplicate_labels, &tree) !=
        QfExitCode_ValueError) {
        return EqualityError;
    }

    // A parent out of range.
    uint32_t out_of_range_parents[2] = {0, 7};
    if (qf_ternary_tree_new(2, 0, out_of_range_parents, pair_labels, &tree) !=
        QfExitCode_ValueError) {
        return EqualityError;
    }

    // A node unreachable from the root: a cycle that does not include it.
    uint32_t cycle_parents[3] = {0, 2, 1};
    QfPauliLabel cycle_labels[3] = {QfPauliLabel_Z, QfPauliLabel_X, QfPauliLabel_X};
    if (qf_ternary_tree_new(3, 0, cycle_parents, cycle_labels, &tree) != QfExitCode_ValueError) {
        return EqualityError;
    }

    // A null array.
    if (qf_ternary_tree_new(2, 0, NULL, pair_labels, &tree) != QfExitCode_NullPointerError) {
        return EqualityError;
    }
    if (qf_ternary_tree_new(2, 0, pair_parents, NULL, &tree) != QfExitCode_NullPointerError) {
        return EqualityError;
    }

    // An empty specification is invalid as a tree, not as a pointer: a zero-length array may be
    // NULL, and is reported the same way as a non-NULL one.
    if (qf_ternary_tree_new(0, 0, NULL, NULL, &tree) != QfExitCode_ValueError) {
        return EqualityError;
    }
    if (qf_ternary_tree_new(0, 0, pair_parents, pair_labels, &tree) != QfExitCode_ValueError) {
        return EqualityError;
    }

    // A misaligned output pointer is reported rather than written through. The buffer is never
    // dereferenced as a pointer, so offsetting into it by one byte is safe to construct. The union
    // gives the buffer a pointer's alignment without `_Alignas`, which MSVC only knows in C11 mode.
    union {
        QfTernaryTree *alignment;
        unsigned char bytes[2 * sizeof(QfTernaryTree *)];
    } storage;
    QfTernaryTree **misaligned = (QfTernaryTree **)(storage.bytes + 1);
    if (qf_ternary_tree_chain(2, QfPauliLabel_Z, misaligned) != QfExitCode_AlignmentError) {
        return EqualityError;
    }

    // A branching outside 1..3 is an error, not silently the branching=3 tree.
    QfPauliLabel all_labels[3] = {QfPauliLabel_X, QfPauliLabel_Y, QfPauliLabel_Z};
    if (qf_ternary_tree_breadth_first(7, 4, all_labels, &tree) != QfExitCode_ValueError) {
        return EqualityError;
    }
    if (qf_ternary_tree_breadth_first(7, 0, all_labels, &tree) != QfExitCode_ValueError) {
        return EqualityError;
    }

    // A null tree is reported rather than aborting, as is a null output pointer.
    QfTernaryTreeEncoding *never_written = NULL;
    if (qf_ternary_tree_encoding_new(NULL, NULL, &never_written) != QfExitCode_NullPointerError) {
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

    // An under-sized register is reported rather than aborting in qk_obs_new.
    QkObs *parity = NULL;
    if (qf_ternary_tree_encoding_total_parity(encoding, 2, &parity) != QfExitCode_ValueError ||
        parity != NULL) {
        result = EqualityError;
    }
    if (qf_ternary_tree_encoding_total_parity(NULL, 4, &parity) != QfExitCode_NullPointerError) {
        result = EqualityError;
    }
    if (qf_ternary_tree_encoding_total_parity(encoding, 4, NULL) != QfExitCode_NullPointerError) {
        result = EqualityError;
    }

    // Null inputs to a mapper are reported rather than aborting.
    if (qf_ferm_op_ternary_tree(NULL, encoding, 4, &unused) != QfExitCode_NullPointerError) {
        result = EqualityError;
    }
    if (qf_ferm_op_ternary_tree(too_wide, NULL, 4, &unused) != QfExitCode_NullPointerError) {
        result = EqualityError;
    }
    if (qf_ferm_op_ternary_tree(too_wide, encoding, 4, NULL) != QfExitCode_NullPointerError) {
        result = EqualityError;
    }

    // The encoding's own weight, which always agrees with the tree's.
    if (qf_ternary_tree_encoding_max_weight(encoding) != qf_ternary_tree_max_weight(tree)) {
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

// Label bytes are validated rather than trusted, whether read through a pointer or passed by value.
// Before C23 a QfPauliLabel is a plain uint8_t, so any byte can arrive, and the root's label is
// documented as ignored.
static int test_label_validation(void) {
    QfTernaryTree *tree;

    // A root whose entries are left zero-initialized: neither the label, which is not a
    // QfPauliLabel value, nor the parent is read.
    uint32_t parents[2] = {0, 0};
    QfPauliLabel zeroed_root[2] = {0, QfPauliLabel_X};
    if (qf_ternary_tree_new(2, 0, parents, zeroed_root, &tree) != QfExitCode_Success) {
        return EqualityError;
    }
    qf_ternary_tree_free(tree);

    // The root need not come first, and it is its entries, wherever it sits, that are ignored.
    uint32_t last_root_parents[2] = {1, 0};
    QfPauliLabel last_root_labels[2] = {QfPauliLabel_X, 0};
    if (qf_ternary_tree_new(2, 1, last_root_parents, last_root_labels, &tree) !=
        QfExitCode_Success) {
        return EqualityError;
    }
    qf_ternary_tree_free(tree);

    // A non-root entry with an invalid label is reported, whether it is the identity's natural
    // encoding or simply out of range.
    uint8_t invalid[2] = {0, 7};
    for (int i = 0; i < 2; i++) {
        QfPauliLabel bad_label[2] = {QfPauliLabel_Z, invalid[i]};
        if (qf_ternary_tree_new(2, 0, parents, bad_label, &tree) != QfExitCode_ValueError) {
            return EqualityError;
        }
    }

    // A label passed by value to the chain constructor is validated just the same.
    for (int i = 0; i < 2; i++) {
        if (qf_ternary_tree_chain(4, invalid[i], &tree) != QfExitCode_ValueError) {
            return EqualityError;
        }
    }

    // An invalid entry of the breadth-first order is reported too.
    QfPauliLabel bad_order[3] = {QfPauliLabel_X, 0, QfPauliLabel_Z};
    if (qf_ternary_tree_breadth_first(4, 3, bad_order, &tree) != QfExitCode_ValueError) {
        return EqualityError;
    }

    // So is a label repeated among the entries that are actually used, but not beyond them.
    QfPauliLabel repeated_order[3] = {QfPauliLabel_X, QfPauliLabel_X, QfPauliLabel_Z};
    if (qf_ternary_tree_breadth_first(4, 2, repeated_order, &tree) != QfExitCode_ValueError) {
        return EqualityError;
    }
    if (qf_ternary_tree_breadth_first(4, 1, repeated_order, &tree) != QfExitCode_Success) {
        return EqualityError;
    }
    qf_ternary_tree_free(tree);

    return Ok;
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
    num_failed += RUN_TEST(test_label_validation);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
