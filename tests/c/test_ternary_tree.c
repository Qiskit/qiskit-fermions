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
#include <math.h>
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

// Returns whether `node`'s links that end in a leg are exactly `expected`, in X, Y, Z order. The C
// API exposes no parent or child accessor, so this is how the tests below see where a link went.
static bool has_free_labels(const QfTernaryTree *tree, uint32_t node, const QfPauliLabel *expected,
                            uint64_t num_expected) {
    QfPauliLabel labels[3];
    uint64_t num_labels = 0;
    if (qf_ternary_tree_leg_labels(tree, node, labels, &num_labels) != QfExitCode_Success ||
        num_labels != num_expected) {
        return false;
    }
    for (uint64_t i = 0; i < num_labels; i++) {
        if (labels[i] != expected[i]) {
            return false;
        }
    }
    return true;
}

// The tree edit operations: a subtree move, a re-root, and a relabel. Each returns a *new* tree, so
// the ownership rule these exercise is that both the input and the output must be freed.
static int test_tree_edits(void) {
    QfTernaryTree *chain;
    if (qf_ternary_tree_chain(6, QfPauliLabel_Z, &chain) != QfExitCode_Success) {
        return RuntimeError;
    }

    int result = Ok;

    // A chain's interior nodes spend Z on the next node, so X and Y are free; the tail has all
    // three. Labels come back in X, Y, Z order.
    QfPauliLabel free_labels[3];
    uint64_t num_free = 0;
    if (qf_ternary_tree_leg_labels(chain, 0, free_labels, &num_free) != QfExitCode_Success) {
        result = RuntimeError;
    } else if (num_free != 2 || free_labels[0] != QfPauliLabel_X ||
               free_labels[1] != QfPauliLabel_Y) {
        result = EqualityError;
    }
    if (qf_ternary_tree_leg_labels(chain, 5, free_labels, &num_free) != QfExitCode_Success) {
        result = RuntimeError;
    } else if (num_free != 3 || free_labels[0] != QfPauliLabel_X ||
               free_labels[1] != QfPauliLabel_Y || free_labels[2] != QfPauliLabel_Z) {
        result = EqualityError;
    }

    const QfPauliLabel x_y[2] = {QfPauliLabel_X, QfPauliLabel_Y};
    const QfPauliLabel y_only[1] = {QfPauliLabel_Y};
    const QfPauliLabel y_z[2] = {QfPauliLabel_Y, QfPauliLabel_Z};

    // Moving a node to the link it already hangs off changes nothing: node 1 still occupies the
    // root's Z link.
    QfTernaryTree *unchanged;
    if (qf_ternary_tree_reparent(chain, 1, 0, QfPauliLabel_Z, &unchanged) != QfExitCode_Success) {
        result = RuntimeError;
    } else {
        if (!has_free_labels(unchanged, 0, x_y, 2) ||
            qf_ternary_tree_max_weight(unchanged) != qf_ternary_tree_max_weight(chain)) {
            result = EqualityError;
        }
        qf_ternary_tree_free(unchanged);
    }

    // Moving the chain's tail onto a free link at the root lowers the Pauli weight.
    QfTernaryTree *moved;
    if (qf_ternary_tree_reparent(chain, 5, 0, QfPauliLabel_X, &moved) != QfExitCode_Success) {
        result = RuntimeError;
    } else {
        if (qf_ternary_tree_max_weight(chain) != 6) {
            result = EqualityError;
        }
        if (qf_ternary_tree_max_weight(moved) != 5) {
            result = EqualityError;
        }
        // The leg count survives every edit, whatever the shape.
        if (qf_ternary_tree_num_legs(moved) != 13) {
            result = EqualityError;
        }
        qf_ternary_tree_free(moved);
    }

    // Re-rooting a 6-chain at an interior node keeps the node and leg counts and makes it
    // shallower. Node 3 keeps its Z child, so the reversed link to node 2 takes the lowest free
    // slot, X, leaving only Y free; node 2 then finds its Z slot free and keeps it for node 1.
    QfTernaryTree *rerooted;
    if (qf_ternary_tree_reroot(chain, 3, &rerooted) != QfExitCode_Success) {
        result = RuntimeError;
    } else {
        if (qf_ternary_tree_num_nodes(rerooted) != 6 || qf_ternary_tree_num_legs(rerooted) != 13) {
            result = EqualityError;
        }
        if (qf_ternary_tree_max_weight(rerooted) >= qf_ternary_tree_max_weight(chain)) {
            result = EqualityError;
        }
        if (!has_free_labels(rerooted, 3, y_only, 1) || !has_free_labels(rerooted, 2, x_y, 2)) {
            result = EqualityError;
        }
        qf_ternary_tree_free(rerooted);
    }

    // Relabeling the root by (Y, Z, X) moves its Z child to the X link, leaving Y and Z free.
    QfTernaryTree *relabeled;
    QfPauliLabel permutation[3] = {QfPauliLabel_Y, QfPauliLabel_Z, QfPauliLabel_X};
    if (qf_ternary_tree_relabel(chain, 0, permutation, &relabeled) != QfExitCode_Success) {
        result = RuntimeError;
    } else {
        if (!has_free_labels(relabeled, 0, y_z, 2)) {
            result = EqualityError;
        }
        qf_ternary_tree_free(relabeled);
    }

    qf_ternary_tree_free(chain);
    return result;
}

// Checks that an edit failed with `expected`. Should it unexpectedly succeed, the tree it wrote to
// `*out` is freed, so a regression shows up as a failure rather than a leak.
static int expect_edit_error(QfExitCode code, QfExitCode expected, QfTernaryTree **out) {
    if (code == QfExitCode_Success) {
        qf_ternary_tree_free(*out);
        *out = NULL;
    }
    return code == expected ? Ok : EqualityError;
}

// Every precondition of the edits must be reported rather than aborting.
static int test_tree_edit_errors(void) {
    QfPauliLabel order[3] = {QfPauliLabel_X, QfPauliLabel_Y, QfPauliLabel_Z};
    QfTernaryTree *tree;
    if (qf_ternary_tree_breadth_first(13, 3, order, &tree) != QfExitCode_Success) {
        return RuntimeError;
    }

    int result = Ok;
    QfTernaryTree *out = NULL;
    const QfExitCode value = QfExitCode_ValueError;
    const QfExitCode null = QfExitCode_NullPointerError;

    // The root has no parent to be moved away from.
    result |=
        expect_edit_error(qf_ternary_tree_reparent(tree, 0, 5, QfPauliLabel_X, &out), value, &out);
    // A node cannot move under itself. Node 4 is a leaf, so its X link is free and only the
    // self-move check can reject the move.
    result |=
        expect_edit_error(qf_ternary_tree_reparent(tree, 4, 4, QfPauliLabel_X, &out), value, &out);
    // Node 0's X link already carries a child in a branching-3 tree.
    result |=
        expect_edit_error(qf_ternary_tree_reparent(tree, 5, 0, QfPauliLabel_X, &out), value, &out);
    // Out of range.
    result |=
        expect_edit_error(qf_ternary_tree_reparent(tree, 99, 0, QfPauliLabel_X, &out), value, &out);
    // Node 1 of this tree has three children, so it cannot take a fourth link for its old parent.
    result |= expect_edit_error(qf_ternary_tree_reroot(tree, 1, &out), value, &out);
    result |= expect_edit_error(qf_ternary_tree_reroot(tree, 99, &out), value, &out);
    // Not a permutation.
    QfPauliLabel repeated[3] = {QfPauliLabel_X, QfPauliLabel_X, QfPauliLabel_Z};
    result |= expect_edit_error(qf_ternary_tree_relabel(tree, 0, repeated, &out), value, &out);
    // A value that is no Pauli label at all, as a label or inside a permutation.
    result |= expect_edit_error(qf_ternary_tree_reparent(tree, 4, 5, 9, &out), value, &out);
    QfPauliLabel invalid[3] = {QfPauliLabel_X, QfPauliLabel_Y, 9};
    result |= expect_edit_error(qf_ternary_tree_relabel(tree, 0, invalid, &out), value, &out);
    // Leg labels of an out-of-range node.
    QfPauliLabel labels[3];
    uint64_t len = 0;
    if (qf_ternary_tree_leg_labels(tree, 99, labels, &len) != QfExitCode_ValueError) {
        result = EqualityError;
    }

    // Null pointers are reported, not dereferenced, whether the output or the tree itself.
    result |=
        expect_edit_error(qf_ternary_tree_reparent(tree, 5, 6, QfPauliLabel_X, NULL), null, &out);
    result |=
        expect_edit_error(qf_ternary_tree_reparent(NULL, 5, 6, QfPauliLabel_X, &out), null, &out);
    result |= expect_edit_error(qf_ternary_tree_reroot(NULL, 0, &out), null, &out);
    result |= expect_edit_error(qf_ternary_tree_relabel(NULL, 0, order, &out), null, &out);
    result |= expect_edit_error(qf_ternary_tree_relabel(tree, 0, NULL, &out), null, &out);
    if (qf_ternary_tree_leg_labels(NULL, 0, labels, &len) != QfExitCode_NullPointerError ||
        qf_ternary_tree_leg_labels(tree, 0, NULL, &len) != QfExitCode_NullPointerError ||
        qf_ternary_tree_leg_labels(tree, 0, labels, NULL) != QfExitCode_NullPointerError) {
        result = EqualityError;
    }

    qf_ternary_tree_free(tree);
    return result;
}

// HATT: a tree grown around the term structure of a Hamiltonian.
static int test_hatt_tree(void) {
    int outcome = Ok;

    // A nearest-neighbor hopping Hamiltonian on four modes, in the Majorana representation the
    // algorithm works in.
    QfMajoranaOperator *ham = qf_maj_op_zero();
    QkComplex64 coeff = {1.0, 0.0};
    for (uint32_t mode = 0; mode < 3; mode++) {
        uint32_t modes[2] = {2 * mode, 2 * (mode + 1) + 1};
        qf_maj_op_add_term(ham, 2, modes, &coeff);
    }

    QfTernaryTree *tree;
    if (qf_ternary_tree_hatt(ham, 4, false, &tree) != QfExitCode_Success) {
        qf_maj_op_free(ham);
        return RuntimeError;
    }
    if (qf_ternary_tree_num_nodes(tree) != 4 || qf_ternary_tree_num_legs(tree) != 9) {
        outcome = EqualityError;
    }

    // The result is an ordinary tree, owned by the caller, so it compiles and maps like any other.
    QfTernaryTreeEncoding *encoding;
    if (qf_ternary_tree_encoding_new(tree, NULL, &encoding) != QfExitCode_Success) {
        outcome = RuntimeError;
    } else {
        QkObs *mapped;
        if (qf_maj_op_ternary_tree(ham, encoding, 4, &mapped) != QfExitCode_Success) {
            outcome = RuntimeError;
        } else {
            qk_obs_free(mapped);
        }
        qf_ternary_tree_encoding_free(encoding);
    }
    qf_ternary_tree_free(tree);

    // The weighted flag reaches the search: with three negligible terms and one dominant one,
    // counting terms and weighting them grow different trees. Counted, mode 0 ends up a leaf, with
    // all three links free; weighted, it becomes the root and keeps only X free.
    {
        QfMajoranaOperator *skewed = qf_maj_op_zero();
        uint32_t skewed_modes[4][2] = {{0, 3}, {1, 2}, {2, 5}, {0, 4}};
        QkComplex64 small = {1e-3, 0.0};
        QkComplex64 large = {1.0, 0.0};
        for (int i = 0; i < 4; i++) {
            qf_maj_op_add_term(skewed, 2, skewed_modes[i], i < 3 ? &small : &large);
        }
        const QfPauliLabel all[3] = {QfPauliLabel_X, QfPauliLabel_Y, QfPauliLabel_Z};
        const QfPauliLabel x_only[1] = {QfPauliLabel_X};
        QfTernaryTree *counted, *weighted;
        if (qf_ternary_tree_hatt(skewed, 3, false, &counted) != QfExitCode_Success) {
            outcome = RuntimeError;
        } else {
            if (!has_free_labels(counted, 0, all, 3)) {
                outcome = EqualityError;
            }
            qf_ternary_tree_free(counted);
        }
        if (qf_ternary_tree_hatt(skewed, 3, true, &weighted) != QfExitCode_Success) {
            outcome = RuntimeError;
        } else {
            if (!has_free_labels(weighted, 0, x_only, 1)) {
                outcome = EqualityError;
            }
            qf_ternary_tree_free(weighted);
        }
        qf_maj_op_free(skewed);
    }

    // Zero modes, and a Hamiltonian wider than the requested encoding, are both reported.
    QfTernaryTree *rejected = NULL;
    outcome |= expect_edit_error(qf_ternary_tree_hatt(ham, 0, false, &rejected),
                                 QfExitCode_ValueError, &rejected);
    outcome |= expect_edit_error(qf_ternary_tree_hatt(ham, 2, false, &rejected),
                                 QfExitCode_ValueError, &rejected);
    // So are null pointers, rather than being dereferenced.
    outcome |= expect_edit_error(qf_ternary_tree_hatt(NULL, 4, false, &rejected),
                                 QfExitCode_NullPointerError, &rejected);
    outcome |= expect_edit_error(qf_ternary_tree_hatt(ham, 4, false, NULL),
                                 QfExitCode_NullPointerError, &rejected);

    // A NaN coefficient is rejected.
    QkComplex64 nan_coeff = {NAN, 0.0};
    uint32_t nan_modes[2] = {0, 1};
    qf_maj_op_add_term(ham, 2, nan_modes, &nan_coeff);
    outcome |= expect_edit_error(qf_ternary_tree_hatt(ham, 4, false, &rejected),
                                 QfExitCode_ValueError, &rejected);

    qf_maj_op_free(ham);
    return outcome;
}

int test_ternary_tree(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_tree_edits);
    num_failed += RUN_TEST(test_tree_edit_errors);
    num_failed += RUN_TEST(test_hatt_tree);
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
