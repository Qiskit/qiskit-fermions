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

// The exact coefficients of the operator built from each fixture are pinned by the Rust unittests
// (`test_to_fermion_operator` / `..._beta` in `crates/core/src/operators/library/fcidump.rs`), so
// re-listing all 40 (resp. 72) of them here would only duplicate that ground truth more weakly.
// These tests instead assert the header fields plus the structural invariants of the resulting
// operator (its term and group counts, and the physical properties an electronic-structure
// Hamiltonian must have), which is what the C binding itself is responsible for conveying.

static int test_from_file(void) {
    QfFCIDump *fcidump = NULL;
    if (qf_fcidump_from_file("../../h2.fcidump", &fcidump) != QfExitCode_Success) {
        return RuntimeError;
    }
    QfFermionOperator *op = qf_ferm_op_from_fcidump(fcidump);

    bool is_correct = (qf_fcidump_norb(fcidump) == 2) && (qf_fcidump_nelec(fcidump) == 2) &&
                      (qf_fcidump_ms2(fcidump) == 0);

    // A spin-symmetric H2 FCIDump carries no beta blocks, so the operator is the one-plus-two-body
    // sum over the alpha integrals only. The leading term is the nuclear-repulsion constant, which
    // the file does carry and which occupies a group of its own: 1 + 40 terms in 1 + 22 groups.
    is_correct = is_correct && (qf_ferm_op_len(op) == 41);
    is_correct = is_correct && qf_ferm_op_has_groups(op) && (qf_ferm_op_num_groups(op) == 23);
    // A real Hamiltonian is Hermitian, conserves particle number, and is at most two-body.
    is_correct = is_correct && qf_ferm_op_is_hermitian(op, 1e-10);
    is_correct = is_correct && qf_ferm_op_conserves_particle_number(op);
    is_correct = is_correct && (qf_ferm_op_max_rank(op) == 4);

    qf_fcidump_free(fcidump);
    qf_ferm_op_free(op);

    if (!is_correct) {
        return EqualityError;
    }
    return Ok;
}

static int test_from_file_beta(void) {
    QfFCIDump *fcidump = NULL;
    if (qf_fcidump_from_file("../../heh.fcidump", &fcidump) != QfExitCode_Success) {
        return RuntimeError;
    }
    QfFermionOperator *op = qf_ferm_op_from_fcidump(fcidump);

    bool is_correct = (qf_fcidump_norb(fcidump) == 2) && (qf_fcidump_nelec(fcidump) == 3) &&
                      (qf_fcidump_ms2(fcidump) == 1);

    // The HeH fixture is spin-polarised, so it carries explicit beta blocks: the operator gains the
    // separate alpha/beta one- and two-body terms, giving 1 + 72 terms across 1 + 36 groups (again
    // counting the constant).
    is_correct = is_correct && (qf_ferm_op_len(op) == 73);
    is_correct = is_correct && qf_ferm_op_has_groups(op) && (qf_ferm_op_num_groups(op) == 37);
    is_correct = is_correct && qf_ferm_op_is_hermitian(op, 1e-10);
    is_correct = is_correct && qf_ferm_op_conserves_particle_number(op);
    is_correct = is_correct && (qf_ferm_op_max_rank(op) == 4);

    qf_fcidump_free(fcidump);
    qf_ferm_op_free(op);

    if (!is_correct) {
        return EqualityError;
    }
    return Ok;
}

/// A path that does not exist must be reported through the exit code, leaving `out` untouched,
/// rather than unwinding a Rust panic across the FFI boundary.
static int test_from_file_missing_file(void) {
    QfFCIDump *fcidump = NULL;
    QfExitCode exit = qf_fcidump_from_file("../../does_not_exist.fcidump", &fcidump);

    if (exit != QfExitCode_ValueError) {
        return RuntimeError;
    }
    if (fcidump != NULL) {
        return NullptrError;
    }
    return Ok;
}

/// The packed integral getters borrow the FCIDump's own buffers, so the exact values are already
/// pinned by the Rust unittests. What the binding is responsible for is the length of each array
/// and the absence convention for the beta blocks, which is what these assert.
static int test_get_packed_arrays(void) {
    QfFCIDump *fcidump = NULL;
    if (qf_fcidump_from_file("../../h2.fcidump", &fcidump) != QfExitCode_Success) {
        return RuntimeError;
    }

    // norb = 2, so npair = 3 and the S8 length is npair * (npair + 1) / 2 = 6.
    double *one_body_a;
    uint64_t one_body_a_len;
    qf_fcidump_get_one_body_tril_a(fcidump, &one_body_a, &one_body_a_len);

    double *two_body_aa;
    uint64_t two_body_aa_len;
    qf_fcidump_get_two_body_tril_aa(fcidump, &two_body_aa, &two_body_aa_len);

    bool is_correct = (one_body_a_len == 3) && (two_body_aa_len == 6);
    is_correct = is_correct && (one_body_a[0] == -1.2563390730032502);
    is_correct = is_correct && (two_body_aa[0] == 0.6757101548035165);

    // A spin-symmetric FCIDump carries no beta blocks at all.
    is_correct = is_correct && !qf_fcidump_is_unrestricted(fcidump);

    // The constant is independent of the beta blocks: this restricted file does carry one.
    is_correct = is_correct && qf_fcidump_has_constant(fcidump);
    is_correct = is_correct && (qf_fcidump_constant(fcidump) == 0.7199689944489797);

    qf_fcidump_free(fcidump);

    if (!is_correct) {
        return EqualityError;
    }
    return Ok;
}

/// The beta blocks appear together, and `two_body_ab` is packed differently from its siblings: it
/// is only 4-fold symmetric, so it holds the full `(npair, npair)` matrix rather than its lower
/// triangle. The off-diagonal check is what would catch a transposed alpha/beta orientation.
static int test_get_packed_arrays_beta(void) {
    QfFCIDump *fcidump = NULL;
    if (qf_fcidump_from_file("../../heh.fcidump", &fcidump) != QfExitCode_Success) {
        return RuntimeError;
    }

    bool is_correct = qf_fcidump_is_unrestricted(fcidump);

    double *one_body_b;
    uint64_t one_body_b_len;
    qf_fcidump_get_one_body_tril_b(fcidump, &one_body_b, &one_body_b_len);

    double *two_body_ab;
    uint64_t two_body_ab_len;
    qf_fcidump_get_two_body_tril_ab(fcidump, &two_body_ab, &two_body_ab_len);

    double *two_body_bb;
    uint64_t two_body_bb_len;
    qf_fcidump_get_two_body_tril_bb(fcidump, &two_body_bb, &two_body_bb_len);

    // npair = 3, so the S8 blocks hold 6 elements while the S4 block holds npair * npair = 9.
    is_correct = is_correct && (one_body_b_len == 3) && (two_body_bb_len == 6);
    is_correct = is_correct && (two_body_ab_len == 9);
    is_correct = is_correct && (one_body_b[0] == -2.6172710340816154);
    is_correct = is_correct && (two_body_bb[0] == 0.9643310447658793);
    // Index 3 is row 1, column 0 of the (3, 3) matrix, whose transpose partner at index 1 differs.
    is_correct = is_correct && (two_body_ab[3] == -0.1830105072322512);
    is_correct = is_correct && (two_body_ab[1] == -0.16158571043232664);

    qf_fcidump_free(fcidump);

    if (!is_correct) {
        return EqualityError;
    }
    return Ok;
}

/// A file without an all-zero index record carries no constant, which must be reported through the
/// predicate rather than as a zero (the two are different facts).
static int test_no_constant(void) {
    QfFCIDump *fcidump = NULL;
    if (qf_fcidump_from_file("../../no_constant.fcidump", &fcidump) != QfExitCode_Success) {
        return RuntimeError;
    }

    bool is_correct = !qf_fcidump_has_constant(fcidump);

    qf_fcidump_free(fcidump);

    if (!is_correct) {
        return EqualityError;
    }
    return Ok;
}

int test_fcidump(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_from_file);
    num_failed += RUN_TEST(test_from_file_beta);
    num_failed += RUN_TEST(test_from_file_missing_file);
    num_failed += RUN_TEST(test_get_packed_arrays);
    num_failed += RUN_TEST(test_get_packed_arrays_beta);
    num_failed += RUN_TEST(test_no_constant);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
