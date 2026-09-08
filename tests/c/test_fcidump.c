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

int test_fcidump(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_from_file);
    num_failed += RUN_TEST(test_from_file_beta);
    num_failed += RUN_TEST(test_from_file_missing_file);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
