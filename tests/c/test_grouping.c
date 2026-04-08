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

static int test_grouping_error(void) {
    uint64_t num_terms = 1;
    uint64_t num_actions = 1;
    bool actions[1] = {true};
    uint32_t modes[1] = {0};
    QkComplex64 coeffs[1] = {{1.0, 0.0}};
    uint32_t boundaries[2] = {0, 1};
    QfFermionOperator *op =
        qf_ferm_op_new(num_terms, num_actions, coeffs, actions, modes, boundaries);

    QfExitCode exit = qf_group_terms_by_electronic_structure(op, 2, false);

    qf_ferm_op_free(op);

    if (exit != QfExitCode_ValueError) {
        return RuntimeError;
    }
    return Ok;
}

static int test_group_terms_by_electronic_structure(void) {
    QfFCIDump *fcidump = qf_fcidump_from_file("../../h2.fcidump");
    QfFermionOperator *op = qf_ferm_op_from_fcidump(fcidump);

    QfFermionOperator *normal = qf_ferm_op_normal_ordered(op);
    qf_ferm_op_free(op);

    QfExitCode exit = qf_group_terms_by_electronic_structure(normal, 4, false);
    qf_ferm_op_free(normal);

    if (exit != QfExitCode_Success) {
        return RuntimeError;
    }
    return Ok;
}

int test_grouping(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_grouping_error);
    num_failed += RUN_TEST(test_group_terms_by_electronic_structure);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
