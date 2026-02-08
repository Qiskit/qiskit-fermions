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

static int test_from_file(void) {
    QfFCIDump *fcidump = qf_fcidump_from_file("../../h2.fcidump");
    QfFermionOperator *op = qf_ferm_op_from_fcidump(fcidump);

    // TODO: actually assert all results!

    uint32_t norb = qf_fcidump_norb(fcidump);
    uint32_t nelec = qf_fcidump_nelec(fcidump);
    uint32_t ms2 = qf_fcidump_ms2(fcidump);

    bool is_correct = (norb == 2) && (nelec == 2) && (ms2 == 0);

    qf_fcidump_free(fcidump);
    qf_ferm_op_free(op);

    if (!is_correct) {
        return EqualityError;
    }
    return Ok;
}

static int test_from_file_beta(void) {
    QfFCIDump *fcidump = qf_fcidump_from_file("../../heh.fcidump");
    QfFermionOperator *op = qf_ferm_op_from_fcidump(fcidump);

    // TODO: actually assert all results!

    uint32_t norb = qf_fcidump_norb(fcidump);
    uint32_t nelec = qf_fcidump_nelec(fcidump);
    uint32_t ms2 = qf_fcidump_ms2(fcidump);

    bool is_correct = (norb == 2) && (nelec == 3) && (ms2 == 1);

    qf_fcidump_free(fcidump);
    qf_ferm_op_free(op);

    if (!is_correct) {
        return EqualityError;
    }
    return Ok;
}

int test_fcidump(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_from_file);
    num_failed += RUN_TEST(test_from_file_beta);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
