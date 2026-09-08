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

    QfExitCode exit = qf_ferm_op_group_terms_by_electronic_structure(op, 2, false);

    qf_ferm_op_free(op);

    if (exit != QfExitCode_ValueError) {
        return RuntimeError;
    }
    return Ok;
}

static int test_group_terms_by_electronic_structure(void) {
    QfFCIDump *fcidump = qf_fcidump_from_file("../../h2.fcidump");
    QfFermionOperator *op = qf_ferm_op_from_fcidump(fcidump);
    qf_fcidump_free(fcidump);

    QfFermionOperator *normal = qf_ferm_op_normal_ordered(op, NULL);
    qf_ferm_op_free(op);

    QfExitCode exit = qf_ferm_op_group_terms_by_electronic_structure(normal, 4, false);
    qf_ferm_op_free(normal);

    if (exit != QfExitCode_Success) {
        return RuntimeError;
    }
    return Ok;
}

static int test_group_coeff_means(void) {
    uint64_t num_terms = 4;
    uint64_t num_actions = 8;
    bool actions[8] = {true, false, true, false, true, false, true, false};
    uint32_t modes[8] = {0, 1, 2, 3, 1, 0, 3, 2};
    // the second term of each group is the negative of the first, so averaging the magnitudes
    // (rather than the signed coefficients) is what makes the means non-zero
    QkComplex64 coeffs[4] = {{1.0, 0.0}, {2.0, 0.0}, {-1.0, 0.0}, {-2.0, 0.0}};
    uint32_t boundaries[5] = {0, 2, 4, 6, 8};
    QfFermionOperator *op =
        qf_ferm_op_new(num_terms, num_actions, coeffs, actions, modes, boundaries);

    uint32_t groups_in[4] = {0, 1, 0, 1};
    qf_ferm_op_set_groups(op, groups_in, num_terms);

    double means[2];
    qf_ferm_op_group_coeff_means(op, means);

    // both averages are exactly representable, so an exact comparison is safe here
    bool correct_mean0 = means[0] == 1.0;
    bool correct_mean1 = means[1] == 2.0;

    // a group index carried by no term weighs 0.0 rather than NaN
    uint32_t sparse_groups[4] = {0, 2, 0, 2};
    qf_ferm_op_set_groups(op, sparse_groups, num_terms);

    double sparse_means[3];
    qf_ferm_op_group_coeff_means(op, sparse_means);

    bool correct_sparse0 = sparse_means[0] == 1.0;
    bool correct_sparse1 = sparse_means[1] == 0.0;
    bool correct_sparse2 = sparse_means[2] == 2.0;

    bool passed_all =
        correct_mean0 && correct_mean1 && correct_sparse0 && correct_sparse1 && correct_sparse2;

    qf_ferm_op_free(op);

    if (!passed_all) {
        return EqualityError;
    }
    return Ok;
}

static int test_set_groups_err(void) {
    uint64_t num_terms = 2;
    uint64_t num_actions = 4;
    bool actions[4] = {true, false, true, false};
    uint32_t modes[4] = {0, 1, 1, 0};
    QkComplex64 coeffs[2] = {{1.0, 0.0}, {1.0, 0.0}};
    uint32_t boundaries[3] = {0, 2, 4};
    QfFermionOperator *op =
        qf_ferm_op_new(num_terms, num_actions, coeffs, actions, modes, boundaries);

    uint32_t groups_in[3] = {0, 1, 2};

    // one index per term is accepted
    bool correct_ok = qf_ferm_op_set_groups(op, groups_in, 2) == QfExitCode_Success;

    // a short array would silently drop the trailing term wherever terms are iterated with their
    // groups, and a long one would make `num_groups` report groups that no term carries
    bool correct_short = qf_ferm_op_set_groups(op, groups_in, 1) == QfExitCode_ValueError;
    bool correct_long = qf_ferm_op_set_groups(op, groups_in, 3) == QfExitCode_ValueError;

    // a rejected assignment leaves the previously accepted grouping in place
    bool unchanged = qf_ferm_op_num_groups(op) == 2;

    bool passed_all = correct_ok && correct_short && correct_long && unchanged;

    qf_ferm_op_free(op);

    if (!passed_all) {
        return EqualityError;
    }
    return Ok;
}

static int test_groups_are_hermitian(void) {
    // a†_0 a_1 and a†_1 a_0 are each other's adjoint, so group 0 is Hermitian; the trailing
    // a†_2 a_3 is unpaired, so group 1 is not
    uint64_t num_terms = 3;
    uint64_t num_actions = 6;
    bool actions[6] = {true, false, true, false, true, false};
    uint32_t modes[6] = {0, 1, 1, 0, 2, 3};
    QkComplex64 coeffs[3] = {{1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}};
    uint32_t boundaries[4] = {0, 2, 4, 6};
    QfFermionOperator *op =
        qf_ferm_op_new(num_terms, num_actions, coeffs, actions, modes, boundaries);

    uint32_t groups_in[3] = {0, 0, 1};
    qf_ferm_op_set_groups(op, groups_in, num_terms);

    bool hermitian[2];
    qf_ferm_op_groups_are_hermitian(op, 1e-8, hermitian);
    bool correct = hermitian[0] && !hermitian[1];

    // a group index carried by no term holds the zero operator, which is Hermitian
    uint32_t sparse_groups[3] = {0, 0, 2};
    qf_ferm_op_set_groups(op, sparse_groups, num_terms);

    bool sparse_hermitian[3];
    qf_ferm_op_groups_are_hermitian(op, 1e-8, sparse_hermitian);
    bool correct_sparse = sparse_hermitian[0] && sparse_hermitian[1] && !sparse_hermitian[2];

    bool passed_all = correct && correct_sparse;

    qf_ferm_op_free(op);

    if (!passed_all) {
        return EqualityError;
    }
    return Ok;
}

static int test_groups_have_uniform_coeffs(void) {
    // group 0 mixes +1 and -1: equal in magnitude, but not as coefficients
    uint64_t num_terms = 3;
    uint64_t num_actions = 6;
    bool actions[6] = {true, false, true, false, true, false};
    uint32_t modes[6] = {0, 1, 1, 0, 2, 3};
    QkComplex64 coeffs[3] = {{1.0, 0.0}, {-1.0, 0.0}, {2.0, 0.0}};
    uint32_t boundaries[4] = {0, 2, 4, 6};
    QfFermionOperator *op =
        qf_ferm_op_new(num_terms, num_actions, coeffs, actions, modes, boundaries);

    uint32_t groups_in[3] = {0, 0, 1};
    qf_ferm_op_set_groups(op, groups_in, num_terms);

    bool uniform_abs[2];
    qf_ferm_op_groups_have_uniform_coeffs(op, 1e-8, true, uniform_abs);
    bool correct_abs = uniform_abs[0] && uniform_abs[1];

    bool uniform_exact[2];
    qf_ferm_op_groups_have_uniform_coeffs(op, 1e-8, false, uniform_exact);
    bool correct_exact = !uniform_exact[0] && uniform_exact[1];

    bool passed_all = correct_abs && correct_exact;

    qf_ferm_op_free(op);

    if (!passed_all) {
        return EqualityError;
    }
    return Ok;
}

int test_grouping(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_grouping_error);
    num_failed += RUN_TEST(test_group_terms_by_electronic_structure);
    num_failed += RUN_TEST(test_group_coeff_means);
    num_failed += RUN_TEST(test_set_groups_err);
    num_failed += RUN_TEST(test_groups_are_hermitian);
    num_failed += RUN_TEST(test_groups_have_uniform_coeffs);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
