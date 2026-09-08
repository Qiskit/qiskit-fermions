# This code is a Qiskit project.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for analysing an existing grouping of operator terms."""

from pathlib import Path

import pytest
from qiskit_fermions.operators import (
    EdgeVertexOperator,
    FermionOperator,
    MajoranaOperator,
    TransferVertexOperator,
    ann,
    cre,
)
from qiskit_fermions.operators.library import FCIDump
from qiskit_fermions.operators.terms.grouping import (
    group_coeff_means,
    group_terms_by_electronic_structure,
    groups_are_hermitian,
    groups_have_uniform_coeffs,
)

ANALYSIS_FUNCTIONS = [
    group_coeff_means,
    groups_are_hermitian,
    groups_have_uniform_coeffs,
]

# One grouped operator per supported type, so that every routine is exercised through the same
# generic code path it takes in ``core`` rather than only on ``FermionOperator``.
GROUPED_OPERATORS = [
    FermionOperator(
        [1.0, 1.0],
        [True, False, True, False],
        [0, 1, 1, 0],
        [0, 2, 4],
    ),
    MajoranaOperator([1.0, 1.0], [0, 1, 2, 3], [0, 2, 4]),
    EdgeVertexOperator([1.0, 1.0], [0, 2], [1, 3], [0, 1, 2]),
    TransferVertexOperator([1.0, 1.0], [0, 2], [1, 3], [0, 1, 2]),
]


def _hopping_pair_plus_unpaired():
    """Returns ``a†_0 a_1 + a†_1 a_0 + a†_2 a_3`` with the first two terms grouped together.

    The leading pair are each other's adjoint, so group 0 is Hermitian; the trailing term is
    unpaired, so group 1 is not.
    """
    op = FermionOperator(
        [1.0, 1.0, 1.0],
        [True, False, True, False, True, False],
        [0, 1, 1, 0, 2, 3],
        [0, 2, 4, 6],
    )
    op.groups = [0, 0, 1]
    return op


@pytest.mark.parametrize("func", ANALYSIS_FUNCTIONS)
def test_returns_none_when_ungrouped(func):
    op = FermionOperator.from_dict({(cre(0), ann(1)): 1.0})
    assert not op.has_groups()
    assert func(op) is None


@pytest.mark.parametrize("func", ANALYSIS_FUNCTIONS)
def test_rejects_unsupported_type(func):
    with pytest.raises(TypeError, match="expects a fermionic, Majorana, edge-vertex or transfer"):
        func("not an operator")


@pytest.mark.parametrize("op", GROUPED_OPERATORS, ids=lambda op: type(op).__name__)
def test_every_operator_type_is_supported(op):
    op.groups = [0, 0]

    assert group_coeff_means(op) == [1.0]
    assert len(groups_are_hermitian(op)) == 1
    assert groups_have_uniform_coeffs(op) == [True]


class TestGroupCoeffMeans:
    """Tests for :func:`.group_coeff_means`."""

    def test_averages_magnitudes(self, subtests):
        # built positionally so that the term order (and hence the pairing with the group
        # indices) is fixed, pinning the exact expected averages: group 0 is
        # (|1.0| + |-3.0 + 4.0j|) / 2 == (1.0 + 5.0) / 2, and group 1 is the lone |2.0|. Note that
        # the magnitude, not the real part, is what gets averaged.
        op = FermionOperator(
            [1.0, -3.0 + 4.0j, 2.0],
            [True, False, True, False, True, False],
            [0, 1, 1, 0, 2, 3],
            [0, 2, 4, 6],
        )

        with subtests.test("assigned"):
            op.groups = [0, 0, 1]
            assert group_coeff_means(op) == [3.0, 2.0]

        with subtests.test("sparse group index"):
            # group 1 is carried by no term at all, so it weighs 0.0 rather than NaN
            op.groups = [0, 0, 2]
            assert group_coeff_means(op) == [3.0, 0.0, 2.0]

    def test_empty_list(self):
        empty = FermionOperator.zero()
        empty.groups = []
        assert group_coeff_means(empty) == []


class TestGroupsAreHermitian:
    """Tests for :func:`.groups_are_hermitian`."""

    def test_per_group_result(self):
        assert groups_are_hermitian(_hopping_pair_plus_unpaired()) == [True, False]

    def test_agrees_with_split_out_groups(self):
        op = _hopping_pair_plus_unpaired()
        assert groups_are_hermitian(op) == [g.is_hermitian() for g in op.split_out_groups()]

    def test_sparse_group_index_is_hermitian(self):
        # a group index carried by no term holds the zero operator, which is Hermitian
        op = _hopping_pair_plus_unpaired()
        op.groups = [0, 0, 2]
        assert groups_are_hermitian(op) == [True, True, False]

    def test_atol_is_honoured(self, subtests):
        op = FermionOperator.from_dict(
            {
                (cre(0), ann(1)): 1.00001j,
                (cre(1), ann(0)): -1j,
            }
        )
        op.groups = [0, 0]

        with subtests.test("default tolerance"):
            assert groups_are_hermitian(op) == [False]

        with subtests.test("loosened tolerance"):
            assert groups_are_hermitian(op, atol=1e-4) == [True]


class TestGroupsHaveUniformCoeffs:
    """Tests for :func:`.groups_have_uniform_coeffs`."""

    def test_compares_magnitudes_by_default(self, subtests):
        op = FermionOperator(
            [1.0, -1.0, 2.0],
            [True, False, True, False, True, False],
            [0, 1, 1, 0, 2, 3],
            [0, 2, 4, 6],
        )
        op.groups = [0, 0, 1]

        with subtests.test("magnitudes"):
            # group 0 mixes +1 and -1: equal in magnitude, but not as coefficients
            assert groups_have_uniform_coeffs(op) == [True, True]

        with subtests.test("exact coefficients"):
            assert groups_have_uniform_coeffs(op, abs=False) == [False, True]

    def test_atol_is_honoured(self, subtests):
        op = FermionOperator(
            [1.0, 1.00001],
            [True, False, True, False],
            [0, 1, 1, 0],
            [0, 2, 4],
        )
        op.groups = [0, 0]

        with subtests.test("default tolerance"):
            assert groups_have_uniform_coeffs(op, abs=False) == [False]

        with subtests.test("loosened tolerance"):
            assert groups_have_uniform_coeffs(op, atol=1e-4, abs=False) == [True]

    def test_empty_list(self):
        empty = FermionOperator.zero()
        empty.groups = []
        assert groups_have_uniform_coeffs(empty) == []


class TestChecksAreIndependent:
    """The Hermiticity and uniformity checks imply each other in neither direction."""

    def test_uniform_but_not_hermitian(self):
        # two independent raising terms: identical coefficients, but not each other's adjoint
        op = FermionOperator.from_dict(
            {
                (cre(0), ann(1)): 1.0,
                (cre(2), ann(3)): 1.0,
            }
        )
        op.groups = [0, 0]

        assert groups_have_uniform_coeffs(op) == [True]
        assert groups_are_hermitian(op) == [False]

    def test_hermitian_but_not_uniform(self):
        # two conjugate pairs at different scales, merged into one group
        op = FermionOperator(
            [1.0, 1.0, 5.0, 5.0],
            [True, False, True, False, True, False, True, False],
            [0, 1, 1, 0, 2, 3, 3, 2],
            [0, 2, 4, 6, 8],
        )
        op.groups = [0, 0, 0, 0]

        assert groups_are_hermitian(op) == [True]
        assert groups_have_uniform_coeffs(op) == [False]

    def test_hermitian_conjugate_pair_with_complex_coeffs(self):
        # equal magnitudes but unequal coefficients, so only the `abs` form calls it uniform
        op = FermionOperator.from_dict(
            {
                (cre(0), ann(1)): 1.0j,
                (cre(1), ann(0)): -1.0j,
            }
        )
        op.groups = [0, 0]

        assert groups_are_hermitian(op) == [True]
        assert groups_have_uniform_coeffs(op) == [True]
        assert groups_have_uniform_coeffs(op, abs=False) == [False]


def test_electronic_structure_grouping_passes_every_check():
    """The grouping this library produces satisfies both conventions the checks describe.

    This is the end-to-end case motivating the helpers: the groups that
    :func:`.group_terms_by_electronic_structure` assigns are exactly the single-generator groups for
    which :func:`.group_coeff_means` is the atomic group magnitude.
    """
    file_path = Path(__file__).parent / "../../../../h2.fcidump"
    fcidump = FCIDump.from_file(str(file_path))

    op = FermionOperator.from_fcidump(fcidump)
    normal = op.normal_ordered().simplify(atol=1e-16)
    group_terms_by_electronic_structure(normal, 2 * fcidump.norb)

    assert all(groups_are_hermitian(normal))
    assert all(groups_have_uniform_coeffs(normal))
    assert len(group_coeff_means(normal)) == normal.num_groups()
