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

import itertools
import pickle

import numpy as np
import pytest
from qiskit_fermions.operators import MajoranaOperator, gamma
from qiskit_fermions.operators.library import anti_commutator, commutator

from .majorana_matrix_oracle import majorana, majorana_operator_matrix, vertex_matrix
from .operator_contract_tests import OperatorContractTests


class TestMajoranaOperator(OperatorContractTests):
    @staticmethod
    def get_class() -> type[MajoranaOperator]:
        return MajoranaOperator

    def test_max_rank(self, subtests):
        cls = self.get_class()

        op = cls.zero()

        with subtests.test("0 for additive identity"):
            assert op.max_rank() == 0

        op += cls.one()

        with subtests.test("0 for multiplicative identity"):
            assert op.max_rank() == 0

        op += cls.from_dict({(0, 1): 1})

        with subtests.test("2"):
            assert op.max_rank() == 2

        op += cls.from_dict({(0, 1, 2, 3): 1})

        with subtests.test("4"):
            assert op.max_rank() == 4

    def test_getters(self, subtests):
        cls = self.get_class()

        coeffs = [1e-10, 2, 3, 4, -4]
        modes = [0, 0, 1, 1]
        boundaries = [0, 0, 1, 2, 3, 4]

        op = cls(coeffs, modes, boundaries)

        with subtests.test("coeffs"):
            assert np.allclose(op.get_coeffs(), coeffs)
        with subtests.test("modes"):
            assert np.all(op.get_modes() == modes)
        with subtests.test("boundaries"):
            assert np.all(op.get_boundaries() == boundaries)

    def test_get_support(self):
        cls = self.get_class()
        op = cls.from_dict({(0, 4): 1, (1, 3, 4, 7): 1})
        assert op.get_support() == {0, 1, 3, 4, 7}

    def test_richcmp(self, subtests):
        cls = self.get_class()

        # two terms, each of length two (boundaries has len(coeffs) + 1 entries)
        coeffs = [1, 2]
        modes = [0, 1, 0, 1]
        boundaries = [0, 2, 4]

        op = cls(coeffs, modes, boundaries)

        with subtests.test("equal"):
            other = cls(coeffs, modes, boundaries)
            # `!=` must be the exact negation of `==`
            assert (op == other) is True
            assert (op != other) is False

        # each field differs individually: `!=` must be the negation of `==`.
        for name, other in [
            ("coeffs", cls([1, 3], modes, boundaries)),
            ("modes", cls(coeffs, [1, 0, 0, 1], boundaries)),
            ("boundaries", cls(coeffs, modes, [0, 1, 4])),
        ]:
            with subtests.test(name):
                assert (op == other) is False
                assert (op != other) is True

    def test_repr(self):
        cls = self.get_class()
        op = cls.from_dict(
            {
                (): 2,
                (gamma(0, False),): 1,
                (gamma(0, False), gamma(0, True)): 0.5,
                (gamma(1, False), gamma(0, True)): -0.5j,
                (gamma(1, True), gamma(1, False)): 1 - 0.5j,
            }
        )
        assert op.equiv(eval(repr(op)))

    def test_len(self, subtests):
        cls = self.get_class()

        with subtests.test("len==0"):
            assert len(cls.zero()) == 0

        with subtests.test("len==1"):
            assert len(cls.one()) == 1

        with subtests.test("len==2"):
            op = cls.from_dict({(): 1, (gamma(0, False), gamma(0, True)): 1})
            assert len(op) == 2

    def test_from_terms(self, subtests):
        cls = self.get_class()
        op = cls.from_dict(
            {
                (): 2,
                (gamma(0, False),): 1,
                (gamma(0, False), gamma(0, True)): 0.5,
                (gamma(1, False), gamma(0, True)): -0.5j,
                (gamma(1, True), gamma(1, False)): 1 - 0.5j,
            }
        )
        with subtests.test("iterator"):
            assert op.equiv(cls.from_terms(op.iter_terms()))
        with subtests.test("list"):
            assert op.equiv(cls.from_terms(list(op.iter_terms())))

    def test_from_terms_with_groups(self, subtests):
        cls = self.get_class()
        op = cls.from_dict(
            {
                (): 2,
                (gamma(0, False),): 1,
                (gamma(0, False), gamma(0, True)): 0.5,
                (gamma(1, False), gamma(0, True)): -0.5j,
                (gamma(1, True), gamma(1, False)): 1 - 0.5j,
            }
        )
        op.groups = [0, 1, 2, 3, 4]
        with subtests.test("iterator"):
            reconstructed = cls.from_terms_with_groups(op.iter_terms_with_groups())
            assert op.equiv(reconstructed)
            assert op.groups == reconstructed.groups
        with subtests.test("list"):
            reconstructed = cls.from_terms_with_groups(list(op.iter_terms_with_groups()))
            assert op.equiv(reconstructed)
            assert op.groups == reconstructed.groups

    def test_pickle(self, subtests):
        cls = self.get_class()
        op = cls.from_dict({(0, 1): 1.0, (2, 3): -1.0})

        with subtests.test("without groups"):
            reconstructed = pickle.loads(pickle.dumps(op))
            assert op.equiv(reconstructed)
            assert reconstructed.groups is None

        with subtests.test("with groups"):
            op.groups = [0, 1]
            reconstructed = pickle.loads(pickle.dumps(op))
            assert op.equiv(reconstructed)
            assert op.groups == reconstructed.groups

    def test_ichop(self):
        cls = self.get_class()
        op = cls.from_dict({(): 1e-4, (0,): 1e-6, (1,): 1e-10})
        op.ichop()
        assert op.equiv(cls.from_dict({(): 1e-4, (0,): 1e-6}))
        op.ichop(1e-5)
        assert op.equiv(cls.from_dict({(): 1e-4}))

    def test_ichop_preserves_complex_coeffs(self):
        cls = self.get_class()
        op = cls.from_dict({(): 1 + 2j, (0,): -3j, (1,): 1e-10})
        op.ichop()
        assert op.equiv(cls.from_dict({(): 1 + 2j, (0,): -3j}))

    def test_simplify(self):
        cls = self.get_class()
        coeffs = [1e-10, 2, 3, 4, -4]
        modes = [0, 0, 1, 1]
        boundaries = [0, 0, 1, 2, 3, 4]
        op = cls(coeffs, modes, boundaries)
        canon = op.simplify()
        assert canon.equiv(cls.from_dict({(0,): 5}), 1e-12)

    def test_simplify_vs_ichop(self):
        cls = self.get_class()
        coeffs = [1e-5] * int(1e5)
        modes = []
        boundaries = [0] + [0] * int(1e5)
        op = cls(coeffs, modes, boundaries)
        canon = op.simplify(1e-4)
        assert canon.equiv(op.one(), 1e-6)
        op.ichop(1e-4)
        assert op.equiv(op.zero(), 1e-6)

    def test_and(self):
        cls = self.get_class()
        op1 = cls.from_dict({(): 2, (gamma(0, False), gamma(0, True)): 3})
        op2 = cls.from_dict({(): 1.5, (gamma(0, True), gamma(0, False)): 4})
        op = op1 & op2
        assert op.equiv(
            cls.from_dict(
                {
                    (): 3,
                    (gamma(0, True), gamma(0, False)): 8,
                    (gamma(0, False), gamma(0, True)): 4.5,
                    (
                        gamma(0, True),
                        gamma(0, False),
                        gamma(0, False),
                        gamma(0, True),
                    ): 12,
                }
            )
        )

    def test_iand(self):
        cls = self.get_class()
        op1 = cls.from_dict({(): 2, (gamma(0, False), gamma(0, True)): 3})
        op2 = cls.from_dict({(): 1.5, (gamma(0, True), gamma(0, False)): 4})
        op1 &= op2
        assert op1.equiv(
            cls.from_dict(
                {
                    (): 3,
                    (gamma(0, True), gamma(0, False)): 8,
                    (gamma(0, False), gamma(0, True)): 4.5,
                    (
                        gamma(0, True),
                        gamma(0, False),
                        gamma(0, False),
                        gamma(0, True),
                    ): 12,
                }
            )
        )

    def test_pow(self, subtests):
        cls = self.get_class()
        op = cls.from_dict({(gamma(0, False),): 2})

        with subtests.test("pow==0"):
            assert (op**0).equiv(cls.one())

        with subtests.test("pow==1"):
            assert (op**1).equiv(op)

        with subtests.test("pow==2"):
            assert (op**2).equiv(cls.from_dict({(gamma(0, False), gamma(0, False)): 4}))

    def test_adjoint(self):
        cls = self.get_class()
        op = cls.from_dict({(): 2j, (gamma(0, False), gamma(0, True)): 3})
        assert op.adjoint().equiv(cls.from_dict({(): -2j, (gamma(0, True), gamma(0, False)): 3}))

    def test_normal_ordered(self, subtests):
        cls = self.get_class()

        with subtests.test("descending order"):
            op = cls.from_dict({(gamma(0, True), gamma(0, False)): 1})
            assert op.normal_ordered(ascending=False).equiv(op)
            expected = cls.from_dict({(gamma(0, False), gamma(0, True)): -1})
            assert op.normal_ordered(ascending=True).equiv(expected)

        with subtests.test("ascending order"):
            op = cls.from_dict({(gamma(0, False), gamma(0, True)): 1})
            assert op.normal_ordered(ascending=True).equiv(op)
            expected = cls.from_dict({(gamma(0, True), gamma(0, False)): -1})
            assert op.normal_ordered(ascending=False).equiv(expected)

        with subtests.test("reorder with reduction"):
            op = cls.from_dict({(gamma(0, True), gamma(0, False), gamma(0, True)): 1})
            expected = cls.from_dict({(gamma(0, False),): -1})
            assert op.normal_ordered().equiv(expected)

        with subtests.test("reorder without reduction"):
            op = cls.from_dict({(gamma(0, True), gamma(0, False), gamma(0, True)): 1})
            expected = cls.from_dict({(gamma(0, True), gamma(0, True), gamma(0, False)): -1})
            assert op.normal_ordered(reduce=False).equiv(expected)

    def test_majorana_matrix_oracle_matches_vertex_matrix(self):
        """Pins the index convention that :func:`majorana_operator_matrix` relies on.

        ``MajoranaOperator`` numbers generators from zero, while the oracle's :func:`majorana`
        follows the 1-based convention of the defining papers. An off-by-one here would make the
        oracle agree with a shifted operator and quietly validate a wrong implementation, so the
        offset is checked against :func:`vertex_matrix` (independently exercised by the edge- and
        transfer-vertex suites) via :math:`V_j = -i \\gamma_{2j-1} \\gamma_{2j}`.
        """
        num_modes = 2
        for mode in range(num_modes):
            lo, hi = sorted((gamma(mode, False), gamma(mode, True)))
            from_majoranas = -1j * majorana(lo + 1, num_modes) @ majorana(hi + 1, num_modes)
            assert np.allclose(from_majoranas, vertex_matrix(mode, num_modes))

    @pytest.mark.parametrize("length", [2, 3])
    @pytest.mark.parametrize("reduce", [True, False])
    @pytest.mark.parametrize("ascending", [True, False])
    def test_normal_ordered_preserves_matrix(self, length, reduce, ascending):
        """Asserts ``normal_ordered`` never changes the operator it represents.

        Reordering and contracting generators is only sound if every swap and every contraction
        carries the right sign, and a sign error is invisible to a test that compares against a
        hand-written expectation derived from the same (possibly wrong) rule. So this compares
        against dense matrices built straight from the Majorana definitions instead, exhaustively
        over every term of the given length. The edge- and transfer-vertex suites have carried this
        check for a while; ``MajoranaOperator`` was the one type whose own oracle it never used.
        """
        cls = self.get_class()
        num_modes = 2
        generators = list(range(2 * num_modes))

        for indices in itertools.product(generators, repeat=length):
            op = cls.from_dict({tuple(indices): 1 - 0.5j})
            reordered = op.normal_ordered(ascending=ascending, reduce=reduce)
            expected = majorana_operator_matrix(op, num_modes)
            actual = majorana_operator_matrix(reordered, num_modes)
            assert np.allclose(actual, expected), f"normal_ordered changed the operator {indices}"

    @pytest.mark.parametrize("ascending", [True, False])
    @pytest.mark.parametrize("length", [2, 3])
    def test_normal_ordered_is_fully_reduced(self, length, ascending):
        """Asserts no reducible pair of adjacent generators survives ``normal_ordered``.

        Two adjacent Majoranas on the same index square to the identity, so a fully reduced term
        never repeats an index in adjacent positions, and its indices are sorted in the requested
        direction.
        """
        cls = self.get_class()
        num_modes = 2
        generators = list(range(2 * num_modes))

        for indices in itertools.product(generators, repeat=length):
            reduced = cls.from_dict({tuple(indices): 1}).normal_ordered(ascending=ascending)
            for remaining, _ in reduced.iter_terms():
                pairs = list(itertools.pairwise(remaining))
                assert all(a != b for a, b in pairs), f"{indices} left {remaining} unreduced"
                if ascending:
                    assert all(a < b for a, b in pairs), f"{indices} left {remaining} unsorted"
                else:
                    assert all(a > b for a, b in pairs), f"{indices} left {remaining} unsorted"

    def test_is_hermitian(self, subtests):
        cls = self.get_class()

        with subtests.test("atol boundary"):
            op = cls.from_dict(
                {
                    (
                        gamma(0, False),
                        gamma(0, True),
                        gamma(1, False),
                        gamma(1, True),
                    ): 1.00001j,
                    (gamma(1, True), gamma(1, False), gamma(0, True), gamma(0, False)): -1j,
                }
            )

            assert not op.is_hermitian()
            assert op.is_hermitian(1e-4)

        with subtests.test("symmetrized operator is Hermitian"):
            op = cls.from_dict({(gamma(0, True), gamma(1, False)): 1.0 + 1.0j})
            assert not op.is_hermitian()
            assert (op + op.adjoint()).is_hermitian()

        with subtests.test("imaginary identity is not Hermitian"):
            assert not (cls.one() * 1j).is_hermitian()

    def test_is_even(self, subtests):
        cls = self.get_class()

        with subtests.test("True"):
            op = cls.from_dict({(gamma(0, True), gamma(0, False)): 1})
            assert op.is_even()

        with subtests.test("False"):
            op = cls.from_dict({(gamma(0, True),): 1})
            assert not op.is_even()

    def test_commutator(self):
        cls = self.get_class()

        op1 = cls.from_dict({(gamma(0, True),): 1})
        op2 = cls.from_dict({(gamma(0, True),): 1})
        comm = commutator(op1, op2)
        comm.ichop()
        assert comm.equiv(cls.zero())

    def test_anti_commutator(self):
        cls = self.get_class()

        op1 = cls.from_dict({(gamma(0, True),): 1})
        op2 = cls.from_dict({(gamma(0, True),): 1})
        comm = anti_commutator(op1, op2)
        assert comm.equiv(cls.from_dict({(gamma(0, True), gamma(0, True)): 2}))

    def test_relabel_modes(self, subtests):
        cls = self.get_class()

        op = cls.from_dict({(0, 1): 1, (0, 0, 2, 3): 1})

        with subtests.test("valid"):
            permutation = [4, 2, 5, 3]
            relabeled = op.relabel_modes(permutation)
            expected = cls.from_dict({(4, 2): 1, (4, 4, 5, 3): 1})
            assert relabeled.equiv(expected)

        with (
            subtests.test("duplicate indices"),
            pytest.raises(ValueError, match="duplicate indices"),
        ):
            permutation = [4, 4, 5, 3]
            op.relabel_modes(permutation)

        with (
            subtests.test("index map too small"),
            pytest.raises(ValueError, match="does not account for the entire length"),
        ):
            permutation = [4, 2, 5]
            op.relabel_modes(permutation)

    def test_split_out_groups(self, subtests):
        cls = self.get_class()

        # NOTE: we rely on Python dict's insertion order to guarantee the correct order of terms in
        # the expected outcome groups
        group0 = {}
        group0[(0, 1)] = 1
        group0[(1, 0)] = 1
        op = cls.from_dict(group0)
        group1 = {(0, 0, 1, 1): 2}
        op += cls.from_dict(group1)

        with subtests.test("num_groups none"):
            assert not op.has_groups()
            assert op.num_groups() is None

        op.groups = [0, 0, 1]

        with subtests.test("num_groups some"):
            assert op.has_groups()
            assert op.num_groups() == 2

        groups = op.split_out_groups()
        expected = [cls.from_dict(group0), cls.from_dict(group1)]

        with subtests.test("split groups"):
            assert all([a.equiv(b) for a, b in zip(groups, expected, strict=True)])

        with subtests.test("split groups reversed"):
            reversed_groups = op.split_out_groups(group_indices=[1, 0])
            assert all(
                a.equiv(b) for a, b in zip(reversed_groups, list(reversed(expected)), strict=True)
            )

        with subtests.test("split groups duplicate"):
            duplicate_groups = op.split_out_groups(group_indices=[0, 0])
            assert all(
                a.equiv(b)
                for a, b in zip(duplicate_groups, [expected[0], expected[0]], strict=True)
            )

        with subtests.test("split groups empty"):
            assert op.split_out_groups(group_indices=[]) == []

    def test_has_groups(self, subtests):
        cls = self.get_class()

        op = cls.from_dict({(0, 1): 1.0})

        with subtests.test("unset"):
            assert not op.has_groups()

        with subtests.test("assigned"):
            op.groups = [0]
            assert op.has_groups()

        with subtests.test("empty list"):
            # an operator may track groups while carrying no group indices at all, which
            # `has_groups` reports as `True` even though `num_groups` is 0
            empty = cls.zero()
            empty.groups = []
            assert empty.has_groups()
            assert empty.num_groups() == 0

        with subtests.test("reset to None"):
            op.groups = None
            assert not op.has_groups()

    def test_set_groups_err(self, subtests):
        cls = self.get_class()

        op = cls.from_dict({(0, 1): 1.0, (2, 3): 1.0})
        assert len(op.get_coeffs()) == 2

        # a short array would silently drop the trailing term wherever terms are iterated together
        # with their groups
        with (
            subtests.test("too few indices"),
            pytest.raises(ValueError, match="expected one group index per term"),
        ):
            op.groups = [0]

        # a long array would make `num_groups` report groups that no term carries
        with (
            subtests.test("too many indices"),
            pytest.raises(ValueError, match="expected one group index per term"),
        ):
            op.groups = [0, 0, 1]

        with subtests.test("a rejected assignment changes nothing"):
            assert not op.has_groups()

        with subtests.test("clearing is always allowed"):
            op.groups = [0, 1]
            op.groups = None
            assert not op.has_groups()

    def test_split_out_groups_err(self):
        cls = self.get_class()

        op = cls.from_dict({(0, 1): 1, (1, 0): 1, (0, 0, 1, 1): 2})
        assert op.split_out_groups() is None
        assert op.split_out_groups(group_indices=[0]) is None
