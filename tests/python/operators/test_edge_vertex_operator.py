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
from qiskit_fermions.operators import EdgeVertexOperator
from qiskit_fermions.operators.library import anti_commutator, commutator

from .majorana_matrix_oracle import edge_matrix, operator_matrix


class TestEdgeVertexOperator:
    @staticmethod
    def get_class() -> type[EdgeVertexOperator]:
        return EdgeVertexOperator

    def test_getters(self, subtests):
        cls = self.get_class()

        coeffs = [1e-10, 2, 3, 4, -4]
        l_indices = [0, 1, 2, 3]
        r_indices = [1, 2, 3, 4]
        boundaries = [0, 0, 1, 2, 3, 4]

        op = cls(coeffs, l_indices, r_indices, boundaries)

        with subtests.test("coeffs"):
            assert np.allclose(op.get_coeffs(), coeffs)
        with subtests.test("left_indices"):
            assert np.all(op.get_left_indices() == l_indices)
        with subtests.test("right_indices"):
            assert np.all(op.get_right_indices() == r_indices)
        with subtests.test("boundaries"):
            assert np.all(op.get_boundaries() == boundaries)

    def test_get_support(self):
        cls = self.get_class()
        op = cls.from_dict({((0, 1), (3, 4)): 1, ((7, 7),): 1})
        assert op.get_support() == {0, 1, 3, 4, 7}

    def test_zero(self):
        cls = self.get_class()
        op = cls.zero()
        assert op == cls.from_dict({})

    def test_one(self):
        cls = self.get_class()
        op = cls.one()
        assert op == cls.from_dict({(): 1})

    def test_richcmp(self, subtests):
        cls = self.get_class()

        # two terms, each of length two (boundaries has len(coeffs) + 1 entries)
        coeffs = [1, 2]
        l_indices = [0, 1, 0, 1]
        r_indices = [1, 2, 1, 2]
        boundaries = [0, 2, 4]

        op = cls(coeffs, l_indices, r_indices, boundaries)

        with subtests.test("equal"):
            other = cls(coeffs, l_indices, r_indices, boundaries)
            # `!=` must be the exact negation of `==`
            assert (op == other) is True
            assert (op != other) is False

        # each field differs individually: `!=` must be the negation of `==`.
        for name, other in [
            ("coeffs", cls([1, 3], l_indices, r_indices, boundaries)),
            ("left_indices", cls(coeffs, [1, 0, 0, 1], r_indices, boundaries)),
            ("right_indices", cls(coeffs, l_indices, [2, 1, 1, 2], boundaries)),
            ("boundaries", cls(coeffs, l_indices, r_indices, [0, 1, 4])),
        ]:
            with subtests.test(name):
                assert (op == other) is False
                assert (op != other) is True

    def test_repr(self):
        cls = self.get_class()
        op = cls.from_dict(
            {
                (): 2,
                ((0, 1),): 1,
                ((3, 4),): 0.5,
                ((0, 1), (2, 3)): -0.5j,
                ((3, 4), (0, 1)): 1 - 0.5j,
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
            op = cls.from_dict({(): 1, ((0, 1),): 1})
            assert len(op) == 2

    def test_iter(self):
        cls = self.get_class()
        op = cls.one()
        assert list(op.iter_terms()) == [([], 1)]

    def test_from_terms(self, subtests):
        cls = self.get_class()
        op = cls.from_dict(
            {
                (): 2,
                ((0, 1),): 1,
                ((3, 4),): 0.5,
                ((0, 1), (2, 3)): -0.5j,
                ((3, 4), (0, 1)): 1 - 0.5j,
            }
        )
        with subtests.test("iterator"):
            assert op.equiv(cls.from_terms(op.iter_terms()))
        with subtests.test("list"):
            assert op.equiv(cls.from_terms(list(op.iter_terms())))

    def test_iter_with_groups(self):
        cls = self.get_class()
        op = cls.one()
        op.groups = [0]
        assert list(op.iter_terms_with_groups()) == [([], 1, 0)]

    def test_from_terms_with_groups(self, subtests):
        cls = self.get_class()
        op = cls.from_dict(
            {
                (): 2,
                ((0, 1),): 1,
                ((3, 4),): 0.5,
                ((0, 1), (2, 3)): -0.5j,
                ((3, 4), (0, 1)): 1 - 0.5j,
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
        op = cls.from_dict({((0, 1),): 1.0, ((3, 4),): -1.0})

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
        op = cls.from_dict({(): 1e-4, ((0, 1),): 1e-6, ((1, 2),): 1e-10})
        op.ichop()
        assert op.equiv(cls.from_dict({(): 1e-4, ((0, 1),): 1e-6}))
        op.ichop(1e-5)
        assert op.equiv(cls.from_dict({(): 1e-4}))

    def test_ichop_preserves_complex_coeffs(self):
        cls = self.get_class()
        op = cls.from_dict({(): 1 + 2j, ((0, 1),): -3j, ((1, 2),): 1e-10})
        op.ichop()
        assert op.equiv(cls.from_dict({(): 1 + 2j, ((0, 1),): -3j}))

    def test_simplify(self):
        cls = self.get_class()
        coeffs = [1e-10, 2, 3, 4, -4]
        l_indices = [0, 0, 2, 2]
        r_indices = [1, 1, 3, 3]
        boundaries = [0, 0, 1, 2, 3, 4]
        op = cls(coeffs, l_indices, r_indices, boundaries)
        canon = op.simplify()
        assert canon.equiv(cls.from_dict({((0, 1),): 5}), 1e-12)

    def test_simplify_vs_ichop(self):
        cls = self.get_class()
        coeffs = [1e-5] * int(1e5)
        l_indices = []
        r_indices = []
        boundaries = [0] + [0] * int(1e5)
        op = cls(coeffs, l_indices, r_indices, boundaries)
        canon = op.simplify(1e-4)
        assert canon.equiv(op.one(), 1e-6)
        op.ichop(1e-4)
        assert op.equiv(op.zero(), 1e-6)

    def test_add(self):
        cls = self.get_class()
        one = cls.one()
        two = cls.from_dict({(): 2})
        three = one + two
        assert three.equiv(cls.from_dict({(): 3}))

    def test_iadd(self):
        cls = self.get_class()
        op = cls.one()
        two = cls.from_dict({(): 2})
        op += two
        assert op.equiv(cls.from_dict({(): 3}))

    def test_sub(self):
        cls = self.get_class()
        one = cls.one()
        two = cls.from_dict({(): 2})
        new_one = two - one
        assert new_one.equiv(one)

    def test_isub(self):
        cls = self.get_class()
        op = cls.from_dict({(): 2})
        one = cls.one()
        op -= one
        assert op.equiv(one)

    def test_mul(self):
        cls = self.get_class()
        one = cls.one()
        three = one * 3
        assert three.equiv(cls.from_dict({(): 3}))

    def test_rmul(self):
        cls = self.get_class()
        one = cls.one()
        three = 3 * one
        assert three.equiv(cls.from_dict({(): 3}))

    def test_imul(self):
        cls = self.get_class()
        op = cls.one()
        op *= 3
        assert op.equiv(cls.from_dict({(): 3}))

    def test_div(self):
        cls = self.get_class()
        three = cls.from_dict({(): 3})
        one_half = three / 2.0
        assert one_half.equiv(cls.from_dict({(): 1.5}))

    def test_idiv(self):
        cls = self.get_class()
        op = cls.from_dict({(): 3})
        op /= 2.0
        assert op.equiv(cls.from_dict({(): 1.5}))

    def test_neg(self):
        cls = self.get_class()
        one = cls.one()
        assert (-one).equiv(cls.from_dict({(): -1}))

    def test_and(self):
        cls = self.get_class()
        op1 = cls.from_dict({(): 2, ((0, 1),): 3})
        op2 = cls.from_dict({(): 1.5, ((1, 2),): 4})
        op = op1 & op2
        assert op.equiv(
            cls.from_dict(
                {
                    (): 3,
                    ((0, 1),): 4.5,
                    ((1, 2),): 8,
                    ((1, 2), (0, 1)): 12,
                }
            )
        )

    def test_iand(self):
        cls = self.get_class()
        op1 = cls.from_dict({(): 2, ((0, 1),): 3})
        op2 = cls.from_dict({(): 1.5, ((1, 2),): 4})
        op1 &= op2
        assert op1.equiv(
            cls.from_dict(
                {
                    (): 3,
                    ((0, 1),): 4.5,
                    ((1, 2),): 8,
                    ((1, 2), (0, 1)): 12,
                }
            )
        )

    def test_pow(self, subtests):
        cls = self.get_class()
        op = cls.from_dict({((0, 1),): 2})

        with subtests.test("pow==0"):
            assert (op**0).equiv(cls.one())

        with subtests.test("pow==1"):
            assert (op**1).equiv(op)

        with subtests.test("pow==2"):
            assert (op**2).equiv(cls.from_dict({((0, 1), (0, 1)): 4}))

    def test_adjoint(self, subtests):
        cls = self.get_class()

        with subtests.test("single-factor terms"):
            op = cls.from_dict({(): 2j, ((0, 1),): 3})
            assert op.adjoint().equiv(cls.from_dict({(): -2j, ((0, 1),): 3}))

        with subtests.test("multi-factor term is reversed"):
            op = cls.from_dict({((0, 0), (0, 1), (1, 2)): 3 - 4j})
            assert op.adjoint().equiv(cls.from_dict({((1, 2), (0, 1), (0, 0)): 3 + 4j}))

        with subtests.test("(A @ B).adjoint() == B.adjoint() @ A.adjoint()"):
            op_a = cls.from_dict({((0, 0), (0, 1)): 2 + 1j})
            op_b = cls.from_dict({((1, 2), (2, 2)): -1 + 3j})
            assert (op_a @ op_b).adjoint().equiv(op_b.adjoint() @ op_a.adjoint())

    def test_is_hermitian(self, subtests):
        cls = self.get_class()

        # V(0) and E(0,1) share the index 0 and therefore anticommute, so
        # (V(0) E(0,1))† = E(0,1) V(0) = -V(0) E(0,1) and the operator is not Hermitian.
        op = cls.from_dict({((0, 0), (0, 1)): 1.0})

        with subtests.test("anticommuting product is not Hermitian"):
            assert not op.is_hermitian()

        with subtests.test("symmetrized product is Hermitian"):
            assert (op + op.adjoint()).is_hermitian()

        with subtests.test("Hermitian only after fusion"):
            # This operator is Hermitian, but recognizing that requires *fusing* `E(1,0) E(2,1)`
            # into a multiple of `E(2,0)`: the two terms of `op - op.adjoint()` cancel only once
            # that contraction is applied. Reordering alone leaves a non-zero remainder, which is
            # why this used to be reported as non-Hermitian.
            op = cls.from_dict(
                {
                    ((2, 0), (2, 2)): 0.1 - 0.4j,
                    ((2, 1), (2, 2), (1, 0)): 0.1 - 0.1j,
                }
            )
            matrix = operator_matrix(op, 3, edge_matrix)
            assert np.allclose(matrix, matrix.conj().T), "test premise: op must be Hermitian"
            assert op.is_hermitian()

    def test_equiv(self):
        cls = self.get_class()
        op = cls.from_dict({(): 1e-7})
        zero = cls.zero()
        assert not op.equiv(zero)
        assert op.equiv(zero, 1e-6)
        assert not op.equiv(zero, 1e-8)

    def test_normal_ordered(self, subtests):
        cls = self.get_class()

        # These cases pin the pure *reordering* behaviour, so they opt out of the contraction that
        # `reduce=True` (the default) would additionally apply. See `test_normal_ordered_reduce`.
        with subtests.test("no change"):
            op = cls.from_dict({((0, 0), (1, 1), (1, 1), (0, 1)): 1})
            assert op.normal_ordered(reduce=False).equiv(op)

        with subtests.test("ordering of vertex and edge operators"):
            op = cls.from_dict({((0, 1), (1, 0), (1, 2), (0, 0), (2, 2)): 1})
            expected = cls.from_dict({((0, 0), (2, 2), (0, 1), (1, 0), (1, 2)): -1})
            assert op.normal_ordered(reduce=False).equiv(expected)

        with subtests.test("edge operators on the same pair of modes commute"):
            # Eq. (5) of arXiv:2512.11418v1 only covers `j != k != l != m`, so it says nothing
            # about two edge operators spanning the *same* pair of modes. Because
            # `E_{kj} = -E_{jk}`, those are collinear and commute: reordering them must not
            # introduce a sign.
            op = cls.from_dict({((1, 0), (0, 1)): 1 + 2j})
            expected = cls.from_dict({((0, 1), (1, 0)): 1 + 2j})
            assert op.normal_ordered(reduce=False).equiv(expected)

            # Two identical edge operators: the sort is a no-op, but the parity rule still runs.
            op = cls.from_dict({((1, 0), (1, 0)): 1 + 2j})
            assert op.normal_ordered(reduce=False).equiv(op)

    def test_normal_ordered_reduce(self, subtests):
        cls = self.get_class()

        with subtests.test("V_j V_j = 1"):
            op = cls.from_dict({((0, 0), (0, 0)): 2 + 1j})
            assert op.normal_ordered().equiv(cls.from_dict({(): 2 + 1j}))

        with subtests.test("E_jk E_jk = 1"):
            op = cls.from_dict({((0, 1), (0, 1)): 2 + 1j})
            assert op.normal_ordered().equiv(cls.from_dict({(): 2 + 1j}))

        with subtests.test("E_jk E_kj = -1"):
            op = cls.from_dict({((0, 1), (1, 0)): 2 + 1j})
            assert op.normal_ordered().equiv(cls.from_dict({(): -2 - 1j}))

        with subtests.test("fusion: E_ab E_bc = -i E_ac"):
            op = cls.from_dict({((0, 1), (1, 2)): 1})
            assert op.normal_ordered().equiv(cls.from_dict({((0, 2),): -1j}))

        with subtests.test("fusion is found regardless of stored orientation"):
            # `E_{1,0} E_{2,1}` shares mode 1, but neither factor is oriented so that the shared
            # mode sits in the inner position. Reducing it requires applying `E_{kj} = -E_{jk}`
            # first, which is what the orientation canonicalization is for.
            op = cls.from_dict({((1, 0), (2, 1)): 1})
            assert op.normal_ordered().equiv(cls.from_dict({((0, 2),): -1j}))

        with subtests.test("nothing left to contract"):
            for actions in [((0, 0), (1, 1)), ((0, 0), (1, 2)), ((0, 1), (2, 3))]:
                op = cls.from_dict({actions: 1})
                reduced = op.normal_ordered()
                assert reduced.equiv(op), f"{actions} should not have been reduced"

    @pytest.mark.parametrize("ascending", [True, False])
    def test_normal_ordered_ascending(self, ascending, subtests):
        cls = self.get_class()

        with subtests.test("orientation is canonicalized"):
            # `E_{1,0} = -E_{0,1}`, so whichever orientation is *not* selected must be rewritten
            # into the one that is, with the sign absorbed into the coefficient.
            op = cls.from_dict({((1, 0),): 1})
            expected_action = (0, 1) if ascending else (1, 0)
            expected_coeff = -1 if ascending else 1
            assert op.normal_ordered(ascending=ascending).equiv(
                cls.from_dict({(expected_action,): expected_coeff})
            )

        with subtests.test("vertex operators are unaffected"):
            op = cls.from_dict({((1, 1),): 1})
            assert op.normal_ordered(ascending=ascending).equiv(op)

    def test_normal_ordered_is_canonical(self):
        """Asserts that two representations of one operator normal-order to the same terms.

        Because ``E_kj = -E_jk``, the same operator can be stored in many ways. Flipping every
        edge operator's orientation (and the coefficient's sign with it) is a no-op on the operator
        itself, so it must be a no-op on the normal-ordered result too.
        """
        cls = self.get_class()
        num_modes = 3
        generators = [(a, b) for a in range(num_modes) for b in range(num_modes)]

        for actions in itertools.product(generators, repeat=3):
            coeff = 1 - 0.5j
            flipped, flipped_coeff = [], coeff
            for left, right in actions:
                if left == right:
                    flipped.append((left, right))
                else:
                    flipped.append((right, left))
                    flipped_coeff = -flipped_coeff

            original = cls.from_dict({tuple(actions): coeff}).normal_ordered().simplify()
            equivalent = cls.from_dict({tuple(flipped): flipped_coeff}).normal_ordered().simplify()
            assert original.equiv(equivalent), f"{actions} is not canonical"

    @pytest.mark.parametrize("length", [2, 3])
    @pytest.mark.parametrize("reduce", [False, True])
    @pytest.mark.parametrize("ascending", [True, False])
    def test_normal_ordered_preserves_matrix(self, length, reduce, ascending):
        """Asserts ``normal_ordered`` never changes the operator it represents.

        Reordering and contracting generators is only sound if every swap and every contraction
        carries the right sign, and a sign error is invisible to a test that compares against a
        hand-written expectation derived from the same (possibly wrong) rule. So this compares
        against dense matrices built straight from the Majorana definitions instead, exhaustively
        over every term of the given length.
        """
        cls = self.get_class()
        num_modes = 3
        generators = [(a, b) for a in range(num_modes) for b in range(num_modes)]

        for actions in itertools.product(generators, repeat=length):
            op = cls.from_dict({tuple(actions): 1 - 0.5j})
            reordered = op.normal_ordered(ascending=ascending, reduce=reduce)
            expected = operator_matrix(op, num_modes, edge_matrix)
            actual = operator_matrix(reordered, num_modes, edge_matrix)
            assert np.allclose(actual, expected), f"normal_ordered changed the operator {actions}"

    @pytest.mark.parametrize("length", [2, 3])
    def test_normal_ordered_is_fully_reduced(self, length):
        """Asserts no reducible pair of adjacent generators survives ``normal_ordered``."""
        cls = self.get_class()
        num_modes = 3
        generators = [(a, b) for a in range(num_modes) for b in range(num_modes)]

        def reducible(actions) -> bool:
            for (a, b), (c, d) in itertools.pairwise(actions):
                if {a, b} == {c, d}:
                    return True
                # two edge operators sharing exactly one mode fuse into a single one
                if a != b and c != d and len({a, b} & {c, d}) == 1:
                    return True
            return False

        for actions in itertools.product(generators, repeat=length):
            reduced = cls.from_dict({tuple(actions): 1}).normal_ordered()
            for remaining, _ in reduced.iter_terms():
                assert not reducible(tuple(remaining)), f"{actions} left {remaining} unreduced"

    def test_commutator(self, subtests):
        cls = self.get_class()

        with subtests.test("3. relation of Eq. (5) from arXiv:2512.11418v1"):
            op1 = cls.from_dict({((0, 0),): 1})
            op2 = cls.from_dict({((1, 1),): 1})
            comm = commutator(op1, op2)
            comm = comm.normal_ordered()
            comm.ichop()
            assert comm.equiv(cls.zero())

        with subtests.test("4. relation of Eq. (5) from arXiv:2512.11418v1"):
            op1 = cls.from_dict({((0, 1),): 1})
            op2 = cls.from_dict({((2, 2),): 1})
            comm = commutator(op1, op2)
            comm = comm.normal_ordered()
            comm.ichop()
            assert comm.equiv(cls.zero())

        with subtests.test("5. relation of Eq. (5) from arXiv:2512.11418v1"):
            op1 = cls.from_dict({((0, 1),): 1})
            op2 = cls.from_dict({((2, 3),): 1})
            comm = commutator(op1, op2)
            comm = comm.normal_ordered()
            comm.ichop()
            assert comm.equiv(cls.zero())

        with subtests.test("edge operators on the same pair of modes"):
            # Not covered by Eq. (5), which requires `j != k != l != m`: since
            # `E_{1,0} = -E_{0,1}`, the two are collinear and therefore commute.
            op1 = cls.from_dict({((0, 1),): 1})
            op2 = cls.from_dict({((1, 0),): 1})
            comm = commutator(op1, op2)
            comm = comm.normal_ordered()
            comm.ichop()
            assert comm.equiv(cls.zero())

    def test_anti_commutator(self, subtests):
        cls = self.get_class()

        with subtests.test("1. relation of Eq. (5) from arXiv:2512.11418v1"):
            op1 = cls.from_dict({((0, 1),): 1})
            op2 = cls.from_dict({((1, 1),): 1})
            comm = anti_commutator(op1, op2)
            comm = comm.normal_ordered()
            comm.ichop()
            assert comm.equiv(cls.zero())

        with subtests.test("2. relation of Eq. (5) from arXiv:2512.11418v1"):
            # Eq. (5) requires `j != k != l`, so the two edge operators must share *exactly one*
            # mode. `E_{0,1}` and `E_{1,2}` share only mode 1. Two edge operators spanning the
            # *same* pair of modes instead commute -- see `test_normal_ordered`.
            op1 = cls.from_dict({((0, 1),): 1})
            op2 = cls.from_dict({((1, 2),): 1})
            comm = anti_commutator(op1, op2)
            comm = comm.normal_ordered()
            comm.ichop()
            assert comm.equiv(cls.zero())

    def test_relabel_modes(self, subtests):
        cls = self.get_class()

        op = cls.from_dict({((0, 1), (2, 3)): 1, ((1, 2), (3, 0)): 1})

        with subtests.test("valid"):
            permutation = [4, 2, 5, 3]
            relabeled = op.relabel_modes(permutation)
            expected = cls.from_dict({((4, 2), (5, 3)): 1, ((2, 5), (3, 4)): 1})
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
        group0[((0, 1),)] = 1
        group0[((2, 3),)] = 1
        op = cls.from_dict(group0)
        group1 = {((1, 0), (2, 3)): 2}
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

        op = cls.from_dict({((0, 1),): 1.0})

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

    def test_group_weights(self, subtests):
        cls = self.get_class()

        op = cls.from_dict({((0, 1),): 1.0, ((2, 3),): -3.0 + 4.0j, ((1, 0), (2, 3)): 2.0})

        with subtests.test("unset"):
            assert op.group_weights() is None

        # NOTE: `from_dict` does not preserve the insertion order of its keys, so the expected
        # weights are derived from the operator's actual coefficient order rather than hardcoded.
        # The magnitude, not the real part, is what gets averaged (hence `abs` below).
        coeffs = op.get_coeffs()
        first_two = (abs(coeffs[0]) + abs(coeffs[1])) / 2
        last = abs(coeffs[2])

        op.groups = [0, 0, 1]

        with subtests.test("assigned"):
            assert op.group_weights() == [first_two, last]

        with subtests.test("sparse group index"):
            # group 1 is carried by no term at all, so it weighs 0.0 rather than NaN
            op.groups = [0, 0, 2]
            assert op.group_weights() == [first_two, 0.0, last]

        with subtests.test("empty list"):
            empty = cls.zero()
            empty.groups = []
            assert empty.group_weights() == []

    def test_split_out_groups_err(self):
        cls = self.get_class()

        op = cls.from_dict({((0, 1),): 1, ((2, 3),): 1, ((1, 0), (2, 3)): 2})
        assert op.split_out_groups() is None
        assert op.split_out_groups(group_indices=[0]) is None
