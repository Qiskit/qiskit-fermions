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

import pickle

import numpy as np
import pytest
from qiskit_fermions.operators import FermionOperator, ann, cre
from qiskit_fermions.operators.library import anti_commutator, commutator


class TestFermionOperator:
    @staticmethod
    def get_class() -> type[FermionOperator]:
        return FermionOperator

    def test_getters(self, subtests):
        cls = self.get_class()

        coeffs = [1e-10, 2, 3, 4, -4]
        actions = [True, True, False, False]
        modes = [0, 0, 1, 1]
        boundaries = [0, 0, 1, 2, 3, 4]

        op = cls(coeffs, actions, modes, boundaries)

        with subtests.test("coeffs"):
            assert np.allclose(op.get_coeffs(), coeffs)
        with subtests.test("actions"):
            assert np.all(op.get_actions() == actions)
        with subtests.test("modes"):
            assert np.all(op.get_modes() == modes)
        with subtests.test("boundaries"):
            assert np.all(op.get_boundaries() == boundaries)

    def test_get_support(self):
        cls = self.get_class()
        op = cls.from_dict(
            {((True, 0), (False, 4)): 1, ((True, 1), (True, 3), (False, 4), (False, 7)): 1}
        )
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
        actions = [True, False, True, False]
        modes = [0, 1, 0, 1]
        boundaries = [0, 2, 4]

        op = cls(coeffs, actions, modes, boundaries)

        with subtests.test("equal"):
            other = cls(coeffs, actions, modes, boundaries)
            # `!=` must be the exact negation of `==`
            assert (op == other) is True
            assert (op != other) is False

        # each field differs individually: `!=` must be the negation of `==`.
        for name, other in [
            ("coeffs", cls([1, 3], actions, modes, boundaries)),
            ("actions", cls(coeffs, [False, True, True, False], modes, boundaries)),
            ("modes", cls(coeffs, actions, [1, 0, 0, 1], boundaries)),
            ("boundaries", cls(coeffs, actions, modes, [0, 1, 4])),
        ]:
            with subtests.test(name):
                assert (op == other) is False
                assert (op != other) is True

    def test_repr(self):
        cls = self.get_class()
        op = cls.from_dict(
            {
                (): 2,
                (cre(1), ann(2)): 1,
                (cre(2), ann(1)): 0.5,
                (cre(3), ann(4)): -0.5j,
                (cre(4), ann(3)): 1 - 0.5j,
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
            op = cls.from_dict({(): 1, (cre(0), ann(1)): 1})
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
                (cre(1), ann(2)): 1,
                (cre(2), ann(1)): 0.5,
                (cre(3), ann(4)): -0.5j,
                (cre(4), ann(3)): 1 - 0.5j,
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
                (cre(1), ann(2)): 1,
                (cre(2), ann(1)): 0.5,
                (cre(3), ann(4)): -0.5j,
                (cre(4), ann(3)): 1 - 0.5j,
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
        op = cls.from_dict({(cre(0), ann(2)): 2.0, (cre(2), ann(0)): 2.0})

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
        op = cls.from_dict({(): 1e-4, ((True, 0),): 1e-6, ((False, 0),): 1e-10})
        op.ichop()
        assert op.equiv(cls.from_dict({(): 1e-4, ((True, 0),): 1e-6}))
        op.ichop(1e-5)
        assert op.equiv(cls.from_dict({(): 1e-4}))

    def test_ichop_preserves_complex_coeffs(self):
        cls = self.get_class()
        op = cls.from_dict({(): 1 + 2j, ((True, 0),): -3j, ((False, 0),): 1e-10})
        op.ichop()
        assert op.equiv(cls.from_dict({(): 1 + 2j, ((True, 0),): -3j}))

    def test_simplify(self):
        cls = self.get_class()
        coeffs = [1e-10, 2, 3, 4, -4]
        actions = [True, True, False, False]
        modes = [0, 0, 1, 1]
        boundaries = [0, 0, 1, 2, 3, 4]
        op = cls(coeffs, actions, modes, boundaries)
        canon = op.simplify()
        assert canon.equiv(cls.from_dict({((True, 0),): 5}), 1e-12)

    def test_simplify_vs_ichop(self):
        cls = self.get_class()
        coeffs = [1e-5] * int(1e5)
        actions = []
        modes = []
        boundaries = [0] + [0] * int(1e5)
        op = cls(coeffs, actions, modes, boundaries)
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
        op1 = cls.from_dict({(): 2, (cre(0), ann(1)): 3})
        op2 = cls.from_dict({(): 1.5, (cre(1), ann(0)): 4})
        op = op1 & op2
        assert op.equiv(
            cls.from_dict(
                {
                    (): 3,
                    (cre(1), ann(0)): 8,
                    (cre(0), ann(1)): 4.5,
                    (cre(1), ann(0), cre(0), ann(1)): 12,
                }
            )
        )

    def test_iand(self):
        cls = self.get_class()
        op1 = cls.from_dict({(): 2, (cre(0), ann(1)): 3})
        op2 = cls.from_dict({(): 1.5, (cre(1), ann(0)): 4})
        op1 &= op2
        assert op1.equiv(
            cls.from_dict(
                {
                    (): 3,
                    (cre(1), ann(0)): 8,
                    (cre(0), ann(1)): 4.5,
                    (cre(1), ann(0), cre(0), ann(1)): 12,
                }
            )
        )

    def test_pow(self, subtests):
        cls = self.get_class()
        op = cls.from_dict({(cre(0),): 2})

        with subtests.test("pow==0"):
            assert (op**0).equiv(cls.one())

        with subtests.test("pow==1"):
            assert (op**1).equiv(op)

        with subtests.test("pow==2"):
            assert (op**2).equiv(cls.from_dict({(cre(0), cre(0)): 4}))

    def test_adjoint(self):
        cls = self.get_class()
        op = cls.from_dict({(): 2j, (cre(0), ann(1)): 3})
        assert op.adjoint().equiv(cls.from_dict({(): -2j, (cre(1), ann(0)): 3}))

    def test_equiv(self):
        cls = self.get_class()
        op = cls.from_dict({(): 1e-7})
        zero = cls.zero()
        assert not op.equiv(zero)
        assert op.equiv(zero, 1e-6)
        assert not op.equiv(zero, 1e-8)

    def test_normal_ordered(self, subtests):
        cls = self.get_class()

        with subtests.test("no change"):
            op = cls.from_dict({((True, 0), (False, 1)): 1})
            assert op.normal_ordered().equiv(op)

        with subtests.test("simple reorder"):
            op = cls.from_dict({((True, 0), (True, 1)): 1})
            expected = cls.from_dict({((True, 1), (True, 0)): -1})
            assert op.normal_ordered().equiv(expected)

        with subtests.test("reorder with new term"):
            op = cls.from_dict({((False, 0), (True, 0)): 1})
            expected = cls.from_dict({(): 1, ((True, 0), (False, 0)): -1})
            assert op.normal_ordered().equiv(expected)

    def test_normal_ordered_sandwich(self, subtests):
        cls = self.get_class()

        with subtests.test("sandwich=none"):
            op = cls.from_dict({((False, 0), (False, 1), (True, 0), (True, 1)): 1})
            expected = cls.from_dict(
                {
                    ((True, 1), (True, 0), (False, 1), (False, 0)): 1,
                    ((True, 0), (False, 0)): 1,
                    ((True, 1), (False, 1)): 1,
                    (): -1,
                }
            )
            assert op.normal_ordered(sandwich=None).equiv(expected)

        with subtests.test("sandwich=True"):
            op = cls.from_dict({((False, 1), (False, 0), (True, 0), (True, 1)): 1})
            expected = cls.from_dict(
                {
                    ((True, 0), (True, 1), (False, 1), (False, 0)): 1,
                    ((True, 0), (False, 0)): -1,
                    ((True, 1), (False, 1)): -1,
                    (): 1,
                }
            )
            assert op.normal_ordered(sandwich=True).equiv(expected)

        with subtests.test("sandwich=False"):
            op = cls.from_dict({((False, 0), (False, 1), (True, 1), (True, 0)): 1})
            expected = cls.from_dict(
                {
                    ((True, 1), (True, 0), (False, 0), (False, 1)): 1,
                    ((True, 0), (False, 0)): -1,
                    ((True, 1), (False, 1)): -1,
                    (): 1,
                }
            )
            assert op.normal_ordered(sandwich=False).equiv(expected)

    def test_is_hermitian(self):
        cls = self.get_class()

        op = cls.from_dict(
            {
                ((True, 0), (False, 1)): 1.00001j,
                ((True, 1), (False, 0)): -1j,
            }
        )

        assert not op.is_hermitian()
        assert op.is_hermitian(1e-4)

    def test_max_rank(self, subtests):
        cls = self.get_class()

        op = cls.zero()

        with subtests.test("0 for additive identity"):
            assert op.max_rank() == 0

        op += cls.one()

        with subtests.test("0 for multiplicative identity"):
            assert op.max_rank() == 0

        op += cls.from_dict({((True, 0), (False, 1)): 1})

        with subtests.test("2"):
            assert op.max_rank() == 2

        op += cls.from_dict({((True, 0), (False, 1), (True, 2), (False, 3)): 1})

        with subtests.test("4"):
            assert op.max_rank() == 4

    def test_conserves_particle_number(self, subtests):
        cls = self.get_class()

        with subtests.test("True"):
            op = cls.from_dict({((True, 0), (False, 1)): 1})
            assert op.conserves_particle_number()

        with subtests.test("False"):
            op = cls.from_dict({((True, 0),): 1})
            assert not op.conserves_particle_number()

    def test_conserves_sector(self, subtests):
        cls = self.get_class()

        with subtests.test("empty block_sizes matches conserves_particle_number"):
            op = cls.from_dict({((True, 0), (False, 1)): 1})
            assert op.conserves_sector([]) == op.conserves_particle_number()
            op = cls.from_dict({((True, 0),): 1})
            assert op.conserves_sector([]) == op.conserves_particle_number()

        with subtests.test("spinless hop conserves single block"):
            op = cls.from_dict({((True, 0), (False, 1)): 1})
            assert op.conserves_sector([2])

        with subtests.test("spin flip conserves number but not Sz split"):
            # a†_0 a_2 moves a particle from the alpha block [0, 2) to the beta block [2, 4).
            op = cls.from_dict({((True, 0), (False, 2)): 1})
            assert op.conserves_particle_number()
            assert op.conserves_sector([4])
            assert not op.conserves_sector([2, 2])

        with subtests.test("bare creation conserves nothing"):
            op = cls.from_dict({((True, 2),): 1})
            assert not op.conserves_sector([4])

        with subtests.test("mode beyond the last block does not conserve"):
            op = cls.from_dict({((True, 0), (False, 4)): 1})
            assert not op.conserves_sector([2, 2])

    def test_commutator(self):
        cls = self.get_class()

        op1 = cls.from_dict({(cre(0),): 1})
        op2 = cls.from_dict({(ann(0),): 1})
        comm = commutator(op1, op2)
        comm = comm.normal_ordered()
        comm.ichop()
        assert comm.equiv(cls.from_dict({(): -1, (cre(0), ann(0)): 2}))

    def test_anti_commutator(self):
        cls = self.get_class()

        op1 = cls.from_dict({(cre(0),): 1})
        op2 = cls.from_dict({(ann(0),): 1})
        comm = anti_commutator(op1, op2)
        comm = comm.normal_ordered()
        comm.ichop()
        assert comm.equiv(cls.one())

    def test_relabel_modes(self, subtests):
        cls = self.get_class()

        op = cls.from_dict({(cre(0), ann(1)): 1, (cre(0), ann(0), cre(2), ann(3)): 1})

        with subtests.test("valid"):
            permutation = [4, 2, 5, 3]
            relabeled = op.relabel_modes(permutation)
            expected = cls.from_dict({(cre(4), ann(2)): 1, (cre(4), ann(4), cre(5), ann(3)): 1})
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
        group0[(cre(0), ann(1))] = 1
        group0[(cre(1), ann(0))] = 1
        op = cls.from_dict(group0)
        group1 = {(cre(0), cre(0), ann(1), ann(1)): 2}
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

        op = cls.from_dict({(cre(0), ann(1)): 1.0})

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

        op = cls.from_dict(
            {
                (cre(0), ann(1)): 1.0,
                (cre(1), ann(0)): -3.0 + 4.0j,
                (cre(0), cre(0), ann(1), ann(1)): 2.0,
            }
        )

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

        with subtests.test("exact averages"):
            # built positionally (rather than via `from_dict`) so that the term order -- and hence
            # the pairing with the group indices below -- is fixed, pinning the exact expected
            # averages: group 0 is (|1.0| + |-3.0 + 4.0j|) / 2 == (1.0 + 5.0) / 2, and group 1 is
            # the lone |2.0|.
            exact = cls(
                [1.0, -3.0 + 4.0j, 2.0],
                [True, False, True, False, True, False],
                [0, 1, 1, 0, 2, 3],
                [0, 2, 4, 6],
            )
            exact.groups = [0, 0, 1]
            assert exact.group_weights() == [3.0, 2.0]

    def test_split_out_groups_err(self):
        cls = self.get_class()

        op = cls.from_dict(
            {(cre(0), ann(1)): 1, (cre(1), ann(0)): 1, (cre(0), cre(0), ann(1), ann(1)): 2}
        )
        assert op.split_out_groups() is None
        assert op.split_out_groups(group_indices=[0]) is None
