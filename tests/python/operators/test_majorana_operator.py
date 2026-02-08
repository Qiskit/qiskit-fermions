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

from abc import ABC, abstractmethod

from qiskit_fermions.operators import MajoranaOperator, gamma
from qiskit_fermions.operators.library import anti_commutator, commutator


class MajoranaOperatorTests(ABC):
    @staticmethod
    @abstractmethod
    def get_class() -> type[MajoranaOperator]: ...

    def test_zero(self):
        cls = self.get_class()
        op = cls.zero()
        assert op.equiv(cls.from_dict({}))

    def test_one(self):
        cls = self.get_class()
        op = cls.one()
        assert op.equiv(cls.from_dict({(): 1}))

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

    def test_iter(self):
        cls = self.get_class()
        op = cls.one()
        assert list(op.iter_terms()) == [([], 1)]

    def test_ichop(self):
        cls = self.get_class()
        op = cls.from_dict({(): 1e-4, (0,): 1e-6, (1,): 1e-10})
        op.ichop()
        assert op.equiv(cls.from_dict({(): 1e-4, (0,): 1e-6}))
        op.ichop(1e-5)
        assert op.equiv(cls.from_dict({(): 1e-4}))

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
            op = cls.from_dict({(gamma(0, True), gamma(0, False)): 1})
            assert op.normal_ordered().equiv(op)

        with subtests.test("simple reorder"):
            op = cls.from_dict({(gamma(0, False), gamma(0, True)): 1})
            expected = cls.from_dict({(gamma(0, True), gamma(0, False)): -1})
            assert op.normal_ordered().equiv(expected)

        with subtests.test("reorder with reduction"):
            op = cls.from_dict({(gamma(0, True), gamma(0, False), gamma(0, True)): 1})
            expected = cls.from_dict({(gamma(0, False),): -1})
            assert op.normal_ordered().equiv(expected)

        with subtests.test("reorder without reduction"):
            op = cls.from_dict({(gamma(0, True), gamma(0, False), gamma(0, True)): 1})
            expected = cls.from_dict({(gamma(0, True), gamma(0, True), gamma(0, False)): -1})
            assert op.normal_ordered(reduce=False).equiv(expected)

    def test_is_hermitian(self):
        cls = self.get_class()

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

    def test_many_body_order(self, subtests):
        cls = self.get_class()

        op = cls.one()

        with subtests.test("0"):
            assert op.many_body_order() == 0

        op += cls.from_dict({(0, 1): 1})

        with subtests.test("2"):
            assert op.many_body_order() == 2

        op += cls.from_dict({(0, 1, 2, 3): 1})

        with subtests.test("4"):
            assert op.many_body_order() == 4

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


class TestMajoranaOperator(MajoranaOperatorTests):
    @staticmethod
    def get_class() -> type[MajoranaOperator]:
        return MajoranaOperator
