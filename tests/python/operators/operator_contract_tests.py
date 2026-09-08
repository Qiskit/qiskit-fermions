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

"""The shared operator contract, exercised identically by all four operator types."""

from __future__ import annotations

import copy
from typing import Protocol


class _OperatorClassProvider(Protocol):
    """The single hook a test class provides: which operator type to exercise."""

    @staticmethod
    def get_class() -> type: ...


class OperatorContractTests:
    """Tests of the arithmetic and iteration contract shared by every operator type.

    Every test here is expressed purely in terms of :meth:`zero`, :meth:`one` and the *identity*
    term ``{(): coeff}``, so none of them needs to know how a particular operator family spells its
    generators. That is what makes it safe to share them: the four operator types take genuinely
    different action tuples (``cre``/``ann`` pairs, Majorana indices, edge/transfer index pairs), and
    a test that touched those could not be lifted here without inventing a translation layer.

    Subclasses supply :meth:`get_class` and inherit these; anything type-specific stays in the
    subclass. Before this existed the tests below were maintained as one byte-identical copy per
    operator type, which is how the per-type asymmetries elsewhere in these files accumulated
    unnoticed.

    Deliberately *not* lifted here, even though the four files hold near-identical copies:
    ``test_commutator``/``test_anti_commutator``. Their subtests pin the algebraic relations of a
    specific representation (Eq. (5) of arXiv:2512.11418v1 for edge-vertex, Eq. (7) for
    transfer-vertex), so folding them together would silently merge assertions about different
    algebras.
    """

    # The subclass provides this; declared for the type checker's benefit.
    get_class: staticmethod

    def test_zero(self: _OperatorClassProvider):
        cls = self.get_class()
        op = cls.zero()
        assert op == cls.from_dict({})

    def test_one(self: _OperatorClassProvider):
        cls = self.get_class()
        op = cls.one()
        assert op == cls.from_dict({(): 1})

    def test_add(self: _OperatorClassProvider):
        cls = self.get_class()
        one = cls.one()
        two = cls.from_dict({(): 2})
        three = one + two
        assert three.equiv(cls.from_dict({(): 3}))

    def test_iadd(self: _OperatorClassProvider):
        cls = self.get_class()
        op = cls.one()
        two = cls.from_dict({(): 2})
        op += two
        assert op.equiv(cls.from_dict({(): 3}))

    def test_sub(self: _OperatorClassProvider):
        cls = self.get_class()
        one = cls.one()
        two = cls.from_dict({(): 2})
        new_one = two - one
        assert new_one.equiv(one)

    def test_isub(self: _OperatorClassProvider):
        cls = self.get_class()
        op = cls.from_dict({(): 2})
        one = cls.one()
        op -= one
        assert op.equiv(one)

    def test_mul(self: _OperatorClassProvider):
        cls = self.get_class()
        one = cls.one()
        three = one * 3
        assert three.equiv(cls.from_dict({(): 3}))

    def test_rmul(self: _OperatorClassProvider):
        cls = self.get_class()
        one = cls.one()
        three = 3 * one
        assert three.equiv(cls.from_dict({(): 3}))

    def test_imul(self: _OperatorClassProvider):
        cls = self.get_class()
        op = cls.one()
        op *= 3
        assert op.equiv(cls.from_dict({(): 3}))

    def test_div(self: _OperatorClassProvider):
        cls = self.get_class()
        three = cls.from_dict({(): 3})
        one_half = three / 2.0
        assert one_half.equiv(cls.from_dict({(): 1.5}))

    def test_idiv(self: _OperatorClassProvider):
        cls = self.get_class()
        op = cls.from_dict({(): 3})
        op /= 2.0
        assert op.equiv(cls.from_dict({(): 1.5}))

    def test_neg(self: _OperatorClassProvider):
        cls = self.get_class()
        one = cls.one()
        assert (-one).equiv(cls.from_dict({(): -1}))

    def test_iter(self: _OperatorClassProvider):
        cls = self.get_class()
        op = cls.one()
        assert list(op.iter_terms()) == [([], 1)]

    def test_iter_with_groups(self: _OperatorClassProvider):
        cls = self.get_class()
        op = cls.one()
        op.groups = [0]
        assert list(op.iter_terms_with_groups()) == [([], 1, 0)]

    def test_equiv(self: _OperatorClassProvider):
        cls = self.get_class()
        op = cls.from_dict({(): 1e-7})
        zero = cls.zero()
        assert not op.equiv(zero)
        assert op.equiv(zero, 1e-6)
        assert not op.equiv(zero, 1e-8)

    def test_matmul(self: _OperatorClassProvider):
        """``@`` composes two operators.

        Declared on ``OperatorTrait`` and covered on the Rust side, but no Python test exercised the
        binding: ``__matmul__`` was only ever reached incidentally, from inside ``test_adjoint``.
        """
        cls = self.get_class()
        two = cls.from_dict({(): 2})
        three = cls.from_dict({(): 3})
        assert (two @ three).equiv(cls.from_dict({(): 6}))

    def test_imatmul(self: _OperatorClassProvider):
        """``@=`` composes in place, matching ``@``.

        ``__imatmul__`` had no Python references at all before this.
        """
        cls = self.get_class()
        op = cls.from_dict({(): 2})
        op @= cls.from_dict({(): 3})
        assert op.equiv(cls.from_dict({(): 6}))

    def test_matmul_leaves_operands_untouched(self: _OperatorClassProvider):
        """The out-of-place form must not mutate either operand."""
        cls = self.get_class()
        two = cls.from_dict({(): 2})
        three = cls.from_dict({(): 3})
        _ = two @ three
        assert two.equiv(cls.from_dict({(): 2}))
        assert three.equiv(cls.from_dict({(): 3}))

    def test_deepcopy_is_independent(self: _OperatorClassProvider):
        """``deepcopy`` must copy the group assignment, not share it.

        ``__deepcopy__`` was previously exercised only incidentally, by one filtering test, and
        nothing asserted that the copy is actually independent of the original.
        """
        cls = self.get_class()
        op = cls.from_dict({(): 1})
        op.groups = [0]

        clone = copy.deepcopy(op)
        clone.groups = [5]
        clone += cls.from_dict({(): 1})

        assert op.groups == [0], "mutating the copy's groups changed the original"
        assert op.equiv(cls.from_dict({(): 1})), "mutating the copy changed the original's terms"

    def test_pow_modulo_is_accepted(self: _OperatorClassProvider):
        """``__pow__`` takes an optional ``modulo`` argument, which no test passed before.

        ``pow(op, exponent, modulo)`` is how Python spells the three-argument form; the operator
        implementation ignores ``modulo``, and passing ``None`` explicitly must behave like omitting
        it rather than raising.
        """
        cls = self.get_class()
        two = cls.from_dict({(): 2})
        assert two.__pow__(2, None).equiv(cls.from_dict({(): 4}))
        assert pow(two, 2, None).equiv(cls.from_dict({(): 4}))
