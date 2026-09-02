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

import pytest
from qiskit_fermions.operators import (
    EdgeVertexOperator,
    FermionOperator,
    MajoranaOperator,
    OperatorTrait,
    TransferVertexOperator,
)

OPERATOR_CLASSES = [
    FermionOperator,
    MajoranaOperator,
    EdgeVertexOperator,
    TransferVertexOperator,
]


def _is_hermitian_via_protocol(op: OperatorTrait, atol: float) -> bool:
    """Calls ``is_hermitian`` through a protocol-typed binding.

    The annotation is the point of this helper: it asserts that the member is reachable through the
    protocol rather than only through a concrete class, independently of which one is passed in.
    """
    return op.is_hermitian(atol)


class TestOperatorTraitConformance:
    """Tests the parts of :class:`.OperatorTrait` that every operator type must provide."""

    @pytest.mark.parametrize("cls", OPERATOR_CLASSES)
    def test_is_hermitian_through_protocol(self, cls, subtests):
        # `zero` and `one` are the additive and multiplicative identities on every operator type,
        # and both are Hermitian. Neither requires knowing the term vocabulary of the specific type,
        # which is what lets this test be generic over all four of them.
        with subtests.test("zero is Hermitian", cls=cls.__name__):
            assert _is_hermitian_via_protocol(cls.zero(), 1e-8)

        with subtests.test("identity is Hermitian", cls=cls.__name__):
            assert _is_hermitian_via_protocol(cls.one(), 1e-8)

        with subtests.test("default atol is accepted", cls=cls.__name__):
            # Pins the default argument that the protocol member declares.
            assert cls.one().is_hermitian()

        with subtests.test("imaginary identity is not Hermitian", cls=cls.__name__):
            # An anti-Hermitian operator, again expressible on every type. Without this case, an
            # implementation that unconditionally returned `True` would satisfy the ones above.
            assert not _is_hermitian_via_protocol(cls.one() * 1j, 1e-8)
