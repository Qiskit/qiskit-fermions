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

"""Commutator functions."""

from typing import Protocol, TypeVar, cast, runtime_checkable

T = TypeVar("T")


@runtime_checkable
class SupportsCommutators(Protocol[T]):
    """A runtime-checkable Protocol indicating support for efficient commutator generation.

    Implementation of this protocol requires the three methods below.

    .. automethod:: _commutator_
    .. automethod:: _anti_commutator_
    .. automethod:: _double_commutator_
    """

    @staticmethod
    def _commutator_(op_a: T, op_b: T) -> T:
        """Computes the commutator of two operator instances.

        See :func:`.commutator` for more details.
        """
        ...

    @staticmethod
    def _anti_commutator_(op_a: T, op_b: T) -> T:
        """Computes the anti-commutator of two operator instances.

        See :func:`.anti_commutator` for more details.
        """
        ...

    @staticmethod
    def _double_commutator_(op_a: T, op_b: T, op_c: T, sign: bool) -> T:
        """Computes the double-commutator of three operator instances.

        See :func:`.double_commutator` for more details.
        """
        ...


def commutator(op_a: SupportsCommutators, op_b: SupportsCommutators) -> SupportsCommutators:
    r"""Computes the commutator of two operators.

    The commutator is defined as:

    .. math::

       [A, B] = AB - BA

    Operators which support this method must implement the :class:`.SupportsCommutators` protocol.

    .. note::
       Both inputs must be of the same operator type. This will also determine the output type.

    .. doctest::
       >>> from qiskit_fermions.operators import FermionOperator
       >>> from qiskit_fermions.operators.library import commutator
       >>> op1 = FermionOperator.from_dict({((True, 0), (False, 0)): 1})
       >>> op2 = FermionOperator.from_dict({((False, 0), (True, 0)): 2})
       >>> comm = commutator(op1, op2)
       >>> comm = comm.normal_ordered()
       >>> canon = comm.simplify()
       >>> assert canon == FermionOperator.zero()

    Args:
        op_a: the operator :math:`A` above.
        op_b: the operator :math:`B` above.

    Returns:
        The commutator :math:`[A, B]`.
    """
    return cast(SupportsCommutators, type(op_a)._commutator_(op_a, op_b))


def anti_commutator(op_a: SupportsCommutators, op_b: SupportsCommutators) -> SupportsCommutators:
    r"""Computes the anti-commutator of two operators.

    The anti-commutator is defined as:

    .. math::

       \{A, B\} = AB + BA

    Operators which support this method must implement the :class:`.SupportsCommutators` protocol.

    .. note::
       Both inputs must be of the same operator type. This will also determine the output type.

    .. doctest::
       >>> from qiskit_fermions.operators import FermionOperator
       >>> from qiskit_fermions.operators.library import anti_commutator
       >>> op1 = FermionOperator.from_dict({((True, 0), (False, 0)): 1})
       >>> op2 = FermionOperator.from_dict({((False, 0), (True, 0)): 2})
       >>> comm = anti_commutator(op1, op2)
       >>> comm = comm.normal_ordered()
       >>> canon = comm.simplify()
       >>> assert canon == FermionOperator.zero()

    Args:
        op_a: the operator :math:`A` above.
        op_b: the operator :math:`B` above.

    Returns:
        The anti-commutator :math:`\{A, B\}`.
    """
    return cast(SupportsCommutators, type(op_a)._anti_commutator_(op_a, op_b))


def double_commutator(
    op_a: SupportsCommutators,
    op_b: SupportsCommutators,
    op_c: SupportsCommutators,
    sign: bool,
) -> SupportsCommutators:
    r"""Computes the double-commutator of three operators.

    The double-commutator is defined as follows (see also Equation (13.6.18) in [1]_):

    If ``sign`` is ``False``, it returns

    .. math::
         [[A, B], C]/2 + [A, [B, C]]/2 = (2ABC + 2CBA - BAC - CAB - ACB - BCA)/2.

    If ``sign`` is ``True``, it returns

    .. math::
         \{[A, B], C\}/2 + \{A, [B, C]\}/2 = (2ABC - 2CBA - BAC + CAB - ACB + BCA)/2.

    Operators which support this method must implement the :class:`.SupportsCommutators` protocol.

    .. note::
       All three inputs must be of the same operator type. This will also determine the output type.

    .. doctest::
       >>> from qiskit_fermions.operators import FermionOperator
       >>> from qiskit_fermions.operators.library import anti_commutator
       >>> op1 = FermionOperator.from_dict({((True, 0), (False, 0)): 1})
       >>> op2 = FermionOperator.from_dict({((False, 0), (True, 0)): 2})
       >>> op3 = FermionOperator.from_dict(
       ...     {((True, 0), (False, 0)): 1, ((False, 0), (True, 0)): 2 + 0.5j}
       ... )
       >>> comm = double_commutator(op1, op2, op3, False)
       >>> comm = comm.normal_ordered()
       >>> canon = comm.simplify()
       >>> assert canon == FermionOperator.zero()

    .. [1] R. McWeeny. Methods of Molecular Quantum Mechanics.
           2nd Edition, Academic Press, 1992. ISBN 0-12-486552-6.

    Args:
        op_a: the operator :math:`A` above.
        op_b: the operator :math:`B` above.
        op_c: the operator :math:`C` above.
        sign: the nature of the outer (anti-)commutator as per the definition above.

    Returns:
        The double-commutator as per the definition above.
    """
    return cast(SupportsCommutators, type(op_a)._double_commutator_(op_a, op_b, op_c, sign))
