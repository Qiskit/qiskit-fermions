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

"""The MajoranaAction type."""

from typing import NewType, cast

MajoranaAction = NewType("MajoranaAction", int)

MajoranaAction.__doc__ = """The MajoranaAction type. See :func:`.gamma` for more details."""


def gamma(mode: int, is_prime: bool) -> MajoranaAction:
    r"""Create a majorana fermion.

    For a given mode ``i``, two majorana fermions can be created:
        - ``gamma(i, False)`` creates :math:`\gamma = a_i^\dagger + a_i`
        - ``gamma(i, True)`` creates :math:`\gamma' = i(a_i^\dagger - a_i)`

    The argument order is ``(mode, is_prime)`` to reflect the natural interpretation of
    majorana operators: first specify the mode, then the variant. Unlike fermionic operators --
    where the distinction between creation and annihilation is fundamental -- the two majorana
    variants are more symmetric and conceptually less different.

    This mirrors the normal-ordering convention for majoranas, where operators are sorted first by
    mode and then by variant. For example, a normally ordered string like
    ``((1, True), (0, False), (0, True)`` is more readable and intuitive than
    ``((True, 1), (False, 0), (True, 0)``.

    Args:
        mode: index of the fermionic mode.
        is_prime: whether to create :math:`\gamma` (False) or :math:`\gamma'` (True).

    Returns:
        A ``MajoranaAction`` object, which is essentially the flat index ``2*mode+int(is_prime)``.
    """
    return cast(MajoranaAction, 2 * mode + int(is_prime))
