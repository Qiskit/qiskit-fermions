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

"""The FermionAction type."""

from typing import TypeAlias

FermionAction: TypeAlias = tuple[bool, int]
"""A fermionic creation or annihilation action.

A 2-tuple ``(action, mode)`` where ``action`` is ``True`` for a creation and ``False`` for an
annihilation action, and ``mode`` is the spin-less fermionic mode index on which to act.
"""


def cre(mode: int) -> FermionAction:
    """Constructs a creation action on the fermionic mode at ``mode``.

    Args:
        mode: the spin-less fermionic mode on which to act.

    Returns:
        A creation :class:`.FermionAction` on ``mode``.
    """
    return (True, mode)


def ann(mode: int) -> FermionAction:
    """Constructs an annihilation action on the fermionic mode at ``mode``.

    Args:
        mode: the spin-less fermionic mode on which to act.

    Returns:
        An annihilation :class:`.FermionAction` on ``mode``.
    """
    return (False, mode)
