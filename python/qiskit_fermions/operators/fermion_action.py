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

import sys
from typing import NamedTuple

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


class FermionAction(NamedTuple):
    """A fermionic creation or annihilation action."""

    action: bool
    """Whether this action is a creation (``True``) or annihilation (``False``) one."""

    mode: int
    """The spin-less fermionic mode index on which to act."""

    @classmethod
    def creation(cls, mode: int) -> Self:
        """Constructs a creation action on the fermionic mode at ``mode``.

        Args:
            mode: the spin-less fermionic mode on which to act.
        """
        return cls(action=True, mode=mode)

    @classmethod
    def annihilation(cls, mode: int) -> Self:
        """Constructs an annihilation action on the fermionic mode at ``mode``.

        Args:
            mode: the spin-less fermionic mode on which to act.
        """
        return cls(action=False, mode=mode)


cre = FermionAction.creation
"""A convenience alias for :meth:`FermionAction.creation`."""

ann = FermionAction.annihilation
"""A convenience alias for :meth:`FermionAction.annihilation`."""
