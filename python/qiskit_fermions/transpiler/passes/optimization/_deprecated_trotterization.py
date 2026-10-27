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

"""Backwards-compatibility shim for the former ``FermionicTrotterization`` pass.

This module is self-contained: deleting it, together with its exports and its test, removes the
shim.
"""

from __future__ import annotations

from collections.abc import Callable

from qiskit.dagcircuit import DAGOpNode
from qiskit.utils import deprecate_func

from qiskit_fermions.circuit.library.synthesis import F2FSynthesisPlugin

from .f2f_synthesis import F2FSynthesis


class FermionicTrotterization(F2FSynthesis):
    """The former pass to Trotterize :class:`.Evolution` gates.

    This is :class:`.F2FSynthesis` restricted to the :class:`.Evolution` gate: the ``synthesis``
    argument is the method for that gate.
    """

    @deprecate_func(
        since="0.4.0",
        package_name="qiskit-fermions",
        additional_msg='Use F2FSynthesis({"Evolution": synthesis}) instead.',
    )
    def __init__(
        self,
        synthesis: F2FSynthesisPlugin,
        *,
        apply: bool = True,
        filter: Callable[[DAGOpNode], bool] | None = None,  # noqa: A002
    ) -> None:
        """Initializing this transpiler pass can be done with the arguments listed below.

        Args:
            synthesis: the synthesis method for :class:`.Evolution` gates.
            apply: see :class:`.F2FSynthesis`.
            filter: see :class:`.F2FSynthesis`.
        """
        super().__init__({"Evolution": synthesis}, apply=apply, filter=filter)
