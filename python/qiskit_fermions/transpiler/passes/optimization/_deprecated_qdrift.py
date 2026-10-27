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

"""Backwards-compatibility shim for the former signature of ``QDriftTrotterization``.

This module is self-contained apart from the dispatch in ``QDriftTrotterization.__new__``: removing
the shim means deleting this module, that ``__new__`` and the test of the shim.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from qiskit.utils import deprecate_func

from qiskit_fermions.circuit import FermionicDAGCircuit
from qiskit_fermions.circuit.library.synthesis import FermionicQDrift

from .f2f_synthesis import F2FSynthesis
from .qdrift import QDriftTrotterization


class LegacyQDriftTrotterization(QDriftTrotterization):
    """**DEPRECATED** The former :class:`.QDriftTrotterization`, which sampled by itself.

    This is what ``QDriftTrotterization(num_terms, *, filter_trivial, weights, rng)`` returns.
    """

    @deprecate_func(
        since="0.4.0",
        package_name="qiskit-fermions",
        additional_msg=(
            "Use F2FSynthesis({'Evolution': FermionicQDrift(num_terms, ...)}) to sample, or "
            "QDriftTrotterization(FermionicQDrift(num_terms, ...)) to also filter the trivial "
            "draws."
        ),
    )
    def __init__(
        self,
        num_terms: int,
        *,
        filter_trivial: bool = False,
        weights: Sequence[float] | np.ndarray | None = None,
        rng: np.random.Generator | int | None = None,
    ) -> None:
        """Initializing this transpiler pass can be done with the arguments listed below.

        Args:
            num_terms: see :class:`.FermionicQDrift`.
            filter_trivial: whether to reject the draws that cannot move a particle.
            weights: see :class:`.FermionicQDrift`.
            rng: see :class:`.FermionicQDrift`.
        """
        super().__init__(FermionicQDrift(num_terms, weights=weights, rng=rng))
        self.num_terms = num_terms
        self.filter_trivial = filter_trivial
        self.weights = self.qdrift.weights

    def run(self, dag: FermionicDAGCircuit) -> FermionicDAGCircuit:
        """Runs this transpilation pass.

        Args:
            dag: the input circuit.

        Returns:
            The output circuit.
        """
        if self.filter_trivial:
            return super().run(dag)
        return F2FSynthesis({"Evolution": self.qdrift}).run(dag)
