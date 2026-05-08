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

"""A QDrift Trotterization optimization pass."""

from __future__ import annotations

import numpy as np
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import TransformationPass

from qiskit_fermions.circuit.library import Evolution


class QDriftTrotterization(TransformationPass):
    """A transpilation pass to Trotterize :class:`.Evolution` gates via the qDRIFT protocol."""

    def __init__(self, num_terms: int, *, rng: np.random.Generator | int | None = None) -> None:
        """Initializes the transpiler pass.

        Args:
            num_terms: the number of terms to include in the qDRIFT Trotterization.
            rng: the random number generator (rng) to be used. When this is an ``int``, the internal
                rng will be initialized with ``np.random.default_rng(seed=rng)``.
        """
        super().__init__()

        self.num_terms = num_terms
        """The number of terms to include in the qDRIFT Trotterization."""

        self._rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Runs this transpilation pass.

        Args:
            dag: the input circuit with fermion-based instructions. Only
                :class:`~qiskit.dagcircuit.DAGOpNode` with :class:`.FermionGate` instances as their
                :attr:`~qiskit.dagcircuit.DAGOpNode.op` are supported.

        Returns:
            The output circuit which is still acting on a fermionic register.
        """
        out_dag = dag.copy_empty_like()

        for node in dag.op_nodes():
            if not isinstance(node.op, Evolution):
                continue

            hamil = node.op.operator
            time = node.op.params[0]
            num_fermions = len(node.qargs)

            if hamil.groups is not None:
                terms = hamil.split_out_groups()
                weights = [np.abs(np.mean(g.get_coeffs())) for g in terms]
            else:
                terms = list(hamil.iter_terms())
                weights = np.abs(hamil.get_coeffs())

            lambd = np.sum(weights)
            delta = (lambd * time) / self.num_terms

            sampled_indices = self._rng.choice(
                np.arange(len(weights)),
                size=self.num_terms,
                p=weights / lambd,
            )

            for ind in sampled_indices:
                term = terms[ind]
                op = hamil.__class__.from_terms([term]) if isinstance(term, tuple) else term
                evo = Evolution(num_fermions, op, time=delta)
                out_dag.apply_operation_back(evo, qargs=out_dag.qubits)

        return out_dag
