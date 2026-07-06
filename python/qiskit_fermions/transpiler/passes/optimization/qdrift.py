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

from typing import Any

import numpy as np

from qiskit_fermions.circuit import FermionicDAGCircuit
from qiskit_fermions.circuit.library import Evolution

from ... import FermionicDAGCircuitPass


class QDriftTrotterization(FermionicDAGCircuitPass):
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

    def run(self, dag: FermionicDAGCircuit) -> FermionicDAGCircuit:
        """Runs this transpilation pass.

        Args:
            dag: the input circuit with fermion-based instructions. Only
                :class:`~qiskit.dagcircuit.DAGOpNode` with :class:`.FermionicGate` instances as their
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
            num_modes = len(node.qargs)

            terms: list[Any]  # can be either a list of operator terms or operator instances
            if hamil.groups is None:
                # NOTE: the qDRIFT protocol replaces the operator's coefficients with identities
                # because the evolution time is entirely dictated by `delta` (computed below), since
                # the operator's coefficients only impact the probability of a term being included.
                terms = [(actions, 1.0) for actions, _ in hamil.iter_terms()]
                weights = np.abs(hamil.get_coeffs())
            else:
                all_coeffs = hamil.get_coeffs()
                groups = hamil.groups
                weights = np.zeros((hamil.num_groups(),))
                np.add.at(weights, groups, np.abs(all_coeffs))
                weights /= np.unique(groups, return_counts=True)[1]
                # NOTE: we do not pre-process the coefficients of these different group terms here,
                # because we would unnecessarily need to loop over all terms in all groups. Instead,
                # we replace the coefficients by identity values only once that particular group
                # term actually gets sampled.
                terms = hamil.split_out_groups()

            lambd = np.sum(weights)
            delta = (lambd * time) / self.num_terms

            sampled_indices = self._rng.choice(
                np.arange(len(weights)),
                size=self.num_terms,
                p=weights / lambd,
            )

            for ind in sampled_indices:
                sampled_term = terms[ind]
                if isinstance(sampled_term, tuple):
                    # in this case, we have already replaced the operator coefficient by 1.0
                    identity_terms = [sampled_term]
                else:
                    # NOTE: as per the comment earlier, we have not replaced the coefficients with
                    # identity values in the case of working with grouped operator terms.
                    identity_terms = [(actions, 1.0) for actions, _ in sampled_term.iter_terms()]

                op = hamil.__class__.from_terms(identity_terms)
                evo = Evolution(num_modes, op, time=delta)
                out_dag.apply_operation_back(evo, qargs=out_dag.qubits)

        return out_dag
