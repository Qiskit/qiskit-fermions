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

import copy
from typing import TYPE_CHECKING

import numpy as np
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import TransformationPass

from qiskit_fermions.circuit.library import Evolution

if TYPE_CHECKING:
    from qiskit_fermions._lib.operators.terms.filtering import filter_diagonal_terms
else:
    from qiskit_fermions.operators.terms.filtering import filter_diagonal_terms


class QDriftTrotterization(TransformationPass):
    """A transpilation pass to Trotterize :class:`.Evolution` gates via the qDRIFT protocol."""

    def __init__(
        self,
        num_terms: int,
        *,
        filter_diagonal_terms: bool = False,
        rng: np.random.Generator | int | None = None,
    ) -> None:
        """Initializes the transpiler pass.

        Args:
            num_terms: the number of terms to include in the qDRIFT Trotterization.
            filter_diagonal_terms: when set to ``True``, terms that are diagonal in the
                occupation-number basis (i.e. products of number operators) are removed from the
                Hamiltonian before the qDRIFT sampling. The time evolution of such terms does not
                affect the sampled bitstrings, so including them would only increase the sampling
                overhead. This automates the manual filtering otherwise required when preparing a
                Hamiltonian for SqDRIFT. The Hamiltonian is assumed to be normal-ordered. See also
                :func:`~qiskit_fermions.operators.terms.filtering.filter_diagonal_terms`.
            rng: the random number generator (rng) to be used. When this is an ``int``, the internal
                rng will be initialized with ``np.random.default_rng(seed=rng)``.
        """
        super().__init__()

        self.num_terms = num_terms
        """The number of terms to include in the qDRIFT Trotterization."""

        self.filter_diagonal_terms = filter_diagonal_terms
        """Whether to filter out diagonal terms before the qDRIFT sampling."""

        self._rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

    def run(self, dag: DAGCircuit) -> DAGCircuit:
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

            if self.filter_diagonal_terms:
                # Work on a copy so that the user's original operator (held by the Evolution gate)
                # is left untouched. Filtering re-indexes any group information to a contiguous
                # range, keeping the grouped branch below consistent.
                hamil = copy.deepcopy(hamil)
                filter_diagonal_terms(hamil)

            if hamil.groups is not None:
                terms = hamil.split_out_groups()
                all_coeffs = hamil.get_coeffs()
                groups = hamil.groups
                max_group_idx = max(groups)
                weights = np.zeros(
                    (max_group_idx + 1),
                )
                np.add.at(weights, groups, np.abs(all_coeffs))
                weights /= np.unique(groups, return_counts=True)[1]
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
                evo = Evolution(num_modes, op, time=delta)
                out_dag.apply_operation_back(evo, qargs=out_dag.qubits)

        return out_dag
