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
from typing import TYPE_CHECKING, Any

import numpy as np

from qiskit_fermions.circuit import FermionicDAGCircuit
from qiskit_fermions.circuit.library import Evolution

from ... import FermionicDAGCircuitPass

if TYPE_CHECKING:
    from qiskit_fermions._lib.operators.terms.filtering import filter_diagonal_terms
else:
    from qiskit_fermions.operators.terms.filtering import filter_diagonal_terms


class QDriftTrotterization(FermionicDAGCircuitPass):
    """A transpilation pass to Trotterize :class:`.Evolution` gates via the qDRIFT protocol."""

    def __init__(
        self,
        num_terms: int,
        *,
        filter_diagonal_terms: bool = False,
        rng: np.random.Generator | int | None = None,
    ) -> None:
        """Initializing this transpiler pass can be done with the arguments listed below.

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

            if self.filter_diagonal_terms:
                # Work on a copy so that the user's original operator (held by the Evolution gate)
                # is left untouched. Filtering re-indexes any group information to a contiguous
                # range, keeping the grouped branch below consistent.
                hamil = copy.deepcopy(hamil)
                filter_diagonal_terms(hamil)

            terms: list[Any]  # can be either a list of operator terms or operator instances
            if hamil.groups is None:
                # NOTE: the qDRIFT protocol normalizes each term to unit magnitude because the
                # evolution time is entirely dictated by `delta` (computed below). Only the
                # magnitude of a coefficient sets its sampling probability, but its sign fixes the
                # direction of the rotation and must be preserved for the Trotterization to
                # approximate the target time evolution.
                terms = [(actions, np.sign(coeff)) for actions, coeff in hamil.iter_terms()]
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
                    # in this case, we have already normalized the coefficient to its sign
                    unit_terms = [sampled_term]
                else:
                    # NOTE: as per the comment earlier, we have not yet normalized the coefficients
                    # of grouped operator terms. Keep each term's sign (dropping only its magnitude)
                    # so that the sampled rotation points in the correct direction.
                    unit_terms = [
                        (actions, np.sign(coeff)) for actions, coeff in sampled_term.iter_terms()
                    ]

                op = hamil.__class__.from_terms(unit_terms)
                evo = Evolution(num_modes, op, time=delta)
                out_dag.apply_operation_back(evo, qargs=out_dag.qubits)

        return out_dag
