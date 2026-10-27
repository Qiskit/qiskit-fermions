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

"""The Givens decomposition of an orbital rotation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from qiskit_fermions.linalg import givens_decomposition

from ._givens_factors import append_factor, givens_block

if TYPE_CHECKING:
    from qiskit_fermions.circuit import FermionicCircuit

    from ..orbital_rotation import OrbitalRotation


class FermionicGivensDecomposition:
    r"""The decomposition of an orbital rotation into Givens rotations and phases.

    The unitary :math:`U` of the :class:`.OrbitalRotation` is factorized with
    :func:`.givens_decomposition` into at most :math:`n(n-1)/2` rotations between *adjacent* modes
    plus :math:`n` single-mode phases, the rectangular mesh of Clements et al. [1]_. Every factor
    :math:`F` is itself an orbital rotation, whose many-body unitary is :math:`e^{-i H_F}` for the
    Hermitian generator :math:`H_F = \sum_{ij} i \log(F)_{ij} \, a^\dagger_i a_j`. It is therefore
    emitted as an :class:`.Evolution` gate at unit time on the one or two modes it acts on.

    The emitted gates are :attr:`~.Evolution.atomic`, and the operator of each holds a single group.
    The terms of a two-mode generator, :math:`a^\dagger_0 a_1` and :math:`a^\dagger_1 a_0`, do not
    commute with each other, so splitting it up any further would only approximate the factor: the
    fermion-to-qubit stage maps and synthesizes it whole. Under Jordan-Wigner, the generator of a
    rotation between adjacent modes is a hop between adjacent qubits, which the qubit stage can
    recognize (see :class:`.HopEvolutionSynthesis`).

    Because :math:`U \mapsto \exp(\sum_{ij} \log(U)_{ij} a^\dagger_i a_j)` is a group homomorphism,
    the product of the factors equals the full rotation exactly. The decomposition itself therefore
    adds no error: only the synthesis of the individual factors can.

    .. [1] W. R. Clements et al., Optimal design for universal multiport interferometers,
       Optica 3, 1460 (2016), `arXiv:1603.08788 <https://arxiv.org/abs/1603.08788>`_.
    """

    def __init__(self, *, atol: float = 1e-12) -> None:
        """Initializing an instance of this synthesis method can be done with the arguments below.

        Args:
            atol: the absolute tolerance below which factors and entries are dropped. This is
                applied to the entries of the generator of a factor and to the angle of a
                single-mode phase.
        """
        self.atol = atol
        """The magnitude below which entries of a generator and angles of a phase are dropped."""

    def synthesize(self, gate: OrbitalRotation) -> FermionicCircuit:
        """Synthesizes the gate into the mesh of atomic :class:`.Evolution` gates.

        See the class documentation for the decomposition this implements.

        Args:
            gate: the gate to synthesize.

        Returns:
            A :class:`.FermionicCircuit` holding one :class:`.Evolution` gate per two-mode rotation
            and per non-trivial phase, in the order in which they are applied. The rotations come
            first, then the phases.
        """
        # deferred import: `qiskit_fermions.circuit` imports this subpackage's siblings at module load
        from qiskit_fermions.circuit import FermionicCircuit

        rotations, phases = givens_decomposition(gate.rotation_unitary)
        definition = FermionicCircuit(gate.num_modes)

        # U = D conj(G_N) ... conj(G_1) and U -> exp(sum log(U)_ij a†_i a_j) is a group homomorphism,
        # so the circuit applies conj(G_1) first and the phases last.
        for c, s, i, j in rotations:
            append_factor(definition, givens_block(c, s), (i, j), atol=self.atol)
        for mode, phase in enumerate(phases):
            angle = float(np.angle(phase))
            if abs(angle) > self.atol:
                append_factor(definition, np.array([[np.exp(1j * angle)]]), (mode,), atol=self.atol)

        return definition
