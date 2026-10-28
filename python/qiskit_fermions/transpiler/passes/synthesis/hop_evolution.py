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

"""High-level synthesis of the evolution under a hop between two qubits."""

from __future__ import annotations

import numbers

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate, XXPlusYYGate
from qiskit.quantum_info import SparseObservable, SparsePauliOp
from qiskit.transpiler.passes.synthesis.plugin import HighLevelSynthesisPlugin

DEFAULT_ATOL = 1e-12
"""The default absolute tolerance on the coefficients of a recognized generator."""

_MAX_TERMS = 4
"""The most terms a hop can have, which bounds the cost of a failed match."""


def match_hop(
    observable: SparseObservable, time: float, *, atol: float = DEFAULT_ATOL
) -> tuple[XXPlusYYGate, tuple[int, int]] | None:
    r"""Recognizes the evolution under a hop between two qubits.

    Let :math:`a` and :math:`b` be two qubits. A hop is the generator

    .. math::

        H = \alpha (X_a X_b + Y_a Y_b) + \beta (X_a Y_b - Y_a X_b),

    which is the image of a fermionic hop between adjacent modes under Jordan-Wigner. With the
    evolution time :math:`t`, :math:`e^{-i t H} = \mathrm{XXPlusYY}(\theta, \varphi)` for
    :math:`\theta = 4 t \sqrt{\alpha^2 + \beta^2}` and :math:`\varphi = \operatorname{atan2}(\beta,
    \alpha)`. The sign of :math:`t` lives in :math:`\theta` alone, so :math:`\varphi` must not depend
    on it. Terms that are absent count as zero, so a pure :math:`XX + YY` or a pure :math:`XY - YX`
    generator also matches.

    The gate is the exact evolution. The :math:`XX + YY` and :math:`XY - YX` parts do not commute
    (unless one of them vanishes), so a product formula only approximates the same evolution, with a
    Trotter error. Nothing about the two qubits is assumed. A hop across other qubits maps to a
    longer Pauli string and is therefore not recognized, which keeps the match an operator identity
    rather than a statement about the state the gate acts on.

    Args:
        observable: the qubit operator.
        time: the evolution time :math:`t` in :math:`e^{-i t H}`.
        atol: the absolute tolerance within which the coefficient pairs must agree and the
            imaginary parts must vanish.

    Returns:
        The gate and the pair of qubit indices it acts on, in the gate's qubit order, or ``None`` if
        ``observable`` is not a hop.
    """
    terms = observable.to_sparse_list()
    if not 0 < len(terms) <= _MAX_TERMS:
        return None

    support: tuple[int, ...] | None = None
    coeffs = {"XX": 0.0, "YY": 0.0, "XY": 0.0, "YX": 0.0}
    for label, indices, coeff in terms:
        if label not in coeffs or len(indices) != 2:
            return None
        if support is None:
            support = tuple(indices)
        elif tuple(indices) != support:
            return None
        if abs(complex(coeff).imag) > atol:
            return None
        coeffs[label] += complex(coeff).real

    assert support is not None
    if not np.isclose(coeffs["XX"], coeffs["YY"], rtol=0.0, atol=atol):
        return None
    if not np.isclose(coeffs["XY"], -coeffs["YX"], rtol=0.0, atol=atol):
        return None

    # the sparse label ``XY`` is X on the first and Y on the second index of ``support``
    alpha, beta = coeffs["XX"], coeffs["XY"]
    if abs(alpha) <= atol and abs(beta) <= atol:
        return None

    theta = 4.0 * time * np.hypot(alpha, beta)
    phi = np.arctan2(beta, alpha)
    return XXPlusYYGate(theta, phi), (support[0], support[1])


class HopEvolutionSynthesis(HighLevelSynthesisPlugin):
    r"""A high-level synthesis plugin for the evolution under a hop between two qubits.

    A :class:`~qiskit.circuit.library.PauliEvolutionGate` whose operator is a hop (see
    :func:`match_hop`), such as the image of a fermionic hop between adjacent modes under
    Jordan-Wigner, is synthesized as a single :class:`~qiskit.circuit.library.XXPlusYYGate`. That is
    the exact evolution. The generic synthesis of the same operator costs several times as many
    two-qubit gates, and a target whose only two-qubit gate is the
    :class:`~qiskit.circuit.library.XXPlusYYGate` cannot be reached by it at all.

    Any other gate is declined (the plugin returns ``None``), so that the next method of the
    :class:`~qiskit.transpiler.passes.synthesis.high_level_synthesis.HLSConfig` is tried.

    .. note::
       The :attr:`~qiskit.circuit.library.PauliEvolutionGate.synthesis` of a matched gate does not
       apply: the result is exact, whereas a product formula approximates the evolution of a hop
       with a complex amplitude (its :math:`XX + YY` and :math:`XY - YX` parts do not commute). A
       matched gate therefore differs from the generic synthesis by that Trotter error, and is the
       more accurate of the two.

    Register it ahead of the default method::

        HLSConfig(PauliEvolution=[HopEvolutionSynthesis(), "default"])

    The :func:`.generate_preset_jw_pass_manager` does so unconditionally, and merges the entry into
    any ``hls_config`` passed to it. That mirrors how the preset has always lowered an orbital
    rotation: it emits an :class:`~qiskit.circuit.library.XXPlusYYGate` for each Givens factor, and
    the configuration cannot turn that off.
    """

    def __init__(self, *, atol: float = DEFAULT_ATOL) -> None:
        """Initializing this plugin can be done with the arguments listed below.

        Args:
            atol: the absolute tolerance on the coefficients of the operator, within which it is
                still considered a hop.
        """
        super().__init__()

        self.atol = atol
        """The absolute coefficient tolerance with which an operator is recognized as a hop."""

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Runs this plugin.

        Args:
            high_level_object: the operation to synthesize.
            coupling_map: unused.
            target: unused.
            qubits: unused.
            options: unused.

        Returns:
            A circuit with a single :class:`~qiskit.circuit.library.XXPlusYYGate` if
            ``high_level_object`` is a :class:`~qiskit.circuit.library.PauliEvolutionGate` of a hop
            with a fixed evolution time, and ``None`` otherwise.
        """
        if not isinstance(high_level_object, PauliEvolutionGate):
            return None

        operator = high_level_object.operator
        if isinstance(operator, SparsePauliOp):
            operator = SparseObservable.from_sparse_pauli_op(operator)
        if not isinstance(operator, SparseObservable):
            return None

        time = high_level_object.params[0]
        if not isinstance(time, numbers.Real):
            return None

        matched = match_hop(operator, float(time), atol=self.atol)
        if matched is None:
            return None

        gate, indices = matched
        circuit = QuantumCircuit(high_level_object.num_qubits)
        circuit.append(gate, list(indices))
        return circuit
