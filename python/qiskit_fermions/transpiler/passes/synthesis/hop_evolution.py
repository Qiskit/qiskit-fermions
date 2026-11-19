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
) -> tuple[QuantumCircuit, tuple[int, ...]] | None:
    r"""Recognizes the evolution under a hop between two qubits, with or without a Z-string.

    Let :math:`a < b` be two qubits and :math:`M` a (possibly empty) set of other qubits. A hop is
    the generator

    .. math::

        H = Z_M \left[ \alpha (X_a X_b + Y_a Y_b) + \beta (X_a Y_b - Y_a X_b) \right],
        \qquad Z_M = \prod_{m \in M} Z_m.

    Under Jordan-Wigner, it is the image of a fermionic hop between adjacent modes for an empty
    :math:`M`, and of one across the modes mapped to the qubits :math:`a < m < b` otherwise. Nothing
    requires :math:`M` to lie between :math:`a` and :math:`b`, though. Terms that are absent count as
    zero, so a pure :math:`XX + YY` or a pure :math:`XY - YX` generator also matches.

    With the evolution time :math:`t`, the hop without the string evolves as
    :math:`e^{-i t h} = \mathrm{XXPlusYY}(\theta, \varphi)` for :math:`\theta = 4 t \sqrt{\alpha^2 +
    \beta^2}` and :math:`\varphi = \operatorname{atan2}(\beta, \alpha)`. The sign of :math:`t` lives
    in :math:`\theta` alone, so :math:`\varphi` must not depend on it. The string :math:`Z_M`
    commutes with :math:`h` and only flips the sign of the evolution time on the odd-parity subspace
    of :math:`M`, and so does conjugating :math:`h` with :math:`Z_a`. The returned circuit therefore
    computes the parity of :math:`M` onto the lowest qubit of :math:`M` with a ladder of
    :class:`~qiskit.circuit.library.CXGate`, applies the :class:`~qiskit.circuit.library.XXPlusYYGate`
    between two :class:`~qiskit.circuit.library.CZGate` controlled on that parity, and undoes the
    ladder. For an empty :math:`M`, it is the :class:`~qiskit.circuit.library.XXPlusYYGate` alone.
    In general, it costs two :class:`~qiskit.circuit.library.CZGate` and :math:`2(|M| - 1)`
    :class:`~qiskit.circuit.library.CXGate` on top of the
    :class:`~qiskit.circuit.library.XXPlusYYGate`.

    The circuit is the exact evolution. The :math:`XX + YY` and :math:`XY - YX` parts do not commute
    (unless one of them vanishes), so a product formula only approximates the same evolution, with a
    Trotter error. Nothing about the qubits is assumed: the match is an operator identity rather
    than a statement about the state the circuit acts on.

    Args:
        observable: the qubit operator.
        time: the evolution time :math:`t` in :math:`e^{-i t H}`.
        atol: the absolute tolerance within which the coefficient pairs must agree and the
            imaginary parts must vanish.

    Returns:
        The circuit and the qubit indices it acts on, such that its qubit ``i`` is the qubit
        ``indices[i]`` of ``observable``, or ``None`` if ``observable`` is not a hop. The indices
        start with :math:`a` and end with :math:`b`, with :math:`M` in ascending order in between.
    """
    terms = observable.to_sparse_list()
    if not 0 < len(terms) <= _MAX_TERMS:
        return None

    support: tuple[int, ...] | None = None
    ends: tuple[int, int] | None = None
    coeffs = {"XX": 0.0, "YY": 0.0, "XY": 0.0, "YX": 0.0}
    for label, indices, coeff in terms:
        # every qubit but the two ends carries a Z
        term_ends = tuple(pos for pos, pauli in enumerate(label) if pauli != "Z")
        if len(term_ends) != 2:
            return None
        if support is None:
            # ``to_sparse_list`` sorts the indices of each term, so ``support`` is sorted
            support, ends = tuple(indices), term_ends
        elif tuple(indices) != support or term_ends != ends:
            return None
        key = label[term_ends[0]] + label[term_ends[1]]
        if key not in coeffs:
            return None
        if abs(complex(coeff).imag) > atol:
            return None
        coeffs[key] += complex(coeff).real

    assert support is not None and ends is not None
    if not np.isclose(coeffs["XX"], coeffs["YY"], rtol=0.0, atol=atol):
        return None
    if not np.isclose(coeffs["XY"], -coeffs["YX"], rtol=0.0, atol=atol):
        return None

    # the key ``XY`` is X on the lower and Y on the higher end
    alpha, beta = coeffs["XX"], coeffs["XY"]
    if abs(alpha) <= atol and abs(beta) <= atol:
        return None

    theta = 4.0 * time * np.hypot(alpha, beta)
    phi = np.arctan2(beta, alpha)

    # local qubit 0 is the end a, the last one the end b, and 1..k the string, in ascending order
    strings = tuple(index for pos, index in enumerate(support) if pos not in ends)
    indices = (support[ends[0]], *strings, support[ends[1]])
    last = len(indices) - 1
    ladder = QuantumCircuit(len(indices))
    for middle in range(last - 1, 1, -1):
        ladder.cx(middle, middle - 1)

    circuit = QuantumCircuit(len(indices))
    if last > 1:
        circuit.compose(ladder, inplace=True)
        circuit.cz(1, 0)
    circuit.append(XXPlusYYGate(theta, phi), [0, last])
    if last > 1:
        circuit.cz(1, 0)
        circuit.compose(ladder.inverse(), inplace=True)
    return circuit, indices


class HopEvolutionSynthesis(HighLevelSynthesisPlugin):
    r"""A high-level synthesis plugin for the evolution under a hop between two qubits.

    A :class:`~qiskit.circuit.library.PauliEvolutionGate` whose operator is a hop (see
    :func:`match_hop`), such as the image of a fermionic hop under Jordan-Wigner, is synthesized
    exactly around a single :class:`~qiskit.circuit.library.XXPlusYYGate`. A hop between adjacent
    modes becomes that gate alone, and a hop across other modes adds the two
    :class:`~qiskit.circuit.library.CZGate` and the :class:`~qiskit.circuit.library.CXGate` ladder
    that account for its Z-string. The generic synthesis of the same operator costs more two-qubit
    gates, and a target whose only two-qubit gate is the
    :class:`~qiskit.circuit.library.XXPlusYYGate` cannot be reached by it for an adjacent hop at
    all.

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
    the configuration cannot turn that off. A factor between modes that are adjacent in the
    :class:`.OrbitalRotation` but not on the register, because its wires are permuted or not
    contiguous, is a hop across the modes in between and is synthesized exactly, too.
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
            The circuit of :func:`match_hop` if ``high_level_object`` is a
            :class:`~qiskit.circuit.library.PauliEvolutionGate` of a hop with a fixed evolution
            time, and ``None`` otherwise.
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

        hop, indices = matched
        circuit = QuantumCircuit(high_level_object.num_qubits)
        circuit.compose(hop, list(indices), inplace=True)
        return circuit
