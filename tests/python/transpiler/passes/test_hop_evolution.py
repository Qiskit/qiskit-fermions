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

"""Hop evolution synthesis tests."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import PauliEvolutionGate, XXPlusYYGate
from qiskit.quantum_info import Operator, SparseObservable, SparsePauliOp
from qiskit.synthesis import SuzukiTrotter
from qiskit.transpiler import generate_preset_pass_manager
from qiskit.transpiler.passes.synthesis.high_level_synthesis import HLSConfig
from qiskit_fermions.circuit import FermionicCircuit
from qiskit_fermions.circuit.library import Evolution
from qiskit_fermions.mappers.library import jordan_wigner
from qiskit_fermions.operators import FermionOperator
from qiskit_fermions.transpiler.passes import HopEvolutionSynthesis
from qiskit_fermions.transpiler.passes.synthesis.hop_evolution import match_hop
from qiskit_fermions.transpiler.presets import generate_preset_jw_pass_manager


def _hop_observable(alpha, beta, qubits, num_qubits):
    return SparseObservable.from_sparse_list(
        [
            ("XX", qubits, alpha),
            ("YY", qubits, alpha),
            ("XY", qubits, beta),
            ("YX", qubits, -beta),
        ],
        num_qubits=num_qubits,
    )


def _evolution_matrix(observable, time):
    return scipy.linalg.expm(
        -1j * time * SparsePauliOp.from_sparse_observable(observable).to_matrix()
    )


def _embedded_matrix(gate, qubits, num_qubits):
    circ = QuantumCircuit(num_qubits)
    circ.append(gate, list(qubits))
    return Operator(circ).data


@pytest.mark.parametrize("qubits", [[0, 1], [1, 0], [0, 2], [3, 1]])
def test_match_hop_is_the_exact_evolution(qubits):
    """Whatever the match returns must be the exact unitary, on any pair of qubits."""
    rng = np.random.default_rng(7)
    num_qubits = 4
    for _ in range(25):
        alpha, beta = rng.normal(size=2)
        time = rng.uniform(-2.0, 2.0)
        observable = _hop_observable(alpha, beta, qubits, num_qubits)

        match = match_hop(observable, time)
        assert match is not None
        gate, gate_qubits = match

        expected = _evolution_matrix(observable, time)
        np.testing.assert_allclose(
            _embedded_matrix(gate, gate_qubits, num_qubits), expected, atol=1e-12
        )


@pytest.mark.parametrize(("alpha", "beta"), [(0.7, 0.0), (0.0, 0.7), (-0.3, 0.0), (0.0, -0.3)])
def test_match_hop_partial_generators(alpha, beta):
    observable = _hop_observable(alpha, beta, [0, 1], 2)
    match = match_hop(observable, 1.3)
    assert match is not None
    np.testing.assert_allclose(
        _embedded_matrix(match[0], match[1], 2), _evolution_matrix(observable, 1.3), atol=1e-12
    )


def test_match_hop_ignores_term_order():
    terms = [("YX", [0, 1], -0.4), ("XX", [0, 1], 0.9), ("XY", [0, 1], 0.4), ("YY", [0, 1], 0.9)]
    observable = SparseObservable.from_sparse_list(terms, num_qubits=2)
    match = match_hop(observable, 0.5)
    assert match is not None
    np.testing.assert_allclose(
        _embedded_matrix(match[0], match[1], 2), _evolution_matrix(observable, 0.5), atol=1e-12
    )


@pytest.mark.parametrize(
    "terms",
    [
        # unequal XX and YY
        [("XX", [0, 1], 1.0), ("YY", [0, 1], 0.5)],
        # XY without the matching -YX
        [("XY", [0, 1], 1.0), ("YX", [0, 1], 1.0)],
        # a Z-string in between (the JW image of a hop across a mode)
        [("XZX", [0, 1, 2], 1.0), ("YZY", [0, 1, 2], 1.0)],
        # an extra, non-matching term
        [("XX", [0, 1], 1.0), ("YY", [0, 1], 1.0), ("ZZ", [0, 1], 0.1)],
        # terms on different pairs of qubits
        [("XX", [0, 1], 1.0), ("YY", [1, 2], 1.0)],
        # a single-qubit term
        [("X", [0], 1.0)],
    ],
)
def test_match_hop_rejects_other_generators(terms):
    observable = SparseObservable.from_sparse_list(terms, num_qubits=3)
    assert match_hop(observable, 1.0) is None


def test_match_hop_rejects_the_empty_operator():
    assert match_hop(SparseObservable.zero(2), 1.0) is None


def test_match_hop_respects_the_tolerance():
    observable = SparseObservable.from_sparse_list(
        [("XX", [0, 1], 1.0), ("YY", [0, 1], 1.0 + 1e-9)], num_qubits=2
    )
    assert match_hop(observable, 1.0) is None
    assert match_hop(observable, 1.0, atol=1e-6) is not None


def _transpile(circ, hls_config, **kwargs):
    return generate_preset_pass_manager(hls_config=hls_config, **kwargs).run(circ)


def test_plugin_emits_one_xx_plus_yy():
    observable = _hop_observable(0.7, 0.2, [0, 1], 2)
    circ = QuantumCircuit(2)
    circ.append(PauliEvolutionGate(observable, time=0.5), [0, 1])
    config = HLSConfig(PauliEvolution=[HopEvolutionSynthesis(), "default"])

    out = _transpile(circ, config, optimization_level=0, basis_gates=["xx_plus_yy", "u3"])

    assert dict(out.count_ops()) == {"xx_plus_yy": 1}
    np.testing.assert_allclose(Operator(out).data, _evolution_matrix(observable, 0.5), atol=1e-12)


def test_plugin_beats_the_generic_synthesis_in_cx_count():
    observable = _hop_observable(0.7, 0.2, [0, 1], 2)
    circ = QuantumCircuit(2)
    circ.append(PauliEvolutionGate(observable, time=0.5), [0, 1])
    config = HLSConfig(PauliEvolution=[HopEvolutionSynthesis(), "default"])

    generic = _transpile(circ, None, optimization_level=1, basis_gates=["cx", "u3"])
    matched = _transpile(circ, config, optimization_level=1, basis_gates=["cx", "u3"])

    assert matched.count_ops()["cx"] == 2
    assert matched.count_ops()["cx"] < generic.count_ops()["cx"]
    # the match is exact, whereas the generic product formula is not: the XX + YY and XY - YX parts
    # of a complex hop do not commute
    exact = _evolution_matrix(observable, 0.5)
    np.testing.assert_allclose(Operator(matched).data, exact, atol=1e-12)
    assert np.abs(Operator(generic).data - exact).max() > 1e-3


def test_plugin_declines_other_operators():
    """Anything that is not a hop is left to the next method, and keeps the product formula."""
    observable = SparseObservable.from_sparse_list(
        [("XZX", [0, 1, 2], 1.0), ("YZY", [0, 1, 2], 1.0)], num_qubits=3
    )
    gate = PauliEvolutionGate(observable, time=0.5, synthesis=SuzukiTrotter(order=2, reps=2))
    circ = QuantumCircuit(3)
    circ.append(gate, [0, 1, 2])
    config = HLSConfig(PauliEvolution=[HopEvolutionSynthesis(), "default"])

    kwargs = {"optimization_level": 0, "basis_gates": ["cx", "u3"]}
    with_plugin = _transpile(circ, config, **kwargs)
    without = _transpile(circ, None, **kwargs)

    assert dict(with_plugin.count_ops()) == dict(without.count_ops())
    assert "xx_plus_yy" not in with_plugin.count_ops()


def test_plugin_ignores_a_user_product_formula_on_a_match():
    observable = _hop_observable(0.7, 0.2, [0, 1], 2)
    gate = PauliEvolutionGate(observable, time=0.5, synthesis=SuzukiTrotter(order=2, reps=3))
    circ = QuantumCircuit(2)
    circ.append(gate, [0, 1])
    config = HLSConfig(PauliEvolution=[HopEvolutionSynthesis(), "default"])

    out = _transpile(circ, config, optimization_level=0, basis_gates=["xx_plus_yy", "u3"])

    assert dict(out.count_ops()) == {"xx_plus_yy": 1}


def test_plugin_declines_a_parameterized_time():
    observable = _hop_observable(0.7, 0.2, [0, 1], 2)
    circ = QuantumCircuit(2)
    circ.append(PauliEvolutionGate(observable, time=Parameter("t")), [0, 1])

    plugin = HopEvolutionSynthesis()
    assert plugin.run(circ.data[0].operation) is None


def test_plugin_accepts_a_sparse_pauli_op():
    op = SparsePauliOp.from_sparse_observable(_hop_observable(0.7, 0.2, [0, 1], 2))
    circ = QuantumCircuit(2)
    circ.append(PauliEvolutionGate(op, time=0.5), [0, 1])
    config = HLSConfig(PauliEvolution=[HopEvolutionSynthesis(), "default"])

    out = _transpile(circ, config, optimization_level=0, basis_gates=["xx_plus_yy", "u3"])

    assert dict(out.count_ops()) == {"xx_plus_yy": 1}


def test_plugin_declines_other_operations():
    assert HopEvolutionSynthesis().run(XXPlusYYGate(0.3)) is None


def _hop_circuit(num_modes, pairs, time):
    circ = FermionicCircuit(num_modes)
    for i, j in pairs:
        hop = FermionOperator.from_dict(
            {((True, i), (False, j)): 0.8, ((True, j), (False, i)): 0.8}
        )
        circ.append(Evolution(num_modes, hop, time=time), circ.modes)
    return circ


def test_preset_emits_xx_plus_yy_for_a_hop():
    circ = _hop_circuit(2, [(0, 1)], 0.6)

    out = generate_preset_jw_pass_manager(
        optimization_level=0, basis_gates=["xx_plus_yy", "u3"]
    ).run(circ)

    assert dict(out.count_ops()) == {"xx_plus_yy": 1}
    hop = jordan_wigner(
        FermionOperator.from_dict({((True, 0), (False, 1)): 0.8, ((True, 1), (False, 0)): 0.8}), 2
    )
    np.testing.assert_allclose(Operator(out).data, _evolution_matrix(hop, 0.6), atol=1e-12)


def test_preset_keeps_a_hop_across_a_mode_generic():
    circ = _hop_circuit(3, [(0, 2)], 0.6)

    out = generate_preset_jw_pass_manager(optimization_level=0, basis_gates=["cx", "u3"]).run(circ)

    assert "xx_plus_yy" not in out.count_ops()
    hop = jordan_wigner(
        FermionOperator.from_dict({((True, 0), (False, 2)): 0.8, ((True, 2), (False, 0)): 0.8}), 3
    )
    np.testing.assert_allclose(Operator(out).data, _evolution_matrix(hop, 0.6), atol=1e-12)


def test_preset_forces_the_hop_synthesis_over_a_user_hls_config():
    """The hop synthesis is unconditional: a listed ``"default"`` does not pre-empt it.

    The configuration given to the preset is extended and left unmodified.
    """
    circ = _hop_circuit(2, [(0, 1)], 0.6)
    config = HLSConfig(PauliEvolution=["default"], use_default_on_unspecified=False)

    out = generate_preset_jw_pass_manager(
        optimization_level=0, basis_gates=["xx_plus_yy", "u3"], hls_config=config
    ).run(circ)

    assert dict(out.count_ops()) == {"xx_plus_yy": 1}
    assert config.methods == {"PauliEvolution": ["default"]}


def test_preset_keeps_the_methods_of_a_user_hls_config():
    """What the user lists for a Pauli evolution still applies to whatever is not a hop."""
    circ = _hop_circuit(3, [(0, 2)], 0.6)
    config = HLSConfig(PauliEvolution=[("default", {})])

    out = generate_preset_jw_pass_manager(
        optimization_level=0, basis_gates=["cx", "u3"], hls_config=config
    ).run(circ)

    assert "xx_plus_yy" not in out.count_ops()
    assert out.count_ops()["cx"] > 0
