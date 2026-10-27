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

"""Tests for the plugin framework of the fermion-to-fermion synthesis pass."""

from __future__ import annotations

import pytest
from qiskit_fermions.circuit import FermionicCircuit
from qiskit_fermions.circuit.library import Evolution, InitializeModes
from qiskit_fermions.circuit.library.synthesis import (
    F2FSynthesisPluginManager,
    FermionicLieTrotter,
    FermionicSuzukiTrotter,
)
from qiskit_fermions.operators import FermionOperator
from qiskit_fermions.transpiler import FermionicCircuitToDAG, FermionicPassManager
from qiskit_fermions.transpiler.passes import F2FSynthesis

NUM_MODES = 3


def _operator() -> FermionOperator:
    return FermionOperator.from_terms_with_groups(
        [
            (((True, 0), (False, 1)), -1.0, 0),
            (((True, 1), (False, 0)), -1.0, 0),
            (((True, 1), (False, 2)), -0.8, 1),
            (((True, 2), (False, 1)), -0.8, 1),
        ]
    )


def _circuit(*extra_gates) -> FermionicCircuit:
    circuit = FermionicCircuit(NUM_MODES)
    for gate in extra_gates:
        circuit.append(gate, circuit.modes)
    circuit.append(Evolution(NUM_MODES, _operator(), time=0.5), circuit.modes)
    return circuit


class _Marker:
    """A plugin that is registered nowhere, and counts the gates it is given."""

    def __init__(self):
        self.seen = []

    def synthesize(self, gate):
        self.seen.append(gate)
        return FermionicCircuit(gate.num_modes)


def test_the_builtin_plugins_are_installed():
    manager = F2FSynthesisPluginManager()

    assert {"LieTrotter", "SuzukiTrotter"} <= set(manager.method_names("Evolution"))
    assert manager.method("Evolution", "LieTrotter") is FermionicLieTrotter
    assert manager.method("Evolution", "SuzukiTrotter") is FermionicSuzukiTrotter
    assert "Evolution" in manager.op_names()


def test_an_unknown_gate_has_no_methods():
    assert F2FSynthesisPluginManager().method_names("NotAGate") == []


def test_config_by_name():
    synth = F2FSynthesis({"Evolution": "LieTrotter"})

    assert isinstance(synth.methods["Evolution"], FermionicLieTrotter)


def test_config_by_name_and_positional_arguments():
    synth = F2FSynthesis({"Evolution": ("LieTrotter", (3,))})

    assert synth.methods["Evolution"].reps == 3


def test_config_by_name_arguments_and_keyword_arguments():
    synth = F2FSynthesis({"Evolution": ("SuzukiTrotter", (), {"order": 4, "reps": 2})})

    method = synth.methods["Evolution"]
    assert isinstance(method, FermionicSuzukiTrotter)
    assert (method.order, method.reps) == (4, 2)


def test_config_by_instance_needs_no_entry_point():
    marker = _Marker()

    synth = F2FSynthesis({"Evolution": marker})

    assert synth.methods["Evolution"] is marker


def test_methods_can_be_populated_by_hand():
    synth = F2FSynthesis()
    assert synth.methods == {}

    synth.methods["Evolution"] = FermionicLieTrotter()

    out = FermionicPassManager(synth).run(_circuit())
    assert out.count_ops()["Evolution"] > 1


def test_an_empty_pass_changes_nothing():
    out = FermionicPassManager(F2FSynthesis()).run(_circuit())

    assert dict(out.count_ops()) == {"Evolution": 1}


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"Evolution": "NoSuchMethod"}, "No synthesis method 'NoSuchMethod'"),
        ({"Evolution": ("NoSuchMethod", ())}, "No synthesis method 'NoSuchMethod'"),
        ({"NotAGate": "Anything"}, "No fermion-to-fermion synthesis is installed for 'NotAGate'"),
    ],
)
def test_an_unknown_gate_or_method_is_rejected_when_the_pass_is_built(config, message):
    with pytest.raises(KeyError, match=message):
        F2FSynthesis(config)


def test_a_gate_without_a_synthesis_cannot_be_configured():
    """``InitializeModes`` breaks down into nothing, so naming a method for it is a mistake."""
    with pytest.raises(KeyError, match="No fermion-to-fermion synthesis is installed"):
        F2FSynthesis({"InitializeModes": "Anything"})


@pytest.mark.parametrize("value", [42, None, ("LieTrotter",), ["LieTrotter"], object()])
def test_a_value_of_no_known_form_is_rejected(value):
    with pytest.raises(TypeError, match="must be a method name"):
        F2FSynthesis({"Evolution": value})  # type: ignore[dict-item]


def test_a_gate_without_a_synthesis_passes_through_whatever_else_is_configured():
    marker = _Marker()
    circuit = _circuit(InitializeModes([1, 0, 1]))

    out = FermionicPassManager(F2FSynthesis({"Evolution": marker})).run(circuit)

    assert [type(g).__name__ for g in marker.seen] == ["Evolution"]
    assert dict(out.count_ops()) == {"InitializeModes": 1}


def test_only_the_configured_gate_types_are_touched():
    marker = _Marker()

    FermionicPassManager(F2FSynthesis({"InitializeModes": marker})).run(_circuit())

    assert marker.seen == []


def test_the_plugin_is_matched_by_gate_type_name():
    first, second = _Marker(), _Marker()
    synth = F2FSynthesis({"Evolution": first})

    synth.methods["Evolution"] = second
    FermionicPassManager(synth).run(_circuit())

    assert first.seen == []
    assert len(second.seen) == 1


def test_a_registered_name_and_an_instance_behave_the_same():
    by_name = FermionicPassManager(F2FSynthesis({"Evolution": ("SuzukiTrotter", (), {"order": 2})}))
    by_instance = FermionicPassManager(F2FSynthesis({"Evolution": FermionicSuzukiTrotter(order=2)}))

    a, b = by_name.run(_circuit()), by_instance.run(_circuit())

    assert dict(a.count_ops()) == dict(b.count_ops())


def test_the_pass_runs_on_a_dag_directly():
    dag = FermionicCircuitToDAG().run(_circuit())

    out = F2FSynthesis({"Evolution": FermionicLieTrotter()}).run(dag)

    assert len(out.op_nodes()) == 2
