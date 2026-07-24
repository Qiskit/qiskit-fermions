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

"""Fermion-to-qubit synthesis tests."""

from __future__ import annotations

import pytest
from qiskit import ClassicalRegister
from qiskit.circuit.library import XGate
from qiskit.passmanager import MultiStagePassManager
from qiskit_fermions.circuit import FermionicCircuit
from qiskit_fermions.circuit.library import Evolution
from qiskit_fermions.mappers.library import jordan_wigner
from qiskit_fermions.operators import FermionOperator
from qiskit_fermions.transpiler import FermionicCircuitToDAG, QuantumDAGToCircuit
from qiskit_fermions.transpiler.passes import (
    F2QSynthesis,
    MapperFnEvolutionSynthesis,
    TrivialF2QLayout,
)


def test_missing_plugin():
    """Test the handling of a missing fermion-to-qubit plugin."""
    hamil = FermionOperator.from_dict(
        {
            ((True, 0), (False, 2)): 2.0,
            ((True, 2), (False, 0)): 2.0,
            ((True, 1), (False, 3)): -2.0,
            ((True, 3), (False, 1)): -2.0,
        }
    )
    time = 1.5
    num_modes = 4
    circ = FermionicCircuit(num_modes)
    evo = Evolution(num_modes, hamil, time=time)
    circ.append(evo, circ.modes)

    pm = MultiStagePassManager(
        input=FermionicCircuitToDAG(),
        layout=TrivialF2QLayout(),
        synthesis=F2QSynthesis(),
        output=QuantumDAGToCircuit(),
    )

    with pytest.raises(TypeError, match="No plugin registered"):
        _ = pm.run(circ)


def test_config_tuple_with_positional_args():
    """The ``(name, args)`` config form forwards positional args to the plugin ``__init__``."""
    synth = F2QSynthesis({"Evolution": ("MapperFn", (jordan_wigner,))})
    assert isinstance(synth.methods["Evolution"], MapperFnEvolutionSynthesis)


def test_config_tuple_with_positional_and_keyword_args():
    """The ``(name, args, kwargs)`` config form forwards both positional and keyword args."""
    synth = F2QSynthesis({"Evolution": ("MapperFn", (), {"mapper_fn": jordan_wigner})})
    assert isinstance(synth.methods["Evolution"], MapperFnEvolutionSynthesis)


def test_run_rejects_non_fermionic_gate():
    """A circuit instruction that is not a FermionicGate raises during synthesis."""
    num_modes = 2
    circ = FermionicCircuit(num_modes)
    # Append a plain (non-fermionic) gate directly on the mode register.
    circ._inner.append(XGate(), [circ.register[0]])

    synth = F2QSynthesis()
    synth.methods["Evolution"] = MapperFnEvolutionSynthesis(jordan_wigner)

    pm = MultiStagePassManager(
        input=FermionicCircuitToDAG(),
        layout=TrivialF2QLayout(),
        synthesis=synth,
        output=QuantumDAGToCircuit(),
    )

    with pytest.raises(ValueError, match="unsupported circuit instruction type"):
        _ = pm.run(circ)


def test_classical_registers_are_preserved():
    """Classical registers on the input circuit are carried over to the synthesized output."""
    hamil = FermionOperator.from_dict({((True, 0), (False, 1)): 1.0, ((True, 1), (False, 0)): 1.0})
    num_modes = 2
    circ = FermionicCircuit(num_modes)
    circ.append(Evolution(num_modes, hamil, time=0.5), circ.modes)
    creg = ClassicalRegister(2, "c")
    circ._inner.add_register(creg)

    synth = F2QSynthesis()
    synth.methods["Evolution"] = MapperFnEvolutionSynthesis(jordan_wigner)

    pm = MultiStagePassManager(
        input=FermionicCircuitToDAG(),
        layout=TrivialF2QLayout(),
        synthesis=synth,
        output=QuantumDAGToCircuit(),
    )

    qu_circ = pm.run(circ)
    assert any(reg.name == "c" and len(reg) == 2 for reg in qu_circ.cregs)
