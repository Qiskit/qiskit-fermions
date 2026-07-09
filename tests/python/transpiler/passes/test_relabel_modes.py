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

"""Mode relabeling optimization tests."""

from __future__ import annotations

import pytest
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.passmanager import MultiStagePassManager
from qiskit_fermions.circuit import FermionicCircuit
from qiskit_fermions.circuit.library import Evolution, InitializeModes
from qiskit_fermions.mappers.library import jordan_wigner
from qiskit_fermions.operators import FermionOperator
from qiskit_fermions.transpiler import FermionicCircuitToDAG, QuantumDAGToCircuit
from qiskit_fermions.transpiler.passes import (
    F2QSynthesis,
    MapperFnEvolutionSynthesis,
    RelabelModes,
    TrivialF2QLayout,
    TrivialOccupationInitializeModesSynthesis,
)
from qiskit_fermions.utils.optionals import HAS_PYOMO

if HAS_PYOMO:
    from pyomo.environ import SolverFactory


def test_relabel_modes_fixed_permutation():
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

    init = InitializeModes([1, 0, 1, 0])
    circ.append(init, circ.modes)

    evo = Evolution(num_modes, hamil, time=time)
    circ.append(evo, circ.modes)

    synth = F2QSynthesis()
    synth.methods["Evolution"] = MapperFnEvolutionSynthesis(jordan_wigner)
    synth.methods["InitializeModes"] = TrivialOccupationInitializeModesSynthesis()

    permutation = [0, 2, 1, 3]
    relabel = RelabelModes(permutation)

    pm = MultiStagePassManager(
        input=FermionicCircuitToDAG(),
        optimization=relabel,
        layout=TrivialF2QLayout(),
        synthesis=synth,
        output=QuantumDAGToCircuit(),
    )

    qu_circ = pm.run(circ)

    gates = qu_circ.data
    assert len(gates) == 3
    assert isinstance(gates[-1].operation, PauliEvolutionGate)

    qu_circ_decomp = qu_circ.decompose(reps=2)
    assert qu_circ_decomp.depth(lambda instr: len(instr.qubits) == 2) == 4


def test_relabel_modes_local_indices():
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
    circ = FermionicCircuit(2 * num_modes)

    init = InitializeModes([1, 0, 1, 0])
    circ.append(init, circ.modes[:num_modes])

    evo = Evolution(num_modes, hamil, time=time)
    circ.append(evo, circ.modes[:num_modes])

    synth = F2QSynthesis()
    synth.methods["Evolution"] = MapperFnEvolutionSynthesis(jordan_wigner)
    synth.methods["InitializeModes"] = TrivialOccupationInitializeModesSynthesis()

    permutation = [0, 2, 1, 3, 4, 5, 6, 7]
    relabel = RelabelModes(permutation)

    pm = MultiStagePassManager(
        input=FermionicCircuitToDAG(),
        optimization=relabel,
        layout=TrivialF2QLayout(),
        synthesis=synth,
        output=QuantumDAGToCircuit(),
    )

    qu_circ = pm.run(circ)

    qu_circ_decomp = qu_circ.decompose(gates_to_decompose=["PauliEvolution"])
    assert qu_circ_decomp.count_ops() == {"x": 2, "rxx": 2, "ryy": 2}


@pytest.mark.skipif(not HAS_PYOMO, reason="Pyomo is required")
def test_relabel_modes_pyomo_optimization():
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

    synth = F2QSynthesis()
    synth.methods["Evolution"] = MapperFnEvolutionSynthesis(jordan_wigner)

    solver = SolverFactory("appsi_highs")
    solver.options["time_limit"] = 60
    relabel = RelabelModes(solver=solver)

    pm = MultiStagePassManager(
        input=FermionicCircuitToDAG(),
        optimization=relabel,
        layout=TrivialF2QLayout(),
        synthesis=synth,
        output=QuantumDAGToCircuit(),
    )

    qu_circ = pm.run(circ)

    gates = qu_circ.data
    assert len(gates) == 1
    assert isinstance(gates[0].operation, PauliEvolutionGate)

    qu_circ_decomp = qu_circ.decompose(reps=2)
    assert qu_circ_decomp.depth(lambda instr: len(instr.qubits) == 2) == 4
