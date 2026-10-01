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

"""Tests that measurements added on a fermionic circuit survive the qubit-level transpilation."""

from __future__ import annotations

from qiskit.providers.basic_provider import BasicSimulator
from qiskit.transpiler import CouplingMap
from qiskit_fermions.circuit import FermionicCircuit
from qiskit_fermions.circuit.library import Evolution, InitializeModes
from qiskit_fermions.operators import FermionOperator, ann, cre
from qiskit_fermions.transpiler.presets import generate_preset_jw_pass_manager


def test_measurements_follow_the_qubits_through_layout_and_routing():
    """Bitstrings come back indexed by fermionic mode even when the qubit stage permutes the qubits.

    This is the reason to add measurements on the *fermionic* circuit rather than on the transpiled
    one: Qiskit rewires a measurement's qubit along with the qubit it follows, so the layout and
    routing permutations are absorbed. Measuring after transpilation instead leaves the caller to undo
    :meth:`~qiskit.transpiler.TranspileLayout.final_index_layout` by hand, and silently returns
    mode-permuted results if they forget.

    The evolution couples the two ends of a linear coupling map, which forces a non-trivial layout;
    its operator is diagonal (a product of number operators), so the state stays a computational-basis
    state and the expected bitstring is exactly the initial occupation. The layout assertion guards the
    test itself -- with a trivial layout it would hold no matter how measurements were handled.
    """
    occupation = [1, 1, 0, 1, 0]
    num_modes = len(occupation)

    hamiltonian = FermionOperator.from_dict({(cre(0), ann(0), cre(4), ann(4)): 1.0})
    circuit = FermionicCircuit(num_modes)
    circuit.append(InitializeModes(occupation), circuit.modes)
    circuit.append(Evolution(num_modes, hamiltonian, time=0.7), circuit.modes)
    circuit.measure_all()

    pass_manager = generate_preset_jw_pass_manager(
        optimization_level=1,
        coupling_map=CouplingMap.from_line(num_modes),
        basis_gates=["cx", "u", "measure"],
        seed_transpiler=7,
    )
    transpiled = pass_manager.run(circuit)

    # guard: if the qubit stage left the qubits in place, this test proves nothing
    assert transpiled.layout.final_index_layout() != list(range(num_modes))

    (bitstring,) = BasicSimulator().run(transpiled, shots=1).result().get_counts()
    # clbit `i` holds mode `i`; Qiskit prints clbit 0 rightmost, hence the negated index
    assert [int(bitstring[~mode]) for mode in range(num_modes)] == occupation
