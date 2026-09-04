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

"""Evolution synthesis tests."""

from __future__ import annotations

from qiskit.circuit.library import PauliEvolutionGate
from qiskit.passmanager import MultiStagePassManager
from qiskit.quantum_info import SparseObservable, SparsePauliOp
from qiskit.synthesis import LieTrotter, SuzukiTrotter
from qiskit_fermions.circuit import FermionicCircuit
from qiskit_fermions.circuit.library import Evolution
from qiskit_fermions.mappers.library import jordan_wigner
from qiskit_fermions.operators import FermionOperator
from qiskit_fermions.transpiler import FermionicCircuitToDAG, QuantumDAGToCircuit
from qiskit_fermions.transpiler.passes import (
    F2QSynthesis,
    MapperFnEvolutionSynthesis,
    TrivialF2QLayout,
    group_wise,
    simplify,
)


def test_evolution_gate_synthesis():
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

    pm = MultiStagePassManager(
        input=FermionicCircuitToDAG(),
        layout=TrivialF2QLayout(),
        synthesis=synth,
        output=QuantumDAGToCircuit(),
    )

    qu_circ = pm.run(circ)

    gates = qu_circ.data
    assert len(gates) == 1
    assert isinstance(gates[0].operation, PauliEvolutionGate)

    qu_circ_decomp = qu_circ.decompose(reps=2)
    assert qu_circ_decomp.depth(lambda instr: len(instr.qubits) == 2) == 16


def test_custom_qubit_ordering():
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

    custom_qubit_ordering = [0, 2, 1, 3]

    def custom_qubit_ordering_mapper_fn(op, num_qubits):
        relabeled = op.relabel_modes(custom_qubit_ordering)
        return jordan_wigner(relabeled, num_qubits)

    synth = F2QSynthesis()
    synth.methods["Evolution"] = MapperFnEvolutionSynthesis(custom_qubit_ordering_mapper_fn)

    pm = MultiStagePassManager(
        input=FermionicCircuitToDAG(),
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


def _single_evolution_circuit(time: float = 1.5) -> FermionicCircuit:
    hamil = FermionOperator.from_dict(
        {
            ((True, 0), (False, 2)): 2.0,
            ((True, 2), (False, 0)): 2.0,
            ((True, 1), (False, 3)): -2.0,
            ((True, 3), (False, 1)): -2.0,
        }
    )
    num_modes = 4
    circ = FermionicCircuit(num_modes)
    circ.append(Evolution(num_modes, hamil, time=time), circ.modes)
    return circ


def _run_with_product_formula(product_formula) -> object:
    synth = F2QSynthesis()
    synth.methods["Evolution"] = MapperFnEvolutionSynthesis(jordan_wigner, product_formula)
    pm = MultiStagePassManager(
        input=FermionicCircuitToDAG(),
        layout=TrivialF2QLayout(),
        synthesis=synth,
        output=QuantumDAGToCircuit(),
    )
    return pm.run(_single_evolution_circuit())


def test_default_product_formula_is_none():
    """Omitting ``product_formula`` leaves the emitted gate on its default synthesis."""
    plugin = MapperFnEvolutionSynthesis(jordan_wigner)
    assert plugin.product_formula is None

    qu_circ = _run_with_product_formula(None)
    (instruction,) = qu_circ.data
    assert isinstance(instruction.operation, PauliEvolutionGate)
    # a PauliEvolutionGate with no explicit synthesis falls back to a single-rep LieTrotter
    assert isinstance(instruction.operation.synthesis, LieTrotter)
    assert instruction.operation.synthesis.reps == 1


def test_product_formula_is_forwarded_to_pauli_evolution_gate():
    """An explicit product formula is attached to the emitted PauliEvolutionGate verbatim."""
    product_formula = SuzukiTrotter(order=2, reps=3)
    qu_circ = _run_with_product_formula(product_formula)
    (instruction,) = qu_circ.data
    assert isinstance(instruction.operation, PauliEvolutionGate)
    assert instruction.operation.synthesis is product_formula


def test_higher_order_product_formula_deepens_synthesis():
    """A higher-order / repeated product formula yields a deeper decomposition."""
    default_circ = _run_with_product_formula(None).decompose(reps=2)
    suzuki_circ = _run_with_product_formula(SuzukiTrotter(order=2, reps=3)).decompose(reps=2)

    two_qubit = lambda instr: len(instr.qubits) == 2  # noqa: E731
    assert suzuki_circ.depth(two_qubit) > default_circ.depth(two_qubit)


def test_product_formula_via_config_kwargs():
    """The product formula flows through the ``F2QSynthesisConfig`` keyword-argument form."""
    product_formula = SuzukiTrotter(order=2, reps=2)
    config = {"Evolution": ("MapperFn", (jordan_wigner,), {"product_formula": product_formula})}
    synth = F2QSynthesis(config)
    assert synth.methods["Evolution"].product_formula is product_formula

    pm = MultiStagePassManager(
        input=FermionicCircuitToDAG(),
        layout=TrivialF2QLayout(),
        synthesis=synth,
        output=QuantumDAGToCircuit(),
    )
    qu_circ = pm.run(_single_evolution_circuit())
    (instruction,) = qu_circ.data
    assert instruction.operation.synthesis is product_formula


def _flow_set_hamiltonian(num_modes: int = 6) -> FermionOperator:
    """A hopping chain grouped into even bonds, odd bonds and the on-site interaction.

    The bonds of one group are vertex-disjoint, so the Pauli terms a group maps to act on disjoint
    qubits and can be scheduled in parallel -- but only if they arrive adjacent to one another.
    """
    terms = []
    for site in range(num_modes - 1):
        group = site % 2
        terms.append((((True, site), (False, site + 1)), -1.0, group))
        terms.append((((True, site + 1), (False, site)), -1.0, group))
    for site in range(0, num_modes - 1, 2):
        terms.append((((True, site), (False, site), (True, site + 1), (False, site + 1)), 2.0, 2))
    return FermionOperator.from_terms_with_groups(terms)


def _synthesize(hamil: FermionOperator, mapper_fn) -> object:
    num_modes = len(hamil.get_support())
    circ = FermionicCircuit(num_modes)
    circ.append(Evolution(num_modes, hamil, time=1.0), circ.modes)

    synth = F2QSynthesis()
    synth.methods["Evolution"] = MapperFnEvolutionSynthesis(mapper_fn)
    pm = MultiStagePassManager(
        input=FermionicCircuitToDAG(),
        layout=TrivialF2QLayout(),
        synthesis=synth,
        output=QuantumDAGToCircuit(),
    )
    return pm.run(circ)


def _mapped_operator(hamil: FermionOperator, mapper_fn) -> SparseObservable:
    """Returns the qubit operator the plugin hands to the emitted gate."""
    (instruction,) = _synthesize(hamil, mapper_fn).data
    return instruction.operation.operator


def test_mapper_output_is_not_simplified():
    """The plugin passes the mapped operator through untouched, term order included."""
    hamil = _flow_set_hamiltonian()
    num_qubits = len(hamil.get_support())

    emitted = _mapped_operator(hamil, group_wise(jordan_wigner))
    expected = group_wise(jordan_wigner)(hamil, num_qubits)

    labels = lambda obs: [(label, tuple(idx)) for label, idx, _ in obs.to_sparse_list()]  # noqa: E731
    assert labels(emitted) == labels(expected)
    # ... and that order is *not* the canonical one that simplifying would impose
    assert labels(emitted) != labels(expected.simplify())


def test_simplify_yields_a_canonical_term_order():
    """``simplify`` pins the term order, which an unwrapped mapper does not guarantee."""
    hamil = _flow_set_hamiltonian()
    num_qubits = len(hamil.get_support())

    labels = lambda obs: [(label, tuple(idx)) for label, idx, _ in obs.to_sparse_list()]  # noqa: E731
    wrapped = simplify(jordan_wigner)
    assert labels(wrapped(hamil, num_qubits)) == labels(jordan_wigner(hamil, num_qubits).simplify())


def test_simplify_preserves_the_operator():
    """Simplifying merges and reorders terms but must not change what the operator *is*."""
    hamil = _flow_set_hamiltonian()
    num_qubits = len(hamil.get_support())

    plain = SparsePauliOp.from_sparse_observable(jordan_wigner(hamil, num_qubits))
    wrapped = SparsePauliOp.from_sparse_observable(simplify(jordan_wigner)(hamil, num_qubits))

    assert plain.simplify().equiv(wrapped.simplify())


def test_group_wise_preserves_the_operator():
    """Mapping group by group and summing must reproduce the monolithic mapping exactly."""
    hamil = _flow_set_hamiltonian()
    num_qubits = len(hamil.get_support())

    monolithic = SparsePauliOp.from_sparse_observable(jordan_wigner(hamil, num_qubits))
    grouped = SparsePauliOp.from_sparse_observable(group_wise(jordan_wigner)(hamil, num_qubits))

    assert monolithic.simplify().equiv(grouped.simplify())


def test_group_wise_emits_each_group_contiguously():
    """The point of the wrapper: a group's terms arrive together rather than interleaved."""
    hamil = _flow_set_hamiltonian()
    num_qubits = len(hamil.get_support())

    per_group = [
        {(label, tuple(idx)) for label, idx, _ in jordan_wigner(group, num_qubits).to_sparse_list()}
        for group in hamil.split_out_groups()
    ]
    emitted = [
        (label, tuple(idx))
        for label, idx, _ in group_wise(jordan_wigner)(hamil, num_qubits).to_sparse_list()
    ]

    # walking the emitted terms, the group index may only ever move forward
    seen: list[int] = []
    for term in emitted:
        owners = [index for index, terms in enumerate(per_group) if term in terms]
        assert owners, f"emitted term {term} belongs to no group"
        if not seen or owners[0] != seen[-1]:
            seen.append(owners[0])
    assert seen == sorted(set(seen)), f"groups are interleaved, not contiguous: {seen}"


def test_group_wise_reduces_two_qubit_depth():
    """Contiguous groups let disjoint rotations share a layer, at an unchanged gate count."""
    hamil = _flow_set_hamiltonian()

    plain = _synthesize(hamil, simplify(jordan_wigner)).decompose(reps=6)
    grouped = _synthesize(hamil, group_wise(jordan_wigner)).decompose(reps=6)

    two_qubit = lambda instr: len(instr.qubits) == 2  # noqa: E731
    assert grouped.depth(two_qubit) < plain.depth(two_qubit)
    assert grouped.count_ops()["cx"] == plain.count_ops()["cx"]


def test_group_wise_passes_an_ungrouped_operator_through():
    """Without groups there is nothing to split, so the wrapper must be a no-op."""
    hamil = FermionOperator.from_dict({((True, 0), (False, 1)): 1.0, ((True, 1), (False, 0)): 1.0})
    assert not hamil.has_groups()

    labels = lambda obs: [(label, tuple(idx)) for label, idx, _ in obs.to_sparse_list()]  # noqa: E731
    assert labels(group_wise(jordan_wigner)(hamil, 2)) == labels(jordan_wigner(hamil, 2))


def test_group_wise_handles_an_empty_grouped_operator():
    """A grouped operator holding no terms maps to the zero observable rather than raising."""
    hamil = FermionOperator.zero()
    hamil.groups = []
    assert hamil.has_groups()

    assert group_wise(jordan_wigner)(hamil, 2) == SparseObservable.zero(2)


def test_adapters_compose_with_a_product_formula():
    """The adapters are plain MapperFunctions, so they coexist with ``product_formula``."""
    product_formula = SuzukiTrotter(order=2, reps=2)
    synth = F2QSynthesis()
    synth.methods["Evolution"] = MapperFnEvolutionSynthesis(
        group_wise(jordan_wigner), product_formula
    )
    pm = MultiStagePassManager(
        input=FermionicCircuitToDAG(),
        layout=TrivialF2QLayout(),
        synthesis=synth,
        output=QuantumDAGToCircuit(),
    )
    (instruction,) = pm.run(_single_evolution_circuit()).data
    assert instruction.operation.synthesis is product_formula
