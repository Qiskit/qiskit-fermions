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

"""Tests of the commuting-diagonalization synthesis plugin and its selectors."""

from __future__ import annotations

import itertools

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import SparseObservable, SparsePauliOp
from qiskit.synthesis import LieTrotter
from qiskit.synthesis.evolution import EvolutionSynthesis
from qiskit.transpiler import CouplingMap, generate_preset_pass_manager
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes.synthesis.high_level_synthesis import HighLevelSynthesis, HLSConfig
from qiskit_fermions.transpiler.passes import CommutingDiagonalizationSynthesis
from qiskit_fermions.transpiler.passes.synthesis.commuting_evolution import (
    DEFAULT_CANDIDATES,
    Context,
    Local,
    MinCost,
    Plain,
    Problem,
    Report,
    Variant,
    abstract_cost,
)

from .groups import GROUPS, non_commuting, plaquettes, same_unitary, zxz_windows

TIME = 0.37


def _gate(op, time=TIME) -> PauliEvolutionGate:
    return PauliEvolutionGate(op, time=time)


def _run(plugin, op, *, line: bool = False, staged: bool = False):
    """Runs the plugin on a gate; ``staged`` passes the physical qubits as the layout stage
    would."""
    kwargs = {}
    if line:
        kwargs["coupling_map"] = CouplingMap.from_line(op.num_qubits)
    if staged:
        kwargs["qubits"] = list(range(op.num_qubits))
    return plugin.run(_gate(op), **kwargs)


def _recording(**kwargs) -> tuple[CommutingDiagonalizationSynthesis, list[Report]]:
    """A plugin whose reports are collected in the returned list."""
    reports: list[Report] = []
    return CommutingDiagonalizationSynthesis(callback=reports.append, **kwargs), reports


class _Fixed:
    """A user candidate that offers one fixed circuit under its own name."""

    def __init__(self, name: str, circuit: QuantumCircuit | None = None):
        self.name = name
        self.circuit = circuit

    def generate(self, problem, context):
        if self.circuit is not None:
            return {"": self.circuit}
        circuit = QuantumCircuit(problem.num_qubits)
        circuit.append(PauliEvolutionGate(problem.operator, problem.time), circuit.qubits)
        return {"": circuit.decompose()}


class TestDeclines:
    def test_non_commuting_terms(self):
        plugin, reports = _recording()
        assert plugin.run(_gate(non_commuting())) is None
        assert [r.declined for r in reports] == ["terms do not commute"]

    def test_complex_coefficients(self):
        """Qiskit refuses to build such a gate, but its operator can be replaced afterwards."""
        gate = _gate(zxz_windows(3))
        gate.operator = SparsePauliOp.from_sparse_list(
            [("XX", [0, 1], 1.0 + 0.5j), ("ZZ", [0, 1], 0.3)], num_qubits=3
        )
        plugin, reports = _recording()
        assert plugin.run(gate) is None
        assert [r.declined for r in reports] == ["complex coefficients"]

    def test_not_a_pauli_evolution(self):
        plugin, reports = _recording()
        assert plugin.run(QuantumCircuit(2).to_gate()) is None
        assert reports == []  # not its business, so not reported

    def test_non_real_time(self):
        from qiskit.circuit import Parameter

        plugin, reports = _recording()
        assert plugin.run(PauliEvolutionGate(zxz_windows(4), time=Parameter("t"))) is None
        assert reports == []

    def test_unroutable_coupling_map(self):
        """A scoring transpile that cannot route the gate declines it instead of failing the whole
        pass."""
        op = SparsePauliOp(["XXXX", "ZZZZ", "YYYY"], [0.3, 0.5, 0.7])
        plugin, reports = _recording()
        assert plugin.run(_gate(op), coupling_map=CouplingMap([(0, 1), (2, 3)])) is None
        assert reports[0].variants and reports[0].chosen is None and reports[0].declined

    def test_no_candidate_offers_a_circuit(self):
        class Nothing:
            name = "nothing"

            def generate(self, problem, context):
                return {}

        plugin, reports = _recording(candidates=[Nothing()])
        assert _run(plugin, zxz_windows(5)) is None
        assert reports[0].variants == () and reports[0].declined

    def test_declined_gate_falls_through_to_the_default_method(self):
        """In an ``HLSConfig`` the next method synthesizes what the plugin declines."""
        qc = QuantumCircuit(2)
        qc.append(_gate(non_commuting()), [0, 1])
        config = HLSConfig(PauliEvolution=[CommutingDiagonalizationSynthesis(), "default"])
        out = HighLevelSynthesis(hls_config=config)(qc)
        assert not any(i.operation.name == "PauliEvolution" for i in out.data)


class TestExactness:
    @pytest.mark.parametrize("name", sorted(GROUPS))
    def test_default_plugin_is_exact(self, name):
        op = GROUPS[name]()
        circuit = _run(CommutingDiagonalizationSynthesis(), op)
        assert circuit is not None
        assert same_unitary(circuit, op, TIME)

    @pytest.mark.parametrize("staged", [False, True])
    @pytest.mark.parametrize("line", [False, True])
    @pytest.mark.parametrize("selector", [MinCost(), MinCost(abstract_cost)], ids=str)
    def test_every_configuration_is_exact(self, selector, line, staged):
        plugin = CommutingDiagonalizationSynthesis(selector=selector)
        op = plaquettes(2)
        circuit = _run(plugin, op, line=line or staged, staged=staged)
        assert circuit is not None
        assert same_unitary(circuit, op, TIME)

    def test_accepts_sparse_observables(self):
        op = zxz_windows(5)
        observable = SparseObservable.from_sparse_list(
            [(label, indices, coeff.real) for label, indices, coeff in op.to_sparse_list()],
            num_qubits=5,
        )
        circuit = CommutingDiagonalizationSynthesis().run(PauliEvolutionGate(observable, time=TIME))
        assert circuit is not None
        assert same_unitary(circuit, op, TIME)

    def test_negative_time(self):
        op = zxz_windows(5)
        circuit = CommutingDiagonalizationSynthesis().run(_gate(op, -0.8))
        assert circuit is not None
        assert same_unitary(circuit, op, -0.8)

    def test_unsimplified_operator(self):
        """Repeated terms are combined before synthesis."""
        op = SparsePauliOp.from_sparse_list(
            [("ZZ", [0, 1], 0.3), ("ZZ", [0, 1], 0.2), ("XX", [0, 1], 0.4), ("II", [0, 1], 0.5)],
            num_qubits=2,
        )
        circuit = CommutingDiagonalizationSynthesis().run(_gate(op))
        assert circuit is not None
        assert same_unitary(circuit, op, TIME)

    def test_gate_with_a_non_product_formula_synthesis_object(self):
        """The plugin reads the operator, not the gate's own synthesis, so it handles any
        ``EvolutionSynthesis``."""

        class Custom(EvolutionSynthesis):
            def synthesize(self, evolution):
                return LieTrotter().synthesize(evolution)

            @property
            def settings(self):
                return {}

        op = zxz_windows(5)
        circuit = CommutingDiagonalizationSynthesis().run(
            PauliEvolutionGate(op, time=TIME, synthesis=Custom())
        )
        assert circuit is not None
        assert same_unitary(circuit, op, TIME)


class TestCandidates:
    def test_only_the_given_candidates_are_asked(self):
        plugin, reports = _recording(candidates=[Local()])
        assert _run(plugin, zxz_windows(5)) is not None
        assert {v.candidate for v in reports[0].variants} == {"local"}

    def test_search_runs_by_default_only_for_a_gate_that_is_not_placed(self):
        plugin, reports = _recording()
        _run(plugin, zxz_windows(6))
        _run(plugin, zxz_windows(6), line=True, staged=True)
        unplaced, placed = ({v.candidate for v in r.variants} for r in reports)
        assert "search" in unplaced
        assert "search" not in placed

    def test_duplicate_names(self):
        with pytest.raises(ValueError, match="unique"):
            CommutingDiagonalizationSynthesis(candidates=[Plain(), Local(), Plain()])

    def test_iterators_are_read_once(self):
        """Candidates and edges given as iterators are not lost to the validation."""
        plugin = CommutingDiagonalizationSynthesis(
            candidates=iter([Plain(), Local()]), edges=iter([(1, 0), (1, 2)])
        )
        assert [c.name for c in plugin.candidates] == ["plain", "local"]
        assert plugin.edges == ((0, 1), (1, 2))

    def test_a_user_candidate_takes_part(self):
        """A candidate of the user's own competes with the built-in ones, and wins when it is the
        only one."""
        op = zxz_windows(5)
        plugin, reports = _recording(candidates=[*DEFAULT_CANDIDATES, _Fixed("stock")])
        assert same_unitary(_run(plugin, op), op, TIME)
        assert "stock" in {v.candidate for v in reports[0].variants}
        plugin, reports = _recording(candidates=[_Fixed("stock")])
        assert same_unitary(_run(plugin, op), op, TIME)
        assert reports[0].chosen.label == "stock"

    def test_twin_fold_wins_on_the_plaquettes(self):
        plugin, reports = _recording()
        assert _run(plugin, plaquettes(4)) is not None
        assert reports[0].chosen.candidate == "twin-fold"


class TestEdges:
    def test_user_edges_are_in_the_circuit_qubits(self):
        """Each gate in a circuit gets the pairs between its own qubits, in its own indices."""
        plugin, reports = _recording(edges=[(k, k + 1) for k in range(7)])
        circuit = QuantumCircuit(8)
        circuit.append(_gate(zxz_windows(4)), [4, 5, 6, 7])
        circuit.append(_gate(SparsePauliOp(["XX", "YY"], [0.4, 0.6])), [1, 3])
        HighLevelSynthesis(hls_config=HLSConfig(PauliEvolution=[plugin])).run(
            circuit_to_dag(circuit)
        )
        by_width = {
            r.problem.num_qubits: r for r in reports
        }  # the two gates may come in either order
        assert by_width[4].context.user_edges == ((0, 1), (1, 2), (2, 3))
        assert by_width[2].context.user_edges == ()  # 1 and 3 are not a pair

    def test_user_edges_of_a_direct_call(self):
        """Without the circuit's qubits, the pairs are in the gate's own indices; the ones past the
        gate are dropped."""
        plugin, reports = _recording(edges=[(0, 1), (1, 2), (2, 3)])
        for op in (SparsePauliOp(["XX", "YY"], [0.4, 0.6]), zxz_windows(4)):
            assert same_unitary(plugin.run(_gate(op)), op, TIME)
        assert [r.context.user_edges for r in reports] == [((0, 1),), ((0, 1), (1, 2), (2, 3))]

    @pytest.mark.parametrize("edges", [[(0, 0)], [(-1, 2)], [(0, 1, 2)]])
    def test_invalid_user_edges(self, edges):
        with pytest.raises(ValueError, match="edges"):
            CommutingDiagonalizationSynthesis(edges=edges)


class TestSelectors:
    @staticmethod
    def _variants(*circuits) -> list[Variant]:
        return [Variant("c", str(i), c) for i, c in enumerate(circuits)]

    @staticmethod
    def _ctx():
        op = SparsePauliOp(["ZZ"], [0.1])
        return Problem.from_operator(op, TIME), Context(2)

    def test_min_cost_takes_the_cheapest_and_the_first_of_ties(self):
        a, b, c = QuantumCircuit(2), QuantumCircuit(2), QuantumCircuit(2)
        a.cx(0, 1)
        a.cx(0, 1)
        b.cx(0, 1)
        c.cz(0, 1)
        variants = self._variants(a, b, c)
        assert MinCost(abstract_cost)(variants, *self._ctx()) is variants[1]

    def test_min_cost_scores_each_variant_once(self):
        calls = []

        def cost(circuit, context):
            calls.append(circuit)
            return 0

        variants = self._variants(QuantumCircuit(2), QuantumCircuit(2), QuantumCircuit(2))
        MinCost(cost)(variants, *self._ctx())
        assert len(calls) == 3

    def test_min_cost_drops_what_it_cannot_route(self):
        def cost(circuit, context):
            if circuit.num_qubits == 3:
                raise TranspilerError("cannot route")
            return 0

        variants = self._variants(QuantumCircuit(3), QuantumCircuit(2))
        assert MinCost(cost)(variants, *self._ctx()) is variants[1]
        assert MinCost(cost)(variants[:1] * 2, *self._ctx()) is None

    def test_min_cost_does_not_score_a_single_variant(self):
        def cost(circuit, context):
            raise AssertionError("scored")

        variants = self._variants(QuantumCircuit(2))
        assert MinCost(cost)(variants, *self._ctx()) is variants[0]
        assert MinCost(cost)([], *self._ctx()) is None

    def test_a_single_variant_pool_takes_that_circuit_unscored(self, monkeypatch):
        """Narrowing the candidates to one variant chooses it directly: no routed transpile runs,
        even on a device."""
        from qiskit_fermions.transpiler.passes.synthesis.commuting_evolution import cost

        def fail(*args, **kwargs):
            raise AssertionError("the routed cost ran")

        monkeypatch.setattr(cost, "transpile", fail)
        op = zxz_windows(6)
        plugin, reports = _recording(candidates=[Local(variants=["ladder"])])
        circuit = _run(plugin, op, line=True, staged=True)
        assert reports[0].chosen.label == "local[ladder]"
        assert same_unitary(circuit, op, TIME)

    def test_a_custom_metric_changes_the_choice(self):
        """The default ranks by two-qubit cost; ranking by the number of single-qubit gates picks a
        different one."""
        op = plaquettes(3)
        by_default = _recording()
        by_one_qubit = _recording(
            selector=MinCost(
                lambda circuit, context: sum(
                    1 for i in circuit.decompose(reps=6).data if len(i.qubits) == 1
                )
            )
        )
        for plugin, _ in (by_default, by_one_qubit):
            _run(plugin, op)
        first, second = by_default[1][0].chosen, by_one_qubit[1][0].chosen
        assert first.label != second.label

    def test_a_custom_selector_sees_every_variant(self):
        seen = []

        def last(variants, problem, context):
            seen.append(len(variants))
            return variants[-1]

        plugin, reports = _recording(selector=last)
        _run(plugin, zxz_windows(5))
        assert seen == [len(reports[0].variants)]
        assert reports[0].chosen is reports[0].variants[-1]

    def test_abstract_cost_never_routes(self, monkeypatch):
        """``MinCost(abstract_cost)`` never runs the routed scoring transpile."""
        from qiskit_fermions.transpiler.passes.synthesis.commuting_evolution import cost

        def fail(*args, **kwargs):
            raise AssertionError("the routed cost ran")

        monkeypatch.setattr(cost, "transpile", fail)
        op = zxz_windows(6)
        plugin = CommutingDiagonalizationSynthesis(selector=MinCost(abstract_cost))
        circuit = _run(plugin, op, line=True, staged=True)
        assert same_unitary(circuit, op, TIME)


class TestStructure:
    def test_deterministic(self):
        op = GROUPS["random"]()
        first = _run(CommutingDiagonalizationSynthesis(), op, line=True, staged=True)
        second = _run(CommutingDiagonalizationSynthesis(), op, line=True, staged=True)
        assert first == second

    def test_report(self):
        plugin, reports = _recording()
        op = zxz_windows(5)
        circuit = _run(plugin, op, line=True, staged=True)
        (report,) = reports
        assert report.declined is None
        assert report.chosen in report.variants and report.chosen.circuit is circuit
        assert report.context.physical_qubits == tuple(range(5))
        assert report.problem.num_qubits == 5 and report.problem.operator.num_terms == len(op)

    def test_no_state_between_gates(self):
        """The same instance gives the same circuit for a gate whatever it synthesized before."""
        op = SparsePauliOp.from_sparse_list([("ZZZZ", [0, 1, 2, 3], 0.9)], num_qubits=4)
        plugin = CommutingDiagonalizationSynthesis()
        reference = plugin.run(_gate(op))
        _run(plugin, plaquettes(3), line=True, staged=True)
        assert plugin.run(_gate(op)) == reference

    def test_selection_follows_the_candidate_order_on_ties(self):
        """Two candidates offering the same circuit: the earlier one wins."""
        circuit = QuantumCircuit(2)
        circuit.append(PauliEvolutionGate(SparsePauliOp(["ZZ"], [0.2]), TIME), circuit.qubits)
        circuit = circuit.decompose()
        for order in itertools.permutations(["a", "b"]):
            plugin, reports = _recording(candidates=[_Fixed(n, circuit) for n in order])
            plugin.run(_gate(SparsePauliOp(["ZZ"], [0.2])))
            assert reports[0].chosen.candidate == order[0]


class TestThroughThePassManager:
    def test_by_instance(self):
        op = plaquettes(2)
        qc = QuantumCircuit(op.num_qubits)
        qc.append(_gate(op), range(op.num_qubits))
        config = HLSConfig(PauliEvolution=[CommutingDiagonalizationSynthesis(), "default"])
        out = generate_preset_pass_manager(
            basis_gates=["cx", "rz", "sx", "x"], optimization_level=1, hls_config=config
        ).run(qc)
        assert same_unitary(out, op, TIME, atol=1e-9)

    def test_by_name_through_qiskits_plugin_manager(self):
        """The entry point is registered with Qiskit's own group and the plugin is built with no
        arguments."""
        from qiskit.transpiler.passes.synthesis.plugin import HighLevelSynthesisPluginManager

        manager = HighLevelSynthesisPluginManager()
        assert "commuting_diagonalization" in manager.method_names("PauliEvolution")
        assert isinstance(
            manager.method("PauliEvolution", "commuting_diagonalization"),
            CommutingDiagonalizationSynthesis,
        )

    def test_by_name_in_a_pass_manager(self):
        op = zxz_windows(5)
        qc = QuantumCircuit(5)
        qc.append(_gate(op), range(5))
        config = HLSConfig(PauliEvolution=["commuting_diagonalization", "default"])
        out = generate_preset_pass_manager(
            basis_gates=["cx", "rz", "sx", "x"], optimization_level=1, hls_config=config
        ).run(qc)
        assert same_unitary(out, op, TIME, atol=1e-9)

    def test_staged_with_a_layout_on_a_line(self):
        """The plugin placed after the layout stage sees the physical qubits and the device."""
        from qiskit.transpiler import PassManager

        op = zxz_windows(6)
        qc = QuantumCircuit(6)
        qc.append(_gate(op), range(6))
        plugin, reports = _recording()
        config = HLSConfig(PauliEvolution=[plugin, "default"])
        pm = PassManager(
            [
                HighLevelSynthesis(
                    hls_config=config, coupling_map=CouplingMap.from_line(6), use_qubit_indices=True
                )
            ]
        )
        out = pm.run(qc)
        assert reports[0].context.physical_qubits == tuple(range(6))
        assert same_unitary(out, op, TIME, atol=1e-9)
