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

"""Custom layout tests."""

from __future__ import annotations

from collections import defaultdict

from qiskit.circuit import QuantumRegister
from qiskit.quantum_info import SparseObservable, SparsePauliOp
from qiskit.transpiler import PassManager
from qiskit_fermions.circuit import FermionCircuit
from qiskit_fermions.circuit.library import Evolution
from qiskit_fermions.operators import MajoranaOperator
from qiskit_fermions.transpiler.passes import CustomF2QLayout, EvolutionSynthesis, F2QSynthesis


def build_fermi_hubbard_4x4(interaction: complex, tunneling: complex):
    """Defines a Fermi-Hubbard Hamiltonian on a 4 by 4 lattice of spinless fermionic sites."""
    interaction /= 4
    tunneling *= 0.5j

    data: dict[tuple[int, ...], complex] = defaultdict(complex)
    for i in range(16):
        row = i // 4
        col = i % 4

        js = []

        # horizontal edges
        if col != 3 and row % 2 == 0:
            js.append(i + 1)
        elif col != 0 and row % 2 == 1:
            js.append(i - 1)

        # vertical edges
        if col % 2 == 1:
            # down: i < j
            j = i + 4
            if j < 16:
                js.append(j)
        else:
            # up: i > j
            j = i - 4
            if j >= 0:
                js.append(j)

        for j in js:
            # interaction
            data[(i, i)] -= interaction
            data[(j, j)] -= interaction
            data[(i, i, j, j)] += interaction

            # tunneling
            data[(i, i, i, j)] += tunneling
            data[(j, j, i, j)] -= tunneling

    return MajoranaOperator.from_dict(data)


def build_derby_klassen_edge_face_map_4x4():
    """Defines the edge-to-face map for a 4 by 4 lattice used in the Derby-Klassen F2Q encoding."""
    edge_face_map = {
        (1, 2): 16,
        (1, 5): 16,
        (6, 2): 16,
        (6, 5): 16,
        (6, 7): 18,
        (6, 10): 18,
        (11, 7): 18,
        (11, 10): 18,
        (4, 5): 17,
        (4, 8): 17,
        (9, 5): 17,
        (9, 8): 17,
        (9, 10): 19,
        (9, 13): 19,
        (14, 10): 19,
        (14, 13): 19,
    }
    return edge_face_map


def derby_klassen(
    op: MajoranaOperator,
    initial_state: list[bool],
    edge_face_map: dict[tuple[int, int], int],
    num_qubits: int,
) -> SparseObservable:
    """Implements the Derby-Klassen fermion-to-qubit encoding."""
    assert op.is_even()

    mapped_operator = SparseObservable.zero(num_qubits)

    for terms, coeff in op.iter_terms():
        mapped_terms = SparseObservable.identity(num_qubits)

        for i, j in zip(terms[::2], terms[1::2], strict=True):
            dij = abs(i - j)

            face_qubit: int | None = None
            if (i, j) in edge_face_map:
                face_qubit = edge_face_map[(i, j)]
            elif (j, i) in edge_face_map:
                face_qubit = edge_face_map[(j, i)]

            if dij == 0:
                # vertex term
                sign = -((-1) ** initial_state[i])
                mapped_terms = mapped_terms.compose(
                    SparseObservable.from_sparse_list([("Z", (i,), sign)], num_qubits), front=True
                )

            elif dij == 1:
                # horizontal edge term
                paulis = "XYY"[: 2 if face_qubit is None else 3]
                indices = (i, j) if face_qubit is None else (i, j, face_qubit)
                mapped_terms = mapped_terms.compose(
                    SparseObservable.from_sparse_list([(paulis, indices, 1.0)], num_qubits),
                    front=True,
                )

            else:
                # vertical edge term
                # NOTE: we hard-code whether an edge points up or down based on the column index
                sign = 1.0 if i % 2 else -1.0
                paulis = "XYX"[: 2 if face_qubit is None else 3]
                indices = (i, j) if face_qubit is None else (i, j, face_qubit)
                mapped_terms = mapped_terms.compose(
                    SparseObservable.from_sparse_list([(paulis, indices, sign)], num_qubits),
                    front=True,
                )

        mapped_operator += coeff * mapped_terms

    return mapped_operator


def test_custom_layout():
    num_fermions = 16
    num_qubits = 20
    hamil = build_fermi_hubbard_4x4(5.0, 5.0)
    print()
    initial_state = [True, False, True, False, False, True, False, True] * 2
    edge_face_map = build_derby_klassen_edge_face_map_4x4()
    qop = derby_klassen(hamil, initial_state, edge_face_map, num_qubits)

    pauli_op = SparsePauliOp.from_sparse_observable(qop)

    # fmt: off
    expected = SparsePauliOp(
        [
            "IIIIIIIIIIIIIIIIIIYY", "IIIIIIIIIIIIIIIIIIXX", "IIIIIIIIIIIIIIIIIIZI", "IIIIIIIIIIIIIIIIIIIZ",
            "IIIIIIIIIIIIIIIIIIZZ", "IIIIIIIIIIIIYIIIIIIY", "IIIIIIIIIIIIXIIIIIIX", "IIIIIIIIIIIIZIIIIIII",
            "IIIIIIIIIIIIZIIIIIIZ", "IIIYIIIIIIIIIIIIIYYI", "IIIYIIIIIIIIIIIIIXXI", "IIIIIIIIIIIIIIIIIZII",
            "IIIIIIIIIIIIIIIIIZZI", "IIIXIIIIIIIIIYIIIIYI", "IIIXIIIIIIIIIXIIIIXI", "IIIIIIIIIIIIIZIIIIII",
            "IIIIIIIIIIIIIZIIIIZI", "IIIIIIIIIIIIIIIIYYII", "IIIIIIIIIIIIIIIIXXII", "IIIIIIIIIIIIIIIIZIII",
            "IIIIIIIIIIIIIIIIZZII", "IIIXIIIIIIIIIIYIIYII", "IIIXIIIIIIIIIIXIIXII", "IIIIIIIIIIIIIIZIIIII",
            "IIIIIIIIIIIIIIZIIZII", "IIIIIIIIIIIIIIIYYIII", "IIIIIIIIIIIIIIIXXIII", "IIIIIIIIIIIIIIIZIIII",
            "IIIIIIIIIIIIIIIZZIII", "IIYIIIIIIIIIYYIIIIII", "IIYIIIIIIIIIXXIIIIII", "IIIIIIIIIIIIZZIIIIII",
            "IIXIIIIIIIIYYIIIIIII", "IIXIIIIIIIIXXIIIIIII", "IIIIIIIIIIIZIIIIIIII", "IIIIIIIIIIIZZIIIIIII",
            "IIIYIIIIIIIIIYYIIIII", "IIIYIIIIIIIIIXXIIIII", "IIIIIIIIIIIIIZZIIIII", "IIXIIIIIIIYIIYIIIIII",
            "IIXIIIIIIIXIIXIIIIII", "IIIIIIIIIIZIIIIIIIII", "IIIIIIIIIIZIIZIIIIII", "IYIIIIIIIIIIIIYYIIII",
            "IYIIIIIIIIIIIIXXIIII", "IIIIIIIIIIIIIIZZIIII", "IXIIIIIIIYIIIIYIIIII", "IXIIIIIIIXIIIIXIIIII",
            "IIIIIIIIIZIIIIIIIIII", "IIIIIIIIIZIIIIZIIIII", "IXIIIIIIYIIIIIIYIIII", "IXIIIIIIXIIIIIIXIIII",
            "IIIIIIIIZIIIIIIIIIII", "IIIIIIIIZIIIIIIZIIII", "IIYIIIIIIIYYIIIIIIII", "IIYIIIIIIIXXIIIIIIII",
            "IIIIIIIIIIZZIIIIIIII", "IIIIYIIIIIIYIIIIIIII", "IIIIXIIIIIIXIIIIIIII", "IIIIZIIIIIIIIIIIIIII",
            "IIIIZIIIIIIZIIIIIIII", "YIIIIIIIIYYIIIIIIIII", "YIIIIIIIIXXIIIIIIIII", "IIIIIIIIIZZIIIIIIIII",
            "XIIIIYIIIIYIIIIIIIII", "XIIIIXIIIIXIIIIIIIII", "IIIIIZIIIIIIIIIIIIII", "IIIIIZIIIIZIIIIIIIII",
            "IYIIIIIIYYIIIIIIIIII", "IYIIIIIIXXIIIIIIIIII", "IIIIIIIIZZIIIIIIIIII", "XIIIIIYIIYIIIIIIIIII",
            "XIIIIIXIIXIIIIIIIIII", "IIIIIIZIIIIIIIIIIIII", "IIIIIIZIIZIIIIIIIIII", "IIIIIIIYYIIIIIIIIIII",
            "IIIIIIIXXIIIIIIIIIII", "IIIIIIIZIIIIIIIIIIII", "IIIIIIIZZIIIIIIIIIII", "IIIIYYIIIIIIIIIIIIII",
            "IIIIXXIIIIIIIIIIIIII", "IIIIZZIIIIIIIIIIIIII", "YIIIIYYIIIIIIIIIIIII", "YIIIIXXIIIIIIIIIIIII",
            "IIIIIZZIIIIIIIIIIIII", "IIIIIIYYIIIIIIIIIIII", "IIIIIIXXIIIIIIIIIIII", "IIIIIIZZIIIIIIIIIIII",
        ],
        coeffs=[
            -2.5 + 0.0j, 2.5 + 0.0j, 3.75 + 0.0j, -2.5 + 0.0j, -1.25 + 0.0j, -2.5 + 0.0j, 2.5 + 0.0j,
            3.75 + 0.0j, -1.25 + 0.0j, 2.5 + 0.0j, -2.5 + 0.0j, -3.75 + 0.0j, -1.25 + 0.0j, 2.5 + 0.0j,
            -2.5 + 0.0j, -5.0 + 0.0j, -1.25 + 0.0j, -2.5 + 0.0j, 2.5 + 0.0j, 2.5 + 0.0j, -1.25 + 0.0j,
            -2.5 + 0.0j, 2.5 + 0.0j, 5.0 + 0.0j, -1.25 + 0.0j, 2.5 + 0.0j, -2.5 + 0.0j, -3.75 + 0.0j,
            -1.25 + 0.0j, -2.5 + 0.0j, 2.5 + 0.0j, -1.25 + 0.0j, 2.5 + 0.0j, -2.5 + 0.0j, -3.75 + 0.0j,
            -1.25 + 0.0j, 2.5 + 0.0j, -2.5 + 0.0j, -1.25 + 0.0j, -2.5 + 0.0j, 2.5 + 0.0j, 5.0 + 0.0j,
            -1.25 + 0.0j, -2.5 + 0.0j, 2.5 + 0.0j, -1.25 + 0.0j, 2.5 + 0.0j, -2.5 + 0.0j, -5.0 + 0.0j,
            -1.25 + 0.0j, -2.5 + 0.0j, 2.5 + 0.0j, 3.75 + 0.0j, -1.25 + 0.0j, -2.5 + 0.0j, 2.5 + 0.0j,
            -1.25 + 0.0j, -2.5 + 0.0j, 2.5 + 0.0j, 2.5 + 0.0j, -1.25 + 0.0j, 2.5 + 0.0j, -2.5 + 0.0j,
            -1.25 + 0.0j, 2.5 + 0.0j, -2.5 + 0.0j, -3.75 + 0.0j, -1.25 + 0.0j, -2.5 + 0.0j, 2.5 + 0.0j,
            -1.25 + 0.0j, -2.5 + 0.0j, 2.5 + 0.0j, 3.75 + 0.0j, -1.25 + 0.0j, 2.5 + 0.0j, -2.5 + 0.0j,
            -2.5 + 0.0j, -1.25 + 0.0j, -2.5 + 0.0j, 2.5 + 0.0j, -1.25 + 0.0j, 2.5 + 0.0j, -2.5 + 0.0j,
            -1.25 + 0.0j, -2.5 + 0.0j, 2.5 + 0.0j, -1.25 + 0.0j,
        ],
    )
    # fmt: on

    layout = [0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12, 16, 17, 18, 19]
    pauli_op = pauli_op.apply_layout(layout)

    diff = (expected - pauli_op).simplify(atol=0.0)
    assert diff == SparsePauliOp.from_sparse_list([], num_qubits)

    circ = FermionCircuit(num_fermions)
    circ.append(Evolution(num_fermions, hamil), circ.fermions)

    layout = CustomF2QLayout({circ.register: QuantumRegister(num_qubits)})

    def mapper_fn(op):
        return derby_klassen(op, initial_state, edge_face_map, num_qubits)

    synth = F2QSynthesis()
    synth.plugins[Evolution] = EvolutionSynthesis(mapper_fn)

    pm = PassManager([layout, synth])

    qu_circ = pm.run(circ._inner)
    qu_circ_decomp = qu_circ.decompose()
    assert qu_circ_decomp.depth(lambda instr: len(instr.qubits) == 2) == 93
