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
from functools import partial

from qiskit.circuit import QuantumRegister
from qiskit.passmanager import MultiStagePassManager
from qiskit.quantum_info import SparseObservable
from qiskit_fermions.circuit import FermionicCircuit
from qiskit_fermions.circuit.library import Evolution
from qiskit_fermions.operators import MajoranaOperator
from qiskit_fermions.transpiler import FermionicCircuitToDAG, QuantumDAGToCircuit
from qiskit_fermions.transpiler.passes import (
    CustomF2QLayout,
    F2QSynthesis,
    MapperFnEvolutionSynthesis,
)


# NOTE: this is a very specific implementation of the Fermi-Hubbard model on a square lattice. It is
# not intended for general purpose use and tailored to the purposes of this test case.
def build_fermi_hubbard_square_lattice(
    ncols: int, nrows: int, interaction: complex, tunneling: complex
) -> MajoranaOperator:
    """Defines a Fermi-Hubbard Hamiltonian on a 4 by 4 lattice of spinless fermionic sites.

    Args:
        ncols: the number of columns in the square lattice of spinless fermionic sites.
        nrows: the number of rows in the square lattice of spinless fermionic sites.
        interaction: the strength of the Coulomb terms.
        tunneling: the strength of the tunneling terms.

    Returns:
        The spinless Fermi-Hubbard Hamiltonian as a Majorana-operator.
    """
    interaction /= 4
    tunneling *= 0.5j

    nsites = nrows * ncols

    data: dict[tuple[int, ...], complex] = defaultdict(complex)
    for i in range(nsites):
        row = i // ncols
        col = i % nrows

        js = []

        # horizontal edges
        if col != (ncols - 1) and row % 2 == 0:
            js.append(i + 1)
        elif col != 0 and row % 2 == 1:
            js.append(i - 1)

        # vertical edges
        if col % 2 == 1:
            # down: i < j
            j = i + nrows
            if j < nsites:
                js.append(j)
        else:
            # up: i > j
            j = i - ncols
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


# NOTE: this implementation of the Derby-Klassen fermion-to-qubit encoding is not necessarily
# general and only implemented for the purposes of this test.
def derby_klassen(
    op: MajoranaOperator,
    num_qubits: int,
    initial_state: list[bool],
    edge_face_map: dict[tuple[int, int], int],
) -> SparseObservable:
    """Implements the Derby-Klassen fermion-to-qubit encoding. [1]_

    Args:
        op: the operator to encode.
        num_qubits: the total number of qubits in the resulting operator.
        initial_state: the initial occupation state of the fermionic modes.
        edge_face_map: a mapping of fermionic lattice edges to auxiliary qubit indices.

    Returns:
        The mapped operator.

    .. [1] C. Derby, J. Klassen, J. Bausch, and T. Cubitt, Compact fermion to qubit mappings,
           Phys. Rev. B 104, 035118 (2021),
           `doi:10.1103/PhysRevB.104.035118 <http://dx.doi.org/10.1103/PhysRevB.104.035118>`_.
    """
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


def test_derby_klassen():
    """Tests the implementation of the Derby-Klassen fermion-to-qubit encoding. [1]_

    .. note::
       This test exists solely as an auxiliary test to `test_custom_layout`.

    .. [1] C. Derby, J. Klassen, J. Bausch, and T. Cubitt, Compact fermion to qubit mappings,
           Phys. Rev. B 104, 035118 (2021),
           `doi:10.1103/PhysRevB.104.035118 <http://dx.doi.org/10.1103/PhysRevB.104.035118>`_.
    """
    num_qubits = 20
    hamil = build_fermi_hubbard_square_lattice(4, 4, 5.0, 5.0)
    initial_state = [bool(int(c)) for c in "1010010110100101"]
    edge_face_map = {
        (1, 2): 16,
        (1, 5): 16,
        (6, 2): 16,
        (6, 5): 16,
        (4, 5): 17,
        (4, 8): 17,
        (9, 5): 17,
        (9, 8): 17,
        (6, 7): 18,
        (6, 10): 18,
        (11, 7): 18,
        (11, 10): 18,
        (9, 10): 19,
        (9, 13): 19,
        (14, 10): 19,
        (14, 13): 19,
    }
    qop = derby_klassen(hamil, num_qubits, initial_state, edge_face_map)

    expected = SparseObservable.from_list(
        [
            ("IIIIIIIIIIIIIIIIIIYY", -2.5),
            ("IIIIIIIIIIIIIIIIIIXX", 2.5),
            ("IIIIIIIIIIIIIIIIIIZI", 3.75),
            ("IIIIIIIIIIIIIIIIIIIZ", -2.5),
            ("IIIIIIIIIIIIIIIIIIZZ", -1.25),
            ("IIIIIIIIIIIIIIIYIIIY", -2.5),
            ("IIIIIIIIIIIIIIIXIIIX", 2.5),
            ("IIIIIIIIIIIIIIIZIIII", 3.75),
            ("IIIIIIIIIIIIIIIZIIIZ", -1.25),
            ("IIIYIIIIIIIIIIIIIYYI", 2.5),
            ("IIIYIIIIIIIIIIIIIXXI", -2.5),
            ("IIIIIIIIIIIIIIIIIZII", -3.75),
            ("IIIIIIIIIIIIIIIIIZZI", -1.25),
            ("IIIXIIIIIIIIIIYIIIYI", 2.5),
            ("IIIXIIIIIIIIIIXIIIXI", -2.5),
            ("IIIIIIIIIIIIIIZIIIII", -5.0),
            ("IIIIIIIIIIIIIIZIIIZI", -1.25),
            ("IIIIIIIIIIIIIIIIYYII", -2.5),
            ("IIIIIIIIIIIIIIIIXXII", 2.5),
            ("IIIIIIIIIIIIIIIIZIII", 2.5),
            ("IIIIIIIIIIIIIIIIZZII", -1.25),
            ("IIIXIIIIIIIIIYIIIYII", -2.5),
            ("IIIXIIIIIIIIIXIIIXII", 2.5),
            ("IIIIIIIIIIIIIZIIIIII", 5.0),
            ("IIIIIIIIIIIIIZIIIZII", -1.25),
            ("IIIIIIIIIIIIYIIIYIII", 2.5),
            ("IIIIIIIIIIIIXIIIXIII", -2.5),
            ("IIIIIIIIIIIIZIIIIIII", -3.75),
            ("IIIIIIIIIIIIZIIIZIII", -1.25),
            ("IIYIIIIIIIIIIIYYIIII", -2.5),
            ("IIYIIIIIIIIIIIXXIIII", 2.5),
            ("IIIIIIIIIIIIIIZZIIII", -1.25),
            ("IIXIIIIIIIIYIIIYIIII", 2.5),
            ("IIXIIIIIIIIXIIIXIIII", -2.5),
            ("IIIIIIIIIIIZIIIIIIII", -3.75),
            ("IIIIIIIIIIIZIIIZIIII", -1.25),
            ("IIIYIIIIIIIIIYYIIIII", 2.5),
            ("IIIYIIIIIIIIIXXIIIII", -2.5),
            ("IIIIIIIIIIIIIZZIIIII", -1.25),
            ("IIXIIIIIIIYIIIYIIIII", -2.5),
            ("IIXIIIIIIIXIIIXIIIII", 2.5),
            ("IIIIIIIIIIZIIIIIIIII", 5.0),
            ("IIIIIIIIIIZIIIZIIIII", -1.25),
            ("IYIIIIIIIIIIYYIIIIII", -2.5),
            ("IYIIIIIIIIIIXXIIIIII", 2.5),
            ("IIIIIIIIIIIIZZIIIIII", -1.25),
            ("IXIIIIIIIYIIIYIIIIII", 2.5),
            ("IXIIIIIIIXIIIXIIIIII", -2.5),
            ("IIIIIIIIIZIIIIIIIIII", -5.0),
            ("IIIIIIIIIZIIIZIIIIII", -1.25),
            ("IXIIIIIIYIIIYIIIIIII", -2.5),
            ("IXIIIIIIXIIIXIIIIIII", 2.5),
            ("IIIIIIIIZIIIIIIIIIII", 3.75),
            ("IIIIIIIIZIIIZIIIIIII", -1.25),
            ("IIYIIIIIIIYYIIIIIIII", -2.5),
            ("IIYIIIIIIIXXIIIIIIII", 2.5),
            ("IIIIIIIIIIZZIIIIIIII", -1.25),
            ("IIIIIIIYIIIYIIIIIIII", -2.5),
            ("IIIIIIIXIIIXIIIIIIII", 2.5),
            ("IIIIIIIZIIIIIIIIIIII", 2.5),
            ("IIIIIIIZIIIZIIIIIIII", -1.25),
            ("YIIIIIIIIYYIIIIIIIII", 2.5),
            ("YIIIIIIIIXXIIIIIIIII", -2.5),
            ("IIIIIIIIIZZIIIIIIIII", -1.25),
            ("XIIIIIYIIIYIIIIIIIII", 2.5),
            ("XIIIIIXIIIXIIIIIIIII", -2.5),
            ("IIIIIIZIIIIIIIIIIIII", -3.75),
            ("IIIIIIZIIIZIIIIIIIII", -1.25),
            ("IYIIIIIIYYIIIIIIIIII", -2.5),
            ("IYIIIIIIXXIIIIIIIIII", 2.5),
            ("IIIIIIIIZZIIIIIIIIII", -1.25),
            ("XIIIIYIIIYIIIIIIIIII", -2.5),
            ("XIIIIXIIIXIIIIIIIIII", 2.5),
            ("IIIIIZIIIIIIIIIIIIII", 3.75),
            ("IIIIIZIIIZIIIIIIIIII", -1.25),
            ("IIIIYIIIYIIIIIIIIIII", 2.5),
            ("IIIIXIIIXIIIIIIIIIII", -2.5),
            ("IIIIZIIIIIIIIIIIIIII", -2.5),
            ("IIIIZIIIZIIIIIIIIIII", -1.25),
            ("IIIIIIYYIIIIIIIIIIII", -2.5),
            ("IIIIIIXXIIIIIIIIIIII", 2.5),
            ("IIIIIIZZIIIIIIIIIIII", -1.25),
            ("YIIIIYYIIIIIIIIIIIII", 2.5),
            ("YIIIIXXIIIIIIIIIIIII", -2.5),
            ("IIIIIZZIIIIIIIIIIIII", -1.25),
            ("IIIIYYIIIIIIIIIIIIII", -2.5),
            ("IIIIXXIIIIIIIIIIIIII", 2.5),
            ("IIIIZZIIIIIIIIIIIIII", -1.25),
        ]
    )

    diff = (expected - qop).simplify(tol=0.0)
    assert diff == SparseObservable.zero(num_qubits)


def test_custom_layout():
    """Tests the fermion-to-qubit transpilation pipeline with a custom fermion-to-qubit encoding.

    The purpose of this test is to ensure that fermion-to-qubit encodings which result in a change
    in the number of bits (fermionic modes vs. qubits) are correctly supported by the transpilation
    pipeline. To this end, this test implements the Derby-Klassen fermion-to-qubit encoding. [1]_

    .. [1] C. Derby, J. Klassen, J. Bausch, and T. Cubitt, Compact fermion to qubit mappings,
           Phys. Rev. B 104, 035118 (2021),
           `doi:10.1103/PhysRevB.104.035118 <http://dx.doi.org/10.1103/PhysRevB.104.035118>`_.
    """
    num_modes = 16
    num_qubits = 20
    hamil = build_fermi_hubbard_square_lattice(4, 4, 5.0, 5.0)

    initial_state = [bool(int(c)) for c in "1010010110100101"]
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
    mapper_fn = partial(derby_klassen, initial_state=initial_state, edge_face_map=edge_face_map)

    circ = FermionicCircuit(num_modes)
    circ.append(Evolution(num_modes, hamil), circ.modes)

    layout = CustomF2QLayout({circ.register: QuantumRegister(num_qubits)})

    synth = F2QSynthesis()
    synth.methods["Evolution"] = MapperFnEvolutionSynthesis(mapper_fn)

    pm = MultiStagePassManager(
        input=FermionicCircuitToDAG(),
        layout=layout,
        synthesis=synth,
        output=QuantumDAGToCircuit(),
    )

    qu_circ = pm.run(circ)
    qu_circ_decomp = qu_circ.decompose()
    assert qu_circ_decomp.depth(lambda instr: len(instr.qubits) == 2) == 93
