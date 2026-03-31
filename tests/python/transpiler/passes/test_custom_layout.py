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

from qiskit.quantum_info import SparseObservable, SparsePauliOp
from qiskit_fermions.operators import MajoranaOperator, gamma


def build_fermi_hubbard_4x4(interaction: complex, tunneling: complex):
    """Defines a Fermi-Hubbard Hamiltonian on a 4 by 4 lattice of spinless fermionic sites."""
    interaction /= 4
    tunneling *= 0.5j

    data: dict[tuple[int, ...], complex] = defaultdict(complex)
    for i in range(16):
        row = i // 4
        col = 3 - i % 4 if row % 2 else i % 4

        js = []

        # horizontal edges
        if i not in {3, 7, 11, 15}:
            js.append(i + 1)

        # vertical edges
        if col % 2 == 1:
            # down: i < j
            row_down = row + 1
            j = 4 * row_down
            if row_down % 2:
                j += 3 - col
            else:
                j += col
            if j < 16:
                js.append(j)
        else:
            # up: i > j
            row_up = row - 1
            j = 4 * row_up
            if row_up % 2:
                j += 3 - col
            else:
                j += col
            if row_up >= 0:
                js.append(j)

        for j in js:
            # interaction
            data[(gamma(i, False), gamma(i, True))] -= interaction
            data[(gamma(j, False), gamma(j, True))] -= interaction
            data[(gamma(i, False), gamma(i, True), gamma(j, False), gamma(j, True))] += interaction

            # tunneling
            data[(gamma(i, False), gamma(i, True), gamma(i, False), gamma(j, False))] += tunneling
            data[(gamma(j, False), gamma(j, True), gamma(i, False), gamma(j, False))] -= tunneling

    return MajoranaOperator.from_dict(data)


def build_derby_klassen_edge_face_map_4x4():
    """Defines the edge-to-face map for a 4 by 4 lattice used in the Derby-Klassen F2Q encoding."""
    edge_face_map = {
        (1, 2): 16,
        (1, 6): 16,
        (5, 2): 16,
        (6, 5): 16,
        (5, 4): 18,
        (5, 10): 18,
        (11, 4): 18,
        (11, 10): 18,
        (7, 6): 17,
        (7, 8): 17,
        (9, 6): 17,
        (9, 8): 17,
        (9, 10): 19,
        (9, 14): 19,
        (13, 10): 19,
        (13, 14): 19,
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
            i = i // 2
            j = j // 2
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

            elif dij == 1 and (i, j) != (8, 7):
                # horizontal edge term
                # FIXME: just because dij == 1, that does not mean we are guaranteed to have a
                # horizontal term in Anthony's index order!
                paulis = "XYY"[: 2 if face_qubit is None else 3]
                indices = (i, j) if face_qubit is None else (i, j, face_qubit)
                mapped_terms = mapped_terms.compose(
                    SparseObservable.from_sparse_list([(paulis, indices, 1.0)], num_qubits),
                    front=True,
                )

            else:
                # vertical edge term
                # NOTE: we hard-code whether an edge points up or down based on the relative size of
                # i and j
                sign = -1.0 if i > j else 1.0
                paulis = "XYX"[: 2 if face_qubit is None else 3]
                indices = (i, j) if face_qubit is None else (i, j, face_qubit)
                mapped_terms = mapped_terms.compose(
                    SparseObservable.from_sparse_list([(paulis, indices, sign)], num_qubits),
                    front=True,
                )

        mapped_operator += coeff * mapped_terms

    return mapped_operator


def test_dk_vertex_horizontal():
    hamil = MajoranaOperator.from_dict(
        {
            (gamma(0, False), gamma(0, True)): -1.25,
            (gamma(1, False), gamma(1, True)): -1.25,
            (gamma(0, False), gamma(0, True), gamma(1, False), gamma(1, True)): 1.25,
        }
    )
    initial_state = [True, False] * 8
    edge_face_map = build_derby_klassen_edge_face_map_4x4()
    num_qubits = 20
    qop = derby_klassen(hamil, initial_state, edge_face_map, num_qubits)
    expected = SparseObservable.from_sparse_list(
        [
            ("Z", (0,), -1.25),
            ("Z", (1,), 1.25),
            ("ZZ", (0, 1), -1.25),
        ],
        num_qubits,
    )
    assert (qop - expected).simplify() == SparseObservable.zero(num_qubits)


def test_dk_vertex_vertical():
    hamil = MajoranaOperator.from_dict(
        {
            (gamma(0, False), gamma(0, True)): -1.25,
            (gamma(7, False), gamma(7, True)): -1.25,
            (gamma(0, False), gamma(0, True), gamma(7, False), gamma(7, True)): 1.25,
        }
    )
    initial_state = [True, False] * 8
    edge_face_map = build_derby_klassen_edge_face_map_4x4()
    num_qubits = 20
    qop = derby_klassen(hamil, initial_state, edge_face_map, num_qubits)
    expected = SparseObservable.from_sparse_list(
        [
            ("Z", (0,), -1.25),
            ("Z", (7,), 1.25),
            ("ZZ", (0, 7), -1.25),
        ],
        num_qubits,
    )
    assert (qop - expected).simplify() == SparseObservable.zero(num_qubits)


def test_dk_edge_horizontal_right():
    hamil = MajoranaOperator.from_dict(
        {
            (gamma(0, False), gamma(0, True), gamma(0, False), gamma(1, False)): 2.5j,
            (gamma(1, False), gamma(1, True), gamma(0, False), gamma(1, False)): -2.5j,
        }
    )
    initial_state = [True, False] * 8
    edge_face_map = build_derby_klassen_edge_face_map_4x4()
    num_qubits = 20
    qop = derby_klassen(hamil, initial_state, edge_face_map, num_qubits)
    expected = SparseObservable.from_sparse_list(
        [
            ("XX", (0, 1), 2.5),
            ("YY", (0, 1), -2.5),
        ],
        num_qubits,
    )
    assert (qop - expected).simplify() == SparseObservable.zero(num_qubits)


def test_dk_edge_horizontal_right_with_face_below():
    hamil = MajoranaOperator.from_dict(
        {
            (gamma(1, False), gamma(1, True), gamma(1, False), gamma(2, False)): 2.5j,
            (gamma(2, False), gamma(2, True), gamma(1, False), gamma(2, False)): -2.5j,
        }
    )
    initial_state = [True, False] * 8
    edge_face_map = build_derby_klassen_edge_face_map_4x4()
    num_qubits = 20
    qop = derby_klassen(hamil, initial_state, edge_face_map, num_qubits)
    expected = SparseObservable.from_sparse_list(
        [
            ("XXY", (1, 2, 16), -2.5),
            ("YYY", (1, 2, 16), 2.5),
        ],
        num_qubits,
    )
    assert (qop - expected).simplify() == SparseObservable.zero(num_qubits)


def test_dk_edge_horizontal_right_with_face_above():
    hamil = MajoranaOperator.from_dict(
        {
            (gamma(8, False), gamma(8, True), gamma(8, False), gamma(9, False)): 2.5j,
            (gamma(9, False), gamma(9, True), gamma(8, False), gamma(9, False)): -2.5j,
        }
    )
    initial_state = [True, False] * 8
    edge_face_map = build_derby_klassen_edge_face_map_4x4()
    num_qubits = 20
    qop = derby_klassen(hamil, initial_state, edge_face_map, num_qubits)
    expected = SparseObservable.from_sparse_list(
        [
            ("XXY", (8, 9, 17), 2.5),
            ("YYY", (8, 9, 17), -2.5),
        ],
        num_qubits,
    )
    assert (qop - expected).simplify() == SparseObservable.zero(num_qubits)


def test_dk_edge_horizontal_left():
    hamil = MajoranaOperator.from_dict(
        {
            (gamma(14, False), gamma(14, True), gamma(14, False), gamma(15, False)): 2.5j,
            (gamma(15, False), gamma(15, True), gamma(14, False), gamma(15, False)): -2.5j,
        }
    )
    initial_state = [True, False] * 8
    edge_face_map = build_derby_klassen_edge_face_map_4x4()
    num_qubits = 20
    qop = derby_klassen(hamil, initial_state, edge_face_map, num_qubits)
    expected = SparseObservable.from_sparse_list(
        [
            ("XX", (14, 15), 2.5),
            ("YY", (14, 15), -2.5),
        ],
        num_qubits,
    )
    assert (qop - expected).simplify() == SparseObservable.zero(num_qubits)


def test_dk_edge_horizontal_left_with_face_above():
    hamil = MajoranaOperator.from_dict(
        {
            (gamma(13, False), gamma(13, True), gamma(13, False), gamma(14, False)): 2.5j,
            (gamma(14, False), gamma(14, True), gamma(13, False), gamma(14, False)): -2.5j,
        }
    )
    initial_state = [True, False] * 8
    edge_face_map = build_derby_klassen_edge_face_map_4x4()
    num_qubits = 20
    qop = derby_klassen(hamil, initial_state, edge_face_map, num_qubits)
    expected = SparseObservable.from_sparse_list(
        [
            ("XXY", (13, 14, 19), -2.5),
            ("YYY", (13, 14, 19), 2.5),
        ],
        num_qubits,
    )
    assert (qop - expected).simplify() == SparseObservable.zero(num_qubits)


def test_dk_edge_horizontal_left_with_face_below():
    hamil = MajoranaOperator.from_dict(
        {
            (gamma(4, False), gamma(4, True), gamma(4, False), gamma(5, False)): 2.5j,
            (gamma(5, False), gamma(5, True), gamma(4, False), gamma(5, False)): -2.5j,
        }
    )
    initial_state = [True, False] * 8
    edge_face_map = build_derby_klassen_edge_face_map_4x4()
    num_qubits = 20
    qop = derby_klassen(hamil, initial_state, edge_face_map, num_qubits)
    expected = SparseObservable.from_sparse_list(
        [
            ("XXY", (4, 5, 18), 2.5),
            ("YYY", (4, 5, 18), -2.5),
        ],
        num_qubits,
    )
    assert (qop - expected).simplify() == SparseObservable.zero(num_qubits)


def test_dk_edge_vertical_up():
    hamil = MajoranaOperator.from_dict(
        {
            (gamma(7, False), gamma(7, True), gamma(7, False), gamma(0, False)): 2.5j,
            (gamma(0, False), gamma(0, True), gamma(7, False), gamma(0, False)): -2.5j,
        }
    )
    initial_state = [True, False] * 8
    edge_face_map = build_derby_klassen_edge_face_map_4x4()
    num_qubits = 20
    qop = derby_klassen(hamil, initial_state, edge_face_map, num_qubits)
    expected = SparseObservable.from_sparse_list(
        [
            ("XX", (0, 7), 2.5),
            ("YY", (0, 7), -2.5),
        ],
        num_qubits,
    )
    assert (qop - expected).simplify() == SparseObservable.zero(num_qubits)


def test_dk_edge_vertical_up_with_face_left():
    hamil = MajoranaOperator.from_dict(
        {
            (gamma(5, False), gamma(5, True), gamma(5, False), gamma(2, False)): 2.5j,
            (gamma(2, False), gamma(2, True), gamma(5, False), gamma(2, False)): -2.5j,
        }
    )
    initial_state = [True, False] * 8
    edge_face_map = build_derby_klassen_edge_face_map_4x4()
    num_qubits = 20
    qop = derby_klassen(hamil, initial_state, edge_face_map, num_qubits)
    expected = SparseObservable.from_sparse_list(
        [
            ("XXX", (2, 5, 16), 2.5),
            ("YYX", (2, 5, 16), -2.5),
        ],
        num_qubits,
    )
    assert (qop - expected).simplify() == SparseObservable.zero(num_qubits)


def test_dk_edge_vertical_up_with_face_right():
    hamil = MajoranaOperator.from_dict(
        {
            (gamma(10, False), gamma(10, True), gamma(10, False), gamma(5, False)): 2.5j,
            (gamma(5, False), gamma(5, True), gamma(10, False), gamma(5, False)): -2.5j,
        }
    )
    initial_state = [True, False] * 8
    edge_face_map = build_derby_klassen_edge_face_map_4x4()
    num_qubits = 20
    qop = derby_klassen(hamil, initial_state, edge_face_map, num_qubits)
    expected = SparseObservable.from_sparse_list(
        [
            ("XXX", (5, 10, 18), -2.5),
            ("YYX", (5, 10, 18), 2.5),
        ],
        num_qubits,
    )
    assert (qop - expected).simplify() == SparseObservable.zero(num_qubits)


def test_dk_edge_vertical_down():
    hamil = MajoranaOperator.from_dict(
        {
            (gamma(3, False), gamma(3, True), gamma(3, False), gamma(4, False)): 2.5j,
            (gamma(4, False), gamma(4, True), gamma(3, False), gamma(4, False)): -2.5j,
        }
    )
    initial_state = [True, False] * 8
    edge_face_map = build_derby_klassen_edge_face_map_4x4()
    num_qubits = 20
    qop = derby_klassen(hamil, initial_state, edge_face_map, num_qubits)
    expected = SparseObservable.from_sparse_list(
        [
            ("XX", (3, 4), -2.5),
            ("YY", (3, 4), 2.5),
        ],
        num_qubits,
    )
    assert (qop - expected).simplify() == SparseObservable.zero(num_qubits)


def test_dk_edge_vertical_down_with_face_right():
    hamil = MajoranaOperator.from_dict(
        {
            (gamma(1, False), gamma(1, True), gamma(1, False), gamma(6, False)): 2.5j,
            (gamma(6, False), gamma(6, True), gamma(1, False), gamma(6, False)): -2.5j,
        }
    )
    initial_state = [True, False] * 8
    edge_face_map = build_derby_klassen_edge_face_map_4x4()
    num_qubits = 20
    qop = derby_klassen(hamil, initial_state, edge_face_map, num_qubits)
    expected = SparseObservable.from_sparse_list(
        [
            ("XXX", (1, 6, 16), -2.5),
            ("YYX", (1, 6, 16), 2.5),
        ],
        num_qubits,
    )
    assert (qop - expected).simplify() == SparseObservable.zero(num_qubits)


def test_dk_edge_vertical_down_with_face_left():
    hamil = MajoranaOperator.from_dict(
        {
            (gamma(6, False), gamma(6, True), gamma(6, False), gamma(9, False)): 2.5j,
            (gamma(9, False), gamma(9, True), gamma(6, False), gamma(9, False)): -2.5j,
        }
    )
    initial_state = [True, False] * 8
    edge_face_map = build_derby_klassen_edge_face_map_4x4()
    num_qubits = 20
    qop = derby_klassen(hamil, initial_state, edge_face_map, num_qubits)
    expected = SparseObservable.from_sparse_list(
        [
            ("XXX", (6, 9, 17), 2.5),
            ("YYX", (6, 9, 17), -2.5),
        ],
        num_qubits,
    )
    assert (qop - expected).simplify() == SparseObservable.zero(num_qubits)


def test_custom_layout():
    hamil = build_fermi_hubbard_4x4(5.0, 5.0)
    initial_state = [True, False] * 8
    edge_face_map = build_derby_klassen_edge_face_map_4x4()
    qop = derby_klassen(hamil, initial_state, edge_face_map, 20)

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

    diff = (expected - pauli_op).simplify(atol=0.0)
    assert diff == SparsePauliOp.from_sparse_list([], 20)
