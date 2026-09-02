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

import numpy as np
import pytest
from qiskit.quantum_info import SparseObservable
from qiskit_fermions.mappers.library import (
    edge_vertex_jordan_wigner,
    edge_vertex_to_fermion,
    fermion_jordan_wigner,
    jordan_wigner,
    majorana_jordan_wigner,
    majorana_to_fermion,
    transfer_vertex_jordan_wigner,
    transfer_vertex_to_fermion,
)
from qiskit_fermions.operators import (
    EdgeVertexOperator,
    FermionOperator,
    MajoranaOperator,
    TransferVertexOperator,
)


def test_fermion_jordan_wigner():
    num_qubits = 4
    op = FermionOperator.from_dict(
        {
            ((True, 0), (True, 1), (False, 0), (False, 1)): -0.4836505304710653,
            ((True, 0), (True, 2), (False, 0), (False, 2)): -0.6757101548035165,
            ((True, 0), (True, 3), (False, 0), (False, 3)): -0.6645817302552967,
            ((True, 0), (True, 2), (False, 1), (False, 3)): -0.18093119978423133,
            ((True, 0), (True, 3), (False, 1), (False, 2)): -0.18093119978423133,
            ((True, 1), (True, 2), (False, 0), (False, 3)): -0.18093119978423133,
            ((True, 1), (True, 3), (False, 0), (False, 2)): -0.18093119978423133,
            ((True, 1), (True, 2), (False, 1), (False, 2)): -0.6645817302552967,
            ((True, 1), (True, 3), (False, 1), (False, 3)): -0.6985737227320183,
            ((True, 2), (True, 3), (False, 2), (False, 3)): -0.4836505304710653,
            ((True, 0), (False, 0)): -1.2563390730032502,
            ((True, 1), (False, 1)): -0.4718960072811406,
            ((True, 2), (False, 2)): -1.2563390730032502,
            ((True, 3), (False, 3)): -0.4718960072811406,
        }
    )
    qop = fermion_jordan_wigner(op, num_qubits)
    assert isinstance(qop, SparseObservable)
    expected = SparseObservable.from_sparse_list(
        [
            ("", [], -0.8105479805373266),
            ("Z", [0], 0.1721839326191555),
            ("Z", [1], -0.22575349222402474),
            ("Z", [2], 0.17218393261915543),
            ("Z", [3], -0.22575349222402474),
            ("ZZ", [0, 1], 0.12091263261776633),
            ("ZZ", [0, 2], 0.16892753870087912),
            ("ZZ", [0, 3], 0.16614543256382416),
            ("YYYY", [0, 1, 2, 3], 0.04523279994605783),
            ("YYXX", [0, 1, 2, 3], 0.04523279994605783),
            ("XXYY", [0, 1, 2, 3], 0.04523279994605783),
            ("XXXX", [0, 1, 2, 3], 0.04523279994605783),
            ("ZZ", [1, 2], 0.16614543256382416),
            ("ZZ", [1, 3], 0.17464343068300459),
            ("ZZ", [2, 3], 0.12091263261776633),
        ],
        num_qubits,
    )
    diff = (qop - expected).simplify()
    assert diff == SparseObservable.zero(num_qubits)


def test_fermion_jordan_wigner_num_qubits_too_small():
    # an operator acting on mode index 3 requires at least 4 qubits; too few qubits must raise a
    # catchable ValueError instead of aborting the interpreter
    op = FermionOperator.from_dict({((True, 3),): 1.0})
    with pytest.raises(ValueError):
        fermion_jordan_wigner(op, 3)

    # exactly enough qubits succeeds
    assert isinstance(fermion_jordan_wigner(op, 4), SparseObservable)


def test_jordan_wigner_dispatches_fermion():
    # the type-agnostic entry point must delegate FermionOperator inputs to fermion_jordan_wigner
    op = FermionOperator.from_dict({((True, 0), (False, 0)): 0.1, ((True, 1), (False, 1)): -0.2})
    dispatched = jordan_wigner(op, 3)
    direct = fermion_jordan_wigner(op, 3)
    assert isinstance(dispatched, SparseObservable)
    assert (dispatched - direct).simplify() == SparseObservable.zero(3)


def test_fermion_jordan_wigner_merges_duplicate_terms():
    # The mapper accumulates each mapped term into per-thread observables using an addition that
    # concatenates rather than merges. Without periodic canonicalization those accumulators grow
    # with the number of *emitted* Pauli terms instead of the number of *distinct* ones, which for
    # an electronic-structure Hamiltonian is an asymptotic rather than a constant factor.
    #
    # Use a two-body operator big enough for the duplication to dominate: every two-body term maps
    # to 16 Pauli products, and terms sharing a mode set collapse onto the same Pauli strings.
    norb = 6
    npair = norb * (norb + 1) // 2
    rng = np.random.default_rng(1234)
    two_body = rng.random(npair * (npair + 1) // 2)
    op = FermionOperator.from_2body_tril_spin_sym(two_body, norb)

    qop = fermion_jordan_wigner(op, 2 * norb)
    simplified = qop.simplify(1e-12)

    # The returned operator is not promised to be fully simplified, and how many duplicates survive
    # depends on how the terms were spread over the worker threads, so the exact count varies with the
    # number of cores available.
    #
    # What must hold is that it does not scale with the number of Pauli terms *emitted*, which is what
    # regressed. Anchor on that count rather than on `simplified.num_terms`: the two are compared at
    # different tolerances (the mapper merges at 1e-18), so a ratio between them is satisfied in part
    # by that gap rather than by how well compaction worked. Every two-body term maps onto 16 Pauli
    # products, so the emitted total is a hard upper bound that the unfixed mapper actually reached.
    emitted = 16 * len(op)
    assert qop.num_terms < emitted / 8, (
        f"got {qop.num_terms} terms, close to the {emitted} emitted: duplicates are not being merged"
    )

    # Compaction must not change the operator, only its representation.
    assert (qop - simplified).simplify(1e-12) == SparseObservable.zero(2 * norb)


def test_fermion_jordan_wigner_merges_identity_terms():
    # An identity Pauli term carries no bit terms at all, so measuring the accumulators by their bit
    # term count alone scored these as free and never merged them however many piled up -- the very
    # blowup that compaction exists to prevent, just in the one shape that measure misses.
    num_repeats = 1 << 20
    op = FermionOperator.from_terms([((), 1.0)] * num_repeats)

    qop = fermion_jordan_wigner(op, 2)
    assert qop.num_terms <= 2, (
        f"got {qop.num_terms} terms from {num_repeats} identity terms: they are not being merged"
    )

    # ... and the result must still be `num_repeats * I`.
    expected = SparseObservable.identity(2) * float(num_repeats)
    assert (qop - expected).simplify(1e-9) == SparseObservable.zero(2)


def test_jordan_wigner_unsupported_type_raises():
    # All four operator types now have a direct implementation, so the guard is exercised with
    # something that is not a fermionic operator at all. It must raise rather than fail obscurely
    # deeper in, and it is what any future operator type will land on until it gains one.
    with pytest.raises(TypeError, match="SparseObservable"):
        jordan_wigner(SparseObservable.identity(2), 2)


# The mapped image of every generator below is a *single* Pauli string, unlike a fermionic action
# which maps onto a two-term sum. Each expectation is therefore one sparse-list entry, hand-computed
# from the definitions in the corresponding docstring.


def test_majorana_jordan_wigner():
    num_qubits = 3
    # gamma_0 gamma_3 = (X_0)(Z_0 Y_1): the two images overlap on qubit 0, where X * Z = -i Y, so the
    # product is a weight-2 string carrying that phase rather than the naive X_0 Y_1.
    #
    # gamma_4 keeps its full Z-string, since nothing cancels against it.
    op = MajoranaOperator.from_dict({(0, 3): 1.0, (4,): 0.5})
    qop = majorana_jordan_wigner(op, num_qubits)
    assert isinstance(qop, SparseObservable)
    expected = SparseObservable.from_sparse_list(
        [
            ("YY", [0, 1], -1.0j),
            ("ZZX", [0, 1, 2], 0.5),
        ],
        num_qubits,
    )
    assert (qop - expected).simplify() == SparseObservable.zero(num_qubits)


def test_edge_vertex_jordan_wigner():
    num_qubits = 4
    # A vertex is a bare Z; an edge is Y..X with the Z-string spanning only the modes strictly
    # between its endpoints, and its sign flips with the index order (E_lr = -E_rl).
    op = EdgeVertexOperator.from_dict({((2, 2),): 2.0, ((0, 3),): 0.5, ((3, 0),): 0.5})
    qop = edge_vertex_jordan_wigner(op, num_qubits)
    assert isinstance(qop, SparseObservable)
    expected = SparseObservable.from_sparse_list(
        [
            ("Z", [2], 2.0),
            ("YZZX", [0, 1, 2, 3], -0.5),
            ("YZZX", [0, 1, 2, 3], 0.5),
        ],
        num_qubits,
    )
    assert (qop - expected).simplify() == SparseObservable.zero(num_qubits)


def test_transfer_vertex_jordan_wigner():
    num_qubits = 3
    # The coefficient is -1/2 for *both* orientations; it is the Pauli letters that swap, which is
    # the opposite of how the edge operator behaves.
    op = TransferVertexOperator.from_dict({((1, 1),): 2.0, ((0, 2),): 1.0, ((2, 0),): 1.0})
    qop = transfer_vertex_jordan_wigner(op, num_qubits)
    assert isinstance(qop, SparseObservable)
    expected = SparseObservable.from_sparse_list(
        [
            ("Z", [1], 2.0),
            ("XZX", [0, 1, 2], -0.5),
            ("YZY", [0, 1, 2], -0.5),
        ],
        num_qubits,
    )
    assert (qop - expected).simplify() == SparseObservable.zero(num_qubits)


@pytest.mark.parametrize(
    "mapper,op,num_qubits",
    [
        # gamma_7 acts on fermionic mode 3, so it needs 4 qubits rather than 8: the bound for a
        # MajoranaOperator is counted in fermionic modes.
        (majorana_jordan_wigner, MajoranaOperator.from_dict({(7,): 1.0}), 3),
        # The largest index sits in the *right* buffer, which a left-only check would miss.
        (edge_vertex_jordan_wigner, EdgeVertexOperator.from_dict({((0, 3),): 1.0}), 3),
        (transfer_vertex_jordan_wigner, TransferVertexOperator.from_dict({((0, 3),): 1.0}), 3),
    ],
)
def test_jordan_wigner_num_qubits_too_small(mapper, op, num_qubits):
    # too few qubits must raise a catchable ValueError instead of aborting the interpreter
    with pytest.raises(ValueError):
        mapper(op, num_qubits)

    # exactly enough qubits succeeds
    assert isinstance(mapper(op, num_qubits + 1), SparseObservable)


@pytest.mark.parametrize(
    "direct,op",
    [
        (majorana_jordan_wigner, MajoranaOperator.from_dict({(0, 3): 1.0, (2, 5, 1): -0.5j})),
        (
            edge_vertex_jordan_wigner,
            EdgeVertexOperator.from_dict({((0, 0),): 2.0, ((0, 2),): 0.5, ((2, 0),): -1.0j}),
        ),
        (
            transfer_vertex_jordan_wigner,
            TransferVertexOperator.from_dict(
                {((1, 1),): 2.0, ((0, 2),): 0.5, ((2, 0),): -1.0j, ((1, 2), (0, 1)): 0.25}
            ),
        ),
    ],
)
def test_jordan_wigner_dispatches_to_direct_implementation(direct, op):
    # the type-agnostic entry point must delegate each operator type to its direct implementation
    num_qubits = 3
    dispatched = jordan_wigner(op, num_qubits)
    assert isinstance(dispatched, SparseObservable)
    assert (dispatched - direct(op, num_qubits)).simplify() == SparseObservable.zero(num_qubits)


@pytest.mark.parametrize(
    "direct,to_fermion,op",
    [
        (
            majorana_jordan_wigner,
            majorana_to_fermion,
            # An odd-length term is not Hermitian and is where a stray factor of i would hide; a
            # repeated index makes the Z-strings cancel to the identity; the complex coefficient
            # catches a conjugation error; the empty term maps onto the identity.
            MajoranaOperator.from_dict(
                {(0,): 1.0, (0, 3): -0.5 + 2.0j, (2, 5, 1): 1.0, (1, 1): 0.75, (): 1.25}
            ),
        ),
        (
            edge_vertex_jordan_wigner,
            edge_vertex_to_fermion,
            # Both orderings of the same pair appear together, so an antisymmetry error cannot hide
            # by being present on both sides of the comparison.
            EdgeVertexOperator.from_dict(
                {((0, 0),): 2.0, ((0, 2),): 0.5, ((2, 0),): -1.0j, ((1, 1), (1, 2)): 0.25}
            ),
        ),
        (
            transfer_vertex_jordan_wigner,
            transfer_vertex_to_fermion,
            TransferVertexOperator.from_dict(
                {((1, 1),): 2.0, ((0, 1),): 0.5, ((1, 0),): -1.0j, ((1, 2), (0, 1)): 0.25}
            ),
        ),
    ],
)
def test_jordan_wigner_matches_route_via_fermion(direct, to_fermion, op):
    # Cross-validates the direct Pauli images against converting to a FermionOperator first. The two
    # routes share no code, so a sign error in the image tables cannot cancel out -- which is what
    # makes this the load-bearing check on the algebra rather than a redundancy test.
    #
    # The converter route is *not* the production path precisely because each fermionic action maps
    # onto a two-term sum, inflating a single Pauli string into up to 4**L terms for a length-L term.
    # These operators are kept small so that blowup stays cheap.
    num_qubits = 3
    via_fermion = fermion_jordan_wigner(to_fermion(op), num_qubits)
    assert (direct(op, num_qubits) - via_fermion).simplify(1e-12) == SparseObservable.zero(
        num_qubits
    )
