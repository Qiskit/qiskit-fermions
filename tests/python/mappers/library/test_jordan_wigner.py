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
from qiskit_fermions.mappers.library import fermion_jordan_wigner, jordan_wigner
from qiskit_fermions.operators import FermionOperator, MajoranaOperator


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
    # any operator type without a direct implementation must raise TypeError (not silently fail)
    maj_op = MajoranaOperator.from_dict({(0, 1): 1.0})
    with pytest.raises(TypeError, match="MajoranaOperator"):
        jordan_wigner(maj_op, 2)
