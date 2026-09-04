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

"""Tests for applying a UCC gate to an ffsim state vector (SupportsApplyUnitary).

The two spinful variants are validated against ffsim's own ``UCCSDOpRestrictedReal`` /
``UCCSDOpUnrestrictedReal``: we take the amplitudes from a random ffsim UCCSD operator, rebuild the
ansatz with our :class:`.UCC` gate, and require the resulting state vectors to agree. This pins down
both the mode convention (ffsim's interleaved ``(orb, spin)`` versus our block-spin register) and the
per-block prefactors of the cluster operator. A separate group of tests uses a directly exponentiated
cluster generator as the oracle, which is independent of the gate's ``Evolution``-based definition.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse.linalg
from qiskit_fermions.circuit import FermionicCircuit
from qiskit_fermions.circuit.library import UCC

ffsim = pytest.importorskip("ffsim")


def _restricted_amplitudes(nocc, nvrt, *, seed):
    """Returns random real ``(t1, t2)`` amplitudes with the restricted exchange symmetry.

    ffsim's ``UCCSDOpRestrictedReal`` only accepts real amplitudes, and its parameterization assumes
    ``t2[i, j, a, b] == t2[j, i, b, a]``; symmetrizing here keeps the operator inside the family both
    implementations describe.
    """
    rng = np.random.default_rng(seed)
    t1 = rng.standard_normal((nocc, nvrt))
    t2 = rng.standard_normal((nocc, nocc, nvrt, nvrt))
    return t1, t2 + t2.transpose(1, 0, 3, 2)


def _unrestricted_amplitudes(nocc_a, nocc_b, nvrt_a, nvrt_b, *, seed):
    """Returns random real unrestricted amplitudes with the per-block exchange symmetry."""
    rng = np.random.default_rng(seed)
    t1a = rng.standard_normal((nocc_a, nvrt_a))
    t1b = rng.standard_normal((nocc_b, nvrt_b))
    t2aa = rng.standard_normal((nocc_a, nocc_a, nvrt_a, nvrt_a))
    t2bb = rng.standard_normal((nocc_b, nocc_b, nvrt_b, nvrt_b))
    t2ab = rng.standard_normal((nocc_a, nocc_b, nvrt_a, nvrt_b))
    t2aa = t2aa + t2aa.transpose(1, 0, 3, 2)
    t2bb = t2bb + t2bb.transpose(1, 0, 3, 2)
    return (t1a, t1b), (t2aa, t2ab, t2bb)


def test_ucc_restricted_matches_ffsim():
    """The restricted UCC gate reproduces ffsim's UCCSDOpRestrictedReal state vector."""
    norb, nocc = 4, 2
    nelec = (nocc, nocc)
    t1, t2 = _restricted_amplitudes(nocc, norb - nocc, seed=1234)

    ucc_op = ffsim.UCCSDOpRestrictedReal(t1=t1, t2=t2)
    reference = ffsim.hartree_fock_state(norb, nelec)
    expected = ffsim.apply_unitary(reference, ucc_op, norb=norb, nelec=nelec)

    gate = UCC(ffsim.UCCSDOpRestrictedReal(t1=t1, t2=t2))
    result = ffsim.apply_unitary(reference, gate, norb=norb, nelec=nelec)
    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_ucc_unrestricted_matches_ffsim():
    """The unrestricted UCC gate reproduces ffsim's UCCSDOpUnrestrictedReal state vector.

    The alpha and beta sectors deliberately carry *different* occupations, which exercises the
    per-spin virtual-orbital offset: using a single shared ``nocc`` for both sectors would place the
    beta excitations on the wrong modes.
    """
    norb = 4
    nocc_a, nocc_b = 2, 1
    nelec = (nocc_a, nocc_b)
    t1, t2 = _unrestricted_amplitudes(nocc_a, nocc_b, norb - nocc_a, norb - nocc_b, seed=2024)

    ucc_op = ffsim.UCCSDOpUnrestrictedReal(t1=t1, t2=t2)
    reference = ffsim.hartree_fock_state(norb, nelec)
    expected = ffsim.apply_unitary(reference, ucc_op, norb=norb, nelec=nelec)

    gate = UCC(ffsim.UCCSDOpUnrestrictedReal(t1=t1, t2=t2))
    result = ffsim.apply_unitary(reference, gate, norb=norb, nelec=nelec)
    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_ucc_restricted_doubles_only_matches_ffsim():
    """A doubles-only (UCCD) ansatz built via ``from_t_amplitudes`` matches ffsim with zero ``t1``."""
    norb, nocc = 4, 2
    nelec = (nocc, nocc)
    _, t2 = _restricted_amplitudes(nocc, norb - nocc, seed=31)

    ucc_op = ffsim.UCCSDOpRestrictedReal(t1=np.zeros((nocc, norb - nocc)), t2=t2)
    reference = ffsim.hartree_fock_state(norb, nelec)
    expected = ffsim.apply_unitary(reference, ucc_op, norb=norb, nelec=nelec)

    gate = UCC(ffsim.UCCSDOpRestrictedReal(t1=np.zeros((t2.shape[0], t2.shape[2])), t2=t2))
    result = ffsim.apply_unitary(reference, gate, norb=norb, nelec=nelec)
    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_ucc_gate_apply_unitary_matches_ffsim():
    """The bare gate's ``_apply_unitary_`` applied to the HF reference matches ffsim's operator.

    UCC is a pure unitary carrying no reference: the caller supplies the reference state, exactly as
    ffsim's own operator does.
    """
    norb, nocc = 4, 2
    nelec = (nocc, nocc)
    t1, t2 = _restricted_amplitudes(nocc, norb - nocc, seed=7)

    ucc_op = ffsim.UCCSDOpRestrictedReal(t1=t1, t2=t2)
    reference = ffsim.hartree_fock_state(norb, nelec)
    expected = ffsim.apply_unitary(reference, ucc_op, norb=norb, nelec=nelec)

    gate = UCC(ffsim.UCCSDOpRestrictedReal(t1=t1, t2=t2))
    result = gate._apply_unitary_(reference.copy(), norb, nelec, copy=True)
    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_ucc_gate_through_circuit_matches_ffsim():
    """A UCC gate appended to a FermionicCircuit and driven by ffsim.apply_unitary matches ffsim."""
    norb, nocc = 4, 2
    nelec = (nocc, nocc)
    t1, t2 = _restricted_amplitudes(nocc, norb - nocc, seed=8)

    ucc_op = ffsim.UCCSDOpRestrictedReal(t1=t1, t2=t2)
    reference = ffsim.hartree_fock_state(norb, nelec)
    expected = ffsim.apply_unitary(reference, ucc_op, norb=norb, nelec=nelec)

    gate = UCC(ffsim.UCCSDOpRestrictedReal(t1=t1, t2=t2))
    circ = FermionicCircuit(2 * norb)
    circ.append(gate, circ.modes)
    result = ffsim.apply_unitary(reference, circ, norb=norb, nelec=nelec)
    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_ucc_matches_directly_exponentiated_generator():
    """The gate matches ``exp(T - T^dagger)`` applied via scipy directly.

    The oracle here is the gate's own cluster generator exponentiated independently of the gate's
    ``Evolution``-based definition, which confirms that the ``e^{T - T^dagger} == e^{-i H}`` rewrite
    with ``H = i (T - T^dagger)`` carries the right sign.
    """
    norb, nocc = 4, 2
    nelec = (2, 2)
    t1, t2 = _restricted_amplitudes(nocc, norb - nocc, seed=11)

    gate = UCC(ffsim.UCCSDOpRestrictedReal(t1=t1, t2=t2))
    reference = ffsim.hartree_fock_state(norb, nelec)

    linop = ffsim.linear_operator(gate.cluster_operator(), norb, nelec)
    expected = scipy.sparse.linalg.expm_multiply(linop, reference, traceA=0.0)

    result = ffsim.apply_unitary(reference, gate, norb=norb, nelec=nelec)
    np.testing.assert_allclose(np.linalg.norm(result), 1.0, atol=1e-10)
    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_ucc_restricted_matches_directly_exponentiated_generator():
    """The restricted gate also matches its own generator exponentiated directly.

    A cross-check independent of ffsim's UCCSD operator: it validates the
    ``e^{T - T^dagger} == e^{-i H}`` rewrite in the spinful block-spin register, where an incorrect
    sign or a missing factor of ``i`` would leave the norm intact but rotate the state differently.
    """
    norb, nocc = 3, 2
    nelec = (nocc, nocc)
    t1, t2 = _restricted_amplitudes(nocc, norb - nocc, seed=12)

    gate = UCC(ffsim.UCCSDOpRestrictedReal(t1=t1, t2=t2))
    reference = ffsim.hartree_fock_state(norb, nelec)

    linop = ffsim.linear_operator(gate.cluster_operator(), norb, nelec)
    expected = scipy.sparse.linalg.expm_multiply(linop, reference, traceA=0.0)

    result = ffsim.apply_unitary(reference, gate, norb=norb, nelec=nelec)
    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_ucc_gate_subset_placement_matches_global_embedding():
    """A UCC placed on a subset of a larger register acts on its absolute (global) modes.

    A 4-mode UCC is appended to a larger register on the non-identity subset ``placement``. The
    oracle is the *same* cluster generator relabeled onto those global modes and exponentiated
    directly. This guards the ``_apply_unitary_placed_`` routing: with the placement ignored, the
    gate would act on the leading modes instead of ``placement``.
    """
    norb_global = 6
    nelec = (2, 1)
    # the gate's 4 block-spin modes land on global alpha modes {1, 2} and beta modes {norb+0, norb+2}
    placement = [1, 2, norb_global + 0, norb_global + 2]

    t1, t2 = _restricted_amplitudes(1, 1, seed=909)
    gate = UCC(ffsim.UCCSDOpRestrictedReal(t1=t1, t2=t2))
    assert gate.num_modes == 4

    reference = ffsim.hartree_fock_state(norb_global, nelec)

    # oracle: the same generator, relabeled from local modes onto the placed global modes
    relabeled = gate.cluster_operator().relabel_modes(placement)
    linop = ffsim.linear_operator(relabeled, norb_global, nelec)
    expected = scipy.sparse.linalg.expm_multiply(linop, reference, traceA=0.0)

    circ = FermionicCircuit(2 * norb_global)
    circ.append(gate, [circ.modes[p] for p in placement])
    result = ffsim.apply_unitary(reference, circ, norb=norb_global, nelec=nelec)

    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_ucc_trotterized_circuit_converges_to_the_exact_gate():
    """The synthesized (Trotterized) circuit approaches the exactly-applied gate as reps increase.

    The gate's definition is a first-order product formula, so the circuit and the exact
    ``expm_multiply`` path agree only up to Trotter error. Splitting the ansatz into ``n`` repetitions
    of ``1/n`` of the amplitudes must reduce that error -- confirming the two paths describe the *same*
    unitary and that the definition's term ordering carries no systematic error beyond Trotterization.
    """
    norb, nocc = 3, 2
    nelec = (nocc, nocc)
    t1, t2 = _restricted_amplitudes(nocc, norb - nocc, seed=17)

    reference = ffsim.hartree_fock_state(norb, nelec)
    exact = ffsim.apply_unitary(
        reference, UCC(ffsim.UCCSDOpRestrictedReal(t1=t1, t2=t2)), norb=norb, nelec=nelec
    )

    errors = []
    for n_reps in (1, 4, 16, 64):
        # n_reps repetitions of a 1/n_reps-strength ansatz, each Trotterized group-by-group
        step = UCC(ffsim.UCCSDOpRestrictedReal(t1=t1 / n_reps, t2=t2 / n_reps))
        circ = FermionicCircuit(2 * norb)
        for _ in range(n_reps):
            circ.append(step, circ.modes)
        # decompose twice to reach the individual per-group evolutions (the product formula)
        vec = ffsim.apply_unitary(reference, circ.decompose().decompose(), norb=norb, nelec=nelec)
        # each product-formula factor is a genuine unitary, so the norm is preserved exactly -- this
        # is what the conjugate-pairing groups of the generator buy us
        np.testing.assert_allclose(np.linalg.norm(vec), 1.0, atol=1e-10)
        errors.append(float(np.abs(vec - exact).max()))

    # strictly decreasing, and roughly first-order: quadrupling the reps cuts the error ~4x. The rate
    # is asserted rather than an absolute floor on the last error, because the achievable error at a
    # fixed rep count depends on the order the product formula applies the groups in -- an ordering
    # detail this test has no business pinning. The rate is the property that actually witnesses
    # "same unitary, first-order splitting".
    assert errors[0] > errors[1] > errors[2] > errors[3], errors
    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    assert all(ratio > 2.5 for ratio in ratios), (errors, ratios)
    # the asymptotic ratio approaches 4; the coarsest steps are not yet in that regime
    assert ratios[-1] > 3.5, (errors, ratios)
