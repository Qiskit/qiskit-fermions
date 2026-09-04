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

"""Structural tests for the UCC ansatz gate.

The amplitudes and their conventions belong to ffsim, so the tests here cover only what this package
adds: reading the spin variant off the ffsim operator type, the cluster generator built from the
amplitudes, and the conjugate-paired grouping that keeps every factor of the product formula
unitary.
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit_fermions.circuit import FermionicCircuit
from qiskit_fermions.circuit.library import UCC

ffsim = pytest.importorskip("ffsim")


def _restricted_op(nocc, nvrt, *, seed, with_final=False):
    """Returns an ffsim restricted UCCSD operator with random real amplitudes."""
    rng = np.random.default_rng(seed)
    t1 = rng.standard_normal((nocc, nvrt)) * 0.05
    t2 = rng.standard_normal((nocc, nocc, nvrt, nvrt)) * 0.05
    t2 = 0.5 * (t2 + t2.transpose(1, 0, 3, 2))  # the symmetry the cluster operator sees
    final = None
    if with_final:
        norb = nocc + nvrt
        final = np.eye(norb)
    return ffsim.UCCSDOpRestrictedReal(t1=t1, t2=t2, final_orbital_rotation=final)


def _unrestricted_op(nocc_a, nocc_b, nvrt_a, nvrt_b, *, seed):
    """Returns an ffsim unrestricted UCCSD operator with random real amplitudes."""
    rng = np.random.default_rng(seed)
    t1 = (
        rng.standard_normal((nocc_a, nvrt_a)) * 0.05,
        rng.standard_normal((nocc_b, nvrt_b)) * 0.05,
    )
    t2aa = rng.standard_normal((nocc_a, nocc_a, nvrt_a, nvrt_a)) * 0.05
    t2bb = rng.standard_normal((nocc_b, nocc_b, nvrt_b, nvrt_b)) * 0.05
    t2ab = rng.standard_normal((nocc_a, nocc_b, nvrt_a, nvrt_b)) * 0.05
    t2aa = 0.5 * (t2aa + t2aa.transpose(1, 0, 3, 2))
    t2bb = 0.5 * (t2bb + t2bb.transpose(1, 0, 3, 2))
    return ffsim.UCCSDOpUnrestrictedReal(t1=t1, t2=(t2aa, t2ab, t2bb))


def test_ucc_restricted_reads_norb_off_the_operator():
    """A restricted operator gives a ``2 * norb``-mode gate."""
    op = _restricted_op(2, 2, seed=13)
    gate = UCC(op)
    assert gate.uccsd_op is op
    assert gate.norb == 4
    assert gate.num_modes == 8
    assert gate.uccsd_op.final_orbital_rotation is None


def test_ucc_unrestricted_reads_norb_off_the_operator():
    """An unrestricted operator gives a ``2 * norb``-mode gate with per-spin amplitudes."""
    gate = UCC(_unrestricted_op(2, 1, 2, 3, seed=19))
    assert gate.norb == 4
    assert len(gate.uccsd_op.t1) == 2
    assert len(gate.uccsd_op.t2) == 3


def test_ucc_accepts_the_complex_operator_flavors():
    """The non-``Real`` operator flavors are accepted too."""
    t1 = np.zeros((1, 1))
    t2 = np.zeros((1, 1, 1, 1))
    assert UCC(ffsim.UCCSDOpRestricted(t1=t1, t2=t2)).num_modes == 4
    assert (
        UCC(
            ffsim.UCCSDOpUnrestricted(t1=(t1, t1), t2=(t2, t2, t2)),
        ).num_modes
        == 4
    )


def test_ucc_rejects_a_non_ffsim_operator():
    """Anything other than one of ffsim's four UCCSD operators is rejected up front."""
    with pytest.raises(TypeError, match="requires one of ffsim's UCCSD operators"):
        UCC(np.zeros((2, 2)))


def test_ucc_cluster_operator_generator_is_anti_hermitian():
    """The cluster generator ``T - T^dagger`` is anti-Hermitian, so ``i (T - T^dagger)`` is Hermitian.

    This is exactly the property :meth:`.UCC._build_definition` relies on to express the ansatz as an
    :class:`.Evolution`, whose operator must be Hermitian for the evolution to be unitary.
    """
    generator = UCC(_restricted_op(2, 2, seed=13)).cluster_operator()
    assert (generator * 1j).is_hermitian()


def test_ucc_cluster_operator_conserves_sector():
    """The cluster generator conserves the particle number of each spin species.

    Every excitation replaces an occupied orbital with a virtual one *within* a spin sector, so the
    generator must preserve both the total particle number and the z-component of spin, the
    condition :meth:`.Evolution._apply_unitary_placed_` enforces before simulating.
    """
    gate = UCC(_restricted_op(2, 2, seed=14))
    generator = gate.cluster_operator()
    assert generator.conserves_particle_number()
    assert generator.conserves_sector([gate.norb, gate.norb])


def test_ucc_generator_groups_are_individually_hermitian():
    """Every group of the Hermitian generator is itself Hermitian, a regression guard.

    :class:`.Evolution` decomposes group-by-group, so each group becomes one factor
    ``exp(-i H_k)`` of the product formula. A factor is unitary only if its ``H_k`` is Hermitian.
    Splitting ``i (T - T^dagger)`` *term*-by-term instead yields non-Hermitian factors (each
    excitation is separated from its conjugate), which makes the synthesized circuit non-unitary:
    it does not even preserve the norm. The generator therefore pairs every excitation with its
    conjugate in a shared group, which this test locks in.
    """
    generator = UCC(_restricted_op(2, 2, seed=18)).cluster_operator() * 1j

    assert generator.is_hermitian()
    assert generator.has_groups()
    assert generator.num_groups() > 1  # genuinely grouped, not one lump
    for group in generator.split_out_groups():
        assert group.is_hermitian()


def test_ucc_generator_groups_are_individually_hermitian_unrestricted():
    """The conjugate-pairing group invariant also holds for the unrestricted variant."""
    generator = UCC(_unrestricted_op(2, 1, 2, 3, seed=19)).cluster_operator() * 1j

    assert generator.is_hermitian()
    for group in generator.split_out_groups():
        assert group.is_hermitian()


def test_ucc_generator_group_layout_is_canonical():
    """The group layout is sorted by mode support, so it cannot depend on term-iteration order.

    The group index decides where in the product formula :class:`.Evolution` places that group's
    factor. Since the excitations do not commute, that placement changes the Trotter error, so it must
    be a function of the amplitudes alone. The indices were once assigned in first-encounter order over
    a ``frozenset``-keyed dict, which tied them to element hashing and hence to ``PYTHONHASHSEED``: the
    synthesized circuit's Trotter error then differed between processes, surfacing as an intermittent
    failure of ``test_ucc_trotterized_circuit_converges_to_the_exact_gate``. Asserting the layout is
    sorted pins that down without needing to vary the hash seed, which a single process cannot do.
    """
    generator = UCC(_restricted_op(2, 2, seed=23)).cluster_operator()

    support_by_group: dict[int, tuple[int, ...]] = {}
    for (actions, _), group in zip(
        generator.iter_terms(), generator.groups, strict=True
    ):  # pragma: no branch
        support = tuple(sorted(mode for _, mode in actions))
        # a term and its conjugate share the mode multiset, so a group has exactly one support
        assert support_by_group.setdefault(group, support) == support

    layout = [support_by_group[group] for group in sorted(support_by_group)]
    assert layout == sorted(layout), layout


def test_ucc_definition_is_a_single_evolution():
    """The gate's definition is one :class:`.Evolution` carrying the whole cluster generator.

    Keeping the generator in a single ``Evolution`` (rather than pre-splitting it) is what lets the
    simulation path exponentiate it exactly while leaving the Trotter decomposition to the
    transpiler.
    """
    gate = UCC(_restricted_op(1, 2, seed=16))
    circuit = FermionicCircuit(gate.num_modes)
    circuit.append(gate, circuit.modes)
    assert dict(circuit.decompose().count_ops()) == {"Evolution": 1}


def test_ucc_definition_decomposes_into_group_evolutions():
    """Decomposing the definition's ``Evolution`` splits it into one evolution per group."""
    gate = UCC(_restricted_op(1, 2, seed=17))
    circuit = FermionicCircuit(gate.num_modes)
    circuit.append(gate, circuit.modes)
    counts = dict(circuit.decompose().decompose().count_ops())
    assert counts["Evolution"] > 1


def test_ucc_stays_hermitian_at_any_decomposition_depth():
    """Every emitted factor must stay Hermitian, so that its exponential stays unitary.

    The cluster generator carries conjugate-paired groups precisely so that each factor is Hermitian.
    Decomposing past those groups used to split them term-by-term, and an individual excitation is
    *not* Hermitian on its own, which produced complex Pauli coefficients that the transpiler
    rejected, and a non-normalized state vector in simulation.
    """
    gate = UCC(_restricted_op(1, 2, seed=18))
    circuit = FermionicCircuit(gate.num_modes)
    circuit.append(gate, circuit.modes)

    at_two_levels = dict(circuit.decompose(reps=2).count_ops())
    for reps in (3, 4, 6):
        decomposed = circuit.decompose(reps=reps)
        assert dict(decomposed.count_ops()) == at_two_levels, f"still expanding at reps={reps}"
        for instruction in decomposed._inner.data:
            operator = instruction.operation.operator
            assert operator.is_hermitian(), f"non-Hermitian factor at reps={reps}"


def test_ucc_final_orbital_rotation_is_appended_per_spin_sector():
    """A wrapped operator's final orbital rotation becomes a closing per-spin OrbitalRotation."""
    gate = UCC(_restricted_op(1, 1, seed=21, with_final=True))
    circuit = FermionicCircuit(gate.num_modes)
    circuit.append(gate, circuit.modes)
    counts = dict(circuit.decompose().count_ops())
    assert counts["Evolution"] == 1
    assert counts["OrbitalRotation"] == 2
