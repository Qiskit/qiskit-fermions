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

"""Structural tests for the UCC ansatz gate."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit_fermions.circuit import FermionicCircuit
from qiskit_fermions.circuit.library import UCC


def _restricted_amplitudes(nocc, nvrt, *, seed):
    """Returns random ``(t1, t2)`` amplitudes with the restricted ``(i, a) <-> (j, b)`` symmetry."""
    rng = np.random.default_rng(seed)
    t1 = rng.standard_normal((nocc, nvrt))
    t2 = rng.standard_normal((nocc, nocc, nvrt, nvrt))
    return t1, t2 + t2.transpose(1, 0, 3, 2)


def _unrestricted_amplitudes(nocc_a, nocc_b, nvrt_a, nvrt_b, *, seed):
    """Returns random unrestricted ``((t1a, t1b), (t2aa, t2ab, t2bb))`` amplitudes."""
    rng = np.random.default_rng(seed)
    t1a = rng.standard_normal((nocc_a, nvrt_a))
    t1b = rng.standard_normal((nocc_b, nvrt_b))
    t2aa = rng.standard_normal((nocc_a, nocc_a, nvrt_a, nvrt_a))
    t2bb = rng.standard_normal((nocc_b, nocc_b, nvrt_b, nvrt_b))
    t2ab = rng.standard_normal((nocc_a, nocc_b, nvrt_a, nvrt_b))
    t2aa = t2aa + t2aa.transpose(1, 0, 3, 2)
    t2bb = t2bb + t2bb.transpose(1, 0, 3, 2)
    return (t1a, t1b), (t2aa, t2ab, t2bb)


def test_ucc_restricted_variant():
    """Restricted amplitudes act on ``2 * norb`` block-spin modes."""
    nocc, nvrt = 2, 2
    t1, t2 = _restricted_amplitudes(nocc, nvrt, seed=0)
    gate = UCC("restricted", t1, t2)
    assert gate.norb == nocc + nvrt
    assert gate.num_modes == 2 * (nocc + nvrt)
    assert gate._variant is UCC.Variant.RESTRICTED


def test_ucc_unrestricted_variant():
    """Unrestricted amplitudes act on ``2 * norb`` block-spin modes."""
    t1, t2 = _unrestricted_amplitudes(2, 1, 2, 3, seed=1)
    gate = UCC("unrestricted", t1, t2)
    assert gate.norb == 4
    assert gate.num_modes == 8
    assert gate._variant is UCC.Variant.UNRESTRICTED


def test_ucc_spinless_variant():
    """Spinless amplitudes act on a single ``norb``-mode register."""
    nocc, nvrt = 2, 2
    t1, t2 = _restricted_amplitudes(nocc, nvrt, seed=2)
    gate = UCC("spinless", t1, t2)
    assert gate.norb == nocc + nvrt
    assert gate.num_modes == nocc + nvrt
    assert gate._variant is UCC.Variant.SPINLESS


def test_ucc_variant_accepts_enum_and_string():
    """The variant may be passed either as the enum or as its string value."""
    t1, t2 = _restricted_amplitudes(1, 2, seed=3)
    assert UCC(UCC.Variant.RESTRICTED, t1, t2)._variant is UCC.Variant.RESTRICTED
    assert UCC("restricted", t1, t2)._variant is UCC.Variant.RESTRICTED


def test_ucc_unknown_variant_raises():
    """An unrecognized variant is rejected with a helpful message."""
    t1, t2 = _restricted_amplitudes(1, 1, seed=4)
    with pytest.raises(ValueError, match="Unknown UCC variant"):
        UCC("balanced", t1, t2)


def test_ucc_inconsistent_t2_shape_raises():
    """A ``t2`` whose shape is not implied by ``t1`` is rejected."""
    t1 = np.zeros((2, 2))
    with pytest.raises(ValueError, match="t2 should have shape"):
        UCC("restricted", t1, np.zeros((2, 2, 3, 3)))


def test_ucc_non_2d_t1_raises():
    """A ``t1`` that is not a 2-dimensional ``(nocc, nvrt)`` matrix is rejected."""
    with pytest.raises(ValueError, match="must be 2-dimensional"):
        UCC("restricted", np.zeros(4), np.zeros((2, 2, 2, 2)))


def test_ucc_unrestricted_mismatched_norb_raises():
    """Unrestricted alpha/beta amplitudes implying different orbital counts are rejected."""
    t1a = np.zeros((2, 2))  # norb = 4
    t1b = np.zeros((1, 3))  # norb = 4
    t1b_bad = np.zeros((1, 4))  # norb = 5
    t2 = (np.zeros((2, 2, 2, 2)), np.zeros((2, 1, 2, 4)), np.zeros((1, 1, 4, 4)))
    with pytest.raises(ValueError, match="different numbers of spatial orbitals"):
        UCC("unrestricted", (t1a, t1b_bad), t2)
    # the consistent counterpart is accepted (guards against the test passing for the wrong reason)
    good_t2 = (np.zeros((2, 2, 2, 2)), np.zeros((2, 1, 2, 3)), np.zeros((1, 1, 3, 3)))
    assert UCC("unrestricted", (t1a, t1b), good_t2).norb == 4


def test_ucc_unrestricted_wrong_arity_raises():
    """The unrestricted variant requires a ``t1`` pair and a ``t2`` triple."""
    t1, t2 = _unrestricted_amplitudes(2, 1, 2, 3, seed=5)
    with pytest.raises(ValueError, match=r"\(t2aa, t2ab, t2bb\) triple"):
        UCC("unrestricted", t1, t2[:2])
    with pytest.raises(ValueError, match=r"\(t1a, t1b\) pair"):
        UCC("unrestricted", (*t1, t1[0]), t2)


def test_ucc_unrestricted_wrong_block_shape_raises():
    """An unrestricted ``t2`` block inconsistent with the ``t1`` amplitudes is rejected."""
    t1, t2 = _unrestricted_amplitudes(2, 1, 2, 3, seed=6)
    bad = (t2[0], np.zeros((2, 1, 2, 2)), t2[2])
    with pytest.raises(ValueError, match="t2ab should have shape"):
        UCC("unrestricted", t1, bad)


def test_ucc_from_t_amplitudes_defaults_t1_to_zero():
    """Omitting ``t1`` yields a doubles-only (UCCD) ansatz with zero singles amplitudes."""
    _, t2 = _restricted_amplitudes(2, 2, seed=7)
    gate = UCC.from_t_amplitudes(t2)
    assert gate._variant is UCC.Variant.RESTRICTED
    np.testing.assert_allclose(gate.t1, 0.0)
    np.testing.assert_allclose(gate.t2, t2)


def test_ucc_from_t_amplitudes_unrestricted_defaults_t1_to_zero():
    """Omitting ``t1`` for the unrestricted variant yields per-spin zero singles amplitudes."""
    _, t2 = _unrestricted_amplitudes(2, 1, 2, 3, seed=8)
    gate = UCC.from_t_amplitudes(t2, variant="unrestricted")
    t1a, t1b = gate.t1
    assert t1a.shape == (2, 2)
    assert t1b.shape == (1, 3)
    np.testing.assert_allclose(t1a, 0.0)
    np.testing.assert_allclose(t1b, 0.0)


@pytest.mark.parametrize(
    ("variant", "nocc", "nvrt"),
    [("restricted", 2, 2), ("spinless", 2, 2), ("restricted", 1, 3)],
)
def test_ucc_num_parameters_matches_to_parameters(variant, nocc, nvrt):
    """``num_parameters`` agrees with the length of the vector ``to_parameters`` produces."""
    t1, t2 = _restricted_amplitudes(nocc, nvrt, seed=9)
    gate = UCC(variant, t1, t2)
    expected = UCC.num_parameters(nocc + nvrt, nocc, variant)
    assert len(gate.to_parameters()) == expected


def test_ucc_num_parameters_unrestricted_matches_to_parameters():
    """``num_parameters`` agrees with ``to_parameters`` for the unrestricted variant."""
    t1, t2 = _unrestricted_amplitudes(2, 1, 2, 3, seed=10)
    gate = UCC("unrestricted", t1, t2)
    assert len(gate.to_parameters()) == UCC.num_parameters(4, (2, 1), "unrestricted")


@pytest.mark.parametrize("variant", ["restricted", "spinless"])
def test_ucc_parameters_round_trip(variant):
    """``from_parameters`` inverts ``to_parameters`` for the single-tensor variants."""
    nocc, nvrt = 2, 2
    t1, t2 = _restricted_amplitudes(nocc, nvrt, seed=11)
    gate = UCC(variant, t1, t2)
    params = gate.to_parameters()
    rebuilt = UCC.from_parameters(params, nocc + nvrt, nocc, variant)
    np.testing.assert_allclose(rebuilt.to_parameters(), params, atol=1e-12)
    np.testing.assert_allclose(rebuilt.t1, t1, atol=1e-12)
    np.testing.assert_allclose(rebuilt.t2, t2, atol=1e-12)


def test_ucc_parameters_round_trip_unrestricted():
    """``from_parameters`` inverts ``to_parameters`` for the unrestricted variant."""
    t1, t2 = _unrestricted_amplitudes(2, 1, 2, 3, seed=12)
    gate = UCC("unrestricted", t1, t2)
    params = gate.to_parameters()
    rebuilt = UCC.from_parameters(params, 4, (2, 1), "unrestricted")
    np.testing.assert_allclose(rebuilt.to_parameters(), params, atol=1e-12)
    for got, want in zip(rebuilt.t2, t2, strict=True):
        np.testing.assert_allclose(got, want, atol=1e-12)


@pytest.mark.parametrize(
    ("variant", "nocc", "antisymmetric"),
    [
        ("restricted", 2, False),
        ("unrestricted", (3, 2), False),
        ("unrestricted", (3, 2), True),
        ("spinless", 3, False),
        ("spinless", 3, True),
    ],
)
def test_ucc_parameters_round_trip_is_two_sided_at_any_scale(variant, nocc, antisymmetric):
    """``to_parameters(from_parameters(p)) == p`` exactly, however large ``p`` is.

    The amplitudes are this ansatz's parameters directly, so both directions are a plain re-indexing
    and the round-trip is scale-free -- pinned here at a scale far beyond the other round-trip tests',
    and to exact equality rather than a tolerance, since no arithmetic is performed on the values.
    """
    norb = 5
    expected = UCC.num_parameters(norb, nocc, variant, antisymmetric=antisymmetric)
    rng = np.random.default_rng(abs(hash((variant, antisymmetric))) % (2**32))
    params = rng.standard_normal(expected) * 20.0

    gate = UCC.from_parameters(params, norb, nocc, variant, antisymmetric=antisymmetric)

    np.testing.assert_allclose(gate.to_parameters(), params, rtol=0, atol=0)


def test_ucc_from_parameters_wrong_length_raises():
    """A parameter vector of the wrong length is rejected."""
    with pytest.raises(ValueError, match="did not match the number expected"):
        UCC.from_parameters(np.zeros(3), 4, 2, "restricted")


def test_ucc_nocc_arity_is_validated():
    """``nocc`` must be a pair for the unrestricted variant and an integer otherwise."""
    with pytest.raises(ValueError, match="requires a \\(nocc_a, nocc_b\\) pair"):
        UCC.num_parameters(4, 2, "unrestricted")
    with pytest.raises(ValueError, match="only valid for the 'unrestricted' variant"):
        UCC.num_parameters(4, (2, 2), "restricted")


def test_ucc_cluster_operator_generator_is_anti_hermitian():
    """The cluster generator ``T - T^dagger`` is anti-Hermitian, so ``i (T - T^dagger)`` is Hermitian.

    This is exactly the property :meth:`.UCC._build_definition` relies on to express the ansatz as an
    :class:`.Evolution`, whose operator must be Hermitian for the evolution to be unitary.
    """
    t1, t2 = _restricted_amplitudes(2, 2, seed=13)
    generator = UCC("restricted", t1, t2).cluster_operator()
    assert (generator * 1j).is_hermitian()


def test_ucc_cluster_operator_conserves_sector():
    """The cluster generator conserves the particle number of each spin species.

    Every excitation replaces an occupied orbital with a virtual one *within* a spin sector, so the
    generator must preserve both the total particle number and the z-component of spin -- the
    condition :meth:`.Evolution._apply_unitary_placed_` enforces before simulating.
    """
    t1, t2 = _restricted_amplitudes(2, 2, seed=14)
    gate = UCC("restricted", t1, t2)
    generator = gate.cluster_operator()
    assert generator.conserves_particle_number()
    assert generator.conserves_sector([gate.norb, gate.norb])


@pytest.mark.parametrize("variant", ["restricted", "spinless"])
def test_ucc_generator_groups_are_individually_hermitian(variant):
    """Every group of the Hermitian generator is itself Hermitian -- a regression guard.

    :class:`.Evolution` decomposes group-by-group, so each group becomes one factor
    ``exp(-i H_k)`` of the product formula. A factor is unitary only if its ``H_k`` is Hermitian.
    Splitting ``i (T - T^dagger)`` *term*-by-term instead yields non-Hermitian factors (each
    excitation is separated from its conjugate), which makes the synthesized circuit non-unitary --
    it does not even preserve the norm. The generator therefore pairs every excitation with its
    conjugate in a shared group, which this test locks in.
    """
    t1, t2 = _restricted_amplitudes(2, 2, seed=18)
    generator = UCC(variant, t1, t2).hermitian_generator()

    assert generator.is_hermitian()
    assert generator.has_groups()
    assert generator.num_groups() > 1  # genuinely grouped, not one lump
    for group in generator.split_out_groups():
        assert group.is_hermitian()


def test_ucc_generator_groups_are_individually_hermitian_unrestricted():
    """The conjugate-pairing group invariant also holds for the unrestricted variant."""
    t1, t2 = _unrestricted_amplitudes(2, 1, 2, 3, seed=19)
    generator = UCC("unrestricted", t1, t2).hermitian_generator()

    assert generator.is_hermitian()
    for group in generator.split_out_groups():
        assert group.is_hermitian()


def test_ucc_spinless_cluster_operator_acts_only_on_norb_modes():
    """The spinless generator stays within the single ``norb``-mode register (no spin offset)."""
    nocc, nvrt = 2, 2
    t1, t2 = _restricted_amplitudes(nocc, nvrt, seed=15)
    gate = UCC("spinless", t1, t2)
    assert max(gate.cluster_operator().get_support()) < nocc + nvrt


def test_ucc_definition_is_a_single_evolution():
    """The gate's definition is one :class:`.Evolution` carrying the whole cluster generator.

    Keeping the generator in a single ``Evolution`` (rather than pre-splitting it) is what lets the
    simulation path exponentiate it exactly while leaving the Trotter decomposition to the
    transpiler.
    """
    t1, t2 = _restricted_amplitudes(1, 2, seed=16)
    gate = UCC("restricted", t1, t2)
    circuit = FermionicCircuit(gate.num_modes)
    circuit.append(gate, circuit.modes)
    assert dict(circuit.decompose().count_ops()) == {"Evolution": 1}


def test_ucc_definition_decomposes_into_term_evolutions():
    """Decomposing the definition's ``Evolution`` splits it into per-term evolutions."""
    t1, t2 = _restricted_amplitudes(1, 2, seed=17)
    gate = UCC("restricted", t1, t2)
    circuit = FermionicCircuit(gate.num_modes)
    circuit.append(gate, circuit.modes)
    counts = dict(circuit.decompose().decompose().count_ops())
    assert counts["Evolution"] > 1


def _antisymmetrize(t2):
    """Projects a same-spin ``t2`` block onto the standard coupled-cluster antisymmetric subspace."""
    t2 = t2 - t2.transpose(1, 0, 2, 3)
    return t2 - t2.transpose(0, 1, 3, 2)


def test_ucc_antisymmetric_num_parameters_is_smaller_unrestricted():
    """``antisymmetric=True`` shrinks the unrestricted parameter count to the smaller subspace.

    Both same-spin blocks drop from the full exchange-symmetric basis
    (``n_pairs * (n_pairs + 1) / 2`` entries) to the strictly-upper-triangular
    ``(i < j, a < b)`` basis, while the singles and the cross-spin ``t2ab`` block are untouched.
    """
    norb, nocc_a, nocc_b = 4, 2, 1
    nvrt_a, nvrt_b = norb - nocc_a, norb - nocc_b

    full = UCC.num_parameters(norb, (nocc_a, nocc_b), "unrestricted")
    reduced = UCC.num_parameters(norb, (nocc_a, nocc_b), "unrestricted", antisymmetric=True)

    unconstrained = nocc_a * nvrt_a + nocc_b * nvrt_b + nocc_a * nocc_b * nvrt_a * nvrt_b
    n_pairs_a, n_pairs_b = nocc_a * nvrt_a, nocc_b * nvrt_b
    assert full == unconstrained + sum(n * (n + 1) // 2 for n in (n_pairs_a, n_pairs_b))
    # only i < j, a < b survives: 1 occupied pair x 1 virtual pair for alpha, none at all for beta
    assert reduced == unconstrained + 1
    assert reduced < full


def test_ucc_antisymmetric_num_parameters_is_smaller_spinless():
    """``antisymmetric=True`` shrinks the spinless parameter count to the smaller subspace."""
    norb, nocc = 4, 2
    nvrt = norb - nocc

    full = UCC.num_parameters(norb, nocc, "spinless")
    reduced = UCC.num_parameters(norb, nocc, "spinless", antisymmetric=True)

    n_pairs = nocc * nvrt
    assert full == nocc * nvrt + n_pairs * (n_pairs + 1) // 2
    assert reduced == nocc * nvrt + 1  # one (i < j) x (a < b) entry
    assert reduced < full


@pytest.mark.parametrize(
    ("variant", "norb", "nocc"),
    [("unrestricted", 5, (3, 2)), ("spinless", 5, 3)],
)
def test_ucc_antisymmetric_parameters_round_trip(variant, norb, nocc):
    """``to_parameters`` inverts ``from_parameters`` in the antisymmetric basis.

    The flag is stored on the instance, so ``to_parameters`` writes out the *same* (smaller) basis
    ``from_parameters`` read -- a round-trip through the full exchange-symmetric basis would return a
    longer vector.
    """
    expected = UCC.num_parameters(norb, nocc, variant, antisymmetric=True)
    rng = np.random.default_rng(20)
    params = rng.standard_normal(expected)

    gate = UCC.from_parameters(params, norb, nocc, variant, antisymmetric=True)
    assert gate.antisymmetric
    assert len(gate.to_parameters()) == expected
    np.testing.assert_allclose(gate.to_parameters(), params, atol=1e-12)


@pytest.mark.parametrize(
    ("variant", "norb", "nocc"),
    [("unrestricted", 5, (3, 2)), ("spinless", 5, 3)],
)
def test_ucc_antisymmetric_from_parameters_builds_antisymmetric_amplitudes(variant, norb, nocc):
    """The amplitudes ``from_parameters(antisymmetric=True)`` builds really are antisymmetric.

    Construction and validation are deliberately independent: ``from_parameters`` expands each
    independent entry into its four sign copies, and the constructor then re-checks the result. This
    test pins down the property itself rather than trusting that composition.
    """
    expected = UCC.num_parameters(norb, nocc, variant, antisymmetric=True)
    rng = np.random.default_rng(21)
    gate = UCC.from_parameters(
        rng.standard_normal(expected), norb, nocc, variant, antisymmetric=True
    )

    blocks = [gate.t2[0], gate.t2[2]] if variant == "unrestricted" else [gate.t2]
    for t2 in blocks:
        np.testing.assert_allclose(t2, -t2.transpose(1, 0, 2, 3), atol=1e-12)
        np.testing.assert_allclose(t2, -t2.transpose(0, 1, 3, 2), atol=1e-12)


def test_ucc_antisymmetric_rejects_non_antisymmetric_amplitudes():
    """A ``t2`` outside the antisymmetric subspace is rejected when the flag is set -- but not else.

    The exchange-symmetric amplitudes ``t2[i, j, a, b] == t2[j, i, b, a]`` used everywhere else are a
    strictly larger family, so the *same* tensor must be accepted with the flag off. Asserting both
    halves keeps the test from passing for the wrong reason (e.g. an unrelated shape error).
    """
    nocc, nvrt = 2, 2
    t1, t2 = _restricted_amplitudes(nocc, nvrt, seed=22)

    with pytest.raises(ValueError, match="are not antisymmetric under occupied"):
        UCC("spinless", t1, t2, antisymmetric=True)

    assert UCC("spinless", t1, t2).norb == nocc + nvrt


def test_ucc_antisymmetric_accepts_antisymmetric_amplitudes():
    """Genuinely antisymmetric amplitudes pass the opt-in validation."""
    nocc, nvrt = 2, 2
    t1, t2 = _restricted_amplitudes(nocc, nvrt, seed=23)
    gate = UCC("spinless", t1, _antisymmetrize(t2), antisymmetric=True)
    assert gate.antisymmetric


def test_ucc_antisymmetric_validates_both_unrestricted_same_spin_blocks():
    """Both unrestricted same-spin blocks are validated, and the cross-spin block is exempt.

    ``t2ab`` carries no antisymmetry (exchanging an alpha with a beta excitation is not an exchange of
    identical operators), so leaving it unsymmetrized must not trip the check.
    """
    (t1a, t1b), (t2aa, t2ab, t2bb) = _unrestricted_amplitudes(2, 2, 2, 2, seed=24)

    # a violation in either same-spin block is caught, named by block
    with pytest.raises(ValueError, match="the t2aa amplitudes are not antisymmetric"):
        UCC("unrestricted", (t1a, t1b), (t2aa, t2ab, _antisymmetrize(t2bb)), antisymmetric=True)
    with pytest.raises(ValueError, match="the t2bb amplitudes are not antisymmetric"):
        UCC("unrestricted", (t1a, t1b), (_antisymmetrize(t2aa), t2ab, t2bb), antisymmetric=True)

    # ... while the raw (unsymmetrized) cross-spin block is accepted
    gate = UCC(
        "unrestricted",
        (t1a, t1b),
        (_antisymmetrize(t2aa), t2ab, _antisymmetrize(t2bb)),
        antisymmetric=True,
    )
    assert gate.antisymmetric


def test_ucc_antisymmetric_atol_is_honored():
    """A violation below ``atol`` is tolerated, and the default tolerance is tight enough to catch it."""
    nocc, nvrt = 2, 2
    t1, t2 = _restricted_amplitudes(nocc, nvrt, seed=25)
    t2 = _antisymmetrize(t2)
    t2[0, 0, 0, 0] = 1e-6  # a small, deliberate antisymmetry violation

    with pytest.raises(ValueError, match="are not antisymmetric"):
        UCC("spinless", t1, t2, antisymmetric=True)
    assert UCC("spinless", t1, t2, antisymmetric=True, atol=1e-4).antisymmetric


@pytest.mark.parametrize("variant", ["restricted", UCC.Variant.RESTRICTED])
def test_ucc_antisymmetric_is_refused_for_restricted(variant):
    """``antisymmetric=True`` is refused for the restricted variant at every entry point.

    The restricted ``t2`` supplies both the same-spin *and* the alpha-beta amplitudes, and the
    antisymmetry only applies to the former; honoring the flag would silently over-constrain the
    cross-spin channel. Refusing explicitly is preferred over a flag that means something else here.
    """
    t1, t2 = _restricted_amplitudes(2, 2, seed=26)
    match = "not supported for the 'restricted' variant"

    with pytest.raises(ValueError, match=match):
        UCC(variant, t1, t2, antisymmetric=True)
    with pytest.raises(ValueError, match=match):
        UCC.from_t_amplitudes(t2, t1=t1, variant=variant, antisymmetric=True)
    with pytest.raises(ValueError, match=match):
        UCC.num_parameters(4, 2, variant, antisymmetric=True)
    with pytest.raises(ValueError, match=match):
        UCC.from_parameters(np.zeros(1), 4, 2, variant, antisymmetric=True)


def test_ucc_antisymmetric_flag_does_not_change_the_operator():
    """The flag only restricts *which* amplitudes are allowed, never what they mean.

    Given amplitudes that satisfy the antisymmetry, the generator must be identical whether or not the
    flag was set -- it selects a parameterization and switches on a validation, and must not sneak an
    extra symmetrization or prefactor into the operator itself.
    """
    t1, t2 = _restricted_amplitudes(2, 2, seed=27)
    t2 = _antisymmetrize(t2)

    checked = UCC("spinless", t1, t2, antisymmetric=True).hermitian_generator()
    unchecked = UCC("spinless", t1, t2).hermitian_generator()
    assert checked.equiv(unchecked)
