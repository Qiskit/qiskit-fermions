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

"""Tests for build_excitation_span_optimization_model.

These tests focus on deterministic model-construction behavior:

- excitation preprocessing and cancellation of repeated indices
- partitioning into 2-body and 4-body excitation sets
- deduplication of processed excitations
- creation of index sets, variables, and constraints
- objective-dependent creation of ``max_obj``
- edge cases such as empty or fully-ignored excitation input

They intentionally do not test solver output or optimality, since the function
only constructs a Pyomo model and does not solve it.
"""

from __future__ import annotations

import pytest
from pyomo.environ import Constraint, Objective
from qiskit_fermions.mappers.optimization.models import build_excitation_span_optimization_model


def _set_members(pyomo_set):
    """Return the members of a Pyomo set as a plain Python set."""
    return set(pyomo_set.data())


def test_empty_excitations_builds_valid_model_for_avg():
    """Build a model successfully when no excitations are provided.

    This verifies the denominator fallback path and confirms that the model
    can still be constructed even when both processed excitation sets are empty.
    """
    model = build_excitation_span_optimization_model([], num_modes=4, objective="avg")

    assert list(model.I) == [0, 1, 2, 3]
    assert list(model.J) == [0, 1, 2, 3]
    assert _set_members(model.P2) == set()
    assert _set_members(model.P4) == set()

    # `avg` objective should not define `max_obj`
    assert not hasattr(model, "max_obj")
    assert isinstance(model.obj, Objective)


def test_ignores_terms_reducing_to_length_zero_or_one():
    """Ignore excitations that cancel down to length 0 or 1.

    Examples:
    - (1, 1) -> length 0
    - (2, 2, 2) -> length 1 after the implemented cancellation logic
    """
    model = build_excitation_span_optimization_model(
        excitations=[(1, 1), (2, 2, 2)],
        num_modes=4,
        objective="avg",
    )

    assert _set_members(model.P2) == set()
    assert _set_members(model.P4) == set()


def test_repeated_indices_cancel_to_a_pair():
    """Reduce repeated indices by pairwise cancellation.

    The implementation removes an index if it appears twice, so
    (1, 2, 2, 3) becomes (1, 3).
    """
    model = build_excitation_span_optimization_model(
        excitations=[(1, 2, 2, 3)],
        num_modes=4,
        objective="avg",
    )

    assert _set_members(model.P2) == {(1, 3)}
    assert _set_members(model.P4) == set()


def test_processed_pairs_and_quads_are_partitioned_correctly():
    """Place processed 2-body and 4-body terms into the expected sets."""
    model = build_excitation_span_optimization_model(
        excitations=[
            (0, 2),  # stays a pair
            (0, 1, 2, 3),  # stays a quad
        ],
        num_modes=4,
        objective="avg",
    )

    assert _set_members(model.P2) == {(0, 2)}
    assert _set_members(model.P4) == {(0, 1, 2, 3)}


def test_duplicate_processed_pairs_are_deduplicated():
    """Store each processed 2-body excitation only once.

    This includes duplicates arising both from identical raw tuples and from
    distinct raw tuples that reduce to the same processed pair.
    """
    model = build_excitation_span_optimization_model(
        excitations=[
            (1, 3),
            (1, 3),
            (1, 2, 2, 3),  # also reduces to (1, 3)
        ],
        num_modes=4,
        objective="avg",
    )

    assert _set_members(model.P2) == {(1, 3)}
    assert len(list(model.P2.data())) == 1


def test_duplicate_processed_quads_are_deduplicated():
    """Store each processed 4-body excitation only once."""
    model = build_excitation_span_optimization_model(
        excitations=[
            (0, 1, 2, 3),
            (0, 1, 2, 3),
        ],
        num_modes=4,
        objective="avg",
    )

    assert _set_members(model.P4) == {(0, 1, 2, 3)}
    assert len(list(model.P4.data())) == 1


def test_length_three_after_processing_raises_assertion():
    """Raise when a processed excitation has length 3.

    The implementation explicitly asserts that post-cancellation tuples of
    length 3 are unsupported.
    """
    with pytest.raises(ValueError):
        build_excitation_span_optimization_model(
            excitations=[(0, 1, 2)],
            num_modes=4,
            objective="avg",
        )


def test_permutation_constraint_blocks_have_expected_size():
    """Create one row-sum and one column-sum constraint per mode."""
    num_modes = 5
    model = build_excitation_span_optimization_model(
        excitations=[(0, 1)],
        num_modes=num_modes,
        objective="avg",
    )

    assert len(model.row_sum) == num_modes
    assert len(model.col_sum) == num_modes
    assert len(model.y_def) == num_modes


def test_x_and_y_variables_are_indexed_over_expected_sets():
    """Index assignment variables consistently with the mode sets."""
    model = build_excitation_span_optimization_model(
        excitations=[(0, 1)],
        num_modes=3,
        objective="avg",
    )

    for i in model.I:
        assert i in model.y

    for i in model.I:
        for j in model.J:
            assert (i, j) in model.x


def test_pair_span_variables_are_created_for_processed_pairs():
    """Create s and t variables indexed by the processed pair set."""
    model = build_excitation_span_optimization_model(
        excitations=[(0, 2)],
        num_modes=3,
        objective="avg",
    )

    assert (0, 2) in model.s
    assert (0, 2) in model.t


def test_quad_span_variables_are_created_for_processed_quads():
    """Create u and v variables indexed by the processed 4-body set."""
    model = build_excitation_span_optimization_model(
        excitations=[(0, 1, 2, 3)],
        num_modes=4,
        objective="avg",
    )

    assert (0, 1, 2, 3) in model.u
    assert (0, 1, 2, 3) in model.v


def test_named_constraints_exist_for_pair_when_max_objective_is_used():
    """Add all expected named constraints for a processed pair.

    For objective='minmax', the pair-specific span constraints and the
    max-span linking constraint should all be present.
    """
    model = build_excitation_span_optimization_model(
        excitations=[(0, 2)],
        num_modes=3,
        objective="minmax",
    )

    assert isinstance(model.component("s_le_y_0_2_1"), Constraint)
    assert isinstance(model.component("s_le_y_0_2_2"), Constraint)
    assert isinstance(model.component("t_ge_y_0_2_1"), Constraint)
    assert isinstance(model.component("t_ge_y_0_2_2"), Constraint)
    assert isinstance(model.component("max_ge_span_0_2"), Constraint)


def test_named_constraints_exist_for_quad_when_max_objective_is_used():
    """Add all expected named constraints for a processed 4-body term."""
    model = build_excitation_span_optimization_model(
        excitations=[(0, 1, 2, 3)],
        num_modes=4,
        objective="minmax",
    )

    for suffix in range(1, 5):
        assert isinstance(model.component(f"u_le_y_0_1_2_3_{suffix}"), Constraint)
        assert isinstance(model.component(f"v_ge_y_0_1_2_3_{suffix}"), Constraint)

    assert isinstance(model.component("max_ge_span_0_1_2_3"), Constraint)


def test_max_obj_exists_for_minmax():
    """Create max_obj for the minmax objective."""
    model = build_excitation_span_optimization_model(
        excitations=[(0, 1)],
        num_modes=2,
        objective="minmax",
    )

    assert hasattr(model, "max_obj")


def test_max_obj_exists_for_multi():
    """Create max_obj for the multi objective."""
    model = build_excitation_span_optimization_model(
        excitations=[(0, 1)],
        num_modes=2,
        objective="multi",
    )

    assert hasattr(model, "max_obj")


def test_max_obj_not_created_for_avg():
    """Do not create max_obj for the avg objective."""
    model = build_excitation_span_optimization_model(
        excitations=[(0, 1)],
        num_modes=2,
        objective="avg",
    )

    assert not hasattr(model, "max_obj")


def test_invalid_objective_raises_value_error():
    """Reject unsupported objective values."""
    with pytest.raises(ValueError, match="Unknown objective"):
        build_excitation_span_optimization_model(
            excitations=[(0, 1)],
            num_modes=2,
            objective="not-a-valid-objective",
        )
