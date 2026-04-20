"""Fermionic mode ordering optimization utilities.

This module provides a mixed-integer optimization routine that computes an index
permutation minimizing excitation spans for fermionic operators when embedded
into a 1D ordered register. The objective can target:
- worst-case span ("minmax"),
- average span ("avg"), or
- a weighted combination of both ("multi").

The implementation uses Pyomo and is compatible with multiple MILP backends
(e.g. HiGHS, CPLEX, Gurobi, GLPK, CBC).
"""

from typing import Literal

import numpy as np
from pyomo.environ import (
    Binary,
    ConcreteModel,
    Constraint,
    Objective,
    RangeSet,
    Reals,
    Set,
    Var,
    minimize,
)
from qiskit_fermions.utils.optionals import HAS_PYOMO


@HAS_PYOMO.require_in_call
def build_excitation_span_optimization_model(
    excitations: list[tuple[int, int] | tuple[int, int, int, int]],
    num_modes: int,
    *,
    objective: Literal["minmax", "multi", "avg"] = "multi",
    mix_delta: float = 0.1,
) -> ConcreteModel:
    """Build a Pyomo model for ordering fermionic modes to minimize excitation spans.

    The model constructs a permutation where each original mode index is assigned to
    exactly one position in a linear ordering. For every 2-mode and 4-mode
    excitation tuple, the span of occupied positions is minimized according to the
    chosen objective.

    Notes:
        - Input tuples are preprocessed by canceling indices that occur twice
          within the same excitation under the function's pair-cancellation
          logic; tuples reducing to length 0 or 1 are ignored.
        - Supported (post-cancellation) tuple lengths are 2 and 4.

    Args:
        excitations (list[tuple[int, int] | tuple[int, int, int, int]]):
            Sequence of excitation index tuples over fermionic mode indices. After the
            function's pair-cancellation preprocessing, supported tuple lengths are 2 and 4.
        num_modes (int):
            Total number of fermionic modes/orbitals to be ordered (size of the register).
        objective (Literal["minmax", "multi", "avg"]):
            Objective mode:
            - ``"minmax"`` minimizes the maximum excitation span.
            - ``"avg"`` minimizes the average excitation span.
            - ``"multi"`` minimizes ``max_span + mix_delta * average_span``.
        mix_delta (float):
            Weight used only when ``objective="multi"``.

    Returns:
        ConcreteModel:
            A Pyomo ``ConcreteModel`` encoding the permutation variables, span variables,
            constraints, and objective. Solve this model with a Pyomo solver to obtain an
            optimized ordering.
    """
    # Preprocess excitations
    i2 = []
    i4 = []
    for ind in excitations:
        deduplicated = []
        ignore = set()
        for i in ind:
            if i in ignore:
                continue
            if i in deduplicated:
                ignore.add(i)
                deduplicated.remove(i)
            else:
                deduplicated.append(i)
        if len(deduplicated) <= 1:
            continue
        if len(deduplicated) == 3:
            raise ValueError(
                f"Unsupported excitation after cancellation: {ind} -> {tuple(deduplicated)}"
            )
        if len(deduplicated) == 2:
            i2 += [tuple(deduplicated)]
        elif len(deduplicated) == 4:
            i4 += [tuple(deduplicated)]

    i2 = [tuple(a) for a in np.unique(i2, axis=0)]
    i4 = [tuple(a) for a in np.unique(i4, axis=0)]

    model = ConcreteModel()

    # Index sets
    model.I = RangeSet(0, num_modes - 1)
    model.J = RangeSet(0, num_modes - 1)
    model.P2 = Set(initialize=i2)
    model.P4 = Set(initialize=i4)

    # Variables
    model.x = Var(
        model.I, model.J, within=Binary
    )  # x[i,j] == 1 if mode i placed at position j
    model.y = Var(model.I, within=Reals)

    model.s = Var(model.P2, within=Reals)
    model.t = Var(model.P2, within=Reals)
    model.u = Var(model.P4, within=Reals)
    model.v = Var(model.P4, within=Reals)

    if objective in ("minmax", "multi"):
        model.max_obj = Var(within=Reals)

    # Permutation constraints: rows and columns sum to 1
    def row_sum_rule(m, i):
        return sum(m.x[i, j] for j in m.J) == 1

    def col_sum_rule(m, j):
        return sum(m.x[i, j] for i in m.I) == 1

    model.row_sum = Constraint(model.I, rule=row_sum_rule)
    model.col_sum = Constraint(model.J, rule=col_sum_rule)

    # y[i] = sum_j j * x[i,j]
    def y_def_rule(m, i):
        return m.y[i] == sum(j * m.x[i, j] for j in m.J)

    model.y_def = Constraint(model.I, rule=y_def_rule)

    # Constraints for P2
    for i, j in i2:
        model.add_component(
            f"s_le_y_{i}_{j}_1", Constraint(expr=model.s[(i, j)] <= model.y[i])
        )
        model.add_component(
            f"s_le_y_{i}_{j}_2", Constraint(expr=model.s[(i, j)] <= model.y[j])
        )
        model.add_component(
            f"t_ge_y_{i}_{j}_1", Constraint(expr=model.t[(i, j)] >= model.y[i])
        )
        model.add_component(
            f"t_ge_y_{i}_{j}_2", Constraint(expr=model.t[(i, j)] >= model.y[j])
        )
        if hasattr(model, "max_obj"):
            model.add_component(
                f"max_ge_span_{i}_{j}",
                Constraint(expr=model.max_obj >= model.t[(i, j)] - model.s[(i, j)] + 1),
            )

    # Constraints for P4
    for i, j, k, l in i4:
        model.add_component(
            f"u_le_y_{i}_{j}_{k}_{l}_1",
            Constraint(expr=model.u[(i, j, k, l)] <= model.y[i]),
        )
        model.add_component(
            f"u_le_y_{i}_{j}_{k}_{l}_2",
            Constraint(expr=model.u[(i, j, k, l)] <= model.y[j]),
        )
        model.add_component(
            f"u_le_y_{i}_{j}_{k}_{l}_3",
            Constraint(expr=model.u[(i, j, k, l)] <= model.y[k]),
        )
        model.add_component(
            f"u_le_y_{i}_{j}_{k}_{l}_4",
            Constraint(expr=model.u[(i, j, k, l)] <= model.y[l]),
        )
        model.add_component(
            f"v_ge_y_{i}_{j}_{k}_{l}_1",
            Constraint(expr=model.v[(i, j, k, l)] >= model.y[i]),
        )
        model.add_component(
            f"v_ge_y_{i}_{j}_{k}_{l}_2",
            Constraint(expr=model.v[(i, j, k, l)] >= model.y[j]),
        )
        model.add_component(
            f"v_ge_y_{i}_{j}_{k}_{l}_3",
            Constraint(expr=model.v[(i, j, k, l)] >= model.y[k]),
        )
        model.add_component(
            f"v_ge_y_{i}_{j}_{k}_{l}_4",
            Constraint(expr=model.v[(i, j, k, l)] >= model.y[l]),
        )
        if hasattr(model, "max_obj"):
            model.add_component(
                f"max_ge_span_{i}_{j}_{k}_{l}",
                Constraint(
                    expr=model.max_obj
                    >= model.v[(i, j, k, l)] - model.u[(i, j, k, l)] + 1
                ),
            )

    # Average objective expression
    denom = len(i2) + len(i4) or 1
    avg_expr = 0
    for p in i2:
        avg_expr = avg_expr + (model.t[p] - model.s[p] + 1) / denom
    for p in i4:
        avg_expr = avg_expr + (model.v[p] - model.u[p] + 1) / denom

    # Define objective
    if objective == "minmax":
        obj_expr = model.max_obj
    elif objective == "multi":
        obj_expr = model.max_obj + mix_delta * avg_expr
    elif objective == "avg":
        obj_expr = avg_expr
    else:
        raise ValueError(f"Unknown objective: {objective}")

    model.obj = Objective(expr=obj_expr, sense=minimize)

    return model
