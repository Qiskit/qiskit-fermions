// This code is a Qiskit project.
//
// (C) Copyright IBM 2026.
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;

use qiskit_fermions_core::operators::terms::grouping::clash as core_clash;

/// Group the operator's terms so that the terms sharing a group index mutually commute.
///
/// Colors the operator's *clash graph* - the graph whose nodes are its terms and whose edges join
/// the pairs that do **not** commute, available as ``op.non_commutation_edges()``. A proper coloring
/// places two non-commuting terms in different groups by construction. Group indices are assigned in
/// place, replacing any the operator already carried.
///
/// As for ``op.non_commutation_edges()``, a term with an exactly zero coefficient clashes with nothing,
/// while every other term counts as written: simplify the operator first to remove near-zero and
/// duplicate terms, which would otherwise inflate the number of groups.
///
/// The coloring is the greedy *largest-first* heuristic (terms with the most clashes are placed
/// first), so the number of groups is an upper bound on the smallest achievable rather than a
/// minimum. The graph itself is never built: memory grows linearly with the number of terms, where
/// the list of clashing pairs grows quadratically. This makes the grouping usable on operators whose
/// ``op.non_commutation_edges()`` would not fit in memory. Building the graph from those edges
/// remains the way to color it with a different strategy.
///
/// Args:
///     op: the operator whose terms to group.
///
/// Raises:
///     TypeError: if ``op`` is not one of the four operator types.
///     ValueError: if ``op`` has more than :math:`2^{32}` terms, or is a :class:`.FermionOperator`
///         with a term of degree above 20 and a non-zero coefficient, exactly when
///         ``op.non_commutation_edges()`` raises. The operator's groups are then left untouched.
///
/// .. note::
///     Whether these commuting groups agree with the line flow sets used to compile the time
///     evolution of lattice Hamiltonians depends on the operator representation, not on the lattice.
///     A flow set forbids two arrows that share a source or a target while permitting a head-to-tail
///     chain. A transfer :math:`T_{jk}` acts on the odd generator of its source and the even
///     generator of its target, so for the hopping terms of a :class:`.TransferVertexOperator` the
///     clash relation *is* that rule. Majorana and edge-vertex bilinears instead act on the same
///     generator of a shared mode however they meet, so there a head-to-tail chain does **not**
///     commute and the two partitions diverge. See the :ref:`grouping guide <grouping_explanation>`.
///
/// Examples:
///     .. plot::
///        :include-source:
///        :nofigs:
///        :context: close-figs
///
///         >>> from qiskit_fermions.operators import MajoranaOperator
///         >>> from qiskit_fermions.operators.terms.grouping import (
///         ...     group_terms_by_non_commutation,
///         ... )
///         >>>
///         >>> # built from the sparse arrays rather than a dictionary, because only the former fixes
///         >>> # the term order that the group indices below are paired with
///         >>> op = MajoranaOperator([1.0, 1.0, 1.0], [0, 1, 1, 2, 3, 4], [0, 2, 4, 6])
///         >>> group_terms_by_non_commutation(op)
///         >>> op.groups
///         [0, 1, 0]
///
///     The two clashing terms land in different groups, while the term commuting with both rejoins
///     the first group.
#[gen_stub_pyfunction(module = "qiskit_fermions._lib.operators.operators_terms.grouping.clash")]
#[pyfunction(name = "group_terms_by_non_commutation")]
pub fn py_group_terms_by_non_commutation(
    #[gen_stub(override_type(type_repr = "OperatorTrait", imports = ("qiskit_fermions.operators.operator_trait.OperatorTrait")))]
    op: &Bound<'_, PyAny>,
) -> PyResult<()> {
    // The two quadratic sweeps touch no Python object, so other Python threads may run meanwhile.
    // The operator itself stays borrowed throughout, so they cannot touch it.
    dispatch!(mut, op, "group_terms_by_non_commutation", |inner| {
        op.py()
            .detach(|| core_clash::group_terms_by_non_commutation(inner))
            .map_err(crate::value_err)?
    })
}

#[pymodule]
pub mod clash {
    #[pymodule_export]
    use super::py_group_terms_by_non_commutation;
}
