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

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;

use crate::operators::edge_vertex_operator::PyEdgeVertexOperator;
use crate::operators::fermion_operator::PyFermionOperator;
use crate::operators::majorana_operator::PyMajoranaOperator;
use crate::operators::transfer_vertex_operator::PyTransferVertexOperator;
use qiskit_fermions_core::operators::terms::grouping::analysis as core_analysis;

/// Applies a group-analysis routine to whichever concrete operator type `op` holds.
///
/// The routines in `core` are generic over the operator type, but Python hands us an untyped object,
/// so each wrapper resolves it here. `cast` is a borrowing type check: on a miss it returns a
/// borrowed error without allocating a `PyErr`, and on a hit it hands back a `&Bound` we borrow
/// from. Every routine reads through a shared borrow, so no clone of the input is needed.
macro_rules! dispatch {
    ($op:expr, $name:literal, |$inner:ident| $call:expr) => {{
        if let Ok(bound) = $op.cast::<PyFermionOperator>() {
            let $inner = &bound.borrow().inner;
            return Ok($call);
        }
        if let Ok(bound) = $op.cast::<PyMajoranaOperator>() {
            let $inner = &bound.borrow().inner;
            return Ok($call);
        }
        if let Ok(bound) = $op.cast::<PyEdgeVertexOperator>() {
            let $inner = &bound.borrow().inner;
            return Ok($call);
        }
        if let Ok(bound) = $op.cast::<PyTransferVertexOperator>() {
            let $inner = &bound.borrow().inner;
            return Ok($call);
        }
        Err(PyTypeError::new_err(concat!(
            $name,
            " expects a fermionic, Majorana, edge-vertex or transfer-vertex operator"
        )))
    }};
}

/// Returns the mean absolute coefficient magnitude of each group.
///
/// The ``i``-th entry is the sum of ``abs(coeff)`` over the terms in group ``i``, divided by the
/// number of terms in that group. If the operator tracks no groups, this returns ``None``.
///
/// This is the sampling weight of a randomized product formula (for example, qDRIFT) that draws
/// whole groups rather than individual terms: it is the magnitude of one *atomic* group, which is
/// the relevant scale because grouping is what makes each sampled piece Hermitian (and hence its
/// time evolution unitary) in the first place.
///
/// Computing it natively is considerably cheaper than reducing ``get_coeffs()`` and ``groups`` in
/// NumPy, because those two accessors each copy one value per *ungrouped* term out of the operator
/// only for it to be aggregated back down to one value per group, whereas this returns just the
/// ``num_groups()`` reduced values.
///
/// .. note::
///    A group index that no term carries weighs ``0.0``, which keeps it out of the sample.
///
/// .. doctest::
///
///     >>> from qiskit_fermions.operators import FermionOperator
///     >>> from qiskit_fermions.operators.terms.grouping import group_coeff_means
///     >>> op = FermionOperator(
///     ...     [1.0, 2.0, -1.0, -2.0],
///     ...     [True, False, True, False, True, False, True, False],
///     ...     [0, 1, 2, 3, 1, 0, 3, 2],
///     ...     [0, 2, 4, 6, 8],
///     ... )
///     >>> print(group_coeff_means(op))
///     None
///     >>> op.groups = [0, 1, 0, 1]
///     >>> group_coeff_means(op)
///     [1.0, 2.0]
///
/// Args:
///     op: the operator whose groups to reduce.
///
/// Returns:
///     The mean absolute coefficient magnitude of each group index, or ``None`` if the operator
///     tracks no groups.
///
/// Raises:
///     TypeError: if ``op`` is not a supported operator type (see
///         :class:`~qiskit_fermions.operators.OperatorTrait`).
#[gen_stub_pyfunction(module = "qiskit_fermions._lib.operators.operators_terms.grouping.analysis")]
#[pyfunction(name = "group_coeff_means")]
pub fn py_group_coeff_means(
    #[gen_stub(override_type(
        type_repr = "OperatorTrait",
        imports = ("qiskit_fermions.operators.operator_trait.OperatorTrait")
    ))]
    op: &Bound<'_, PyAny>,
) -> PyResult<Option<Vec<f64>>> {
    dispatch!(op, "group_coeff_means", |inner| {
        core_analysis::group_coeff_means(inner)
    })
}

/// Returns, for each group, whether the operator formed by that group's terms is Hermitian.
///
/// Groups prescribe no meaning of their own; this checks one *common* convention, namely that each
/// group is separately Hermitian. That is the property a randomized product formula relies on when it
/// samples whole groups, since only a Hermitian group has a unitary time evolution. If the operator
/// tracks no groups, this returns ``None``.
///
/// An empty group counts as Hermitian, because the zero operator is.
///
/// .. note::
///    This inherits the one-sided guarantee of :meth:`~qiskit_fermions.operators.OperatorTrait.is_hermitian`:
///    a ``True`` entry is always reliable, while a ``False`` entry is reliable only for operator
///    types whose normal form is a genuine canonical form. For the others the check is
///    *conservative* and can report ``False`` for a group that is in fact Hermitian.
///
/// .. note::
///    This is *not* implied by (nor does it imply) :func:`.groups_have_uniform_coeffs`. Uniform
///    coefficients do not make a group Hermitian, and a Hermitian group may mix magnitudes.
///
/// .. doctest::
///
///     >>> from qiskit_fermions.operators import FermionOperator
///     >>> from qiskit_fermions.operators.terms.grouping import groups_are_hermitian
///     >>> # built from the sparse arrays rather than a dictionary, because only the former
///     >>> # fixes the term order that the group indices are paired with
///     >>> op = FermionOperator(
///     ...     [1.0, 1.0, 1.0],
///     ...     [True, False, True, False, True, False],
///     ...     [0, 1, 1, 0, 2, 3],
///     ...     [0, 2, 4, 6],
///     ... )
///     >>> op.groups = [0, 0, 1]  # the leading conjugate pair, then one unpaired term
///     >>> groups_are_hermitian(op)
///     [True, False]
///
/// Args:
///     op: the operator whose groups to check.
///     atol: The numerical accuracy upto which coefficients are considered equal. This value
///         defaults to ``1e-8``.
///
/// Returns:
///     One flag per group index, or ``None`` if the operator tracks no groups.
///
/// Raises:
///     TypeError: if ``op`` is not a supported operator type (see
///         :class:`~qiskit_fermions.operators.OperatorTrait`).
#[gen_stub_pyfunction(module = "qiskit_fermions._lib.operators.operators_terms.grouping.analysis")]
#[pyfunction(name = "groups_are_hermitian")]
#[pyo3(signature = (op, *, atol=1e-8))]
pub fn py_groups_are_hermitian(
    #[gen_stub(override_type(
        type_repr = "OperatorTrait",
        imports = ("qiskit_fermions.operators.operator_trait.OperatorTrait")
    ))]
    op: &Bound<'_, PyAny>,
    atol: f64,
) -> PyResult<Option<Vec<bool>>> {
    dispatch!(op, "groups_are_hermitian", |inner| {
        core_analysis::groups_are_hermitian(inner, atol)
    })
}

/// Returns, for each group, whether all of its coefficients are numerically equal.
///
/// With ``abs`` (the default), the coefficient *magnitudes* are compared, which is the assumption
/// that :func:`.group_coeff_means` makes when it averages ``abs(coeff)``. Without it, the
/// coefficients must match exactly. Note that a Hermitian group may legitimately fail the stricter
/// form, since a conjugate pair with complex coefficients has equal magnitudes but unequal
/// coefficients. If the operator tracks no groups, this returns ``None``.
///
/// An empty or single-term group is trivially uniform.
///
/// .. note::
///    This is *not* a weaker form of :func:`.groups_are_hermitian`: neither implies the other.
///    Uniform coefficients do not make a group Hermitian, and a Hermitian group can mix
///    magnitudes.
///
/// .. doctest::
///
///     >>> from qiskit_fermions.operators import FermionOperator
///     >>> from qiskit_fermions.operators.terms.grouping import groups_have_uniform_coeffs
///     >>> op = FermionOperator.from_dict(
///     ...     {
///     ...         ((True, 0), (False, 1)): 1.0j,
///     ...         ((True, 1), (False, 0)): -1.0j,
///     ...     }
///     ... )
///     >>> op.groups = [0, 0]
///     >>> groups_have_uniform_coeffs(op)  # equal magnitudes
///     [True]
///     >>> groups_have_uniform_coeffs(op, abs=False)  # but not equal coefficients
///     [False]
///
/// Args:
///     op: the operator whose groups to check.
///     atol: The numerical accuracy upto which coefficients are considered equal. This value
///         defaults to ``1e-8``.
///     abs: Whether to compare coefficient magnitudes rather than the coefficients themselves. This
///         value defaults to ``True``.
///
/// Returns:
///     One flag per group index, or ``None`` if the operator tracks no groups.
///
/// Raises:
///     TypeError: if ``op`` is not a supported operator type (see
///         :class:`~qiskit_fermions.operators.OperatorTrait`).
#[gen_stub_pyfunction(module = "qiskit_fermions._lib.operators.operators_terms.grouping.analysis")]
#[pyfunction(name = "groups_have_uniform_coeffs")]
#[pyo3(signature = (op, *, atol=1e-8, abs=true))]
pub fn py_groups_have_uniform_coeffs(
    #[gen_stub(override_type(
        type_repr = "OperatorTrait",
        imports = ("qiskit_fermions.operators.operator_trait.OperatorTrait")
    ))]
    op: &Bound<'_, PyAny>,
    atol: f64,
    abs: bool,
) -> PyResult<Option<Vec<bool>>> {
    dispatch!(op, "groups_have_uniform_coeffs", |inner| {
        core_analysis::groups_have_uniform_coeffs(inner, atol, abs)
    })
}

#[pymodule]
pub mod analysis {
    #[pymodule_export]
    use super::py_group_coeff_means;
    #[pymodule_export]
    use super::py_groups_are_hermitian;
    #[pymodule_export]
    use super::py_groups_have_uniform_coeffs;
}
