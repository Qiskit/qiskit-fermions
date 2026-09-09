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
use qiskit_fermions_core::operators::terms::ordering::group::group_order;

/// Returns a copy of an operator with its terms ordered by group index.
///
/// The terms are sorted by their group index alone, which makes each group one contiguous run of
/// terms and the group indices non-decreasing. The sort is stable, so terms within a group keep their
/// relative order: ordering canonically first and by group second therefore refines the canonical
/// order rather than replacing it with an arbitrary permutation. The terms themselves are left
/// untouched -- this only reorders them, it does not simplify or normal-order the operator.
///
/// This works for any of the built-in operator types implementing the
/// :class:`~qiskit_fermions.operators.OperatorTrait` protocol; the returned operator is of the same
/// type as the input.
///
/// Group indices say only which terms belong together, so this changes the operator's *layout*, not
/// its value. What the layout buys is lookup cost:
/// :meth:`~qiskit_fermions.operators.OperatorTrait.split_out_groups` has to scan every term to find
/// the requested groups in general, but on a group-ordered operator each group's terms form one
/// range it can locate by binary search, so a lookup costs what the *requested* groups cost rather
/// than what the *held* terms cost. Hoisting this call out of a loop that repeatedly samples a few
/// groups from a large operator is what makes those lookups affordable.
///
/// .. note::
///    An operator tracking no groups (see for example
///    :attr:`~qiskit_fermions.operators.FermionOperator.groups`) has nothing to order by and is
///    returned as an unchanged copy, so this composes in a pipeline without a guard at every step.
///    The result tracks no groups either.
///
/// .. doctest::
///
///     >>> from qiskit_fermions.operators import FermionOperator
///     >>> from qiskit_fermions.operators.terms.ordering import group_order
///     >>> op = FermionOperator.from_dict(
///     ...     {
///     ...         ((True, 0), (False, 1)): 1.0,
///     ...         ((True, 1), (False, 0)): 2.0,
///     ...         ((True, 2), (False, 3)): 3.0,
///     ...     }
///     ... )
///     >>> op.groups = [1, 0, 1]  # group 1's terms are not contiguous
///     >>> ordered = group_order(op)
///     >>> print(ordered.groups)
///     [0, 1, 1]
///
/// Args:
///     op: the operator whose terms to reorder.
///
/// Returns:
///     A new operator of the same type with its terms ordered by group index.
///
/// Raises:
///     TypeError: if ``op`` is not a supported operator type (see
///         :class:`~qiskit_fermions.operators.OperatorTrait`).
#[gen_stub_pyfunction(module = "qiskit_fermions._lib.operators.operators_terms.ordering.group")]
#[pyfunction(name = "group_order")]
#[gen_stub(override_return_type(
    type_repr = "OperatorTrait",
    imports = ("qiskit_fermions.operators.operator_trait.OperatorTrait")
))]
pub fn py_group_order(
    #[gen_stub(override_type(
        type_repr = "OperatorTrait",
        imports = ("qiskit_fermions.operators.operator_trait.OperatorTrait")
    ))]
    op: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let py = op.py();
    // Dispatch on the concrete operator type, exactly as `canonical_order` does. `cast` is a
    // borrowing type check: on a miss it returns a borrowed error without allocating a `PyErr`, and
    // on a hit it hands back a `&Bound` we borrow from. `group_order` reorders through a shared
    // borrow and returns a fresh operator, so no clone of the input is needed.
    macro_rules! dispatch {
        ($($py_op:ty),+ $(,)?) => {
            $(
                if let Ok(op) = op.cast::<$py_op>() {
                    let ordered = <$py_op>::from(group_order(&op.borrow().inner));
                    return Ok(ordered.into_pyobject(py)?.into_any().unbind());
                }
            )+
        };
    }
    dispatch!(
        PyFermionOperator,
        PyMajoranaOperator,
        PyEdgeVertexOperator,
        PyTransferVertexOperator,
    );
    Err(PyTypeError::new_err(
        "group_order expects a fermionic, Majorana, edge-vertex or transfer-vertex operator",
    ))
}

#[pymodule]
pub mod group {
    #[pymodule_export]
    use super::py_group_order;
}
