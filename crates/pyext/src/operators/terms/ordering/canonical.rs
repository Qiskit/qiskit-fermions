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
use qiskit_fermions_core::operators::terms::ordering::canonical::canonical_order;

/// Returns a copy of an operator with its terms sorted into a canonical order.
///
/// The terms are sorted into a canonical order that depends only on each term's structure (the
/// operator string it represents) and not on its coefficient. The order is therefore deterministic
/// for a given set of terms regardless of how the operator was assembled. The terms themselves are
/// left untouched — this only reorders them, it does not simplify or normal-order the operator.
///
/// This works for any of the built-in operator types implementing the
/// :class:`~qiskit_fermions.operators.OperatorTrait` protocol; the returned operator is of the same
/// type as the input.
///
/// .. note::
///    Any group indices (see e.g.
///    :attr:`~qiskit_fermions.operators.FermionOperator.groups`) are preserved: each term carries
///    its group index along as it moves, since a group index is a per-term tag independent of term
///    order. Terms of the same group are simply no longer contiguous afterwards. If the input has
///    no groups, neither does the result.
///
/// .. doctest::
///
///     >>> from qiskit_fermions.operators import FermionOperator
///     >>> from qiskit_fermions.operators.terms.ordering import canonical_order
///     >>> op = FermionOperator.from_dict(
///     ...     {
///     ...         ((True, 1), (False, 0)): 1.0,  # a†_1 a_0
///     ...         ((True, 0), (False, 1)): 2.0,  # a†_0 a_1
///     ...     }
///     ... )
///     >>> ordered = canonical_order(op)
///     >>> list(ordered.iter_terms())
///     [([(True, 0), (False, 1)], (2+0j)), ([(True, 1), (False, 0)], (1+0j))]
///
/// Args:
///     op: the operator whose terms to reorder.
///
/// Returns:
///     A new operator of the same type with its terms in canonical order.
///
/// Raises:
///     TypeError: if ``op`` is not a supported operator type (see
///         :class:`~qiskit_fermions.operators.OperatorTrait`).
#[gen_stub_pyfunction(module = "qiskit_fermions._lib.operators.operators_terms.ordering.canonical")]
#[pyfunction(name = "canonical_order")]
#[gen_stub(override_return_type(
    type_repr = "OperatorTrait",
    imports = ("qiskit_fermions.operators.operator_trait.OperatorTrait")
))]
pub fn py_canonical_order(
    #[gen_stub(override_type(
        type_repr = "OperatorTrait",
        imports = ("qiskit_fermions.operators.operator_trait.OperatorTrait")
    ))]
    op: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let py = op.py();
    // Dispatch on the concrete operator type. `cast` is a borrowing type check: on a miss it
    // returns a borrowed error without allocating a `PyErr`, and on a hit it hands back a `&Bound`
    // we borrow from. `canonical_order` reorders through a shared borrow and returns a fresh
    // operator, so no clone of the input is needed.
    macro_rules! dispatch {
        ($($py_op:ty),+ $(,)?) => {
            $(
                if let Ok(op) = op.cast::<$py_op>() {
                    let ordered = <$py_op>::from(canonical_order(&op.borrow().inner));
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
        "canonical_order expects a fermionic, Majorana, edge-vertex or transfer-vertex operator",
    ))
}

#[pymodule]
pub mod canonical {
    #[pymodule_export]
    use super::py_canonical_order;
}
