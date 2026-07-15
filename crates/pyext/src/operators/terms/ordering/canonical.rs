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
/// This works for every operator type; the returned operator is of the same type as the input.
///
/// .. note::
///    Any group indices (see e.g.
///    :attr:`~qiskit_fermions.operators.FermionOperator.groups`) are *not* preserved: the returned
///    operator has no groups, since a canonical reordering does not respect group boundaries.
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
///     TypeError: if ``op`` is not one of the supported operator types.
#[gen_stub_pyfunction(module = "qiskit_fermions.operators.terms.ordering.canonical")]
#[pyfunction(name = "canonical_order")]
pub fn py_canonical_order(op: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let py = op.py();
    if let Ok(fer_op) = op.extract::<PyFermionOperator>() {
        let inner = canonical_order(fer_op.inner);
        return Ok(PyFermionOperator { inner }
            .into_pyobject(py)?
            .into_any()
            .unbind());
    }
    if let Ok(maj_op) = op.extract::<PyMajoranaOperator>() {
        let inner = canonical_order(maj_op.inner);
        return Ok(PyMajoranaOperator { inner }
            .into_pyobject(py)?
            .into_any()
            .unbind());
    }
    if let Ok(edge_op) = op.extract::<PyEdgeVertexOperator>() {
        let inner = canonical_order(edge_op.inner);
        return Ok(PyEdgeVertexOperator { inner }
            .into_pyobject(py)?
            .into_any()
            .unbind());
    }
    if let Ok(transfer_op) = op.extract::<PyTransferVertexOperator>() {
        let inner = canonical_order(transfer_op.inner);
        return Ok(PyTransferVertexOperator { inner }
            .into_pyobject(py)?
            .into_any()
            .unbind());
    }
    Err(PyTypeError::new_err(
        "canonical_order expects a fermionic, Majorana, edge-vertex or transfer-vertex operator",
    ))
}

#[pymodule]
pub mod canonical {
    #[pymodule_export]
    use super::py_canonical_order;
}
