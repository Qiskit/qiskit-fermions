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

use qiskit_fermions_core::mappers::library::ternary_tree::{
    edge_vertex_ternary_tree, fermion_ternary_tree, majorana_ternary_tree,
    transfer_vertex_ternary_tree,
};

use super::into_py_obs;
use crate::mappers::ternary_tree::PyTernaryTreeEncoding;
use crate::operators::edge_vertex_operator::PyEdgeVertexOperator;
use crate::operators::fermion_operator::PyFermionOperator;
use crate::operators::majorana_operator::PyMajoranaOperator;
use crate::operators::transfer_vertex_operator::PyTransferVertexOperator;

/// Map a :class:`.FermionOperator` to a :class:`~qiskit.quantum_info.SparseObservable` under a
/// ternary-tree encoding.
///
/// Args:
///     op: the fermionic operator to map.
///     encoding: the compiled encoding to map through.
///     num_qubits: the number of qubits for the resulting qubit operator. Must be at least the
///         encoding's mode count; a larger value pads with the identity.
///
/// Returns:
///     The mapped qubit operator. The result is `not` guaranteed to be fully simplified; call
///     :meth:`~qiskit.quantum_info.SparseObservable.simplify` to combine any remaining duplicate
///     terms. Duplicates are merged as the result is assembled, to bound the memory required, so the
///     exact number of terms returned may vary with the number of threads used. Any grouping carried
///     by ``op`` is not reflected in the result, as a
///     :class:`~qiskit.quantum_info.SparseObservable` does not track groups.
///
/// Raises:
///     ValueError: if ``op`` acts on a mode the encoding does not cover.
///
/// Usage
/// =====
///
/// On a chain along ``Z`` this reproduces :func:`.fermion_jordan_wigner` exactly:
///
/// .. doctest::
///
///     >>> from qiskit_fermions.mappers import TernaryTree, TernaryTreeEncoding
///     >>> from qiskit_fermions.mappers.library import (
///     ...     fermion_jordan_wigner, fermion_ternary_tree,
///     ... )
///     >>> from qiskit_fermions.operators import FermionOperator
///     >>> fop = FermionOperator.from_dict({((True, 0), (False, 0)): 0.1})
///     >>> enc = TernaryTreeEncoding(TernaryTree.chain(2, "Z"))
///     >>> fermion_ternary_tree(fop, enc, 2).simplify()
///     <SparseObservable with 2 terms on 2 qubits: (0.05+0j)() + (-0.05+0j)(Z_0)>
///     >>> bool((fermion_ternary_tree(fop, enc, 2) - fermion_jordan_wigner(fop, 2)).simplify().num_terms == 0)
///     True
///
/// Other trees trade that encoding's Pauli weight, linear in the number of modes, for a shallower one.
#[gen_stub_pyfunction(module = "qiskit_fermions._lib.mappers.mappers_library.ternary_tree")]
#[pyfunction(name = "fermion_ternary_tree")]
#[gen_stub(override_return_type(type_repr="qiskit.quantum_info.SparseObservable", imports=("qiskit.quantum_info")))]
pub fn py_fermion_ternary_tree(
    op: &Bound<PyFermionOperator>,
    encoding: &PyTernaryTreeEncoding,
    num_qubits: u32,
) -> PyResult<Py<PyAny>> {
    // NOTE: borrowed rather than taken by value, as in the Jordan-Wigner bindings: the operator is
    // `Clone`, so extracting it would copy every term buffer purely to read it.
    let obs = fermion_ternary_tree(&op.borrow().inner, &encoding.inner, num_qubits)
        .map_err(crate::value_err)?;
    Ok(unsafe { into_py_obs(obs) })
}

/// Map a :class:`.MajoranaOperator` to a :class:`~qiskit.quantum_info.SparseObservable` under a
/// ternary-tree encoding.
///
/// Each Majorana operator's image is a single Pauli string, whatever the tree, so a term that is a
/// product of several of them composes those strings rather than inflating into a sum.
///
/// Args:
///     op: the Majorana operator to map.
///     encoding: the compiled encoding to map through.
///     num_qubits: the number of qubits for the resulting qubit operator. Must be at least the
///         encoding's mode count; a larger value pads with the identity.
///
/// Returns:
///     The mapped qubit operator, subject to the same simplification and grouping caveats as
///     :func:`.fermion_ternary_tree`.
///
/// Raises:
///     ValueError: if ``op`` acts on a mode the encoding does not cover. Note that Majorana index
///         :math:`m` acts on mode :math:`\lfloor m/2 \rfloor`.
///
/// Usage
/// =====
///
/// .. doctest::
///
///     >>> from qiskit_fermions.mappers import TernaryTree, TernaryTreeEncoding
///     >>> from qiskit_fermions.mappers.library import majorana_ternary_tree
///     >>> from qiskit_fermions.operators import MajoranaOperator
///     >>> mop = MajoranaOperator.from_dict({(0, 1): 1.0})
///     >>> enc = TernaryTreeEncoding(TernaryTree.breadth_first(2, 3))
///     >>> majorana_ternary_tree(mop, enc, 2).simplify()
///     <SparseObservable with 1 term on 2 qubits: (0+1j)(Z_1 Z_0)>
///
/// Note that the product of two Majorana strings carries a phase that depends on how their paths
/// overlap, and so on the tree.
#[gen_stub_pyfunction(module = "qiskit_fermions._lib.mappers.mappers_library.ternary_tree")]
#[pyfunction(name = "majorana_ternary_tree")]
#[gen_stub(override_return_type(type_repr="qiskit.quantum_info.SparseObservable", imports=("qiskit.quantum_info")))]
pub fn py_majorana_ternary_tree(
    op: &Bound<PyMajoranaOperator>,
    encoding: &PyTernaryTreeEncoding,
    num_qubits: u32,
) -> PyResult<Py<PyAny>> {
    let obs = majorana_ternary_tree(&op.borrow().inner, &encoding.inner, num_qubits)
        .map_err(crate::value_err)?;
    Ok(unsafe { into_py_obs(obs) })
}

/// Map an :class:`.EdgeVertexOperator` to a :class:`~qiskit.quantum_info.SparseObservable` under a
/// ternary-tree encoding.
///
/// Args:
///     op: the edge-vertex operator to map.
///     encoding: the compiled encoding to map through.
///     num_qubits: the number of qubits for the resulting qubit operator. Must be at least the
///         encoding's mode count; a larger value pads with the identity.
///
/// Returns:
///     The mapped qubit operator, subject to the same simplification and grouping caveats as
///     :func:`.fermion_ternary_tree`.
///
/// Raises:
///     ValueError: if ``op`` acts on a mode the encoding does not cover.
///
/// Note:
///     Unlike under Jordan-Wigner, a vertex operator is not in general the weight-1 Pauli
///     :math:`Z_l`: that is a property of the chain, where the two Majorana strings' :math:`Z` chains
///     cancel. On the balanced tree at :math:`n = 4` a vertex operator has weight 3.
#[gen_stub_pyfunction(module = "qiskit_fermions._lib.mappers.mappers_library.ternary_tree")]
#[pyfunction(name = "edge_vertex_ternary_tree")]
#[gen_stub(override_return_type(type_repr="qiskit.quantum_info.SparseObservable", imports=("qiskit.quantum_info")))]
pub fn py_edge_vertex_ternary_tree(
    op: &Bound<PyEdgeVertexOperator>,
    encoding: &PyTernaryTreeEncoding,
    num_qubits: u32,
) -> PyResult<Py<PyAny>> {
    let obs = edge_vertex_ternary_tree(&op.borrow().inner, &encoding.inner, num_qubits)
        .map_err(crate::value_err)?;
    Ok(unsafe { into_py_obs(obs) })
}

/// Map a :class:`.TransferVertexOperator` to a :class:`~qiskit.quantum_info.SparseObservable` under a
/// ternary-tree encoding.
///
/// Args:
///     op: the transfer-vertex operator to map.
///     encoding: the compiled encoding to map through.
///     num_qubits: the number of qubits for the resulting qubit operator. Must be at least the
///         encoding's mode count; a larger value pads with the identity.
///
/// Returns:
///     The mapped qubit operator, subject to the same simplification and grouping caveats as
///     :func:`.fermion_ternary_tree`.
///
/// Raises:
///     ValueError: if ``op`` acts on a mode the encoding does not cover.
#[gen_stub_pyfunction(module = "qiskit_fermions._lib.mappers.mappers_library.ternary_tree")]
#[pyfunction(name = "transfer_vertex_ternary_tree")]
#[gen_stub(override_return_type(type_repr="qiskit.quantum_info.SparseObservable", imports=("qiskit.quantum_info")))]
pub fn py_transfer_vertex_ternary_tree(
    op: &Bound<PyTransferVertexOperator>,
    encoding: &PyTernaryTreeEncoding,
    num_qubits: u32,
) -> PyResult<Py<PyAny>> {
    let obs = transfer_vertex_ternary_tree(&op.borrow().inner, &encoding.inner, num_qubits)
        .map_err(crate::value_err)?;
    Ok(unsafe { into_py_obs(obs) })
}

#[pymodule]
pub mod ternary_tree {
    #[pymodule_export]
    use super::py_fermion_ternary_tree;

    #[pymodule_export]
    use super::py_majorana_ternary_tree;

    #[pymodule_export]
    use super::py_edge_vertex_ternary_tree;

    #[pymodule_export]
    use super::py_transfer_vertex_ternary_tree;
}
