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

use crate::operators::edge_vertex_operator::PyEdgeVertexOperator;
use crate::operators::fermion_operator::PyFermionOperator;
use crate::operators::majorana_operator::PyMajoranaOperator;
use crate::operators::transfer_vertex_operator::PyTransferVertexOperator;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use qiskit_fermions_core::mappers::library::jordan_wigner::{
    edge_vertex_jordan_wigner, fermion_jordan_wigner, majorana_jordan_wigner,
    transfer_vertex_jordan_wigner,
};
use qiskit_pyo3_ffi as ffi;

/// Converts a mapped `QkObs` into the Python `SparseObservable` that owns it.
///
/// # Safety
///
/// `obs` must be a valid, uniquely-owned `QkObs` pointer; ownership transfers to Python.
unsafe fn into_py_obs(obs: *mut qiskit_pyo3_ffi::QkObs) -> Py<PyAny> {
    unsafe {
        let py = Python::assume_attached();
        let py_obs = ffi::qk_obs_to_python(obs);
        Bound::from_owned_ptr(py, py_obs).into()
    }
}

/// Map a :class:`.FermionOperator` to a :class:`~qiskit.quantum_info.SparseObservable` under the
/// Jordan-Wigner transformation. [1]_
///
/// Fermionic mode :math:`j` is mapped to qubit :math:`j` of the resulting
/// :class:`~qiskit.quantum_info.SparseObservable` (i.e. the identity is used on any qubit outside
/// the operator's support). This follows Qiskit's little-endian qubit ordering, where the qubit
/// index in a Pauli label such as ``X_2 Z_1 Z_0`` is the mode index.
///
/// Args:
///     op: the fermionic operator to map.
///     num_qubits: the number of qubits for the resulting qubit operator. This must be strictly
///         greater than the largest mode index in ``op`` (any additional qubits are padded with the
///         identity).
///
/// Returns:
///     The mapped qubit operator. The result is `not` guaranteed to be fully simplified; call
///     :meth:`~qiskit.quantum_info.SparseObservable.simplify` to combine any remaining duplicate
///     terms. Duplicates are merged as the result is assembled, to bound the memory required, so
///     the exact number of terms returned may vary with the number of threads used. This does not
///     affect the operator that the result represents.
///
/// Raises:
///     ValueError: if ``num_qubits`` is too small to hold the operator's support, i.e. if it is
///         not larger than the largest mode index acted upon by ``op``.
///
/// Memory usage
/// ============
///
/// This mapping is parallelized for speed, and that choice costs memory. Each worker thread
/// accumulates into an observable of its own, and the terms are handed to whichever thread is free
/// rather than partitioned by which Pauli strings they produce, so every thread ends up holding
/// roughly a full copy of the mapped operator. Peak memory therefore grows with the number of
/// threads: expect on the order of the mapped operator's size *times* the thread count, plus the
/// input operator.
///
/// If memory matters more than wall-clock time, there are two things to reach for:
///
/// * Reduce the thread count, using rayon's ``RAYON_NUM_THREADS`` environment variable. Peak memory
///   falls roughly in proportion, and the mapping takes correspondingly longer.
/// * Map the operator in batches and combine the results yourself, which bounds the peak by the
///   batch size at the cost of repeating the merging work:
///
///   .. code-block:: python
///
///      import itertools
///
///      terms = op.iter_terms()
///      total = None
///      while batch := list(itertools.islice(terms, 100_000)):
///          partial = fermion_jordan_wigner(FermionOperator.from_terms(batch), num_qubits)
///          partial = partial.simplify()
///          total = partial if total is None else (total + partial).simplify()
///
///   Note that the terms are streamed rather than materialized: ``list(op.iter_terms())`` on a
///   large operator costs more than the mapping itself.
///
/// Definition
/// ==========
///
/// The Jordan-Wigner transformation maps fermionic creation and annihilation operators to spin (or
/// in this case, qubit) operators:
///
/// .. math::
///
///    a^\dagger_j \rightarrow \bigotimes_{k\lt j} \sigma^Z_k \otimes \sigma^-_j ~~\text{and}~~
///    a_j \rightarrow \bigotimes_{k\lt j} \sigma^Z_k \otimes \sigma^+_j \, ,
///
/// where :math:`a^\dagger_j` (:math:`a_j`) is the fermionic creation (annihilation) operator
/// acting on the :math:`j`-th spin-less fermionic mode, :math:`\sigma^P` with
/// :math:`P \in \{X,Y,Z\}` are the spin-:math:`\frac{1}{2}` Pauli operators and
/// :math:`\sigma^\pm = (\sigma^X \pm \mathrm{i} \sigma^Y) / 2`.
///
/// This mapping preserves the fermionic anti-commutation relations by introducing a chain of
/// :math:`\sigma^Z` operators on all qubits preceding the acted-upon index :math:`j`.
///
/// Usage
/// =====
///
/// Since a :class:`.FermionOperator` does not determine a fixed number of modes which it acts
/// upon, one can specify the number of qubits to map onto when calling this function.
///
/// .. doctest::
///
///     >>> from qiskit_fermions.mappers.library import fermion_jordan_wigner
///     >>> from qiskit_fermions.operators import FermionOperator
///     >>> fop = FermionOperator.from_dict(
///     ...     {
///     ...         (): 2.0,
///     ...         ((True, 0), (False, 0)): 0.1,
///     ...         ((True, 1), (False, 2), (True, 2), (False, 1)): -1.0j,
///     ...     }
///     ... )
///     >>> qop = fermion_jordan_wigner(fop, 4)
///     >>> qop.simplify()
///     <SparseObservable with 5 terms on 4 qubits: (2.05-0.25j)() + (-0.05+0j)(Z_0) + (0+0.25j)(Z_1) + (0+0.25j)(Z_2 Z_1) + (0-0.25j)(Z_2)>
///
/// .. [1] P. Jordan and E. Wigner, Über das Paulische Äquivalenzverbot,
///        Zeitschrift für Physik 47, No. 9. (1928), pp. 631–651,
///        `doi:10.1007/BF01331938 <https://link.springer.com/article/10.1007/BF01331938>`_.
#[gen_stub_pyfunction(module = "qiskit_fermions._lib.mappers.mappers_library.jordan_wigner")]
#[pyfunction(name = "fermion_jordan_wigner")]
#[gen_stub(override_return_type(type_repr="qiskit.quantum_info.SparseObservable", imports=("qiskit.quantum_info")))]
pub fn py_fermion_jordan_wigner(
    op: &Bound<PyFermionOperator>,
    num_qubits: u32,
) -> PyResult<Py<PyAny>> {
    // NOTE: borrowed rather than taken by value. `PyFermionOperator` is `Clone`, so extracting it
    // by value would copy every term buffer of the input operator purely to read it.
    let obs = fermion_jordan_wigner(&op.borrow().inner, num_qubits).map_err(crate::value_err)?;
    Ok(unsafe { into_py_obs(obs) })
}

/// Map a :class:`.MajoranaOperator` to a :class:`~qiskit.quantum_info.SparseObservable` under the
/// Jordan-Wigner transformation. [1]_
///
/// Majorana mode :math:`m` acts on the fermionic mode :math:`\lfloor m/2 \rfloor`, which is mapped to
/// the qubit of the same index. This follows Qiskit's little-endian qubit ordering.
///
/// Args:
///     op: the Majorana operator to map.
///     num_qubits: the number of qubits for the resulting qubit operator. Note that this is counted
///         in *fermionic* modes, so it must be strictly greater than the largest Majorana index in
///         ``op`` divided by two (any additional qubits are padded with the identity).
///
/// Returns:
///     The mapped qubit operator. The result is `not` guaranteed to be fully simplified; call
///     :meth:`~qiskit.quantum_info.SparseObservable.simplify` to combine any remaining duplicate
///     terms. Duplicates are merged as the result is assembled, to bound the memory required, so
///     the exact number of terms returned may vary with the number of threads used. This does not
///     affect the operator that the result represents.
///
/// Raises:
///     ValueError: if ``num_qubits`` is too small to hold the operator's support, i.e. if it is not
///         larger than the largest *fermionic* mode acted upon by ``op``.
///
/// Definition
/// ==========
///
/// With the :class:`.MajoranaOperator` convention that even indices carry
/// :math:`\gamma_j = a^\dagger_j + a_j` and odd ones :math:`\gamma'_j = i(a^\dagger_j - a_j)`, a
/// single Majorana operator maps onto a single Pauli string,
///
/// .. math::
///
///    \gamma_j \rightarrow \bigotimes_{k\lt j} \sigma^Z_k \otimes \sigma^X_j ~~\text{and}~~
///    \gamma'_j \rightarrow \bigotimes_{k\lt j} \sigma^Z_k \otimes \sigma^Y_j \, .
///
/// This is also what makes it cheaper than converting to a :class:`.FermionOperator` first and
/// calling :func:`.fermion_jordan_wigner`: each fermionic action maps onto a *two*-term sum, so that
/// route inflates a single Pauli string into up to :math:`4^L` terms for a term built from :math:`L`
/// Majorana operators, before merging them back down. The saving therefore grows with the length of
/// the terms; for single-operator terms there is no blowup to avoid and the two routes cost about the
/// same.
///
/// Usage
/// =====
///
/// .. doctest::
///
///     >>> from qiskit_fermions.mappers.library import majorana_jordan_wigner
///     >>> from qiskit_fermions.operators import MajoranaOperator, gamma
///     >>> mop = MajoranaOperator.from_dict(
///     ...     {
///     ...         (gamma(0, False), gamma(1, False)): 0.5,
///     ...         (gamma(0, True), gamma(1, True)): 0.5,
///     ...     }
///     ... )
///     >>> majorana_jordan_wigner(mop, 2).simplify()
///     <SparseObservable with 2 terms on 2 qubits: (0+0.5j)(Y_1 X_0) + (0-0.5j)(X_1 Y_0)>
///
/// .. [1] P. Jordan and E. Wigner, Über das Paulische Äquivalenzverbot,
///        Zeitschrift für Physik 47, No. 9. (1928), pp. 631–651,
///        `doi:10.1007/BF01331938 <https://link.springer.com/article/10.1007/BF01331938>`_.
#[gen_stub_pyfunction(module = "qiskit_fermions._lib.mappers.mappers_library.jordan_wigner")]
#[pyfunction(name = "majorana_jordan_wigner")]
#[gen_stub(override_return_type(type_repr="qiskit.quantum_info.SparseObservable", imports=("qiskit.quantum_info")))]
pub fn py_majorana_jordan_wigner(
    op: &Bound<PyMajoranaOperator>,
    num_qubits: u32,
) -> PyResult<Py<PyAny>> {
    // NOTE: borrowed rather than taken by value, as in `py_fermion_jordan_wigner` above.
    let obs = majorana_jordan_wigner(&op.borrow().inner, num_qubits).map_err(crate::value_err)?;
    Ok(unsafe { into_py_obs(obs) })
}

/// Map an :class:`.EdgeVertexOperator` to a :class:`~qiskit.quantum_info.SparseObservable` under the
/// Jordan-Wigner transformation. [1]_
///
/// Fermionic mode :math:`j` is mapped to qubit :math:`j` of the resulting
/// :class:`~qiskit.quantum_info.SparseObservable`, following Qiskit's little-endian qubit ordering.
///
/// Args:
///     op: the edge-vertex operator to map.
///     num_qubits: the number of qubits for the resulting qubit operator. This must be strictly
///         greater than the largest mode index in ``op`` (any additional qubits are padded with the
///         identity).
///
/// Returns:
///     The mapped qubit operator. The result is `not` guaranteed to be fully simplified; call
///     :meth:`~qiskit.quantum_info.SparseObservable.simplify` to combine any remaining duplicate
///     terms. Duplicates are merged as the result is assembled, to bound the memory required, so
///     the exact number of terms returned may vary with the number of threads used. This does not
///     affect the operator that the result represents.
///
/// Raises:
///     ValueError: if ``num_qubits`` is too small to hold the operator's support, i.e. if it is
///         not larger than the largest mode index acted upon by ``op``.
///
/// Definition
/// ==========
///
/// Writing :math:`l_\text{min}` and :math:`l_\text{max}` for the smaller and larger of the two
/// indices, the generalized edge operators map onto single Pauli strings,
///
/// .. math::
///
///    \begin{align}
///    V_l = E_{ll} &\rightarrow \sigma^Z_l \, , \nonumber \\
///    E_{lr} &\rightarrow \mp \, \sigma^Y_{l_\text{min}}
///            \left( \bigotimes_{l_\text{min} \lt k \lt l_\text{max}} \sigma^Z_k \right)
///            \sigma^X_{l_\text{max}} \nonumber
///    \end{align}
///
/// where the sign is negative for :math:`l \lt r` and positive otherwise. The :math:`\sigma^Z`
/// chains of the two underlying Majorana operators cancel below the lower index, which is why the
/// :math:`\sigma^Z` string spans only the modes strictly *between* the two endpoints.
///
/// .. note::
///    Reversing the two indices leaves the Pauli string unchanged and flips only the sign, which is
///    the antisymmetry :math:`E_{lr} = -E_{rl}`. Contrast :func:`.transfer_vertex_jordan_wigner`,
///    where the coefficient is the same for both orientations and the Pauli letters change instead.
///
/// .. note::
///    These Pauli strings differ from those in Eq. (10) of [2]_ by an exchange of :math:`\sigma^X`
///    and :math:`\sigma^Y` on the two endpoints. This is a single-qubit basis choice -- both
///    conventions satisfy every defining relation of the algebra -- and the one used here is the one
///    consistent with :func:`.edge_vertex_to_fermion`, so that mapping an operator directly agrees
///    with converting it to a :class:`.FermionOperator` first.
///
/// Mapping directly also avoids an intermediate blowup: each fermionic action maps onto a *two*-term
/// sum, so routing a term built from :math:`L` edge operators through a :class:`.FermionOperator`
/// inflates a single Pauli string into up to :math:`4^L` terms before merging them back down. The
/// saving grows with the length of the terms; for single-operator terms the two routes cost about the
/// same.
///
/// Usage
/// =====
///
/// .. doctest::
///
///     >>> from qiskit_fermions.mappers.library import edge_vertex_jordan_wigner
///     >>> from qiskit_fermions.operators import EdgeVertexOperator
///     >>> eop = EdgeVertexOperator.from_dict({((0, 0),): 2.0, ((0, 1),): 0.5})
///     >>> edge_vertex_jordan_wigner(eop, 2).simplify()
///     <SparseObservable with 2 terms on 2 qubits: (2+0j)(Z_0) + (-0.5+0j)(X_1 Y_0)>
///
/// .. [1] P. Jordan and E. Wigner, Über das Paulische Äquivalenzverbot,
///        Zeitschrift für Physik 47, No. 9. (1928), pp. 631–651,
///        `doi:10.1007/BF01331938 <https://link.springer.com/article/10.1007/BF01331938>`_.
/// .. [2] L. Gandon et al., Fermionic quantum simulation with flow sets,
///        `arXiv:2512.11418 <https://arxiv.org/abs/2512.11418>`_.
#[gen_stub_pyfunction(module = "qiskit_fermions._lib.mappers.mappers_library.jordan_wigner")]
#[pyfunction(name = "edge_vertex_jordan_wigner")]
#[gen_stub(override_return_type(type_repr="qiskit.quantum_info.SparseObservable", imports=("qiskit.quantum_info")))]
pub fn py_edge_vertex_jordan_wigner(
    op: &Bound<PyEdgeVertexOperator>,
    num_qubits: u32,
) -> PyResult<Py<PyAny>> {
    // NOTE: borrowed rather than taken by value, as in `py_fermion_jordan_wigner` above.
    let obs =
        edge_vertex_jordan_wigner(&op.borrow().inner, num_qubits).map_err(crate::value_err)?;
    Ok(unsafe { into_py_obs(obs) })
}

/// Map a :class:`.TransferVertexOperator` to a :class:`~qiskit.quantum_info.SparseObservable` under
/// the Jordan-Wigner transformation. [1]_
///
/// Fermionic mode :math:`j` is mapped to qubit :math:`j` of the resulting
/// :class:`~qiskit.quantum_info.SparseObservable`, following Qiskit's little-endian qubit ordering.
///
/// Args:
///     op: the transfer-vertex operator to map.
///     num_qubits: the number of qubits for the resulting qubit operator. This must be strictly
///         greater than the largest mode index in ``op`` (any additional qubits are padded with the
///         identity).
///
/// Returns:
///     The mapped qubit operator. The result is `not` guaranteed to be fully simplified; call
///     :meth:`~qiskit.quantum_info.SparseObservable.simplify` to combine any remaining duplicate
///     terms. Duplicates are merged as the result is assembled, to bound the memory required, so
///     the exact number of terms returned may vary with the number of threads used. This does not
///     affect the operator that the result represents.
///
/// Raises:
///     ValueError: if ``num_qubits`` is too small to hold the operator's support, i.e. if it is
///         not larger than the largest mode index acted upon by ``op``.
///
/// Definition
/// ==========
///
/// Writing :math:`l_\text{min}` and :math:`l_\text{max}` for the smaller and larger of the two
/// indices, the generalized transfer operators map onto single Pauli strings,
///
/// .. math::
///
///    \begin{align}
///    V_l = T_{ll} &\rightarrow \sigma^Z_l \, , \nonumber \\
///    T_{lr} &\rightarrow -\frac{1}{2} \, \sigma^P_{l_\text{min}}
///            \left( \bigotimes_{l_\text{min} \lt k \lt l_\text{max}} \sigma^Z_k \right)
///            \sigma^P_{l_\text{max}} \nonumber
///    \end{align}
///
/// where :math:`P = X` for :math:`l \lt r` and :math:`P = Y` otherwise.
///
/// .. note::
///    The index order works the opposite way round to :func:`.edge_vertex_jordan_wigner`: the
///    coefficient is :math:`-1/2` for **both** orientations and it is the Pauli letters that swap.
///    :math:`T_{lr}` and :math:`T_{rl}` are genuinely different operators, with no antisymmetry
///    relating them.
///
/// .. note::
///    As for :func:`.edge_vertex_jordan_wigner`, these Pauli strings differ from Eq. (10) of [2]_ by
///    a single-qubit basis choice; the convention used here is the one consistent with
///    :func:`.transfer_vertex_to_fermion`.
///
/// Mapping directly also avoids an intermediate blowup: each fermionic action maps onto a *two*-term
/// sum, so routing a term built from :math:`L` transfer operators through a
/// :class:`.FermionOperator` inflates a single Pauli string into up to :math:`4^L` terms. The saving
/// grows with the length of the terms; for single-operator terms the two routes cost about the same.
///
/// Usage
/// =====
///
/// .. doctest::
///
///     >>> from qiskit_fermions.mappers.library import transfer_vertex_jordan_wigner
///     >>> from qiskit_fermions.operators import TransferVertexOperator
///     >>> top = TransferVertexOperator.from_dict({((0, 1),): 1.0, ((1, 0),): 1.0})
///     >>> transfer_vertex_jordan_wigner(top, 2).simplify()
///     <SparseObservable with 2 terms on 2 qubits: (-0.5+0j)(X_1 X_0) + (-0.5+0j)(Y_1 Y_0)>
///
/// .. [1] P. Jordan and E. Wigner, Über das Paulische Äquivalenzverbot,
///        Zeitschrift für Physik 47, No. 9. (1928), pp. 631–651,
///        `doi:10.1007/BF01331938 <https://link.springer.com/article/10.1007/BF01331938>`_.
/// .. [2] L. Gandon et al., Fermionic quantum simulation with flow sets,
///        `arXiv:2512.11418 <https://arxiv.org/abs/2512.11418>`_.
#[gen_stub_pyfunction(module = "qiskit_fermions._lib.mappers.mappers_library.jordan_wigner")]
#[pyfunction(name = "transfer_vertex_jordan_wigner")]
#[gen_stub(override_return_type(type_repr="qiskit.quantum_info.SparseObservable", imports=("qiskit.quantum_info")))]
pub fn py_transfer_vertex_jordan_wigner(
    op: &Bound<PyTransferVertexOperator>,
    num_qubits: u32,
) -> PyResult<Py<PyAny>> {
    // NOTE: borrowed rather than taken by value, as in `py_fermion_jordan_wigner` above.
    let obs =
        transfer_vertex_jordan_wigner(&op.borrow().inner, num_qubits).map_err(crate::value_err)?;
    Ok(unsafe { into_py_obs(obs) })
}

#[pymodule]
pub mod jordan_wigner {
    #[pymodule_export]
    use super::py_fermion_jordan_wigner;

    #[pymodule_export]
    use super::py_majorana_jordan_wigner;

    #[pymodule_export]
    use super::py_edge_vertex_jordan_wigner;

    #[pymodule_export]
    use super::py_transfer_vertex_jordan_wigner;
}
