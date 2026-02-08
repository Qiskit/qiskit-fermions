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

use crate::operators::fermion_operator::PyFermionOperator;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use qiskit_fermions_core::mappers::library::jordan_wigner::jordan_wigner;

/// Map a :class:`.FermionOperator` to a :class:`~qiskit.quantum_info.SparseObservable` under the
/// Jordan-Wigner transformation. [1]_
///
/// Args:
///     op: the fermionic operator to map.
///     num_qubits: the number of qubits for the resulting qubit operator.
///
/// Returns:
///     The mapped qubit operator.
///
/// ----
///
/// Definition
/// ==========
///
/// The Jordan-Wigner transformation maps fermionic creation and annihilation operators to spin (or
/// in this case, qubit) operators:
///
/// .. math::
///
///    a^\dagger_j \rightarrow \bigotimes_{k\lt j} \sigma^Z_k \otimes \sigma^+_j ~~\text{and}~~
///    a_j \rightarrow \bigotimes_{k\lt j} \sigma^Z_k \otimes \sigma^-_j \, ,
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
///     >>> from qiskit_fermions.mappers.library import jordan_wigner
///     >>> from qiskit_fermions.operators import FermionOperator
///     >>> fop = FermionOperator.from_dict(
///     ...     {
///     ...         (): 2.0,
///     ...         ((True, 0), (False, 0)): 0.1,
///     ...         ((True, 1), (False, 2), (True, 2), (False, 1)): -1.0j,
///     ...     }
///     ... )
///     >>> qop = jordan_wigner(fop, 4)
///     >>> qop.simplify()
///     <SparseObservable with 5 terms on 4 qubits: (2.05-0.25j)() + (-0.05+0j)(Z_0) + (0+0.25j)(Z_1) + (0+0.25j)(Z_2 Z_1) + (0-0.25j)(Z_2)>
///
/// ----
///
/// .. [1] P. Jordan and E. Wigner, Über das Paulische Äquivalenzverbot,
///        Zeitschrift für Physik 47, No. 9. (1928), pp. 631–651,
///        `doi:10.1007/BF01331938 <https://link.springer.com/article/10.1007/BF01331938>`_.
#[gen_stub_pyfunction(module = "qiskit_fermions.mappers.library.jordan_wigner")]
#[pyfunction(name = "jordan_wigner")]
#[gen_stub(override_return_type(type_repr="qiskit.quantum_info.SparseObservable", imports=("qiskit.quantum_info")))]
pub fn py_jordan_wigner(op: PyFermionOperator, num_qubits: u32) -> Py<PyAny> {
    let obs = jordan_wigner(&op.inner, num_qubits);
    unsafe {
        let py = Python::assume_attached();
        let py_obs = qiskit_sys::qk_obs_to_python(obs);
        Bound::from_owned_ptr(py, py_obs).into()
    }
}

#[pymodule]
pub mod jordan_wigner {
    #[pymodule_export]
    use super::py_jordan_wigner;
}
