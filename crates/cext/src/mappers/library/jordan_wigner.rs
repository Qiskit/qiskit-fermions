// This code is a Qiskit project.
//
// (C) Copyright IBM 2026.
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use crate::pointers::const_ptr_as_ref;

use qiskit_fermions_core::mappers::library::jordan_wigner::jordan_wigner;
use qiskit_fermions_core::operators::fermion_operator::FermionOperator;

/// @ingroup qf_mapper_library
///
/// @brief Applies the Jordan-Wigner transformation to an operator.
///
/// @param op A pointer to the fermionic operator to be mapped.
/// @param num_qubits The number of qubits of the resulting operator.
///
/// @return A pointer to the created qubit operator.
///
/// @rst
///
/// Map a :c:struct:`QfFermionOperator` to a
/// :external+cqiskit:doc:`QkObs <cdoc/qk-obs>` under the Jordan-Wigner
/// transformation. [1]_
///
/// ----
///
/// Definition
/// ----------
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
/// .. [1] P. Jordan and E. Wigner, Über das Paulische Äquivalenzverbot,
///        Zeitschrift für Physik 47, No. 9. (1928), pp. 631–651,
///        `doi:10.1007/BF01331938 <https://link.springer.com/article/10.1007/BF01331938>`_.
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     // define some kind of fermionic operator
///     QfFermionOperator *hamil = qf_ferm_op_one();
///
///     // and map it to a qubit operator
///     QkObs *result = qf_jordan_wigner(hamil, 4);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_jordan_wigner(
    op: *const FermionOperator,
    num_qubits: u32,
) -> *mut qiskit_sys::QkObs {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let op = unsafe { const_ptr_as_ref(op) };

    jordan_wigner(op, num_qubits)
}
