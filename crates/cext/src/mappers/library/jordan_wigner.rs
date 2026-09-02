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

use crate::exit_codes::ExitCode;
use crate::pointers::const_ptr_as_ref;

use qiskit_fermions_core::mappers::library::jordan_wigner::{
    edge_vertex_jordan_wigner, fermion_jordan_wigner, majorana_jordan_wigner,
    transfer_vertex_jordan_wigner,
};
use qiskit_fermions_core::operators::edge_vertex_operator::EdgeVertexOperator;
use qiskit_fermions_core::operators::fermion_operator::FermionOperator;
use qiskit_fermions_core::operators::majorana_operator::MajoranaOperator;
use qiskit_fermions_core::operators::transfer_vertex_operator::TransferVertexOperator;

/// @ingroup qf_mapper_library
///
/// @brief Applies the Jordan-Wigner transformation to an operator.
///
/// @param op A pointer to the fermionic operator to be mapped.
/// @param num_qubits The number of qubits of the resulting operator. This must be strictly greater
///        than the largest mode index acted upon by ``op``.
/// @param out A pointer to where the created qubit operator will be written on success. It is left
///        untouched if the transformation fails.
///
/// @return An exit code. This is ``>0`` if an error occurred. In particular, a
///         ``QfExitCode_ValueError`` is returned if ``num_qubits`` is too small to hold the
///         operator's support.
///
/// @rst
///
/// Map a :c:struct:`QfFermionOperator` to a
/// :external+cqiskit:doc:`QkObs <cdoc/qk-obs>` under the Jordan-Wigner
/// transformation. [JW-ferm]_
///
/// Definition
/// ----------
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
/// .. [JW-ferm] P. Jordan and E. Wigner, Über das Paulische Äquivalenzverbot,
///        Zeitschrift für Physik 47, No. 9. (1928), pp. 631–651,
///        `doi:10.1007/BF01331938 <https://link.springer.com/article/10.1007/BF01331938>`_.
///
/// Memory usage
/// ------------
///
/// The result is not guaranteed to be fully simplified: duplicate terms are merged as it is
/// assembled, to bound the memory required, but some may remain. Call ``qk_obs_canonicalize`` if you
/// need them all combined. The exact number of terms returned may therefore vary with the number of
/// threads used, which does not affect the operator that the result represents.
///
/// This mapping is parallelized for speed, and that choice costs memory. Each worker thread
/// accumulates into an observable of its own, and the terms are handed to whichever thread is free
/// rather than partitioned by which Pauli strings they produce, so every thread ends up holding
/// roughly a full copy of the mapped operator. Peak memory therefore grows with the number of
/// threads: expect on the order of the mapped operator's size *times* the thread count, plus the
/// input operator.
///
/// If memory matters more than wall-clock time, reduce the thread count through rayon's
/// ``RAYON_NUM_THREADS`` environment variable -- peak memory falls roughly in proportion, and the
/// mapping takes correspondingly longer -- or map the operator in batches and add the partial results
/// together yourself, which bounds the peak by the batch size at the cost of repeating the merging
/// work.
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
///     QkObs *result;
///     QfExitCode exit = qf_ferm_op_jordan_wigner(hamil, 4, &result);
///
///     assert(exit == QfExitCode_Success);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ferm_op_jordan_wigner(
    op: *const FermionOperator,
    num_qubits: u32,
    out: *mut *mut qiskit_sys::QkObs,
) -> ExitCode {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let op = unsafe { const_ptr_as_ref(op) };

    match fermion_jordan_wigner(op, num_qubits) {
        Ok(obs) => {
            // SAFETY: Per documentation, `out` is non-null and aligned.
            unsafe { out.write(obs) };
            ExitCode::Success
        }
        Err(_) => ExitCode::ValueError,
    }
}

/// @ingroup qf_mapper_library
///
/// @brief Applies the Jordan-Wigner transformation to a Majorana operator.
///
/// @param op A pointer to the Majorana operator to be mapped.
/// @param num_qubits The number of qubits of the resulting operator. Note that this is counted in
///        *fermionic* modes, so it must be strictly greater than the largest Majorana index acted
///        upon by ``op`` divided by two.
/// @param out A pointer to where the created qubit operator will be written on success. It is left
///        untouched if the transformation fails.
///
/// @return An exit code. This is ``>0`` if an error occurred. In particular, a
///         ``QfExitCode_ValueError`` is returned if ``num_qubits`` is too small to hold the
///         operator's support.
///
/// @rst
///
/// Map a :c:struct:`QfMajoranaOperator` to a
/// :external+cqiskit:doc:`QkObs <cdoc/qk-obs>` under the Jordan-Wigner
/// transformation. [JW-maj]_
///
/// Definition
/// ----------
///
/// With the :c:struct:`QfMajoranaOperator` convention that even indices carry
/// :math:`\gamma_j = a^\dagger_j + a_j` and odd ones :math:`\gamma'_j = i(a^\dagger_j - a_j)`, the
/// Majorana index :math:`m` acts on the fermionic mode :math:`\lfloor m/2 \rfloor` and maps onto a
/// single Pauli string,
///
/// .. math::
///
///    \gamma_j \rightarrow \bigotimes_{k\lt j} \sigma^Z_k \otimes \sigma^X_j ~~\text{and}~~
///    \gamma'_j \rightarrow \bigotimes_{k\lt j} \sigma^Z_k \otimes \sigma^Y_j \, .
///
/// This also avoids an intermediate blowup relative to converting to a
/// :c:struct:`QfFermionOperator` first: each fermionic action maps onto a *two*-term sum, so that
/// route inflates a single Pauli string into up to :math:`4^L` terms for a term built from :math:`L`
/// Majorana operators. The saving grows with the length of the terms; for single-operator terms the
/// two routes cost about the same.
///
/// .. [JW-maj] P. Jordan and E. Wigner, Über das Paulische Äquivalenzverbot,
///        Zeitschrift für Physik 47, No. 9. (1928), pp. 631–651,
///        `doi:10.1007/BF01331938 <https://link.springer.com/article/10.1007/BF01331938>`_.
///
/// Memory usage
/// ------------
///
/// See :c:func:`qf_ferm_op_jordan_wigner`; the same parallelization and merging behaviour applies.
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     // define some kind of Majorana operator
///     QfMajoranaOperator *hamil = qf_maj_op_one();
///
///     // and map it to a qubit operator
///     QkObs *result;
///     QfExitCode exit = qf_maj_op_jordan_wigner(hamil, 4, &result);
///
///     assert(exit == QfExitCode_Success);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_maj_op_jordan_wigner(
    op: *const MajoranaOperator,
    num_qubits: u32,
    out: *mut *mut qiskit_sys::QkObs,
) -> ExitCode {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let op = unsafe { const_ptr_as_ref(op) };

    match majorana_jordan_wigner(op, num_qubits) {
        Ok(obs) => {
            // SAFETY: Per documentation, `out` is non-null and aligned.
            unsafe { out.write(obs) };
            ExitCode::Success
        }
        Err(_) => ExitCode::ValueError,
    }
}

/// @ingroup qf_mapper_library
///
/// @brief Applies the Jordan-Wigner transformation to an edge-vertex operator.
///
/// @param op A pointer to the edge-vertex operator to be mapped.
/// @param num_qubits The number of qubits of the resulting operator. This must be strictly greater
///        than the largest mode index acted upon by ``op``.
/// @param out A pointer to where the created qubit operator will be written on success. It is left
///        untouched if the transformation fails.
///
/// @return An exit code. This is ``>0`` if an error occurred. In particular, a
///         ``QfExitCode_ValueError`` is returned if ``num_qubits`` is too small to hold the
///         operator's support.
///
/// @rst
///
/// Map a :c:struct:`QfEdgeVertexOperator` to a
/// :external+cqiskit:doc:`QkObs <cdoc/qk-obs>` under the Jordan-Wigner
/// transformation. [JW-edge]_
///
/// Definition
/// ----------
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
/// where the sign is negative for :math:`l \lt r` and positive otherwise. The :math:`\sigma^Z` chains
/// of the two underlying Majorana operators cancel below the lower index, which is why the
/// :math:`\sigma^Z` string spans only the modes strictly *between* the two endpoints.
///
/// .. note::
///    Reversing the two indices leaves the Pauli string unchanged and flips only the sign, which is
///    the antisymmetry :math:`E_{lr} = -E_{rl}`. Contrast
///    :c:func:`qf_transfer_op_jordan_wigner`, where the coefficient is the same for both
///    orientations and the Pauli letters change instead.
///
/// .. note::
///    These Pauli strings differ from those in Eq. (10) of [Gandon-edge]_ by an exchange of :math:`\sigma^X`
///    and :math:`\sigma^Y` on the two endpoints. This is a single-qubit basis choice -- both
///    conventions satisfy every defining relation of the algebra -- and the one used here is the one
///    consistent with :c:func:`qf_edge_vertex_to_fermion`, so that mapping an operator directly
///    agrees with converting it to a :c:struct:`QfFermionOperator` first.
///
/// .. [JW-edge] P. Jordan and E. Wigner, Über das Paulische Äquivalenzverbot,
///        Zeitschrift für Physik 47, No. 9. (1928), pp. 631–651,
///        `doi:10.1007/BF01331938 <https://link.springer.com/article/10.1007/BF01331938>`_.
/// .. [Gandon-edge] L. Gandon et al., Fermionic quantum simulation with flow sets,
///        `arXiv:2512.11418 <https://arxiv.org/abs/2512.11418>`_.
///
/// Memory usage
/// ------------
///
/// See :c:func:`qf_ferm_op_jordan_wigner`; the same parallelization and merging behaviour applies.
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     // define some kind of edge-vertex operator
///     QfEdgeVertexOperator *hamil = qf_edge_op_one();
///
///     // and map it to a qubit operator
///     QkObs *result;
///     QfExitCode exit = qf_edge_op_jordan_wigner(hamil, 4, &result);
///
///     assert(exit == QfExitCode_Success);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_edge_op_jordan_wigner(
    op: *const EdgeVertexOperator,
    num_qubits: u32,
    out: *mut *mut qiskit_sys::QkObs,
) -> ExitCode {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let op = unsafe { const_ptr_as_ref(op) };

    match edge_vertex_jordan_wigner(op, num_qubits) {
        Ok(obs) => {
            // SAFETY: Per documentation, `out` is non-null and aligned.
            unsafe { out.write(obs) };
            ExitCode::Success
        }
        Err(_) => ExitCode::ValueError,
    }
}

/// @ingroup qf_mapper_library
///
/// @brief Applies the Jordan-Wigner transformation to a transfer-vertex operator.
///
/// @param op A pointer to the transfer-vertex operator to be mapped.
/// @param num_qubits The number of qubits of the resulting operator. This must be strictly greater
///        than the largest mode index acted upon by ``op``.
/// @param out A pointer to where the created qubit operator will be written on success. It is left
///        untouched if the transformation fails.
///
/// @return An exit code. This is ``>0`` if an error occurred. In particular, a
///         ``QfExitCode_ValueError`` is returned if ``num_qubits`` is too small to hold the
///         operator's support.
///
/// @rst
///
/// Map a :c:struct:`QfTransferVertexOperator` to a
/// :external+cqiskit:doc:`QkObs <cdoc/qk-obs>` under the Jordan-Wigner
/// transformation. [JW-transfer]_
///
/// Definition
/// ----------
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
///    The index order works the opposite way round to :c:func:`qf_edge_op_jordan_wigner`: the
///    coefficient is :math:`-1/2` for **both** orientations and it is the Pauli letters that swap.
///    :math:`T_{lr}` and :math:`T_{rl}` are genuinely different operators, with no antisymmetry
///    relating them.
///
/// .. note::
///    As for :c:func:`qf_edge_op_jordan_wigner`, these Pauli strings differ from Eq. (10) of [Gandon-transfer]_ by
///    a single-qubit basis choice; the convention used here is the one consistent with
///    :c:func:`qf_transfer_vertex_to_fermion`.
///
/// .. [JW-transfer] P. Jordan and E. Wigner, Über das Paulische Äquivalenzverbot,
///        Zeitschrift für Physik 47, No. 9. (1928), pp. 631–651,
///        `doi:10.1007/BF01331938 <https://link.springer.com/article/10.1007/BF01331938>`_.
/// .. [Gandon-transfer] L. Gandon et al., Fermionic quantum simulation with flow sets,
///        `arXiv:2512.11418 <https://arxiv.org/abs/2512.11418>`_.
///
/// Memory usage
/// ------------
///
/// See :c:func:`qf_ferm_op_jordan_wigner`; the same parallelization and merging behaviour applies.
///
/// Example
/// -------
///
/// .. code-block:: c
///     :linenos:
///
///     // define some kind of transfer-vertex operator
///     QfTransferVertexOperator *hamil = qf_transfer_op_one();
///
///     // and map it to a qubit operator
///     QkObs *result;
///     QfExitCode exit = qf_transfer_op_jordan_wigner(hamil, 4, &result);
///
///     assert(exit == QfExitCode_Success);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_transfer_op_jordan_wigner(
    op: *const TransferVertexOperator,
    num_qubits: u32,
    out: *mut *mut qiskit_sys::QkObs,
) -> ExitCode {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let op = unsafe { const_ptr_as_ref(op) };

    match transfer_vertex_jordan_wigner(op, num_qubits) {
        Ok(obs) => {
            // SAFETY: Per documentation, `out` is non-null and aligned.
            unsafe { out.write(obs) };
            ExitCode::Success
        }
        Err(_) => ExitCode::ValueError,
    }
}
