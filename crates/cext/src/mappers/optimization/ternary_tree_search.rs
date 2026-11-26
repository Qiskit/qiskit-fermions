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
use crate::pointers::{check_ptr, const_ptr_as_ref};

use qiskit_fermions_core::mappers::optimization::ternary_tree_search::hatt_tree;
use qiskit_fermions_core::mappers::ternary_tree::TernaryTree;
use qiskit_fermions_core::operators::majorana_operator::MajoranaOperator;

/// @ingroup qf_ternary_tree
///
/// @brief Grows a ternary tree adapted to a Hamiltonian, following the HATT algorithm.
///
/// @param hamiltonian A pointer to the Majorana operator to adapt to.
/// @param num_modes The number of fermionic modes, which is also the number of qubits.
/// @param weighted Whether to weight each term by the magnitude of its coefficient when scoring a
///        candidate merge. ``false`` counts terms and ignores their coefficients, as in the HATT
///        publication.
/// @param out A pointer to where the created tree will be written on success. It is left untouched if
///        the search fails.
///
/// @return An exit code. This is ``>0`` if an error occurred. In particular, a
///         ``QfExitCode_ValueError`` is returned if ``num_modes`` is zero or larger than a ternary
///         tree can hold, if ``hamiltonian`` acts on a mode the requested encoding does not cover,
///         or if a coefficient is infinite or NaN. A null or misaligned pointer is reported as
///         ``QfExitCode_NullPointerError`` or ``QfExitCode_AlignmentError``.
///
/// @rst
///
/// The algorithm is topology-blind: it shapes the tree around which Majorana operators appear
/// together in the Hamiltonian being mapped.
///
/// The tree is grown bottom-up. Each step picks the ``X`` and ``Z`` children of a new node to minimize
/// the Hamiltonian's Pauli weight on its qubit, and gives the ``Y`` link the subtree holding the
/// partner of the Majorana operator on the ``X`` side. That is the vacuum-preserving variant of the
/// algorithm, and it matches how :c:func:`qf_ternary_tree_encoding_new` pairs legs into Majorana
/// operators. Node ``j`` of the returned tree hosts mode ``j``.
///
/// Summed over all steps, the cost is the total Pauli weight of the terms of ``hamiltonian`` as given,
/// each term's image counted on its own: nothing is normal-ordered, merged or dropped first. If
/// ``hamiltonian`` is normal-ordered and simplified beforehand, its distinct terms map to distinct
/// Pauli strings, and the cost equals the total Pauli weight of the mapped operator after the same
/// simplification.
///
/// Ties between equal-cost candidates, which the reference leaves open, go to the lowest subtree
/// indices, legs first in Majorana order. So the result is deterministic, and relabeling the modes of
/// ``hamiltonian`` changes which candidate wins a tie.
///
/// .. code-block:: c
///
///     // gamma_0 gamma_3, one term of a hopping between modes 0 and 1.
///     uint32_t modes[2] = {0, 3};
///     QkComplex64 coeff = {0.0, 1.0};
///     QfMajoranaOperator *ham = qf_maj_op_zero();
///     qf_maj_op_add_term(ham, 2, modes, &coeff);
///
///     QfTernaryTree *tree;
///     QfExitCode code = qf_ternary_tree_hatt(ham, 2, false, &tree);
///     assert(code == QfExitCode_Success);
///
///     qf_ternary_tree_free(tree);
///     qf_maj_op_free(ham);
///
/// The tree that comes back is a valid encoding like any other, so it is passed to
/// :c:func:`qf_ternary_tree_encoding_new` and then to the mappers exactly as a
/// :c:func:`qf_ternary_tree_breadth_first` tree would be.
///
/// .. warning::
///
///     This is a greedy heuristic, and Pauli weight is not the circuit cost, so it does not always
///     win: the reference itself reports cases where Jordan-Wigner gives fewer CNOT gates or a
///     shallower circuit. Measure on your own Hamiltonian rather than assuming an improvement.
///
/// @endrst
///
/// # Safety
///
/// `hamiltonian` must be a valid pointer to a `QfMajoranaOperator`, and `out` must be non-null and
/// aligned. A null or misaligned pointer is reported rather than dereferenced.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ternary_tree_hatt(
    hamiltonian: *const MajoranaOperator,
    num_modes: u32,
    weighted: bool,
    out: *mut *mut TernaryTree,
) -> ExitCode {
    // Checked rather than left to `const_ptr_as_ref`, whose panic on a null pointer cannot unwind
    // out of an `extern "C"` function and would abort the caller instead.
    if let Err(err) = check_ptr(hamiltonian).and_then(|()| check_ptr(out)) {
        return err.into();
    }
    // SAFETY: Checked above to be non-null and aligned.
    let hamiltonian = unsafe { const_ptr_as_ref(hamiltonian) };

    match hatt_tree(hamiltonian, num_modes, weighted) {
        Ok(tree) => {
            // SAFETY: Checked above to be non-null and aligned.
            unsafe { out.write(Box::into_raw(Box::new(tree))) };
            ExitCode::Success
        }
        Err(_) => ExitCode::ValueError,
    }
}
