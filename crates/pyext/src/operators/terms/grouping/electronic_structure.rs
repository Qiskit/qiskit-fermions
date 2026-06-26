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
use pyo3::{exceptions::PyValueError, prelude::*};
use pyo3_stub_gen::derive::*;
use qiskit_fermions_core::operators::terms::grouping::electronic_structure::group_terms_by_electronic_structure;

/// Groups the terms of an operator by their electronic structure.
///
/// This function automatically populates the
/// :attr:`~qiskit_fermions.operators.FermionOperator.groups` attribute of the provided
/// :class:`~qiskit_fermions.operators.FermionOperator` such that terms satisfying a symmetric
/// perturbation present in electronic-structure Hamiltonians are grouped.
///
/// .. caution::
///    The provided operator *must* be normal-ordered! This is an underlying assumption of the
///    implementation that is *not* being verified! See
///    :meth:`~qiskit_fermions.operators.FermionOperator.normal_ordered` for how to get an operator
///    of that form.
///
/// More concretely, given an electronic-structure Hamiltonian of the form
///
/// .. math::
///
///    \mathcal{H} = \sum_{ij} c_{ij} a^\dagger_i a_j
///         + \sum_{ijkl} c_{ijkl} a^\dagger_i a^\dagger_j a_k a_l \, ,
///
/// this function will group 1-body terms with permutational symmetry of ``(i, j)`` as well as the
/// 2-body terms with permutational symmetry of ``(i, j, k, l)``. For the 2-body terms, not all
/// permutations will be grouped. Instead, the ``two_body_physicist_order`` determines how the four
/// indices get grouped into pairs of two within which permutational symmetries exist:
///
/// - ``two_body_physicist_order=False`` (`default`): ``(i, l)`` and ``(j, k)``
/// - ``two_body_physicist_order=True``: ``(i, k)`` and ``(j, l)``
///
/// .. code-block:: python
///
///    from qiskit_fermions.operators import FermionOperator
///    from qiskit_fermions.operators.library import FCIDump
///
///    fcidump = FCIDump.from_file("molecule.fcidump")
///    operator = FermionOperator.from_fcidump(fcidump)
///
///    normal = op.normal_ordered().simplify(atol=0.0)
///
///    group_terms_by_electronic_structure(normal, 2 * fcidump.norb, two_body_physicist_order=False)
///
///    assert normal.groups is not None
///
/// Args:
///     op: the normal-ordered operator whose terms to group.
///     num_modes: the number of spin-less fermionic modes in the system.
///     two_body_physicist_order: whether the 2-body terms are stored in physicists order.
///
/// Raises:
///     ValueError: if an unexpected term is encountered.
#[gen_stub_pyfunction(module = "qiskit_fermions.operators.terms.grouping.electronic_structure")]
#[pyfunction(name = "group_terms_by_electronic_structure")]
#[pyo3(signature = (op, num_modes, *, two_body_physicist_order=false))]
pub fn py_group_terms_by_electronic_structure(
    op: &Bound<PyFermionOperator>,
    num_modes: u32,
    two_body_physicist_order: bool,
) -> PyResult<()> {
    let res = group_terms_by_electronic_structure(
        &mut op.borrow_mut().inner,
        num_modes,
        two_body_physicist_order,
    );
    match res {
        Ok(_) => Ok(()),
        Err(e) => Err(PyValueError::new_err(e.to_string())),
    }
}

#[pymodule]
pub mod electronic_structure {
    #[pymodule_export]
    use super::py_group_terms_by_electronic_structure;
}
