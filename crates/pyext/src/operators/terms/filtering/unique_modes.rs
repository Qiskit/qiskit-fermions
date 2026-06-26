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
use qiskit_fermions_core::operators::terms::filtering::unique_modes::filter_terms_by_num_unique_modes;

/// Filters out the terms of an operator that act on too few unique modes.
///
/// This removes every term from the provided
/// :class:`~qiskit_fermions.operators.FermionOperator` that acts on fewer than ``min_unique_modes``
/// *unique* fermionic modes. The number of unique modes is the number of distinct mode indices a
/// term acts on, irrespective of how many actions it has.
///
/// This is most useful when preparing an electronic-structure Hamiltonian for time evolution: such
/// a Hamiltonian contains terms whose inclusion in a time-evolution circuit has no impact on the
/// sampled bitstrings and, thus, only results in an increased sampling overhead. Filtering these
/// out with ``min_unique_modes=2`` removes
///
/// - the constant energy offset (0 unique modes), whose time evolution only introduces a global
///   phase, and
/// - the number operators :math:`a^\dagger_i a_i` (1 unique mode), whose time evolution amounts to
///   single-qubit Z rotations that do not affect the sampled bitstrings.
///
/// .. note::
///    The operator is modified *in place*. If it tracks group indices (see
///    :attr:`~qiskit_fermions.operators.FermionOperator.groups`), surviving terms retain their
///    relative grouping but the group indices are reassigned to a contiguous range starting from 0.
///    You must therefore *not* rely on the specific group index of any term being preserved across
///    a call to this function.
///
/// .. doctest::
///
///     >>> from qiskit_fermions.operators import FermionOperator
///     >>> from qiskit_fermions.operators.terms.filtering import (
///     ...     filter_terms_by_num_unique_modes,
///     ... )
///     >>> op = FermionOperator.from_dict(
///     ...     {
///     ...         (): 1.0,  # constant offset (0 unique modes)
///     ...         ((True, 0), (False, 0)): 2.0,  # number operator (1 unique mode)
///     ...         ((True, 0), (False, 1)): 3.0,  # hopping term (2 unique modes)
///     ...     }
///     ... )
///     >>> filter_terms_by_num_unique_modes(op, 2)
///     >>> list(op.iter_terms())
///     [([(True, 0), (False, 1)], (3+0j))]
///
/// Args:
///     op: the operator whose terms to filter.
///     min_unique_modes: the minimum number of unique modes a term must act on to be kept.
#[gen_stub_pyfunction(module = "qiskit_fermions.operators.terms.filtering.unique_modes")]
#[pyfunction(name = "filter_terms_by_num_unique_modes")]
#[pyo3(signature = (op, min_unique_modes))]
pub fn py_filter_terms_by_num_unique_modes(op: &Bound<PyFermionOperator>, min_unique_modes: u32) {
    filter_terms_by_num_unique_modes(&mut op.borrow_mut().inner, min_unique_modes);
}

#[pymodule]
pub mod unique_modes {
    #[pymodule_export]
    use super::py_filter_terms_by_num_unique_modes;
}
