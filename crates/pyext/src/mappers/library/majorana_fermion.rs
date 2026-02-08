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
use crate::operators::majorana_operator::PyMajoranaOperator;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use qiskit_fermions_core::mappers::library::majorana_fermion::{
    fermion_to_majorana, majorana_to_fermion,
};

/// Map a :class:`.FermionOperator` to a :class:`.MajoranaOperator`.
///
/// Args:
///     fer_op: the fermionic operator to map.
///
/// Returns:
///     The mapped majorana operator.
///
/// ----
///
/// Definition
/// ==========
///
/// This function implements the simple transformation:
///
/// .. math::
///
///    a^\dagger_j \rightarrow \frac{1}{2} (\gamma_j - i \gamma'_j) ~~\text{and}~~
///    a_j \rightarrow \frac{1}{2} (\gamma_j + i \gamma'_j)
///
/// where :math:`a^\dagger_j` (:math:`a_j`) is the fermionic creation (annihilation) operator
/// acting on the :math:`j`-th spin-less fermionic mode, and :math:`\gamma_j`/:math:`\gamma'_j` are
/// the two Majorana fermion operators. In the case of the :class:`.MajoranaOperator` these will be
/// stored on the even and odd Majorana modes, respectively.
///
/// Usage
/// =====
///
/// .. doctest::
///     >>> from qiskit_fermions.mappers.library import fermion_to_majorana
///     >>> from qiskit_fermions.operators import FermionOperator
///     >>> fer_op = FermionOperator.from_dict({((True, 0), (False, 0)): 1})
///     >>> maj_op = fermion_to_majorana(fer_op)
///     >>> print(maj_op.normal_ordered().simplify())
///      5.000000e-1 +0.000000e0j * ()
///      0.000000e0+5.000000e-1j * (1 0)
///
/// ..
#[gen_stub_pyfunction(module = "qiskit_fermions.mappers.library.majorana")]
#[pyfunction(name = "fermion_to_majorana")]
pub fn py_fermion_to_majorana(fer_op: PyFermionOperator) -> PyMajoranaOperator {
    PyMajoranaOperator {
        inner: fermion_to_majorana(&fer_op.inner),
    }
}

/// Map a :class:`.MajoranaOperator` to a :class:`.FermionOperator`.
///
/// Args:
///     maj_op: the Majorana operator to map.
///
/// Returns:
///     The mapped fermion operator.
///
/// ----
///
/// Definition
/// ==========
///
/// This function implements the simple transformation:
///
/// .. math::
///
///    \gamma_j \rightarrow a^\dagger_j + a_j ~~\text{and}~~
///    \gamma'_j \rightarrow i (a^\dagger_j - a_j)
///
/// where :math:`\gamma_j`/:math:`\gamma'_j` are the two Majorana fermion operators (stored on the
/// even and odd modes, respectively), and :math:`a^\dagger_j` (:math:`a_j`) is the fermionic
/// creation (annihilation) operator acting on the :math:`j`-th spin-less fermionic mode.
///
/// Usage
/// =====
///
/// .. doctest::
///     >>> from qiskit_fermions.mappers.library import majorana_to_fermion
///     >>> from qiskit_fermions.operators import MajoranaOperator
///     >>> maj_op = MajoranaOperator.from_dict({(0, 1): 1})
///     >>> fer_op = majorana_to_fermion(maj_op)
///     >>> print(fer_op.normal_ordered().simplify())
///      0.000000e0 -1.000000e0j * ()
///      0.000000e0 +2.000000e0j * (+_0 -_0)
///
/// ..
#[gen_stub_pyfunction(module = "qiskit_fermions.mappers.library.majorana")]
#[pyfunction(name = "majorana_to_fermion")]
pub fn py_majorana_to_fermion(maj_op: PyMajoranaOperator) -> PyFermionOperator {
    PyFermionOperator {
        inner: majorana_to_fermion(&maj_op.inner),
    }
}

#[pymodule]
pub mod majorana_fermion {
    #[pymodule_export]
    use super::py_fermion_to_majorana;

    #[pymodule_export]
    use super::py_majorana_to_fermion;
}
