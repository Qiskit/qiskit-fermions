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
use pyo3::types::PyType;
use pyo3_stub_gen::derive::*;
use qiskit_fermions_core::operators::fermion_operator::FermionOperator;
use qiskit_fermions_core::operators::library::fcidump::FCIDump;

/// An electronic structure Hamiltonian in FCIDump format.
///
/// ----
///
/// Definition
/// ==========
///
/// The FCIDump format was originally defined by Knowles and Handy, 1989 [1]_.
/// It is a widespread format for exporting electronic structure Hamiltonians in a plain-text file.
///
/// The present data structure only stores the information relevant for constructing the second
/// quantized operator.
/// However, this implementation goes beyond the original definition by supporting unrestricted
/// spin data to be loaded. The table below outlines how integrals are associated with spin species
/// based on the intervals in which the indices fall (assuming a header with ``NORB=n``):
///
/// ================== ================ ================ ================ ================
/// Integral Type      i                j                k                l
/// ================== ================ ================ ================ ================
/// Constant           :math:`{0}`      :math:`{0}`      :math:`{0}`      :math:`{0}`
/// 1-body alpha       :math:`{0}`      :math:`{0}`      :math:`[1,n]`    :math:`[1,n]`
/// 1-body beta        :math:`{0}`      :math:`{0}`      :math:`[n+1,2n]` :math:`[n+1,2n]`
/// 2-body alpha-alpha :math:`[1,n]`    :math:`[1,n]`    :math:`[1,n]`    :math:`[1,n]`
/// 2-body alpha-beta  :math:`[1,n]`    :math:`[1,n]`    :math:`[n+1,2n]` :math:`[n+1,2n]`
/// 2-body beta-beta   :math:`[n+1,2n]` :math:`[n+1,2n]` :math:`[n+1,2n]` :math:`[n+1,2n]`
/// ================== ================ ================ ================ ================
///
/// The only required values are the 1-body alpha-spin integrals.
///
/// .. note::
///    The implementation of this data structure is opaque to Python and only provides a few
///    attributes and methods documented at the end of this page.
///
/// ----
///
/// Conversion
/// ==========
///
/// Operator implementations which can be constructed from an instance of :class:`.FCIDump` provide
/// a ``from_fcidump`` classmethod:
///
/// .. table::
///
///   ===================================== =================================================================
///   :meth:`.FermionOperator.from_fcidump` Constructs a :class:`.FermionOperator` from an :class:`.FCIDump`.
///   ===================================== =================================================================
///
/// ----
///
/// .. [1] P. J. Knowles and N. C. Handy, Computer Physics Communications 54 (1989) 75-83.
#[gen_stub_pyclass]
#[pyclass(module = "qiskit_fermions.operators.library.fcidump", name = "FCIDump")]
#[derive(Clone)]
pub struct PyFCIDump {
    pub inner: FCIDump,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyFCIDump {
    /// Parses an FCIDump file.
    ///
    /// Assuming you have an FCIDump file called ``molecule.fcidump``, you use this method like so:
    ///
    /// .. code-block:: python
    ///
    ///    from qiskit_fermions.operators.library import FCIDump
    ///
    ///    fcidump = FCIDump.from_file("molecule.fcidump")
    ///
    /// Args:
    ///     file_path: the path to the FCIDump file.
    ///
    /// Returns:
    ///     The constructed data structure.
    #[classmethod]
    fn from_file(_cls: &Bound<'_, PyType>, file_path: String) -> Self {
        Self {
            inner: FCIDump::from_file(file_path),
        }
    }

    /// Returns the number of orbitals.
    ///
    /// This number, :math:`n`, is extracted from the ``NORB=n`` field in the header of the FCIDump
    /// file.
    #[getter]
    fn norb(&self) -> u32 {
        self.inner.norb
    }

    /// Returns the number of electrons.
    ///
    /// This number, :math:`n`, is extracted from the ``NELEC=n`` field in the header of the
    /// FCIDump file.
    #[getter]
    fn nelec(&self) -> u32 {
        self.inner.nelec
    }

    /// Returns the spin quantum number, :math:`2S`.
    ///
    /// This number, :math:`S`, is extracted from the ``MS2=S`` field in the header of the FCIDump
    /// file.
    #[getter]
    fn ms2(&self) -> u32 {
        self.inner.ms2
    }
}

#[pymodule]
pub mod fcidump {
    #[pymodule_export]
    use super::PyFCIDump;
}

#[gen_stub_pymethods]
#[pymethods]
impl PyFermionOperator {
    /// Constructs a :class:`.FermionOperator` from an :class:`.FCIDump` data structure.
    ///
    /// Assuming you have an FCIDump file called ``molecule.fcidump``, you can construct the
    /// second-quantized operator like so:
    ///
    /// .. code-block:: python
    ///
    ///    from qiskit_fermions.operators import FermionOperator
    ///    from qiskit_fermions.operators.library import FCIDump
    ///
    ///    fcidump = FCIDump.from_file("molecule.fcidump")
    ///    operator = FermionOperator.from_fcidump(fcidump)
    ///
    /// Args:
    ///     fcidump: the FCIDump data structure.
    ///
    /// Returns:
    ///     The constructed operator.
    #[classmethod]
    fn from_fcidump(_cls: &Bound<'_, PyType>, fcidump: PyFCIDump) -> Self {
        Self {
            inner: FermionOperator::from(&fcidump.inner),
        }
    }
}
