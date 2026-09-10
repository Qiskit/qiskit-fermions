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
use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;
use pyo3::types::PyType;
use pyo3_stub_gen::derive::*;
use qiskit_fermions_core::operators::fermion_operator::FermionOperator;
use qiskit_fermions_core::operators::library::fcidump::{FCIDump, FCIDumpError};

/// An electronic structure Hamiltonian in FCIDump format.
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
/// The two-body integrals are expected in `chemist` ordering, :math:`(ij|kl)`, matching the FCIDump
/// convention.
///
/// .. _FCIDump-implementation:
///
/// Implementation
/// ==============
///
/// The integrals are stored as flattened arrays exploiting their permutational symmetry, which is
/// also the layout the :class:`.FermionOperator` electronic-integral constructors expect. Writing
/// :math:`n` for the number of orbitals (:attr:`norb`) and
/// :math:`\text{npair} = n (n + 1) / 2`, a single data structure contains up to 5 arrays:
///
/// .. table::
///
///    =============== =========================================== ==========================================
///    ``one_body_a``  :math:`\text{npair}`                        The :math:`\alpha`-spin 1-body integrals.
///    ``one_body_b``  :math:`\text{npair}`                        The :math:`\beta`-spin 1-body integrals.
///    ``two_body_aa`` :math:`\text{npair} (\text{npair} + 1) / 2` The :math:`\alpha\alpha` 2-body integrals.
///    ``two_body_ab`` :math:`\text{npair}^2`                      The :math:`\alpha\beta` 2-body integrals.
///    ``two_body_bb`` :math:`\text{npair} (\text{npair} + 1) / 2` The :math:`\beta\beta` 2-body integrals.
///    =============== =========================================== ==========================================
///
/// The 1-body arrays are the flattened lower triangle of an :math:`(n, n)` matrix, indexed as
/// ``ia = i * (i + 1) // 2 + a`` with ``a <= i``. The ``two_body_aa`` and ``two_body_bb`` arrays are
/// 8-fold (S8) symmetric: the flattened lower triangle of a
/// :math:`(\text{npair}, \text{npair})` matrix, whose own two axes are each such a pair index.
///
/// .. warning::
///    ``two_body_ab`` is packed *differently* from its siblings. It is only 4-fold (S4) symmetric, so
///    it holds the **full** :math:`(\text{npair}, \text{npair})` matrix in row-major order, indexed as
///    ``iajb = ia * npair + jb``. Its row pair indexes the :math:`\alpha`-spin species and its column
///    pair the :math:`\beta`-spin species, so it is *not* symmetric under exchanging the two pairs.
///
/// .. note::
///    You can access **read-only copies** of these internal arrays via their respective methods:
///    :meth:`.get_one_body_tril_a`, :meth:`.get_one_body_tril_b`, :meth:`.get_two_body_tril_aa`,
///    :meth:`.get_two_body_tril_ab`, and :meth:`.get_two_body_tril_bb`.
///
/// The 1-body :math:`\beta`-spin and the :math:`\alpha\beta` / :math:`\beta\beta` 2-body arrays are
/// only present for a file carrying unrestricted spin data. They are absent together, so
/// :attr:`is_unrestricted` reports on all three at once.
///
/// The stored values are the integrals as they appear in the file. In particular, they do **not**
/// carry the conventional factor of :math:`\frac{1}{2}` that the 2-body operator terms do, so a
/// coefficient of an operator built via :meth:`.FermionOperator.from_fcidump` is half the
/// corresponding value returned here.
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
/// .. [1] P. J. Knowles and N. C. Handy, Computer Physics Communications 54 (1989) 75-83.
#[gen_stub_pyclass]
#[gen_stub(module = "qiskit_fermions._lib.operators.operators_library.fcidump")]
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
    ///
    /// Raises:
    ///     OSError: if ``file_path`` cannot be opened or read.
    ///     ValueError: if the file does not honour the FCIDump format (a missing header namelist, a
    ///         missing ``NORB`` or ``NELEC`` field, or a malformed ``MS2`` field), or if it carries
    ///         an MO energy value, which is not supported yet.
    #[classmethod]
    fn from_file(_cls: &Bound<'_, PyType>, file_path: String) -> PyResult<Self> {
        // A bad path is by far the likeliest failure here, so it maps onto `OSError` (carrying the
        // underlying `io::Error`) while the format violations map onto `ValueError`. Returning a
        // `PyResult` at all is the point: the core reports these as an `FCIDumpError`, and without
        // this conversion a panic would cross the FFI boundary as `PanicException`, which derives
        // from `BaseException` and so slips through an ordinary `except Exception`.
        FCIDump::from_file(file_path)
            .map(|inner| Self { inner })
            .map_err(|err| match err {
                // `Display` for these two variants already interpolates the underlying
                // `io::Error`, so the message needs no further decoration.
                FCIDumpError::Open { .. } | FCIDumpError::Read { .. } => {
                    ::pyo3::exceptions::PyOSError::new_err(err.to_string())
                }
                _ => crate::value_err(err),
            })
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
    /// This number is extracted from the ``NELEC`` field in the header of the FCIDump file.
    #[getter]
    fn nelec(&self) -> u32 {
        self.inner.nelec
    }

    /// Returns twice the spin quantum number, :math:`2S`.
    ///
    /// This number, :math:`2S`, is extracted from the ``MS2=2S`` field in the header of the FCIDump
    /// file.
    #[getter]
    fn ms2(&self) -> u32 {
        self.inner.ms2
    }

    /// Returns the constant energy offset, if the file provides one.
    ///
    /// This is the value of the integral line whose four indices are all zero, which for an
    /// electronic structure Hamiltonian is the nuclear-repulsion energy. It is ``None`` when the file
    /// carries no such line.
    ///
    /// .. seealso::
    ///    :meth:`.FermionOperator.from_fcidump`, which includes this value as the identity term.
    ///
    /// .. doctest::
    ///
    ///     >>> from qiskit_fermions.operators.library import FCIDump
    ///     >>> fcidump = FCIDump.from_file("tests/h2.fcidump")
    ///     >>> fcidump.constant
    ///     0.7199689944489797
    ///
    /// Returns:
    ///     The constant energy offset, or ``None`` when the file provides none.
    #[getter]
    fn constant(&self) -> Option<f64> {
        self.inner.constant
    }

    /// Whether this data structure carries unrestricted (spin-dependent) integrals.
    ///
    /// The :math:`\beta`-spin 1-body and the :math:`\alpha\beta` / :math:`\beta\beta` 2-body arrays
    /// are present exactly when this is ``True``; they are absent together.
    ///
    /// .. doctest::
    ///
    ///     >>> from qiskit_fermions.operators.library import FCIDump
    ///     >>> FCIDump.from_file("tests/h2.fcidump").is_unrestricted
    ///     False
    ///     >>> FCIDump.from_file("tests/heh.fcidump").is_unrestricted
    ///     True
    ///
    /// Returns:
    ///     Whether the beta-spin integral arrays are present.
    #[getter]
    fn is_unrestricted(&self) -> bool {
        self.inner.one_body_b.is_some()
    }

    /// Returns a read-only copy of the :math:`\alpha`-spin 1-body integrals.
    ///
    /// .. note::
    ///    This method returns a **copy** of the internal data.
    ///
    /// .. seealso::
    ///    The explanation of the internal data structure, :ref:`here <FCIDump-implementation>`, and
    ///    :meth:`.FermionOperator.from_1body_tril_spin_sym`, which consumes this array.
    ///
    /// .. doctest::
    ///
    ///     >>> from qiskit_fermions.operators.library import FCIDump
    ///     >>> fcidump = FCIDump.from_file("tests/h2.fcidump")
    ///     >>> fcidump.get_one_body_tril_a().shape
    ///     (3,)
    ///
    /// Returns:
    ///     The flattened lower-triangular 1-body integrals, of length
    ///     :math:`\text{npair} = n (n + 1) / 2`.
    fn get_one_body_tril_a<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.one_body_a.clone().into_pyarray(py)
    }

    /// Returns a read-only copy of the :math:`\beta`-spin 1-body integrals.
    ///
    /// .. note::
    ///    This method returns a **copy** of the internal data.
    ///
    /// .. seealso::
    ///    The explanation of the internal data structure, :ref:`here <FCIDump-implementation>`, and
    ///    :meth:`.FermionOperator.from_1body_tril_spin`, which consumes this array.
    ///
    /// Returns:
    ///     The flattened lower-triangular 1-body integrals, of length
    ///     :math:`\text{npair} = n (n + 1) / 2`, or ``None`` when
    ///     :attr:`is_unrestricted` is ``False``.
    fn get_one_body_tril_b<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .one_body_b
            .as_ref()
            .map(|arr| arr.clone().into_pyarray(py))
    }

    /// Returns a read-only copy of the :math:`\alpha\alpha`-spin 2-body integrals.
    ///
    /// The values are in `chemist` ordering, :math:`(ij|kl)`, and carry no factor of
    /// :math:`\frac{1}{2}`.
    ///
    /// .. note::
    ///    This method returns a **copy** of the internal data.
    ///
    /// .. seealso::
    ///    The explanation of the internal data structure, :ref:`here <FCIDump-implementation>`, and
    ///    :meth:`.FermionOperator.from_2body_tril_spin_sym`, which consumes this array.
    ///
    /// .. doctest::
    ///
    ///     >>> from qiskit_fermions.operators.library import FCIDump
    ///     >>> fcidump = FCIDump.from_file("tests/h2.fcidump")
    ///     >>> fcidump.get_two_body_tril_aa().shape
    ///     (6,)
    ///
    /// Returns:
    ///     The 8-fold symmetric (S8) flattened 2-body integrals, of length
    ///     :math:`\text{npair} (\text{npair} + 1) / 2`.
    fn get_two_body_tril_aa<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.two_body_aa.clone().into_pyarray(py)
    }

    /// Returns a read-only copy of the :math:`\alpha\beta`-spin 2-body integrals.
    ///
    /// The values are in `chemist` ordering, :math:`(ij|kl)`, and carry no factor of
    /// :math:`\frac{1}{2}`.
    ///
    /// .. warning::
    ///    Unlike the other 2-body arrays, this one is only 4-fold (S4) symmetric: it holds the
    ///    **full** :math:`(\text{npair}, \text{npair})` matrix, whose row pair indexes the
    ///    :math:`\alpha`-spin and whose column pair indexes the :math:`\beta`-spin species. It is
    ///    therefore *not* symmetric under exchanging the two pairs.
    ///
    /// .. note::
    ///    This method returns a **copy** of the internal data.
    ///
    /// .. seealso::
    ///    The explanation of the internal data structure, :ref:`here <FCIDump-implementation>`, and
    ///    :meth:`.FermionOperator.from_2body_tril_spin`, which consumes this array.
    ///
    /// Returns:
    ///     The 4-fold symmetric (S4) flattened 2-body integrals, of length
    ///     :math:`\text{npair}^2`, or ``None`` when :attr:`is_unrestricted` is ``False``.
    fn get_two_body_tril_ab<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .two_body_ab
            .as_ref()
            .map(|arr| arr.clone().into_pyarray(py))
    }

    /// Returns a read-only copy of the :math:`\beta\beta`-spin 2-body integrals.
    ///
    /// The values are in `chemist` ordering, :math:`(ij|kl)`, and carry no factor of
    /// :math:`\frac{1}{2}`.
    ///
    /// .. note::
    ///    This method returns a **copy** of the internal data.
    ///
    /// .. seealso::
    ///    The explanation of the internal data structure, :ref:`here <FCIDump-implementation>`, and
    ///    :meth:`.FermionOperator.from_2body_tril_spin`, which consumes this array.
    ///
    /// Returns:
    ///     The 8-fold symmetric (S8) flattened 2-body integrals, of length
    ///     :math:`\text{npair} (\text{npair} + 1) / 2`, or ``None`` when
    ///     :attr:`is_unrestricted` is ``False``.
    fn get_two_body_tril_bb<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .two_body_bb
            .as_ref()
            .map(|arr| arr.clone().into_pyarray(py))
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
    ///     The constructed operator. When the FCIDump provides a constant (e.g. nuclear-repulsion)
    ///     energy, it is included as the identity term ``()``.
    #[classmethod]
    fn from_fcidump(_cls: &Bound<'_, PyType>, fcidump: PyFCIDump) -> Self {
        FermionOperator::from(&fcidump.inner).into()
    }
}
