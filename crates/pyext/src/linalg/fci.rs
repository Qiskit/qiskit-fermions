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

use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;

use qiskit_fermions_core::linalg::fci::{
    FciMatvecError, MAX_ORBITALS, occupation_axis_mask, slater_determinant_statevector,
};

/// A matvec closure closing over an operator and a fixed FCI sector.
///
/// This is the operator-agnostic contract behind [`FciLinearOperator`]: given a state vector, it
/// produces `op @ vec` in the sector's basis ordering (or an [`FciMatvecError`]). Each operator
/// class builds its own closures -- capturing an owned copy of the operator (and its adjoint) plus
/// the sector -- so the wrapper stays decoupled from any particular operator data structure. The
/// `Send + Sync` bound is required for the wrapper to be a `#[pyclass]`.
type Matvec = Box<dyn Fn(&[Complex64]) -> Result<Vec<Complex64>, FciMatvecError> + Send + Sync>;

/// A minimal, native `scipy.sparse.linalg.LinearOperator`-compatible view of an operator restricted
/// to a fixed FCI sector.
///
/// This class duck-types the subset of the SciPy ``LinearOperator`` interface that
/// :func:`scipy.sparse.linalg.expm_multiply` requires: it exposes :attr:`shape`, :attr:`dtype`,
/// :meth:`matvec`, and :meth:`rmatvec`. (``expm_multiply`` needs the adjoint action -- ``rmatvec``
/// -- because its internal one-norm estimator operates on ``A.H``.) It is **not** part of the public
/// API; it is an internal wrapper returned by the ``_linear_operator_`` protocol methods on the
/// operator classes and consumed by the simulation path.
#[gen_stub_pyclass]
#[gen_stub(module = "qiskit_fermions._lib.linalg.fci")]
#[pyclass(module = "qiskit_fermions.linalg.fci", name = "FciLinearOperator")]
pub struct FciLinearOperator {
    dim: usize,
    trace: Complex64,
    matvec: Matvec,
    rmatvec: Matvec,
}

impl FciLinearOperator {
    /// Constructs a wrapper of dimension `dim` backed by the given `matvec`/`rmatvec` closures.
    ///
    /// `dim` is the FCI sector dimension (the length of the state vectors this operator acts on) and
    /// is exposed to Python as the square :attr:`shape` `(dim, dim)`. `matvec` applies the operator;
    /// `rmatvec` applies its adjoint (`A.H @ v`), which SciPy's `expm_multiply` requires.
    pub fn new(dim: usize, trace: Complex64, matvec: Matvec, rmatvec: Matvec) -> Self {
        Self {
            dim,
            trace,
            matvec,
            rmatvec,
        }
    }

    /// Applies `kernel` to `vec`.
    ///
    /// `vec` must be a contiguous one-dimensional `complex128` array of length `dim`. The public
    /// `_linear_operator_` wrapper (in :mod:`qiskit_fermions.operators`) is responsible for coercing
    /// SciPy's probe vectors (real dtype, or non-contiguous `(dim, 1)` columns) into this layout
    /// before they reach here, so the coercion cost is paid with numpy handles bound once per
    /// operator rather than re-resolved on every matvec.
    fn apply<'py>(
        py: Python<'py>,
        kernel: &Matvec,
        vec: PyReadonlyArray1<'py, Complex64>,
    ) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
        let out = kernel(vec.as_slice()?).map_err(crate::value_err)?;
        Ok(out.into_pyarray(py))
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl FciLinearOperator {
    /// The shape of the operator as a square ``(dim, dim)`` tuple.
    #[getter]
    fn shape(&self) -> (usize, usize) {
        (self.dim, self.dim)
    }

    /// The dtype of the operator, always ``numpy.dtype("complex128")``.
    #[getter]
    fn dtype<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let numpy = py.import("numpy")?;
        numpy.getattr("dtype")?.call1(("complex128",))
    }

    /// The exact trace of the operator on this FCI sector.
    #[getter]
    fn trace(&self) -> Complex64 {
        self.trace
    }

    /// Applies the operator to a state vector: returns ``op @ vec``.
    ///
    /// Args:
    ///     vec: the input state vector, a contiguous one-dimensional ``complex128`` array of length
    ///         ``dim``.
    ///
    /// Returns:
    ///     The transformed state vector of length ``dim``.
    ///
    /// Raises:
    ///     ValueError: if ``vec`` has the wrong length, or the underlying operator acts on a mode
    ///         index outside the sector.
    fn matvec<'py>(
        &self,
        py: Python<'py>,
        vec: PyReadonlyArray1<'py, Complex64>,
    ) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
        Self::apply(py, &self.matvec, vec)
    }

    /// Applies the operator's adjoint to a state vector: returns ``op.H @ vec``.
    ///
    /// SciPy's ``expm_multiply`` requires this because its one-norm estimator operates on ``A.H``.
    ///
    /// Args:
    ///     vec: the input state vector, a contiguous one-dimensional ``complex128`` array of length
    ///         ``dim``.
    ///
    /// Returns:
    ///     The adjoint-transformed state vector of length ``dim``.
    ///
    /// Raises:
    ///     ValueError: if ``vec`` has the wrong length, or the underlying operator acts on a mode
    ///         index outside the sector.
    fn rmatvec<'py>(
        &self,
        py: Python<'py>,
        vec: PyReadonlyArray1<'py, Complex64>,
    ) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
        Self::apply(py, &self.rmatvec, vec)
    }
}

/// Builds the FCI state vector of a single occupation determinant.
///
/// This is the state-vector seed a Jordan-Wigner occupation prepares from the vacuum, produced
/// directly in the FCI basis ordering (the same ordering the matvec kernels and ``ffsim`` use). The
/// occupied orbitals of each spin sector are passed as bitmasks (bit ``p`` set iff orbital ``p`` is
/// occupied).
///
/// Args:
///     norb: the number of spatial orbitals.
///     alpha_str: the occupation bitmask of the alpha sector (or of the single spinless sector).
///     beta_str: the occupation bitmask of the beta sector, or ``None`` for a spinless system. When
///         ``None`` the returned vector has length ``C(norb, popcount(alpha_str))``; otherwise it has
///         length ``C(norb, popcount(alpha_str)) * C(norb, popcount(beta_str))`` with the block-spin
///         flat index ``addr_a * dim_b + addr_b``.
///
/// Returns:
///     The one-hot Slater-determinant state vector as a ``complex128`` array.
///
/// Raises:
///     ValueError: if ``norb`` exceeds the maximum number of orbitals the bitmask representation
///         supports (64), an occupation bit is set outside ``0..norb``, or the spinful FCI
///         dimension overflows the addressable range.
#[gen_stub_pyfunction(module = "qiskit_fermions._lib.linalg.fci")]
#[pyfunction(name = "slater_determinant_statevector", signature = (norb, alpha_str, beta_str=None))]
pub fn py_slater_determinant_statevector(
    py: Python<'_>,
    norb: u32,
    alpha_str: u64,
    beta_str: Option<u64>,
) -> PyResult<Bound<'_, PyArray1<Complex64>>> {
    if norb > MAX_ORBITALS {
        return Err(crate::value_err(format!(
            "norb={norb} exceeds the maximum of {MAX_ORBITALS} orbitals"
        )));
    }
    let vec =
        slater_determinant_statevector(norb, alpha_str, beta_str).map_err(crate::value_err)?;
    Ok(vec.into_pyarray(py))
}

/// Builds a boolean mask over an FCI sector's addresses selecting a partial-occupation subspace.
///
/// The mask has length ``C(norb, nocc)``; entry ``addr`` is ``True`` iff the determinant at that
/// address has **all** orbitals in the ``occupied`` bitmask set **and all** orbitals in the ``empty``
/// bitmask clear. Orbitals in neither mask are unconstrained, so a partial constraint selects a whole
/// family of determinants (fixing only some orbitals accepts every determinant that agrees there,
/// whatever the free orbitals do). This is the per-axis subspace test behind
/// :class:`.InitializeModes`'s validator: an incoming state passes iff its amplitude is confined to
/// the ``True`` entries of this mask along the constrained spin axis.
///
/// Args:
///     norb: the number of spatial orbitals.
///     nocc: the number of occupied orbitals of the sector (its address space is ``C(norb, nocc)``).
///     occupied: a bitmask over ``0..norb`` whose set bits mark orbitals forced to be occupied.
///     empty: a bitmask over ``0..norb`` whose set bits mark orbitals forced to be empty.
///
/// Returns:
///     A one-dimensional ``bool`` array of length ``C(norb, nocc)``.
///
/// Raises:
///     ValueError: if ``norb`` exceeds the maximum number of orbitals the bitmask representation
///         supports (64); if a constraint bit is set outside ``0..norb``; if an orbital is
///         constrained to be both occupied and empty; or if the constraint is unsatisfiable (more
///         than ``nocc`` orbitals forced occupied, or too few free orbitals left to reach ``nocc``).
#[gen_stub_pyfunction(module = "qiskit_fermions._lib.linalg.fci")]
#[pyfunction(name = "occupation_axis_mask")]
// `pyo3-stub-gen` has no NumPyScalar impl for `bool`, so the array return type cannot be derived;
// spell it out to match the `numpy.typing.NDArray[...]` convention the sibling functions render.
#[gen_stub(override_return_type(
    type_repr = "numpy.typing.NDArray[numpy.bool_]",
    imports = ("numpy", "numpy.typing")
))]
pub fn py_occupation_axis_mask(
    py: Python<'_>,
    norb: u32,
    nocc: u32,
    occupied: u64,
    empty: u64,
) -> PyResult<Bound<'_, PyArray1<bool>>> {
    if norb > MAX_ORBITALS {
        return Err(crate::value_err(format!(
            "norb={norb} exceeds the maximum of {MAX_ORBITALS} orbitals"
        )));
    }
    let mask = occupation_axis_mask(norb, nocc, occupied, empty).map_err(crate::value_err)?;
    Ok(mask.into_pyarray(py))
}

#[pymodule]
pub mod fci {
    #[pymodule_export]
    use super::FciLinearOperator;
    #[pymodule_export]
    use super::py_occupation_axis_mask;
    #[pymodule_export]
    use super::py_slater_determinant_statevector;
}
