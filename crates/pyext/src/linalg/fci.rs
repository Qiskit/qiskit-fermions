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
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;

use qiskit_fermions_core::linalg::fci::FciMatvecError;

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
#[pyclass(module = "qiskit_fermions.linalg.fci", name = "FciLinearOperator")]
pub struct FciLinearOperator {
    dim: usize,
    matvec: Matvec,
    rmatvec: Matvec,
}

impl FciLinearOperator {
    /// Constructs a wrapper of dimension `dim` backed by the given `matvec`/`rmatvec` closures.
    ///
    /// `dim` is the FCI sector dimension (the length of the state vectors this operator acts on) and
    /// is exposed to Python as the square :attr:`shape` `(dim, dim)`. `matvec` applies the operator;
    /// `rmatvec` applies its adjoint (`A.H @ v`), which SciPy's `expm_multiply` requires.
    pub fn new(dim: usize, matvec: Matvec, rmatvec: Matvec) -> Self {
        Self {
            dim,
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
        let out = kernel(vec.as_slice()?).map_err(|e| PyValueError::new_err(e.to_string()))?;
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

#[pymodule]
pub mod fci {
    #[pymodule_export]
    use super::FciLinearOperator;
}
