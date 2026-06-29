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
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use qiskit_fermions_core::linalg::givens::{GivensRotation, givens_decomposition};

/// Decomposes a unitary matrix into Givens rotations and diagonal phases.
///
/// The :math:`n \times n` unitary matrix, :math:`U`, can be decomposed into a diagonal matrix,
/// :math:`D`, and sequence of :math:`2 \times 2` Givens rotations, :math:`G`, acting on adjacent
/// indices. This algorithm [1]_ requires at most :math:`n (n-1) / 2` such Givens rotations.
///
/// Each Givens rotation is defined by a 4-tuple, ``(c, s, i, j)``, with:
///
/// * ``c``: the real-valued cosine
/// * ``s``: the complex-valued sine
/// * ``i``: the first row index
/// * ``j``: the second row index
///
/// which result in a matrix of the form:
///
/// .. math::
///
///    \begin{pmatrix}
///    c & s \\
///    -s^\dagger & c
///    \end{pmatrix}
///
/// Args:
///     unitary: the unitary matrix, :math:`U`, to be decomposed.
///
/// Returns:
///     A 2-tuple consisting of
///
///     * the sequence of Givens rotations represented as 4-tuples as explained above
///     * the vector of complex phases of the diagonal matrix, :math:`D`
///
/// .. [1] W. R. Clements et al., Optimal design for universal multiport interferometers,
///        Optica 3, 1460-1465 (2016),
///        `doi:10.1364/OPTICA.3.001460 <https://doi.org/10.1364/OPTICA.3.001460>`_.
#[gen_stub_pyfunction(module = "qiskit_fermions.linalg.givens")]
#[pyfunction(name = "givens_decomposition")]
pub fn py_givens_decomposition(
    unitary: PyReadonlyArray2<Complex64>,
) -> (Vec<GivensRotation>, Vec<Complex64>) {
    // Convert numpy array to ndarray - direct conversion
    let ndarray = unitary.as_array().to_owned();
    givens_decomposition(ndarray)
}

#[pymodule]
pub mod givens {
    #[pymodule_export]
    use super::py_givens_decomposition;
}
