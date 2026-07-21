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
use qiskit_fermions_core::linalg::givens::{
    GivensRotation, givens_decomposition, givens_decomposition_slater,
};

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
/// The original unitary is recovered by processing the returned rotations in `reverse` order and
/// right-multiplying the diagonal matrix by the `element-wise complex conjugate` of each rotation
/// matrix :math:`G_k` (as defined above). That is, for :math:`N` returned rotations,
///
/// .. math::
///
///    U = D \cdot \overline{G_N} \cdot \overline{G_{N-1}} \cdots \overline{G_1},
///
/// where :math:`\overline{G_k}` denotes element-wise conjugation (not the conjugate transpose) and
/// each :math:`G_k` acts only on rows/columns :math:`i` and :math:`j` of its rotation.
///
/// .. doctest::
///
///     >>> import numpy as np
///     >>> from qiskit_fermions.linalg import givens_decomposition
///     >>> unitary = np.array([[0.6, 0.8j], [0.8, -0.6j]], dtype=complex)
///     >>> rotations, phases = givens_decomposition(unitary)
///     >>> reconstructed = np.diag(phases).astype(complex)
///     >>> for c, s, i, j in rotations[::-1]:
///     ...     givens_mat = np.eye(2, dtype=complex)
///     ...     givens_mat[np.ix_((i, j), (i, j))] = [[c, s], [-s.conjugate(), c]]
///     ...     reconstructed = reconstructed @ givens_mat.conj()
///     >>> bool(np.allclose(reconstructed, unitary))
///     True
///
/// .. [1] W. R. Clements et al., Optimal design for universal multiport interferometers,
///        Optica 3, 1460-1465 (2016),
///        `doi:10.1364/OPTICA.3.001460 <https://doi.org/10.1364/OPTICA.3.001460>`_.
#[gen_stub_pyfunction(module = "qiskit_fermions._lib.linalg.givens")]
#[pyfunction(name = "givens_decomposition")]
pub fn py_givens_decomposition(
    unitary: PyReadonlyArray2<Complex64>,
) -> (Vec<GivensRotation>, Vec<Complex64>) {
    // Convert numpy array to ndarray - direct conversion
    let ndarray = unitary.as_array().to_owned();
    givens_decomposition(ndarray)
}

/// Decomposes the occupied orbitals of a Slater determinant into Givens rotations.
///
/// This is the rectangular counterpart of :func:`givens_decomposition`, specialized for Slater
/// determinant `state preparation`. Given the coefficient matrix of the occupied orbitals of a
/// Slater determinant, it returns a sequence of Givens rotations that, applied to the reference
/// configuration :math:`\lvert 1 \cdots 1 0 \cdots 0 \rangle` (the first :math:`m` orbitals
/// occupied), prepares the Slater determinant. Here ``orbital_coeffs`` is an :math:`m \times n`
/// matrix whose rows are the :math:`m` occupied orbitals expressed in a basis of :math:`n` spatial
/// orbitals (:math:`m \le n`); its rows are assumed to be orthonormal.
///
/// Unlike :func:`givens_decomposition`, this decomposition only needs to realize the :math:`m`
/// occupied orbitals rather than a full :math:`n \times n` orbital rotation, so it uses at most
/// :math:`m (n - m)` Givens rotations arranged in a diamond-shaped pattern (versus the
/// :math:`n (n - 1) / 2` brick-wall of the square decomposition). The decomposition contains `no`
/// diagonal phases, because a global phase and any rotation within the occupied space leave the
/// prepared Slater determinant unchanged.
///
/// Each Givens rotation is defined by a 4-tuple, ``(c, s, i, j)``, with:
///
/// * ``c``: the real-valued cosine
/// * ``s``: the complex-valued sine
/// * ``i``: the first index
/// * ``j``: the second (adjacent) index
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
///     orbital_coeffs: the :math:`m \times n` matrix of occupied-orbital coefficients.
///
/// Returns:
///     The sequence of Givens rotations represented as 4-tuples as explained above.
///
/// The occupied orbitals are recovered by applying the returned rotations, `in order`, to the
/// columns of the :math:`m \times n` reference :math:`\begin{pmatrix} I_m & 0 \end{pmatrix}`, where
/// each rotation acting on indices :math:`i` and :math:`j` sends
///
/// .. math::
///
///    v_i \mapsto c \, v_i + s^\dagger v_j, \qquad v_j \mapsto c \, v_j - s \, v_i.
///
/// The result spans the same occupied space as ``orbital_coeffs`` (they define the same Slater
/// determinant), so the squared overlap :math:`\lvert \det(A B^\dagger) \rvert^2` between the
/// reconstructed orbitals :math:`A` and the target :math:`B` is one.
///
/// .. doctest::
///
///     >>> import numpy as np
///     >>> from qiskit_fermions.linalg import givens_decomposition_slater
///     >>> # two occupied orbitals in a basis of three, with orthonormal rows
///     >>> base = np.array([[0.8, 0.6, 0.0], [-0.48, 0.64, 0.6]])
///     >>> orbital_coeffs = (base * np.array([[1.0], [1j]])).astype(complex)
///     >>> rotations = givens_decomposition_slater(orbital_coeffs)
///     >>> m, n = orbital_coeffs.shape
///     >>> reconstructed = np.eye(m, n, dtype=complex)
///     >>> for c, s, i, j in rotations:
///     ...     col_i, col_j = reconstructed[:, i].copy(), reconstructed[:, j].copy()
///     ...     reconstructed[:, i] = c * col_i + s.conjugate() * col_j
///     ...     reconstructed[:, j] = c * col_j - s * col_i
///     >>> overlap = abs(np.linalg.det(reconstructed @ orbital_coeffs.conj().T)) ** 2
///     >>> bool(np.isclose(overlap, 1.0))
///     True
///
/// .. seealso::
///    :func:`givens_decomposition` for the square (full orbital rotation) decomposition.
#[gen_stub_pyfunction(module = "qiskit_fermions._lib.linalg.givens")]
#[pyfunction(name = "givens_decomposition_slater")]
pub fn py_givens_decomposition_slater(
    orbital_coeffs: PyReadonlyArray2<Complex64>,
) -> Vec<GivensRotation> {
    let ndarray = orbital_coeffs.as_array().to_owned();
    givens_decomposition_slater(ndarray)
}

#[pymodule]
pub mod givens {
    #[pymodule_export]
    use super::py_givens_decomposition;
    #[pymodule_export]
    use super::py_givens_decomposition_slater;
}
