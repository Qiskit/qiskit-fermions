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
use numpy::{IntoPyArray, PyArray2, PyArray4, PyReadonlyArray2, PyReadonlyArray4};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use qiskit_fermions_core::linalg::double_factorized::{
    DoubleFactorizedT2AlphaBetaTerm, DoubleFactorizedTerm, double_factorized,
    double_factorized_t2, double_factorized_t2_alpha_beta, modified_cholesky, reconstruct_t2,
    reconstruct_t2_alpha_beta,
};

/// A double-factorized term as exposed to Python: a 2-tuple of the diagonal Coulomb matrix and the
/// orbital rotation, both as numpy arrays.
type PyDoubleFactorizedTerm<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<Complex64>>);

/// An alpha-beta double-factorized term as exposed to Python: a 2-tuple of the three diagonal
/// Coulomb matrices ``(aa, ab, bb)`` and the two orbital rotations ``(alpha, beta)``.
type PyDoubleFactorizedT2AlphaBetaTerm<'py> = (
    [Bound<'py, PyArray2<f64>>; 3],
    [Bound<'py, PyArray2<Complex64>>; 2],
);

/// Converts a single core [`DoubleFactorizedTerm`] into its Python representation.
fn term_into_py(py: Python<'_>, term: DoubleFactorizedTerm) -> PyDoubleFactorizedTerm<'_> {
    let (diag_coulomb, orbital_rotation) = term;
    (
        diag_coulomb.into_pyarray(py),
        orbital_rotation.into_pyarray(py),
    )
}

/// Converts a single core [`DoubleFactorizedT2AlphaBetaTerm`] into its Python representation.
fn alpha_beta_term_into_py(
    py: Python<'_>,
    term: DoubleFactorizedT2AlphaBetaTerm,
) -> PyDoubleFactorizedT2AlphaBetaTerm<'_> {
    let DoubleFactorizedT2AlphaBetaTerm {
        diag_coulomb: [aa, ab, bb],
        orbital_rotations: [alpha, beta],
    } = term;
    (
        [aa.into_pyarray(py), ab.into_pyarray(py), bb.into_pyarray(py)],
        [alpha.into_pyarray(py), beta.into_pyarray(py)],
    )
}

/// Modified Cholesky decomposition of a Hermitian positive-semidefinite matrix.
///
/// Decomposes ``mat`` into a sum of outer products :math:`M = \sum_i v_i v_i^\dagger`, returning the
/// vectors :math:`v_i` as the columns of the result. The number of terms is governed by ``tol``,
/// while ``max_vecs`` (default: the matrix dimension) is always respected even if it forces the
/// truncation error above ``tol``.
///
/// Arguments:
///     mat: the Hermitian positive-semidefinite matrix, :math:`M`, to decompose.
///     tol: the tolerance controlling the truncation error of the decomposition.
///     max_vecs: the maximum number of Cholesky vectors to retain. Defaults to the dimension of
///         ``mat``.
///
/// Returns:
///     A 2-dimensional array whose columns are the Cholesky vectors :math:`v_i`.
#[gen_stub_pyfunction(module = "qiskit_fermions.linalg.double_factorized")]
#[pyfunction(name = "modified_cholesky", signature = (mat, tol, max_vecs=None))]
pub fn py_modified_cholesky<'py>(
    py: Python<'py>,
    mat: PyReadonlyArray2<Complex64>,
    tol: f64,
    max_vecs: Option<usize>,
) -> Bound<'py, PyArray2<Complex64>> {
    modified_cholesky(&mat.as_array().to_owned(), tol, max_vecs).into_pyarray(py)
}

/// Double-factorized decomposition of a real two-body tensor.
///
/// Represents
/// :math:`h_{pqrs} = \sum_t \sum_{kl} Z^{(t)}_{kl} U^{(t)}_{pk} U^{(t)}_{qk} U^{(t)}_{rl} U^{(t)}_{sl}`,
/// returning a list of :math:`(Z, U)` terms where each :math:`Z` is a real symmetric diagonal
/// Coulomb matrix and each :math:`U` is a unitary orbital rotation.
///
/// When ``cholesky`` is ``True`` (the default behavior in the original library) the outer
/// factorization uses a modified Cholesky decomposition; otherwise it uses a truncated
/// eigendecomposition.
///
/// Arguments:
///     two_body_tensor: the real two-body tensor of shape ``(norb, norb, norb, norb)``.
///     tol: the tolerance controlling the truncation error of the decomposition.
///     max_vecs: the maximum number of terms to retain. Defaults to ``norb * (norb + 1) / 2``.
///     cholesky: whether to use the modified Cholesky decomposition (``True``) or a truncated
///         eigendecomposition (``False``) for the outer factorization.
///
/// Returns:
///     A list of :math:`(Z, U)` 2-tuples, where ``Z`` is the diagonal Coulomb matrix and ``U`` is
///     the orbital rotation.
#[gen_stub_pyfunction(module = "qiskit_fermions.linalg.double_factorized")]
#[pyfunction(name = "double_factorized", signature = (two_body_tensor, tol, max_vecs=None, cholesky=true))]
pub fn py_double_factorized<'py>(
    py: Python<'py>,
    two_body_tensor: PyReadonlyArray4<f64>,
    tol: f64,
    max_vecs: Option<usize>,
    cholesky: bool,
) -> Vec<PyDoubleFactorizedTerm<'py>> {
    double_factorized(&two_body_tensor.as_array().to_owned(), tol, max_vecs, cholesky)
        .into_iter()
        .map(|term| term_into_py(py, term))
        .collect()
}

/// Double-factorized decomposition of spin-restricted :math:`t_2` amplitudes.
///
/// Factorizes :math:`t_{ijab}` (with :math:`i, j` occupied and :math:`a, b` virtual) into a list of
/// :math:`(Z, U)` terms suitable for reconstruction by :func:`reconstruct_t2`. The number of terms
/// is truncated to ``max_terms`` (default: all).
///
/// Arguments:
///     t2_amplitudes: the :math:`t_2` amplitudes of shape ``(nocc, nocc, nvrt, nvrt)``.
///     tol: the tolerance controlling the truncation error of the decomposition.
///     max_terms: the maximum number of terms to retain. Defaults to all.
///
/// Returns:
///     A list of :math:`(Z, U)` 2-tuples, where ``Z`` is the diagonal Coulomb matrix and ``U`` is
///     the orbital rotation.
#[gen_stub_pyfunction(module = "qiskit_fermions.linalg.double_factorized")]
#[pyfunction(name = "double_factorized_t2", signature = (t2_amplitudes, tol, max_terms=None))]
pub fn py_double_factorized_t2<'py>(
    py: Python<'py>,
    t2_amplitudes: PyReadonlyArray4<Complex64>,
    tol: f64,
    max_terms: Option<usize>,
) -> Vec<PyDoubleFactorizedTerm<'py>> {
    double_factorized_t2(&t2_amplitudes.as_array().to_owned(), tol, max_terms)
        .into_iter()
        .map(|term| term_into_py(py, term))
        .collect()
}

/// Reconstructs spin-restricted :math:`t_2` amplitudes from a double-factorized decomposition.
///
/// Computes
/// :math:`i \sum_k \sum_{pq} Z^{(k)}_{pq} U^{(k)}_{ap} U^{(k)*}_{ip} U^{(k)}_{bq} U^{(k)*}_{jq}`
/// and slices to the occupied/virtual block ``[:nocc, :nocc, nocc:, nocc:]``.
///
/// Arguments:
///     terms: the list of :math:`(Z, U)` 2-tuples produced by :func:`double_factorized_t2`.
///     nocc: the number of occupied orbitals.
///
/// Returns:
///     The reconstructed :math:`t_2` amplitudes of shape ``(nocc, nocc, nvrt, nvrt)``.
#[gen_stub_pyfunction(module = "qiskit_fermions.linalg.double_factorized")]
#[pyfunction(name = "reconstruct_t2", signature = (terms, nocc))]
pub fn py_reconstruct_t2<'py>(
    py: Python<'py>,
    terms: Vec<(PyReadonlyArray2<'py, f64>, PyReadonlyArray2<'py, Complex64>)>,
    nocc: usize,
) -> Bound<'py, PyArray4<Complex64>> {
    let terms: Vec<DoubleFactorizedTerm> = terms
        .into_iter()
        .map(|(z, u)| (z.as_array().to_owned(), u.as_array().to_owned()))
        .collect();
    reconstruct_t2(&terms, nocc).into_pyarray(py)
}

/// Double-factorized decomposition of alpha-beta (spin-unrestricted) :math:`t_2` amplitudes.
///
/// Returns a list of terms suitable for reconstruction by :func:`reconstruct_t2_alpha_beta`. The
/// number of terms is truncated to ``max_terms`` (default: all).
///
/// Arguments:
///     t2_amplitudes: the alpha-beta :math:`t_2` amplitudes of shape
///         ``(nocc_a, nocc_b, nvrt_a, nvrt_b)``.
///     tol: the tolerance controlling the truncation error of the decomposition.
///     max_terms: the maximum number of terms to retain. Defaults to all.
///
/// Returns:
///     A list of 2-tuples, each consisting of
///
///     * a 3-tuple of the diagonal Coulomb matrices in the order ``(aa, ab, bb)``
///     * a 2-tuple of the orbital rotations in the order ``(alpha, beta)``
#[gen_stub_pyfunction(module = "qiskit_fermions.linalg.double_factorized")]
#[pyfunction(name = "double_factorized_t2_alpha_beta", signature = (t2_amplitudes, tol, max_terms=None))]
pub fn py_double_factorized_t2_alpha_beta<'py>(
    py: Python<'py>,
    t2_amplitudes: PyReadonlyArray4<Complex64>,
    tol: f64,
    max_terms: Option<usize>,
) -> Vec<PyDoubleFactorizedT2AlphaBetaTerm<'py>> {
    double_factorized_t2_alpha_beta(&t2_amplitudes.as_array().to_owned(), tol, max_terms)
        .into_iter()
        .map(|term| alpha_beta_term_into_py(py, term))
        .collect()
}

/// Reconstructs alpha-beta :math:`t_2` amplitudes from a double-factorized decomposition.
///
/// Arguments:
///     terms: the list of terms produced by :func:`double_factorized_t2_alpha_beta`, each a 2-tuple
///         of the ``(aa, ab, bb)`` diagonal Coulomb matrices and the ``(alpha, beta)`` orbital
///         rotations.
///     norb: the number of orbitals.
///     nocc_a: the number of alpha-occupied orbitals.
///     nocc_b: the number of beta-occupied orbitals.
///
/// Returns:
///     The reconstructed alpha-beta :math:`t_2` amplitudes of shape
///     ``(nocc_a, nocc_b, nvrt_a, nvrt_b)``.
#[gen_stub_pyfunction(module = "qiskit_fermions.linalg.double_factorized")]
#[pyfunction(name = "reconstruct_t2_alpha_beta", signature = (terms, norb, nocc_a, nocc_b))]
pub fn py_reconstruct_t2_alpha_beta<'py>(
    py: Python<'py>,
    terms: Vec<(
        [PyReadonlyArray2<'py, f64>; 3],
        [PyReadonlyArray2<'py, Complex64>; 2],
    )>,
    norb: usize,
    nocc_a: usize,
    nocc_b: usize,
) -> Bound<'py, PyArray4<Complex64>> {
    let terms: Vec<DoubleFactorizedT2AlphaBetaTerm> = terms
        .into_iter()
        .map(|([aa, ab, bb], [alpha, beta])| DoubleFactorizedT2AlphaBetaTerm {
            diag_coulomb: [
                aa.as_array().to_owned(),
                ab.as_array().to_owned(),
                bb.as_array().to_owned(),
            ],
            orbital_rotations: [alpha.as_array().to_owned(), beta.as_array().to_owned()],
        })
        .collect();
    reconstruct_t2_alpha_beta(&terms, norb, nocc_a, nocc_b).into_pyarray(py)
}

#[pymodule]
pub mod double_factorized {
    #[pymodule_export]
    use super::py_modified_cholesky;
    #[pymodule_export]
    use super::py_double_factorized;
    #[pymodule_export]
    use super::py_double_factorized_t2;
    #[pymodule_export]
    use super::py_reconstruct_t2;
    #[pymodule_export]
    use super::py_double_factorized_t2_alpha_beta;
    #[pymodule_export]
    use super::py_reconstruct_t2_alpha_beta;
}
