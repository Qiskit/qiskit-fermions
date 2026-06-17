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

// The explicit (exact) decompositions in this module were translated from the Python
// implementation in the `ffsim` library
// (https://github.com/qiskit-community/ffsim, Apache-2.0, (C) IBM), specifically
// `ffsim/linalg/double_factorized_decomposition.py`. The `optimize=True` "compressed"
// code paths from the original are intentionally not ported here.

use nalgebra::DMatrix;
use ndarray::{Array1, Array2, Array4};
use num_complex::Complex64;

/// Accuracy threshold below which factorization terms are discarded by default.
///
/// Matches the default tolerance used by the original `ffsim` routines; callers can pass this
/// for the `tol` argument of the public decomposition functions.
pub const DEFAULT_TOL: f64 = 1e-8;

/// A single term of a double-factorized decomposition: a real symmetric diagonal Coulomb
/// matrix `Z` paired with a unitary orbital rotation `U`.
pub type DoubleFactorizedTerm = (Array2<f64>, Array2<Complex64>);

/// A single term of an alpha-beta (spin-unrestricted) `t2` double factorization.
///
/// The three diagonal Coulomb matrices are the (alpha-alpha, alpha-beta, beta-beta) blocks,
/// and the two orbital rotations are the (alpha, beta) rotations.
#[derive(Debug, Clone)]
pub struct DoubleFactorizedT2AlphaBetaTerm {
    /// Diagonal Coulomb matrices in the order `[aa, ab, bb]`.
    pub diag_coulomb: [Array2<f64>; 3],
    /// Orbital rotations in the order `[alpha, beta]`.
    pub orbital_rotations: [Array2<Complex64>; 2],
}

// ---------------------------------------------------------------------------------------------
// nalgebra <-> ndarray conversion helpers
//
// nalgebra is column-major; ndarray (by default) is row-major. To convert an ndarray matrix
// into a nalgebra matrix we build it from the row-major data interpreted as `(ncols, nrows)`
// column-major and transpose, or more simply use `DMatrix::from_row_slice`. To go back, we
// read the nalgebra matrix in row order via element access.
// ---------------------------------------------------------------------------------------------

/// Converts a row-major [`Array2`] into a column-major nalgebra [`DMatrix`].
fn ndarray_to_nalgebra<T: nalgebra::Scalar + Copy>(mat: &Array2<T>) -> DMatrix<T> {
    let (nrows, ncols) = mat.dim();
    // `from_row_slice` interprets the slice in row-major order, matching ndarray's standard
    // layout. Ensure standard layout first in case the input is a transposed/strided view.
    let standard = mat.as_standard_layout();
    DMatrix::from_row_slice(nrows, ncols, standard.as_slice().unwrap())
}

/// Converts a column-major nalgebra [`DMatrix`] into a row-major [`Array2`].
fn nalgebra_to_ndarray<T: nalgebra::Scalar + Copy>(mat: &DMatrix<T>) -> Array2<T> {
    let (nrows, ncols) = (mat.nrows(), mat.ncols());
    Array2::from_shape_fn((nrows, ncols), |(i, j)| mat[(i, j)])
}

// ---------------------------------------------------------------------------------------------
// Eigendecomposition / SVD wrappers
// ---------------------------------------------------------------------------------------------

/// Hermitian eigendecomposition of a complex Hermitian matrix.
///
/// Returns eigenvalues sorted in ascending order (matching numpy's `eigh`) together with the
/// corresponding eigenvectors as the columns of the returned matrix.
fn hermitian_eigh(mat: &Array2<Complex64>) -> (Array1<f64>, Array2<Complex64>) {
    let nalg = ndarray_to_nalgebra(mat);
    let eigen = nalg.symmetric_eigen();
    sort_eigen_ascending(&eigen.eigenvalues, &eigen.eigenvectors)
}

/// Sorts eigenpairs in ascending order of eigenvalue and converts them to ndarray types.
fn sort_eigen_ascending<T: nalgebra::Scalar + Copy>(
    eigenvalues: &nalgebra::DVector<f64>,
    eigenvectors: &DMatrix<T>,
) -> (Array1<f64>, Array2<T>) {
    let dim = eigenvalues.len();
    let mut order: Vec<usize> = (0..dim).collect();
    order.sort_by(|&a, &b| eigenvalues[a].partial_cmp(&eigenvalues[b]).unwrap());

    let sorted_vals = Array1::from_shape_fn(dim, |i| eigenvalues[order[i]]);
    let sorted_vecs = Array2::from_shape_fn((eigenvectors.nrows(), dim), |(row, col)| {
        eigenvectors[(row, order[col])]
    });
    (sorted_vals, sorted_vecs)
}

/// Thin SVD of a complex matrix.
///
/// Returns `(left_vecs, singular_vals, right_vecs)` where `left_vecs` has the left singular
/// vectors as columns, `singular_vals` is sorted in descending order, and `right_vecs` has the
/// right singular vectors as rows (i.e. it is `V^H`, matching `scipy.linalg.svd`).
fn thin_svd(mat: &Array2<Complex64>) -> (Array2<Complex64>, Array1<f64>, Array2<Complex64>) {
    let nalg = ndarray_to_nalgebra(mat);
    // `SVD::new` computes the thin SVD and sorts singular values in descending order.
    let svd = nalg.svd(true, true);
    let u: DMatrix<Complex64> = svd.u.expect("left singular vectors were requested");
    let v_t: DMatrix<Complex64> = svd.v_t.expect("right singular vectors were requested");
    let singular_vals =
        Array1::from_shape_fn(svd.singular_values.len(), |i| svd.singular_values[i]);
    (
        nalgebra_to_ndarray(&u),
        singular_vals,
        nalgebra_to_ndarray(&v_t),
    )
}

// ---------------------------------------------------------------------------------------------
// Truncation helpers
// ---------------------------------------------------------------------------------------------

/// Computes the number of trailing (smallest) values to discard so that the cumulative sum of
/// their absolute values stays strictly below `tol`.
///
/// This reproduces numpy's `searchsorted(cumsum(abs(values[::-1])), tol)` with the default
/// `side="left"`: the result is the number of entries of the reversed cumulative sum that are
/// strictly less than `tol`.
fn cumulative_discard_count(ascending_abs_tail: impl Iterator<Item = f64>, tol: f64) -> usize {
    let mut running = 0.0;
    let mut discard = 0;
    for value in ascending_abs_tail {
        running += value;
        if running < tol {
            discard += 1;
        } else {
            break;
        }
    }
    discard
}

/// Truncated Hermitian eigendecomposition.
///
/// Sorts eigenpairs by descending absolute eigenvalue, then discards the smallest-magnitude
/// eigenpairs whose cumulative absolute sum stays below `tol`, capping the number of retained
/// vectors at `max_vecs` (default: all).
fn truncated_eigh(
    mat: &Array2<Complex64>,
    tol: f64,
    max_vecs: Option<usize>,
) -> (Array1<f64>, Array2<Complex64>) {
    let (eigs, vecs) = hermitian_eigh(mat);
    let dim = eigs.len();

    // Order by descending |eigenvalue|.
    let mut order: Vec<usize> = (0..dim).collect();
    order.sort_by(|&a, &b| eigs[b].abs().partial_cmp(&eigs[a].abs()).unwrap());

    // The reversed (ascending-magnitude) tail used for the cumulative discard count.
    let ascending_abs = order.iter().rev().map(|&i| eigs[i].abs());
    let n_discard = cumulative_discard_count(ascending_abs, tol);

    let n_vecs = max_vecs.unwrap_or(dim).min(dim - n_discard);

    let kept_eigs = Array1::from_shape_fn(n_vecs, |i| eigs[order[i]]);
    let kept_vecs =
        Array2::from_shape_fn((vecs.nrows(), n_vecs), |(row, col)| vecs[[row, order[col]]]);
    (kept_eigs, kept_vecs)
}

/// Truncated thin SVD.
///
/// Discards the smallest singular values whose cumulative sum stays below `tol`, capping the
/// number of retained vectors at `max_vecs` (default: all). Singular values are non-negative
/// and already sorted in descending order.
fn truncated_svd(
    mat: &Array2<Complex64>,
    tol: f64,
    max_vecs: Option<usize>,
) -> (Array2<Complex64>, Array1<f64>, Array2<Complex64>) {
    let (left, singular_vals, right) = thin_svd(mat);
    let dim = singular_vals.len();

    let ascending = (0..dim).rev().map(|i| singular_vals[i]);
    let n_discard = cumulative_discard_count(ascending, tol);

    let n_vecs = max_vecs.unwrap_or(dim).min(dim - n_discard);

    let kept_left = Array2::from_shape_fn((left.nrows(), n_vecs), |(row, col)| left[[row, col]]);
    let kept_s = Array1::from_shape_fn(n_vecs, |i| singular_vals[i]);
    let kept_right = Array2::from_shape_fn((n_vecs, right.ncols()), |(row, col)| right[[row, col]]);
    (kept_left, kept_s, kept_right)
}

// ---------------------------------------------------------------------------------------------
// Modified Cholesky
// ---------------------------------------------------------------------------------------------

/// Modified Cholesky decomposition of a Hermitian positive-semidefinite matrix.
///
/// Decomposes `mat` into a sum of outer products `mat = Σ_i v_i v_i†`, returning the vectors
/// `v_i` as the columns of the result. The number of terms is governed by `tol`, while
/// `max_vecs` (default: the matrix dimension) is always respected even if it forces the
/// truncation error above `tol`.
pub fn modified_cholesky(
    mat: &Array2<Complex64>,
    tol: f64,
    max_vecs: Option<usize>,
) -> Array2<Complex64> {
    let dim = mat.nrows();
    if dim == 0 {
        return Array2::zeros((0, 0));
    }
    let max_vecs = max_vecs.unwrap_or(dim);

    let mut cholesky_vecs: Array2<Complex64> = Array2::zeros((dim, max_vecs + 1));
    let mut errors: Array1<f64> = Array1::from_shape_fn(dim, |i| mat[[i, i]].re);

    // Mirrors the Python `for index in range(max_vecs + 1)` loop: on a clean run `index` ends at
    // `max_vecs` (so the final filled column is excluded), and on an early `break` it equals the
    // number of columns filled so far. The returned slice is `cholesky_vecs[:, :index]`.
    let mut index = max_vecs;
    for current in 0..=max_vecs {
        // Index of the maximum remaining error.
        let mut max_error_index = 0;
        let mut max_error = errors[0];
        for i in 1..dim {
            if errors[i] > max_error {
                max_error = errors[i];
                max_error_index = i;
            }
        }
        if max_error < tol {
            index = current;
            break;
        }

        // New Cholesky vector starts as the selected column of `mat`.
        for row in 0..dim {
            cholesky_vecs[[row, current]] = mat[[row, max_error_index]];
        }
        // Orthogonalize against the previously computed vectors.
        if current > 0 {
            for k in 0..current {
                let coeff = cholesky_vecs[[max_error_index, k]].conj();
                for row in 0..dim {
                    let update = cholesky_vecs[[row, k]] * coeff;
                    cholesky_vecs[[row, current]] -= update;
                }
            }
        }
        // Normalize and update the running errors.
        let scale = max_error.sqrt();
        for row in 0..dim {
            cholesky_vecs[[row, current]] /= scale;
            errors[row] -= cholesky_vecs[[row, current]].norm_sqr();
        }
    }

    cholesky_vecs.slice(ndarray::s![.., ..index]).to_owned()
}

// ---------------------------------------------------------------------------------------------
// Two-body tensor: double_factorized
// ---------------------------------------------------------------------------------------------

/// Double-factorized decomposition of a real two-body tensor.
///
/// Represents `h_{pqrs} = Σ_t Σ_{kl} Z^{(t)}_{kl} U^{(t)}_{pk} U^{(t)}_{qk} U^{(t)}_{rl} U^{(t)}_{sl}`,
/// returning a list of `(Z, U)` terms where each `Z` is a real symmetric diagonal Coulomb matrix
/// and each `U` is a unitary orbital rotation.
///
/// When `cholesky` is `true` (the default behavior in the original library) the outer
/// factorization uses a modified Cholesky decomposition; otherwise it uses a truncated
/// eigendecomposition.
pub fn double_factorized(
    two_body_tensor: &Array4<f64>,
    tol: f64,
    max_vecs: Option<usize>,
    cholesky: bool,
) -> Vec<DoubleFactorizedTerm> {
    let norb = two_body_tensor.shape()[0];
    if norb == 0 {
        return Vec::new();
    }
    let max_vecs = max_vecs.unwrap_or(norb * (norb + 1) / 2);

    let norb = two_body_tensor.shape()[0];
    let reshaped = reshaped_two_body(two_body_tensor);

    // Produce the outer factorization columns and their coefficients. The Cholesky path folds the
    // scale into the vectors themselves (unit coefficients); the eigh path keeps the eigenvalues
    // as coefficients.
    let (outer_columns, outer_coeffs) = if cholesky {
        let cholesky_vecs = modified_cholesky(&reshaped, tol, Some(max_vecs));
        let n_vecs = cholesky_vecs.ncols();
        (cholesky_vecs, vec![1.0; n_vecs])
    } else {
        let (outer_eigs, outer_vecs) = truncated_eigh(&reshaped, tol, Some(max_vecs));
        let coeffs = outer_eigs.to_vec();
        (outer_vecs, coeffs)
    };

    // Each column reshapes to a `(norb, norb)` matrix.
    let outer_mats: Vec<Array2<Complex64>> = (0..outer_coeffs.len())
        .map(|t| Array2::from_shape_fn((norb, norb), |(p, q)| outer_columns[[p * norb + q, t]]))
        .collect();
    terms_from_outer_matrices(&outer_mats, &outer_coeffs)
}

/// Reshapes a two-body tensor `(norb, norb, norb, norb)` into a `(norb², norb²)` complex matrix.
fn reshaped_two_body(two_body_tensor: &Array4<f64>) -> Array2<Complex64> {
    let norb = two_body_tensor.shape()[0];
    Array2::from_shape_fn((norb * norb, norb * norb), |(row, col)| {
        let p = row / norb;
        let q = row % norb;
        let r = col / norb;
        let s = col % norb;
        Complex64::new(two_body_tensor[[p, q, r, s]], 0.0)
    })
}

/// Builds a diagonal Coulomb matrix as the scaled outer product `coeff * eigs_k[k] * eigs_l[l]`.
///
/// The result has shape `(eigs_k.len(), eigs_l.len())`. Passing the same slice for both arguments
/// yields the common symmetric `coeff * eigs[k] * eigs[l]` form.
fn diag_coulomb_outer(coeff: f64, eigs_k: &[f64], eigs_l: &[f64]) -> Array2<f64> {
    Array2::from_shape_fn((eigs_k.len(), eigs_l.len()), |(k, l)| {
        coeff * eigs_k[k] * eigs_l[l]
    })
}

/// Builds the per-term `(Z, U)` factors from a set of reshaped outer matrices, scaling each
/// diagonal Coulomb matrix by an associated outer coefficient.
fn terms_from_outer_matrices(
    outer_mats: &[Array2<Complex64>],
    outer_coeffs: &[f64],
) -> Vec<DoubleFactorizedTerm> {
    outer_mats
        .iter()
        .zip(outer_coeffs.iter())
        .map(|(mat, &coeff)| {
            let (eigs, rotation) = hermitian_eigh(mat);
            let eigs = eigs.as_slice().unwrap();
            let diag_coulomb = diag_coulomb_outer(coeff, eigs, eigs);
            (diag_coulomb, rotation)
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------
// t2 amplitudes (spin-restricted)
// ---------------------------------------------------------------------------------------------

/// Places a flat occupied×virtual vector into the (virtual, occupied) block of a `(norb, norb)`
/// zero matrix. The flat index runs in `product(occupied, virtual)` order (occupied outer,
/// virtual inner): `mat[row, col]` for `row` in `[nocc, norb)` and `col` in `[0, nocc)`.
fn place_ov_block(norb: usize, nocc: usize, get: impl Fn(usize) -> Complex64) -> Array2<Complex64> {
    let mut mat: Array2<Complex64> = Array2::zeros((norb, norb));
    let mut entry = 0;
    for col in 0..nocc {
        for row in nocc..norb {
            mat[[row, col]] = get(entry);
            entry += 1;
        }
    }
    mat
}

/// The "quadrature" gadget that turns a placement matrix into a Hermitian one-body tensor:
/// `0.5 * (1 - sign*i) * (mat + sign*i*mat†)`.
fn quadrature(mat: &Array2<Complex64>, sign: f64) -> Array2<Complex64> {
    let i = Complex64::new(0.0, 1.0);
    let prefactor = Complex64::new(0.5, 0.0) * (Complex64::new(1.0, 0.0) - sign * i);
    let (nrows, ncols) = mat.dim();
    Array2::from_shape_fn((nrows, ncols), |(r, c)| {
        let adjoint = mat[[c, r]].conj();
        prefactor * (mat[[r, c]] + sign * i * adjoint)
    })
}

/// Double-factorized decomposition of spin-restricted `t2` amplitudes.
///
/// Factorizes `t_{ijab}` (with `i,j` occupied and `a,b` virtual) into a list of `(Z, U)` terms
/// suitable for reconstruction by [`reconstruct_t2`]. The number of terms is truncated to
/// `max_terms` (default: all).
pub fn double_factorized_t2(
    t2_amplitudes: &Array4<Complex64>,
    tol: f64,
    max_terms: Option<usize>,
) -> Vec<DoubleFactorizedTerm> {
    let nocc = t2_amplitudes.shape()[0];
    let nvrt = t2_amplitudes.shape()[2];
    let norb = nocc + nvrt;

    // Transpose axes (0, 2, 1, 3) then reshape to (nocc*nvrt, nocc*nvrt).
    let t2_mat = Array2::from_shape_fn((nocc * nvrt, nocc * nvrt), |(row, col)| {
        let i = row / nvrt;
        let a = row % nvrt;
        let j = col / nvrt;
        let b = col % nvrt;
        t2_amplitudes[[i, j, a, b]]
    });

    let (outer_eigs, outer_vecs) = truncated_eigh(&t2_mat, tol, None);
    let n_vecs = outer_eigs.len();

    let mut terms: Vec<DoubleFactorizedTerm> = Vec::with_capacity(2 * n_vecs);
    for t in 0..n_vecs {
        // Place the outer vector into the occupied x virtual block.
        let mat = place_ov_block(norb, nocc, |entry| outer_vecs[[entry, t]]);

        for (sign, coeff_sign) in [(1.0, 1.0), (-1.0, -1.0)] {
            let one_body = quadrature(&mat, sign);
            let (eigs, rotation) = hermitian_eigh(&one_body);
            let coeff = coeff_sign * outer_eigs[t];
            let eigs = eigs.as_slice().unwrap();
            let diag_coulomb = diag_coulomb_outer(coeff, eigs, eigs);
            terms.push((diag_coulomb, rotation));
        }
    }

    if let Some(max_terms) = max_terms {
        terms.truncate(max_terms);
    }
    terms
}

/// Reconstructs spin-restricted `t2` amplitudes from a double-factorized decomposition.
///
/// Computes `i * Σ_k Σ_{pq} Z^{(k)}_{pq} U^{(k)}_{ap} U^{(k)*}_{ip} U^{(k)}_{bq} U^{(k)*}_{jq}`
/// and slices to the occupied/virtual block `[:nocc, :nocc, nocc:, nocc:]`.
pub fn reconstruct_t2(terms: &[DoubleFactorizedTerm], nocc: usize) -> Array4<Complex64> {
    let norb = if terms.is_empty() {
        0
    } else {
        terms[0].1.nrows()
    };
    let nvrt = norb - nocc;
    let i_unit = Complex64::new(0.0, 1.0);

    let mut result: Array4<Complex64> = Array4::zeros((nocc, nocc, nvrt, nvrt));
    for (z, u) in terms {
        for occ_i in 0..nocc {
            for occ_j in 0..nocc {
                for vrt_a in 0..nvrt {
                    for vrt_b in 0..nvrt {
                        let a = nocc + vrt_a;
                        let b = nocc + vrt_b;
                        let mut acc = Complex64::new(0.0, 0.0);
                        for p in 0..norb {
                            for q in 0..norb {
                                acc += z[[p, q]]
                                    * u[[a, p]]
                                    * u[[occ_i, p]].conj()
                                    * u[[b, q]]
                                    * u[[occ_j, q]].conj();
                            }
                        }
                        result[[occ_i, occ_j, vrt_a, vrt_b]] += i_unit * acc;
                    }
                }
            }
        }
    }
    result
}

// ---------------------------------------------------------------------------------------------
// t2 amplitudes (alpha-beta / unrestricted)
// ---------------------------------------------------------------------------------------------

/// Double-factorized decomposition of alpha-beta (spin-unrestricted) `t2` amplitudes.
///
/// Returns a list of [`DoubleFactorizedT2AlphaBetaTerm`] suitable for reconstruction by
/// [`reconstruct_t2_alpha_beta`]. The number of terms is truncated to `max_terms` (default: all).
pub fn double_factorized_t2_alpha_beta(
    t2_amplitudes: &Array4<Complex64>,
    tol: f64,
    max_terms: Option<usize>,
) -> Vec<DoubleFactorizedT2AlphaBetaTerm> {
    let nocc_a = t2_amplitudes.shape()[0];
    let nocc_b = t2_amplitudes.shape()[1];
    let nvrt_a = t2_amplitudes.shape()[2];
    let nvrt_b = t2_amplitudes.shape()[3];
    let norb = nocc_a + nvrt_a;

    // Transpose axes (0, 2, 1, 3) then reshape to (nocc_a*nvrt_a, nocc_b*nvrt_b).
    let t2_mat = Array2::from_shape_fn((nocc_a * nvrt_a, nocc_b * nvrt_b), |(row, col)| {
        let i = row / nvrt_a;
        let a = row % nvrt_a;
        let j = col / nvrt_b;
        let b = col % nvrt_b;
        t2_amplitudes[[i, j, a, b]]
    });

    let (left, singular_vals, right) = truncated_svd(&t2_mat, tol, None);
    let n_vecs = singular_vals.len();

    // The four per-outer-vector "rows" each pair an alpha one-body tensor (from `left_mat`) with
    // a beta one-body tensor (from `right_mat`), built via the quadrature gadget. The layout
    // mirrors ffsim's `one_body_tensors[n_vecs, 2, 2, 2, norb, norb]` reshaped to
    // `(n_vecs, 4, 2, norb, norb)`:
    //   row 0: alpha=quad(left, +1), beta=quad(+right, +1)
    //   row 1: alpha=quad(left, +1), beta=quad(-right, +1)
    //   row 2: alpha=quad(left, -1), beta=quad(+right, -1)
    //   row 3: alpha=quad(left, -1), beta=quad(-right, -1)
    // and `coeffs = 0.5 * [1, -1, -1, 1] * singular_val`.
    let coeff_signs = [1.0, -1.0, -1.0, 1.0];

    let mut terms: Vec<DoubleFactorizedT2AlphaBetaTerm> = Vec::with_capacity(4 * n_vecs);
    for t in 0..n_vecs {
        // Place the left/right singular vectors into the occupied x virtual blocks.
        let left_mat = place_ov_block(norb, nocc_a, |entry| left[[entry, t]]);
        let right_mat = place_ov_block(norb, nocc_b, |entry| right[[t, entry]]);
        let neg_right_mat = right_mat.mapv(|x| -x);

        let rows: [(f64, &Array2<Complex64>); 4] = [
            (1.0, &right_mat),
            (1.0, &neg_right_mat),
            (-1.0, &right_mat),
            (-1.0, &neg_right_mat),
        ];

        for (row_idx, (sign, beta_mat)) in rows.iter().enumerate() {
            let one_body_a = quadrature(&left_mat, *sign);
            let one_body_b = quadrature(beta_mat, *sign);
            let (eigs_a, rotation_a) = hermitian_eigh(&one_body_a);
            let (eigs_b, rotation_b) = hermitian_eigh(&one_body_b);

            let coeff = 0.5 * coeff_signs[row_idx] * singular_vals[t];

            // The big diagonal Coulomb matrix is the `coeff`-scaled outer product of the
            // concatenated alpha/beta eigenvalues, sliced into the aa/ab/bb blocks.
            let alpha = eigs_a.as_slice().unwrap();
            let beta = eigs_b.as_slice().unwrap();
            let aa = diag_coulomb_outer(coeff, alpha, alpha);
            let ab = diag_coulomb_outer(coeff, alpha, beta);
            let bb = diag_coulomb_outer(coeff, beta, beta);

            terms.push(DoubleFactorizedT2AlphaBetaTerm {
                diag_coulomb: [aa, ab, bb],
                orbital_rotations: [rotation_a, rotation_b],
            });
        }
    }

    if let Some(max_terms) = max_terms {
        terms.truncate(max_terms);
    }
    terms
}

/// Reconstructs alpha-beta `t2` amplitudes from a double-factorized decomposition.
///
/// Each term's three diagonal Coulomb blocks are assembled into a `(2*norb, 2*norb)` matrix and
/// the two rotations into a block-diagonal `(2*norb, 2*norb)` rotation; the same contraction as
/// [`reconstruct_t2`] is then applied and sliced to the alpha-occupied / beta-occupied /
/// alpha-virtual / beta-virtual block.
pub fn reconstruct_t2_alpha_beta(
    terms: &[DoubleFactorizedT2AlphaBetaTerm],
    norb: usize,
    nocc_a: usize,
    nocc_b: usize,
) -> Array4<Complex64> {
    let nvrt_a = norb - nocc_a;
    let nvrt_b = norb - nocc_b;
    let big = 2 * norb;
    let i_unit = Complex64::new(0.0, 1.0);

    let mut result: Array4<Complex64> = Array4::zeros((nocc_a, nocc_b, nvrt_a, nvrt_b));
    for term in terms {
        let [aa, ab, bb] = &term.diag_coulomb;
        let [rot_a, rot_b] = &term.orbital_rotations;

        // Expanded (2*norb, 2*norb) diagonal Coulomb matrix: [[aa, ab], [ab^T, bb]].
        let z = Array2::from_shape_fn((big, big), |(row, col)| {
            if row < norb && col < norb {
                aa[[row, col]]
            } else if row < norb {
                ab[[row, col - norb]]
            } else if col < norb {
                ab[[col, row - norb]] // ab^T
            } else {
                bb[[row - norb, col - norb]]
            }
        });
        // Block-diagonal rotation: diag(rot_a, rot_b).
        let u = Array2::from_shape_fn((big, big), |(row, col)| {
            if row < norb && col < norb {
                rot_a[[row, col]]
            } else if row >= norb && col >= norb {
                rot_b[[row - norb, col - norb]]
            } else {
                Complex64::new(0.0, 0.0)
            }
        });

        // Contract and slice to [:nocc_a, norb:norb+nocc_b, nocc_a:norb, norb+nocc_b:].
        for occ_i in 0..nocc_a {
            for occ_j in 0..nocc_b {
                let global_j = norb + occ_j;
                for vrt_a in 0..nvrt_a {
                    let a = nocc_a + vrt_a;
                    for vrt_b in 0..nvrt_b {
                        let b = norb + nocc_b + vrt_b;
                        let mut acc = Complex64::new(0.0, 0.0);
                        for p in 0..big {
                            for q in 0..big {
                                acc += z[[p, q]]
                                    * u[[a, p]]
                                    * u[[occ_i, p]].conj()
                                    * u[[b, q]]
                                    * u[[global_j, q]].conj();
                            }
                        }
                        result[[occ_i, occ_j, vrt_a, vrt_b]] += i_unit * acc;
                    }
                }
            }
        }
    }
    result
}

impl DoubleFactorizedT2AlphaBetaTerm {
    /// The number of orbitals `norb` represented by this term.
    pub fn norb(&self) -> usize {
        self.orbital_rotations[0].nrows()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::Array2;

    use crate::random::random_unitary;

    /// Returns `true` if two 4-D tensors agree elementwise within `tol`.
    fn tensors_approx_equal(a: &Array4<Complex64>, b: &Array4<Complex64>, tol: f64) -> bool {
        if a.shape() != b.shape() {
            return false;
        }
        a.iter().zip(b.iter()).all(|(x, y)| (x - y).norm() <= tol)
    }

    /// Builds a random Hermitian positive-semidefinite matrix `A A†` of the given dimension.
    fn random_psd(dim: usize) -> Array2<Complex64> {
        let a = random_unitary(dim);
        a.dot(&a.t().mapv(|x| x.conj()))
    }

    /// Builds a random real two-body tensor whose reshaped `(norb², norb²)` matrix is symmetric
    /// positive semidefinite — the structure of an electronic-structure two-electron integral
    /// tensor, and required by the modified-Cholesky path.
    ///
    /// We construct `h_{pqrs} = Σ_t g^{(t)}_{pq} g^{(t)}_{rs}` for random real symmetric `g^{(t)}`,
    /// which makes the reshaped matrix `M[pq, rs] = Σ_t vec(g^{(t)})_{pq} vec(g^{(t)})_{rs}` PSD by
    /// construction. This tensor is exactly double-factorizable by both the Cholesky and eigh
    /// paths.
    fn random_two_body_tensor(norb: usize) -> Array4<f64> {
        let n_terms = norb; // enough terms for a nontrivial spectrum
        let mut gs: Vec<Array2<f64>> = Vec::with_capacity(n_terms);
        for term in 0..n_terms {
            let g = Array2::<f64>::from_shape_fn((norb, norb), |(p, q)| {
                ((p + 1) as f64 * 0.37 - (q as f64) * 0.21 + term as f64 * 0.5).sin()
            });
            gs.push(&g + &g.t());
        }
        Array4::from_shape_fn((norb, norb, norb, norb), |(p, q, r, s)| {
            gs.iter().map(|g| g[[p, q]] * g[[r, s]]).sum()
        })
    }

    /// Reconstructs a two-body tensor from its double-factorized terms.
    fn reconstruct_two_body(terms: &[DoubleFactorizedTerm], norb: usize) -> Array4<f64> {
        let mut result = Array4::<f64>::zeros((norb, norb, norb, norb));
        for (z, u) in terms {
            for p in 0..norb {
                for q in 0..norb {
                    for r in 0..norb {
                        for s in 0..norb {
                            let mut acc = 0.0;
                            for k in 0..norb {
                                for l in 0..norb {
                                    acc += z[[k, l]]
                                        * (u[[p, k]] * u[[q, k]] * u[[r, l]] * u[[s, l]]).re;
                                }
                            }
                            result[[p, q, r, s]] += acc;
                        }
                    }
                }
            }
        }
        result
    }

    fn tensors_real_approx_equal(a: &Array4<f64>, b: &Array4<f64>, tol: f64) -> bool {
        if a.shape() != b.shape() {
            return false;
        }
        a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() <= tol)
    }

    #[test]
    fn test_nalgebra_ndarray_roundtrip() {
        for dim in 1..=5 {
            let mat = random_unitary(dim);
            let back = nalgebra_to_ndarray(&ndarray_to_nalgebra(&mat));
            assert!(
                mat.iter()
                    .zip(back.iter())
                    .all(|(x, y)| (x - y).norm() < 1e-12),
                "round-trip conversion failed for dim {dim}"
            );
        }
    }

    #[test]
    fn test_modified_cholesky() {
        for dim in 1..=6 {
            let mat = random_psd(dim);
            let vecs = modified_cholesky(&mat, 1e-12, None);
            // Reconstruct: sum_i v_i v_i^dagger.
            let reconstructed = vecs.dot(&vecs.t().mapv(|x| x.conj()));
            assert!(
                mat.iter()
                    .zip(reconstructed.iter())
                    .all(|(x, y)| (x - y).norm() < 1e-8),
                "modified Cholesky reconstruction failed for dim {dim}"
            );
        }
    }

    #[test]
    fn test_double_factorized_cholesky() {
        for norb in 2..=4 {
            let tensor = random_two_body_tensor(norb);
            let terms = double_factorized(&tensor, 1e-10, None, true);
            let reconstructed = reconstruct_two_body(&terms, norb);
            assert!(
                tensors_real_approx_equal(&tensor, &reconstructed, 1e-8),
                "cholesky double factorization round-trip failed for norb {norb}"
            );
        }
    }

    #[test]
    fn test_double_factorized_eigh() {
        for norb in 2..=4 {
            let tensor = random_two_body_tensor(norb);
            let terms = double_factorized(&tensor, 1e-10, None, false);
            let reconstructed = reconstruct_two_body(&terms, norb);
            assert!(
                tensors_real_approx_equal(&tensor, &reconstructed, 1e-8),
                "eigh double factorization round-trip failed for norb {norb}"
            );
        }
    }

    #[test]
    fn test_double_factorized_max_vecs_truncates() {
        let norb = 4;
        let tensor = random_two_body_tensor(norb);
        let full = double_factorized(&tensor, 1e-10, None, true);
        let truncated = double_factorized(&tensor, 1e-10, Some(1), true);
        assert!(truncated.len() <= full.len());
        assert!(truncated.len() <= 1);
    }

    #[test]
    fn test_double_factorized_t2_roundtrip() {
        for (nocc, nvrt) in [(1usize, 2usize), (2, 2), (2, 3)] {
            let norb = nocc + nvrt;
            // Build a random t2 from a known decomposition so reconstruction is exact.
            let t2 = random_t2(nocc, nvrt);
            let terms = double_factorized_t2(&t2, 1e-12, None);
            let reconstructed = reconstruct_t2(&terms, nocc);
            assert!(
                tensors_approx_equal(&t2, &reconstructed, 1e-8),
                "t2 round-trip failed for nocc {nocc} nvrt {nvrt} (norb {norb})"
            );
        }
    }

    #[test]
    fn test_double_factorized_t2_alpha_beta_roundtrip() {
        for (nocc, nvrt) in [(1usize, 2usize), (2, 2)] {
            let norb = nocc + nvrt;
            let t2 = random_t2_alpha_beta(nocc, nvrt);
            let terms = double_factorized_t2_alpha_beta(&t2, 1e-12, None);
            let reconstructed = reconstruct_t2_alpha_beta(&terms, norb, nocc, nocc);
            assert!(
                tensors_approx_equal(&t2, &reconstructed, 1e-8),
                "alpha-beta t2 round-trip failed for nocc {nocc} nvrt {nvrt}"
            );
        }
    }

    /// Builds a random alpha-beta `t2` tensor `t_{ijab}` (alpha indices `i,a`, beta indices
    /// `j,b`). The opposite-spin amplitudes carry no permutation symmetry, so any tensor is
    /// exactly representable by the SVD-based factorization. The values are real here, matching
    /// `ffsim.random.random_t2_amplitudes` with `dtype=float`.
    fn random_t2_alpha_beta(nocc: usize, nvrt: usize) -> Array4<Complex64> {
        let mut counter = 0.0_f64;
        Array4::from_shape_fn((nocc, nocc, nvrt, nvrt), |_| {
            counter += 1.0;
            Complex64::new((counter * 0.53).cos(), 0.0)
        })
    }

    /// Builds a random `t2` tensor `t_{ijab}` with the symmetry of restricted CCSD amplitudes,
    /// matching `ffsim.random.random_t2_amplitudes`.
    ///
    /// The amplitudes satisfy `t2[i,j,a,b] == t2[j,i,b,a]`, which makes the reshaped matrix
    /// `t2_mat = t2.transpose(0,2,1,3).reshape(nocc*nvrt, nocc*nvrt)` real symmetric and hence
    /// exactly representable by the explicit (Hermitian-eigendecomposition) factorization.
    fn random_t2(nocc: usize, nvrt: usize) -> Array4<Complex64> {
        // Deterministic pseudo-random real values, assigned to the mirrored positions
        // (i,a,j,b) and (j,b,i,a) so the CCSD symmetry holds exactly.
        let mut t2 = Array4::<Complex64>::zeros((nocc, nocc, nvrt, nvrt));
        let mut counter = 0.0_f64;
        for i in 0..nocc {
            for a in 0..nvrt {
                for j in 0..nocc {
                    for b in 0..nvrt {
                        // Iterate over combinations_with_replacement of (occ, virt) pairs.
                        if (i, a) <= (j, b) {
                            counter += 1.0;
                            let val = Complex64::new((counter * 0.7).sin(), 0.0);
                            t2[[i, j, a, b]] = val;
                            t2[[j, i, b, a]] = val;
                        }
                    }
                }
            }
        }
        t2
    }
}
