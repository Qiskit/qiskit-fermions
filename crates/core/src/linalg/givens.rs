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

// The original implementation of this module was translated from Python with the assistance of the
// AI coding tool Bob 1.0.0. It has been reviewed and significantly refactored manually since then.

use ndarray::{Array2, s};
use num_complex::Complex64;

/// Accuracy below which a floating point number is considered equivalent to zero.
const EPSILON: f64 = 1e-10;

/// Type alias for a Givens rotation: (real cosine, complex sine, row/col index i, row/col index j)
pub type GivensRotation = (f64, Complex64, usize, usize);

/// Enum for tracking matrix slice operations
#[derive(Debug, Clone, Copy)]
enum SliceType {
    Row,
    Column,
}

/// Computes the real cosine and complex sine of a Givens rotation resulting in a plane rotation as:
///
///   [  c         s ] [ a ] = [ r ]
///   [ -conjg(s)  c ] [ b ]   [ 0 ]
///
/// BLAS provides an implementation of zrotg, but in this implementation we guard also against
/// scenarios where either `a` or `b` are very close to zero.
fn zrotg(a: Complex64, b: Complex64) -> (f64, Complex64) {
    // Handle case that a is zero
    if a.norm() < EPSILON {
        (0.0, Complex64::new(1.0, 0.0))
    }
    // Handle case that b is zero and a is nonzero
    else if b.norm() < EPSILON {
        (1.0, Complex64::new(0.0, 0.0))
    }
    // Handle case that a and b are both nonzero
    else {
        let hypotenuse = (a.norm().powi(2) + b.norm().powi(2)).sqrt();
        let c = a.norm() / hypotenuse;
        let sign_a = a / a.norm();
        let s = sign_a * b.conj() / hypotenuse;

        (c, s)
    }
}

/// Applies a [GivensRotation] to a specific slice (row or column) of a target matrix.
///
/// This function performs the following in-place operation:
/// - For columns: target[:, i] and target[:, j] are transformed by the Givens rotation
/// - For rows: target[i, :] and target[j, :] are transformed by the Givens rotation
///
/// # Arguments
///
/// * `target` - The target matrix to transform
/// * `givens` - The Givens rotation to apply
/// * `slice_type` - Whether to apply to rows or columns
///
/// BLAS provides an implementation of zrot, but this is not currently exposed correctly in the
/// LAPACK headers and therefore not picked up by the bindings available to Rust.
fn zrot(target: &mut Array2<Complex64>, givens: &GivensRotation, slice_type: SliceType) {
    let (c, s, i, j) = givens;

    match slice_type {
        SliceType::Column => {
            // Apply to columns: target[:, i] and target[:, j]
            let n_rows = target.nrows();

            let (mut col_i, mut col_j) = target.multi_slice_mut((s![.., *i], s![.., *j]));

            (0..n_rows).for_each(|row| {
                let tmp = c * col_i[row] + s * col_j[row];
                col_j[row] = -s.conj() * col_i[row] + c * col_j[row];
                col_i[row] = tmp;
            });
        }
        SliceType::Row => {
            // Apply to rows: target[i, :] and target[j, :]
            let n_cols = target.ncols();

            let (mut row_i, mut row_j) = target.multi_slice_mut((s![*i, ..], s![*j, ..]));

            (0..n_cols).for_each(|col| {
                let tmp = c * row_i[col] + s * row_j[col];
                row_j[col] = -s.conj() * row_i[col] + c * row_j[col];
                row_i[col] = tmp;
            });
        }
    }
}

/// Decomposes a unitary matrix into Givens rotations and diagonal phases.
///
/// This function implements the Givens decomposition algorithm, which expresses a unitary matrix
/// as a product of Givens rotations (2x2 unitary matrices acting on pairs of indices) and a
/// diagonal phase matrix.
///
/// # Arguments
///
/// * `unitary` - A unitary matrix to decompose
///
/// # Returns
///
/// A tuple containing:
/// - A vector of Givens rotations, each given as (real cosine, complex sine, index i, index j)
/// - A vector of complex diagonal phase factors
pub fn givens_decomposition(unitary: Array2<Complex64>) -> (Vec<GivensRotation>, Vec<Complex64>) {
    let n = unitary.nrows();
    let mut matrix = unitary.clone();
    let mut left_rotations: Vec<GivensRotation> = Vec::new();
    let mut right_rotations: Vec<GivensRotation> = Vec::new();

    // Compute left and right Givens rotations
    for i in 0..n - 1 {
        if i % 2 == 0 {
            // Even iterations: rotate columns by right multiplication
            for j in 0..=i {
                let target_index = i - j;
                let row = n - j - 1;

                if matrix[[row, target_index]].norm() > EPSILON {
                    // Zero out element at target index in given row
                    let (c, s) =
                        zrotg(matrix[[row, target_index + 1]], matrix[[row, target_index]]);

                    let givens = (c, s, target_index + 1, target_index);
                    right_rotations.push(givens);

                    zrot(&mut matrix, &givens, SliceType::Column);
                }
            }
        } else {
            // Odd iterations: rotate rows by left multiplication
            for j in 0..=i {
                let target_index = n - i + j - 1;
                let col = j;

                if matrix[[target_index, col]].norm() > EPSILON {
                    // Zero out element at target index in given column
                    let (c, s) =
                        zrotg(matrix[[target_index - 1, col]], matrix[[target_index, col]]);

                    let givens = (c, s, target_index - 1, target_index);
                    left_rotations.push(givens);

                    zrot(&mut matrix, &givens, SliceType::Row);
                }
            }
        }
    }

    // Convert left rotations to right rotations
    for (c_left, s_left, i, j) in left_rotations.into_iter().rev() {
        // Get current matrix scaling factors
        let phase_i = matrix[[i, i]];
        let phase_j = matrix[[j, j]];

        // Compute scaled Givens rotation - this will be the left-to-right converted rotation
        let (c_right, s_right) = zrotg(phase_j * c_left, phase_i * s_left.conj());

        // Store as right rotation (transposed)
        right_rotations.push((c_right, -s_right.conj(), i, j));

        // Update diagonal phases - from the diagonal values of the dot product between the adjoint
        // (original) left Givens rotation with the (new) right Givens rotation.
        matrix[[i, i]] = c_left * phase_i * c_right + s_left * phase_j * s_right.conj();
        matrix[[j, j]] = s_left.conj() * phase_i * s_right + c_left * phase_j * c_right;
    }

    // Extract diagonal phases
    let diagonal_phases = matrix.diag().to_vec();

    (right_rotations, diagonal_phases)
}

/// Decomposes the occupied-orbital coefficients of a Slater determinant into Givens rotations.
///
/// This is the rectangular counterpart of [`givens_decomposition`], specialized for Slater
/// determinant `state preparation`. The input `orbital_coeffs` is an :math:`m \times n` matrix whose
/// :math:`m` rows are the occupied orbitals expressed in a basis of :math:`n` spatial orbitals
/// (:math:`m \le n`); its rows are assumed to be orthonormal.
///
/// Applying the returned rotations, in order, to the reference configuration in which the first
/// :math:`m` orbitals are occupied prepares the target Slater determinant. Because preparing a
/// Slater determinant only requires realizing the :math:`m` occupied orbitals -- rather than a full
/// :math:`n \times n` orbital rotation -- this uses at most :math:`m (n - m)` Givens rotations
/// arranged in a diamond-shaped pattern (versus the brick-wall :math:`n (n - 1) / 2` of the square
/// decomposition). The decomposition carries `no` diagonal phases: a global phase and any rotation
/// within the occupied space leave the prepared Slater determinant unchanged, so both are discarded.
///
/// # Arguments
///
/// * `orbital_coeffs` - the :math:`m \times n` matrix of occupied-orbital coefficients.
///
/// # Returns
///
/// A vector of Givens rotations, each given as (real cosine, complex sine, index i, index j) acting
/// on the adjacent indices i and j.
pub fn givens_decomposition_slater(orbital_coeffs: Array2<Complex64>) -> Vec<GivensRotation> {
    let m = orbital_coeffs.nrows();
    let n = orbital_coeffs.ncols();
    let mut matrix = orbital_coeffs;

    // Zero out the top-right corner by rotating rows; this is a no-op on the prepared determinant
    // (it mixes only unoccupied orbitals) but brings the matrix into the shape the column sweep
    // below expects.
    if n > m {
        let n_minus_m = n - m;
        for j in (n_minus_m + 1..n).rev() {
            // zero out the entries in column j from the top down
            for i in 0..(j - n_minus_m) {
                if matrix[[i, j]].norm() > EPSILON {
                    let (c, s) = zrotg(matrix[[i + 1, j]], matrix[[i, j]]);
                    zrot(&mut matrix, &(c, s, i + 1, i), SliceType::Row);
                }
            }
        }
    }

    // Decompose the matrix into Givens rotations, zeroing each row's trailing entries by rotating
    // adjacent columns.
    let mut rotations: Vec<GivensRotation> = Vec::new();
    for i in 0..m {
        let j_max = n - m + i;
        for j in (i + 1..=j_max).rev() {
            if matrix[[i, j]].norm() > EPSILON {
                let (c, s) = zrotg(matrix[[i, j - 1]], matrix[[i, j]]);
                rotations.push((c, s, j, j - 1));
                zrot(&mut matrix, &(c, s, j - 1, j), SliceType::Column);
            }
        }
    }

    rotations.reverse();
    rotations
}

#[cfg(test)]
mod tests {
    use super::*;

    use nalgebra::DMatrix;
    use ndarray::arr1;

    use crate::random::random_unitary;
    use crate::testing::matrices_approx_equal;

    /// Reconstructs a 2D array from the decomposition into 2x2 Givens rotations.
    ///
    /// # Arguments
    ///
    /// * `rotations` - the vector of Givens rotations in the form (real cosine, complex sine, i, j)
    /// * `phases` - the vector of complex diagonal phase factors
    ///
    /// # Returns
    ///
    /// The reconstructed 2D unitary matrix.
    fn reconstruct_from_decomposition(
        rotations: &[(f64, Complex64, usize, usize)],
        phases: &[Complex64],
    ) -> Array2<Complex64> {
        let mut reconstructed = Array2::from_diag(&arr1(phases));

        // Apply rotations in reverse order
        rotations.iter().rev().for_each(|(c, s, i, j)| {
            zrot(
                &mut reconstructed,
                &(*c, s.conj(), *j, *i), // Note: reversed indices
                SliceType::Column,
            );
        });

        reconstructed
    }

    #[test]
    fn test_givens_decomposition() {
        for dim in 1..=10 {
            let mat = random_unitary(dim);
            let (rotations, phases) = givens_decomposition(mat.clone());
            let reconstructed = reconstruct_from_decomposition(&rotations, &phases);

            assert!(
                matrices_approx_equal(&mat, &reconstructed, 1e-8),
                "Givens decomposition and reconstruction failed for dimension {}",
                dim
            );
        }
    }

    /// Reconstructs the `m x n` occupied-orbital coefficients from a Slater decomposition.
    ///
    /// Starting from the reference configuration (the first `m` of `n` orbitals occupied, i.e. the
    /// `m x n` truncated identity), the rotations are applied in order to the columns. This is the
    /// forward of the elimination performed by [`givens_decomposition_slater`].
    fn reconstruct_slater(rotations: &[GivensRotation], m: usize, n: usize) -> Array2<Complex64> {
        let mut reconstructed = Array2::eye(n);
        reconstructed = reconstructed.slice(s![..m, ..]).to_owned();

        for &(c, s, i, j) in rotations {
            for row in 0..m {
                let col_i = reconstructed[[row, i]];
                let col_j = reconstructed[[row, j]];
                reconstructed[[row, j]] = c * col_j - s * col_i;
                reconstructed[[row, i]] = c * col_i + s.conj() * col_j;
            }
        }

        reconstructed
    }

    /// Squared overlap `|det(A B^dagger)|^2` between two sets of occupied orbitals `A` and `B`.
    ///
    /// Both are `m x n` matrices with orthonormal rows; the overlap is 1 iff they span the same
    /// occupied space (i.e. define the same Slater determinant up to a global phase).
    fn slater_fidelity(a: &Array2<Complex64>, b: &Array2<Complex64>) -> f64 {
        let m = a.nrows();
        // gram = A * B^dagger, an m x m matrix
        let gram = DMatrix::from_fn(m, m, |i, j| {
            (0..a.ncols())
                .map(|k| a[[i, k]] * b[[j, k]].conj())
                .sum::<Complex64>()
        });
        gram.determinant().norm().powi(2)
    }

    #[test]
    fn test_givens_decomposition_slater() {
        for (n, m) in [(6, 3), (7, 2), (5, 4), (4, 4), (5, 1), (8, 4)] {
            // draw a random unitary and take m of its columns (as rows) for the occupied orbitals
            let unitary = random_unitary(n);
            let target = unitary.t().slice(s![..m, ..]).to_owned();

            let rotations = givens_decomposition_slater(target.clone());

            assert!(
                rotations.len() <= m * (n - m),
                "Slater decomposition for (n={n}, m={m}) used {} rotations, exceeding the m(n-m)={} bound",
                rotations.len(),
                m * (n - m),
            );
            assert!(
                rotations.iter().all(|&(_, _, i, j)| i.abs_diff(j) == 1),
                "Slater decomposition for (n={n}, m={m}) contains a non-adjacent rotation",
            );

            let reconstructed = reconstruct_slater(&rotations, m, n);
            let fidelity = slater_fidelity(&reconstructed, &target);
            assert!(
                (fidelity - 1.0).abs() < 1e-8,
                "Slater decomposition for (n={n}, m={m}) reconstructed with fidelity {fidelity}",
            );
        }
    }
}
