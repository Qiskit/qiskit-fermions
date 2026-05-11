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

#[cfg(test)]
mod tests {
    use super::*;

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
}
