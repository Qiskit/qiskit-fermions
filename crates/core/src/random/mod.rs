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

use nalgebra::DMatrix;
use ndarray::{Array1, Array2};
use num_complex::Complex64;

/// The algorithm implemented here is described in https://arxiv.org/abs/math-ph/0609050.
pub fn random_unitary(dim: usize) -> Array2<Complex64> {
    // Create random complex matrix
    let z_real = DMatrix::new_random(dim, dim);
    let z_imag = DMatrix::new_random(dim, dim);
    let z = DMatrix::from_fn(dim, dim, |i, j| {
        Complex64::new(z_real[(i, j)], z_imag[(i, j)])
    });

    // QR decomposition
    let qr = z.qr();

    // Convert nalgebra to ndarray
    let q_ndarray = Array2::from_shape_vec((dim, dim), qr.q().data.as_vec().clone()).unwrap();

    // Extract diagonal phases
    let phase = Array1::from_iter(qr.r().diagonal().iter().map(|x| x / x.norm()));

    // A truly Haar measure random Unitary needs to be scaled by the diagonal phases
    q_ndarray * phase
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::testing::matrices_approx_equal;

    #[test]
    fn test_random_unitary() {
        (1..=10).for_each(|dim| {
            let unitary = DMatrix::from_iterator(dim, dim, random_unitary(dim).iter().cloned());
            let approx_eye = Array2::from_shape_vec(
                (dim, dim),
                (unitary.adjoint() * unitary).data.as_vec().clone(),
            )
            .unwrap();
            let eye = Array2::eye(dim);
            assert!(
                matrices_approx_equal(&approx_eye, &eye, 1e-8),
                "The produced random matrix of dimension {}x{} was not unitary!",
                dim,
                dim,
            );
        });
    }
}
