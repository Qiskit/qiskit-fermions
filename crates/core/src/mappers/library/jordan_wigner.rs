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

use crate::operators::fermion_operator::{FermionAction, FermionOperator};
use crate::operators::{CoherenceError, OperatorTrait};
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

cfg_select! {
    feature = "pyext" => {
        use num_complex::Complex64 as QkComplex64;
        extern crate qiskit_pyo3_ffi as ffi;
        use ffi::QkBitTerm::X as QkBitTermX;
        use ffi::QkBitTerm::Y as QkBitTermY;
        use ffi::QkBitTerm::Z as QkBitTermZ;
    }
    feature = "cext" => {
        extern crate qiskit_sys as ffi;
        use ffi::QkComplex64 as QkComplex64;
        use ffi::QkBitTerm_QkBitTerm_X as QkBitTermX;
        use ffi::QkBitTerm_QkBitTerm_Y as QkBitTermY;
        use ffi::QkBitTerm_QkBitTerm_Z as QkBitTermZ;
    }
}

fn map_action(action: FermionAction, num_qubits: u32) -> *mut ffi::QkObs {
    let fer_idx = *action.1 as usize;
    let im = if *action.0 { -0.5 } else { 0.5 };
    let mut coeffs: Vec<QkComplex64> = vec![
        QkComplex64 { re: 0.5, im: 0.0 },
        QkComplex64 { re: 0.0, im },
    ];

    let mut bit_terms = Vec::<ffi::QkBitTerm>::new();
    let mut indices = Vec::<u32>::new();
    for qb_idx in 0..fer_idx {
        bit_terms.push(QkBitTermZ);
        indices.push(qb_idx as u32);
    }
    bit_terms.push(QkBitTermX);
    indices.push(fer_idx as u32);
    for qb_idx in 0..fer_idx {
        bit_terms.push(QkBitTermZ);
        indices.push(qb_idx as u32);
    }
    bit_terms.push(QkBitTermY);
    indices.push(fer_idx as u32);

    let mut boundaries: Vec<usize> = vec![0, fer_idx + 1, 2 * fer_idx + 2];

    unsafe {
        ffi::qk_obs_new(
            num_qubits,
            coeffs.len().try_into().unwrap(),
            bit_terms.len().try_into().unwrap(),
            coeffs.as_mut_ptr(),
            bit_terms.as_mut_ptr(),
            indices.as_mut_ptr(),
            boundaries.as_mut_ptr(),
        )
    }
}

// NOTE: https://stackoverflow.com/a/50341075
struct Wrapper {
    ptr: *mut ffi::QkObs,
}
unsafe impl Send for Wrapper {}

// TODO: can we clean up the coding pattern of overwriting a data structure in-place to avoid the
// repetitive re-allocations?
pub fn jordan_wigner(
    fer_op: &FermionOperator,
    num_qubits: u32,
) -> Result<*mut ffi::QkObs, CoherenceError> {
    // Each mode index `j` maps onto qubit `j`, so the operator's largest mode index must fit
    // within `num_qubits`. Without this check, the underlying `qk_obs_*` calls receive an
    // out-of-range qubit index and abort the process with a non-unwinding panic.
    if let Some(&max_mode) = fer_op.modes.iter().max()
        && max_mode >= num_qubits
    {
        return Err(CoherenceError::NumQubitsTooSmall {
            num_qubits,
            max_mode,
        });
    }

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(0)
        .build()
        .unwrap();

    let mut qubit_ops = vec![];
    for _ in 0..pool.current_num_threads() {
        qubit_ops.push(Arc::new(Mutex::new(Wrapper {
            ptr: unsafe { ffi::qk_obs_zero(num_qubits) },
        })));
    }

    pool.install(|| {
        fer_op.iter().par_bridge().for_each(|term| {
            let qk_coeff = QkComplex64 {
                re: term.coeff.re,
                im: term.coeff.im,
            };

            let mut mapped_term = unsafe { ffi::qk_obs_identity(num_qubits) };

            term.iter().for_each(|action| {
                let mapped_action = map_action(action, num_qubits);
                let new_term = unsafe { ffi::qk_obs_compose(mapped_action, mapped_term) };
                unsafe { ffi::qk_obs_free(mapped_action) };
                unsafe { ffi::qk_obs_free(mapped_term) };
                mapped_term = new_term;
            });

            let canon_term = unsafe { ffi::qk_obs_canonicalize(mapped_term, 1e-18) };
            unsafe { ffi::qk_obs_free(mapped_term) };

            let qubit_op = qubit_ops[pool.current_thread_index().unwrap()]
                // this should never lock because we have one item per thread
                .lock()
                .unwrap();

            unsafe { ffi::qk_obs_scaled_add_inplace(qubit_op.ptr, canon_term, &qk_coeff) };

            unsafe { ffi::qk_obs_free(canon_term) };
        });
    });

    let mapped_operator: Wrapper = qubit_ops
        .par_iter()
        .fold(
            || Wrapper {
                ptr: unsafe { ffi::qk_obs_zero(num_qubits) },
            },
            {
                |op1: Wrapper, op2| {
                    let op_locked = op2.lock().unwrap();
                    unsafe { ffi::qk_obs_add_inplace(op1.ptr, op_locked.ptr) };
                    unsafe { ffi::qk_obs_free(op_locked.ptr) };
                    op1
                }
            },
        )
        .reduce(
            || Wrapper {
                ptr: unsafe { ffi::qk_obs_zero(num_qubits) },
            },
            {
                |op1, op2| {
                    let num_add_terms1 = unsafe { ffi::qk_obs_num_terms(op1.ptr) };
                    let num_add_terms2 = unsafe { ffi::qk_obs_num_terms(op2.ptr) };
                    if num_add_terms1 > num_add_terms2 {
                        unsafe { ffi::qk_obs_add_inplace(op1.ptr, op2.ptr) };
                        unsafe { ffi::qk_obs_free(op2.ptr) };
                        op1
                    } else {
                        unsafe { ffi::qk_obs_add_inplace(op2.ptr, op1.ptr) };
                        unsafe { ffi::qk_obs_free(op1.ptr) };
                        op2
                    }
                }
            },
        );

    Ok(mapped_operator.ptr)
}

#[cfg(test)]
mod tests {
    use super::*;

    use num_complex::Complex64;

    #[test]
    fn test_jordan_wigner() {
        let fer_op = FermionOperator {
            coeffs: vec![
                -1.2563390730032502,
                -1.2563390730032502,
                -2.3575299028703285e-16,
                -2.3575299028703285e-16,
                -2.3575299028703285e-16,
                -2.3575299028703285e-16,
                -0.4718960072811406,
                -0.4718960072811406,
                0.33785507740175824,
                0.33785507740175824,
                0.33785507740175824,
                0.33785507740175824,
                0.09046559989211567,
                0.09046559989211567,
                0.09046559989211567,
                0.09046559989211567,
                0.09046559989211567,
                0.09046559989211567,
                0.09046559989211567,
                0.09046559989211567,
                0.09046559989211567,
                0.09046559989211567,
                0.09046559989211567,
                0.09046559989211567,
                0.09046559989211567,
                0.09046559989211567,
                0.09046559989211567,
                0.09046559989211567,
                0.3322908651276483,
                0.3322908651276483,
                0.3322908651276483,
                0.3322908651276483,
                0.3322908651276483,
                0.3322908651276483,
                0.3322908651276483,
                0.3322908651276483,
                0.34928686136600917,
                0.34928686136600917,
                0.34928686136600917,
                0.34928686136600917,
            ]
            .iter()
            .map(|c| Complex64::new(*c, 0.0))
            .collect(),
            actions: vec![
                true, false, true, false, true, false, true, false, true, false, true, false, true,
                false, true, false, true, true, false, false, true, true, false, false, true, true,
                false, false, true, true, false, false, true, true, false, false, true, true,
                false, false, true, true, false, false, true, true, false, false, true, true,
                false, false, true, true, false, false, true, true, false, false, true, true,
                false, false, true, true, false, false, true, true, false, false, true, true,
                false, false, true, true, false, false, true, true, false, false, true, true,
                false, false, true, true, false, false, true, true, false, false, true, true,
                false, false, true, true, false, false, true, true, false, false, true, true,
                false, false, true, true, false, false, true, true, false, false, true, true,
                false, false, true, true, false, false, true, true, false, false, true, true,
                false, false, true, true, false, false, true, true, false, false,
            ],
            modes: vec![
                0, 0, 2, 2, 1, 0, 0, 1, 3, 2, 2, 3, 1, 1, 3, 3, 0, 0, 0, 0, 2, 0, 0, 2, 0, 2, 2, 0,
                2, 2, 2, 2, 1, 1, 0, 0, 3, 1, 0, 2, 1, 3, 2, 0, 3, 3, 2, 2, 0, 1, 0, 1, 2, 1, 0, 3,
                0, 3, 2, 1, 2, 3, 2, 3, 1, 0, 1, 0, 3, 0, 1, 2, 1, 2, 3, 0, 3, 2, 3, 2, 0, 0, 1, 1,
                2, 0, 1, 3, 0, 2, 3, 1, 2, 2, 3, 3, 1, 0, 0, 1, 3, 0, 0, 3, 1, 2, 2, 1, 3, 2, 2, 3,
                0, 1, 1, 0, 2, 1, 1, 2, 0, 3, 3, 0, 2, 3, 3, 2, 1, 1, 1, 1, 3, 1, 1, 3, 1, 3, 3, 1,
                3, 3, 3, 3,
            ],
            boundaries: vec![
                0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68,
                72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140,
                144,
            ],
            groups: None,
        };
        let qb_op = jordan_wigner(&fer_op, 4).unwrap();

        let mut coeffs: Vec<QkComplex64> = vec![
            QkComplex64 {
                re: -0.8105479805373261,
                im: 0.0,
            },
            QkComplex64 {
                re: -0.22575349222402477,
                im: 0.0,
            },
            QkComplex64 {
                re: 0.17218393261915543,
                im: 0.0,
            },
            QkComplex64 {
                re: 0.12091263261776633,
                im: 0.0,
            },
            QkComplex64 {
                re: 0.17218393261915554,
                im: 0.0,
            },
            QkComplex64 {
                re: 0.16892753870087912,
                im: 0.0,
            },
            QkComplex64 {
                re: -0.22575349222402477,
                im: 0.0,
            },
            QkComplex64 {
                re: 0.16614543256382416,
                im: 0.0,
            },
            QkComplex64 {
                re: 0.04523279994605783,
                im: 0.0,
            },
            QkComplex64 {
                re: 0.04523279994605783,
                im: 0.0,
            },
            QkComplex64 {
                re: 0.04523279994605783,
                im: 0.0,
            },
            QkComplex64 {
                re: 0.04523279994605783,
                im: 0.0,
            },
            QkComplex64 {
                re: 0.16614543256382416,
                im: 0.0,
            },
            QkComplex64 {
                re: 0.17464343068300459,
                im: 0.0,
            },
            QkComplex64 {
                re: 0.12091263261776633,
                im: 0.0,
            },
        ];

        let mut bit_terms: Vec<ffi::QkBitTerm> = vec![
            QkBitTermZ, QkBitTermZ, QkBitTermZ, QkBitTermZ, QkBitTermZ, QkBitTermZ, QkBitTermZ,
            QkBitTermZ, QkBitTermZ, QkBitTermZ, QkBitTermY, QkBitTermY, QkBitTermY, QkBitTermY,
            QkBitTermY, QkBitTermY, QkBitTermX, QkBitTermX, QkBitTermX, QkBitTermX, QkBitTermY,
            QkBitTermY, QkBitTermX, QkBitTermX, QkBitTermX, QkBitTermX, QkBitTermZ, QkBitTermZ,
            QkBitTermZ, QkBitTermZ, QkBitTermZ, QkBitTermZ,
        ];

        let mut indices: Vec<u32> = vec![
            1, 0, 0, 1, 2, 0, 2, 3, 0, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 1, 2, 1,
            3, 2, 3,
        ];

        let mut boundaries: Vec<usize> =
            vec![0, 0, 1, 2, 4, 5, 7, 8, 10, 14, 18, 22, 26, 28, 30, 32];

        let mut expected = unsafe {
            ffi::qk_obs_new(
                4,
                coeffs.len().try_into().unwrap(),
                bit_terms.len().try_into().unwrap(),
                coeffs.as_mut_ptr(),
                bit_terms.as_mut_ptr(),
                indices.as_mut_ptr(),
                boundaries.as_mut_ptr(),
            )
        };

        let factor = QkComplex64 { re: -1.0, im: 0.0 };
        expected = unsafe { ffi::qk_obs_multiply(expected, &factor) };

        let mut diff = unsafe { ffi::qk_obs_add(qb_op, expected) };

        diff = unsafe { ffi::qk_obs_canonicalize(diff, 1e-6) };

        let zero = unsafe { ffi::qk_obs_zero(4) };

        let equal = unsafe { ffi::qk_obs_equal(diff, zero) };

        assert!(equal)
    }

    #[test]
    fn test_jordan_wigner_num_qubits_too_small() {
        // an operator acting on mode index 3 requires at least 4 qubits
        let fer_op = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            actions: vec![true],
            modes: vec![3],
            boundaries: vec![0, 1],
            groups: None,
        };

        // too few qubits must be reported instead of aborting the process
        let err = jordan_wigner(&fer_op, 3).unwrap_err();
        assert!(matches!(
            err,
            CoherenceError::NumQubitsTooSmall {
                num_qubits: 3,
                max_mode: 3
            }
        ));

        // exactly enough qubits succeeds
        assert!(jordan_wigner(&fer_op, 4).is_ok());
    }
}
