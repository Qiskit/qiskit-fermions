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
use rayon::prelude::*;
use std::mem::MaybeUninit;
use std::sync::{Arc, Mutex};

fn map_action(action: FermionAction, num_qubits: u32) -> *mut qiskit_sys::QkObs {
    let fer_idx = *action.1 as usize;
    let im = if *action.0 { -0.5 } else { 0.5 };
    let mut coeffs: Vec<qiskit_sys::QkComplex64> = vec![
        qiskit_sys::QkComplex64 { re: 0.5, im: 0.0 },
        qiskit_sys::QkComplex64 { re: 0.0, im },
    ];

    let mut bit_terms = Vec::<qiskit_sys::QkBitTerm>::new();
    let mut indices = Vec::<u32>::new();
    for qb_idx in 0..fer_idx {
        bit_terms.push(qiskit_sys::QkBitTerm_QkBitTerm_Z);
        indices.push(qb_idx as u32);
    }
    bit_terms.push(qiskit_sys::QkBitTerm_QkBitTerm_X);
    indices.push(fer_idx as u32);
    for qb_idx in 0..fer_idx {
        bit_terms.push(qiskit_sys::QkBitTerm_QkBitTerm_Z);
        indices.push(qb_idx as u32);
    }
    bit_terms.push(qiskit_sys::QkBitTerm_QkBitTerm_Y);
    indices.push(fer_idx as u32);

    let mut boundaries: Vec<usize> = vec![0, fer_idx + 1, 2 * fer_idx + 2];

    unsafe {
        qiskit_sys::qk_obs_new(
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
    ptr: *mut qiskit_sys::QkObs,
}
unsafe impl Send for Wrapper {}

// TODO: can we clean up the coding pattern of overwriting a data structure in-place to avoid the
// repetitive re-allocations?
pub fn jordan_wigner(fer_op: &FermionOperator, num_qubits: u32) -> *mut qiskit_sys::QkObs {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(0)
        .build()
        .unwrap();

    let mut qubit_ops = vec![];
    for _ in 0..pool.current_num_threads() {
        qubit_ops.push(Arc::new(Mutex::new(Wrapper {
            ptr: unsafe { qiskit_sys::qk_obs_zero(num_qubits) },
        })));
    }

    pool.install(|| {
        fer_op.iter().par_bridge().for_each(|term| {
            let qk_coeff = qiskit_sys::QkComplex64 {
                re: term.coeff.re,
                im: term.coeff.im,
            };

            let mut mapped_term = unsafe { qiskit_sys::qk_obs_identity(num_qubits) };

            term.iter().for_each(|action| {
                let mapped_action = map_action(action, num_qubits);
                let new_term = unsafe { qiskit_sys::qk_obs_compose(mapped_action, mapped_term) };
                unsafe { qiskit_sys::qk_obs_free(mapped_action) };
                unsafe { qiskit_sys::qk_obs_free(mapped_term) };
                mapped_term = new_term;
            });

            let scaled_term = unsafe { qiskit_sys::qk_obs_multiply(mapped_term, &qk_coeff) };
            unsafe { qiskit_sys::qk_obs_free(mapped_term) };

            let canon_term = unsafe { qiskit_sys::qk_obs_canonicalize(scaled_term, 1e-18) };
            unsafe { qiskit_sys::qk_obs_free(scaled_term) };

            let qubit_op = qubit_ops[pool.current_thread_index().unwrap()]
                // this should never lock because we have one item per thread
                .lock()
                .unwrap();

            // PERF: we are addinf the terms one-by-one manually since this is significantly more
            // efficient that many repetitive calls to qk_obs_add. In-place addition support within
            // Qiskit would alleviate the need for this.
            let num_add_terms = unsafe { qiskit_sys::qk_obs_num_terms(canon_term) };
            let mut term = MaybeUninit::uninit();
            (0..num_add_terms).for_each(|j| unsafe {
                qiskit_sys::qk_obs_term(canon_term, j as u64, term.as_mut_ptr());
                qiskit_sys::qk_obs_add_term(qubit_op.ptr, term.as_ptr());
            });

            unsafe { qiskit_sys::qk_obs_free(canon_term) };
        });
    });

    let mapped_operator: Wrapper = qubit_ops
        .par_iter()
        .fold(
            || Wrapper {
                ptr: unsafe { qiskit_sys::qk_obs_zero(num_qubits) },
            },
            {
                |op1: Wrapper, op2| {
                    let op_locked = op2.lock().unwrap();

                    let num_add_terms = unsafe { qiskit_sys::qk_obs_num_terms(op_locked.ptr) };
                    let mut term = MaybeUninit::uninit();
                    (0..num_add_terms).for_each(|j| unsafe {
                        qiskit_sys::qk_obs_term(op_locked.ptr, j as u64, term.as_mut_ptr());
                        qiskit_sys::qk_obs_add_term(op1.ptr, term.as_ptr());
                    });

                    unsafe { qiskit_sys::qk_obs_free(op_locked.ptr) };

                    op1
                }
            },
        )
        .reduce(
            || Wrapper {
                ptr: unsafe { qiskit_sys::qk_obs_zero(num_qubits) },
            },
            {
                |op1, op2| {
                    let num_add_terms1 = unsafe { qiskit_sys::qk_obs_num_terms(op1.ptr) };
                    let num_add_terms2 = unsafe { qiskit_sys::qk_obs_num_terms(op2.ptr) };
                    if num_add_terms1 > num_add_terms2 {
                        let mut term = MaybeUninit::uninit();
                        (0..num_add_terms2).for_each(|j| unsafe {
                            qiskit_sys::qk_obs_term(op2.ptr, j as u64, term.as_mut_ptr());
                            qiskit_sys::qk_obs_add_term(op1.ptr, term.as_ptr());
                        });
                        unsafe { qiskit_sys::qk_obs_free(op2.ptr) };
                        op1
                    } else {
                        let mut term = MaybeUninit::uninit();
                        (0..num_add_terms1).for_each(|j| unsafe {
                            qiskit_sys::qk_obs_term(op1.ptr, j as u64, term.as_mut_ptr());
                            qiskit_sys::qk_obs_add_term(op2.ptr, term.as_ptr());
                        });
                        unsafe { qiskit_sys::qk_obs_free(op1.ptr) };
                        op2
                    }
                }
            },
        );

    mapped_operator.ptr
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
            indices: vec![
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
        };
        let qb_op = jordan_wigner(&fer_op, 4);

        let mut coeffs: Vec<qiskit_sys::QkComplex64> = vec![
            qiskit_sys::QkComplex64 {
                re: -0.8105479805373261,
                im: 0.0,
            },
            qiskit_sys::QkComplex64 {
                re: -0.22575349222402477,
                im: 0.0,
            },
            qiskit_sys::QkComplex64 {
                re: 0.17218393261915543,
                im: 0.0,
            },
            qiskit_sys::QkComplex64 {
                re: 0.12091263261776633,
                im: 0.0,
            },
            qiskit_sys::QkComplex64 {
                re: 0.17218393261915554,
                im: 0.0,
            },
            qiskit_sys::QkComplex64 {
                re: 0.16892753870087912,
                im: 0.0,
            },
            qiskit_sys::QkComplex64 {
                re: -0.22575349222402477,
                im: 0.0,
            },
            qiskit_sys::QkComplex64 {
                re: 0.16614543256382416,
                im: 0.0,
            },
            qiskit_sys::QkComplex64 {
                re: 0.04523279994605783,
                im: 0.0,
            },
            qiskit_sys::QkComplex64 {
                re: 0.04523279994605783,
                im: 0.0,
            },
            qiskit_sys::QkComplex64 {
                re: 0.04523279994605783,
                im: 0.0,
            },
            qiskit_sys::QkComplex64 {
                re: 0.04523279994605783,
                im: 0.0,
            },
            qiskit_sys::QkComplex64 {
                re: 0.16614543256382416,
                im: 0.0,
            },
            qiskit_sys::QkComplex64 {
                re: 0.17464343068300459,
                im: 0.0,
            },
            qiskit_sys::QkComplex64 {
                re: 0.12091263261776633,
                im: 0.0,
            },
        ];

        let mut bit_terms: Vec<qiskit_sys::QkBitTerm> = vec![
            qiskit_sys::QkBitTerm_QkBitTerm_Z,
            qiskit_sys::QkBitTerm_QkBitTerm_Z,
            qiskit_sys::QkBitTerm_QkBitTerm_Z,
            qiskit_sys::QkBitTerm_QkBitTerm_Z,
            qiskit_sys::QkBitTerm_QkBitTerm_Z,
            qiskit_sys::QkBitTerm_QkBitTerm_Z,
            qiskit_sys::QkBitTerm_QkBitTerm_Z,
            qiskit_sys::QkBitTerm_QkBitTerm_Z,
            qiskit_sys::QkBitTerm_QkBitTerm_Z,
            qiskit_sys::QkBitTerm_QkBitTerm_Z,
            qiskit_sys::QkBitTerm_QkBitTerm_Y,
            qiskit_sys::QkBitTerm_QkBitTerm_Y,
            qiskit_sys::QkBitTerm_QkBitTerm_Y,
            qiskit_sys::QkBitTerm_QkBitTerm_Y,
            qiskit_sys::QkBitTerm_QkBitTerm_Y,
            qiskit_sys::QkBitTerm_QkBitTerm_Y,
            qiskit_sys::QkBitTerm_QkBitTerm_X,
            qiskit_sys::QkBitTerm_QkBitTerm_X,
            qiskit_sys::QkBitTerm_QkBitTerm_X,
            qiskit_sys::QkBitTerm_QkBitTerm_X,
            qiskit_sys::QkBitTerm_QkBitTerm_Y,
            qiskit_sys::QkBitTerm_QkBitTerm_Y,
            qiskit_sys::QkBitTerm_QkBitTerm_X,
            qiskit_sys::QkBitTerm_QkBitTerm_X,
            qiskit_sys::QkBitTerm_QkBitTerm_X,
            qiskit_sys::QkBitTerm_QkBitTerm_X,
            qiskit_sys::QkBitTerm_QkBitTerm_Z,
            qiskit_sys::QkBitTerm_QkBitTerm_Z,
            qiskit_sys::QkBitTerm_QkBitTerm_Z,
            qiskit_sys::QkBitTerm_QkBitTerm_Z,
            qiskit_sys::QkBitTerm_QkBitTerm_Z,
            qiskit_sys::QkBitTerm_QkBitTerm_Z,
        ];

        let mut indices: Vec<u32> = vec![
            1, 0, 0, 1, 2, 0, 2, 3, 0, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 1, 2, 1,
            3, 2, 3,
        ];

        let mut boundaries: Vec<usize> =
            vec![0, 0, 1, 2, 4, 5, 7, 8, 10, 14, 18, 22, 26, 28, 30, 32];

        let mut expected = unsafe {
            qiskit_sys::qk_obs_new(
                4,
                coeffs.len().try_into().unwrap(),
                bit_terms.len().try_into().unwrap(),
                coeffs.as_mut_ptr(),
                bit_terms.as_mut_ptr(),
                indices.as_mut_ptr(),
                boundaries.as_mut_ptr(),
            )
        };

        let factor = qiskit_sys::QkComplex64 { re: -1.0, im: 0.0 };
        expected = unsafe { qiskit_sys::qk_obs_multiply(expected, &factor) };

        let mut diff = unsafe { qiskit_sys::qk_obs_add(qb_op, expected) };

        diff = unsafe { qiskit_sys::qk_obs_canonicalize(diff, 1e-6) };

        let zero = unsafe { qiskit_sys::qk_obs_zero(4) };

        let equal = unsafe { qiskit_sys::qk_obs_equal(diff, zero) };

        assert!(equal)
    }
}
