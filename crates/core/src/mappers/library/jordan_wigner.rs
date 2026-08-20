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

/// Tolerance used for every `qk_obs_canonicalize` call in this module.
///
/// This is deliberately far below any physically meaningful coefficient: canonicalization is used
/// here only to *merge* duplicate Pauli terms, never to discard physics. Raising it would silently
/// change the mapped operator.
const CANONICALIZE_TOL: f64 = 1e-18;

/// Weight a single bit term carries in [`Wrapper::cost`]: a `QkBitTerm` plus its `u32` qubit index.
const WEIGHT_PER_BIT_TERM: usize = size_of::<u32>() + 1;

/// Weight a single Pauli term carries on top of its bit terms: a `Complex64` coefficient plus the
/// `usize` boundary delimiting it.
///
/// Counted separately from [`WEIGHT_PER_BIT_TERM`] because it is charged per *term* rather than per
/// bit term, and the two cannot be folded into one figure without assuming an average Pauli weight.
/// It also dominates for low-weight terms -- and an identity term has no bit terms at all, so a
/// measure built on bit terms alone would score it as free. That this weight is *non-zero* is the
/// load-bearing property; the particular figures only set the relative cost of a term against a bit
/// term.
const WEIGHT_PER_TERM: usize = size_of::<f64>() * 2 + size_of::<usize>();

/// Smallest accumulator size worth compacting at all, in the units of [`Wrapper::cost`].
///
/// Below this the duplication cannot amount to enough memory to be worth a merge pass, so small
/// operators are mapped without ever being compacted.
const MIN_COMPACTION_FLOOR: usize = (1 << 16) * WEIGHT_PER_BIT_TERM;

/// How far an accumulator may grow past its last merged size before it is merged again.
///
/// This is what bounds the accumulators: a merge leaves behind exactly the distinct content, so the
/// next merge is triggered at this multiple of it and the peak is `factor * distinct` -- proportional
/// to the operator being built, whatever the duplication happens to be.
///
/// The value trades merging work against that peak, and nothing more. It cannot run away: the trigger
/// is recomputed from the *measured* post-merge size every cycle, so it tracks the distinct content
/// rather than compounding against it. An earlier version varied the factor at runtime, from the
/// reduction each merge achieved, on the theory that a fixed one would let the post-merge size creep
/// upwards from cycle to cycle. That cannot happen for the reason just given, and the adaptive factor
/// measurably spent 93% of its time pinned to the bottom of its own clamp range, so it was replaced
/// by the constant it was effectively computing.
const GROWTH_FACTOR: f64 = 1.5;

/// Canonicalizes `obs` in place, freeing the original, to merge duplicate Pauli terms.
///
/// Takes ownership of the pointer it is given and returns a new one; the input must not be used
/// afterwards.
///
/// # Safety
///
/// `obs` must be a valid, uniquely-owned `QkObs` pointer.
unsafe fn compact(obs: *mut ffi::QkObs) -> *mut ffi::QkObs {
    let compacted = unsafe { ffi::qk_obs_canonicalize(obs, CANONICALIZE_TOL) };
    unsafe { ffi::qk_obs_free(obs) };
    compacted
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
    /// [`Wrapper::cost`] of this accumulator just after it was last compacted, or 0 if it never was.
    ///
    /// Compaction is triggered by growth relative to this rather than by an absolute size, so that
    /// an accumulator which is already mostly distinct terms is not compacted repeatedly to no
    /// effect.
    compacted_cost: usize,
}
unsafe impl Send for Wrapper {}

impl Wrapper {
    /// Creates a new, empty accumulator.
    fn zero(num_qubits: u32) -> Self {
        Self {
            ptr: unsafe { ffi::qk_obs_zero(num_qubits) },
            compacted_cost: 0,
        }
    }

    /// Returns a weighted size for this accumulator's term buffers.
    ///
    /// Both triggers below are driven by this rather than by `qk_obs_len` alone. A term carries its
    /// coefficient and boundary on top of its bit terms, and that part is charged per term: an
    /// *identity* term has no bit terms whatsoever, so a measure built on `qk_obs_len` alone scores
    /// it as free and neither trigger ever fires however many of them pile up. Operators whose
    /// mapped image is identity-dominated -- a constant offset repeated, say -- would then grow
    /// unbounded, which is exactly the regression the compaction exists to prevent.
    fn cost(&self) -> usize {
        let num_bit_terms = unsafe { ffi::qk_obs_len(self.ptr) } as usize;
        let num_terms = unsafe { ffi::qk_obs_num_terms(self.ptr) } as usize;
        num_bit_terms * WEIGHT_PER_BIT_TERM + num_terms * WEIGHT_PER_TERM
    }

    /// Merges duplicate terms once the accumulator has grown enough past its last merged size to be
    /// worth the work.
    fn compact_if_grown(&mut self) {
        let cost = self.cost();

        let grown = (self.compacted_cost as f64 * GROWTH_FACTOR) as usize;
        if cost <= MIN_COMPACTION_FLOOR.max(grown) {
            return;
        }

        self.ptr = unsafe { compact(self.ptr) };
        self.compacted_cost = self.cost();
    }

    /// Merges duplicate terms unconditionally.
    ///
    /// Used for the result that is handed back, so that its size does not depend on whether the last
    /// combine happened to trip the growth trigger.
    fn compact_now(&mut self) {
        let cost = self.cost();

        // Nothing has been added since the last compaction, so there is provably nothing left to
        // merge. Canonicalizing is not free -- it builds a map over every term and allocates a whole
        // new observable -- and this operator is the largest one the mapper handles, so the check is
        // well worth making. Callers who want the result fully simplified do that themselves.
        //
        // This is a sound "nothing was added" proof only because every append raises the cost:
        // `WEIGHT_PER_TERM` is non-zero, so even appending a bit-term-free identity term moves it.
        // Comparing `qk_obs_len` alone would not do, since that is blind to such an append.
        if cost == self.compacted_cost {
            return;
        }

        self.ptr = unsafe { compact(self.ptr) };
        self.compacted_cost = self.cost();
    }
}

pub fn fermion_jordan_wigner(
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
        qubit_ops.push(Arc::new(Mutex::new(Wrapper::zero(num_qubits))));
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

            let canon_term = unsafe { compact(mapped_term) };

            // NOTE: `mut` so that the accumulator can be swapped for its compacted self below.
            // This relies on there being exactly one accumulator per worker thread, which is what
            // makes the `lock()` uncontended: `par_bridge` only ever runs this closure on a pool
            // worker, so `current_thread_index()` is both `Some` and unique per concurrent call.
            let mut qubit_op = qubit_ops[pool.current_thread_index().unwrap()]
                // this should never lock because we have one item per thread
                .lock()
                .unwrap();

            unsafe { ffi::qk_obs_scaled_add_inplace(qubit_op.ptr, canon_term, &qk_coeff) };

            unsafe { ffi::qk_obs_free(canon_term) };

            // Merge the duplicates accumulated so far. Without this the accumulator grows with the
            // number of emitted terms instead of the number of distinct ones.
            qubit_op.compact_if_grown();
        });
    });

    let mapped_operator: Wrapper = qubit_ops
        .par_iter()
        .fold(|| Wrapper::zero(num_qubits), {
            |mut op1: Wrapper, op2| {
                let op_locked = op2.lock().unwrap();
                unsafe { ffi::qk_obs_add_inplace(op1.ptr, op_locked.ptr) };
                unsafe { ffi::qk_obs_free(op_locked.ptr) };
                // Compact as we go: `qk_obs_add_inplace` concatenates, so folding the
                // per-thread accumulators would otherwise hold their combined size at once.
                op1.compact_if_grown();
                op1
            }
        })
        .reduce(|| Wrapper::zero(num_qubits), {
            |op1, op2| {
                let num_add_terms1 = unsafe { ffi::qk_obs_num_terms(op1.ptr) } as usize;
                let num_add_terms2 = unsafe { ffi::qk_obs_num_terms(op2.ptr) } as usize;
                // Add into whichever side is larger so the bigger buffer is never copied, then
                // merge the duplicates the concatenation introduced.
                let mut acc = if num_add_terms1 > num_add_terms2 {
                    unsafe { ffi::qk_obs_add_inplace(op1.ptr, op2.ptr) };
                    unsafe { ffi::qk_obs_free(op2.ptr) };
                    op1
                } else {
                    unsafe { ffi::qk_obs_add_inplace(op2.ptr, op1.ptr) };
                    unsafe { ffi::qk_obs_free(op1.ptr) };
                    op2
                };
                acc.compact_if_grown();
                acc
            }
        });

    // Merge the result unconditionally. The combine above stops as soon as the growth trigger is
    // satisfied, which would otherwise leave the size of the returned operator dependent on how the
    // work happened to be split -- the caller would see very different term counts for the same
    // operator from one run to the next.
    let mut mapped_operator = mapped_operator;
    mapped_operator.compact_now();

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
        let qb_op = fermion_jordan_wigner(&fer_op, 4).unwrap();

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
        let err = fermion_jordan_wigner(&fer_op, 3).unwrap_err();
        assert!(matches!(
            err,
            CoherenceError::NumQubitsTooSmall {
                num_qubits: 3,
                max_mode: 3
            }
        ));

        // exactly enough qubits succeeds
        assert!(fermion_jordan_wigner(&fer_op, 4).is_ok());
    }

    /// Appends the weight-1 Pauli `Z_{qubit}` to `acc`, as the mapper's inner loop does.
    fn append_z(acc: &mut Wrapper, qubit: u32) {
        let mut coeffs: Vec<QkComplex64> = vec![QkComplex64 { re: 1.0, im: 0.0 }];
        let mut bit_terms: Vec<ffi::QkBitTerm> = vec![QkBitTermZ];
        let mut indices: Vec<u32> = vec![qubit];
        let mut boundaries: Vec<usize> = vec![0, 1];
        let one = QkComplex64 { re: 1.0, im: 0.0 };
        unsafe {
            let term = ffi::qk_obs_new(
                NUM_QUBITS,
                1,
                1,
                coeffs.as_mut_ptr(),
                bit_terms.as_mut_ptr(),
                indices.as_mut_ptr(),
                boundaries.as_mut_ptr(),
            );
            ffi::qk_obs_scaled_add_inplace(acc.ptr, term, &one);
            ffi::qk_obs_free(term);
        }
    }

    /// Qubit count used by the accumulator-level tests below.
    const NUM_QUBITS: u32 = 64;

    #[test]
    fn test_accumulator_stays_proportional_to_its_distinct_content() {
        // This is the guarantee that bounds the accumulators: a merge leaves exactly the distinct
        // content behind, and the next merge triggers at `GROWTH_FACTOR` times that, so the peak is a
        // fixed multiple of the distinct content however much duplication passes through.
        //
        // Driven against the accumulator directly rather than through `fermion_jordan_wigner`, whose
        // final unconditional compaction would mask the trigger's behaviour.
        //
        // The workload matters. Appending one Pauli string over and over merges back to a single term
        // every time, which makes the bound trivial. Merging must be only *partly* effective for the
        // proportionality to mean anything, so each cycle appends one string from a bounded distinct
        // set along with a batch of duplicates. The scale is chosen to carry the accumulator well past
        // `MIN_COMPACTION_FLOOR`, below which nothing is compacted at all.
        let mut acc = Wrapper::zero(NUM_QUBITS);

        let num_cycles = 2000;
        let dupes_per_cycle = 64;
        let mut peak = 0;
        let mut num_compactions = 0;
        for cycle in 0..num_cycles {
            append_z(&mut acc, (cycle % NUM_QUBITS as usize) as u32);
            for _ in 0..dupes_per_cycle {
                append_z(&mut acc, 0);
                // Detect a merge by the cost *dropping*: `compacted_cost` settles at the distinct
                // content and then stops changing, so comparing it would miss every merge after the
                // first.
                let before = acc.cost();
                acc.compact_if_grown();
                let after = acc.cost();
                if after < before {
                    num_compactions += 1;
                }
                peak = peak.max(before);
            }
        }

        // The trigger must actually have fired, or the bound below is vacuous.
        assert!(
            num_compactions > 2,
            "growth trigger barely fired ({num_compactions} times), the bound below proves nothing"
        );

        // Only `NUM_QUBITS` distinct strings are ever appended, so every compaction takes the
        // accumulator back to that tiny distinct content and the next one triggers at the floor. The
        // peak therefore stays within the growth factor of the floor, plus one append's slack for the
        // term that trips it.
        let bound = (MIN_COMPACTION_FLOOR as f64 * GROWTH_FACTOR) as usize
            + WEIGHT_PER_TERM
            + WEIGHT_PER_BIT_TERM;
        assert!(
            peak <= bound,
            "accumulator grew beyond its distinct content: peak {peak} exceeds {bound}"
        );

        // ... and the workload must have been large enough for that to constrain anything: an
        // accumulator growing with the number of *appends* would have reached this instead.
        let unmerged = num_cycles * (dupes_per_cycle + 1) * (WEIGHT_PER_TERM + WEIGHT_PER_BIT_TERM);
        assert!(
            unmerged > 2 * bound,
            "workload too small ({unmerged} unmerged vs bound {bound}) to constrain anything"
        );
        unsafe { ffi::qk_obs_free(acc.ptr) };
    }

    #[test]
    fn test_identity_only_terms_are_compacted() {
        // An identity Pauli term has *no* bit terms, so it costs nothing in `qk_obs_len` while still
        // occupying a coefficient and a boundary. Measuring the accumulators in `qk_obs_len` alone
        // therefore left identity-dominated operators growing without bound -- neither trigger could
        // ever fire -- which is precisely the regression compaction exists to prevent.
        //
        // Sized to clear `MIN_COMPACTION_FLOOR` by enough for the growth trigger to fire a handful of
        // times (measured: 7 merges here), which is what makes this exercise the incremental path
        // rather than only the final unconditional one. Going higher only costs time: at `1 << 21`
        // this ran 15x slower for 148 merges instead of 7, and caught nothing extra.
        let num_repeats = 1 << 17;
        let fer_op = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0); num_repeats],
            actions: vec![],
            modes: vec![],
            // Every term is empty, i.e. a bare coefficient, which maps onto the identity.
            boundaries: vec![0; num_repeats + 1],
            groups: None,
        };

        let qb_op = fermion_jordan_wigner(&fer_op, 2).unwrap();
        let num_terms = unsafe { ffi::qk_obs_num_terms(qb_op) } as usize;
        assert!(
            num_terms <= 2,
            "expected identity terms to be merged, got {num_terms} from {num_repeats} copies"
        );

        // ... and the operator must still be `num_repeats * I`.
        let mut coeffs: Vec<QkComplex64> = vec![QkComplex64 {
            re: -(num_repeats as f64),
            im: 0.0,
        }];
        let mut boundaries: Vec<usize> = vec![0, 0];
        // The single term is the identity, so it has no bit terms; the arrays are still real
        // allocations rather than null pointers, since `qk_obs_new` dereferences what it is handed.
        let mut bit_terms: Vec<ffi::QkBitTerm> = vec![QkBitTermZ];
        let mut indices: Vec<u32> = vec![0];
        let negated = unsafe {
            ffi::qk_obs_new(
                2,
                1,
                0,
                coeffs.as_mut_ptr(),
                bit_terms.as_mut_ptr(),
                indices.as_mut_ptr(),
                boundaries.as_mut_ptr(),
            )
        };
        let diff = unsafe { ffi::qk_obs_add(qb_op, negated) };
        let diff = unsafe { ffi::qk_obs_canonicalize(diff, 1e-9) };
        let zero = unsafe { ffi::qk_obs_zero(2) };
        assert!(unsafe { ffi::qk_obs_equal(diff, zero) });
    }

    #[test]
    fn test_jordan_wigner_merges_duplicate_terms() {
        // The accumulators are grown with an addition that concatenates rather than merges, so
        // without periodic canonicalization the result grows with the number of terms mapped even
        // when they all map onto the same Pauli strings.
        //
        // Repeat one number operator many times: every copy maps to the same two Pauli terms, so a
        // compacting mapper returns a handful of terms regardless of how many copies there are.
        //
        // Sized as in `test_identity_only_terms_are_compacted`: far enough past
        // `MIN_COMPACTION_FLOOR` for the growth trigger to fire repeatedly (measured: 11 merges), and
        // no further, since the assertion below is satisfied by orders of magnitude either way.
        let num_repeats = 1 << 17;
        let fer_op = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0); num_repeats],
            actions: [true, false].repeat(num_repeats),
            modes: vec![0; 2 * num_repeats],
            boundaries: (0..=num_repeats).map(|i| 2 * i).collect(),
            groups: None,
        };

        let qb_op = fermion_jordan_wigner(&fer_op, 2).unwrap();
        let num_terms = unsafe { ffi::qk_obs_num_terms(qb_op) } as usize;

        // `a^dagger_0 a_0` maps onto the identity and `Z_0`, so the merged result is tiny. Allow
        // generous slack for whatever is left over between compactions, but assert the count does
        // not scale with `num_repeats` (which is what regressed): the growth rule keeps the
        // accumulators near the compaction floor, which is far below the number of copies here.
        assert!(
            num_terms < num_repeats / 8,
            "expected duplicates to be merged, got {num_terms} terms from {num_repeats} copies"
        );

        // The operator must still be `num_repeats * (I - Z_0) / 2`.
        let canon = unsafe { compact(qb_op) };
        let expected_coeff = QkComplex64 {
            re: 0.5 * num_repeats as f64,
            im: 0.0,
        };
        let mut coeffs: Vec<QkComplex64> = vec![
            expected_coeff,
            QkComplex64 {
                re: -expected_coeff.re,
                im: 0.0,
            },
        ];
        let mut bit_terms: Vec<ffi::QkBitTerm> = vec![QkBitTermZ];
        let mut indices: Vec<u32> = vec![0];
        let mut boundaries: Vec<usize> = vec![0, 0, 1];
        let expected = unsafe {
            ffi::qk_obs_new(
                2,
                coeffs.len().try_into().unwrap(),
                bit_terms.len().try_into().unwrap(),
                coeffs.as_mut_ptr(),
                bit_terms.as_mut_ptr(),
                indices.as_mut_ptr(),
                boundaries.as_mut_ptr(),
            )
        };
        let factor = QkComplex64 { re: -1.0, im: 0.0 };
        let negated = unsafe { ffi::qk_obs_multiply(expected, &factor) };
        let diff = unsafe { ffi::qk_obs_add(canon, negated) };
        let diff = unsafe { ffi::qk_obs_canonicalize(diff, 1e-9) };
        let zero = unsafe { ffi::qk_obs_zero(2) };
        assert!(unsafe { ffi::qk_obs_equal(diff, zero) });
    }
}
