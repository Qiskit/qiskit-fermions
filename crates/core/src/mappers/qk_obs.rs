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

//! Shared plumbing for the mappers that emit a Qiskit `QkObs`.
//!
//! A fermion-to-qubit mapper in this crate is, in the end, a choice of Pauli image for each generator
//! of the operator algebra it consumes. Everything wrapped around that choice (building a
//! one-string observable, composing a term's images in order, and accumulating the mapped terms in
//! parallel under a memory bound) is identical whatever the encoding, and lives here.
//!
//! Nothing in this module knows about any particular encoding. It was factored out of
//! [`library::jordan_wigner`](crate::mappers::library::jordan_wigner), which is why the accumulator's
//! tuning is documented in terms of the operators that mapper sees.

use rayon::prelude::*;
use std::sync::{Arc, Mutex};

cfg_select! {
    feature = "pyext" => {
        pub(crate) use num_complex::Complex64 as QkComplex64;
        extern crate qiskit_pyo3_ffi as ffi;
        pub(crate) use qiskit_pyo3_ffi as ffi_backend;
        pub(crate) use ffi::QkBitTerm::X as QK_BIT_TERM_X;
        pub(crate) use ffi::QkBitTerm::Y as QK_BIT_TERM_Y;
        pub(crate) use ffi::QkBitTerm::Z as QK_BIT_TERM_Z;
    }
    feature = "cext" => {
        extern crate qiskit_sys as ffi;
        pub(crate) use ffi::QkComplex64 as QkComplex64;
        pub(crate) use qiskit_sys as ffi_backend;
        pub(crate) use ffi::QkBitTerm_QkBitTerm_X as QK_BIT_TERM_X;
        pub(crate) use ffi::QkBitTerm_QkBitTerm_Y as QK_BIT_TERM_Y;
        pub(crate) use ffi::QkBitTerm_QkBitTerm_Z as QK_BIT_TERM_Z;
    }
}

// The `QkBitTerm` values are re-exported from here so that the backend selection above is the only
// place in the crate that spells the two FFI backends' differing names for them. Call sites use the
// re-exported *values*, so they read identically on both backends.

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
pub(crate) unsafe fn compact(obs: *mut ffi::QkObs) -> *mut ffi::QkObs {
    let compacted = unsafe { ffi::qk_obs_canonicalize(obs, CANONICALIZE_TOL) };
    unsafe { ffi::qk_obs_free(obs) };
    compacted
}

// NOTE: https://stackoverflow.com/a/50341075
pub(crate) struct Wrapper {
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
    /// mapped image is identity-dominated (a constant offset repeated, say) would then grow
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
        // merge. Canonicalizing is not free (it builds a map over every term and allocates a whole
        // new observable), and this operator is the largest one the mapper handles, so the check is
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

/// Maps every term of an operator onto a `QkObs` in parallel, merging duplicates as it goes.
///
/// This is the single copy of the mapping driver shared by all four public mappers in this module:
/// the thread pool, the per-worker accumulators and the compaction schedule live here and nowhere
/// else, so a fix to any of them lands once.
///
/// It is generic over the term iterator and a per-term closure rather than over
/// [`OperatorTrait`](crate::operators::OperatorTrait): each operator's `*TermView` exposes `iter()`
/// as an *inherent* method rather than a trait one, so a term's actions are only reachable from code
/// that knows the concrete view type. Handing in a closure keeps that knowledge at the call site.
///
/// `map_term` returns the term's *unscaled* Pauli image together with the coefficient to apply, so
/// that the scaling stays in the one `qk_obs_scaled_add_inplace` below instead of costing an extra
/// observable per term. The returned pointer is consumed (freed) here.
///
/// # Panics
///
/// `map_term` must not unwind: it runs inside `for_each`, and a panic there would leak every
/// accumulator. The four implementations in this module are `qk_obs_*` calls plus arithmetic.
pub(crate) fn map_operator<T, I, F>(terms: I, num_qubits: u32, map_term: F) -> *mut ffi::QkObs
where
    I: Iterator<Item = T> + Send,
    T: Send,
    F: Fn(&T, u32) -> (*mut ffi::QkObs, QkComplex64) + Sync,
{
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(0)
        .build()
        .unwrap();

    let mut qubit_ops = vec![];
    for _ in 0..pool.current_num_threads() {
        qubit_ops.push(Arc::new(Mutex::new(Wrapper::zero(num_qubits))));
    }

    pool.install(|| {
        terms.par_bridge().for_each(|term| {
            let (mapped_term, qk_coeff) = map_term(&term, num_qubits);

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

    mapped_operator.ptr
}

/// Composes the Pauli images of a term's actions, left to right.
///
/// `image` is the per-action image builder. The composition order is load-bearing and matches the
/// order the actions appear in the term: `qk_obs_compose(new, acc)` appends `new` on the *right* of
/// what has been built so far. Reversing the operands silently yields the adjoint-ordered product,
/// which for a non-commuting word is a different operator.
///
/// The single-action case is by far the most common one for the vertex/edge/transfer Hamiltonians
/// these mappers see, and its image is already the whole term, so it skips the identity and the
/// compose entirely.
pub(crate) fn compose_actions<A, I, G>(actions: I, num_qubits: u32, image: G) -> *mut ffi::QkObs
where
    I: ExactSizeIterator<Item = A>,
    G: Fn(A, u32) -> *mut ffi::QkObs,
{
    let mut actions = actions;
    if actions.len() == 1 {
        return image(actions.next().unwrap(), num_qubits);
    }

    let mut mapped_term = unsafe { ffi::qk_obs_identity(num_qubits) };
    actions.for_each(|action| {
        let mapped_action = image(action, num_qubits);
        let new_term = unsafe { ffi::qk_obs_compose(mapped_action, mapped_term) };
        unsafe { ffi::qk_obs_free(mapped_action) };
        unsafe { ffi::qk_obs_free(mapped_term) };
        mapped_term = new_term;
    });
    mapped_term
}

/// Returns the `QkComplex64` mirror of a term coefficient.
pub(crate) fn qk_coeff(coeff: num_complex::Complex64) -> QkComplex64 {
    QkComplex64 {
        re: coeff.re,
        im: coeff.im,
    }
}
/// Builds a single Pauli string as an observable, from its bit terms and their qubit indices.
///
/// Unlike a fermionic action (a *two*-term sum), every generator of the Majorana, edge-vertex and
/// transfer-vertex algebras has an image consisting of exactly one Pauli string, so all three
/// mappers below build their images through this.
pub(crate) fn one_pauli_string(
    num_qubits: u32,
    coeff: QkComplex64,
    bit_terms: &mut [ffi::QkBitTerm],
    indices: &mut [u32],
) -> *mut ffi::QkObs {
    let mut coeffs = [coeff];
    let mut boundaries = [0usize, bit_terms.len()];
    unsafe {
        ffi::qk_obs_new(
            num_qubits,
            1,
            bit_terms.len().try_into().unwrap(),
            coeffs.as_mut_ptr(),
            bit_terms.as_mut_ptr(),
            indices.as_mut_ptr(),
            boundaries.as_mut_ptr(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Appends the weight-1 Pauli `Z_{qubit}` to `acc`, as the mapper's inner loop does.
    fn append_z(acc: &mut Wrapper, qubit: u32) {
        let mut coeffs: Vec<QkComplex64> = vec![QkComplex64 { re: 1.0, im: 0.0 }];
        let mut bit_terms: Vec<ffi::QkBitTerm> = vec![QK_BIT_TERM_Z];
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
}
