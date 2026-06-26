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

use crate::operators::fermion_operator::{FermionOperator, FermionOperatorTermView};
use std::collections::HashMap;

pub mod diagonal;
pub mod unique_modes;

/// Retains only those terms of `op` for which `keep` returns `true`, dropping the rest.
///
/// This mutates `op` in place by rebuilding its flat term storage from the surviving terms. If
/// `op` tracks group indices (see [`FermionOperator::groups`]), surviving terms retain their
/// relative grouping but the group indices are reassigned to a contiguous range starting from 0.
/// This is necessary to keep the grouping information consistent after terms (and possibly entire
/// groups) have been removed. Callers must therefore *not* rely on the specific group index of any
/// term being preserved across a call to this function.
fn retain_terms(op: &mut FermionOperator, keep: impl Fn(FermionOperatorTermView) -> bool) {
    // We compute the new flat storage for the surviving terms into fresh vectors and only assign
    // them back at the end, because `op.iter()` borrows `op` immutably for the duration of the
    // loop.
    let mut coeffs = Vec::with_capacity(op.coeffs.len());
    let mut actions = Vec::with_capacity(op.actions.len());
    let mut modes = Vec::with_capacity(op.modes.len());
    let mut boundaries = vec![0usize];
    let mut groups = op
        .groups
        .as_ref()
        .map(|_| Vec::with_capacity(op.coeffs.len()));
    // Maps an original group index to its reassigned, contiguous index. Entries are created lazily
    // in order of first appearance among the surviving terms, so the resulting indices span
    // 0..k without gaps even when entire groups are dropped.
    let mut group_remap: HashMap<u32, u32> = HashMap::new();

    for (idx, term) in op.iter().enumerate() {
        if !keep(term) {
            continue;
        }
        coeffs.push(term.coeff);
        actions.extend_from_slice(term.actions);
        modes.extend_from_slice(term.modes);
        boundaries.push(modes.len());
        if let (Some(dst), Some(src)) = (groups.as_mut(), op.groups.as_ref()) {
            let next = group_remap.len() as u32;
            let new_idx = *group_remap.entry(src[idx]).or_insert(next);
            dst.push(new_idx);
        }
    }

    op.coeffs = coeffs;
    op.actions = actions;
    op.modes = modes;
    op.boundaries = boundaries;
    op.groups = groups;
}
