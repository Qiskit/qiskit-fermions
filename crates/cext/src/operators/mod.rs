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

pub mod edge_vertex_operator;
pub mod fermion_operator;
pub mod library;
pub mod majorana_operator;
pub mod terms;
pub mod transfer_vertex_operator;

use qiskit_fermions_core::operators::terms::grouping::clash::Edge;

/// @ingroup qf_operators
///
/// @brief Frees an edge array returned by a ``qf_op_type_non_commutation_edges`` call.
///
/// @param edges A pointer to the edge array to free. Passing ``NULL`` is a no-op.
/// @param num_edges The edge count that was written alongside the array, unchanged.
///
/// @rst
///
/// The array is owned by the library rather than the caller, so it must be released through this
/// function rather than with ``free``. One function serves every operator representation, since the
/// allocation carries no type information.
///
/// The array is released by its size, which this function rebuilds from ``num_edges``, so the two
/// must be exactly the pair that one successful call wrote. Any other count is undefined behavior,
/// and so is freeing the same array twice. A failed call writes neither out-parameter, so start both
/// from ``NULL`` and ``0`` as below: the free is then a no-op whether or not the call succeeded,
/// rather than releasing whatever a previous call left in them.
///
/// .. code-block:: c
///     :linenos:
///
///     uint32_t *edges = NULL;
///     uint64_t num_edges = 0;
///     QfExitCode exit = qf_maj_op_non_commutation_edges(op, &edges, &num_edges);
///     // ... use the edges ...
///     qf_edges_free(edges, num_edges);
///
/// @endrst
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_edges_free(edges: *mut u32, num_edges: u64) {
    if edges.is_null() {
        return;
    }
    // A count that does not fit a `usize` cannot describe an array in this address space, so it
    // cannot be one `leak_edges` handed out. Leaking is then the only option that is not undefined
    // behavior, where truncating it would release a wrongly sized allocation.
    let Ok(len) = usize::try_from(num_edges) else {
        return;
    };
    // Reconstitutes the boxed slice `leak_edges` leaked, so that it is dropped with the layout it was
    // allocated with: `num_edges` arrays of two `u32`s each.
    // SAFETY: Per documentation, `edges` and `num_edges` are a pair handed out by one of the
    // `non_commutation_edges` functions, which leaked a boxed slice of exactly this many edges.
    drop(unsafe {
        Box::from_raw(std::ptr::slice_from_raw_parts_mut(
            edges.cast::<Edge>(),
            len,
        ))
    });
}

/// Hands an edge list to C as the flat array that [`qf_edges_free`] releases.
///
/// Every `qf_op_type_non_commutation_edges` accessor goes through here, so that the allocation and
/// its release are written down once. The edges are passed on as a boxed slice rather than a `Vec`,
/// because a boxed slice's capacity *is* its length: `qf_edges_free` only receives the edge count
/// back, and must be able to rebuild the exact layout from it.
///
/// No index is copied. An edge is an array of two `u32`s, so `n` edges already lie in memory as
/// `2 n` consecutive `u32`s, with the same size and alignment either way, and the buffer is handed
/// out as it is. Core sizes the list exactly, so turning it into a boxed slice does not reallocate
/// either.
///
/// # Safety
///
/// Both out-pointers must be non-null and aligned.
pub(crate) unsafe fn leak_edges(
    edges: Vec<Edge>,
    edges_out: *mut *mut u32,
    num_edges_out: *mut u64,
) {
    let num_edges = edges.len() as u64;
    let ptr = Box::into_raw(edges.into_boxed_slice()).cast::<u32>();

    // SAFETY: Guaranteed by the caller.
    unsafe {
        *edges_out = ptr;
        *num_edges_out = num_edges;
    }
}
