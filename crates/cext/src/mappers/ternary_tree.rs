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

use crate::exit_codes::ExitCode;
use crate::pointers::const_ptr_as_ref;

use qiskit_fermions_core::mappers::ternary_tree::encoding::{ModeMap, TernaryTreeEncoding};
use qiskit_fermions_core::mappers::ternary_tree::{PauliLabel, TernaryTree};

/// One node's entry in a tree specification: its parent and the label of the link descending to it.
///
/// A node whose `is_root` is `true` is the root; its `parent` and `label` are then ignored. Exactly one
/// entry of a specification must be a root.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct QfTernaryTreeNode {
    /// The index of this node's parent.
    pub parent: u32,
    /// The label of the link descending from `parent` to this node.
    pub label: PauliLabel,
    /// Whether this node is the root, in which case `parent` and `label` are ignored.
    pub is_root: bool,
}

/// @ingroup qf_ternary_tree
///
/// @brief Constructs a ternary tree from a parent-and-label specification.
///
/// @param num_nodes The number of nodes, which is both the number of fermionic modes and the number
///        of qubits the encoding uses.
/// @param spec An array of ``num_nodes`` entries, one per node. Exactly one must have ``is_root``
///        set.
/// @param out A pointer to where the created tree will be written on success. It is left untouched
///        if construction fails.
///
/// @return An exit code. This is ``>0`` if an error occurred. In particular, a
///         ``QfExitCode_ValueError`` is returned if the specification does not describe a single
///         connected ternary tree -- if it names no root or several, repeats a label on one node,
///         references a node out of range, or leaves some node unreachable from the root.
///
/// @rst
///
/// A ternary tree on :math:`N` nodes defines a fermion-to-qubit encoding of :math:`N` fermionic modes
/// onto :math:`N` qubits. Each node is a qubit carrying three downward **links**, labelled ``X``,
/// ``Y`` and ``Z``. A link either descends to a child node (an *edge*) or terminates (a *leg*), and
/// each leg's path back to the root spells a Pauli string: crossing the link labelled :math:`P` that
/// descends *from* node :math:`u` contributes :math:`P` on qubit :math:`u`.
///
/// Every tree of this shape yields a valid encoding, and the familiar ones are particular shapes. This
/// is the general constructor; :c:func:`qf_ternary_tree_chain` and
/// :c:func:`qf_ternary_tree_breadth_first` build the two uniform families more conveniently.
///
/// The tree must be freed with :c:func:`qf_ternary_tree_free`.
///
/// Example
/// -------
///
/// .. code-block:: c
///
///     // A tree in which node 0 branches three ways.
///     QfTernaryTreeNode spec[4] = {
///         {0, QfPauliLabel_Z, true},
///         {0, QfPauliLabel_X, false},
///         {0, QfPauliLabel_Y, false},
///         {0, QfPauliLabel_Z, false},
///     };
///
///     QfTernaryTree *tree;
///     QfExitCode exit = qf_ternary_tree_new(4, spec, &tree);
///
///     assert(exit == QfExitCode_Success);
///
/// @endrst
///
/// # Safety
///
/// `spec` must point to an array of at least `num_nodes` initialized `QfTernaryTreeNode` values, and
/// `out` must be non-null and aligned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ternary_tree_new(
    num_nodes: u32,
    spec: *const QfTernaryTreeNode,
    out: *mut *mut TernaryTree,
) -> ExitCode {
    if spec.is_null() || out.is_null() {
        return ExitCode::NullPointerError;
    }
    // SAFETY: Per documentation, `spec` points to `num_nodes` initialized entries.
    let entries = unsafe { ::std::slice::from_raw_parts(spec, num_nodes as usize) };
    let parsed: Vec<Option<(u32, PauliLabel)>> = entries
        .iter()
        .map(|entry| {
            if entry.is_root {
                None
            } else {
                Some((entry.parent, entry.label))
            }
        })
        .collect();

    match TernaryTree::try_new(&parsed) {
        Ok(tree) => {
            // SAFETY: Per documentation, `out` is non-null and aligned.
            unsafe { out.write(Box::into_raw(Box::new(tree))) };
            ExitCode::Success
        }
        Err(_) => ExitCode::ValueError,
    }
}

/// @ingroup qf_ternary_tree
///
/// @brief Constructs a linear chain descending along one label.
///
/// @param num_nodes The number of nodes, and hence of modes and qubits.
/// @param label The link the chain descends along.
/// @param out A pointer to where the created tree will be written on success.
///
/// @return An exit code. This is ``>0`` if an error occurred, in particular if ``num_nodes`` is zero.
///
/// @rst
///
/// ``QfPauliLabel_Z`` gives the Jordan-Wigner encoding and ``QfPauliLabel_X`` the parity encoding.
/// Both have Pauli weight ``num_nodes``, the worst a tree on that many nodes can do; see
/// :c:func:`qf_ternary_tree_breadth_first` for the optimal one.
///
/// The tree must be freed with :c:func:`qf_ternary_tree_free`.
///
/// Example
/// -------
///
/// .. code-block:: c
///
///     // The Jordan-Wigner encoding of four modes: a chain descending along Z.
///     QfTernaryTree *chain;
///     QfExitCode exit = qf_ternary_tree_chain(4, QfPauliLabel_Z, &chain);
///
///     assert(exit == QfExitCode_Success);
///     // A chain is the worst case: one Majorana operator spans the whole register.
///     assert(qf_ternary_tree_max_weight(chain) == 4);
///
///     qf_ternary_tree_free(chain);
///
/// @endrst
///
/// # Safety
///
/// `out` must be non-null and aligned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ternary_tree_chain(
    num_nodes: u32,
    label: PauliLabel,
    out: *mut *mut TernaryTree,
) -> ExitCode {
    if out.is_null() {
        return ExitCode::NullPointerError;
    }
    match TernaryTree::chain(num_nodes, label) {
        Ok(tree) => {
            // SAFETY: Per documentation, `out` is non-null and aligned.
            unsafe { out.write(Box::into_raw(Box::new(tree))) };
            ExitCode::Success
        }
        Err(_) => ExitCode::ValueError,
    }
}

/// @ingroup qf_ternary_tree
///
/// @brief Constructs a tree by attaching up to `branching` children to each node in turn.
///
/// @param num_nodes The number of nodes, and hence of modes and qubits.
/// @param branching How many children to attach per node, at most 3.
/// @param order The three labels to attach children along, in order. Only the first `branching`
///        entries are used, and they must be distinct.
/// @param out A pointer to where the created tree will be written on success.
///
/// @return An exit code. This is ``>0`` if an error occurred, in particular if ``num_nodes`` is zero
///         or the used part of ``order`` repeats a label.
///
/// @rst
///
/// A ``branching`` of 3 gives the balanced tree, whose Pauli weight
/// :math:`\lceil \log_3(2N+1) \rceil` is optimal over *all* fermion-to-qubit mappings. A ``branching``
/// of 2 gives a binary-branching tree of weight :math:`\lfloor \log_2 N \rfloor + 1`. That matches the
/// Bravyi-Kitaev weight but is *not* that encoding, except when ``num_nodes`` is a power of two:
/// Bravyi-Kitaev's tree is the irregular partial-sum shape, built through
/// :c:func:`qf_ternary_tree_new`.
///
/// ``order`` is part of the encoding's definition: it changes which Pauli strings are produced, though
/// not their weight.
///
/// The tree must be freed with :c:func:`qf_ternary_tree_free`.
///
/// Example
/// -------
///
/// .. code-block:: c
///
///     QfPauliLabel order[3] = {QfPauliLabel_X, QfPauliLabel_Y, QfPauliLabel_Z};
///
///     QfTernaryTree *tree;
///     QfExitCode exit = qf_ternary_tree_breadth_first(13, 3, order, &tree);
///
///     assert(exit == QfExitCode_Success);
///     // The balanced tree needs weight 3 where a chain of 13 modes needs 13.
///     assert(qf_ternary_tree_max_weight(tree) == 3);
///
/// @endrst
///
/// # Safety
///
/// `order` must point to an array of at least 3 initialized `QfPauliLabel` values, and `out` must be
/// non-null and aligned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ternary_tree_breadth_first(
    num_nodes: u32,
    branching: u8,
    order: *const PauliLabel,
    out: *mut *mut TernaryTree,
) -> ExitCode {
    if order.is_null() || out.is_null() {
        return ExitCode::NullPointerError;
    }
    // SAFETY: Per documentation, `order` points to 3 initialized entries.
    let order = unsafe { ::std::slice::from_raw_parts(order, 3) };
    let order = [order[0], order[1], order[2]];

    match TernaryTree::breadth_first(num_nodes, branching, order) {
        Ok(tree) => {
            // SAFETY: Per documentation, `out` is non-null and aligned.
            unsafe { out.write(Box::into_raw(Box::new(tree))) };
            ExitCode::Success
        }
        Err(_) => ExitCode::ValueError,
    }
}

/// @ingroup qf_ternary_tree
///
/// @brief Returns the number of nodes of a tree, i.e. its mode and qubit count.
///
/// @param tree A pointer to the tree.
///
/// @return The number of nodes.
///
/// # Safety
///
/// `tree` must be a non-null, aligned pointer to a `QfTernaryTree`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ternary_tree_num_nodes(tree: *const TernaryTree) -> u32 {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    unsafe { const_ptr_as_ref(tree) }.num_nodes()
}

/// @ingroup qf_ternary_tree
///
/// @brief Returns the number of legs of a tree, always ``2 * num_nodes + 1``.
///
/// @param tree A pointer to the tree.
///
/// @return The number of legs.
///
/// @rst
///
/// This holds for every shape: a node with fewer children has more *legs*, not fewer Pauli strings. Of
/// these, ``2 * num_nodes`` pair into the Majorana operators and exactly one is left over, carrying the
/// total fermionic parity -- see :c:func:`qf_ternary_tree_encoding_total_parity`.
///
/// @endrst
///
/// # Safety
///
/// `tree` must be a non-null, aligned pointer to a `QfTernaryTree`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ternary_tree_num_legs(tree: *const TernaryTree) -> u32 {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    unsafe { const_ptr_as_ref(tree) }.num_legs()
}

/// @ingroup qf_ternary_tree
///
/// @brief Returns the largest Pauli weight of any single Majorana operator the tree generates.
///
/// @param tree A pointer to the tree.
///
/// @return The maximum Pauli weight.
///
/// @rst
///
/// This is the figure of merit a tree is chosen for. Note that it is the weight of one *Majorana*
/// operator; a fermionic creation or annihilation operator is a sum of two of them, and so may reach
/// twice this.
///
/// @endrst
///
/// # Safety
///
/// `tree` must be a non-null, aligned pointer to a `QfTernaryTree`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ternary_tree_max_weight(tree: *const TernaryTree) -> u32 {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    unsafe { const_ptr_as_ref(tree) }.max_weight()
}

/// @ingroup qf_ternary_tree
///
/// @brief Frees a ternary tree.
///
/// @param tree A pointer to the tree to free. Passing ``NULL`` is a no-op.
///
/// # Safety
///
/// `tree` must be a pointer returned by one of the `qf_ternary_tree_*` constructors and not already
/// freed, or `NULL`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ternary_tree_free(tree: *mut TernaryTree) {
    if !tree.is_null() {
        if !tree.is_aligned() {
            panic!("Attempted to free a non-aligned pointer.")
        }
        // SAFETY: We have verified the pointer is non-null and aligned, so it should be readable by
        // Box.
        unsafe {
            let _ = Box::from_raw(tree);
        }
    }
}

/// @ingroup qf_ternary_tree
///
/// @brief Compiles a ternary tree into the Pauli images of its Majorana operators.
///
/// @param tree A pointer to the tree defining the encoding.
/// @param mode_map Which mode each node serves, as an array of ``qf_ternary_tree_num_nodes(tree)``
///        entries indexed by node, or ``NULL`` for the identity assignment.
/// @param out A pointer to where the created encoding will be written on success.
///
/// @return An exit code. This is ``>0`` if an error occurred. In particular, a
///         ``QfExitCode_ValueError`` is returned if ``mode_map`` is not a permutation of the tree's
///         nodes.
///
/// @rst
///
/// Pairing the tree's legs into Majorana operators and building their Pauli strings is done once,
/// here, rather than per mapped operator, so one encoding should be reused across every operator
/// mapped through it. The encoding carries no qubit count of its own: the mappers take one, so a
/// single encoding can serve registers of different widths.
///
/// The pairing follows Algorithm 1 of [Bonsai]_: from each node, descend its ``X`` link and then ``Z``
/// links until reaching a leg, which carries :math:`\gamma` of that node's mode; the ``Y`` link
/// likewise gives :math:`\gamma'`. This is what makes the all-zero computational state the fermionic
/// vacuum, so it is not an adjustable convention.
///
/// The encoding must be freed with :c:func:`qf_ternary_tree_encoding_free`.
///
/// Example
/// -------
///
/// ``mode_map`` is indexed by *node* and names the *mode* that node serves, so it reassigns which
/// mode's Majorana operators each pair of Pauli strings represents without changing the strings
/// themselves. Pass ``NULL`` when node ``j`` should serve mode ``j``.
///
/// .. code-block:: c
///
///     QfTernaryTree *tree;
///     qf_ternary_tree_chain(4, QfPauliLabel_Z, &tree);
///
///     // Node j serves mode j.
///     QfTernaryTreeEncoding *identity;
///     QfExitCode exit = qf_ternary_tree_encoding_new(tree, NULL, &identity);
///     assert(exit == QfExitCode_Success);
///
///     // Reverse the assignment: node 0 now serves mode 3, node 1 mode 2, and so on.
///     uint32_t mode_map[4] = {3, 2, 1, 0};
///     QfTernaryTreeEncoding *reversed;
///     exit = qf_ternary_tree_encoding_new(tree, mode_map, &reversed);
///     assert(exit == QfExitCode_Success);
///
///     // Both cover the same modes; only which node serves which mode differs.
///     assert(qf_ternary_tree_encoding_num_modes(reversed) == 4);
///
///     qf_ternary_tree_encoding_free(identity);
///     qf_ternary_tree_encoding_free(reversed);
///     qf_ternary_tree_free(tree);
///
/// Reordering the modes of a fixed tree is one of the cheapest ways to lower the Pauli weight of a
/// particular Hamiltonian, which is why the assignment is separate from the tree.
///
/// .. [Bonsai] A. Miller, Z. Zimborás, S. Knecht, S. Maniscalco and G. García-Pérez, Bonsai
///    algorithm: grow your own fermion-to-qubit mappings, PRX Quantum 4, 030314 (2023),
///    `arXiv:2212.09731 <https://arxiv.org/abs/2212.09731>`__.
///
/// @endrst
///
/// # Safety
///
/// `tree` must be a valid `QfTernaryTree` pointer, `mode_map` must be `NULL` or point to an array of
/// at least `qf_ternary_tree_num_nodes(tree)` initialized `uint32_t` values, and `out` must be
/// non-null and aligned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ternary_tree_encoding_new(
    tree: *const TernaryTree,
    mode_map: *const u32,
    out: *mut *mut TernaryTreeEncoding,
) -> ExitCode {
    if out.is_null() {
        return ExitCode::NullPointerError;
    }
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let tree = unsafe { const_ptr_as_ref(tree) };

    let mode_map = if mode_map.is_null() {
        ModeMap::Identity
    } else {
        // SAFETY: Per documentation, `mode_map` points to `num_nodes` initialized entries.
        let modes = unsafe { ::std::slice::from_raw_parts(mode_map, tree.num_nodes() as usize) };
        ModeMap::Permutation(modes.to_vec())
    };

    match TernaryTreeEncoding::new(tree, &mode_map) {
        Ok(encoding) => {
            // SAFETY: Per documentation, `out` is non-null and aligned.
            unsafe { out.write(Box::into_raw(Box::new(encoding))) };
            ExitCode::Success
        }
        Err(_) => ExitCode::ValueError,
    }
}

/// @ingroup qf_ternary_tree
///
/// @brief Returns the number of fermionic modes a compiled encoding covers.
///
/// @param encoding A pointer to the encoding.
///
/// @return The number of modes, equal to the tree's node count.
///
/// # Safety
///
/// `encoding` must be a non-null, aligned pointer to a `QfTernaryTreeEncoding`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ternary_tree_encoding_num_modes(
    encoding: *const TernaryTreeEncoding,
) -> u32 {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    unsafe { const_ptr_as_ref(encoding) }.num_modes()
}

/// @ingroup qf_ternary_tree
///
/// @brief Builds the total fermionic parity observable of a compiled encoding.
///
/// @param encoding A pointer to the compiled encoding.
/// @param num_qubits The number of qubits for the resulting observable. This must be at least the
///        encoding's mode count; a larger value pads with the identity.
///
/// @return The total-parity observable: a single Pauli string with a :math:`\pm 1` coefficient. The
///         caller takes ownership and must free it with ``qk_obs_free``. Returns ``NULL`` if
///         ``num_qubits`` is below the encoding's mode count.
///
/// @rst
///
/// A tree's :math:`2N+1` legs pair into :math:`2N` Majorana operators, leaving exactly one over, and
/// that leftover is the total parity :math:`\prod_j (1 - 2 n_j)`.
///
/// .. important::
///    This is an **observable to measure, not a constraint to impose**. Both eigenvalues are
///    physical, so do not project a state onto the :math:`+1` sector: a ternary-tree encoding spends
///    one qubit per mode and is a unitary isomorphism onto the *whole* :math:`2^N`-dimensional space,
///    with both parity sectors represented at multiplicity :math:`2^{N-1}` and no unphysical subspace
///    to project away.
///
///    Local encodings that add ancilla qubits do work that way, using more qubits than modes so that
///    a stabilizer subspace is what keeps them faithful. Ternary trees add no such qubits.
///
/// Note also that the Pauli string is *not* in general the all-:math:`Z` string: it is the leg reached
/// by the all-:math:`Z` *path*, which acts as the identity on every node not on that path. On the
/// balanced tree at four modes it is :math:`Z_3 Z_0`.
///
/// @endrst
///
/// # Safety
///
/// `encoding` must be a non-null, aligned pointer to a `QfTernaryTreeEncoding`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ternary_tree_encoding_total_parity(
    encoding: *const TernaryTreeEncoding,
    num_qubits: u32,
) -> *mut qiskit_sys::QkObs {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let encoding = unsafe { const_ptr_as_ref(encoding) };
    // An under-sized register is reported as `NULL` rather than allowed to reach `qk_obs_new`, whose
    // non-unwinding panic would abort the caller's process.
    encoding
        .total_parity(num_qubits)
        .unwrap_or(std::ptr::null_mut())
}

/// @ingroup qf_ternary_tree
///
/// @brief Returns the largest Pauli weight of any single Majorana operator of a compiled encoding.
///
/// @param encoding A pointer to the encoding.
///
/// @return The largest weight over the encoding's ``2N`` Majorana operators.
///
/// @rst
///
/// This is the encoding-level counterpart of :c:func:`qf_ternary_tree_max_weight`, and it is the more
/// meaningful of the two: it is restricted to the ``2N`` legs that actually carry a Majorana operator,
/// whereas the tree-level function ranges over all ``2N+1`` legs and so can also report the leftover
/// parity leg.
///
/// @endrst
///
/// # Safety
///
/// `encoding` must be a non-null, aligned pointer to a `QfTernaryTreeEncoding`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ternary_tree_encoding_max_weight(
    encoding: *const TernaryTreeEncoding,
) -> u32 {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    unsafe { const_ptr_as_ref(encoding) }.max_weight()
}

/// @ingroup qf_ternary_tree
///
/// @brief Frees a compiled ternary-tree encoding.
///
/// @param encoding A pointer to the encoding to free. Passing ``NULL`` is a no-op.
///
/// # Safety
///
/// `encoding` must be a pointer returned by `qf_ternary_tree_encoding_new` and not already freed, or
/// `NULL`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ternary_tree_encoding_free(encoding: *mut TernaryTreeEncoding) {
    if !encoding.is_null() {
        if !encoding.is_aligned() {
            panic!("Attempted to free a non-aligned pointer.")
        }
        // SAFETY: We have verified the pointer is non-null and aligned, so it should be readable by
        // Box.
        unsafe {
            let _ = Box::from_raw(encoding);
        }
    }
}
