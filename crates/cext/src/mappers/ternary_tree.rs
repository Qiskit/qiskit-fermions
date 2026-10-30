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
use crate::pointers::{check_ptr, const_ptr_as_ref, try_slice_from_ptr};

use qiskit_fermions_core::mappers::ternary_tree::encoding::{ModeMap, TernaryTreeEncoding};
use qiskit_fermions_core::mappers::ternary_tree::{PauliLabel, TernaryTree};

/// @ingroup qf_ternary_tree
///
/// @brief Constructs a ternary tree from a parent-and-label specification.
///
/// @param num_nodes The number of nodes, which is both the number of fermionic modes and the number
///        of qubits the encoding uses.
/// @param root The index of the root node.
/// @param parents An array of ``num_nodes`` entries: the index of each node's parent. The root has no
///        parent, so its entry is never read and may hold any value, zero included. The array is not
///        read when ``num_nodes`` is zero, and may then be ``NULL``.
/// @param labels An array of ``num_nodes`` ``QfPauliLabel`` values: the label of the link descending
///        from each node's parent to it. The root has no such link, so its entry is never read and may
///        hold any value, zero included. The array is not read when ``num_nodes`` is zero, and may
///        then be ``NULL``.
/// @param out A pointer to where the created tree will be written on success. It is left untouched
///        if construction fails.
///
/// @return An exit code. This is ``>0`` if an error occurred. In particular, a
///         ``QfExitCode_ValueError`` is returned if the specification does not describe a single
///         connected ternary tree -- if it is empty, ``root`` is not a node, a label is repeated on
///         one node, a non-root entry of ``labels`` is not a ``QfPauliLabel`` value, a parent is
///         not a node, or some node is unreachable from the root.
///
/// @rst
///
/// A ternary tree on :math:`N` nodes defines a fermion-to-qubit encoding of :math:`N` fermionic modes
/// onto :math:`N` qubits. Each node is a qubit carrying three downward **links**, labeled ``X``,
/// ``Y`` and ``Z``. A link either descends to a child node (an *edge*) or terminates (a *leg*), and
/// each leg's path back to the root spells a Pauli string: crossing the link labeled :math:`P` that
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
///     // A tree in which node 0, the root, branches three ways. The root's entries are never read,
///     // so any value works.
///     #define IGNORED 0
///
///     uint32_t parents[4] = {IGNORED, 0, 0, 0};
///     QfPauliLabel labels[4] = {IGNORED, QfPauliLabel_X, QfPauliLabel_Y, QfPauliLabel_Z};
///
///     QfTernaryTree *tree;
///     QfExitCode exit = qf_ternary_tree_new(4, 0, parents, labels, &tree);
///
///     assert(exit == QfExitCode_Success);
///
/// @endrst
///
/// # Safety
///
/// `parents` and `labels` must each point to an array of at least `num_nodes` initialized values
/// (they are not read when `num_nodes` is zero), and `out` must be non-null and aligned. The entries
/// of `labels` need not hold valid labels: an invalid one is reported as `QfExitCode_ValueError`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ternary_tree_new(
    num_nodes: u32,
    root: u32,
    parents: *const u32,
    labels: *const PauliLabel,
    out: *mut *mut TernaryTree,
) -> ExitCode {
    // Alignment matters as much as nullness: both arrays are read and `out` is written. An empty
    // specification reads neither array, so they may be NULL, as a zero-length C array conventionally
    // may; that leaves it to be rejected below as the ValueError it is.
    if let Err(err) = check_ptr(out) {
        return err.into();
    }
    // SAFETY: Per documentation, each array holds `num_nodes` initialized entries. The labels are read
    // as the `u8`s they are laid out as, never as `PauliLabel`s: C can store any byte there (the
    // type is a plain `uint8_t` before C23), and a `PauliLabel` holding an invalid byte would be
    // undefined behavior before it could be checked. Each is validated below.
    let parents = match unsafe { try_slice_from_ptr(parents, num_nodes as usize) } {
        Ok(parents) => parents,
        Err(err) => return err.into(),
    };
    let labels = match unsafe { try_slice_from_ptr(labels.cast::<u8>(), num_nodes as usize) } {
        Ok(labels) => labels,
        Err(err) => return err.into(),
    };
    // This also rejects the empty specification, for which no index is a node.
    if root >= num_nodes {
        return ExitCode::ValueError;
    }
    // The root's entries are documented as ignored, so they are not read at all.
    let parsed: Result<Vec<Option<(u32, PauliLabel)>>, u8> = (0..num_nodes as usize)
        .map(|node| {
            if node == root as usize {
                return Ok(None);
            }
            Ok(Some((parents[node], PauliLabel::try_from(labels[node])?)))
        })
        .collect();
    let Ok(parsed) = parsed else {
        return ExitCode::ValueError;
    };

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
/// @param label The link the chain descends along, as a ``QfPauliLabel`` value. It is passed as a
///        plain byte, so that an invalid value is reported as a ``QfExitCode_ValueError``.
/// @param out A pointer to where the created tree will be written on success.
///
/// @return An exit code. This is ``>0`` if an error occurred, in particular a
///         ``QfExitCode_ValueError`` if ``num_nodes`` is zero or ``label`` is not a ``QfPauliLabel``
///         value.
///
/// @rst
///
/// ``QfPauliLabel_Z`` gives the Jordan-Wigner encoding and ``QfPauliLabel_X`` the parity encoding.
/// Both have Pauli weight ``num_nodes``, the worst a tree on that many nodes can do; see
/// :c:func:`qf_ternary_tree_breadth_first` for the optimal one.
///
/// This is only a convenience: it is :c:func:`qf_ternary_tree_breadth_first` with a branching of
/// one, whose ``order`` then needs just the one label.
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
/// `out` must be non-null and aligned. `label` need not hold a valid label: an invalid one is reported
/// as `QfExitCode_ValueError`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ternary_tree_chain(
    num_nodes: u32,
    // Taken as the `u8` a `QfPauliLabel` is laid out as, never as a `PauliLabel`: C can pass any
    // byte here (the type is a plain `uint8_t` before C23), and receiving an invalid one as a
    // `PauliLabel` would be undefined behavior before it could be checked.
    label: u8,
    out: *mut *mut TernaryTree,
) -> ExitCode {
    if let Err(err) = check_ptr(out) {
        return err.into();
    }
    let Ok(label) = PauliLabel::try_from(label) else {
        return ExitCode::ValueError;
    };
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
/// @param branching How many children to attach per node, from 1 to 3.
/// @param order The three labels to attach children along, in order. Only the first `branching`
///        entries are used, and they must be distinct, but all three must be ``QfPauliLabel`` values.
/// @param out A pointer to where the created tree will be written on success.
///
/// @return An exit code. This is ``>0`` if an error occurred, in particular if ``num_nodes`` is zero,
///         ``branching`` is not between 1 and 3, an entry of ``order`` is not a ``QfPauliLabel``
///         value, or the used part of ``order`` repeats a label.
///
/// @rst
///
/// A ``branching`` of 3 gives the balanced tree, whose Pauli weight
/// :math:`\lceil \log_3(2N+1) \rceil` is optimal over *all* fermion-to-qubit mappings. A ``branching``
/// of 2 gives a binary-branching tree of weight :math:`\lfloor \log_2 N \rfloor + 1`. That matches the
/// Bravyi-Kitaev weight but is *not* that encoding, for any ``order``: Bravyi-Kitaev's tree is the
/// irregular partial-sum shape, built through :c:func:`qf_ternary_tree_new`.
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
/// `order` must point to an array of at least 3 initialized `QfPauliLabel` entries, and `out` must be
/// non-null and aligned. The entries need not hold valid labels: an invalid one is reported as
/// `QfExitCode_ValueError`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ternary_tree_breadth_first(
    num_nodes: u32,
    branching: u8,
    order: *const PauliLabel,
    out: *mut *mut TernaryTree,
) -> ExitCode {
    if let Err(err) = check_ptr(order).and_then(|()| check_ptr(out)) {
        return err.into();
    }
    // SAFETY: Per documentation, `order` points to 3 initialized entries. They are read as the `u8`s
    // they are laid out as and validated before becoming `PauliLabel`s, for the reason given in
    // `qf_ternary_tree_new`.
    let order = unsafe { ::std::slice::from_raw_parts(order.cast::<u8>(), 3) };
    let (Ok(first), Ok(second), Ok(third)) = (
        PauliLabel::try_from(order[0]),
        PauliLabel::try_from(order[1]),
        PauliLabel::try_from(order[2]),
    ) else {
        return ExitCode::ValueError;
    };
    let order = [first, second, third];

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
/// operator, and that no Pauli term of a fermionic creation or annihilation operator is heavier,
/// since that is a sum of two Majorana operators. The two terms generally act on different qubits,
/// though, so the operator as a whole can be supported on more.
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
///         ``QfExitCode_NullPointerError`` is returned if ``tree`` or ``out`` is ``NULL``, and a
///         ``QfExitCode_ValueError`` if ``mode_map`` is not a permutation of the tree's nodes.
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
/// `tree` must be `NULL` or a valid `QfTernaryTree` pointer, `mode_map` must be `NULL` or point to an
/// array of at least `qf_ternary_tree_num_nodes(tree)` initialized `uint32_t` values, and `out` must
/// be `NULL` or aligned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ternary_tree_encoding_new(
    tree: *const TernaryTree,
    mode_map: *const u32,
    out: *mut *mut TernaryTreeEncoding,
) -> ExitCode {
    // Checked rather than left to `const_ptr_as_ref`, whose panic on a null pointer cannot unwind
    // across `extern "C"` and would abort the caller's process. A caller that ignored a failed
    // constructor holds exactly such a pointer.
    if let Err(err) = check_ptr(tree).and_then(|()| check_ptr(out)) {
        return err.into();
    }
    // SAFETY: Checked non-null and aligned above, and valid per documentation.
    let tree = unsafe { const_ptr_as_ref(tree) };

    let mode_map = if mode_map.is_null() {
        ModeMap::Identity
    } else {
        // SAFETY: Per documentation, `mode_map` points to `num_nodes` initialized entries.
        match unsafe { try_slice_from_ptr(mode_map, tree.num_nodes() as usize) } {
            Ok(modes) => ModeMap::Permutation(modes.to_vec()),
            Err(err) => return err.into(),
        }
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
/// @param out A pointer to where the total-parity observable will be written on success: a single
///        :math:`Z` string with coefficient :math:`+1`. The caller takes ownership and must free it
///        with ``qk_obs_free``. It is left untouched if the construction fails.
///
/// @return An exit code. This is ``>0`` if an error occurred. In particular, a
///         ``QfExitCode_NullPointerError`` is returned if ``encoding`` or ``out`` is ``NULL``, and a
///         ``QfExitCode_ValueError`` if ``num_qubits`` is below the encoding's mode count.
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
/// Note also that the Pauli string is *not* in general the all-:math:`Z` string: it is the leg reached
/// by the all-:math:`Z` *path*, which acts as the identity on every node not on that path. On the
/// balanced tree at four modes it is :math:`Z_3 Z_0`.
///
/// Example
/// -------
///
/// .. code-block:: c
///
///     QkObs *parity;
///     QfExitCode exit = qf_ternary_tree_encoding_total_parity(encoding, 4, &parity);
///
///     assert(exit == QfExitCode_Success);
///
///     qk_obs_free(parity);
///
/// @endrst
///
/// # Safety
///
/// `encoding` must be `NULL` or a valid `QfTernaryTreeEncoding` pointer, and `out` must be `NULL` or
/// aligned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_ternary_tree_encoding_total_parity(
    encoding: *const TernaryTreeEncoding,
    num_qubits: u32,
    out: *mut *mut qiskit_sys::QkObs,
) -> ExitCode {
    if let Err(err) = check_ptr(encoding).and_then(|()| check_ptr(out)) {
        return err.into();
    }
    // SAFETY: Checked non-null and aligned above, and valid per documentation.
    let encoding = unsafe { const_ptr_as_ref(encoding) };
    // An under-sized register is reported rather than allowed to reach `qk_obs_new`, whose
    // non-unwinding panic would abort the caller's process.
    match encoding.total_parity(num_qubits) {
        Ok(obs) => {
            // SAFETY: Checked non-null and aligned above.
            unsafe { out.write(obs) };
            ExitCode::Success
        }
        Err(_) => ExitCode::ValueError,
    }
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
/// This is the encoding-level counterpart of :c:func:`qf_ternary_tree_max_weight`, and always agrees
/// with it: although it ranges only over the ``2N`` legs that carry a Majorana operator, the leftover
/// parity leg is never deeper than all of them. It lets an encoding report its weight once the tree it
/// was compiled from has been freed.
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
