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

//! Ternary trees, the structures that define a fermion-to-qubit encoding.
//!
//! A ternary tree on `N` nodes defines an encoding of `N` fermionic modes onto `N` qubits. Every
//! familiar encoding is one particular tree: Jordan-Wigner is a chain, the balanced tree achieves the
//! optimal Pauli weight, and everything in between is reachable. This module holds the tree and the
//! [`encoding`] it compiles into; the mappers that consume them live in
//! [`library::ternary_tree`](crate::mappers::library::ternary_tree).
//!
//! Building a tree, pairing its legs into Majorana operators and checking its invariants are all pure
//! index arithmetic, so future tree-search code (Bonsai, Treespilation) can work on trees without
//! building an observable. Producing one is a single method,
//! [`TernaryTreeEncoding::total_parity`](encoding::TernaryTreeEncoding::total_parity); everything else
//! here is index bookkeeping.
//!
//! # References
//!
//! 1. A. Miller, Z. Zimborás, S. Knecht, S. Maniscalco and G. García-Pérez, *Bonsai algorithm:
//!    grow your own fermion-to-qubit mappings*, PRX Quantum **4**, 030314 (2023),
//!    [arXiv:2212.09731](https://arxiv.org/abs/2212.09731).
//! 2. Z. Jiang, A. Kalev, W. Mruczkiewicz and H. Neven, *Optimal fermion-to-qubit mapping via ternary
//!    trees with applications to reduced quantum states learning*, Quantum **4**, 276 (2020),
//!    [arXiv:1910.10746](https://arxiv.org/abs/1910.10746).

use thiserror::Error;

pub mod encoding;

/// Index of a node in a [`TernaryTree`].
///
/// Nodes are qubits: node `u` is qubit `u`. This is a `u32` to match the mode indices of the operator
/// types, which is the same index space once a [`ModeMap`](encoding::ModeMap) is applied.
pub type NodeId = u32;

/// Index of a leg in a [`TernaryTree`].
///
/// There are always exactly `2 * num_nodes + 1` legs; see [`TernaryTree::num_legs`].
pub type LegId = u32;

/// The Pauli labelling one of a node's three downward links.
///
/// The discriminants are the symplectic `(z, x)` bit pair Qiskit uses, with `z` in the low bit and
/// `x` in the next: `Z = 0b01`, `X = 0b10`, `Y = 0b11`. That makes them the discriminants of
/// Qiskit's `QkBitTerm`, so a compiled encoding can be handed to the FFI without translating each
/// label. Both the size and each individual discriminant are asserted at compile time where that is
/// relied upon, in [`library::ternary_tree`](crate::mappers::library::ternary_tree).
///
/// `0b00` would be the natural encoding of the identity, which is why no `I` variant is named: a
/// link always carries a non-trivial Pauli, and identity on a qubit is that qubit's absence from a
/// leg's sparse Pauli string. Qiskit omits it from `QkBitTerm` for the same reason.
///
/// The discriminants deliberately do *not* agree with [`Self::slot`], which orders the labels `X`,
/// `Y`, `Z` rather than `Z`, `X`, `Y`. The two answer to different authorities and neither can yield:
/// the discriminants are a wire format fixed by Qiskit's FFI, while the slot order is the sequence
/// Algorithm 1 of Ref. 1 pairs the links in, and it fixes [`LegId`] numbering. So `slot` is written
/// as an explicit `match` rather than arithmetic on the discriminant -- deriving one from the other
/// would weld the two requirements together, and changing either would silently reorder the other.
/// Nothing sorts or compares a `PauliLabel`, so the discriminant order carries no meaning of its own
/// beyond the numeric values themselves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum PauliLabel {
    /// The `X` Pauli.
    X = 2,
    /// The `Y` Pauli.
    Y = 3,
    /// The `Z` Pauli. Distinguished only by the pairing rule, which descends along `Z`.
    Z = 1,
}

impl PauliLabel {
    /// The three labels, in the order used to index a node's slots.
    ///
    /// Deliberately not `pub`: it is an internal convenience, not API. cbindgen exports a `pub`
    /// associated constant whose type is an exported enum, so making this public would put a
    /// `QfPauliLabel_ALL` initializer-list macro in the C header, which no caller asked for.
    /// [`Self::iter`] is the public way to enumerate the labels.
    const ALL: [PauliLabel; 3] = [PauliLabel::X, PauliLabel::Y, PauliLabel::Z];

    /// Returns the three labels in slot order, i.e. `X`, `Y`, `Z`.
    ///
    /// Rust has no built-in way to enumerate an enum's variants, so this is how a caller outside the
    /// crate walks a node's three links: pair it with [`Self::slot`] to index into a `[Slot; 3]`.
    #[inline]
    pub fn iter() -> impl ExactSizeIterator<Item = PauliLabel> {
        Self::ALL.into_iter()
    }

    /// Returns the index of this label within a node's `[Slot; 3]`.
    ///
    /// Note this is *not* `self as u8 - 1`: the slot order is `X`, `Y`, `Z` while the discriminants
    /// ascend `Z`, `X`, `Y`. See the type's documentation for why the two are independent.
    ///
    /// `const` so that the round-trip assertion below can call it.
    #[inline]
    pub const fn slot(self) -> usize {
        match self {
            PauliLabel::X => 0,
            PauliLabel::Y => 1,
            PauliLabel::Z => 2,
        }
    }
}

/// [`PauliLabel::iter`] and [`PauliLabel::slot`] must agree: `iter` yields the labels in slot order,
/// and `slot` names the index of each. [`TernaryTree::try_from_children`] zips a caller's
/// `[Option<NodeId>; 3]` against `iter`, so a disagreement would mislabel every edge there while
/// leaving `try_new` correct, and the result would still satisfy every structural invariant -- so
/// nothing else would catch it.
const _: () = {
    assert!(PauliLabel::ALL[PauliLabel::X.slot()] as u8 == PauliLabel::X as u8);
    assert!(PauliLabel::ALL[PauliLabel::Y.slot()] as u8 == PauliLabel::Y as u8);
    assert!(PauliLabel::ALL[PauliLabel::Z.slot()] as u8 == PauliLabel::Z as u8);
};

/// What hangs off one of a node's three downward links.
///
/// Every node has exactly three, which is what makes the leg count independent of the tree's shape: a
/// node with fewer children does not have fewer links, it has more *legs*. See
/// [`TernaryTree::num_legs`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slot {
    /// The link descends to a child node.
    Edge(NodeId),
    /// The link terminates. It still generates a Pauli string.
    Leg(LegId),
}

/// Errors from building a [`TernaryTree`].
///
/// Kept separate from [`CoherenceError`](crate::operators::CoherenceError), which describes the
/// coherence of *operator* index buffers: a tree is a different kind of object, and folding these in
/// would give that enum variants no operator can produce.
#[derive(Error, Debug, PartialEq, Eq)]
pub enum TernaryTreeError {
    /// Two of a node's links carry the same Pauli label.
    #[error("node {node} has more than one link labelled {label:?}")]
    DuplicateLabel {
        /// The offending node.
        node: NodeId,
        /// The label given twice.
        label: PauliLabel,
    },
    /// A node was given more than three children.
    #[error("node {node} has {num_children} children; a ternary tree node has at most 3")]
    TooManyChildren {
        /// The offending node.
        node: NodeId,
        /// How many children it was given.
        num_children: usize,
    },
    /// The specification does not describe a single connected tree.
    ///
    /// Raised when some node cannot be reached from the root, which is what would otherwise let a
    /// forest or a disconnected component through. The leg count `2N+1` holds only for a connected
    /// tree, so this is checked rather than assumed.
    #[error("node {node} is not reachable from the root ({root})")]
    Disconnected {
        /// A node with no path to the root.
        node: NodeId,
        /// The root the tree was built around.
        root: NodeId,
    },
    /// The specification names more than one root, or none at all.
    #[error("expected exactly one root (a node with no parent), found {num_roots}")]
    NotATree {
        /// How many parentless nodes the specification contained.
        num_roots: usize,
    },
    /// A specification entry refers to a node outside the tree.
    #[error("the specification references node {node}, but the tree has only {num_nodes} nodes")]
    NodeOutOfRange {
        /// The out-of-range index.
        node: NodeId,
        /// The number of nodes the tree has.
        num_nodes: u32,
    },
    /// A node was named as its own parent.
    #[error("node {node} is its own parent")]
    SelfParent {
        /// The offending node.
        node: NodeId,
    },
    /// The tree has no nodes.
    ///
    /// A zero-mode encoding is degenerate rather than useful, and `2N+1 = 1` would leave a single leg
    /// with nothing to pair it with.
    #[error("a ternary tree must have at least one node")]
    Empty,
    /// The tree is too large for its leg indices to be addressed.
    #[error("a ternary tree may have at most {max} nodes, got {num_nodes}")]
    TooLarge {
        /// The number of nodes requested.
        num_nodes: u32,
        /// The largest supported node count.
        max: u32,
    },
    /// A [`ModeMap`](encoding::ModeMap) has the wrong length.
    #[error("the mode map has {len} entries, but the tree has {num_nodes} nodes")]
    ModeMapLength {
        /// The length given.
        len: usize,
        /// The tree's node count.
        num_nodes: u32,
    },
    /// A [`ModeMap`](encoding::ModeMap) names the same mode twice.
    #[error("the mode map is not a permutation: mode {mode} appears more than once")]
    ModeMapNotAPermutation {
        /// The repeated mode.
        mode: u32,
    },
    /// A [`ModeMap`](encoding::ModeMap) names a mode outside the tree.
    ///
    /// Distinct from [`Self::NodeOutOfRange`]: the offending index is a *mode*, and a mode map is
    /// read as node-indexed entries holding mode values, so reporting it as a node misdirects.
    #[error("the mode map references mode {mode}, but the tree has only {num_nodes} nodes")]
    ModeOutOfRange {
        /// The out-of-range mode.
        mode: u32,
        /// The number of nodes the tree has.
        num_nodes: u32,
    },
}

/// The largest node count whose leg indices (`2N+1`) still fit in a [`LegId`].
pub const MAX_NODES: u32 = (u32::MAX - 1) / 2;

/// A rooted ternary tree defining a fermion-to-qubit encoding.
///
/// Each node is a qubit and carries three downward **links**, labelled `X`, `Y` and `Z`. A link
/// either descends to a child node (an *edge*) or terminates (a *leg*). Each leg's upward path to the
/// root spells a Pauli string: crossing the link labelled `P` that descends *from* node `u`
/// contributes `P` on qubit `u`, with the identity everywhere else.
///
/// # Validity is structural
///
/// Any rooted, connected tree of this shape yields a valid set of `2N+1` Hermitian, involutory,
/// mutually anticommuting operators -- whatever its shape, whatever the assignment of qubits to
/// nodes, and whatever permutation of `{X,Y,Z}` each node uses. Anticommutation follows from the
/// first-common-ancestor argument: two distinct legs' paths agree above their deepest shared node,
/// carry two *different* labels at it, and at least one acts trivially below, so the strings overlap
/// non-trivially on exactly one qubit. There is consequently nothing to verify at runtime beyond the
/// structural invariants enforced when a tree is constructed, and the mappers never check
/// anticommutation.
///
/// # Constructing one
///
/// [`try_new`](Self::try_new) and [`try_from_children`](Self::try_from_children) build an arbitrary
/// tree; they are the general interface, and the shapes the literature names are all reachable
/// through them. [`chain`](Self::chain) and [`breadth_first`](Self::breadth_first) are conveniences
/// for the two uniform families:
///
/// ```text
/// Encoding            Constructor                        Pauli weight
/// ------------------  ---------------------------------  --------------------------
/// Jordan-Wigner       chain(n, PauliLabel::Z)            n
/// Parity              chain(n, PauliLabel::X)            n
/// Balanced (JKMN)     breadth_first(n, 3, [X, Y, Z])     ceil(log3(2n+1)), optimal
/// Binary-branching    breadth_first(n, 2, [Z, X, Y])     floor(log2 n) + 1
/// ```
///
/// Those conveniences cover the *uniform* families only. The trees that motivate the framework are not
/// uniform: the Bonsai algorithm grows a spanning tree of a hardware coupling graph, so its branching
/// follows the device's connectivity. Those are built with [`Self::try_new`], which is why that is the
/// general interface and the conveniences are only conveniences.
///
/// # Bravyi-Kitaev
///
/// Bravyi-Kitaev is in this family but has no constructor here, its tree being irregular: over a range of
/// nodes `[l, r]`, the node placed is `l + h - 1` for the largest power of two `h <= r - l + 1`, with `X`
/// descending to the block before it, `Z` to the remainder after it, and `Y` left as a leg. The encoding
/// tests pin it at `N = 4` and `N = 6`; note that its root is the largest `2^k - 1` that is `<= N-1`
/// rather than `N-1` itself, so a check run only at power-of-two sizes cannot tell those two rules apart.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TernaryTree {
    /// The one node with no parent; every leg's path terminates here.
    root: NodeId,
    /// One `[Slot; 3]` per node, indexed by [`PauliLabel::slot`].
    slots: Vec<[Slot; 3]>,
    /// The parent of each node and the label of the link descending to it; `None` for the root.
    ///
    /// Cached rather than searched for. Walking a leg to the root is then `O(depth)` with no downward
    /// scan, and the tree edits that tree-search algorithms perform (reparenting a subtree,
    /// re-rooting) become local rather than `O(N)`.
    parent: Vec<Option<(NodeId, PauliLabel)>>,
    /// The `(node, label)` whose link terminates in each leg, indexed by [`LegId`].
    ///
    /// The inverse of the [`Slot::Leg`] entries, kept so that enumerating legs is a slice scan rather
    /// than a filter over all `3N` slots.
    legs: Vec<(NodeId, PauliLabel)>,
}

impl TernaryTree {
    /// Builds a tree from a parent-and-label specification.
    ///
    /// `spec[u]` gives node `u`'s parent and the label of the link descending to it, or `None` if `u`
    /// is the root. Exactly one entry must be `None`. Every link not named by some entry becomes a
    /// leg. Legs are numbered by ascending node, and within a node in [`PauliLabel::slot`] order
    /// (`X`, `Y`, `Z`), which fixes [`LegId`] deterministically for a given tree.
    ///
    /// This is the general constructor: any ternary tree is expressible as such a specification,
    /// including the irregular, connectivity-derived trees that the Bonsai algorithm produces.
    pub fn try_new(spec: &[Option<(NodeId, PauliLabel)>]) -> Result<Self, TernaryTreeError> {
        let num_nodes = Self::check_size(spec.len())?;

        let mut slots = vec![[Slot::Leg(0); 3]; spec.len()];
        let mut assigned = vec![[false; 3]; spec.len()];
        let mut parent = vec![None; spec.len()];
        let mut num_children = vec![0usize; spec.len()];
        let mut roots = Vec::new();

        for (child, entry) in spec.iter().enumerate() {
            let child = child as NodeId;
            let Some(&(node, label)) = entry.as_ref() else {
                roots.push(child);
                continue;
            };
            if node >= num_nodes {
                return Err(TernaryTreeError::NodeOutOfRange { node, num_nodes });
            }
            if node == child {
                return Err(TernaryTreeError::SelfParent { node });
            }
            let idx = node as usize;
            if assigned[idx][label.slot()] {
                return Err(TernaryTreeError::DuplicateLabel { node, label });
            }
            assigned[idx][label.slot()] = true;
            num_children[idx] += 1;
            // A node has three slots, so `DuplicateLabel` already bounds the count at 3; this stays
            // as a guard for `try_from_children`, which counts children before it has labels.
            if num_children[idx] > 3 {
                return Err(TernaryTreeError::TooManyChildren {
                    node,
                    num_children: num_children[idx],
                });
            }
            slots[idx][label.slot()] = Slot::Edge(child);
            parent[child as usize] = Some((node, label));
        }

        if roots.len() != 1 {
            return Err(TernaryTreeError::NotATree {
                num_roots: roots.len(),
            });
        }
        let root = roots[0];

        // Every unassigned link becomes a leg. Numbering them here, by ascending node and within a
        // node in `PauliLabel::slot` order, is what makes `LegId` a function of the tree alone.
        let mut legs = Vec::new();
        for (node, node_assigned) in assigned.iter().enumerate() {
            for label in PauliLabel::ALL {
                if !node_assigned[label.slot()] {
                    slots[node][label.slot()] = Slot::Leg(legs.len() as LegId);
                    legs.push((node as NodeId, label));
                }
            }
        }

        let tree = Self {
            root,
            slots,
            parent,
            legs,
        };
        tree.check_connected()?;
        Ok(tree)
    }

    /// Builds a tree from a child specification.
    ///
    /// `children[u][k]` is node `u`'s child along the label with [`PauliLabel::slot`] `k`, if any. The
    /// dual of [`try_new`](Self::try_new), offered because a top-down description is the natural one
    /// when a tree is generated rather than transcribed.
    pub fn try_from_children(
        children: &[[Option<NodeId>; 3]],
        root: NodeId,
    ) -> Result<Self, TernaryTreeError> {
        let num_nodes = Self::check_size(children.len())?;
        if root >= num_nodes {
            return Err(TernaryTreeError::NodeOutOfRange {
                node: root,
                num_nodes,
            });
        }

        let mut spec: Vec<Option<(NodeId, PauliLabel)>> = vec![None; children.len()];
        // Tracks whether a node has already been claimed, so that a node named as the child of two
        // different parents is reported rather than silently reparented. `try_new` sees only the
        // surviving entry and would call the result disconnected instead.
        let mut claimed = vec![false; children.len()];
        for (node, links) in children.iter().enumerate() {
            for (label, child) in PauliLabel::iter().zip(links) {
                let Some(&child) = child.as_ref() else {
                    continue;
                };
                if child >= num_nodes {
                    return Err(TernaryTreeError::NodeOutOfRange {
                        node: child,
                        num_nodes,
                    });
                }
                if child == root || claimed[child as usize] {
                    // Two parents for one node, or a child pointing at the root: either way the
                    // specification is not a tree rooted where it claims.
                    return Err(TernaryTreeError::NotATree {
                        num_roots: if child == root { 0 } else { 2 },
                    });
                }
                claimed[child as usize] = true;
                spec[child as usize] = Some((node as NodeId, label));
            }
        }

        Self::try_new(&spec)
    }

    /// Builds a linear chain descending along `along`.
    ///
    /// Node `u+1` hangs off node `u` via its `along` link; every other link of every node becomes a
    /// leg. `along = Z` is the Jordan-Wigner encoding and `along = X` the parity encoding. Both have
    /// Pauli weight `num_nodes`, the worst a tree on that many nodes can do -- a balanced tree is
    /// exponentially better (see [`breadth_first`](Self::breadth_first)).
    pub fn chain(num_nodes: u32, along: PauliLabel) -> Result<Self, TernaryTreeError> {
        Self::check_size(num_nodes as usize)?;
        let mut spec: Vec<Option<(NodeId, PauliLabel)>> = Vec::with_capacity(num_nodes as usize);
        spec.push(None);
        for node in 1..num_nodes {
            spec.push(Some((node - 1, along)));
        }
        Self::try_new(&spec)
    }

    /// Builds a tree by breadth-first filling.
    ///
    /// Attaches up to `branching` children to each node in turn, using the first `branching` labels of
    /// `order`. `(3, [X, Y, Z])` gives the balanced tree of Ref. 2, whose Pauli weight
    /// `ceil(log3(2n+1))` is optimal over *all* fermion-to-qubit mappings; `(2, [Z, X, Y])` gives a
    /// binary-branching tree of weight `floor(log2 n) + 1`.
    ///
    /// # Note
    ///
    /// The `(2, ...)` tree is *not* the Bravyi-Kitaev encoding, even though it matches its weight.
    /// Bravyi-Kitaev's tree is the irregular partial-sum shape, which a breadth-first fill reproduces
    /// only when `n` is a power of two; see the [type documentation](Self) for how to build it.
    ///
    /// `order` changes which Pauli strings the encoding produces, so it is part of the encoding's
    /// definition rather than an implementation detail; it does not, however, change the weight
    /// scaling.
    pub fn breadth_first(
        num_nodes: u32,
        branching: u8,
        order: [PauliLabel; 3],
    ) -> Result<Self, TernaryTreeError> {
        Self::check_size(num_nodes as usize)?;
        if branching == 0 && num_nodes > 1 {
            // Nothing could be attached, so every node but the root would be unreachable. Report the
            // cause rather than letting the connectivity check describe the symptom.
            return Err(TernaryTreeError::Disconnected { node: 1, root: 0 });
        }
        if branching > 3 {
            // Rejected rather than clamped: the docs present 3 as a limit, so silently returning the
            // `branching = 3` tree would answer a question the caller did not ask.
            return Err(TernaryTreeError::TooManyChildren {
                node: 0,
                num_children: branching as usize,
            });
        }
        let branching = branching as usize;
        if let Some(dup) = (0..branching).find(|&i| order[..i].contains(&order[i])) {
            return Err(TernaryTreeError::DuplicateLabel {
                node: 0,
                label: order[dup],
            });
        }

        let mut spec: Vec<Option<(NodeId, PauliLabel)>> = vec![None; num_nodes as usize];
        let mut next: u32 = 1;
        'outer: for node in 0..num_nodes {
            for &label in &order[..branching] {
                if next >= num_nodes {
                    break 'outer;
                }
                spec[next as usize] = Some((node, label));
                next += 1;
            }
        }
        Self::try_new(&spec)
    }

    /// Validates a node count and returns it as a `u32`.
    fn check_size(len: usize) -> Result<u32, TernaryTreeError> {
        if len == 0 {
            return Err(TernaryTreeError::Empty);
        }
        if len > MAX_NODES as usize {
            return Err(TernaryTreeError::TooLarge {
                num_nodes: len.try_into().unwrap_or(u32::MAX),
                max: MAX_NODES,
            });
        }
        Ok(len as u32)
    }

    /// Checks that every node is reachable from the root.
    ///
    /// This is what makes the `2N+1` leg count sound. The slot layout gives `3N` links and the
    /// specification supplies `N-1` edges, so `L = 2N+1` follows *only* if those edges form a
    /// connected tree; a specification naming one root while wiring the rest into a cycle would
    /// otherwise pass unnoticed.
    fn check_connected(&self) -> Result<(), TernaryTreeError> {
        let mut seen = vec![false; self.slots.len()];
        let mut stack = vec![self.root];
        seen[self.root as usize] = true;
        while let Some(node) = stack.pop() {
            for label in PauliLabel::ALL {
                if let Slot::Edge(child) = self.slots[node as usize][label.slot()]
                    && !seen[child as usize]
                {
                    seen[child as usize] = true;
                    stack.push(child);
                }
            }
        }
        match seen.iter().position(|&s| !s) {
            Some(node) => Err(TernaryTreeError::Disconnected {
                node: node as NodeId,
                root: self.root,
            }),
            None => Ok(()),
        }
    }

    /// Returns the number of nodes, which is both the number of qubits and the number of modes.
    #[inline]
    pub fn num_nodes(&self) -> u32 {
        self.slots.len() as u32
    }

    /// Returns the number of legs, always `2 * num_nodes() + 1`.
    ///
    /// This holds for *every* shape, which is the property the whole framework rests on. Each of the
    /// `N` nodes has three links, so there are `3N` of them; a connected tree on `N` nodes has `N-1`
    /// edges; the rest are legs, giving `3N - (N-1) = 2N+1`. A node with fewer children therefore does
    /// not yield fewer strings -- its unused links are legs, and a leg generates a string just as an
    /// edge's descendants do.
    ///
    /// The `2N` of these that pair into Majorana operators, and the one that does not, are determined
    /// by [`TernaryTreeEncoding`](encoding::TernaryTreeEncoding).
    #[inline]
    pub fn num_legs(&self) -> u32 {
        2 * self.num_nodes() + 1
    }

    /// Returns the root node.
    #[inline]
    pub fn root(&self) -> NodeId {
        self.root
    }

    /// Returns `node`'s parent and the label of the link descending to it, or `None` for the root.
    #[inline]
    pub fn parent_of(&self, node: NodeId) -> Option<(NodeId, PauliLabel)> {
        self.parent[node as usize]
    }

    /// Returns what hangs off `node`'s `label` link.
    #[inline]
    pub fn child_of(&self, node: NodeId, label: PauliLabel) -> Slot {
        self.slots[node as usize][label.slot()]
    }

    /// Returns the `(node, label)` whose link terminates in `leg`.
    #[inline]
    pub fn leg(&self, leg: LegId) -> (NodeId, PauliLabel) {
        self.legs[leg as usize]
    }

    /// Iterates over the legs in [`LegId`] order.
    pub fn legs(&self) -> impl ExactSizeIterator<Item = (NodeId, PauliLabel)> + '_ {
        self.legs.iter().copied()
    }

    /// Returns the Pauli string of `leg` as `(qubit, label)` pairs, ordered leaf-to-root.
    ///
    /// The qubits are distinct but *not* sorted: they appear in the order the path visits them, which
    /// follows the tree's depth rather than the node numbering.
    /// [`TernaryTreeEncoding`](encoding::TernaryTreeEncoding) sorts them, because `SparseObservable`
    /// requires each term's qubit indices to be term-wise sorted.
    pub fn leg_path(&self, leg: LegId) -> Vec<(NodeId, PauliLabel)> {
        let (node, label) = self.legs[leg as usize];
        let mut path = Vec::with_capacity(self.leg_depth(leg) as usize);
        path.push((node, label));
        let mut current = node;
        while let Some((parent, parent_label)) = self.parent[current as usize] {
            path.push((parent, parent_label));
            current = parent;
        }
        path
    }

    /// Returns the depth of `leg`, i.e. the Pauli weight of the operator it generates.
    pub fn leg_depth(&self, leg: LegId) -> u32 {
        let (node, _) = self.legs[leg as usize];
        let mut depth = 1;
        let mut current = node;
        while let Some((parent, _)) = self.parent[current as usize] {
            depth += 1;
            current = parent;
        }
        depth
    }

    /// Returns the summed depth of every leg, i.e. the total support of all `2N+1` Pauli strings.
    ///
    /// This is what a compiled encoding's flat buffers have to hold, so it sizes them exactly.
    pub fn total_leg_depth(&self) -> u32 {
        (0..self.num_legs()).map(|leg| self.leg_depth(leg)).sum()
    }

    /// Returns the largest Pauli weight of any single generated operator.
    ///
    /// This is the figure of merit a tree is chosen for. It is the weight of one *Majorana* operator;
    /// a fermionic creation or annihilation operator is a sum of two of them, and so may reach twice
    /// this.
    pub fn max_weight(&self) -> u32 {
        (0..self.num_legs())
            .map(|leg| self.leg_depth(leg))
            .max()
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Jordan-Wigner tree: a chain along `Z`.
    fn jordan_wigner(n: u32) -> TernaryTree {
        TernaryTree::chain(n, PauliLabel::Z).unwrap()
    }

    /// The parity tree: a chain along `X`.
    fn parity(n: u32) -> TernaryTree {
        TernaryTree::chain(n, PauliLabel::X).unwrap()
    }

    /// The balanced (JKMN) tree.
    fn balanced(n: u32) -> TernaryTree {
        TernaryTree::breadth_first(n, 3, PauliLabel::ALL).unwrap()
    }

    /// A binary-branching tree: the same weight as Bravyi-Kitaev, but not that encoding's tree.
    fn binary(n: u32) -> TernaryTree {
        TernaryTree::breadth_first(n, 2, [PauliLabel::Z, PauliLabel::X, PauliLabel::Y]).unwrap()
    }

    /// The four families the literature names, as `(name, tree)` for a given size.
    fn families(n: u32) -> Vec<(&'static str, TernaryTree)> {
        vec![
            ("jordan_wigner", jordan_wigner(n)),
            ("parity", parity(n)),
            ("balanced", balanced(n)),
            ("binary", binary(n)),
        ]
    }

    /// Builds a pseudorandom irregular tree on `n` nodes.
    ///
    /// Each node after the root is attached to a random earlier node that still has a free link, which
    /// produces genuinely lopsided shapes rather than the uniform ones the constructors make. A tiny
    /// xorshift keeps this dependency-free and deterministic.
    fn random_tree(n: u32, seed: u64) -> TernaryTree {
        let mut state = seed | 1;
        let mut next_rand = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        let mut spec: Vec<Option<(NodeId, PauliLabel)>> = vec![None; n as usize];
        // Free links, as (node, label) pairs; the root starts with all three.
        let mut free: Vec<(NodeId, PauliLabel)> = PauliLabel::ALL.iter().map(|&l| (0, l)).collect();
        for child in 1..n {
            let pick = (next_rand() % free.len() as u64) as usize;
            let (node, label) = free.swap_remove(pick);
            spec[child as usize] = Some((node, label));
            free.extend(PauliLabel::ALL.iter().map(|&l| (child, l)));
        }
        TernaryTree::try_new(&spec).unwrap()
    }

    #[test]
    fn test_leg_count_is_2n_plus_1() {
        // The invariant the framework rests on: it holds for every shape, so a mapper never has to
        // ask how balanced its tree is.
        for n in 1..=12 {
            for (name, tree) in families(n) {
                assert_eq!(
                    tree.legs().len() as u32,
                    2 * n + 1,
                    "{name} at n={n} has the wrong number of legs"
                );
                assert_eq!(tree.num_legs(), 2 * n + 1);
            }
        }

        // ... including shapes no constructor here produces.
        for seed in 0..50u64 {
            let n = 1 + (seed % 12) as u32;
            let tree = random_tree(n, seed + 1);
            assert_eq!(
                tree.legs().len() as u32,
                2 * n + 1,
                "random tree (n={n}, seed={seed}) has the wrong number of legs"
            );
        }
    }

    #[test]
    fn test_legs_overlap_on_exactly_one_qubit() {
        // Anticommutation of the generated operators follows from this and nothing else: two distinct
        // legs overlap non-trivially on exactly one qubit. Asserting the structural property rather
        // than multiplying matrices tests the *reason*, and costs nothing at any size.
        let mut trees: Vec<(String, TernaryTree)> = Vec::new();
        for n in [1, 2, 3, 4, 5, 7, 13] {
            for (name, tree) in families(n) {
                trees.push((format!("{name} n={n}"), tree));
            }
        }
        for seed in 0..50u64 {
            let n = 1 + (seed % 12) as u32;
            trees.push((format!("random seed={seed}"), random_tree(n, seed + 1)));
        }

        for (name, tree) in &trees {
            let paths: Vec<Vec<(NodeId, PauliLabel)>> =
                (0..tree.num_legs()).map(|l| tree.leg_path(l)).collect();
            for (i, left) in paths.iter().enumerate() {
                for right in paths.iter().skip(i + 1) {
                    let overlap = left
                        .iter()
                        .filter(|(qubit, label)| {
                            right.iter().any(|(q, l)| q == qubit && l != label)
                        })
                        .count();
                    assert_eq!(
                        overlap, 1,
                        "{name}: two legs must anticommute on exactly one qubit, found {overlap}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_leg_paths_have_distinct_qubits() {
        // Each string is a product of Paulis on *distinct* qubits, which is what makes it Hermitian
        // and involutory with coefficient +1 -- no phase correction anywhere in the mappers.
        for seed in 0..20u64 {
            let n = 1 + (seed % 10) as u32;
            let tree = random_tree(n, seed + 1);
            for leg in 0..tree.num_legs() {
                let path = tree.leg_path(leg);
                let mut qubits: Vec<NodeId> = path.iter().map(|(q, _)| *q).collect();
                qubits.sort_unstable();
                let len = qubits.len();
                qubits.dedup();
                assert_eq!(qubits.len(), len, "leg {leg} visits a qubit twice");
            }
        }
    }

    #[test]
    fn test_chain_weight_is_linear() {
        // The Jordan-Wigner chain is the worst case, and the contrast that makes the balanced tree
        // worth having.
        for n in 1..=10 {
            assert_eq!(jordan_wigner(n).max_weight(), n, "JW weight at n={n}");
            assert_eq!(parity(n).max_weight(), n, "parity weight at n={n}");
        }
    }

    #[test]
    fn test_balanced_weight_bound() {
        // `ceil(log3(2n+1))`, which Ref. 2's Theorem 1 shows is optimal over all mappings. The
        // complete sizes are the ones that saturate it exactly.
        for (n, expected) in [(1, 1), (4, 2), (13, 3), (40, 4), (121, 5)] {
            assert_eq!(
                balanced(n).max_weight(),
                expected,
                "balanced tree at n={n} should have weight {expected}"
            );
        }

        // In between, the bound still holds.
        for n in 1..=60u32 {
            let bound = (0..).find(|&h| 3u64.pow(h) >= (2 * n as u64 + 1)).unwrap();
            assert!(
                balanced(n).max_weight() <= bound,
                "balanced tree at n={n} exceeded ceil(log3(2n+1))={bound}"
            );
        }
    }

    #[test]
    fn test_binary_branching_weight_is_logarithmic() {
        // Not the Bravyi-Kitaev tree (that one is the irregular Fenwick shape), but the same
        // `floor(log2 n) + 1` weight.
        for (n, expected) in [(4, 3), (7, 3), (13, 4), (40, 6)] {
            assert_eq!(binary(n).max_weight(), expected, "binary tree at n={n}");
        }
    }

    #[test]
    fn test_breadth_first_order_changes_strings_not_weight() {
        // The child-label order is part of the encoding's definition (it changes which strings are
        // produced) but it does not change the weight scaling. Pinned so that neither half of that
        // drifts unnoticed.
        for n in [4u32, 7, 13, 40] {
            let a = TernaryTree::breadth_first(n, 2, PauliLabel::ALL).unwrap();
            let b = binary(n);
            assert_eq!(
                a.max_weight(),
                b.max_weight(),
                "label order must not change the weight at n={n}"
            );
            if n > 2 {
                assert_ne!(a, b, "label order must change the tree at n={n}");
            }
        }
    }

    /// A `branching` above 3 is an error, not silently the `branching = 3` tree.
    #[test]
    fn test_breadth_first_rejects_excess_branching() {
        for branching in [4u8, 5, u8::MAX] {
            assert_eq!(
                TernaryTree::breadth_first(7, branching, PauliLabel::ALL),
                Err(TernaryTreeError::TooManyChildren {
                    node: 0,
                    num_children: branching as usize
                }),
                "branching={branching} must be rejected rather than clamped"
            );
        }
        // Three is still the largest accepted value.
        assert!(TernaryTree::breadth_first(7, 3, PauliLabel::ALL).is_ok());
    }

    #[test]
    fn test_jordan_wigner_tree_shape() {
        // The chain along Z is the encoding this crate already implements directly: the leg on node
        // `j`'s X link is reached through `Z` on every node below it. This is what ties the framework
        // to `library::jordan_wigner`'s convention.
        let n = 5;
        let tree = jordan_wigner(n);
        for node in 0..n {
            for label in [PauliLabel::X, PauliLabel::Y] {
                let Slot::Leg(leg) = tree.child_of(node, label) else {
                    panic!("node {node}'s {label:?} link should be a leg in the JW chain");
                };
                let path = tree.leg_path(leg);
                assert_eq!(path[0], (node, label));
                // ... and the rest of the path is `Z` on every strictly smaller node, in order.
                let rest: Vec<(NodeId, PauliLabel)> = path[1..].to_vec();
                let expected: Vec<(NodeId, PauliLabel)> =
                    (0..node).rev().map(|q| (q, PauliLabel::Z)).collect();
                assert_eq!(rest, expected, "JW Z-string below node {node}");
            }
        }
        // The chain's last node keeps its Z link as the one leftover leg.
        assert!(matches!(tree.child_of(n - 1, PauliLabel::Z), Slot::Leg(_)));
    }

    #[test]
    fn test_try_new_round_trips_an_irregular_tree() {
        // The general path, exercised on a shape neither convenience constructor can make: node 0
        // branches three ways, node 1 twice, and the rest hang off in a chain.
        let spec = vec![
            None,
            Some((0, PauliLabel::X)),
            Some((0, PauliLabel::Y)),
            Some((0, PauliLabel::Z)),
            Some((1, PauliLabel::X)),
            Some((1, PauliLabel::Z)),
            Some((4, PauliLabel::Y)),
        ];
        let tree = TernaryTree::try_new(&spec).unwrap();
        assert_eq!(tree.num_nodes(), 7);
        assert_eq!(tree.num_legs(), 15);
        assert_eq!(tree.root(), 0);

        // The specification is recoverable from the tree.
        for (node, entry) in spec.iter().enumerate() {
            assert_eq!(tree.parent_of(node as NodeId), *entry);
        }

        // ... and `try_from_children` describes the same tree.
        let mut children = vec![[None; 3]; 7];
        for (node, entry) in spec.iter().enumerate() {
            if let Some((parent, label)) = entry {
                children[*parent as usize][label.slot()] = Some(node as NodeId);
            }
        }
        assert_eq!(TernaryTree::try_from_children(&children, 0).unwrap(), tree);
    }

    #[test]
    fn test_mixed_branching_tree() {
        // The case that motivates the framework, and the one neither `chain` nor `breadth_first` can
        // produce: a tree whose branching is uneven, as a spanning tree of hardware connectivity
        // would be.
        //
        // Its branching profile is genuinely mixed (four leaves, sixteen nodes with one child, one
        // with two and one with three), so it exercises the leg bookkeeping in a way the uniform
        // families cannot. That, and the `2N+1` leg count holding regardless, is all this asserts.
        //
        // The tree is hand-built rather than computed, and deliberately so: it is a *structural*
        // fixture, not the output of any algorithm. Do not read a provenance into it. It is in
        // particular not what the Bonsai algorithm would produce: at node 4 the `Z` link descends
        // to the shallower arm (subtree depth 3) while `X` takes the deeper one (depth 4), which no
        // longest-path-first labelling rule would do, and no parent graph for it exists in this
        // repository.
        let z = PauliLabel::Z;
        let x = PauliLabel::X;
        let y = PauliLabel::Y;
        let spec = vec![
            Some((1, z)),  // 0
            Some((2, z)),  // 1
            Some((3, z)),  // 2
            None,          // 3  <- root
            Some((3, x)),  // 4
            Some((4, z)),  // 5
            Some((5, z)),  // 6
            Some((6, z)),  // 7
            Some((7, z)),  // 8
            Some((10, z)), // 9
            Some((11, z)), // 10
            Some((0, z)),  // 11
            Some((3, y)),  // 12
            Some((12, z)), // 13
            Some((13, z)), // 14
            Some((14, z)), // 15
            Some((15, z)), // 16
            Some((18, z)), // 17
            Some((19, z)), // 18
            Some((20, z)), // 19
            Some((21, z)), // 20
            Some((4, x)),  // 21
        ];
        let tree = TernaryTree::try_new(&spec).unwrap();

        assert_eq!(tree.num_nodes(), 22);
        assert_eq!(
            tree.num_legs(),
            45,
            "2N+1 must hold for an irregular tree too"
        );
        assert_eq!(tree.root(), 3);

        // The branching really is mixed, or this test would prove nothing beyond the chain case.
        let mut profile = [0usize; 4];
        for node in 0..tree.num_nodes() {
            let children = PauliLabel::ALL
                .iter()
                .filter(|&&l| matches!(tree.child_of(node, l), Slot::Edge(_)))
                .count();
            profile[children] += 1;
        }
        assert_eq!(
            profile,
            [4, 16, 1, 1],
            "expected the measured Bonsai branching profile"
        );

        // Weights are far below the chain's, which is the point of growing such a tree.
        assert!(
            tree.max_weight() <= 7,
            "expected weight <= 7, got {}",
            tree.max_weight()
        );
    }

    #[test]
    fn test_rejects_empty() {
        assert_eq!(TernaryTree::try_new(&[]), Err(TernaryTreeError::Empty));
        assert_eq!(
            TernaryTree::chain(0, PauliLabel::Z),
            Err(TernaryTreeError::Empty)
        );
    }

    #[test]
    fn test_rejects_duplicate_label() {
        let spec = vec![None, Some((0, PauliLabel::X)), Some((0, PauliLabel::X))];
        assert_eq!(
            TernaryTree::try_new(&spec),
            Err(TernaryTreeError::DuplicateLabel {
                node: 0,
                label: PauliLabel::X
            })
        );
    }

    #[test]
    fn test_rejects_out_of_range_node() {
        let spec = vec![None, Some((7, PauliLabel::X))];
        assert_eq!(
            TernaryTree::try_new(&spec),
            Err(TernaryTreeError::NodeOutOfRange {
                node: 7,
                num_nodes: 2
            })
        );
        let children = vec![[Some(5), None, None], [None; 3]];
        assert_eq!(
            TernaryTree::try_from_children(&children, 0),
            Err(TernaryTreeError::NodeOutOfRange {
                node: 5,
                num_nodes: 2
            })
        );
    }

    #[test]
    fn test_rejects_self_parent() {
        let spec = vec![None, Some((1, PauliLabel::X))];
        assert_eq!(
            TernaryTree::try_new(&spec),
            Err(TernaryTreeError::SelfParent { node: 1 })
        );
    }

    #[test]
    fn test_rejects_wrong_root_count() {
        // Two roots ...
        let spec = vec![None, None];
        assert_eq!(
            TernaryTree::try_new(&spec),
            Err(TernaryTreeError::NotATree { num_roots: 2 })
        );
        // ... and none at all, which is a cycle.
        let spec = vec![Some((1, PauliLabel::X)), Some((0, PauliLabel::X))];
        assert_eq!(
            TernaryTree::try_new(&spec),
            Err(TernaryTreeError::NotATree { num_roots: 0 })
        );
    }

    #[test]
    fn test_rejects_disconnected() {
        // A lone root plus a two-node cycle: exactly one node has no parent, so the root count is
        // right, yet the nodes in the cycle cannot be reached. This is the case the `|E| = N-1`
        // arithmetic alone would let through, and why connectivity is checked.
        let spec = vec![None, Some((2, PauliLabel::X)), Some((1, PauliLabel::X))];
        assert!(matches!(
            TernaryTree::try_new(&spec),
            Err(TernaryTreeError::Disconnected { .. })
        ));
    }

    #[test]
    fn test_rejects_two_parents_for_one_node() {
        let children = vec![[Some(2), None, None], [Some(2), None, None], [None; 3]];
        assert_eq!(
            TernaryTree::try_from_children(&children, 0),
            Err(TernaryTreeError::NotATree { num_roots: 2 })
        );
    }

    #[test]
    fn test_single_node_tree() {
        // The degenerate but legal case: one qubit, three legs, two of which pair into the mode's
        // Majorana operators.
        let tree = jordan_wigner(1);
        assert_eq!(tree.num_nodes(), 1);
        assert_eq!(tree.num_legs(), 3);
        assert_eq!(tree.max_weight(), 1);
        for label in PauliLabel::ALL {
            assert!(matches!(tree.child_of(0, label), Slot::Leg(_)));
        }
    }
}
