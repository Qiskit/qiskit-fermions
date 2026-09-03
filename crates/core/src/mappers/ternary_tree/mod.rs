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
//! familiar encoding is one particular tree: Jordan-Wigner is a chain, the balanced tree achieves
//! the optimal Pauli weight, and everything in between is reachable. This module holds the tree and
//! the [`encoding`] it compiles into; the mappers that consume them live in
//! [`library::ternary_tree`](crate::mappers::library::ternary_tree).
//!
//! Building a tree, pairing its legs into Majorana operators and checking its invariants are all
//! pure index arithmetic, so future tree-search code (Bonsai, HATT, Treespilation) can work on
//! trees without building an observable; the only method here that produces one is
//! [`TernaryTreeEncoding::total_parity`](encoding::TernaryTreeEncoding::total_parity).
//!
//! # References
//!
//! 1. A. Miller, Z. Zimborás, S. Knecht, S. Maniscalco and G. García-Pérez, *Bonsai algorithm:
//!    grow your own fermion-to-qubit mappings*, PRX Quantum **4**, 030314 (2023),
//!    [arXiv:2212.09731](https://arxiv.org/abs/2212.09731).
//! 2. Z. Jiang, A. Kalev, W. Mruczkiewicz and H. Neven, *Optimal fermion-to-qubit mapping via
//!    ternary trees with applications to reduced quantum states learning*, Quantum **4**, 276
//!    (2020), [arXiv:1910.10746](https://arxiv.org/abs/1910.10746).

use std::hash::{Hash, Hasher};
use thiserror::Error;

pub mod encoding;

/// Index of a node in a [`TernaryTree`].
///
/// Nodes are qubits: node `u` is qubit `u`. This is a `u32` to match the mode indices of the
/// operator types, which is the same index space once a [`ModeMap`](encoding::ModeMap) is applied.
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
/// label. Both the size and each individual discriminant are asserted at compile time, beside the
/// code that reinterprets one as the other.
///
/// `0b00` would be the natural encoding of the identity, which is why no `I` variant is named: a
/// link always carries a non-trivial Pauli, and identity on a qubit is that qubit's absence from a
/// leg's sparse Pauli string. Qiskit omits it from `QkBitTerm` for the same reason.
///
/// The discriminants deliberately do *not* agree with [`Self::slot`], which orders the labels `X`,
/// `Y`, `Z` rather than `Z`, `X`, `Y`. The two answer to different authorities and neither can
/// yield: the discriminants are a wire format fixed by Qiskit's FFI, while the slot order is the
/// sequence Algorithm 1 of Ref. 1 pairs the links in, and it fixes [`LegId`] numbering. So `slot`
/// is written as an explicit `match` rather than arithmetic on the discriminant -- deriving one
/// from the other would weld the two requirements together, and changing either would silently
/// reorder the other. Nothing sorts or compares a `PauliLabel`, so the discriminant order carries
/// no meaning of its own beyond the numeric values themselves.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PauliLabel {
    /// The `X` Pauli.
    X = 2,
    /// The `Y` Pauli.
    Y = 3,
    /// The `Z` Pauli, along which the pairing rule of Ref. 1 descends. See [`Self::slot`] for
    /// the order of the labels.
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
    /// Rust has no built-in way to enumerate an enum's variants, so this is how a caller outside
    /// the crate walks a node's three links: pair it with [`Self::slot`] to index into a
    /// `[Slot; 3]`.
    #[inline]
    pub fn iter() -> impl ExactSizeIterator<Item = PauliLabel> {
        Self::ALL.into_iter()
    }

    /// Returns the index of this label within a node's `[Slot; 3]`.
    ///
    /// The slots are ordered `X`, `Y`, `Z`, independently of the discriminants (see the type's
    /// documentation).
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

/// Recovers a label from its discriminant, rejecting every byte that is not one.
///
/// This is how a label read from foreign memory becomes a [`PauliLabel`]: a caller across the C API
/// can hand over any byte, and every value but these three is an invalid `PauliLabel`, so it must
/// be checked as a plain `u8` before it becomes one. The rejected byte is returned as the error.
impl TryFrom<u8> for PauliLabel {
    type Error = u8;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            v if v == PauliLabel::X as u8 => Ok(PauliLabel::X),
            v if v == PauliLabel::Y as u8 => Ok(PauliLabel::Y),
            v if v == PauliLabel::Z as u8 => Ok(PauliLabel::Z),
            other => Err(other),
        }
    }
}

/// [`PauliLabel::iter`] and [`PauliLabel::slot`] must agree: `iter` yields the labels in slot
/// order, and `slot` names the index of each. A caller that zips a node's three links against
/// `iter` would otherwise mislabel every edge while the tree still satisfied every structural
/// invariant, so nothing else would catch it. The assertion keeps the two orderings the enum
/// encodes (the order `iter` yields and the order `slot` indexes) from drifting apart.
const _: () = {
    assert!(PauliLabel::ALL[PauliLabel::X.slot()] as u8 == PauliLabel::X as u8);
    assert!(PauliLabel::ALL[PauliLabel::Y.slot()] as u8 == PauliLabel::Y as u8);
    assert!(PauliLabel::ALL[PauliLabel::Z.slot()] as u8 == PauliLabel::Z as u8);
};

/// What hangs off one of a node's three downward links.
///
/// Every node has exactly three, which is what makes the leg count independent of the tree's shape:
/// a node with fewer children does not have fewer links, it has more *legs*. See
/// [`TernaryTree::num_legs`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Slot {
    /// The link descends to a child node.
    Edge(NodeId),
    /// The link terminates. It still generates a Pauli string.
    Leg(LegId),
}

/// Errors from working with a [`TernaryTree`].
///
/// Kept separate from [`CoherenceError`](crate::operators::CoherenceError), which describes the
/// coherence of *operator* index buffers: a tree is a different kind of object, and folding these
/// in would give that enum variants no operator can produce.
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
    /// [`TernaryTree::breadth_first`] was asked for a `branching` outside `1..=3`.
    ///
    /// Reported against the argument rather than a node, as is [`Self::DuplicateOrderLabel`]:
    /// `branching` describes the tree as a whole, so there is no node to blame.
    #[error("branching must be between 1 and 3, got {branching}")]
    InvalidBranching {
        /// The rejected value.
        branching: u8,
    },
    /// [`TernaryTree::breadth_first`] was given an `order` that repeats a label it uses.
    ///
    /// Only the first `branching` entries of `order` name child links, so a repeat beyond
    /// them is harmless and not reported. Like [`Self::InvalidBranching`], this is reported
    /// against the argument rather than a node.
    #[error("order repeats the label {label:?} among its first {branching} entries")]
    DuplicateOrderLabel {
        /// The label given twice.
        label: PauliLabel,
        /// How many leading entries of `order` are in use.
        branching: u8,
    },
    /// The specification does not describe a single connected tree.
    ///
    /// Raised when some node cannot be reached from the root. With exactly one root, every other
    /// node has a parent, so this can only happen when a node sits on a cycle detached from the
    /// root. A leg below such a node has no path to the root, so its Pauli string is undefined.
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
    /// A zero-mode encoding is degenerate rather than useful, and `2N+1 = 1` would leave a single
    /// leg with nothing to pair it with.
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
}

/// The largest node count whose leg indices (`2N+1`) still fit in a [`LegId`].
pub const MAX_NODES: u32 = (u32::MAX - 1) / 2;

/// A rooted ternary tree defining a fermion-to-qubit encoding.
///
/// Each node is a qubit and carries three downward **links**, labelled `X`, `Y` and `Z`. A link
/// either descends to a child node (an *edge*) or terminates (a *leg*). Each leg's upward path to
/// the root spells a Pauli string: crossing the link labelled `P` that descends *from* node `u`
/// contributes `P` on qubit `u`, with the identity everywhere else.
///
/// # Validity is structural
///
/// Any rooted, connected tree of this shape yields a valid set of `2N+1` Hermitian, involutory,
/// mutually anticommuting operators -- whatever its shape, whatever the assignment of qubits to
/// nodes, and whatever permutation of `{X,Y,Z}` each node uses. Anticommutation follows from the
/// first-common-ancestor argument: two distinct legs' paths agree above their deepest shared node,
/// carry two *different* labels at it, and at least one acts trivially below, so the strings
/// overlap non-trivially on exactly one qubit. There is consequently nothing to verify at runtime
/// beyond the structural invariants enforced when a tree is constructed, and the mappers never
/// check anticommutation.
///
/// # Constructing one
///
/// [`Self::try_new`] builds an arbitrary tree; it is the general interface, and the shapes the
/// literature names are all reachable through it. [`Self::chain`] and [`Self::breadth_first`] are
/// conveniences for the two uniform families:
///
/// ```text
/// Encoding            Constructor                        Pauli weight
/// ------------------  ---------------------------------  --------------------------
/// Jordan-Wigner       chain(n, PauliLabel::Z)            n
/// Parity              chain(n, PauliLabel::X)            n
/// Balanced (Ref. 2)   breadth_first(n, 3, [X, Y, Z])     ceil(log3(2n+1)), optimal
/// Binary-branching    breadth_first(n, 2, [Z, X, Y])     floor(log2 n) + 1
/// ```
///
/// Those conveniences cover the *uniform* families only. The trees that motivate the framework are
/// not uniform: the Bonsai algorithm grows a spanning tree of a hardware coupling graph, so its
/// branching follows the device's connectivity. Those are built with [`Self::try_new`], which is
/// why that is the general interface and the conveniences are only conveniences.
///
/// # Bravyi-Kitaev
///
/// Bravyi-Kitaev is in this family but has no constructor here, its tree being irregular. The
/// worked example "a Bravyi-Kitaev tree" in the mappers guide builds it through
/// [`Self::try_new`].
#[derive(Clone, Debug)]
pub struct TernaryTree {
    /// The one node with no parent; every leg's path terminates here.
    root: NodeId,
    /// One `[Slot; 3]` per node, indexed by [`PauliLabel::slot`].
    slots: Vec<[Slot; 3]>,
    /// The parent of each node and the label of the link descending to it; `None` for the root.
    ///
    /// The inverse of the [`Slot::Edge`] entries, kept so that walking a leg to the root is
    /// `O(depth)` with no downward scan.
    parent: Vec<Option<(NodeId, PauliLabel)>>,
    /// The `(node, label)` whose link terminates in each leg, indexed by [`LegId`].
    ///
    /// The inverse of the [`Slot::Leg`] entries, kept so that enumerating legs is a slice scan
    /// rather than a filter over all `3N` slots.
    legs: Vec<(NodeId, PauliLabel)>,
    /// The number of nodes on each node's path to the root, the root included.
    ///
    /// This is the Pauli weight of every leg hanging off that node, so caching it makes
    /// [`Self::leg_depth`], [`Self::max_weight`] and [`Self::total_leg_depth`] lookups rather than
    /// walks: without it they cost `O(N * depth)`, which is quadratic on a chain.
    ///
    /// Derived from `parent`: a node's depth is one more than its parent's, and the root's is `1`.
    depth: Vec<u32>,
}

/// Compares `slots` alone.
///
/// The slots determine every other field: the root is the one node that no slot names as a
/// child, the parent and leg tables are their inverses, and the depths follow from the parents.
/// The fields are private and only the constructors set them, so two trees are equal exactly
/// when their slots are, and comparing the derived tables as well would only repeat the work.
impl PartialEq for TernaryTree {
    fn eq(&self, other: &Self) -> bool {
        self.slots == other.slots
    }
}

impl Eq for TernaryTree {}

/// Hashes `slots` alone, for the reason given for the [`PartialEq`] implementation, so equal
/// trees hash equally.
impl Hash for TernaryTree {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.slots.hash(state);
    }
}

impl TernaryTree {
    /// Builds a tree from a parent-and-label specification.
    ///
    /// `spec[u]` gives node `u`'s parent and the label of the link descending to it, or `None` if
    /// `u` is the root. Exactly one entry must be `None`. Every link not named by some entry
    /// becomes a leg. Legs are numbered by ascending node, and within a node in
    /// [`PauliLabel::slot`] order (`X`, `Y`, `Z`), which fixes [`LegId`] deterministically for a
    /// given tree.
    ///
    /// This is the general constructor: any ternary tree is expressible as such a specification,
    /// including the irregular, connectivity-derived trees that the Bonsai algorithm produces.
    pub fn try_new(spec: &[Option<(NodeId, PauliLabel)>]) -> Result<Self, TernaryTreeError> {
        let num_nodes = Self::check_size(spec.len())?;

        let mut slots = vec![[Slot::Leg(0); 3]; spec.len()];
        let mut assigned = vec![[false; 3]; spec.len()];
        let mut parent = vec![None; spec.len()];
        let mut roots = Vec::new();

        for (child, entry) in (0..num_nodes).zip(spec) {
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

        // No placeholder survives: every leg slot names a leg that names it back.
        debug_assert!(slots.iter().enumerate().all(|(node, node_slots)| {
            PauliLabel::ALL
                .into_iter()
                .all(|label| match node_slots[label.slot()] {
                    Slot::Leg(leg) => legs[leg as usize] == (node as NodeId, label),
                    Slot::Edge(_) => true,
                })
        }));

        let depth = Self::node_depths(root, &slots)?;
        Ok(Self {
            root,
            slots,
            parent,
            legs,
            depth,
        })
    }

    /// Builds a linear chain descending along `along`.
    ///
    /// Node `u+1` hangs off node `u` via its `along` link; every other link of every node becomes a
    /// leg. `along = Z` is the Jordan-Wigner encoding and `along = X` the parity encoding. Both
    /// have Pauli weight `num_nodes`, the worst a tree on that many nodes can do -- a balanced tree
    /// is exponentially better (see [`breadth_first`](Self::breadth_first)).
    ///
    /// This is only a convenience: it is [`breadth_first`](Self::breadth_first) with a branching of
    /// one, whose `order` then needs just the one label.
    pub fn chain(num_nodes: u32, along: PauliLabel) -> Result<Self, TernaryTreeError> {
        // One child per node is a chain. Only the first label of `order` is read at a branching
        // of one, so repeating `along` for the rest is accepted.
        Self::breadth_first(num_nodes, 1, [along; 3])
    }

    /// Builds a tree by breadth-first filling.
    ///
    /// Attaches up to `branching` children to each node in turn, using the first `branching` labels
    /// of `order`. `(3, [X, Y, Z])` gives the balanced tree of Ref. 2, whose Pauli weight
    /// `ceil(log3(2n+1))` is optimal over *all* fermion-to-qubit mappings; `(2, [Z, X, Y])` gives a
    /// binary-branching tree of weight `floor(log2 n) + 1`.
    ///
    /// # Note
    ///
    /// The `(2, ...)` tree is *not* the Bravyi-Kitaev encoding, even though it matches its weight.
    /// Bravyi-Kitaev's tree is the irregular partial-sum shape, which no breadth-first fill
    /// reproduces beyond a single node, whatever the `order`; the mappers guide shows how to build
    /// it in the worked example "a Bravyi-Kitaev tree".
    ///
    /// `order` changes which Pauli strings the encoding produces, so it is part of the encoding's
    /// definition rather than an implementation detail; it does not, however, change the weight
    /// scaling.
    pub fn breadth_first(
        num_nodes: u32,
        branching: u8,
        order: [PauliLabel; 3],
    ) -> Result<Self, TernaryTreeError> {
        // Reject before allocating: a huge node count must not reach the `Vec` below.
        Self::check_size(num_nodes as usize)?;
        // Above 3 is rejected rather than clamped: the docs present 3 as a limit, so silently
        // returning the `branching = 3` tree would answer a question the caller did not ask. Zero
        // is rejected even for a single node, where it happens to build the one-node tree, so that
        // the valid range does not depend on `num_nodes`.
        if !(1..=3).contains(&branching) {
            return Err(TernaryTreeError::InvalidBranching { branching });
        }
        if let Some(dup) = (0..branching as usize).find(|&i| order[..i].contains(&order[i])) {
            return Err(TernaryTreeError::DuplicateOrderLabel {
                label: order[dup],
                branching,
            });
        }

        let mut spec: Vec<Option<(NodeId, PauliLabel)>> = vec![None; num_nodes as usize];
        let mut next: u32 = 1;
        'outer: for node in 0..num_nodes {
            for &label in &order[..branching as usize] {
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

    /// Checks that every node is reachable from `root`, and returns each node's depth.
    ///
    /// With exactly one parentless node, every other node has a parent, so the only way to be
    /// unreachable is to sit on a cycle that never meets the root. Such a specification still has
    /// `2N+1` legs, but the path from a leg on that cycle would never reach the root, so its Pauli
    /// string, its depth and the anticommutation argument would all be undefined.
    ///
    /// The depths come for free from the same walk, which visits every node exactly once. A depth
    /// of `0` marks a node not yet reached, which is unambiguous because the root's depth is `1`.
    fn node_depths(root: NodeId, slots: &[[Slot; 3]]) -> Result<Vec<u32>, TernaryTreeError> {
        let mut depth = vec![0u32; slots.len()];
        let mut stack = vec![root];
        depth[root as usize] = 1;
        while let Some(node) = stack.pop() {
            for label in PauliLabel::ALL {
                if let Slot::Edge(child) = slots[node as usize][label.slot()] {
                    depth[child as usize] = depth[node as usize] + 1;
                    stack.push(child);
                }
            }
        }
        match depth.iter().position(|&d| d == 0) {
            Some(node) => Err(TernaryTreeError::Disconnected {
                node: node as NodeId,
                root,
            }),
            None => Ok(depth),
        }
    }

    /// Returns the number of nodes, which is both the number of qubits and the number of modes.
    #[inline]
    pub fn num_nodes(&self) -> u32 {
        self.slots.len() as u32
    }

    /// Returns the number of legs, always `2 * num_nodes() + 1`.
    ///
    /// This holds for *every* shape, which is the property the whole framework rests on. Each of
    /// the `N` nodes has three links, so there are `3N` of them; a connected tree on `N` nodes has
    /// `N-1` edges; the rest are legs, giving `3N - (N-1) = 2N+1`. A node with fewer children
    /// therefore does not yield fewer strings -- its unused links are legs, and a leg generates a
    /// string just as an edge's descendants do.
    ///
    /// The `2N` of these that pair into Majorana operators, and the one that does not, are
    /// determined by [`TernaryTreeEncoding`](encoding::TernaryTreeEncoding).
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

    /// Returns the Pauli string of `leg` as `(qubit, label)` pairs, ordered leaf-to-root.
    ///
    /// The qubits are distinct but *not* sorted: they appear in the order the path visits them,
    /// which follows the tree's depth rather than the node numbering.
    /// [`TernaryTreeEncoding`](encoding::TernaryTreeEncoding) sorts them, because
    /// `SparseObservable` requires each term's qubit indices to be term-wise sorted.
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
    #[inline]
    pub fn leg_depth(&self, leg: LegId) -> u32 {
        let (node, _) = self.legs[leg as usize];
        self.depth[node as usize]
    }

    /// Returns the summed depth of every leg, i.e. the total support of all `2N+1` Pauli strings.
    ///
    /// A compiled encoding sizes its flat buffers from this, less the leftover leg's string, which
    /// it keeps separately. It is a `usize` because it grows quadratically on a chain and overflows
    /// a `u32` long before [`MAX_NODES`] is reached.
    pub fn total_leg_depth(&self) -> usize {
        (0..self.num_legs())
            .map(|leg| self.leg_depth(leg) as usize)
            .sum()
    }

    /// Returns the largest Pauli weight of any single generated operator.
    ///
    /// This is the figure of merit a tree is chosen for. It is the weight of one *Majorana*
    /// operator, and no Pauli term of a fermionic creation or annihilation operator is heavier,
    /// since that is a sum of two Majorana operators. The two terms generally act on different
    /// qubits, though, so the operator as a whole can be supported on more.
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

    /// The balanced tree of Ref. 2.
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
    /// Each node after the root is attached to a random earlier node that still has a free link,
    /// which produces genuinely lopsided shapes rather than the uniform ones the constructors make.
    /// A tiny xorshift keeps this dependency-free and deterministic.
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
                    tree.legs.len() as u32,
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
                tree.legs.len() as u32,
                2 * n + 1,
                "random tree (n={n}, seed={seed}) has the wrong number of legs"
            );
        }
    }

    #[test]
    fn test_legs_overlap_on_exactly_one_qubit() {
        // Anticommutation of the generated operators follows from this and nothing else: two
        // distinct legs overlap non-trivially on exactly one qubit. Asserting the structural
        // property rather than multiplying matrices tests the *reason*, and costs nothing at any
        // size.
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
    fn test_chain_is_breadth_first_with_a_branching_of_one() {
        // `chain` is documented as a convenience for this, so the equality is pinned here in case
        // its implementation ever stops delegating. Only the first label of `order` is read at a
        // branching of one, so the other two must not matter.
        use PauliLabel::{X, Y, Z};
        let cases = [
            (X, [X, Y, Z]),
            (Y, [Y, X, X]),
            (Z, [Z, Z, Z]),
            (Z, [Z, Y, X]),
        ];
        for n in [1, 2, 7, 20] {
            for (along, order) in cases {
                assert_eq!(
                    TernaryTree::chain(n, along),
                    TernaryTree::breadth_first(n, 1, order),
                    "chain along {along:?} at n={n}"
                );
            }
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
            assert_ne!(a, b, "label order must change the tree at n={n}");
        }
    }

    /// A `branching` outside `1..=3` is an error naming the argument, not silently the
    /// `branching = 3` tree, and not a node-level error pointing at the wrong fix.
    #[test]
    fn test_breadth_first_rejects_invalid_branching() {
        for branching in [0u8, 4, 5, u8::MAX] {
            for num_nodes in [1, 7] {
                assert_eq!(
                    TernaryTree::breadth_first(num_nodes, branching, PauliLabel::ALL),
                    Err(TernaryTreeError::InvalidBranching { branching }),
                    "branching={branching} at n={num_nodes} must be rejected"
                );
            }
        }
        // One and three are the bounds of the accepted range.
        assert!(TernaryTree::breadth_first(7, 1, PauliLabel::ALL).is_ok());
        assert!(TernaryTree::breadth_first(7, 3, PauliLabel::ALL).is_ok());
    }

    #[test]
    fn test_jordan_wigner_tree_shape() {
        // The chain along Z is the encoding this crate already implements directly: the leg on node
        // `j`'s X link is reached through `Z` on every node below it. This is what ties the
        // framework to `library::jordan_wigner`'s convention.
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
    fn test_irregular_tree_recovers_its_specification() {
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
    }

    #[test]
    fn test_mixed_branching_tree() {
        // The case that motivates the framework, and the one neither `chain` nor `breadth_first`
        // can produce: a tree whose branching is uneven, as a spanning tree of hardware
        // connectivity would be.
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
            "expected the fixture's branching profile"
        );

        // Weights are far below the chain's, which is the point of growing such a tree.
        assert!(
            tree.max_weight() <= 7,
            "expected weight <= 7, got {}",
            tree.max_weight()
        );
    }

    #[test]
    fn test_breadth_first_rejects_a_repeated_order_label() {
        use PauliLabel::{X, Y, Z};
        // The repeat is reported against the argument whatever the node count, including
        // the single-node tree that has no children for it to affect.
        for num_nodes in [1, 7] {
            assert_eq!(
                TernaryTree::breadth_first(num_nodes, 2, [X, X, Z]),
                Err(TernaryTreeError::DuplicateOrderLabel {
                    label: X,
                    branching: 2
                })
            );
        }
        assert_eq!(
            TernaryTree::breadth_first(7, 3, [X, Z, Z]),
            Err(TernaryTreeError::DuplicateOrderLabel {
                label: Z,
                branching: 3
            })
        );
        // Entries beyond the first `branching` are never used, so repeating one is fine.
        assert!(TernaryTree::breadth_first(7, 1, [X, X, Z]).is_ok());
        assert!(TernaryTree::breadth_first(7, 2, [X, Y, Y]).is_ok());
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
    fn test_rejects_too_many_nodes() {
        // `MAX_NODES` is about two billion, so no tree that large can be built here. `check_size`
        // is a pure function of the length, though, so the boundary is tested on it directly.
        let max = MAX_NODES as usize;
        assert_eq!(TernaryTree::check_size(max), Ok(MAX_NODES));
        assert_eq!(
            TernaryTree::check_size(max + 1),
            Err(TernaryTreeError::TooLarge {
                num_nodes: MAX_NODES + 1,
                max: MAX_NODES
            })
        );
        assert_eq!(TernaryTree::check_size(0), Err(TernaryTreeError::Empty));
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
    fn test_equality_and_hash_follow_the_slots() {
        // Equality and hashing look at `slots` alone, on the grounds that it determines every
        // other field. Pinned against a comparison of all of them, so that a field which stopped
        // being derived would be noticed here and not as two "equal" trees that behave
        // differently.
        let mut trees: Vec<TernaryTree> = (1..=8)
            .flat_map(|n| families(n).into_iter().map(|(_, tree)| tree))
            .collect();
        trees.extend((0..30u64).map(|seed| random_tree(1 + (seed % 6) as u32, seed + 1)));

        let hash = |tree: &TernaryTree| {
            let mut hasher = std::hash::DefaultHasher::new();
            tree.hash(&mut hasher);
            hasher.finish()
        };
        // Destructured, so that a field added to the tree is a compile error here (and an unused
        // variable, which the lints reject) until it is compared too.
        let same_fields = |a: &TernaryTree, b: &TernaryTree| {
            let TernaryTree {
                root,
                slots,
                parent,
                legs,
                depth,
            } = a;
            *root == b.root
                && *slots == b.slots
                && *parent == b.parent
                && *legs == b.legs
                && *depth == b.depth
        };
        let mut num_equal = 0;
        for a in &trees {
            for b in &trees {
                assert_eq!(a == b, same_fields(a, b));
                if a == b {
                    assert_eq!(hash(a), hash(b));
                    num_equal += 1;
                }
            }
        }
        // Besides each tree equalling itself, the single-node trees of the four families are the
        // same tree, so some distinct pairs really are equal.
        assert!(num_equal > trees.len());
    }

    #[test]
    fn test_cached_depths_match_the_paths() {
        // The depth cache stands in for walking each leg to the root, so it must agree with that
        // walk on every shape, the irregular ones included.
        for seed in 0..30u64 {
            let n = 1 + (seed % 12) as u32;
            let tree = random_tree(n, seed + 1);
            for leg in 0..tree.num_legs() {
                assert_eq!(tree.leg_depth(leg) as usize, tree.leg_path(leg).len());
            }
            let walked: usize = (0..tree.num_legs()).map(|l| tree.leg_path(l).len()).sum();
            assert_eq!(tree.total_leg_depth(), walked);
        }
    }

    #[test]
    fn test_total_leg_depth_does_not_overflow_u32() {
        // A chain's summed depth is roughly `N^2`, which passes `u32::MAX` at about 66,000 nodes,
        // far below `MAX_NODES`. The exact sum is `N(N+1)` over the paired legs plus `N` for the
        // leftover.
        let n: u32 = 70_000;
        let tree = TernaryTree::chain(n, PauliLabel::Z).unwrap();
        let expected = n as usize * (n as usize + 1) + n as usize;
        assert!(expected > u32::MAX as usize);
        assert_eq!(tree.total_leg_depth(), expected);
        assert_eq!(tree.max_weight(), n);
    }

    #[test]
    fn test_label_round_trips_through_its_discriminant() {
        for label in PauliLabel::ALL {
            assert_eq!(PauliLabel::try_from(label as u8), Ok(label));
        }
        // `0` is the identity's natural encoding and the value a zero-initialized C struct holds,
        // so it is the likeliest invalid byte to arrive.
        for byte in [0u8, 4, u8::MAX] {
            assert_eq!(PauliLabel::try_from(byte), Err(byte));
        }
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
