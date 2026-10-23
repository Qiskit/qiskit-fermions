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

//! Compiling a ternary tree into the Pauli images of the Majorana operators.

use super::{LegId, NodeId, PauliLabel, Slot, TernaryTree};
use crate::mappers::qk_obs::{QkComplex64, as_bit_terms, ffi_backend as ffi, one_pauli_string};
use std::hash::{Hash, Hasher};
use thiserror::Error;

/// How fermionic modes are assigned to the nodes of a tree.
///
/// A tree says which Pauli strings exist; this says which mode's Majorana operators each pair of
/// them represents. The two are separate because every tree-search method varies one while holding
/// the other fixed -- reordering the modes of a fixed tree is one of the cheapest ways to lower the
/// weight of a particular Hamiltonian.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModeMap {
    /// Mode `j` is paired at node `j`.
    Identity,
    /// Entry `u` is the mode paired at node `u`; the entries are a permutation of
    /// `0..num_nodes`.
    Permutation(Vec<u32>),
}

impl ModeMap {
    /// Returns the mode paired at `node`.
    #[inline]
    fn mode_of(&self, node: NodeId) -> u32 {
        match self {
            ModeMap::Identity => node,
            ModeMap::Permutation(modes) => modes[node as usize],
        }
    }

    /// Checks that this map is a permutation of `0..num_nodes`.
    fn validate(&self, num_nodes: u32) -> Result<(), TernaryTreeEncodingError> {
        let ModeMap::Permutation(modes) = self else {
            return Ok(());
        };
        if modes.len() != num_nodes as usize {
            return Err(TernaryTreeEncodingError::ModeMapLength {
                len: modes.len(),
                num_nodes,
            });
        }
        let mut seen = vec![false; num_nodes as usize];
        for &mode in modes {
            if mode >= num_nodes {
                return Err(TernaryTreeEncodingError::ModeOutOfRange { mode, num_nodes });
            }
            if seen[mode as usize] {
                return Err(TernaryTreeEncodingError::ModeMapNotAPermutation { mode });
            }
            seen[mode as usize] = true;
        }
        Ok(())
    }
}

/// Errors from working with a [`TernaryTreeEncoding`].
///
/// Kept separate from [`TernaryTreeError`](super::TernaryTreeError), which covers the tree
/// itself: these concern the [`ModeMap`] a tree is compiled under and the register an encoding
/// is used on, and are only reachable once a valid tree exists.
#[derive(Error, Debug, PartialEq, Eq)]
pub enum TernaryTreeEncodingError {
    /// A [`ModeMap`] has the wrong length.
    #[error("the mode map has {len} entries, but the tree has {num_nodes} nodes")]
    ModeMapLength {
        /// The length given.
        len: usize,
        /// The tree's node count.
        num_nodes: u32,
    },
    /// A [`ModeMap`] names the same mode twice.
    #[error("the mode map is not a permutation: mode {mode} appears more than once")]
    ModeMapNotAPermutation {
        /// The repeated mode.
        mode: u32,
    },
    /// A [`ModeMap`] names a mode outside the tree.
    ///
    /// Distinct from [`NodeOutOfRange`](super::TernaryTreeError::NodeOutOfRange): the offending
    /// index is a *mode*, and a mode map is read as node-indexed entries holding mode values, so
    /// reporting it as a node misdirects.
    #[error("the mode map references mode {mode}, but the tree has only {num_nodes} nodes")]
    ModeOutOfRange {
        /// The out-of-range mode.
        mode: u32,
        /// The number of nodes the tree has.
        num_nodes: u32,
    },
    /// The register has fewer qubits than the encoding has modes.
    ///
    /// A ternary-tree encoding spends one qubit per mode, and a mode's image can act on any of the
    /// encoding's qubits, so the register must cover all of them even when the operator being
    /// mapped touches only low modes.
    ///
    /// Distinct from [`Self::ModeOutsideEncoding`], which no register size can fix.
    #[error(
        "num_qubits ({num_qubits}) is too small for an encoding of {num_modes} modes, which needs \
         one qubit per mode"
    )]
    NumQubitsTooSmall {
        /// The register size given.
        num_qubits: u32,
        /// The number of modes, and hence qubits, the encoding needs.
        num_modes: u32,
    },
    /// The operator acts on a mode that the encoding has no Pauli image for.
    ///
    /// Distinct from [`Self::NumQubitsTooSmall`] because no value of `num_qubits` fixes it: the
    /// encoding's extent is fixed when it is built, so it is the encoding that has to grow.
    #[error(
        "the operator acts on mode index {max_mode}, but the encoding covers only {num_modes} modes"
    )]
    ModeOutsideEncoding {
        /// The number of modes the encoding covers.
        num_modes: u32,
        /// The largest mode the operator acts on.
        max_mode: u32,
    },
}

/// A ternary tree compiled into the Pauli strings of its Majorana operators.
///
/// Holds the `2N` Majorana strings in a flat, sorted form, together with the string of each mode's
/// vertex operator and of the total parity. This is the artifact the mappers consume: after it is
/// built, mapping a generator is a slice lookup, or the product of two of them.
///
/// # The pairing
///
/// The tree yields `2N+1` mutually anticommuting strings, one more than the `2N` Majorana operators
/// of `N` modes. Which string is dropped, and which of a pair is `gamma` rather than `gamma'`, is
/// fixed by Algorithm 1 of the Bonsai paper (Ref. 1 in the [module documentation](super)): from
/// node `u`, follow its `X` link and then descend along `Z` links until reaching a leg (that leg
/// carries `gamma` of the mode at `u`); repeat from the `Y` link for `gamma'`. This consumes
/// exactly `2N` legs, and the one left over is the leg reached by descending `Z` from the root.
///
/// The rule is not an arbitrary convention: it is what makes `|0...0>` the fermionic vacuum. Any
/// injective assignment of Majorana operators to legs gives a valid algebra, but only this one
/// preserves the vacuum and makes `i gamma_2j gamma_2j+1` a number operator.
///
/// # The vertex operators
///
/// The pairing also fixes every vertex operator `V_j = -i gamma_2j gamma_2j+1` in closed form,
/// phase included. If mode `j` sits at node `u`, its two strings share every factor above `u` (both
/// legs are reached through `u`), carry `X` and `Y` on `u` itself, and below `u` carry `Z` on two
/// disjoint chains: the `Z`-descents from `u`'s `X` and `Y` children. Multiplying qubit by qubit,
/// the shared factors square to the identity and `X Y = i Z`, so
///
/// ```text
///     V_j  =  Z_u  prod_{w in desc_X(u)} Z_w  prod_{w in desc_Y(u)} Z_w
/// ```
///
/// with coefficient exactly `+1` on *every* tree and under every mode map. Only the support depends
/// on the tree. Each vertex image is therefore a `Z` string, and they all commute.
///
/// # The leftover string
///
/// The unpaired leg is the total fermionic parity `prod_j V_j`. Multiplying the `Z` strings above
/// out, every node off the root's `Z`-descent appears in exactly two of them (its own, and that of
/// the node whose `X` or `Y` link the node's `Z` chain hangs from), while every node on it appears
/// in its own alone. So the parity is the `Z` string on the root's `Z`-descent, again with
/// coefficient `+1`: precisely the string of the leftover leg. Two things about it are easy to get
/// wrong:
///
/// * Its Pauli string is **not** in general the all-`Z` string. It is the leg reached by the
///   all-`Z` *path*, whose links descend from whichever nodes happen to lie on that path, so it
///   carries the identity on every node that does not. On the balanced tree at `N = 4` it is
///   `Z_3 Z_0`.
/// * It is an **observable to measure, not a constraint to impose**. A ternary-tree encoding spends
///   one qubit per mode and is a unitary isomorphism onto the whole `2^N`-dimensional space, so
///   both parity sectors are physical and nothing may be projected away.
#[derive(Clone, Debug)]
pub struct TernaryTreeEncoding {
    /// The tree's node count, which is both the mode count and the qubits the strings span.
    num_modes: u32,
    /// The Pauli image of each Majorana operator, indexed as `2 * mode + is_prime`.
    ///
    /// Stored in Majorana order rather than leg order so that the two strings of a mode sit next to
    /// each other: a fermionic action's image is their two-term sum, which can then be handed to
    /// `qk_obs_new` as one contiguous span with no gathering.
    majoranas: PauliTable,
    /// The Pauli string of each mode's vertex operator `-i gamma_2j gamma_2j+1`, indexed by mode.
    ///
    /// Every one is a `Z` string with coefficient `+1` (see the type-level documentation), so only
    /// the string is stored. Precomputed because vertex operators dominate the edge-vertex and
    /// transfer-vertex Hamiltonians these encodings map, and their images depend on the encoding
    /// alone.
    vertices: PauliTable,
    /// The qubits the total parity acts on, ascending: `Z` on every node of the root's
    /// `Z`-descent, with coefficient `+1`.
    parity: Vec<u32>,
}

/// Compares the Majorana images alone.
///
/// Everything else the encoding stores is a function of them: the mode count is half their
/// number, a mode's vertex operator is the product of its two Majorana operators, and the total
/// parity is the product of the vertex operators. So two encodings are equal exactly when they
/// assign the same Pauli string to every Majorana operator, and comparing the other tables as
/// well would only repeat the work.
impl PartialEq for TernaryTreeEncoding {
    fn eq(&self, other: &Self) -> bool {
        self.majoranas == other.majoranas
    }
}

impl Eq for TernaryTreeEncoding {}

/// Hashes the Majorana images alone, for the reason given for the [`PartialEq`] implementation,
/// so equal encodings hash equally.
impl Hash for TernaryTreeEncoding {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.majoranas.hash(state);
    }
}

/// A list of Pauli strings in flat storage, laid out like a `SparseObservable`'s
/// `{bit_terms, indices, boundaries}` and handed out in that order: the labels first, then the
/// qubits they act on.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct PauliTable {
    /// `labels[boundaries[i]..boundaries[i + 1]]` is the Pauli acting on each qubit of string `i`.
    labels: Vec<PauliLabel>,
    /// The qubit each entry of `labels` acts on, ascending within a string.
    indices: Vec<u32>,
    /// Start of each string's slice into `labels` and `indices`, with a trailing total.
    boundaries: Vec<usize>,
}

impl PauliTable {
    /// Creates an empty table sized for `num_strings` strings of `total_support` entries in all.
    fn with_capacity(num_strings: usize, total_support: usize) -> Self {
        let mut boundaries = Vec::with_capacity(num_strings + 1);
        boundaries.push(0);
        Self {
            labels: Vec::with_capacity(total_support),
            indices: Vec::with_capacity(total_support),
            boundaries,
        }
    }

    /// Appends a string, which must already be in ascending qubit order.
    fn push(&mut self, string: impl IntoIterator<Item = (u32, PauliLabel)>) {
        for (qubit, label) in string {
            self.labels.push(label);
            self.indices.push(qubit);
        }
        self.boundaries.push(self.labels.len());
    }

    /// Returns strings `first..last`, back to back, as parallel label and index slices.
    #[inline]
    fn span(&self, first: usize, last: usize) -> (&[PauliLabel], &[u32]) {
        let range = self.boundaries[first]..self.boundaries[last];
        (&self.labels[range.clone()], &self.indices[range])
    }

    /// Returns string `index` as parallel label and index slices.
    #[inline]
    fn get(&self, index: usize) -> (&[PauliLabel], &[u32]) {
        self.span(index, index + 1)
    }

    /// Returns the number of qubits string `index` acts on.
    #[inline]
    fn weight(&self, index: usize) -> usize {
        self.boundaries[index + 1] - self.boundaries[index]
    }
}

impl TernaryTreeEncoding {
    /// Compiles `tree` under `mode_map` into the Pauli images of its Majorana operators.
    ///
    /// The compiled strings span exactly the tree's nodes. The qubit count of a *mapped operator*
    /// is a property of that mapping rather than of the encoding, so it is passed to the mappers
    /// instead, which pad with the identity exactly as the Jordan-Wigner ones do.
    pub fn new(tree: &TernaryTree, mode_map: &ModeMap) -> Result<Self, TernaryTreeEncodingError> {
        let num_nodes = tree.num_nodes();
        mode_map.validate(num_nodes)?;

        // The total parity and the vertex operators come first, in the closed forms derived in
        // the type-level documentation: each is the `Z` string on a node set read straight off
        // the tree, with coefficient `+1`, so neither needs a product taken. Their sizes fix how
        // large the Majorana table is. `z_descent` collects the nodes a walk along `Z` edges
        // passes through, starting from a link.
        let z_descent = |mut slot: Slot, string: &mut Vec<(u32, PauliLabel)>| {
            while let Slot::Edge(node) = slot {
                string.push((node, PauliLabel::Z));
                slot = tree.child_of(node, PauliLabel::Z);
            }
        };

        let mut string = Vec::new();
        z_descent(Slot::Edge(tree.root()), &mut string);
        let mut parity: Vec<u32> = string.iter().map(|&(qubit, _)| qubit).collect();
        // Every string is sorted by qubit. `SparseObservable` requires each term's indices to be
        // sorted, but nothing checks it on the way in, and `qk_obs_canonicalize` merges duplicates
        // only when they are stored identically, so an unsorted string would silently defeat the
        // memory bound in `qk_obs`.
        parity.sort_unstable();

        // Vertices are stored by mode, so the node serving each mode is needed.
        let mut node_of_mode = vec![0; num_nodes as usize];
        for node in 0..num_nodes {
            node_of_mode[mode_map.mode_of(node) as usize] = node;
        }
        // Every node off the root's `Z`-descent lies in exactly two vertex strings and every
        // node on it in exactly one (see the type-level documentation), so they hold
        // `2N - |parity|` entries between them and the buffers are sized once.
        let mut vertices =
            PauliTable::with_capacity(num_nodes as usize, 2 * num_nodes as usize - parity.len());
        for &node in &node_of_mode {
            string.clear();
            string.push((node, PauliLabel::Z));
            z_descent(tree.child_of(node, PauliLabel::X), &mut string);
            z_descent(tree.child_of(node, PauliLabel::Y), &mut string);
            // Sorted for the same reason as the parity above.
            string.sort_unstable_by_key(|&(qubit, _)| qubit);
            vertices.push(string.iter().copied());
        }

        // Bonsai Algorithm 1. `descend` walks from a link along `Z` edges to the leg it ends in.
        let descend = |mut slot: Slot| -> LegId {
            loop {
                match slot {
                    Slot::Leg(leg) => return leg,
                    Slot::Edge(node) => slot = tree.child_of(node, PauliLabel::Z),
                }
            }
        };

        let mut majorana_leg = vec![LegId::MAX; 2 * num_nodes as usize];
        for node in 0..num_nodes {
            let mode = mode_map.mode_of(node);
            for (label, offset) in [(PauliLabel::X, 0u32), (PauliLabel::Y, 1)] {
                majorana_leg[(2 * mode + offset) as usize] = descend(tree.child_of(node, label));
            }
        }
        // The mode map is a permutation, so every slot was written. The descent is also injective
        // over the whole tree (this is the content of Algorithm 1), so no leg was paired twice.
        // Only an invalid tree could break either, and a collision would leave one Majorana
        // operator aliasing another with no diagnostic downstream. These guard the walk against a
        // future edit, and cost nothing in a release build.
        debug_assert!(
            majorana_leg.iter().all(|&leg| leg != LegId::MAX),
            "a Majorana operator was left without a leg"
        );
        debug_assert!(
            {
                let mut seen = vec![false; tree.num_legs() as usize];
                majorana_leg
                    .iter()
                    .all(|&leg| !std::mem::replace(&mut seen[leg as usize], true))
            },
            "the Z-descent paired a leg twice"
        );

        // These are the strings of every leg but the leftover one, which carries the parity and
        // whose depth is `parity.len()`. What remains of the total support is exactly what they
        // need, so the buffers are sized once. The bound is quadratic in the node count on a
        // chain, which is inherent in storing every string rather than a cost of the bound.
        let mut majoranas =
            PauliTable::with_capacity(majorana_leg.len(), tree.total_leg_depth() - parity.len());
        for &leg in &majorana_leg {
            let mut path = tree.leg_path(leg);
            // Sorted for the same reason as the parity above.
            path.sort_unstable_by_key(|&(qubit, _)| qubit);
            majoranas.push(path);
        }

        Ok(Self {
            num_modes: num_nodes,
            majoranas,
            vertices,
            parity,
        })
    }

    /// Returns the number of fermionic modes this encoding covers.
    ///
    /// Equal to the tree's node count: a ternary-tree encoding uses exactly one qubit per mode.
    #[inline]
    pub fn num_modes(&self) -> u32 {
        self.num_modes
    }

    /// Returns the Pauli image of Majorana operator `majorana`, indexed as `2 * mode + is_prime`.
    ///
    /// The string is Hermitian and squares to the identity with coefficient exactly `1`: it is a
    /// product of Paulis on distinct qubits, so no phase correction is needed.
    ///
    /// With [`Self::num_modes`], this is the whole public surface a fermion-to-qubit mapping needs:
    /// every generator of all four operator algebras is a product of Majorana operators, so mapping
    /// one is a lookup here per factor. The other accessors are crate-internal shortcuts the
    /// mappers take for the common cases.
    #[inline]
    pub fn majorana_image(&self, majorana: u32) -> (&[PauliLabel], &[u32]) {
        self.majoranas.get(majorana as usize)
    }

    /// Returns the images of `gamma_2j` and `gamma_2j+1` of `mode`, back to back, and the length of
    /// the first.
    ///
    /// The two strings are adjacent in storage, so a fermionic action's two-term image is built
    /// from this span directly.
    #[inline]
    pub(crate) fn mode_images(&self, mode: u32) -> (&[PauliLabel], &[u32], usize) {
        let even = 2 * mode as usize;
        let (labels, indices) = self.majoranas.span(even, even + 2);
        (labels, indices, self.majoranas.weight(even))
    }

    /// Returns the image of the vertex operator `-i gamma_2j gamma_2j+1` of `mode` as parallel
    /// label and ascending-index slices.
    ///
    /// The image is a `Z` string whose coefficient is always exactly `+1` (see the type-level
    /// documentation), so no coefficient is returned.
    #[inline]
    pub(crate) fn vertex_image(&self, mode: u32) -> (&[PauliLabel], &[u32]) {
        self.vertices.get(mode as usize)
    }

    /// Builds the total fermionic parity observable of this encoding.
    ///
    /// The tree's `2N+1` legs pair into `2N` Majorana operators, leaving exactly one over, and that
    /// leftover is the total parity
    ///
    /// ```text
    ///     prod_j (-i gamma_2j gamma_2j+1)  =  prod_j (1 - 2 n_j)
    /// ```
    ///
    /// i.e. the product of the *vertex* operators, each of which is the `+-1`-valued parity of one
    /// mode. That product is the leftover leg's string with coefficient exactly `+1` on every tree
    /// (derived in the type-level documentation), so it is emitted directly. The sign convention
    /// matters: `+i` in place of `-i` would give `prod_j (2 n_j - 1)`, which differs by `(-1)^N`
    /// and so happens to agree for an even number of modes (a discrepancy that only an odd-`N` test
    /// reveals).
    ///
    /// # An observable to measure, not a constraint to impose
    ///
    /// Both eigenvalues of this operator are physical, so a caller must not project onto the `+1`
    /// sector. A ternary-tree encoding spends one qubit per mode and is a unitary isomorphism onto
    /// the whole `2^N`-dimensional space: both parity sectors are represented, each with
    /// multiplicity `2^(N-1)`, and there is no unphysical subspace to project away.
    ///
    /// Note also that the string is *not* in general the all-`Z` string; on a balanced tree at
    /// `N = 4` it is `Z_3 Z_0`.
    ///
    /// `num_qubits` must be at least the encoding's mode count; any excess carries the identity, so
    /// the observable can be built over the same register as a mapped Hamiltonian. Fewer is an
    /// error rather than a truncation, which keeps the bound the same as the mappers': an
    /// under-sized register could otherwise push an out-of-range qubit index into `qk_obs_new`,
    /// whose non-unwinding panic aborts the process.
    pub fn total_parity(
        &self,
        num_qubits: u32,
    ) -> Result<*mut ffi::QkObs, TernaryTreeEncodingError> {
        if num_qubits < self.num_modes() {
            return Err(TernaryTreeEncodingError::NumQubitsTooSmall {
                num_qubits,
                num_modes: self.num_modes(),
            });
        }
        // Built here rather than stored: this runs once per use, not once per mapped term.
        let labels = vec![PauliLabel::Z; self.parity.len()];
        Ok(one_pauli_string(
            num_qubits,
            QkComplex64 { re: 1.0, im: 0.0 },
            as_bit_terms(&labels),
            &self.parity,
        ))
    }

    /// Returns the largest Pauli weight of any single Majorana operator.
    ///
    /// Always equal to the tree's [`TernaryTree::max_weight`], even though that ranges over all
    /// `2N+1` legs and this only over the `2N` paired ones: the leftover leg can never be the
    /// unique deepest. It is the leg reached by descending `Z` from the root, so it hangs off the
    /// first node `w` on that path without a `Z` child, and has `w`'s own depth (a leg is as deep
    /// as the node it hangs off, per [`TernaryTree::leg_depth`]). The `X` link of `w` leads (then
    /// along `Z`) to the leg that carries `gamma` of `w`'s mode, which lies at least that deep. It
    /// is provided so that an encoding can report its weight without the tree it was compiled from.
    pub fn max_weight(&self) -> u32 {
        (0..2 * self.num_modes as usize)
            .map(|m| self.majoranas.weight(m) as u32)
            .max()
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::HashSet;

    use crate::mappers::qk_obs::{assert_obs_equal, compact};

    fn jw(n: u32) -> TernaryTree {
        TernaryTree::chain(n, PauliLabel::Z).unwrap()
    }

    fn balanced(n: u32) -> TernaryTree {
        TernaryTree::breadth_first(n, 3, PauliLabel::ALL).unwrap()
    }

    fn compile(tree: &TernaryTree) -> TernaryTreeEncoding {
        TernaryTreeEncoding::new(tree, &ModeMap::Identity).unwrap()
    }

    /// Returns leg `leg`'s string in ascending qubit order, as the encoding would store it.
    ///
    /// The encoding stores only the paired legs, so the leftover one is read off the tree.
    fn sorted_leg(tree: &TernaryTree, leg: LegId) -> (Vec<PauliLabel>, Vec<u32>) {
        let mut path = tree.leg_path(leg);
        path.sort_unstable_by_key(|&(qubit, _)| qubit);
        let (indices, labels): (Vec<u32>, Vec<PauliLabel>) = path.into_iter().unzip();
        (labels, indices)
    }

    /// Returns the string of every leg of `tree`, each in ascending qubit order.
    fn leg_strings(tree: &TernaryTree) -> Vec<(Vec<PauliLabel>, Vec<u32>)> {
        (0..tree.num_legs())
            .map(|leg| sorted_leg(tree, leg))
            .collect()
    }

    #[test]
    fn test_pairing_consumes_each_leg_exactly_once() {
        // The structural core of Algorithm 1: the 2N Majorana strings plus the parity string are
        // exactly the 2N+1 strings of the tree's legs, each once. If the Z-descent ever collided,
        // one Majorana operator would alias another and the encoding would not be invertible.
        // Phrased on the compiled strings because the encoding does not keep which leg carries
        // which operator, and a string identifies its leg: no two legs share one.
        for n in 1..=13u32 {
            for tree in [
                jw(n),
                balanced(n),
                TernaryTree::chain(n, PauliLabel::X).unwrap(),
            ] {
                let enc = compile(&tree);
                let legs: HashSet<_> = leg_strings(&tree).into_iter().collect();
                assert_eq!(
                    legs.len(),
                    (2 * n + 1) as usize,
                    "two legs share a string at n={n}"
                );

                let mut compiled: Vec<(Vec<PauliLabel>, Vec<u32>)> = (0..2 * n)
                    .map(|m| {
                        let (labels, indices) = enc.majorana_image(m);
                        (labels.to_vec(), indices.to_vec())
                    })
                    .collect();
                compiled.push((vec![PauliLabel::Z; enc.parity.len()], enc.parity.clone()));
                // 2N+1 compiled strings that cover all 2N+1 distinct leg strings repeat none.
                assert_eq!(
                    compiled.into_iter().collect::<HashSet<_>>(),
                    legs,
                    "the compiled strings are not the legs' strings at n={n}"
                );
            }
        }
    }

    #[test]
    fn test_tables_are_sized_exactly() {
        // The tables are allocated once, from sizes derived rather than counted: the Majorana
        // strings are every leg's string but the leftover one, whose depth is the parity's size.
        // A wrong derivation would only show up as a silent reallocation, so it is pinned here.
        for n in 1..=13u32 {
            for tree in [
                jw(n),
                balanced(n),
                TernaryTree::chain(n, PauliLabel::X).unwrap(),
            ] {
                let enc = compile(&tree);
                assert_eq!(
                    enc.majoranas.labels.len(),
                    tree.total_leg_depth() - enc.parity.len(),
                    "Majorana support at n={n}"
                );
                // ... and the vertex strings hold `2N - |parity|` entries between them.
                assert_eq!(
                    enc.vertices.labels.len(),
                    2 * n as usize - enc.parity.len(),
                    "vertex support at n={n}"
                );
            }
        }
    }

    #[test]
    fn test_jordan_wigner_majorana_images() {
        // The convention this crate already uses, reproduced by the chain along Z:
        //     gamma_2j  -> Z_0 ... Z_{j-1} X_j
        //     gamma_2j+1 -> Z_0 ... Z_{j-1} Y_j
        // Asserted on the compiled table, before any observable is built, so a mismatch is
        // localized here rather than in the mapper.
        let n = 5;
        let enc = compile(&jw(n));
        for mode in 0..n {
            for (offset, endpoint) in [(0u32, PauliLabel::X), (1, PauliLabel::Y)] {
                let (labels, indices) = enc.majorana_image(2 * mode + offset);
                let expected_indices: Vec<u32> = (0..=mode).collect();
                assert_eq!(
                    indices,
                    expected_indices.as_slice(),
                    "gamma_{} support",
                    2 * mode + offset
                );
                let expected_labels: Vec<PauliLabel> = (0..mode)
                    .map(|_| PauliLabel::Z)
                    .chain(std::iter::once(endpoint))
                    .collect();
                assert_eq!(
                    labels,
                    expected_labels.as_slice(),
                    "gamma_{} labels",
                    2 * mode + offset
                );
            }
        }
    }

    #[test]
    fn test_bravyi_kitaev_majorana_images() {
        // Bravyi-Kitaev is a member of this family: compiling its tree under this module's own
        // pairing rule reproduces the encoding's Majorana strings term for term. Weight alone
        // cannot stand in for this check, since the uniform `breadth_first(n, 2, ..)` tree matches
        // the weight while producing different strings.
        //
        // The N = 4 tree: root at 3, node 1 off its `X` link, nodes 0 and 2 off node 1's `X` and
        // `Z`. Only `X` and `Z` carry edges; a node's `Y` link is always a leg.
        let tree = TernaryTree::try_new(&[
            Some((1, PauliLabel::X)),
            Some((3, PauliLabel::X)),
            Some((1, PauliLabel::Z)),
            None,
        ])
        .unwrap();
        let enc = compile(&tree);

        // Standard Bravyi-Kitaev, from the Seeley-Richard-Love form
        //     gamma_2j   = X_{U(j)} X_j Z_{P(j)},   gamma_2j+1 = X_{U(j)} Y_j Z_{R(j)}
        // with the update, parity and remainder sets of N = 4.
        use PauliLabel::{X, Y, Z};
        let expected: [(&[PauliLabel], &[u32]); 8] = [
            (&[X, X, X], &[0, 1, 3]),
            (&[Y, X, X], &[0, 1, 3]),
            (&[Z, X, X], &[0, 1, 3]),
            (&[Y, X], &[1, 3]),
            (&[Z, X, X], &[1, 2, 3]),
            (&[Z, Y, X], &[1, 2, 3]),
            (&[Z, Z, X], &[1, 2, 3]),
            (&[Y], &[3]),
        ];
        for (m, (labels, indices)) in expected.iter().enumerate() {
            let (got_labels, got_indices) = enc.majorana_image(m as u32);
            assert_eq!(got_indices, *indices, "gamma_{m} support");
            assert_eq!(got_labels, *labels, "gamma_{m} labels");
        }

        // The weight is Bravyi-Kitaev's `floor(log2 n) + 1`, which the uniform tree also reaches --
        // which is exactly why weight alone cannot stand in for the check above.
        assert_eq!(tree.max_weight(), 3);
    }

    #[test]
    fn test_bravyi_kitaev_root_is_not_the_last_node() {
        // N = 4 above is a power of two, where the root `2^k - 1` coincides with `N-1`. That
        // coincidence makes a power-of-two-only check unable to tell the two rules apart, so N = 6
        // is pinned as well: there the root is 3, not 5.
        use PauliLabel::{X, Y, Z};
        let tree = TernaryTree::try_new(&[
            Some((1, X)),
            Some((3, X)),
            Some((1, Z)),
            None,
            Some((5, X)),
            Some((3, Z)),
        ])
        .unwrap();
        let enc = compile(&tree);

        let expected: [(&[PauliLabel], &[u32]); 12] = [
            (&[X, X, X], &[0, 1, 3]),
            (&[Y, X, X], &[0, 1, 3]),
            (&[Z, X, X], &[0, 1, 3]),
            (&[Y, X], &[1, 3]),
            (&[Z, X, X], &[1, 2, 3]),
            (&[Z, Y, X], &[1, 2, 3]),
            (&[Z, Z, X], &[1, 2, 3]),
            (&[Y], &[3]),
            (&[Z, X, X], &[3, 4, 5]),
            (&[Z, Y, X], &[3, 4, 5]),
            (&[Z, Z, X], &[3, 4, 5]),
            (&[Z, Y], &[3, 5]),
        ];
        for (m, (labels, indices)) in expected.iter().enumerate() {
            let (got_labels, got_indices) = enc.majorana_image(m as u32);
            assert_eq!(got_indices, *indices, "gamma_{m} support");
            assert_eq!(got_labels, *labels, "gamma_{m} labels");
        }

        // `floor(log2 6) + 1`, and the `Y` link of every node is a leg rather than an edge.
        assert_eq!(tree.max_weight(), 3);
    }

    #[test]
    fn test_strings_are_sorted_and_distinct() {
        // Sortedness is load-bearing rather than cosmetic: duplicate Pauli terms are merged only
        // when stored identically, so an unsorted emission would quietly break the mappers' memory
        // bound.
        for n in [1u32, 2, 4, 7, 13] {
            for tree in [jw(n), balanced(n)] {
                let enc = compile(&tree);
                for m in 0..2 * n {
                    let (labels, indices) = enc.majorana_image(m);
                    assert_eq!(indices.len(), labels.len());
                    assert!(
                        indices.windows(2).all(|w| w[0] < w[1]),
                        "gamma_{m} at n={n} is not strictly ascending: {indices:?}"
                    );
                }
                for mode in 0..n {
                    let (_, indices) = enc.vertex_image(mode);
                    assert!(
                        indices.windows(2).all(|w| w[0] < w[1]),
                        "vertex {mode} at n={n} is not strictly ascending: {indices:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_parity_is_not_always_all_z() {
        // A trap worth pinning: the leftover leg is the all-Z *path*, not the all-Z *string*. On
        // the balanced tree at N=4 it is `Z_3 Z_0` -- weight 2 on 4 qubits.
        assert_eq!(compile(&balanced(4)).parity, [0, 3]);

        // On the Jordan-Wigner chain it *is* the full all-Z string, which is why the JW case alone
        // would not have caught this.
        assert_eq!(compile(&jw(4)).parity, [0, 1, 2, 3]);
    }

    /// The total parity is the string of the one leg that no Majorana operator carries, with
    /// coefficient exactly `+1`.
    ///
    /// That leg is found by elimination: it is the one leg string of the tree that none of the
    /// compiled Majorana images equals. The independent check that the parity is the product of
    /// the vertex operators, odd mode counts included, lives with the mappers, which can build
    /// that product from the Majorana strings.
    #[test]
    fn test_total_parity_is_the_leftover_leg() {
        for tree in [
            jw(4),
            balanced(4),
            jw(5),
            balanced(13),
            TernaryTree::chain(4, PauliLabel::X).unwrap(),
        ] {
            let enc = compile(&tree);
            let parity = unsafe { compact(enc.total_parity(enc.num_modes()).unwrap()) };
            assert_eq!(
                unsafe { ffi::qk_obs_num_terms(parity) },
                1,
                "total parity must be a single Pauli string"
            );
            // ... and that string is the encoding's leftover leg: the one leg whose string no
            // Majorana operator carries.
            let images: Vec<_> = (0..2 * enc.num_modes())
                .map(|m| enc.majorana_image(m))
                .collect();
            let leftover: Vec<_> = leg_strings(&tree)
                .into_iter()
                .filter(|(labels, indices)| {
                    !images
                        .iter()
                        .any(|&(l, i)| l == labels.as_slice() && i == indices.as_slice())
                })
                .collect();
            assert_eq!(leftover.len(), 1, "exactly one leg is left over");
            let (labels, indices) = &leftover[0];
            let expected = one_pauli_string(
                enc.num_modes(),
                QkComplex64 { re: 1.0, im: 0.0 },
                as_bit_terms(labels),
                indices,
            );
            assert_obs_equal(parity, expected, enc.num_modes());
        }
    }

    #[test]
    fn test_mode_map_permutes_the_pairing() {
        // A permuted mode map must relabel which mode each string serves, without touching the
        // strings themselves.
        let tree = balanced(4);
        let identity = compile(&tree);
        // Not its own inverse, so the direction is pinned too: node `u` serves mode `modes[u]`.
        let modes = [2u32, 0, 3, 1];
        let permuted =
            TernaryTreeEncoding::new(&tree, &ModeMap::Permutation(modes.to_vec())).unwrap();

        for node in 0..4u32 {
            for offset in 0..2u32 {
                assert_eq!(
                    identity.majorana_image(2 * node + offset),
                    permuted.majorana_image(2 * modes[node as usize] + offset)
                );
            }
        }
        // The parity is a property of the tree, not the mode labeling.
        assert_eq!(identity.parity, permuted.parity);
    }

    #[test]
    fn test_equality_and_hash_follow_the_majorana_images() {
        // Equality and hashing look at the Majorana images alone, on the grounds that they
        // determine everything else the encoding stores. Pinned against a comparison of every
        // stored field, so a field that stopped being derived from them would be noticed here and
        // not as two "equal" encodings that behave differently.
        let mut encodings = Vec::new();
        for n in 1..=5u32 {
            for tree in [
                jw(n),
                balanced(n),
                TernaryTree::chain(n, PauliLabel::X).unwrap(),
            ] {
                encodings.push(compile(&tree));
                let reversed = ModeMap::Permutation((0..n).rev().collect());
                encodings.push(TernaryTreeEncoding::new(&tree, &reversed).unwrap());
            }
        }

        // Destructured, so that a field added to the encoding is a compile error here (and an
        // unused variable, which the lints reject) until it is compared too.
        let same_fields = |a: &TernaryTreeEncoding, b: &TernaryTreeEncoding| {
            let TernaryTreeEncoding {
                num_modes,
                majoranas,
                vertices,
                parity,
            } = a;
            *num_modes == b.num_modes
                && *majoranas == b.majoranas
                && *vertices == b.vertices
                && *parity == b.parity
        };
        let hash = |enc: &TernaryTreeEncoding| {
            let mut hasher = std::hash::DefaultHasher::new();
            enc.hash(&mut hasher);
            hasher.finish()
        };
        let mut num_equal_pairs = 0;
        for (i, a) in encodings.iter().enumerate() {
            for (j, b) in encodings.iter().enumerate() {
                assert_eq!(a == b, same_fields(a, b));
                if a == b {
                    assert_eq!(hash(a), hash(b));
                    if i != j {
                        num_equal_pairs += 1;
                    }
                }
            }
        }
        // Besides each encoding equalling itself, the one-node trees are all the same tree and
        // the reversal of one mode is the identity, so distinct inputs really do compare equal.
        assert!(num_equal_pairs > 0);
    }

    #[test]
    fn test_images_span_only_the_trees_nodes() {
        // The compiled strings are a property of the tree alone: they never mention a qubit beyond
        // its nodes. That is what lets a mapper pad to any wider register without recompiling,
        // since the extra qubits simply carry the identity.
        let tree = jw(2);
        let enc = TernaryTreeEncoding::new(&tree, &ModeMap::Identity).unwrap();
        assert_eq!(enc.num_modes(), 2);
        for m in 0..4 {
            let (_, indices) = enc.majorana_image(m);
            assert!(indices.iter().all(|&q| q < 2));
        }
    }

    #[test]
    fn test_rejects_bad_mode_map() {
        let tree = jw(3);
        assert_eq!(
            TernaryTreeEncoding::new(&tree, &ModeMap::Permutation(vec![0, 1])),
            Err(TernaryTreeEncodingError::ModeMapLength {
                len: 2,
                num_nodes: 3
            })
        );
        assert_eq!(
            TernaryTreeEncoding::new(&tree, &ModeMap::Permutation(vec![0, 1, 1])),
            Err(TernaryTreeEncodingError::ModeMapNotAPermutation { mode: 1 })
        );
        // An out-of-range entry is a *mode*, so it is reported as one; `NodeOutOfRange` would
        // describe the map's index rather than its value.
        assert_eq!(
            TernaryTreeEncoding::new(&tree, &ModeMap::Permutation(vec![0, 1, 9])),
            Err(TernaryTreeEncodingError::ModeOutOfRange {
                mode: 9,
                num_nodes: 3
            })
        );
    }

    /// The encoding's weight, over the paired legs only, always equals the tree's, over all legs.
    ///
    /// The leftover leg can never be the unique deepest (see [`TernaryTreeEncoding::max_weight`]),
    /// so the restriction never changes the answer. Checked across every uniform family and label
    /// order, where the leftover leg sits at very different depths relative to the rest.
    #[test]
    fn test_max_weight_matches_the_tree() {
        use PauliLabel::{X, Y, Z};
        let orders = [
            [X, Y, Z],
            [X, Z, Y],
            [Y, X, Z],
            [Y, Z, X],
            [Z, X, Y],
            [Z, Y, X],
        ];
        for n in 1..=40u32 {
            for branching in 1..=3 {
                for order in orders {
                    let tree = TernaryTree::breadth_first(n, branching, order).unwrap();
                    assert_eq!(
                        compile(&tree).max_weight(),
                        tree.max_weight(),
                        "n={n}, branching={branching}, order={order:?}"
                    );
                }
            }
        }
        assert_eq!(compile(&balanced(13)).max_weight(), 3);
        assert_eq!(compile(&jw(5)).max_weight(), 5);
    }

    /// Each precomputed vertex image is the product `-i gamma_2j gamma_2j+1` of the stored strings,
    /// with coefficient exactly `+1`.
    ///
    /// The closed form the encoding builds the vertices from is derived rather than computed, so
    /// this checks it against `qk_obs_compose` over every uniform family, label order and a
    /// non-trivial mode map. A tree on which the derivation failed would show up as a vertex off by
    /// a sign.
    #[test]
    fn test_vertex_images_are_the_majorana_products() {
        use PauliLabel::{X, Y, Z};
        let orders = [
            [X, Y, Z],
            [X, Z, Y],
            [Y, X, Z],
            [Y, Z, X],
            [Z, X, Y],
            [Z, Y, X],
        ];
        let trees = [1u32, 4, 5, 13].into_iter().flat_map(|n| {
            (1..=3).flat_map(move |branching| {
                orders
                    .into_iter()
                    .map(move |order| TernaryTree::breadth_first(n, branching, order).unwrap())
            })
        });
        for tree in trees {
            let reversed: Vec<u32> = (0..tree.num_nodes()).rev().collect();
            let enc = TernaryTreeEncoding::new(&tree, &ModeMap::Permutation(reversed)).unwrap();
            let n = enc.num_modes();
            let one = QkComplex64 { re: 1.0, im: 0.0 };
            for mode in 0..n {
                let (labels, indices) = enc.majorana_image(2 * mode);
                let even = one_pauli_string(n, one, as_bit_terms(labels), indices);
                let (labels, indices) = enc.majorana_image(2 * mode + 1);
                let odd = one_pauli_string(n, one, as_bit_terms(labels), indices);
                let product = unsafe { ffi::qk_obs_compose(odd, even) };
                unsafe { ffi::qk_obs_free(even) };
                unsafe { ffi::qk_obs_free(odd) };
                unsafe {
                    ffi::qk_obs_multiply_inplace(product, &QkComplex64 { re: 0.0, im: -1.0 })
                };

                let (labels, indices) = enc.vertex_image(mode);
                assert!(labels.iter().all(|&label| label == Z));
                let vertex = one_pauli_string(n, one, as_bit_terms(labels), indices);
                assert_obs_equal(vertex, product, n);
            }
        }
    }
}
