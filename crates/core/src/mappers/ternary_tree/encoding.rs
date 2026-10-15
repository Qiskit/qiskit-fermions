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

use super::{LegId, NodeId, PauliLabel, Slot, TernaryTree, TernaryTreeError};
use crate::mappers::qk_obs::{QkComplex64, bit_terms, ffi_backend as ffi, one_pauli_string};
use crate::operators::CoherenceError;

/// How fermionic modes are assigned to the nodes of a tree.
///
/// A tree says which Pauli strings exist; this says which mode's Majorana operators each pair of them
/// represents. The two are separate because every tree-search method varies one while holding the
/// other fixed -- reordering the modes of a fixed tree is one of the cheapest ways to lower the weight
/// of a particular Hamiltonian.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModeMap {
    /// Mode `j` is paired at node `j`.
    Identity,
    /// `mode_of_node[u]` is the mode paired at node `u`; a permutation of `0..num_nodes`.
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
    fn validate(&self, num_nodes: u32) -> Result<(), TernaryTreeError> {
        let ModeMap::Permutation(modes) = self else {
            return Ok(());
        };
        if modes.len() != num_nodes as usize {
            return Err(TernaryTreeError::ModeMapLength {
                len: modes.len(),
                num_nodes,
            });
        }
        let mut seen = vec![false; num_nodes as usize];
        for &mode in modes {
            if mode >= num_nodes {
                return Err(TernaryTreeError::ModeOutOfRange { mode, num_nodes });
            }
            if seen[mode as usize] {
                return Err(TernaryTreeError::ModeMapNotAPermutation { mode });
            }
            seen[mode as usize] = true;
        }
        Ok(())
    }
}

/// A ternary tree compiled into the Pauli strings of its Majorana operators.
///
/// Holds the `2N+1` leg strings in a flat, sorted form and records which of them carries each Majorana
/// operator. This is the artifact the mappers consume: after it is built, mapping a generator is a
/// slice lookup.
///
/// # The pairing
///
/// The tree yields `2N+1` mutually anticommuting strings, one more than the `2N` Majorana operators of
/// `N` modes. Which string is dropped, and which of a pair is `gamma` rather than `gamma'`, is fixed
/// by Algorithm 1 of the Bonsai paper (Ref. 1 in the [module documentation](super)): from node `u`,
/// follow its `X` link and then descend along `Z` links until reaching a leg (that leg carries `gamma`
/// of the mode at `u`); repeat from the `Y` link for `gamma'`. This consumes exactly `2N` legs, and
/// the one left over is the leg reached by descending `Z` from the root.
///
/// The rule is not an arbitrary convention: it is what makes `|0...0>` the fermionic vacuum. Any
/// injective assignment of Majorana operators to legs gives a valid algebra, but only this one
/// preserves the vacuum and makes `i gamma_2j gamma_2j+1` a number operator.
///
/// # The leftover string
///
/// The unpaired leg is the total fermionic parity, `prod_j (i gamma_2j gamma_2j+1)`, up to a phase
/// that depends on the tree. Two things about it are easy to get wrong:
///
/// * Its Pauli string is **not** in general the all-`Z` string. It is the leg reached by the all-`Z`
///   *path*, whose links descend from whichever nodes happen to lie on that path, so it carries the
///   identity on every node that does not. On the balanced tree at `N = 4` it is `Z_3 Z_0`.
/// * It is an **observable to measure, not a constraint to impose**. A ternary-tree encoding spends
///   one qubit per mode and is a unitary isomorphism onto the whole `2^N`-dimensional space, so both
///   parity sectors are physical and nothing may be projected away. Encodings that add ancilla qubits
///   (more qubits than modes) are the ones with a stabilizer subspace; a tree has none.
///
/// Its phase is computed by the mapper, which has Pauli multiplication available; this module records
/// only which leg it is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TernaryTreeEncoding {
    /// The tree's node count, which is both the mode count and the qubits the strings span.
    num_modes: u32,
    /// Flat storage of the leg strings, mirroring the operator types' `{modes, boundaries}` layout.
    /// `qubits[boundaries[l]..boundaries[l + 1]]` is leg `l`'s support, ascending.
    qubits: Vec<u32>,
    /// The label acting on each entry of `qubits`.
    labels: Vec<PauliLabel>,
    /// Start of each leg's slice into `qubits` and `labels`, with a trailing total; length
    /// `num_legs + 1`.
    boundaries: Vec<usize>,
    /// `majorana_leg[m]` carries Majorana `m`, indexed as `2 * mode + is_prime`.
    majorana_leg: Vec<LegId>,
    /// The single leg the pairing leaves over: total parity.
    parity_leg: LegId,
}

impl TernaryTreeEncoding {
    /// Compiles `tree` under `mode_map` into the Pauli images of its Majorana operators.
    ///
    /// The compiled strings span exactly the tree's nodes. The qubit count of a *mapped operator* is
    /// a property of that mapping rather than of the encoding, so it is passed to the mappers
    /// instead, which pad with the identity exactly as the Jordan-Wigner ones do.
    pub fn new(tree: &TernaryTree, mode_map: &ModeMap) -> Result<Self, TernaryTreeError> {
        let num_nodes = tree.num_nodes();
        mode_map.validate(num_nodes)?;

        let num_legs = tree.num_legs();
        // The total support summed over all legs is exactly the sum of their depths, which is what
        // `total_leg_depth` returns, so both flat buffers are sized once and never reallocate.
        let total_support = tree.total_leg_depth() as usize;
        let mut qubits = Vec::with_capacity(total_support);
        let mut labels = Vec::with_capacity(total_support);
        let mut boundaries = Vec::with_capacity(num_legs as usize + 1);
        boundaries.push(0usize);
        for leg in 0..num_legs {
            let mut path = tree.leg_path(leg);
            // Sort by qubit index: `SparseObservable` *requires* each term's indices to be
            // term-wise sorted, and documents that algorithms may rely on it. The path is generated
            // leaf-to-root, i.e. in depth order, which bears no relation to the node numbering, so
            // sorting here is what makes the emitted observable valid at all.
            //
            // Nothing enforces it on the way in: `qk_obs_add_term` accepts unsorted indices and
            // returns success, while Qiskit's own Python constructor rejects the same data with
            // `ValueError`. The visible consequence is that `qk_obs_canonicalize` compares stored
            // bit-term sequences, so duplicates of one Pauli string stored in two orders never
            // merge, silently defeating the memory bound in `qk_obs` without ever giving a wrong
            // answer. That is the symptom; the class invariant is the reason.
            path.sort_unstable_by_key(|&(qubit, _)| qubit);
            for (qubit, label) in path {
                qubits.push(qubit);
                labels.push(label);
            }
            boundaries.push(qubits.len());
        }

        // Bonsai Algorithm 1. `descend` walks from a link along `Z` edges to the leg it terminates in.
        let descend = |mut slot: Slot| -> LegId {
            loop {
                match slot {
                    Slot::Leg(leg) => return leg,
                    Slot::Edge(node) => slot = tree.child_of(node, PauliLabel::Z),
                }
            }
        };

        let mut majorana_leg = vec![LegId::MAX; 2 * num_nodes as usize];
        let mut paired = vec![false; num_legs as usize];
        for node in 0..num_nodes {
            let mode = mode_map.mode_of(node);
            for (label, offset) in [(PauliLabel::X, 0u32), (PauliLabel::Y, 1)] {
                let leg = descend(tree.child_of(node, label));
                // The descent is injective over the whole tree (this is the content of Algorithm 1),
                // so a collision would mean the walk, not the tree, is wrong. Checked rather than
                // assumed because a silent collision would leave one Majorana operator aliasing
                // another and the encoding non-invertible.
                debug_assert!(
                    !paired[leg as usize],
                    "leg {leg} was paired twice; the Z-descent is not injective"
                );
                paired[leg as usize] = true;
                majorana_leg[(2 * mode + offset) as usize] = leg;
            }
        }

        let parity_leg = paired
            .iter()
            .position(|&p| !p)
            .expect("the pairing consumes 2N of 2N+1 legs, so exactly one is left over")
            as LegId;

        Ok(Self {
            num_modes: num_nodes,
            qubits,
            labels,
            boundaries,
            majorana_leg,
            parity_leg,
        })
    }

    /// Returns the number of fermionic modes this encoding covers.
    ///
    /// Equal to the tree's node count: a ternary-tree encoding uses exactly one qubit per mode.
    #[inline]
    pub fn num_modes(&self) -> u32 {
        self.num_modes
    }

    /// Returns the Pauli image of leg `leg`, as parallel ascending-qubit and label slices.
    ///
    /// Addressed by [`LegId`], i.e. over all `2N+1` legs including the unpaired one. Not public:
    /// outside this crate an encoding is consumed one Majorana operator at a time, through
    /// [`Self::majorana_image`], which delegates here. The leg-indexed view is what the tests need to
    /// reach the unpaired leg, which no Majorana index addresses.
    #[inline]
    pub(crate) fn leg_image(&self, leg: LegId) -> (&[u32], &[PauliLabel]) {
        let range = self.boundaries[leg as usize]..self.boundaries[leg as usize + 1];
        (&self.qubits[range.clone()], &self.labels[range])
    }

    /// Returns the Pauli image of Majorana operator `majorana`, indexed as `2 * mode + is_prime`.
    ///
    /// The string is Hermitian and squares to the identity with coefficient exactly `1`: it is a
    /// product of Paulis on distinct qubits, so no phase correction is needed.
    ///
    /// With [`Self::num_modes`], this is the whole public surface a fermion-to-qubit mapping needs:
    /// every generator of all four operator algebras is a product of Majorana operators, so mapping
    /// one is a lookup here per factor. The leg-level accessors behind it are crate-internal.
    #[inline]
    pub fn majorana_image(&self, majorana: u32) -> (&[u32], &[PauliLabel]) {
        self.leg_image(self.majorana_leg[majorana as usize])
    }

    /// Returns the leg left unpaired, which carries the total fermionic parity.
    ///
    /// See the type-level documentation: this is an observable rather than a stabilizer, and its
    /// string is not in general all-`Z`.
    ///
    /// Test-only, and `pub(crate)` because one of those tests lives in the mapper module: the parity
    /// observable is built by [`Self::total_parity`], which multiplies the vertex operators out to
    /// recover the tree-dependent phase that this leg alone does not carry. The tests pin *which* leg is left over and that its string is not all-`Z`, which cannot
    /// go through [`Self::majorana_image`] since the leftover leg is precisely the one no Majorana
    /// operator maps to.
    #[cfg(test)]
    #[inline]
    pub(crate) fn parity_leg(&self) -> LegId {
        self.parity_leg
    }

    /// Builds the total fermionic parity observable of this encoding.
    ///
    /// The tree's `2N+1` legs pair into `2N` Majorana operators, leaving exactly one over, and that
    /// leftover is the total parity. Its Pauli string is read from the encoding; its phase is computed here
    /// by multiplying the pairs out, because the phase depends on the tree and assuming `+1` would be wrong
    /// for some shapes.
    ///
    /// The product taken is
    ///
    /// ```text
    ///     prod_j (-i gamma_2j gamma_2j+1)  =  prod_j (1 - 2 n_j)
    /// ```
    ///
    /// i.e. of the *vertex* operators, each of which is the `+-1`-valued parity of one mode. The sign
    /// matters: `+i` in place of `-i` would give `prod_j (2 n_j - 1)`, which differs by `(-1)^N` and so
    /// happens to agree for an even number of modes -- a discrepancy that only an odd-`N` test reveals.
    ///
    /// # An observable to measure, not a constraint to impose
    ///
    /// Both eigenvalues of this operator are physical, so a caller must not project onto the `+1` sector.
    /// A ternary-tree encoding spends one qubit per mode and is a unitary isomorphism onto the whole
    /// `2^N`-dimensional space: both parity sectors are represented, each with multiplicity `2^(N-1)`, and
    /// there is no unphysical subspace to project away. Encodings that add ancilla qubits (more qubits than
    /// modes) are the ones made faithful by a stabilizer subspace; a tree has none.
    ///
    /// Note also that the string is *not* in general the all-`Z` string; on a balanced tree at `N = 4` it
    /// is `Z_3 Z_0`.
    ///
    /// `num_qubits` must be at least the encoding's mode count; any excess carries the identity, so the
    /// observable can be built over the same register as a mapped Hamiltonian. Fewer is an error
    /// rather than a truncation: the vertex operators span every node of the tree, so an under-sized
    /// register would push an out-of-range qubit index into `qk_obs_new`, whose non-unwinding panic
    /// aborts the process.
    pub fn total_parity(&self, num_qubits: u32) -> Result<*mut ffi::QkObs, CoherenceError> {
        if num_qubits < self.num_modes() {
            return Err(CoherenceError::NumQubitsTooSmallForEncoding {
                num_qubits,
                num_modes: self.num_modes(),
            });
        }
        // One Majorana operator's image, as a single Pauli string with coefficient 1.
        let image = |majorana: u32| {
            let (qubits, labels) = self.majorana_image(majorana);
            one_pauli_string(
                num_qubits,
                QkComplex64 { re: 1.0, im: 0.0 },
                &mut bit_terms(labels),
                &mut qubits.to_vec(),
            )
        };

        let mut parity = unsafe { ffi::qk_obs_identity(num_qubits) };
        for mode in 0..self.num_modes() {
            // The vertex operator `-i gamma_2j gamma_2j+1`. `qk_obs_compose(second, first)` applies
            // `first` then `second`, so this is the product in that order.
            let even = image(2 * mode);
            let odd = image(2 * mode + 1);
            let term = unsafe { ffi::qk_obs_compose(odd, even) };
            unsafe { ffi::qk_obs_free(even) };
            unsafe { ffi::qk_obs_free(odd) };
            unsafe { ffi::qk_obs_multiply_inplace(term, &QkComplex64 { re: 0.0, im: -1.0 }) };

            let next = unsafe { ffi::qk_obs_compose(term, parity) };
            unsafe { ffi::qk_obs_free(term) };
            unsafe { ffi::qk_obs_free(parity) };
            parity = next;
        }
        Ok(parity)
    }

    /// Returns the largest Pauli weight of any single Majorana operator.
    pub fn max_weight(&self) -> u32 {
        (0..2 * self.num_modes)
            .map(|m| self.majorana_image(m).0.len() as u32)
            .max()
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::mappers::qk_obs::compact;

    fn jw(n: u32) -> TernaryTree {
        TernaryTree::chain(n, PauliLabel::Z).unwrap()
    }

    fn balanced(n: u32) -> TernaryTree {
        TernaryTree::breadth_first(n, 3, PauliLabel::ALL).unwrap()
    }

    fn compile(tree: &TernaryTree) -> TernaryTreeEncoding {
        TernaryTreeEncoding::new(tree, &ModeMap::Identity).unwrap()
    }

    /// Asserts two observables represent the same operator.
    fn assert_obs_equal(got: *mut ffi::QkObs, expected: *mut ffi::QkObs, num_qubits: u32) {
        let factor = QkComplex64 { re: -1.0, im: 0.0 };
        let negated = unsafe { ffi::qk_obs_multiply(expected, &factor) };
        let diff = unsafe { ffi::qk_obs_add(got, negated) };
        let diff = unsafe { ffi::qk_obs_canonicalize(diff, 1e-9) };
        let zero = unsafe { ffi::qk_obs_zero(num_qubits) };
        assert!(unsafe { ffi::qk_obs_equal(diff, zero) });
    }

    #[test]
    fn test_pairing_consumes_each_leg_exactly_once() {
        // The structural core of Algorithm 1: the 2N paired legs plus the leftover are a permutation
        // of all 2N+1 legs. If the Z-descent ever collided, one Majorana operator would alias another
        // and the encoding would not be invertible.
        for n in 1..=13u32 {
            for tree in [
                jw(n),
                balanced(n),
                TernaryTree::chain(n, PauliLabel::X).unwrap(),
            ] {
                let enc = compile(&tree);
                let mut seen = vec![false; tree.num_legs() as usize];
                for m in 0..2 * n {
                    let leg = enc.majorana_leg[m as usize];
                    assert!(!seen[leg as usize], "leg {leg} used twice at n={n}");
                    seen[leg as usize] = true;
                }
                assert!(
                    !seen[enc.parity_leg() as usize],
                    "the parity leg must not also be paired"
                );
                seen[enc.parity_leg() as usize] = true;
                assert!(seen.iter().all(|&s| s), "not every leg was accounted for");
            }
        }
    }

    #[test]
    fn test_jordan_wigner_majorana_images() {
        // The convention this crate already uses, reproduced by the chain along Z:
        //     gamma_2j  -> Z_0 ... Z_{j-1} X_j
        //     gamma_2j+1 -> Z_0 ... Z_{j-1} Y_j
        // Asserted on the compiled table, before any observable is built, so a mismatch is localised
        // here rather than in the mapper.
        let n = 5;
        let enc = compile(&jw(n));
        for mode in 0..n {
            for (offset, endpoint) in [(0u32, PauliLabel::X), (1, PauliLabel::Y)] {
                let (qubits, labels) = enc.majorana_image(2 * mode + offset);
                let expected_qubits: Vec<u32> = (0..=mode).collect();
                assert_eq!(
                    qubits,
                    expected_qubits.as_slice(),
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
        // Bravyi-Kitaev is a member of this family: compiling its tree under this module's own pairing
        // rule reproduces the encoding's Majorana strings term for term. Weight alone cannot stand in
        // for this check, since the uniform `breadth_first(n, 2, ..)` tree matches the weight while
        // producing different strings.
        //
        // The N = 4 tree: root at 3, node 1 off its `X` link, nodes 0 and 2 off node 1's `X` and `Z`.
        // Only `X` and `Z` carry edges; a node's `Y` link is always a leg.
        //
        // Expectations verified against OpenFermion's `bravyi_kitaev` for N = 2..20.
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
        let expected: [(&[u32], &[PauliLabel]); 8] = [
            (&[0, 1, 3], &[X, X, X]),
            (&[0, 1, 3], &[Y, X, X]),
            (&[0, 1, 3], &[Z, X, X]),
            (&[1, 3], &[Y, X]),
            (&[1, 2, 3], &[Z, X, X]),
            (&[1, 2, 3], &[Z, Y, X]),
            (&[1, 2, 3], &[Z, Z, X]),
            (&[3], &[Y]),
        ];
        for (m, (qubits, labels)) in expected.iter().enumerate() {
            let (got_qubits, got_labels) = enc.majorana_image(m as u32);
            assert_eq!(got_qubits, *qubits, "gamma_{m} support");
            assert_eq!(got_labels, *labels, "gamma_{m} labels");
        }

        // The weight is Bravyi-Kitaev's `floor(log2 n) + 1`, which the uniform tree also reaches --
        // which is exactly why weight alone cannot stand in for the check above.
        assert_eq!(tree.max_weight(), 3);
    }

    #[test]
    fn test_bravyi_kitaev_root_is_not_the_last_node() {
        // N = 4 above is a power of two, where the root `2^k - 1` coincides with `N-1`. That
        // coincidence makes a power-of-two-only check unable to tell the two rules apart, so N = 6 is
        // pinned as well: there the root is 3, not 5. It is also the smallest size at which the block
        // split differs from a breadth-first fill.
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

        let expected: [(&[u32], &[PauliLabel]); 12] = [
            (&[0, 1, 3], &[X, X, X]),
            (&[0, 1, 3], &[Y, X, X]),
            (&[0, 1, 3], &[Z, X, X]),
            (&[1, 3], &[Y, X]),
            (&[1, 2, 3], &[Z, X, X]),
            (&[1, 2, 3], &[Z, Y, X]),
            (&[1, 2, 3], &[Z, Z, X]),
            (&[3], &[Y]),
            (&[3, 4, 5], &[Z, X, X]),
            (&[3, 4, 5], &[Z, Y, X]),
            (&[3, 4, 5], &[Z, Z, X]),
            (&[3, 5], &[Z, Y]),
        ];
        for (m, (qubits, labels)) in expected.iter().enumerate() {
            let (got_qubits, got_labels) = enc.majorana_image(m as u32);
            assert_eq!(got_qubits, *qubits, "gamma_{m} support");
            assert_eq!(got_labels, *labels, "gamma_{m} labels");
        }

        // `floor(log2 6) + 1`, and the `Y` link of every node is a leg rather than an edge.
        assert_eq!(tree.max_weight(), 3);
    }

    #[test]
    fn test_strings_are_sorted_and_distinct() {
        // Sortedness is load-bearing rather than cosmetic: duplicate Pauli terms are merged only when
        // stored identically, so an unsorted emission would quietly break the mappers' memory bound.
        for n in [1u32, 2, 4, 7, 13] {
            for tree in [jw(n), balanced(n)] {
                let enc = compile(&tree);
                for leg in 0..tree.num_legs() {
                    let (qubits, labels) = enc.leg_image(leg);
                    assert_eq!(qubits.len(), labels.len());
                    assert!(
                        qubits.windows(2).all(|w| w[0] < w[1]),
                        "leg {leg} at n={n} is not strictly ascending: {qubits:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_parity_leg_is_not_always_all_z() {
        // A trap worth pinning: the leftover leg is the all-Z *path*, not the all-Z *string*. On the
        // balanced tree at N=4 it is `Z_3 Z_0` -- weight 2 on 4 qubits.
        let enc = compile(&balanced(4));
        let (qubits, labels) = enc.leg_image(enc.parity_leg());
        assert_eq!(qubits, &[0, 3]);
        assert_eq!(labels, &[PauliLabel::Z, PauliLabel::Z]);

        // On the Jordan-Wigner chain it *is* the full all-Z string, which is why the JW case alone
        // would not have caught this.
        let enc = compile(&jw(4));
        let (qubits, labels) = enc.leg_image(enc.parity_leg());
        assert_eq!(qubits, &[0, 1, 2, 3]);
        assert_eq!(labels, &[PauliLabel::Z; 4]);
    }

    /// The total-parity string is the leftover leg, and its phase is real.
    #[test]
    fn test_total_parity_is_the_leftover_leg() {
        for enc in [compile(&jw(4)), compile(&balanced(4)), compile(&jw(5))] {
            let parity = unsafe { compact(enc.total_parity(enc.num_modes()).unwrap()) };
            assert_eq!(
                unsafe { ffi::qk_obs_num_terms(parity) },
                1,
                "total parity must be a single Pauli string"
            );
            // ... and that string is the encoding's leftover leg.
            let (qubits, labels) = enc.leg_image(enc.parity_leg());
            let expected = one_pauli_string(
                enc.num_modes(),
                // The phase is +-1 and tree-dependent; read it off rather than assume it.
                unsafe { *ffi::qk_obs_coeffs(parity) },
                &mut bit_terms(labels),
                &mut qubits.to_vec(),
            );
            assert_obs_equal(parity, expected, enc.num_modes());
        }
    }

    #[test]
    fn test_mode_map_permutes_the_pairing() {
        // A permuted mode map must relabel which mode each leg serves, without touching the strings
        // themselves.
        let tree = balanced(4);
        let identity = compile(&tree);
        let reversed = ModeMap::Permutation(vec![3, 2, 1, 0]);
        let permuted = TernaryTreeEncoding::new(&tree, &reversed).unwrap();

        for node in 0..4u32 {
            for offset in 0..2u32 {
                // Node `u` serves mode `3-u` under the reversed map.
                assert_eq!(
                    identity.majorana_leg[(2 * node + offset) as usize],
                    permuted.majorana_leg[(2 * (3 - node) + offset) as usize]
                );
            }
        }
        // The leftover leg is a property of the tree, not the mode labelling.
        assert_eq!(identity.parity_leg(), permuted.parity_leg());
    }

    #[test]
    fn test_images_span_only_the_trees_nodes() {
        // The compiled strings are a property of the tree alone: they never mention a qubit beyond its
        // nodes. That is what lets a mapper pad to any wider register without recompiling, since the
        // extra qubits simply carry the identity.
        let tree = jw(2);
        let enc = TernaryTreeEncoding::new(&tree, &ModeMap::Identity).unwrap();
        assert_eq!(enc.num_modes(), 2);
        for m in 0..4 {
            assert!(enc.majorana_image(m).0.iter().all(|&q| q < 2));
        }
    }

    #[test]
    fn test_rejects_bad_mode_map() {
        let tree = jw(3);
        assert_eq!(
            TernaryTreeEncoding::new(&tree, &ModeMap::Permutation(vec![0, 1])),
            Err(TernaryTreeError::ModeMapLength {
                len: 2,
                num_nodes: 3
            })
        );
        assert_eq!(
            TernaryTreeEncoding::new(&tree, &ModeMap::Permutation(vec![0, 1, 1])),
            Err(TernaryTreeError::ModeMapNotAPermutation { mode: 1 })
        );
        // An out-of-range entry is a *mode*, so it is reported as one; `NodeOutOfRange` would
        // describe the map's index rather than its value.
        assert_eq!(
            TernaryTreeEncoding::new(&tree, &ModeMap::Permutation(vec![0, 1, 9])),
            Err(TernaryTreeError::ModeOutOfRange {
                mode: 9,
                num_nodes: 3
            })
        );
    }

    #[test]
    fn test_max_weight_matches_the_tree() {
        // The encoding's weight is the tree's, restricted to the paired legs. They agree here because
        // the leftover leg is never the deepest, but assert rather than assume it.
        for n in [1u32, 4, 13, 40] {
            let tree = balanced(n);
            let enc = compile(&tree);
            assert!(enc.max_weight() <= tree.max_weight());
        }
        assert_eq!(compile(&balanced(13)).max_weight(), 3);
        assert_eq!(compile(&jw(5)).max_weight(), 5);
    }
}
