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

//! HATT: growing a ternary tree adapted to a particular Hamiltonian.

use super::SearchError;
use crate::mappers::ternary_tree::{MAX_NODES, NodeId, PauliLabel, TernaryTree};
use crate::operators::OperatorTrait;
use crate::operators::majorana_operator::MajoranaOperator;

/// Grows a ternary tree adapted to `hamiltonian`, following the HATT algorithm of Ref. 1.
///
/// The algorithm is topology-blind: it shapes the tree around which Majorana operators actually
/// appear together in the operator being mapped.
///
/// # The algorithm
///
/// This is the vacuum-preserving variant of Ref. 1 (its Algorithms 2 and 3). It works bottom-up and
/// greedily. The `2N+1` legs start out as separate subtrees, and leg `m` stands for Majorana operator
/// `m`, except the last, which is left over for the total parity. Each of the `N` steps picks two
/// subtrees `O_X` and `O_Z` and makes them the `X` and `Z` children of a new node. The `Y` child is
/// not free: if the leg reached by descending `Z` links from `O_X` carries `gamma_2l`, the `Y` child
/// is the subtree whose `Z`-descent carries its partner `gamma_2l+1`. The new node then hosts mode
/// `l`, and the returned tree numbers it `l`.
///
/// That derivation is the same rule [`TernaryTreeEncoding`](Encoding) uses to pair legs into
/// Majorana operators (Algorithm 1 of the Bonsai paper). So the encoding compiled from the
/// returned tree, under the identity mode map, assigns every Majorana operator to exactly the leg
/// the search tracked it on. That is what makes the cost below the real one.
///
/// Ref. 1 first describes an unconstrained variant (its Algorithm 1) that chooses all three
/// children freely. That variant does not preserve the vacuum, and Ref. 1 reports that its Pauli
/// weights differ from this variant's by about 0.43% on average. It is not implemented, because it
/// fixes leg `m` to Majorana operator `m`, whereas a [`TernaryTreeEncoding`](Encoding) derives
/// that assignment from the shape of the tree.
///
/// [Encoding]: crate::mappers::ternary_tree::encoding::TernaryTreeEncoding
///
/// # The cost
///
/// A step is scored by the Pauli weight the Hamiltonian acquires on the new qubit. A term acts there
/// as `X^a Y^b Z^c`, where `a`, `b` and `c` are the parities of how many of its Majorana operators
/// lie under each child. That is the identity, and the term costs nothing, exactly when the three
/// parities are equal: all even, or all odd, since `XYZ` is proportional to the identity. Summed
/// over all steps, the cost is therefore the total Pauli weight of the terms of `hamiltonian` as
/// given, each term's image counted on its own.
///
/// The terms are used exactly as given: nothing is normal-ordered, merged or dropped. A term and a
/// reordering of it that would cancel are both counted, and so is a term with a negligible
/// coefficient. If `hamiltonian` is normal-ordered and simplified first, its distinct terms map to
/// distinct Pauli strings, and the cost equals the total Pauli weight of the mapped operator after
/// the same simplification.
///
/// With `weighted == false`, each term that does not act as the identity contributes `1`, which is
/// the cost of Ref. 1 and ignores the coefficients. With `weighted == true`, each such term
/// contributes `|coefficient|` instead, which deviates from the published algorithm.
///
/// # Complexity
///
/// Ref. 1 states `O(N^3)` for this variant and leaves the cost evaluation out of the figure. Here,
/// because the `Y` child is a function of `O_X`, the cost of every `(O_X, O_Z)` pair is assembled
/// from per-subtree sums, giving `O(T k^2 + N^2)` per step for `T` terms of at most `k` Majorana
/// operators each, and `O(N T k^2 + N^3)` overall.
///
/// # Determinism
///
/// Ref. 1 does not specify how to break ties between equal-cost candidates. Ties go to the lowest
/// subtree indices, legs first in Majorana order and then nodes in the order they were created:
/// first by the lower of the `X` and `Y` children, then by the `Z` child. That is the order in
/// which a scan over the live subtrees first meets each candidate, and it reproduces the published
/// Fermi-Hubbard results wherever the Hamiltonian conventions agree. Because legs are numbered by
/// mode, relabeling the modes of `hamiltonian` changes which candidate wins a tie. Weighted costs are
/// compared with a tolerance relative to the total weight, so that floating-point rounding does not
/// decide a tie.
///
/// # References
///
/// 1. Y. Liu, K. Yao, J. Hong, J. Froustey, E. Rrapaj, C. Iancu, G. Li and Y. Shi, *HATT: Hamiltonian
///    Adaptive Ternary Tree for Optimizing Fermion-to-Qubit Mapping*,
///    [arXiv:2409.02010](https://arxiv.org/abs/2409.02010).
pub fn hatt_tree(
    hamiltonian: &MajoranaOperator,
    num_modes: u32,
    weighted: bool,
) -> Result<TernaryTree, SearchError> {
    grow(hamiltonian, num_modes, weighted).map(|(tree, _)| tree)
}

/// Runs the search, returning the tree together with the summed cost of its merges.
///
/// The total is exposed to the tests only: it must equal the mapped Pauli weight, which is the check
/// that the greedy objective describes the encoding actually produced.
fn grow(
    hamiltonian: &MajoranaOperator,
    num_modes: u32,
    weighted: bool,
) -> Result<(TernaryTree, f64), SearchError> {
    if num_modes == 0 {
        return Err(SearchError::NoModes);
    }
    if num_modes > MAX_NODES {
        return Err(SearchError::TooLarge {
            num_nodes: num_modes,
            max: MAX_NODES,
        });
    }
    if let Some(&max_mode) = hamiltonian.modes.iter().max()
        && max_mode / 2 >= num_modes
    {
        return Err(SearchError::ModeCountMismatch {
            num_modes,
            max_mode: max_mode / 2,
        });
    }

    // A non-finite coefficient would make every weighted comparison meaningless (a NaN sorts below
    // every number), and in a Hamiltonian it is a bug upstream rather than a value to optimize for.
    if let Some(term) = hamiltonian
        .iter()
        .find(|term| !(term.coeff.re.is_finite() && term.coeff.im.is_finite()))
    {
        return Err(SearchError::NonFiniteCoefficient {
            modes: term.modes.to_vec(),
        });
    }

    // Weighted costs only ever compare sums of `|coefficient|`, so dividing every coefficient by the
    // largest component magnitude changes no choice. It keeps each weight at most `sqrt(2)`, so no
    // magnitude or sum of them overflows, however large the finite coefficients are. The constant
    // term is left out, as from the cost: a huge constant would otherwise flush every weight to zero.
    let scale = hamiltonian
        .iter()
        .filter(|term| !term.modes.is_empty())
        .map(|term| term.coeff.re.abs().max(term.coeff.im.abs()))
        .fold(0.0, f64::max);
    let scale = if scale > 0.0 { scale } else { 1.0 };

    // The terms, as Majorana index lists with their weights. Extracted once: every step re-reads
    // them, so re-walking the operator would dominate.
    let terms: Vec<(Vec<u32>, f64)> = hamiltonian
        .iter()
        .filter(|term| !term.modes.is_empty())
        .map(|term| {
            let weight = if weighted {
                (term.coeff / scale).norm()
            } else {
                1.0
            };
            (term.modes.to_vec(), weight)
        })
        .collect();
    // Costs within this of each other tie. Every cost is a signed sum of term weights, so rounding
    // errors scale with the total weight; unweighted costs are small integers and stay exact.
    let tolerance = TIE_TOLERANCE * terms.iter().map(|(_, weight)| weight).sum::<f64>();

    let mut builder = TreeBuilder::new(num_modes, terms.len());
    let mut total = 0.0;
    for _ in 0..num_modes {
        let (merge, merge_cost) = builder
            .best_merge(&terms, tolerance)
            .expect("every step before the tree is complete has a candidate merge");
        total += merge_cost;
        builder.merge(merge);
    }

    // The total is reported in the coefficients' own units, so undo the scaling.
    let total = if weighted { total * scale } else { total };
    Ok((builder.finish(), total))
}

/// The relative tolerance within which two candidate costs count as a tie.
const TIE_TOLERANCE: f64 = 1e-12;

/// A subtree, identified as in Ref. 1: `0..=2N` are the legs, and `2N + 1 + i` is the node created in
/// step `i`.
type Subtree = usize;

/// Bottom-up accumulator for a HATT tree.
///
/// Each merge consumes **three** live subtrees (the new node's `X`, `Y` and `Z` children) and produces
/// one, so the live set shrinks by two per step and holds exactly the root after `N` steps.
struct TreeBuilder {
    /// The number of modes, `N`.
    num_modes: usize,
    /// The live subtrees, in the order they became live.
    live: Vec<Subtree>,
    /// For every subtree, the leg reached by descending `Z` links from its root.
    z_descent: Vec<usize>,
    /// For every leg that is the `Z`-descent of a live subtree, that subtree.
    ///
    /// By Lemma 1 of Ref. 1, a leg is still unpaired exactly when it is the `Z`-descent of a live
    /// subtree, so this finds the `Y` child in constant time.
    z_ancestor: Vec<Subtree>,
    /// For every leg, the live subtree containing it.
    owner: Vec<Subtree>,
    /// For every live subtree, the legs beneath it.
    legs: Vec<Vec<usize>>,
    /// For every node, in creation order: the mode it hosts and its `X`, `Y` and `Z` children.
    nodes: Vec<(u32, [Subtree; 3])>,
    /// Scratch space reused by every step, sized once.
    scratch: Scratch,
}

/// A candidate merge, as compared by [`TreeBuilder::best_merge`].
#[derive(Clone, Copy)]
struct Merge {
    /// The Hamiltonian's Pauli weight on the new qubit.
    cost: f64,
    /// The `X`, `Y` and `Z` children.
    children: [Subtree; 3],
}

impl Merge {
    /// The tie-break key: the lower of the `X` and `Y` children, then the `Z` child.
    ///
    /// A scan over the live subtrees in index order meets the pair `{X, Y}` first at the lower of
    /// the two, whichever of them carries the even Majorana operator, so this is the order in which
    /// such a scan first encounters each candidate. No two candidates share a key, since `X` fixes
    /// `Y`.
    fn key(&self) -> (Subtree, Subtree) {
        let [x, y, z] = self.children;
        (x.min(y), z)
    }
}

/// The candidates whose cost lies within a tolerance of the lowest cost seen so far.
///
/// Comparing candidates pairwise with a tolerance is not transitive, so the winner of a scan would
/// depend on the scan order. Keeping every near-tie until the scan ends and only then taking the
/// lowest [`Merge::key`] makes the choice "the lowest key among the costs within `tolerance` of the
/// minimum", whatever the order.
struct NearTies {
    /// The lowest cost offered so far.
    lowest: f64,
    /// Every offered candidate within the tolerance of `lowest`.
    candidates: Vec<Merge>,
}

impl NearTies {
    fn new() -> Self {
        Self {
            lowest: f64::INFINITY,
            candidates: Vec::new(),
        }
    }

    fn clear(&mut self) {
        self.lowest = f64::INFINITY;
        self.candidates.clear();
    }

    /// Considers `merge`, dropping every kept candidate it beats by more than `tolerance`.
    fn offer(&mut self, merge: Merge, tolerance: f64) {
        if merge.cost < self.lowest {
            self.lowest = merge.cost;
            let bound = self.lowest + tolerance;
            self.candidates.retain(|kept| kept.cost <= bound);
        }
        if merge.cost <= self.lowest + tolerance {
            self.candidates.push(merge);
        }
    }

    /// The kept candidate with the lowest key.
    fn best(&self) -> Option<Merge> {
        self.candidates.iter().copied().min_by_key(Merge::key)
    }
}

/// Per-step buffers of [`TreeBuilder::best_merge`], kept so that a step allocates nothing.
struct Scratch {
    /// For every term, the live subtrees holding an odd number of its Majorana operators.
    odd: Vec<Vec<Subtree>>,
    /// For every subtree, the terms it holds an odd number of Majorana operators of.
    terms_of: Vec<Vec<usize>>,
    /// For every subtree, the summed weight of `terms_of`.
    odd_weight: Vec<f64>,
    /// For every subtree, the correction to subtract when it is the `Z` child.
    correction: Vec<f64>,
    /// The subtrees whose `correction` is non-zero.
    touched: Vec<Subtree>,
    /// For every term, the last `X` candidate that visited it, to visit each term once.
    visited_by: Vec<usize>,
    /// The candidates of the current step that are still in the running.
    near: NearTies,
}

impl TreeBuilder {
    /// Starts from the `2N+1` legs, each its own live subtree.
    fn new(num_modes: u32, num_terms: usize) -> Self {
        let num_modes = num_modes as usize;
        let num_legs = 2 * num_modes + 1;
        let num_subtrees = num_legs + num_modes;
        let mut z_descent = vec![usize::MAX; num_subtrees];
        let mut legs = vec![Vec::new(); num_subtrees];
        for leg in 0..num_legs {
            z_descent[leg] = leg;
            legs[leg].push(leg);
        }
        Self {
            num_modes,
            live: (0..num_legs).collect(),
            z_descent,
            z_ancestor: (0..num_legs).collect(),
            owner: (0..num_legs).collect(),
            legs,
            nodes: Vec::with_capacity(num_modes),
            scratch: Scratch {
                odd: vec![Vec::new(); num_terms],
                terms_of: vec![Vec::new(); num_subtrees],
                odd_weight: vec![0.0; num_subtrees],
                correction: vec![0.0; num_subtrees],
                touched: Vec::new(),
                visited_by: vec![usize::MAX; num_terms],
                near: NearTies::new(),
            },
        }
    }

    /// Returns the cheapest merge `(X, Y, Z)` and its cost, or `None` if no merge is possible.
    ///
    /// For a merge, let `a`, `b` and `c` say whether a term has an odd number of Majorana operators
    /// under `X`, `Y` and `Z`. The term costs its weight unless `a == b == c`. Since `Y` is fixed by
    /// `X`, the cost of `(X, Z)` is assembled as
    ///
    /// ```text
    ///     base(X) + odd_weight(Z) - correction(X, Z)
    /// ```
    ///
    /// where `base(X)` sums the weights of the terms with `a || b`, and `odd_weight(Z)` those of the
    /// terms with `c`. A term with both `a || b` and `c` is counted twice by the two sums, so the
    /// correction subtracts it once, or twice when also `a && b`, since all three parities then agree
    /// and the term costs nothing. Only terms touching `X` or `Y` need a correction, which is what
    /// makes the per-step cost independent of the number of `(X, Z)` pairs.
    ///
    /// Among the candidates whose cost is within `tolerance` of the lowest, the one with the lowest
    /// [`Merge::key`] wins; see [`NearTies`].
    fn best_merge(
        &mut self,
        terms: &[(Vec<u32>, f64)],
        tolerance: f64,
    ) -> Option<([Subtree; 3], f64)> {
        let Scratch {
            odd,
            terms_of,
            odd_weight,
            correction,
            touched,
            visited_by,
            near,
        } = &mut self.scratch;

        // The odd-parity subtrees of every term, and the reverse index.
        for &subtree in &self.live {
            terms_of[subtree].clear();
            odd_weight[subtree] = 0.0;
        }
        for (index, (modes, weight)) in terms.iter().enumerate() {
            let owners = &mut odd[index];
            owners.clear();
            owners.extend(modes.iter().map(|&m| self.owner[m as usize]));
            owners.sort_unstable();
            // Keep each subtree that appears an odd number of times, once.
            let mut kept = 0;
            let mut i = 0;
            while i < owners.len() {
                let mut j = i;
                while j < owners.len() && owners[j] == owners[i] {
                    j += 1;
                }
                if (j - i) % 2 == 1 {
                    owners[kept] = owners[i];
                    kept += 1;
                }
                i = j;
            }
            owners.truncate(kept);
            for &subtree in owners.iter() {
                terms_of[subtree].push(index);
                odd_weight[subtree] += weight;
            }
            visited_by[index] = usize::MAX;
        }

        let spare = 2 * self.num_modes;
        near.clear();
        for &x in &self.live {
            // Only an `X` child whose `Z`-descent is an even Majorana operator is a valid choice: the
            // spare leg never pairs, and an odd one is the same merge as its partner with `X` and `Y`
            // swapped, which Ref. 1 normalizes to this one.
            let leg = self.z_descent[x];
            if leg == spare || leg % 2 == 1 {
                continue;
            }
            let y = self.z_ancestor[leg + 1];

            let mut base = 0.0;
            for &index in terms_of[x].iter().chain(&terms_of[y]) {
                if visited_by[index] == x {
                    continue;
                }
                visited_by[index] = x;
                let owners = &odd[index];
                let weight = terms[index].1;
                base += weight;
                // `owners` is sorted, so membership is a binary search over at most `k` entries.
                let both = owners.binary_search(&x).is_ok() && owners.binary_search(&y).is_ok();
                for &z in owners {
                    if z == x || z == y {
                        continue;
                    }
                    if correction[z] == 0.0 {
                        touched.push(z);
                    }
                    correction[z] += if both { 2.0 * weight } else { weight };
                }
            }

            for &z in &self.live {
                if z == x || z == y {
                    continue;
                }
                let cost = base + odd_weight[z] - correction[z];
                near.offer(
                    Merge {
                        cost,
                        children: [x, y, z],
                    },
                    tolerance,
                );
            }

            for z in touched.drain(..) {
                correction[z] = 0.0;
            }
        }

        near.best().map(|merge| (merge.children, merge.cost))
    }

    /// Merges three live subtrees as the `X`, `Y` and `Z` children of a new node.
    fn merge(&mut self, children: [Subtree; 3]) {
        let [x, _, z] = children;
        let node = 2 * self.num_modes + 1 + self.nodes.len();
        // The node pairs the even Majorana operator under `X` with its partner under `Y`.
        let mode = (self.z_descent[x] / 2) as u32;
        self.nodes.push((mode, children));

        // The node's `Z`-descent is its `Z` child's, and that leg stays unpaired.
        let leg = self.z_descent[z];
        self.z_descent[node] = leg;
        self.z_ancestor[leg] = node;

        let mut legs = Vec::new();
        for child in children {
            for &leg in &self.legs[child] {
                self.owner[leg] = node;
            }
            legs.append(&mut self.legs[child]);
        }
        self.legs[node] = legs;

        self.live.retain(|subtree| !children.contains(subtree));
        self.live.push(node);
    }

    /// Converts the merges into a tree, numbering each node by the mode it hosts.
    fn finish(self) -> TernaryTree {
        let first_node = 2 * self.num_modes + 1;
        let mut spec: Vec<Option<(NodeId, PauliLabel)>> = vec![None; self.num_modes];
        for &(mode, children) in &self.nodes {
            for (child, label) in
                children
                    .into_iter()
                    .zip([PauliLabel::X, PauliLabel::Y, PauliLabel::Z])
            {
                // A leg child leaves the link as a leg, which `try_new` fills in.
                if child >= first_node {
                    let (child_mode, _) = self.nodes[child - first_node];
                    spec[child_mode as usize] = Some((mode, label));
                }
            }
        }
        // `try_new` re-checks connectivity and the label invariants, so a bug here panics rather than
        // producing a wrong encoding.
        TernaryTree::try_new(&spec).expect("the merges always describe a valid tree")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mappers::ternary_tree::encoding::{ModeMap, TernaryTreeEncoding};
    use num_complex::Complex64;

    fn majorana_op(terms: &[(&[u32], f64)]) -> MajoranaOperator {
        let mut coeffs = Vec::new();
        let mut modes = Vec::new();
        let mut boundaries = vec![0usize];
        for (term, coeff) in terms {
            coeffs.push(Complex64::new(*coeff, 0.0));
            modes.extend_from_slice(term);
            boundaries.push(modes.len());
        }
        MajoranaOperator {
            coeffs,
            modes,
            boundaries,
            groups: None,
        }
    }

    /// The Pauli weight of every term of `hamiltonian` under `tree`'s encoding, summed, weighting
    /// each term as the search does for the same `weighted`.
    ///
    /// Computed from the encoding's Majorana images alone, independently of the search: per qubit,
    /// a term's Pauli operator is the product of its factors' labels there, which is the identity
    /// exactly when the parities of `X`, `Y` and `Z` factors agree.
    fn mapped_weight(hamiltonian: &MajoranaOperator, tree: &TernaryTree, weighted: bool) -> f64 {
        let encoding = TernaryTreeEncoding::new(tree, &ModeMap::Identity).unwrap();
        let num_qubits = tree.num_nodes() as usize;
        let mut total = 0.0;
        for term in hamiltonian.iter() {
            let mut parities = vec![[false; 3]; num_qubits];
            for &majorana in term.modes {
                let (labels, qubits) = encoding.majorana_image(majorana);
                for (&label, &qubit) in labels.iter().zip(qubits) {
                    let slot = match label {
                        PauliLabel::X => 0,
                        PauliLabel::Y => 1,
                        PauliLabel::Z => 2,
                    };
                    parities[qubit as usize][slot] ^= true;
                }
            }
            let weight = parities
                .iter()
                .filter(|[a, b, c]| !(a == b && b == c))
                .count() as f64;
            total += weight * if weighted { term.coeff.norm() } else { 1.0 };
        }
        total
    }

    /// A deterministic pseudo-random Hamiltonian: `num_terms` terms of 1 to 4 Majorana operators.
    fn pseudo_random(num_modes: u32, num_terms: usize, seed: u64) -> MajoranaOperator {
        let mut state = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        let mut owned: Vec<(Vec<u32>, f64)> = Vec::new();
        for _ in 0..num_terms {
            let len = 1 + next() % 4;
            let modes: Vec<u32> = (0..len).map(|_| next() % (2 * num_modes)).collect();
            let coeff = 0.1 + f64::from(next() % 100) / 10.0;
            owned.push((modes, coeff));
        }
        let refs: Vec<(&[u32], f64)> = owned.iter().map(|(m, c)| (m.as_slice(), *c)).collect();
        majorana_op(&refs)
    }

    #[test]
    fn test_cost_is_the_mapped_pauli_weight() {
        // The search's objective must describe the encoding it returns. If the search tracked
        // Majorana operators on other legs than the encoding pairs them onto, or scored a term that
        // is the identity on a qubit, the two would disagree.
        for num_modes in 1..=7u32 {
            for seed in 0..6u64 {
                let hamiltonian = pseudo_random(num_modes, 3 * num_modes as usize, seed);
                for weighted in [false, true] {
                    let (tree, total) = grow(&hamiltonian, num_modes, weighted).unwrap();
                    let expected = mapped_weight(&hamiltonian, &tree, weighted);
                    assert!(
                        (total - expected).abs() <= 1e-9 * expected.max(1.0),
                        "n={num_modes} seed={seed} weighted={weighted}: search cost {total} but the \
                         encoding maps to Pauli weight {expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_reproduces_the_worked_example() {
        // Ref. 1, Eq. (3): `a0^ a0 + 2 a1^ a2^ a1 a2` in Majorana form. Its first step merges legs 0,
        // 1 and the spare leg 6 at cost 1, the unique minimum, so mode 0 sits above the spare leg.
        let hamiltonian = majorana_op(&[
            (&[0, 1], 0.5),
            (&[2, 3], -0.5),
            (&[4, 5], -0.5),
            (&[2, 3, 4, 5], 0.5),
        ]);
        let (tree, total) = grow(&hamiltonian, 3, false).unwrap();
        for label in PauliLabel::iter() {
            assert!(matches!(
                tree.child_of(0, label),
                crate::mappers::ternary_tree::Slot::Leg(_)
            ));
        }
        // The spare leg under node 0's `Z` link is the leftover: the root's `Z`-descent reaches it.
        let mut node = tree.root();
        while let crate::mappers::ternary_tree::Slot::Edge(child) =
            tree.child_of(node, PauliLabel::Z)
        {
            node = child;
        }
        assert_eq!(node, 0);
        // Step 1 costs 1, and each later step 2. In step 2 the lowest-index tie-break picks the
        // children (2, 3, 4), where Ref. 1's illustration picks (2, 3, 7) at the same cost, so from
        // here the tree differs from the paper's figure while every step costs the same.
        assert_eq!(total, 5.0);
        assert_eq!(mapped_weight(&hamiltonian, &tree, false), 5.0);
    }

    #[test]
    fn test_all_odd_term_costs_nothing() {
        // `gamma_0 gamma_1 gamma_2` has one factor under each child of a node whose children are
        // legs 0, 1 and 2, and so is proportional to the identity there: `X Y Z = i I`.
        let hamiltonian = majorana_op(&[(&[0, 1, 2], 1.0)]);
        let (tree, total) = grow(&hamiltonian, 2, false).unwrap();
        assert_eq!(total, mapped_weight(&hamiltonian, &tree, false));
        // The best a 2-mode tree can do: identity on the qubit holding all three, one Pauli above.
        assert_eq!(total, 1.0);
    }

    #[test]
    fn test_best_merge_is_the_brute_force_minimum() {
        // The assembled cost `base + odd_weight - correction` must agree with a direct count for
        // *every* candidate, not just the chosen one, or a formula that overestimates the merges it
        // does not pick would go unnoticed. So at every step, enumerate the candidates by hand, score
        // each from the parities of its children directly, and check the search picks the same one.
        for num_modes in 1..=5u32 {
            for seed in 0..6u64 {
                for weighted in [false, true] {
                    let hamiltonian = pseudo_random(num_modes, 3 * num_modes as usize, seed);
                    let terms: Vec<(Vec<u32>, f64)> = hamiltonian
                        .iter()
                        .filter(|term| !term.modes.is_empty())
                        .map(|term| {
                            let weight = if weighted { term.coeff.norm() } else { 1.0 };
                            (term.modes.to_vec(), weight)
                        })
                        .collect();
                    let tolerance = TIE_TOLERANCE * terms.iter().map(|(_, w)| w).sum::<f64>();
                    let mut builder = TreeBuilder::new(num_modes, terms.len());
                    let spare = 2 * num_modes as usize;
                    for step in 0..num_modes {
                        let mut candidates: Vec<Merge> = Vec::new();
                        for &x in &builder.live {
                            let leg = builder.z_descent[x];
                            if leg == spare || leg % 2 == 1 {
                                continue;
                            }
                            let y = builder.z_ancestor[leg + 1];
                            for &z in &builder.live {
                                if z == x || z == y {
                                    continue;
                                }
                                let cost: f64 = terms
                                    .iter()
                                    .filter(|(modes, _)| {
                                        let parity = |child: Subtree| {
                                            modes
                                                .iter()
                                                .filter(|&&m| builder.owner[m as usize] == child)
                                                .count()
                                                % 2
                                        };
                                        let (a, b, c) = (parity(x), parity(y), parity(z));
                                        !(a == b && b == c)
                                    })
                                    .map(|(_, weight)| weight)
                                    .sum();
                                candidates.push(Merge {
                                    cost,
                                    children: [x, y, z],
                                });
                            }
                        }
                        // The documented rule, applied in two passes: the lowest cost, then the
                        // lowest key among the costs within the tolerance of it.
                        let lowest = candidates
                            .iter()
                            .map(|m| m.cost)
                            .fold(f64::INFINITY, f64::min);
                        let expected = candidates
                            .iter()
                            .filter(|m| m.cost <= lowest + tolerance)
                            .min_by_key(|m| m.key())
                            .unwrap();
                        let (children, cost) = builder.best_merge(&terms, tolerance).unwrap();
                        let context = format!("n={num_modes} seed={seed} w={weighted} step={step}");
                        assert!((cost - expected.cost).abs() <= tolerance, "{context}");
                        assert_eq!(children, expected.children, "{context}");
                        builder.merge(children);
                    }
                }
            }
        }
    }

    #[test]
    fn test_near_ties_do_not_depend_on_the_scan_order() {
        // Three costs spaced 0.6 tolerance apart: each neighbor is within the tolerance, the outer
        // two are not. A pairwise comparison would let the scan order pick the winner; the rule picks
        // the lowest key among the costs within the tolerance of the minimum, here keys 1 and 2.
        let merge = |cost: f64, x: Subtree| Merge {
            cost,
            children: [x, 100, 200],
        };
        let candidates = [merge(1.2, 0), merge(0.6, 1), merge(0.0, 2)];
        let orders: [[usize; 3]; 3] = [[0, 1, 2], [2, 1, 0], [1, 0, 2]];
        for order in orders {
            let mut near = NearTies::new();
            for index in order {
                near.offer(candidates[index], 1.0);
            }
            assert_eq!(near.best().unwrap().children[0], 1, "order {order:?}");
        }
    }

    #[test]
    fn test_ties_go_to_the_lowest_indices() {
        // With no terms every candidate is free, so the tie-break alone builds the tree. Step 1 takes
        // legs 0 and 1 (the pair with the lowest index) with leg 2, the lowest remaining, under `Z`.
        // Step 2 then has a single candidate: the new node, whose `Z`-descent carries Majorana 2,
        // paired with leg 3, over the spare leg 4.
        let tree = hatt_tree(&MajoranaOperator::zero(), 2, false).unwrap();
        let expected = TernaryTree::try_new(&[Some((1, PauliLabel::X)), None]).unwrap();
        assert_eq!(tree, expected);
    }

    #[test]
    fn test_weighting_can_change_the_tree() {
        // Counting terms and weighting them by magnitude are different objectives, so on some
        // Hamiltonians they must grow different trees, or the switch would do nothing. Neither is
        // claimed to win on the other's objective: both are greedy, so neither is globally optimal.
        let differs = (0..20u64).any(|seed| {
            let hamiltonian = pseudo_random(5, 12, seed);
            hatt_tree(&hamiltonian, 5, false) != hatt_tree(&hamiltonian, 5, true)
        });
        assert!(differs, "the coefficient weights never changed the tree");
    }

    #[test]
    fn test_huge_finite_coefficients_do_not_overflow() {
        // Each coefficient is finite, but their magnitudes sum past `f64::MAX`. Unscaled, the costs
        // became `inf - inf = NaN` and no candidate survived; scaled, the search matches the same
        // Hamiltonian at an ordinary scale.
        let terms: [&[u32]; 4] = [&[0, 2, 3], &[1], &[1, 2, 3], &[0, 1, 3]];
        let at = |coeff: f64| majorana_op(&terms.map(|term| (term, coeff)));
        assert_eq!(
            hatt_tree(&at(1e308), 2, true).unwrap(),
            hatt_tree(&at(1.0), 2, true).unwrap()
        );
    }

    #[test]
    fn test_a_huge_constant_term_does_not_flush_the_weights() {
        // The constant term costs nothing, so it must not set the scale either: dividing by it would
        // flush these weights to zero and leave every choice to the tie-break.
        let terms: [(&[u32], f64); 4] = [
            (&[0, 3], 1e-3),
            (&[1, 2], 1e-3),
            (&[2, 5], 1e-3),
            (&[0, 4], 1.0),
        ];
        let reference = hatt_tree(&majorana_op(&terms), 3, true).unwrap();
        assert_ne!(
            reference,
            hatt_tree(&majorana_op(&terms), 3, false).unwrap(),
            "test premise: the weights decide the tree"
        );
        let mut with_constant = terms.map(|(modes, coeff)| (modes, coeff * 1e-30)).to_vec();
        with_constant.push((&[], 1e300));
        assert_eq!(
            hatt_tree(&majorana_op(&with_constant), 3, true).unwrap(),
            reference
        );
    }

    #[test]
    fn test_adapts_to_the_hamiltonian() {
        // The point of HATT: the tree it grows should depend on the Hamiltonian it is given. Two
        // operators with different term structure on the same mode count should not generally give
        // the same tree.
        let clustered = majorana_op(&[(&[0, 1], 1.0), (&[0, 2], 1.0), (&[1, 2], 1.0)]);
        let spread = majorana_op(&[(&[0, 7], 1.0), (&[2, 5], 1.0), (&[3, 6], 1.0)]);
        let from_clustered = hatt_tree(&clustered, 4, false).unwrap();
        let from_spread = hatt_tree(&spread, 4, false).unwrap();

        // Both valid ...
        assert_eq!(from_clustered.num_legs(), 9);
        assert_eq!(from_spread.num_legs(), 9);
        // ... and the greedy objective genuinely responded to the input.
        assert_ne!(
            from_clustered, from_spread,
            "the tree should adapt to the Hamiltonian's term structure"
        );
    }

    #[test]
    fn test_rejects_bad_input() {
        let hamiltonian = majorana_op(&[(&[0, 1], 1.0)]);
        assert_eq!(hatt_tree(&hamiltonian, 0, false), Err(SearchError::NoModes));
        // Rejected before anything of that size is allocated.
        assert_eq!(
            hatt_tree(&hamiltonian, u32::MAX, false),
            Err(SearchError::TooLarge {
                num_nodes: u32::MAX,
                max: MAX_NODES
            })
        );
        // Majorana 15 acts on mode 7, which a 4-mode encoding does not cover.
        let too_wide = majorana_op(&[(&[15], 1.0)]);
        assert_eq!(
            hatt_tree(&too_wide, 4, false),
            Err(SearchError::ModeCountMismatch {
                num_modes: 4,
                max_mode: 7
            })
        );
        // A non-finite coefficient is rejected whether or not the coefficients are used.
        for bad in [f64::NAN, -f64::NAN, f64::INFINITY] {
            let hamiltonian = majorana_op(&[(&[0, 1], 1.0), (&[1, 2], bad)]);
            for weighted in [false, true] {
                assert_eq!(
                    hatt_tree(&hamiltonian, 2, weighted),
                    Err(SearchError::NonFiniteCoefficient { modes: vec![1, 2] })
                );
            }
        }
    }

    #[test]
    fn test_empty_hamiltonian_still_builds() {
        // Every candidate scores zero, so the tie-break alone decides. It must still produce a tree.
        let empty = MajoranaOperator::zero();
        let tree = hatt_tree(&empty, 4, false).unwrap();
        assert_eq!(tree.num_nodes(), 4);
        assert_eq!(tree.num_legs(), 9);
    }
}
