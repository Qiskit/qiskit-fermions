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

//! The *clash graph* of an operator: which of its terms fail to commute.
//!
//! Every operator term is a node; an edge joins two terms that do not commute. Commutation is a
//! relation between two whole terms, so the graph is an all-to-all comparison over the terms: `O(T^2)`
//! pairs, equivalently an adjacency matrix over them.
//!
//! This is the fermionic analogue of Qiskit's `PauliList.noncommutation_graph`, one level up the
//! stack: the relation is read off the fermionic algebra rather than off a qubit mapping, so it holds
//! independently of which mapper is applied later.
//!
//! Where the sibling [`super::electronic_structure`] keys terms by their index structure, this module
//! keys them by their commutation relations, and colors the resulting graph to obtain groups whose
//! terms mutually commute.

use std::cmp::Reverse;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

use num_complex::Complex64;
use rayon::prelude::*;
use thiserror::Error;

use crate::operators::OperatorTrait;
use crate::operators::terms::majorana_expansion::{
    MajoranaExpansion, Monomial, Support, compose_into, count_shared_generators,
};

/// The largest degree a [`FermionOperator`](crate::operators::fermion_operator::FermionOperator)
/// term may have for the clash graph to be computed.
///
/// A fermionic term of degree `d` expands into `2^d` Majorana monomials, and every term's expansion is
/// held in memory for the whole computation. At this bound a single term already expands into about a
/// million monomials. Past it, the expansions alone could exhaust memory, which aborts the process
/// rather than raising an error, so such a term is refused before anything is allocated.
///
/// The bound applies to single terms, not to pairs. Comparing two terms that are both odd or that
/// overlap costs the product of their expansions, in time and, through the sums `terms_commute`
/// keeps, in memory. Two such terms of high degree can therefore still exhaust memory well within the
/// bound. Such a pair would take an impractical time to finish anyway, so this is not guarded.
pub const MAX_FERMION_TERM_DEGREE: usize = 20;

/// The largest number of terms an operator may have for the clash graph to be computed.
///
/// Terms are identified by `u32` indices, in the edges and in the group indices alike, so this is the
/// number of distinct values a `u32` holds.
pub const MAX_CLASH_TERMS: u64 = 1 << 32;

/// One edge of the clash graph: the indices of two terms that do not commute, smaller first.
///
/// An array rather than a tuple, because an array's layout is guaranteed: a list of edges is then
/// also a flat list of twice as many indices, which the C API hands out without copying.
pub type Edge = [u32; 2];

/// Error cases of the clash-graph computations.
#[derive(Error, Debug)]
pub enum ClashError {
    #[error(
        "the operator has {num_terms} terms, but the clash graph supports at most {max}, \
        since it indexes terms with 32-bit integers",
        max = MAX_CLASH_TERMS
    )]
    TooManyTerms { num_terms: usize },
    #[error(
        "term {term} has degree {degree}, but the clash graph supports terms of degree at most {max}",
        max = MAX_FERMION_TERM_DEGREE
    )]
    DegreeTooLarge { term: usize, degree: usize },
}

/// Whether two Majorana monomials commute.
///
/// For products of distinct Majorana generators `A` and `B`,
///
/// ```text
///     A B = (-1)^(|A| |B| - |A intersect B|) B A
/// ```
///
/// so they commute exactly when `|A| |B| - |A intersect B|` is even. The product `|A| |B|` is odd only
/// when both factors are, which reduces the whole parity to two list lengths and one merge walk, with
/// no multiplication.
#[inline]
fn monomials_commute(left: &[u64], right: &[u64]) -> bool {
    let both_odd = left.len() & right.len() & 1;
    let shared_odd = count_shared_generators(left, right) & 1;
    (both_odd ^ shared_odd) == 0
}

/// One term's Majorana expansion, together with the summary the pair loop screens on.
struct Expansion {
    /// The term as a sum of monomials, with distinct supports and no identity component.
    monomials: Vec<Monomial>,
    /// Every generator that any of the monomials touches: the union of their supports.
    support: Support,
    /// Whether the monomials hold an odd number of generators. They all agree: each factor of a term
    /// contributes one generator to every monomial, and a generator repeated within a term cancels in
    /// pairs, so every monomial has the parity of the term's degree.
    odd: bool,
}

impl Expansion {
    /// Summarizes a term's monomials, first merging those that share a support.
    ///
    /// A fermionic term repeats supports: the two choices for `a^dagger_p a_p` that pick the same
    /// generator twice both collapse onto the identity. Merging sums their coefficients, and drops any
    /// that cancel, so the pair loop runs over each support once. The identity component is dropped as
    /// well: it commutes with everything, so it never contributes to a commutator. The sums are exact
    /// for the reason given at [`terms_commute`], so neither drop needs a tolerance.
    ///
    /// A single monomial is left alone, so the single-monomial representations keep their fast path
    /// unchanged.
    fn new(monomials: Vec<Monomial>) -> Self {
        let monomials = if monomials.len() < 2 {
            monomials
        } else {
            let mut merged: HashMap<Support, Complex64> = HashMap::with_capacity(monomials.len());
            for (support, coeff) in monomials {
                *merged.entry(support).or_insert(Complex64::ZERO) += coeff;
            }
            // A `HashMap` iterates in an arbitrary order, which can differ from run to run, so the
            // monomials come out in an arbitrary order too. Nothing depends on that order: every
            // answer derived from them is an exact sum (see `terms_commute`), which reordering its
            // terms cannot change.
            merged
                .into_iter()
                .filter(|(support, coeff)| *coeff != Complex64::ZERO && !support.is_empty())
                .collect()
        };

        let mut support: Support = monomials
            .iter()
            .flat_map(|(monomial, _)| monomial.iter().copied())
            .collect();
        support.sort_unstable();
        support.dedup();
        let odd = monomials
            .first()
            .is_some_and(|(monomial, _)| monomial.len() % 2 == 1);
        Self {
            monomials,
            support,
            odd,
        }
    }
}

/// Whether two supports share a generator.
///
/// The same merge walk as [`count_shared_generators`], but stopping at the first common entry.
fn supports_overlap(left: &[u64], right: &[u64]) -> bool {
    let (mut i, mut j) = (0, 0);
    while i < left.len() && j < right.len() {
        match left[i].cmp(&right[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => return true,
        }
    }
    false
}

/// Working memory for [`terms_commute`], reused across every pair one worker tests.
#[derive(Default)]
struct Scratch {
    /// The running sums of the contributions sharing a product support.
    sums: HashMap<Support, Complex64>,
    /// Supports evicted from `sums` when it was last cleared, kept only for their allocations.
    spare: Vec<Support>,
    /// The product of the monomial pair under test.
    product: Support,
}

/// Whether two terms, given as their Majorana expansions, commute.
///
/// The single-monomial case is the common one (every term of a Majorana, edge-vertex or
/// transfer-vertex operator expands to exactly one monomial) and reduces to the parity bit above.
///
/// Before the general case comes a cheap screen. When the supports of the two terms share no
/// generator, every monomial pair has `|A intersect B| = 0`, so it commutes exactly when the two are
/// not both odd. When at least one term is even, that settles it without looking at a single monomial
/// pair. Two odd terms on disjoint supports anticommute pair by pair, and those contributions can
/// still cancel, so they go through the general case.
///
/// In general a term is a *sum* of monomials. Only *anticommuting* monomial pairs contribute to
/// `[A, B]`, each contributing `2 c_i d_j A_i B_j`, but those contributions can cancel against one
/// another - so the commutator vanishes exactly when the contributions sharing a product support sum
/// to zero. Testing merely whether every monomial pair commutes would be sound but strictly too
/// strong: it reports a clash for terms that genuinely commute through cancellation, which would
/// inflate the number of groups.
///
/// The zero test is exact, with no tolerance. The expansion drops each term's own coefficient, so
/// every `c_i` of a degree-`d` term is a power of `i` times `2^-d`, and every contribution is a power
/// of `i` times `2^(1 - d1 - d2)`. The real and imaginary parts of each running sum are therefore
/// integer multiples of that one power of two, at most `2^(d1 + d2)` of them, which an `f64`
/// represents exactly whenever `d1 + d2 <= 53`. The loop below runs `2^(d1 + d2)` times, so a pair
/// past that bound could never be finished anyway. A tolerance could only do harm: the contributions
/// have that fixed size regardless of the operator's coefficients, so any tolerance above it would
/// declare genuinely non-commuting terms to commute.
///
/// `scratch` holds those sums. It is reused rather than created, so a whole worker's share of the
/// `O(T^2)` sweep runs without allocating: each product is composed into one shared buffer and
/// looked up by reference, and only a product not seen before for this pair takes a key, recycled
/// from the supports the previous pair left behind.
fn terms_commute(left: &Expansion, right: &Expansion, scratch: &mut Scratch) -> bool {
    if let ([(left_support, _)], [(right_support, _)]) =
        (left.monomials.as_slice(), right.monomials.as_slice())
    {
        return monomials_commute(left_support, right_support);
    }
    let screened_out = !(left.odd && right.odd || supports_overlap(&left.support, &right.support));
    if screened_out {
        return true;
    }

    let Scratch {
        sums,
        spare,
        product,
    } = scratch;
    spare.extend(sums.drain().map(|(support, _)| support));
    for (left_support, left_coeff) in &left.monomials {
        for (right_support, right_coeff) in &right.monomials {
            if monomials_commute(left_support, right_support) {
                continue;
            }
            let negated = compose_into(left_support, right_support, product);
            let sign = if negated { -2.0 } else { 2.0 };
            let contribution = left_coeff * right_coeff * sign;
            if let Some(total) = sums.get_mut(product.as_slice()) {
                *total += contribution;
            } else {
                let mut key = spare.pop().unwrap_or_default();
                key.clone_from(product);
                sums.insert(key, contribution);
            }
        }
    }
    sums.values().all(|total| *total == Complex64::ZERO)
}

/// Computes the Majorana expansion of every term, once.
///
/// The terms expand independently, so they do so in parallel, like the sweeps that consume them;
/// `collect` keeps them in term order.
fn expand_all<OpType>(op: &OpType) -> Result<Vec<Expansion>, ClashError>
where
    OpType: MajoranaExpansion + OperatorTrait + Sync,
{
    check_term_count(op.num_terms())?;
    // Checked before anything is allocated, because exceeding the bound is exactly the case in which
    // allocating would abort.
    op.check_expandable()?;
    Ok((0..op.num_terms())
        .into_par_iter()
        .map(|index| expand_term(op, index))
        .collect())
}

/// Checks that every term index, and so every group index, fits the `u32` it is reported as.
///
/// Once this passes, the `as u32` conversions of term and group indices below cannot truncate.
fn check_term_count(num_terms: usize) -> Result<(), ClashError> {
    if num_terms as u64 > MAX_CLASH_TERMS {
        return Err(ClashError::TooManyTerms { num_terms });
    }
    Ok(())
}

/// Computes the Majorana expansion of term `index`, which [`MajoranaExpansion::check_expandable`]
/// must already have accepted.
fn expand_term<OpType>(op: &OpType, index: usize) -> Expansion
where
    OpType: MajoranaExpansion + OperatorTrait,
{
    let mut monomials = Vec::new();
    // The expansions drop each term's coefficient, which is only sound for a *non-zero* scalar. A
    // term with an exactly zero coefficient is the zero operator and commutes with everything, and an
    // empty sum of monomials says precisely that: `terms_commute` then finds no contribution at all,
    // so the term stays a node of the graph without any edges. Near-zero coefficients are
    // deliberately not special-cased, since that would need a tolerance; that is what simplifying
    // the operator beforehand is for.
    if op.coeffs()[index] != Complex64::ZERO {
        op.majorana_monomials_into(index, &mut monomials);
    }
    Expansion::new(monomials)
}

/// Returns the pairs of terms of `op` that do **not** commute.
///
/// A term whose coefficient is exactly zero commutes with everything, so it has no edges. Every other
/// term counts as written, so simplify the operator first to remove near-zero and duplicate terms,
/// which would otherwise inflate the graph.
///
/// Each pair is reported once as `(i, j)` with `i < j`, in ascending lexicographic order. Indices
/// refer to the operator's *current* term order, which operators do not promise to preserve across
/// transformations - build the operator from its sparse arrays, or impose
/// [`canonical_order`](crate::operators::terms::ordering::canonical::canonical_order) first, when the
/// result has to be reproducible.
///
/// # Errors
///
/// Returns [`ClashError::TooManyTerms`] if the operator has more than [`MAX_CLASH_TERMS`] terms, and
/// [`ClashError::DegreeTooLarge`] if a term with a non-zero coefficient exceeds
/// [`MAX_FERMION_TERM_DEGREE`], which only a
/// [`FermionOperator`](crate::operators::fermion_operator::FermionOperator) term can.
pub fn non_commutation_edges<OpType>(op: &OpType) -> Result<Vec<Edge>, ClashError>
where
    OpType: MajoranaExpansion + OperatorTrait + Sync,
{
    let expansions = expand_all(op)?;
    // The sweep is `O(T^2)` and its rows are independent, so they run in parallel on rayon's global
    // pool (sized by `RAYON_NUM_THREADS` when set). Each row `i` yields its own ascending run of
    // `(i, j)` with `i < j`, and `collect` keeps the runs in row order, so concatenating them gives
    // the edges already sorted, exactly as a serial upper-triangular walk would. The expansions are
    // only read; the scratch map is mutated on every pair, so each worker gets its own.
    let rows: Vec<Vec<Edge>> = (0..expansions.len())
        .into_par_iter()
        .map_init(Scratch::default, |scratch, i| {
            let mut row = Vec::new();
            for j in (i + 1)..expansions.len() {
                if !terms_commute(&expansions[i], &expansions[j], scratch) {
                    // `expand_all` bounded the term count, so neither index truncates.
                    row.push([i as u32, j as u32]);
                }
            }
            row
        })
        .collect();
    drop(expansions);

    // The edge list is quadratic in the number of terms, so it is the one allocation that matters.
    // Each row is released as soon as it is copied, rather than all of them once the copy completes
    // (as `concat` would), so the rows and the result are never both held in full. The result is
    // sized exactly, which also lets a caller hand it on as a boxed slice without reallocating.
    let mut edges = Vec::with_capacity(rows.iter().map(Vec::len).sum());
    for row in rows {
        edges.extend_from_slice(&row);
    }
    Ok(edges)
}

/// A set of colors, as a bitset that grows to the largest color inserted.
#[derive(Default)]
struct ColorSet {
    words: Vec<u64>,
}

impl ColorSet {
    fn insert(&mut self, color: usize) {
        let word = color / 64;
        if word >= self.words.len() {
            self.words.resize(word + 1, 0);
        }
        self.words[word] |= 1 << (color % 64);
    }

    /// The smallest color not in the set.
    fn first_free(&self) -> usize {
        self.words
            .iter()
            .position(|&word| word != u64::MAX)
            .map_or(64 * self.words.len(), |index| {
                64 * index + self.words[index].trailing_ones() as usize
            })
    }
}

/// Groups the terms of `op` so that the terms sharing a group index mutually commute.
///
/// Colors the clash graph: a proper coloring puts two non-commuting terms in different groups by
/// construction. The coloring is the greedy *largest-first* heuristic, so the number of groups is an
/// upper bound on the smallest achievable rather than a minimum.
///
/// The graph is never built. Storing it costs one entry per clashing pair, and clashes grow with the
/// square of the number of terms: a molecular Hamiltonian with a couple of hundred thousand terms
/// clashes in billions of pairs, tens of gigabytes before any coloring starts. Instead, the coloring
/// streams over the terms twice, holding one count and one color per term, plus one bit per group
/// for each term of the few hundred it colors at a time:
///
/// 1. Count every term's clashes (its degree in the clash graph).
/// 2. Visit the terms by descending degree, ties in term order, and give each the smallest color that
///    none of the clashing terms visited before it holds.
///
/// That is exactly the largest-first strategy, including its tie-breaking, so the groups are the same
/// as coloring the materialized graph would give. It costs two sweeps over the pairs where building
/// the graph costs one, in exchange for memory linear in the number of terms. Callers who want the
/// graph itself, to color it differently or to draw it, can still build it from
/// [`non_commutation_edges`].
///
/// # Note
///
/// Whether these commuting groups agree with the line flow sets used to compile lattice time
/// evolutions depends on the operator representation, not on the lattice. A flow set forbids two
/// arrows that share a source or a target while permitting a head-to-tail chain. A transfer `T_jk`
/// acts on the odd generator of its source and the even generator of its target, so for the hopping
/// terms of a [`TransferVertexOperator`](crate::operators::transfer_vertex_operator::TransferVertexOperator)
/// the clash relation *is* that rule. Majorana and edge-vertex bilinears instead act on the same
/// generator of a shared mode however they meet, so there a head-to-tail chain does not commute and
/// the two partitions diverge.
///
/// # Errors
///
/// Fails exactly when [`non_commutation_edges`] does, leaving the operator's groups untouched.
pub fn group_terms_by_non_commutation<OpType>(op: &mut OpType) -> Result<(), ClashError>
where
    OpType: MajoranaExpansion + OperatorTrait + Sync,
{
    let expansions = expand_all(op)?;
    let num_terms = expansions.len();

    // Pass 1: every term's degree. Each pair is tested once, from the row of its smaller index, and
    // credited to both of its terms; the atomics let the rows run in parallel exactly as in
    // `non_commutation_edges`. A relaxed ordering suffices, since the counts are only read after the
    // parallel loop has joined. Giving each share of the rows a private count vector instead, summed at
    // the end, avoids the atomic writes, but measured slower on a molecular Hamiltonian, and it
    // multiplies this pass's memory by the number of shares.
    let degrees: Vec<AtomicUsize> = (0..num_terms).map(|_| AtomicUsize::new(0)).collect();
    (0..num_terms)
        .into_par_iter()
        .for_each_init(Scratch::default, |scratch, i| {
            let mut own = 0;
            for j in (i + 1)..num_terms {
                if !terms_commute(&expansions[i], &expansions[j], scratch) {
                    own += 1;
                    degrees[j].fetch_add(1, AtomicOrdering::Relaxed);
                }
            }
            degrees[i].fetch_add(own, AtomicOrdering::Relaxed);
        });

    // Largest first, ties in term order: `sort_by_key` is stable. Together with the first-fit below,
    // that fixes the coloring completely, so the groups depend on the clash graph and the term order
    // alone, never on how the work was split across threads.
    let mut order: Vec<usize> = (0..num_terms).collect();
    order.sort_by_key(|&term| Reverse(degrees[term].load(AtomicOrdering::Relaxed)));

    // Pass 2: first-fit in that order. A term's colored neighbors are exactly the clashing terms
    // visited before it, so it is tested against those alone. The terms must be colored one after
    // another, yet most of that work does not depend on the order: the order is cut into blocks, and
    // every term of a block first scans all *earlier* blocks, which are already colored, in
    // parallel, one long serial scan per term. Only the clashes inside the block are then resolved
    // serially, a triangle of `block_len^2 / 2` pairs per block, so `T * block_len / 2` in all.
    //
    // That serial share must stay small next to the `T^2 / 2` pairs the threads share, so a block
    // holds a small fraction of `T / threads` terms. It must also hold at least a term per thread,
    // or the parallel scan has nothing to spread, and never more than a few hundred, past which the
    // serial triangle stops paying for itself. Splitting finer still, one parallel scan per term,
    // costs more in forking than the scans are worth. (`max` before `min`, not `clamp`, which would
    // panic on a machine with more than 512 threads.)
    //
    // The expansions are re-allocated in visiting order first. Each one owns heap data, and scanning
    // them in degree order while that data sits in term order turns every comparison into scattered
    // memory reads; on a molecular Hamiltonian that made this pass four times slower than pass 1.
    // They are expanded afresh rather than cloned, so that they are never held twice: expanding costs
    // time linear in the number of terms, next to the quadratic sweeps on either side of it. As in
    // `expand_all`, the terms expand in parallel; each worker takes a contiguous run of the order and
    // allocates it in turn, so neighbors in visiting order stay close in memory.
    drop(expansions);
    let sorted: Vec<Expansion> = order
        .par_iter()
        .map(|&term| expand_term(op, term))
        .collect();
    let threads = rayon::current_num_threads();
    let block_len = (num_terms / (4 * threads)).max(threads).min(512);
    let mut scratch = Scratch::default();
    let mut position_color = vec![0usize; num_terms];
    for block_start in (0..num_terms).step_by(block_len) {
        let block_end = num_terms.min(block_start + block_len);
        // Each term of the block records the colors its clashing predecessors hold as a set, one bit
        // per color in use, rather than as a list with one entry per clash: a term can clash with
        // nearly all of its predecessors, but those share only as many colors as there are groups.
        let mut taken: Vec<ColorSet> = (block_start..block_end)
            .into_par_iter()
            .map_init(Scratch::default, |scratch, position| {
                let mut colors = ColorSet::default();
                for other in 0..block_start {
                    if !terms_commute(&sorted[position], &sorted[other], scratch) {
                        colors.insert(position_color[other]);
                    }
                }
                colors
            })
            .collect();
        for position in block_start..block_end {
            let colors = &mut taken[position - block_start];
            for other in block_start..position {
                if !terms_commute(&sorted[position], &sorted[other], &mut scratch) {
                    colors.insert(position_color[other]);
                }
            }
            position_color[position] = colors.first_free();
        }
    }
    // The colors serve as the group indices directly. They are dense already: first-fit gives a term
    // color `c` only when its neighbors colored before it hold every color below `c`, so no color is
    // ever skipped. They are also below the number of terms, which `expand_all` bounded, so the
    // conversion cannot truncate.
    let mut group_indices = vec![0u32; num_terms];
    for (position, &term) in order.iter().enumerate() {
        group_indices[term] = position_color[position] as u32;
    }

    // One group index was set per term, so the length check cannot fire.
    op.set_groups(Some(group_indices))
        .expect("one group index is set per term");
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;

    use crate::operators::edge_vertex_operator::EdgeVertexOperator;
    use crate::operators::fermion_operator::FermionOperator;
    use crate::operators::library::commutators::{anti_commutator, commutator};
    use crate::operators::majorana_operator::MajoranaOperator;
    use crate::operators::transfer_vertex_operator::TransferVertexOperator;

    /// Whether an operator is the zero operator.
    ///
    /// `simplify` alone is not enough: it cancels *like* terms but does not reorder the generators
    /// within a term, so `E_jk V_k` and `V_k E_jk` would survive as two distinct terms even though
    /// they differ only by a sign. Normal-ordering first is what makes cancellation visible, and each
    /// type spells that differently - hence the trait rather than a single generic helper.
    trait ZeroCheck {
        fn is_zero(&self) -> bool;
    }

    /// Normal-orders, then requires every surviving coefficient to vanish.
    macro_rules! impl_zero_check {
        ($type:ty, $normal:expr) => {
            impl ZeroCheck for $type {
                fn is_zero(&self) -> bool {
                    let normal: $type = $normal(self);
                    normal
                        .simplify(1e-12)
                        .coeffs()
                        .iter()
                        .all(|c| c.norm() < 1e-12)
                }
            }
        };
    }

    impl_zero_check!(MajoranaOperator, |op: &MajoranaOperator| op
        .normal_ordered(true, true));
    impl_zero_check!(EdgeVertexOperator, |op: &EdgeVertexOperator| op
        .normal_ordered(true, true));
    impl_zero_check!(TransferVertexOperator, |op: &TransferVertexOperator| op
        .normal_ordered(true));
    impl_zero_check!(FermionOperator, |op: &FermionOperator| op
        .normal_ordered(None));

    fn maj(modes: &[u32]) -> MajoranaOperator {
        MajoranaOperator {
            coeffs: vec![Complex64::ONE],
            modes: modes.to_vec(),
            boundaries: vec![0, modes.len()],
            groups: None,
        }
    }

    fn edge(pairs: &[(u32, u32)]) -> EdgeVertexOperator {
        EdgeVertexOperator {
            coeffs: vec![Complex64::ONE],
            left_indices: pairs.iter().map(|p| p.0).collect(),
            right_indices: pairs.iter().map(|p| p.1).collect(),
            boundaries: vec![0, pairs.len()],
            groups: None,
        }
    }

    fn transfer(pairs: &[(u32, u32)]) -> TransferVertexOperator {
        TransferVertexOperator {
            coeffs: vec![Complex64::ONE],
            left_indices: pairs.iter().map(|p| p.0).collect(),
            right_indices: pairs.iter().map(|p| p.1).collect(),
            boundaries: vec![0, pairs.len()],
            groups: None,
        }
    }

    fn ferm(actions: &[bool], modes: &[u32]) -> FermionOperator {
        FermionOperator {
            coeffs: vec![Complex64::ONE],
            actions: actions.to_vec(),
            modes: modes.to_vec(),
            boundaries: vec![0, modes.len()],
            groups: None,
        }
    }

    /// Whether the predicate says these two single-term operators commute.
    fn predicted<OpType>(a: &OpType, b: &OpType) -> bool
    where
        OpType: MajoranaExpansion + OperatorTrait,
    {
        let (left, right) = expand_pair(a, b);
        terms_commute(&left, &right, &mut Scratch::default())
    }

    /// Expands the single terms of two operators.
    fn expand_pair<OpType>(a: &OpType, b: &OpType) -> (Expansion, Expansion)
    where
        OpType: MajoranaExpansion + OperatorTrait,
    {
        let mut left = Vec::new();
        let mut right = Vec::new();
        a.majorana_monomials_into(0, &mut left);
        b.majorana_monomials_into(0, &mut right);
        (Expansion::new(left), Expansion::new(right))
    }

    /// Ground truth from the library's own algebra: `[A, B] == 0`.
    fn actually_commutes<OpType>(a: &OpType, b: &OpType) -> bool
    where
        OpType: OperatorTrait + crate::operators::OperatorMacro + ZeroCheck,
    {
        commutator(a, b).is_zero()
    }

    /// The predicate must agree with `commutator` on every pair. This is the test that would catch a
    /// wrong Majorana image for any of the four types.
    macro_rules! assert_agrees {
        ($terms:expr, $label:expr) => {{
            let terms = $terms;
            let mut checked = 0;
            for i in 0..terms.len() {
                for j in i..terms.len() {
                    let pred = predicted(&terms[i], &terms[j]);
                    let actual = actually_commutes(&terms[i], &terms[j]);
                    assert_eq!(
                        pred, actual,
                        "{} disagreed for terms {i} and {j}: predicted {pred}, actual {actual}",
                        $label
                    );
                    checked += 1;
                }
            }
            checked
        }};
    }

    #[test]
    fn test_predicate_matches_commutator_majorana() {
        // Every monomial up to degree 3 over 4 modes, i.e. 8 Majorana generators. Enumerated by
        // subset bitmask rather than by hand-rolled combination arithmetic.
        let mut terms = Vec::new();
        for mask in 1u32..(1 << 8) {
            if mask.count_ones() > 3 {
                continue;
            }
            let modes: Vec<u32> = (0..8u32).filter(|g| mask >> g & 1 == 1).collect();
            terms.push(maj(&modes));
        }
        let checked = assert_agrees!(terms, "MajoranaOperator");
        assert!(
            checked > 2000,
            "expected a broad sweep, checked only {checked}"
        );
    }

    #[test]
    fn test_predicate_matches_commutator_edge_vertex() {
        let mut terms = Vec::new();
        for left in 0..4u32 {
            for right in 0..4u32 {
                terms.push(edge(&[(left, right)]));
            }
        }
        // Products of two generators exercise the symmetric-difference support.
        for pairs in [
            vec![(0, 1), (1, 2)],
            vec![(0, 0), (1, 1)],
            vec![(0, 1), (2, 3)],
            vec![(1, 2), (2, 3)],
            vec![(0, 1), (1, 2), (2, 3)],
        ] {
            terms.push(edge(&pairs));
        }
        assert_agrees!(terms, "EdgeVertexOperator");
    }

    #[test]
    fn test_predicate_matches_commutator_transfer_vertex() {
        let mut terms = Vec::new();
        for left in 0..4u32 {
            for right in 0..4u32 {
                terms.push(transfer(&[(left, right)]));
            }
        }
        for pairs in [
            vec![(0, 1), (1, 2)],
            vec![(0, 0), (1, 1)],
            vec![(0, 1), (2, 3)],
            vec![(1, 2), (2, 3)],
            vec![(0, 1), (1, 2), (2, 3)],
        ] {
            terms.push(transfer(&pairs));
        }
        assert_agrees!(terms, "TransferVertexOperator");
    }

    #[test]
    fn test_predicate_matches_commutator_fermion() {
        // A fermionic term is a *sum* of Majorana monomials, so this exercises the
        // cancellation-aware branch rather than the parity fast path.
        let mut terms = Vec::new();
        for p in 0..4u32 {
            for q in 0..4u32 {
                terms.push(ferm(&[true, false], &[p, q]));
            }
        }
        terms.push(ferm(&[true, true, false, false], &[0, 1, 1, 0]));
        terms.push(ferm(&[true, true, false, false], &[0, 2, 2, 0]));
        terms.push(ferm(&[true, false, true, false], &[0, 1, 2, 3]));
        // Odd terms, which the disjoint-support screen must *not* wave through: two of them on
        // disjoint modes anticommute.
        for p in 0..4u32 {
            terms.push(ferm(&[true], &[p]));
            terms.push(ferm(&[false], &[p]));
        }
        terms.push(ferm(&[true, true, false], &[0, 1, 1]));
        terms.push(ferm(&[true, false, false], &[2, 2, 3]));
        assert_agrees!(terms, "FermionOperator");
    }

    #[test]
    fn test_expansion_merges_repeated_supports() {
        // `a^dagger_0 a_0` expands into four monomials, two of which collapse onto the identity and
        // the other two onto `gamma_0 gamma_1`, with equal amplitudes. Merging leaves only the latter.
        let number = ferm(&[true, false], &[0, 0]);
        let mut monomials = Vec::new();
        number.majorana_monomials_into(0, &mut monomials);
        assert_eq!(monomials.len(), 4);
        let expansion = Expansion::new(monomials);
        assert_eq!(expansion.monomials.len(), 1);
        assert_eq!(expansion.monomials[0].0.len(), 2);
        assert!(!expansion.odd);
        assert_eq!(expansion.support, expansion.monomials[0].0);
    }

    #[test]
    fn test_support_screen_respects_parity() {
        // Even terms on disjoint modes commute and are screened out ...
        assert!(predicted(
            &ferm(&[true, false], &[0, 1]),
            &ferm(&[true, false], &[2, 3])
        ));
        // ... but odd ones on disjoint modes anticommute, so the screen must not apply to them.
        assert!(!predicted(&ferm(&[true], &[0]), &ferm(&[false], &[1])));
        // An odd term against an even one on disjoint modes commutes again.
        assert!(predicted(
            &ferm(&[true], &[0]),
            &ferm(&[true, false], &[1, 2])
        ));
    }

    #[test]
    fn test_fermion_needs_cancellation_not_just_pairwise() {
        // The cheaper "every monomial pair commutes" rule is sound but too strict. Find a fermionic
        // pair that genuinely commutes *through cancellation*, and assert our predicate says so while
        // the pairwise rule would not.
        let mut found = false;
        for p in 0..4u32 {
            for q in 0..4u32 {
                for r in 0..4u32 {
                    for s in 0..4u32 {
                        let a = ferm(&[true, false], &[p, q]);
                        let b = ferm(&[true, false], &[r, s]);
                        if !actually_commutes(&a, &b) {
                            continue;
                        }
                        // The *merged* monomials, so that what is found here is cancellation
                        // between contributions, not merely monomials merged away beforehand.
                        let (left, right) = expand_pair(&a, &b);
                        let all_pairs_commute = left.monomials.iter().all(|(ls, _)| {
                            right
                                .monomials
                                .iter()
                                .all(|(rs, _)| monomials_commute(ls, rs))
                        });
                        if !all_pairs_commute {
                            // Commutes overall, yet some monomial pair anticommutes: cancellation.
                            assert!(
                                terms_commute(&left, &right, &mut Scratch::default()),
                                "cancellation-aware predicate must accept a+_{p}a_{q} / a+_{r}a_{s}"
                            );
                            found = true;
                        }
                    }
                }
            }
        }
        assert!(
            found,
            "expected at least one fermionic pair commuting only through cancellation"
        );
    }

    /// The (anti)commutation relations stated in `EdgeVertexOperator`'s own docstring.
    #[test]
    fn test_edge_vertex_documented_relations() {
        // Indices chosen distinct: j=0, k=1, l=2, m=3.
        let (j, k, l, m) = (0u32, 1u32, 2u32, 3u32);
        // {E_jk, V_k} = 0
        assert!(!predicted(&edge(&[(j, k)]), &edge(&[(k, k)])));
        assert!(anti_commutator(&edge(&[(j, k)]), &edge(&[(k, k)])).is_zero());
        // {E_jk, E_kl} = 0
        assert!(!predicted(&edge(&[(j, k)]), &edge(&[(k, l)])));
        // [V_k, V_l] = 0
        assert!(predicted(&edge(&[(k, k)]), &edge(&[(l, l)])));
        // [E_jk, V_l] = 0
        assert!(predicted(&edge(&[(j, k)]), &edge(&[(l, l)])));
        // [E_jk, E_lm] = 0
        assert!(predicted(&edge(&[(j, k)]), &edge(&[(l, m)])));
        // [E_jk, E_kj] = 0 -- the two orientations differ only by a sign, so they share a support.
        assert!(predicted(&edge(&[(j, k)]), &edge(&[(k, j)])));
    }

    /// The relations stated in `TransferVertexOperator`'s own docstring.
    #[test]
    fn test_transfer_vertex_documented_relations() {
        let (j, k, l, m) = (0u32, 1u32, 2u32, 3u32);
        // {T_jk, V_k} = 0
        assert!(!predicted(&transfer(&[(j, k)]), &transfer(&[(k, k)])));
        // {T_jk, T_lk} = 0
        assert!(!predicted(&transfer(&[(j, k)]), &transfer(&[(l, k)])));
        // [V_k, V_l] = 0
        assert!(predicted(&transfer(&[(k, k)]), &transfer(&[(l, l)])));
        // [T_jk, V_l] = 0
        assert!(predicted(&transfer(&[(j, k)]), &transfer(&[(l, l)])));
        // [T_jk, T_lm] = 0
        assert!(predicted(&transfer(&[(j, k)]), &transfer(&[(l, m)])));
        // [T_jk, T_kj] = 0
        assert!(predicted(&transfer(&[(j, k)]), &transfer(&[(k, j)])));
        // [T_jk, T_km] = 0 -- a head-to-tail chain, which *commutes*: `T_jk` acts on the even
        // generator of `k` and `T_km` on its odd one, so they share no generator. This is exactly
        // the configuration a flow set permits, and what a Majorana bilinear chain does not do.
        assert!(predicted(&transfer(&[(j, k)]), &transfer(&[(k, m)])));
    }

    /// Builds a `MajoranaOperator` whose terms are the given mode pairs, in the given order.
    fn maj_pairs(pairs: &[(u32, u32)]) -> MajoranaOperator {
        let mut modes = Vec::new();
        let mut boundaries = vec![0];
        for &(a, b) in pairs {
            modes.push(a);
            modes.push(b);
            boundaries.push(modes.len());
        }
        MajoranaOperator {
            coeffs: vec![Complex64::ONE; pairs.len()],
            modes,
            boundaries,
            groups: None,
        }
    }

    /// Asserts that no clash edge lies inside a group. This is the definitional property of the
    /// grouping, and unlike a group *count* it cannot drift when the coloring heuristic changes.
    fn assert_proper_coloring(edges: &[Edge], groups: &[u32]) {
        for &[left, right] in edges {
            assert_ne!(
                groups[left as usize], groups[right as usize],
                "terms {left} and {right} do not commute yet share a group"
            );
        }
    }

    #[test]
    fn test_edges_are_sorted_and_unique() {
        let mut op = maj_pairs(&[(0, 1), (1, 2), (3, 0), (1, 4), (5, 2), (5, 4), (4, 3)]);
        let edges = non_commutation_edges(&op).unwrap();
        assert!(!edges.is_empty(), "this graph must have clashes");
        for &[left, right] in &edges {
            assert!(left < right, "edges must be reported with i < j");
        }
        let mut sorted = edges.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(edges, sorted, "edges must be sorted and free of duplicates");

        // The grouping must be a proper coloring of exactly those edges.
        group_terms_by_non_commutation(&mut op).unwrap();
        let groups = op.groups().expect("groups were just assigned").to_vec();
        assert_proper_coloring(&edges, &groups);
    }

    #[test]
    fn test_groups_are_dense() {
        let mut op = maj_pairs(&[(0, 1), (1, 2), (2, 3), (3, 0)]);
        group_terms_by_non_commutation(&mut op).unwrap();
        let groups = op.groups().expect("groups were just assigned").to_vec();

        let mut seen: Vec<u32> = groups.clone();
        seen.sort_unstable();
        seen.dedup();
        let expected: Vec<u32> = (0..seen.len() as u32).collect();
        assert_eq!(seen, expected, "group indices must be dense and start at 0");
    }

    /// A commuting operator needs a single group; a fully clashing one needs a group per term.
    #[test]
    fn test_extremes() {
        // Disjoint supports: everything commutes.
        let mut commuting = maj_pairs(&[(0, 1), (2, 3), (4, 5)]);
        assert!(non_commutation_edges(&commuting).unwrap().is_empty());
        group_terms_by_non_commutation(&mut commuting).unwrap();
        assert_eq!(commuting.num_groups(), Some(1));

        // Single generators sharing nothing pairwise still anticommute (odd times odd, no overlap).
        let mut clashing = MajoranaOperator {
            coeffs: vec![Complex64::ONE; 3],
            modes: vec![0, 1, 2],
            boundaries: vec![0, 1, 2, 3],
            groups: None,
        };
        let edges = non_commutation_edges(&clashing).unwrap();
        assert_eq!(
            edges.len(),
            3,
            "three distinct single generators pairwise anticommute"
        );
        group_terms_by_non_commutation(&mut clashing).unwrap();
        assert_eq!(
            clashing.num_groups(),
            Some(3),
            "a triangle needs three colors"
        );
    }

    #[test]
    fn test_term_count_bound() {
        // Building an operator this large is out of the question, so the check itself is tested.
        let max = MAX_CLASH_TERMS as usize;
        assert!(check_term_count(max).is_ok());
        match check_term_count(max + 1) {
            Err(ClashError::TooManyTerms { num_terms }) => {
                assert_eq!(num_terms as u64, MAX_CLASH_TERMS + 1);
            }
            other => panic!("expected the term bound to be enforced, got {other:?}"),
        }
        // The largest index the bound admits is exactly the largest `u32`.
        assert_eq!(MAX_CLASH_TERMS - 1, u64::from(u32::MAX));
    }

    #[test]
    fn test_fermion_degree_bound() {
        /// A creation-only term on `degree` distinct modes, followed by a degree-1 term.
        fn with_degree(degree: usize) -> FermionOperator {
            FermionOperator {
                coeffs: vec![Complex64::ONE; 2],
                actions: vec![true; degree + 1],
                modes: (0..degree as u32).chain([0]).collect(),
                boundaries: vec![0, degree, degree + 1],
                groups: None,
            }
        }

        // The bound itself is accepted. Only the check runs here, since expanding a term at the
        // bound is deliberately expensive.
        assert!(
            with_degree(MAX_FERMION_TERM_DEGREE)
                .check_expandable()
                .is_ok()
        );

        let mut op = with_degree(MAX_FERMION_TERM_DEGREE + 1);
        match non_commutation_edges(&op) {
            Err(ClashError::DegreeTooLarge { term, degree }) => {
                assert_eq!(term, 0);
                assert_eq!(degree, MAX_FERMION_TERM_DEGREE + 1);
            }
            other => panic!("expected the degree bound to be enforced, got {other:?}"),
        }
        assert!(group_terms_by_non_commutation(&mut op).is_err());
        assert!(
            op.groups().is_none(),
            "a failed grouping must not assign groups"
        );

        // A zero coefficient is never expanded, so its degree is no obstacle: the term is the zero
        // operator, which commutes with everything.
        op.coeffs[0] = Complex64::ZERO;
        assert_eq!(non_commutation_edges(&op).unwrap(), Vec::<Edge>::new());
        group_terms_by_non_commutation(&mut op).unwrap();
        assert_eq!(op.groups().map(|groups| groups.to_vec()), Some(vec![0, 0]));
    }

    #[test]
    fn test_zero_coefficient_terms_are_isolated() {
        // Three distinct single generators pairwise anticommute, but a zero multiple of one of them
        // is the zero operator, which commutes with everything.
        let mut op = MajoranaOperator {
            coeffs: vec![Complex64::ZERO, Complex64::ONE, Complex64::ONE],
            modes: vec![0, 1, 2],
            boundaries: vec![0, 1, 2, 3],
            groups: None,
        };
        assert_eq!(non_commutation_edges(&op).unwrap(), vec![[1, 2]]);
        group_terms_by_non_commutation(&mut op).unwrap();
        assert_eq!(op.num_groups(), Some(2));

        // Negative zero is zero, too, and the fermionic path takes the same route.
        let fermion = FermionOperator {
            coeffs: vec![Complex64::new(-0.0, 0.0), Complex64::ONE],
            actions: vec![true, true],
            modes: vec![0, 1],
            boundaries: vec![0, 1, 2],
            groups: None,
        };
        assert!(non_commutation_edges(&fermion).unwrap().is_empty());
    }

    #[test]
    fn test_empty_operator() {
        let mut op = MajoranaOperator {
            coeffs: vec![],
            modes: vec![],
            boundaries: vec![0],
            groups: None,
        };
        assert!(non_commutation_edges(&op).unwrap().is_empty());
        group_terms_by_non_commutation(&mut op).unwrap();
        assert_eq!(op.groups(), Some(&[][..]));
    }

    /// For Majorana bilinears the grouping produced here is **not** the flow-set partition, and this
    /// pins the difference.
    ///
    /// The grouping guide hand-assigns groups that resemble line flow sets to a small lattice of
    /// Majorana bilinears. A flow set forbids two arrows sharing a source or a target, but permits a
    /// head-to-tail chain, and two bilinears in such a chain share a generator, so they do not
    /// commute. The guide's own grouping therefore contains non-commuting pairs, while this routine's
    /// grouping never does. Both are legitimate; they answer different questions. Contrast
    /// [`test_transfer_clashes_are_flow_set_conflicts`], where the two relations agree.
    #[test]
    fn test_flow_sets_are_not_commuting_groups() {
        let edges_of_graph = [(0, 1), (1, 2), (3, 0), (1, 4), (5, 2), (5, 4), (4, 3)];
        let mut op = maj_pairs(&edges_of_graph);
        let clashes = non_commutation_edges(&op).unwrap();

        // The guide's hand-written flow sets, verbatim from `docs/guides/grouping.rst`.
        let flow_sets = [0u32, 0, 1, 2, 1, 3, 3];
        let intra_group_clashes = clashes
            .iter()
            .filter(|[l, r]| flow_sets[*l as usize] == flow_sets[*r as usize])
            .count();
        assert!(
            intra_group_clashes > 0,
            "the guide's flow sets are expected to contain non-commuting pairs; if this ever \
             becomes 0, the two relations have stopped diverging and the docs must be revisited"
        );
        // Specifically, the head-to-tail chain (0,1) -> (1,2), which are terms 0 and 1.
        assert!(
            clashes.contains(&[0, 1]),
            "terms (0,1) and (1,2) share mode 1 and must clash"
        );
        assert_eq!(
            flow_sets[0], flow_sets[1],
            "yet the guide places them in the same flow set"
        );

        // Our own grouping, by contrast, is a proper coloring by construction.
        group_terms_by_non_commutation(&mut op).unwrap();
        let groups = op.groups().expect("groups were just assigned").to_vec();
        assert_proper_coloring(&clashes, &groups);
    }

    /// For transfer hopping terms the clash relation *is* the flow-set rule, on any directed graph.
    ///
    /// A transfer acts on the odd generator of its source and the even generator of its target, so
    /// two transfers share a generator exactly when they share a source or a target, and a
    /// head-to-tail chain shares none. Checked on every simple directed graph over four modes.
    #[test]
    fn test_transfer_clashes_are_flow_set_conflicts() {
        let arrows: Vec<(u32, u32)> = (0..4u32)
            .flat_map(|j| (0..4u32).filter(move |&k| k != j).map(move |k| (j, k)))
            .collect();
        for mask in 1u32..(1 << arrows.len()) {
            let pairs: Vec<(u32, u32)> = (0..arrows.len())
                .filter(|bit| mask >> bit & 1 == 1)
                .map(|bit| arrows[bit])
                .collect();
            let op = TransferVertexOperator {
                coeffs: vec![Complex64::ONE; pairs.len()],
                left_indices: pairs.iter().map(|p| p.0).collect(),
                right_indices: pairs.iter().map(|p| p.1).collect(),
                boundaries: (0..=pairs.len()).collect(),
                groups: None,
            };
            let mut conflicts = Vec::new();
            for i in 0..pairs.len() {
                for j in (i + 1)..pairs.len() {
                    if pairs[i].0 == pairs[j].0 || pairs[i].1 == pairs[j].1 {
                        conflicts.push([i as u32, j as u32]);
                    }
                }
            }
            assert_eq!(
                non_commutation_edges(&op).unwrap(),
                conflicts,
                "transfer clashes and flow-set conflicts disagree on {pairs:?}"
            );
        }
    }

    /// Largest-first coloring of an explicit graph, written out plainly as the reference the streaming
    /// implementation must reproduce: sort by descending degree with a stable sort, then give each
    /// term the smallest color its already-colored neighbors leave free.
    fn reference_largest_first(num_terms: usize, edges: &[Edge]) -> Vec<u32> {
        let mut neighbors = vec![Vec::new(); num_terms];
        for &[left, right] in edges {
            neighbors[left as usize].push(right as usize);
            neighbors[right as usize].push(left as usize);
        }
        let mut order: Vec<usize> = (0..num_terms).collect();
        order.sort_by_key(|&term| std::cmp::Reverse(neighbors[term].len()));
        let mut coloring: Vec<Option<usize>> = vec![None; num_terms];
        for term in order {
            let taken: std::collections::HashSet<usize> = neighbors[term]
                .iter()
                .filter_map(|&n| coloring[n])
                .collect();
            coloring[term] = Some((0..).find(|color| !taken.contains(color)).unwrap());
        }
        coloring
            .into_iter()
            .map(|color| color.unwrap() as u32)
            .collect()
    }

    /// The streaming coloring never builds the graph, yet must give exactly the groups that
    /// largest-first coloring of the materialized graph gives, ties included.
    #[test]
    fn test_color_set_finds_the_smallest_free_color() {
        let mut colors = ColorSet::default();
        assert_eq!(colors.first_free(), 0);
        colors.insert(1);
        assert_eq!(colors.first_free(), 0);
        // A full word sends the search on to the next one, which need not exist yet.
        (0..64).for_each(|color| colors.insert(color));
        assert_eq!(colors.first_free(), 64);
        colors.insert(65);
        assert_eq!(colors.first_free(), 64);
        colors.insert(64);
        assert_eq!(colors.first_free(), 66);
    }

    #[test]
    fn test_streaming_coloring_matches_largest_first() {
        // Every Majorana monomial up to degree 3 over 8 generators: dense, with many degree ties.
        let mut pairs_modes = Vec::new();
        let mut boundaries = vec![0];
        for mask in 1u32..(1 << 8) {
            if mask.count_ones() <= 3 {
                pairs_modes.extend((0..8u32).filter(|g| mask >> g & 1 == 1));
                boundaries.push(pairs_modes.len());
            }
        }
        let mut majorana = MajoranaOperator {
            coeffs: vec![Complex64::ONE; boundaries.len() - 1],
            modes: pairs_modes,
            boundaries,
            groups: None,
        };
        let edges = non_commutation_edges(&majorana).unwrap();
        let expected = reference_largest_first(majorana.num_terms(), &edges);
        group_terms_by_non_commutation(&mut majorana).unwrap();
        assert_eq!(majorana.groups().unwrap(), &expected[..]);

        // A fermionic operator, through the cancellation-aware path.
        let mut actions = Vec::new();
        let mut modes = Vec::new();
        let mut boundaries = vec![0];
        for p in 0..4u32 {
            for q in 0..4u32 {
                actions.extend([true, false]);
                modes.extend([p, q]);
                boundaries.push(modes.len());
            }
        }
        let mut fermion = FermionOperator {
            coeffs: vec![Complex64::ONE; boundaries.len() - 1],
            actions,
            modes,
            boundaries,
            groups: None,
        };
        let edges = non_commutation_edges(&fermion).unwrap();
        let expected = reference_largest_first(fermion.num_terms(), &edges);
        group_terms_by_non_commutation(&mut fermion).unwrap();
        assert_eq!(fermion.groups().unwrap(), &expected[..]);
    }

    /// Lattice partitions, asserted as upper bounds so a change of coloring heuristic cannot break
    /// them spuriously. The exact 3x3 value is pinned separately below.
    #[test]
    fn test_lattice_partitions() {
        /// Both orientations of every bond of a `rows x cols` open square lattice, as transfer terms.
        fn square_transfers(rows: u32, cols: u32) -> Vec<(u32, u32)> {
            let index = |r: u32, c: u32| r * cols + c;
            let mut bonds = Vec::new();
            for r in 0..rows {
                for c in 0..cols {
                    if c + 1 < cols {
                        bonds.push((index(r, c), index(r, c + 1)));
                    }
                    if r + 1 < rows {
                        bonds.push((index(r, c), index(r + 1, c)));
                    }
                }
            }
            // Flow sets live on directed arrows, so carry both orientations of each bond.
            bonds
                .into_iter()
                .flat_map(|(j, k)| [(j, k), (k, j)])
                .collect()
        }

        fn transfer_op(pairs: &[(u32, u32)]) -> TransferVertexOperator {
            let mut left = Vec::new();
            let mut right = Vec::new();
            let mut boundaries = vec![0];
            for &(l, r) in pairs {
                left.push(l);
                right.push(r);
                boundaries.push(left.len());
            }
            TransferVertexOperator {
                coeffs: vec![Complex64::ONE; pairs.len()],
                left_indices: left,
                right_indices: right,
                boundaries,
                groups: None,
            }
        }

        for (rows, cols, bound) in [
            (1u32, 4u32, 2u32),
            (1, 6, 2),
            (2, 3, 3),
            (3, 3, 4),
            (4, 4, 4),
        ] {
            let pairs = square_transfers(rows, cols);
            let mut op = transfer_op(&pairs);
            let clashes = non_commutation_edges(&op).unwrap();
            group_terms_by_non_commutation(&mut op).unwrap();
            let groups = op.groups().expect("groups were just assigned").to_vec();
            assert_proper_coloring(&clashes, &groups);
            let num_groups = op.num_groups().expect("grouped");
            assert!(
                num_groups <= bound,
                "{rows}x{cols}: expected at most {bound} groups, got {num_groups}"
            );
        }
    }

    /// Characterizes the *current* greedy heuristic on a 3x3 lattice.
    ///
    /// Unlike the bounds above this is an exact value, so it may legitimately need updating if the
    /// coloring heuristic ever changes. The edge count, by contrast, is a property of the commutation
    /// predicate alone and must not change.
    #[test]
    fn test_three_by_three_characterization() {
        let index = |r: u32, c: u32| r * 3 + c;
        let mut bonds = Vec::new();
        for r in 0..3 {
            for c in 0..3 {
                if c + 1 < 3 {
                    bonds.push((index(r, c), index(r, c + 1)));
                }
                if r + 1 < 3 {
                    bonds.push((index(r, c), index(r + 1, c)));
                }
            }
        }
        let pairs: Vec<(u32, u32)> = bonds
            .into_iter()
            .flat_map(|(j, k)| [(j, k), (k, j)])
            .collect();
        assert_eq!(pairs.len(), 24, "3x3 has 12 bonds, 24 directed arrows");

        let mut left = Vec::new();
        let mut right = Vec::new();
        let mut boundaries = vec![0];
        for &(l, r) in &pairs {
            left.push(l);
            right.push(r);
            boundaries.push(left.len());
        }
        let mut op = TransferVertexOperator {
            coeffs: vec![Complex64::ONE; pairs.len()],
            left_indices: left,
            right_indices: right,
            boundaries,
            groups: None,
        };

        let clashes = non_commutation_edges(&op).unwrap();
        assert_eq!(
            clashes.len(),
            44,
            "the clash count is fixed by the predicate"
        );
        group_terms_by_non_commutation(&mut op).unwrap();
        assert_eq!(
            op.num_groups(),
            Some(4),
            "greedy coloring of the 3x3 clash graph currently yields 4 groups"
        );
    }
}
