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

//! Algorithms that *choose* a ternary tree, rather than merely represent one.
//!
//! [`TernaryTree`](crate::mappers::ternary_tree::TernaryTree) says what an encoding is; the
//! algorithms here decide which encoding to use. [`hatt_tree`] grows a tree adapted to a
//! Hamiltonian's term structure.
//!
//! Neither the balanced tree nor Jordan-Wigner uses any such information: the balanced tree minimizes
//! the Pauli weight of every single operator and knows nothing of the Hamiltonian, while
//! Jordan-Wigner is the maximally unbalanced tree.
//!
//! # References
//!
//! 1. Y. Liu, K. Yao, J. Hong, J. Froustey, E. Rrapaj, C. Iancu, G. Li and Y. Shi, *HATT: Hamiltonian
//!    Adaptive Ternary Tree for Optimizing Fermion-to-Qubit Mapping*,
//!    [arXiv:2409.02010](https://arxiv.org/abs/2409.02010).

use thiserror::Error;

pub mod hatt;

pub use hatt::hatt_tree;

/// Errors from the tree-search algorithms.
///
/// Kept separate from [`TernaryTreeError`](crate::mappers::ternary_tree::TernaryTreeError) for the
/// same reason that one is separate from `CoherenceError`: these describe a bad *input to a search*
/// (such as a Hamiltonian whose mode count disagrees), which no tree construction can produce.
#[derive(Error, Debug, PartialEq, Eq)]
pub enum SearchError {
    /// The requested encoding covers no modes.
    #[error("an encoding must cover at least one mode")]
    NoModes,
    /// The requested tree has more nodes than a ternary tree can hold.
    #[error("a ternary tree holds at most {max} nodes, but {num_nodes} were requested")]
    TooLarge {
        /// The requested number of nodes.
        num_nodes: u32,
        /// The largest number of nodes a tree can hold.
        max: u32,
    },
    /// The Hamiltonian acts on a mode the requested encoding does not cover.
    #[error(
        "the Hamiltonian acts on mode {max_mode}, which an encoding of {num_modes} modes does not cover"
    )]
    ModeCountMismatch {
        /// The requested mode count.
        num_modes: u32,
        /// The largest mode the Hamiltonian acts on.
        max_mode: u32,
    },
    /// A term of the Hamiltonian has a coefficient that is infinite or NaN.
    #[error("the Hamiltonian term on Majorana operators {modes:?} has a non-finite coefficient")]
    NonFiniteCoefficient {
        /// The Majorana operators of the offending term.
        modes: Vec<u32>,
    },
}
