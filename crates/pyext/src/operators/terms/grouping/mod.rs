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

use pyo3::prelude::*;

/// Applies a grouping routine to whichever concrete operator type `op` holds.
///
/// The routines in `core` are generic over the operator type, but Python hands us an untyped object,
/// so each wrapper resolves it here. `cast` is a borrowing type check: on a miss it returns a
/// borrowed error without allocating a `PyErr`, and on a hit it hands back a `&Bound` we borrow
/// from, so no clone of the input is needed.
///
/// The borrow mode comes first: `ref` for a routine that only reads the operator, `mut` for one that
/// writes group indices back into it.
macro_rules! dispatch {
    ($mode:ident, $op:expr, $name:literal, |$inner:ident| $call:expr) => {{
        if let Ok(bound) = $op.cast::<$crate::operators::fermion_operator::PyFermionOperator>() {
            let $inner = dispatch!(@borrow $mode, bound);
            return Ok($call);
        }
        if let Ok(bound) = $op.cast::<$crate::operators::majorana_operator::PyMajoranaOperator>() {
            let $inner = dispatch!(@borrow $mode, bound);
            return Ok($call);
        }
        if let Ok(bound) =
            $op.cast::<$crate::operators::edge_vertex_operator::PyEdgeVertexOperator>()
        {
            let $inner = dispatch!(@borrow $mode, bound);
            return Ok($call);
        }
        if let Ok(bound) =
            $op.cast::<$crate::operators::transfer_vertex_operator::PyTransferVertexOperator>()
        {
            let $inner = dispatch!(@borrow $mode, bound);
            return Ok($call);
        }
        Err(::pyo3::exceptions::PyTypeError::new_err(concat!(
            $name,
            " expects a fermionic, Majorana, edge-vertex or transfer-vertex operator"
        )))
    }};
    (@borrow ref, $bound:ident) => {
        &$bound.borrow().inner
    };
    (@borrow mut, $bound:ident) => {
        &mut $bound.borrow_mut().inner
    };
}

pub mod analysis;
pub mod clash;
pub mod electronic_structure;

#[pymodule]
pub mod grouping {
    #[pymodule_export]
    use super::analysis::analysis;
    #[pymodule_export]
    use super::clash::clash;
    #[pymodule_export]
    use super::electronic_structure::electronic_structure;
}
