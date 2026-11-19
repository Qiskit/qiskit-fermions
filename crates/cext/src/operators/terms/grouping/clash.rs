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

use paste::paste;

use crate::exit_codes::ExitCode;
use crate::pointers::mut_ptr_as_ref;

use qiskit_fermions_core::operators::edge_vertex_operator::EdgeVertexOperator;
use qiskit_fermions_core::operators::fermion_operator::FermionOperator;
use qiskit_fermions_core::operators::majorana_operator::MajoranaOperator;
use qiskit_fermions_core::operators::terms::grouping::clash::group_terms_by_non_commutation;
use qiskit_fermions_core::operators::transfer_vertex_operator::TransferVertexOperator;

/// Generates the commuting-grouping entry point for one operator representation.
///
/// Unlike the per-type `qf_<prefix>_non_commutation_edges` accessors, which live beside their
/// operators because each is documented as part of its operator's API, this grouping is documented
/// once generically and so is macro-generated here, alongside the other grouping strategies.
///
/// The prose documentation lives in `docs/cdoc/qf-operators-terms-grouping.rst` rather than in doc
/// comments here, for the reason given at `impl_group_analysis`: the generated signatures are
/// declared by hand in `cbindgen.toml` and so carry no doc comments into the header.
macro_rules! impl_clash_grouping {
    ($type:ty, $prefix:ident) => {
        paste! {
            /// @ingroup qf_operator_terms_grouping
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qf_ $prefix _group_terms_by_non_commutation>](
                op: *mut $type,
            ) -> ExitCode {
                // SAFETY: Per documentation, the pointer is non-null and aligned.
                let op = unsafe { mut_ptr_as_ref(op) };

                match group_terms_by_non_commutation(op) {
                    Ok(()) => ExitCode::Success,
                    Err(_) => ExitCode::ValueError,
                }
            }
        }
    };
}

// NOTE: cbindgen cannot expand procedural macros without the nightly rust toolchain. Remember to
// declare the generated C function signatures in cbindgen.toml!
impl_clash_grouping!(FermionOperator, ferm_op);
impl_clash_grouping!(MajoranaOperator, maj_op);
impl_clash_grouping!(EdgeVertexOperator, edge_op);
impl_clash_grouping!(TransferVertexOperator, transfer_op);
