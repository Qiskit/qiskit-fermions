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

use crate::pointers::const_ptr_as_ref;

use qiskit_fermions_core::operators::edge_vertex_operator::EdgeVertexOperator;
use qiskit_fermions_core::operators::fermion_operator::FermionOperator;
use qiskit_fermions_core::operators::majorana_operator::MajoranaOperator;
use qiskit_fermions_core::operators::terms::grouping::analysis;
use qiskit_fermions_core::operators::transfer_vertex_operator::TransferVertexOperator;

/// Generates the group-analysis functions for one operator representation.
///
/// The four routines are generic over the operator type in `core`, but C has no such notion, so each
/// one needs a concrete function per representation. Their bodies differ only in the operator type
/// and the `qf_<prefix>_` name, which is what this macro fills in.
///
/// The prose documentation lives in `docs/cdoc/qf-operators-terms-grouping.rst` rather than in
/// doc comments here, because cbindgen cannot expand procedural macros without the nightly
/// toolchain: the generated signatures are declared by hand in `cbindgen.toml` and so carry no
/// doc comments into the header. The same trade-off is made for the commutator functions.
macro_rules! impl_group_analysis {
    ($name:ty, $prefix:ident) => {
        paste::item! {
            /// The panic message shared by the four functions below.
            ///
            /// Built per operator representation so that it names the concrete guard function the
            /// caller was supposed to call, rather than a generic placeholder.
            const [<$prefix:upper _GROUPS_EXPECTED_MSG>]: &str = concat!(
                "Expected groups to be present. It is the user's responsibility to check this via ",
                stringify!([<qf_ $prefix _has_groups>]),
                " before calling this function.",
            );

            /// @ingroup qf_operator_terms_grouping
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qf_ $prefix _group_coeff_means>](
                op: *const $name,
                means_out: *mut f64,
            ) {
                // SAFETY: Per documentation, the pointers are non-null and aligned.
                let op = unsafe { const_ptr_as_ref(op) };

                let means = analysis::group_coeff_means(op).expect(
                    [<$prefix:upper _GROUPS_EXPECTED_MSG>],
                );
                for (i, mean) in means.iter().enumerate() {
                    // SAFETY: Per documentation, `means_out` is sized to the number of groups.
                    unsafe { means_out.add(i).write(*mean) };
                }
            }

            /// @ingroup qf_operator_terms_grouping
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qf_ $prefix _groups_are_hermitian>](
                op: *const $name,
                atol: f64,
                hermitian_out: *mut bool,
            ) {
                // SAFETY: Per documentation, the pointers are non-null and aligned.
                let op = unsafe { const_ptr_as_ref(op) };

                let hermitian = analysis::groups_are_hermitian(op, atol).expect(
                    [<$prefix:upper _GROUPS_EXPECTED_MSG>],
                );
                for (i, flag) in hermitian.iter().enumerate() {
                    // SAFETY: Per documentation, `hermitian_out` is sized to the number of groups.
                    unsafe { hermitian_out.add(i).write(*flag) };
                }
            }

            /// @ingroup qf_operator_terms_grouping
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qf_ $prefix _groups_have_uniform_coeffs>](
                op: *const $name,
                atol: f64,
                abs: bool,
                uniform_out: *mut bool,
            ) {
                // SAFETY: Per documentation, the pointers are non-null and aligned.
                let op = unsafe { const_ptr_as_ref(op) };

                let uniform = analysis::groups_have_uniform_coeffs(op, atol, abs).expect(
                    [<$prefix:upper _GROUPS_EXPECTED_MSG>],
                );
                for (i, flag) in uniform.iter().enumerate() {
                    // SAFETY: Per documentation, `uniform_out` is sized to the number of groups.
                    unsafe { uniform_out.add(i).write(*flag) };
                }
            }
        }
    };
}

// NOTE: cbindgen cannot expand procedural macros without the nightly rust toolchain. Remember to
// declare the generated C function signatures in cbindgen.toml!
impl_group_analysis!(FermionOperator, ferm_op);
impl_group_analysis!(MajoranaOperator, maj_op);
impl_group_analysis!(EdgeVertexOperator, edge_op);
impl_group_analysis!(TransferVertexOperator, transfer_op);

#[cfg(test)]
mod tests {
    use super::{FERM_OP_GROUPS_EXPECTED_MSG, TRANSFER_OP_GROUPS_EXPECTED_MSG};

    /// The guard-function name in the panic message must match the operator representation.
    #[test]
    fn test_groups_expected_message_names_its_guard() {
        assert_eq!(
            FERM_OP_GROUPS_EXPECTED_MSG,
            "Expected groups to be present. It is the user's responsibility to check this via \
            qf_ferm_op_has_groups before calling this function."
        );
        assert!(TRANSFER_OP_GROUPS_EXPECTED_MSG.contains("qf_transfer_op_has_groups"));
    }
}
