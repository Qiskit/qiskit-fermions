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
use qiskit_fermions_core::operators::fermion_operator::FermionOperator;
use qiskit_fermions_core::operators::library::commutators::{
    anti_commutator, commutator, double_commutator,
};
use qiskit_fermions_core::operators::majorana_operator::MajoranaOperator;

macro_rules! impl_commutators {
    ($name:ty, $prefix:ident) => {
        paste::item! {
            /// @ingroup qf_commutators
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qf_ $prefix _commutator>](
                op_a: *const $name,
                op_b: *const $name,
            ) -> *mut $name {
                // SAFETY: Per documentation, the pointers are non-null and aligned.
                let op_a = unsafe { const_ptr_as_ref(op_a) };
                let op_b = unsafe { const_ptr_as_ref(op_b) };

                let result = commutator(op_a, op_b);
                Box::into_raw(Box::new(result))
            }

            /// @ingroup qf_commutators
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qf_ $prefix _anti_commutator>](
                op_a: *const $name,
                op_b: *const $name,
            ) -> *mut $name {
                // SAFETY: Per documentation, the pointers are non-null and aligned.
                let op_a = unsafe { const_ptr_as_ref(op_a) };
                let op_b = unsafe { const_ptr_as_ref(op_b) };

                let result = anti_commutator(op_a, op_b);
                Box::into_raw(Box::new(result))
            }

            /// @ingroup qf_commutators
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qf_ $prefix _double_commutator>](
                op_a: *const $name,
                op_b: *const $name,
                op_c: *const $name,
                sign: bool,
            ) -> *mut $name {
                // SAFETY: Per documentation, the pointers are non-null and aligned.
                let op_a = unsafe { const_ptr_as_ref(op_a) };
                let op_b = unsafe { const_ptr_as_ref(op_b) };
                let op_c = unsafe { const_ptr_as_ref(op_c) };

                let result = double_commutator(op_a, op_b, op_c, sign);
                Box::into_raw(Box::new(result))
            }
        }
    };
}

// NOTE: cbindgen cannot expand procedural macros without the nightly rust toolchain. Remember to
// declare the generated C function signatures in cbindgen.toml!
impl_commutators!(FermionOperator, ferm_op);
impl_commutators!(MajoranaOperator, maj_op);
