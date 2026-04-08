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

use crate::exit_codes::ExitCode;
use crate::pointers::mut_ptr_as_ref;

use qiskit_fermions_core::operators::fermion_operator::FermionOperator;
use qiskit_fermions_core::operators::grouping::electronic_structure::group_terms_by_electronic_structure;

/// TODO:
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qf_group_terms_by_electronic_structure(
    op: *mut FermionOperator,
    num_modes: u32,
    two_body_physicist_order: bool,
) -> ExitCode {
    let op = unsafe { mut_ptr_as_ref(op) };

    let res = group_terms_by_electronic_structure(op, num_modes, two_body_physicist_order);
    match res {
        Ok(_) => ExitCode::Success,
        Err(_) => ExitCode::ValueError,
    }
}
