// This code is a Qiskit project.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_circuit() {
        unsafe {
            let qc = qk_circuit_new(0, 0);
            assert_eq!(0, qk_circuit_num_qubits(qc));
            assert_eq!(0, qk_circuit_num_clbits(qc));
            assert_eq!(0, qk_circuit_num_instructions(qc));
            let mut op_counts = qk_circuit_count_ops(qc);
            assert_eq!(0, op_counts.len);

            qk_circuit_free(qc);
            qk_opcounts_clear(&mut op_counts);
        }
    }
}
