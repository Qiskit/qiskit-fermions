// This code is a Qiskit project.
//
// (C) Copyright IBM 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use std::env;
use std::path::Path;

fn main() {
    let qiskit_lib = env::var("QISKIT_LIB").unwrap();

    let qiskit_lib_path = Path::new(&qiskit_lib);

    match qiskit_lib_path.try_exists() {
        Ok(b) => match b {
            true => {}
            false => panic!("Qiskit path does not exist"),
        },
        Err(e) => panic!("{e:?}"),
    }

    let qiskit_lib_dir = qiskit_lib_path.parent().unwrap().to_str().unwrap();
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", qiskit_lib_dir);
}
