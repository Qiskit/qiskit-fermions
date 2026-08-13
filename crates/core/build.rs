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
    println!("cargo:rerun-if-changed=build.rs");
    // The rpath emitted below is derived from `QISKIT_LIB`, so a change to it has to re-run this
    // script.  Without this, an existing `target/` directory keeps the rpath baked in from whatever
    // `QISKIT_LIB` was set when it was first built.  That is easy to miss because the stale path
    // often still *exists* (CI workspace paths are stable), so the link succeeds and silently
    // points at the wrong Qiskit library rather than failing outright.
    println!("cargo::rerun-if-env-changed=QISKIT_LIB");

    if cfg!(feature = "cext") {
        let qiskit_lib = env::var("QISKIT_LIB").unwrap();

        let qiskit_lib_path = Path::new(&qiskit_lib);

        assert!(
            qiskit_lib_path.try_exists().unwrap(),
            "Qiskit path does not exist"
        );

        let qiskit_lib_dir = qiskit_lib_path.parent().unwrap().to_str().unwrap();
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", qiskit_lib_dir);
    }
}
