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

use std::env;
use std::path::{Path, PathBuf};

/// Wraps doc comments carried over from the C headers in an ```` ```ignore ```` fence, so that
/// rustdoc does not try to compile the C snippets in them as Rust doctests.
///
/// Named to distinguish it from [`bindgen::CargoCallbacks`], which is registered alongside it below
/// and would otherwise shadow this type.
#[derive(Debug)]
struct CommentCallbacks;

impl bindgen::callbacks::ParseCallbacks for CommentCallbacks {
    fn process_comment(&self, comment: &str) -> Option<String> {
        Some(format!("````ignore\n{}\n````", comment))
    }
}

fn generate_bindings_c() {
    let qiskit_lib = env::var("QISKIT_LIB").unwrap();
    let qiskit_include = env::var("QISKIT_INCLUDE").unwrap();

    let qiskit_lib_path = Path::new(&qiskit_lib);

    match qiskit_lib_path.try_exists() {
        Ok(b) => match b {
            true => {}
            false => panic!("Qiskit path does not exist"),
        },
        Err(e) => panic!("{e:?}"),
    }

    let qiskit_lib_dir = qiskit_lib_path.parent().unwrap().to_str().unwrap();

    println!("cargo:rustc-link-search={}", qiskit_lib_dir);
    if std::env::var_os("CARGO_CFG_TARGET_OS").unwrap() == "windows" {
        println!("cargo:rustc-link-lib=qiskit_cext.dll");
    } else {
        println!("cargo:rustc-link-lib=qiskit");
    }

    let bindings: bindgen::Bindings = bindgen::Builder::default()
        .clang_arg(format!("-I{}", qiskit_include))
        .header(format!("{}/qiskit.h", qiskit_include))
        // `CargoCallbacks` emits a `rerun-if-changed` line for every header bindgen actually
        // parses.  The `rerun-if-env-changed` declarations in `main` only cover the *paths* to the
        // Qiskit C API changing -- but those paths are constant in CI by construction, while their
        // *contents* move whenever the Qiskit version being built against does.  Without this, an
        // existing `target/` directory reuses bindings generated from the previous headers, which
        // can silently mismatch the library actually being linked.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .parse_callbacks(Box::new(CommentCallbacks))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo::rerun-if-env-changed=QISKIT_LIB");
    println!("cargo::rerun-if-env-changed=QISKIT_INCLUDE");

    generate_bindings_c();
}
