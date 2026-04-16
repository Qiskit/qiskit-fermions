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

#[derive(Debug)]
struct CargoCallbacks;

impl bindgen::callbacks::ParseCallbacks for CargoCallbacks {
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
        .parse_callbacks(Box::new(CargoCallbacks))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}

fn generate_bindings_py() {
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
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", qiskit_lib_dir);
    println!("cargo:rustc-link-arg={}", qiskit_lib);

    let bindings: bindgen::Bindings = bindgen::Builder::default()
        .clang_arg(format!("-I{}", qiskit_include))
        .clang_arg("-DQISKIT_C_PYTHON_INTERFACE=1")
        .raw_line("use pyo3::ffi::PyObject;")
        .header(format!("{}/qiskit.h", qiskit_include))
        .allowlist_item("^(qk_.*)$")
        .allowlist_item("^(Qk.*)$")
        .blocklist_item("^(Py.*)$")
        .opaque_type("PyObject")
        .parse_callbacks(Box::new(CargoCallbacks))
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

    if cfg!(feature = "python_binding") {
        generate_bindings_py();
    } else {
        generate_bindings_c();
    }
}
