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
use std::process::Command;

enum InstallMethod {
    Clone,
    Path(String),
}

#[derive(Debug)]
struct CargoCallbacks;

impl bindgen::callbacks::ParseCallbacks for CargoCallbacks {
    fn process_comment(&self, comment: &str) -> Option<String> {
        Some(format!("````ignore\n{}\n````", comment))
    }
}

// There are two installation methods:
// - Clone (no path specified): Automatically clones and builds the qiskit c api from source
//     Set envvar export QISKIT_CEXT_INSTALL_METHOD="clone" to use the clone method. WARNING, cloning and building from
//     source is very slow.
// - Path (Manually specified path): Uses qiskit c api binary or source from a path
//     export QISKIT_CEXT_INSTALL_METHOD="path"
//     export QISKIT_CEXT_PATH="path/to/qiskit-cext-dir"
fn check_installation_method() -> InstallMethod {
    let qiskit_cext_path = env::var("QISKIT_CEXT_PATH");
    match env::var("QISKIT_CEXT_INSTALL_METHOD") {
        Ok(val) => match val.as_str() {
            "path" => InstallMethod::Path(qiskit_cext_path.expect("QISKIT_CEXT_PATH is unset")),
            "clone" => InstallMethod::Clone,
            _ => panic!(
                "\"{}\" is not a valid input to QISKIT_CEXT_INSTALL_METHOD, please specify one of the following options: (\"path\", \"clone\")",
                val
            ),
        },
        Err(e) => match e {
            env::VarError::NotPresent => InstallMethod::Clone,
            env::VarError::NotUnicode(_) => {
                panic!("Envvar QISKIT_CEXT_INSTALL_METHOD is not unicode")
            }
        },
    }
}

fn clone_qiskit(source_path: &Path) {
    let url = "https://github.com/Qiskit/qiskit.git";
    match git2::Repository::clone(url, source_path) {
        Ok(repo) => {
            println!("Repository successfully cloned");
            let refname = env!("CARGO_PKG_VERSION");
            if !refname.contains("dev") {
                let (obj, _) = repo
                    .revparse_ext(refname)
                    .unwrap_or_else(|_| panic!("{} not found in repo", refname));
                repo.checkout_tree(&obj, None)
                    .unwrap_or_else(|_| panic!("failed to checkout {}", refname));
            }
        }
        Err(e) => match e.code() {
            git2::ErrorCode::Exists => {
                println!("Repository already exists");
                let refname = env!("CARGO_PKG_VERSION");
                if !refname.contains("dev") {
                    let repo = git2::Repository::open(source_path)
                        .unwrap_or_else(|_| panic!("Invalid repo at {:?}", source_path));
                    let (obj, _) = repo
                        .revparse_ext(refname)
                        .unwrap_or_else(|_| panic!("{} not found in repo", refname));
                    // Reset the repository in case of any untracked changes
                    repo.reset(&obj, git2::ResetType::Soft, None)
                        .expect("Error resetting repository.");
                    repo.checkout_tree(&obj, None)
                        .unwrap_or_else(|_| panic!("failed to checkout {}", refname));
                }
            }
            _ => panic!("Git clone failed: {e:?}"),
        },
    }
}

fn build_qiskit(source_path: &Path) {
    let _ = Command::new("make")
        .current_dir(source_path)
        .env("CARGO_BUILD_TARGET", env::var("TARGET").unwrap())
        .arg(format!(
            "C_CARGO_TARGET_DIR=target/{}/release",
            env::var("TARGET").unwrap()
        ))
        .arg("c")
        .status()
        .expect("Dynamically linked library generation failed");
}

fn build_qiskit_from_source() {
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let source_path = Path::new(&out_dir).join("qiskit_c_lib");
    let source_path = source_path.as_path();

    clone_qiskit(source_path);

    match source_path.try_exists() {
        Ok(b) => match b {
            true => {}
            false => panic!("Qiskit source path does not exist"),
        },
        Err(e) => panic!("{e:?}"),
    }

    build_qiskit(source_path);

    let repo_dir_str = source_path.to_str().unwrap();
    generate_bindings(repo_dir_str);
}

fn generate_bindings(qiskit_path_str: &str) {
    let qiskit_path = Path::new(&qiskit_path_str);

    match qiskit_path.try_exists() {
        Ok(b) => match b {
            true => {}
            false => panic!("Qiskit path does not exist"),
        },
        Err(e) => panic!("{e:?}"),
    }

    let subpath = if cfg!(feature = "python_binding") {
        "qiskit"
    } else {
        "dist/c/lib"
    };

    println!("cargo:rustc-link-search={}/{}", qiskit_path_str, subpath);
    println!("cargo:rustc-link-lib=qiskit");

    let bindings: bindgen::Bindings = if cfg!(feature = "python_binding") {
        bindgen::Builder::default()
            .clang_arg("-DQISKIT_C_PYTHON_INTERFACE=1")
            .raw_line("use pyo3::ffi::PyObject;")
            .header(format!("{}/dist/c/include/qiskit.h", qiskit_path_str))
            .header(format!(
                "{}/dist/c/include/qiskit/complex.h",
                qiskit_path_str
            ))
            .allowlist_item("^(qk_.*)$")
            .allowlist_item("^(Qk.*)$")
            .blocklist_item("^(Py.*)$")
            .opaque_type("PyObject")
            .parse_callbacks(Box::new(CargoCallbacks))
            .generate()
            .expect("Unable to generate bindings")
    } else {
        bindgen::Builder::default()
            .header(format!("{}/dist/c/include/qiskit.h", qiskit_path_str))
            .header(format!(
                "{}/dist/c/include/qiskit/complex.h",
                qiskit_path_str
            ))
            .parse_callbacks(Box::new(CargoCallbacks))
            .generate()
            .expect("Unable to generate bindings")
    };

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo::rerun-if-env-changed=QISKIT_CEXT_INSTALL_METHOD");
    println!("cargo::rerun-if-env-changed=QISKIT_CEXT_PATH");

    let install_method = check_installation_method();

    match install_method {
        InstallMethod::Clone => {
            println!("cargo::warning=Cloning and building from source is very slow");
            build_qiskit_from_source();
        }
        InstallMethod::Path(path) => {
            generate_bindings(&path);
        }
    };
}
