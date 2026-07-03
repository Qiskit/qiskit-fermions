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

// The `cext` and `pyext` features select different FFI backends (`qiskit-sys` versus
// `qiskit-pyo3-ffi`) for the same `ffi` alias, so they are mutually exclusive. Enabling both at
// once would let Cargo's feature unification compile this crate against the wrong backend. Guard
// against it explicitly rather than fail with a confusing type mismatch downstream.
#[cfg(all(feature = "cext", feature = "pyext"))]
compile_error!("features `cext` and `pyext` are mutually exclusive; enable exactly one");

pub mod linalg;
pub mod mappers;
pub mod operators;
pub mod random;
pub mod testing;
