# Rust crates

This document explains the structure and roles of the various Rust crates that are being developed
in this package.

## `core`

This crate implements the core functionality of this entire package in Rust.
Its module structure mirrors that of the public Python API.

Below are some guidelines that should be adhered to when working on this crate:

- This crate shall **NOT** depend on `pyo3`. This enforces a clean separation of the Rust core from
  the Python API and will allow the C API built on top of this core to be fully independent from
  Python (in the future, once the Qiskit Rust core fulfills the same criterion).
- As long as we do not plan a public Rust API, this crate can depend on Qiskit's FFI crates. If we
  desire to publish a Rust crate in the future, this dependency may be problematic (TODO: figure out
  the implications of this in more detail). If that time comes, we may choose to separate those parts
  of this crate that depend on Qiskit out into a separate crate. That should leave us with a core data
  model crate in Rust, whose usage may be useful to other Rust packages (e.g. `ffsim`).

The Qiskit FFI backend is feature-gated, because the `pyext` and `cext` crates require *different,
mutually exclusive* backends:

- The `cext` feature pulls in `qiskit-sys` (raw bindings to the compiled `libqiskit`).
- The `pyext` feature pulls in `qiskit-pyo3-ffi` (Qiskit's PyO3 FFI crate).

Both dependencies are `optional`, and the default feature set enables neither. Enabling both `cext`
and `pyext` at once is a hard error — a `compile_error!` in `src/lib.rs` guards against it, since
Cargo's feature unification would otherwise silently compile the crate against the wrong backend.

## `pyext`

This crate defines the public Python API on top of the `core` crate.
As such, it obviously depends on the `pyo3` crate.
It depends on `core` with its `pyext` feature and links Qiskit through the `qiskit-pyo3-ffi` crate.
Using `qiskit-pyo3-ffi` (rather than linking against a locally compiled Qiskit acceleration binary)
removes the hard-coded link step and is what unlocks Windows support for the Python package.

Just like the `core` crate, the module structure mirrors that of the public Python API and we
leverage `pyo3`'s modern `#[pymodule]` and `#[pymodule_export]` macros to implement the same Python
module layout. Note, that we try to automatically register these nested Python modules in
`sys.modules` during the loading of `python/qiskit_fermions/__init__.py`.

Additionally, the `pyo3-stub-gen` crate is leveraged to generate
[Python stub files](https://typing.python.org/en/latest/guides/writing_stubs.html) to provide proper
type hinting to the Python package as well as to extract the Python API documentation and doctests
from the `pyext` docstrings.

## `cext`

This crate defines the public C API.
Even though C does not have a module structure like Rust or Python, this crate still mirrors the
module structure for ease of navigation.
It depends on `core` with its `cext` feature and links Qiskit's compiled core through the
`qiskit-sys` crate.

The C API header gets generated using `cbindgen` which is done automatically during this crate's
build process.

<!-- vim: set tw=100: -->
