# Testing

Given the multi-language nature of this package, we are dealing with one test suite for each
language. These are discussed individually below.

## Rust

Unittests of Rust crates are included directly in the source code. They can get executed using
`cargo test` for which a simple `make` target is provided in this repo, to ensure that the dynamic
library dependencies get linked correctly:
```bash
make testrust
```

This will run the tests included in the `core` crate which contains all native Rust logic of this
package.
Refer to [this readme](`crates/README.md`) for an explanation of the different crates.

> **Note:** `make testrust` builds the `core` crate with its `cext` feature (linking `qiskit-sys`),
> so — just like the C API — it requires `QISKIT_LIB` and `QISKIT_INCLUDE` to point at your compiled
> Qiskit **C** library. See the [C installation instructions](../docs/install-c.rst) for how to set
> these. (The Python test target does *not* need them, because `pyext` links Qiskit through
> `qiskit-pyo3-ffi` instead.)

If at some point we start writing integration tests for the Rust API, these should be implemented in
`crates/core/tests` to ensure that `cargo test` will find them.
See also [here](https://doc.rust-lang.org/rust-by-example/testing/integration_testing.html).

## Python

Unittest for the Python API are placed in `tests/python`. Additionally, we should strive for good
examples in the API documentation which is written in the docstrings of the `pyext` crate. These
should be formatted as [doctests](https://docs.python.org/3/library/doctest.html).

All of these tests are detected and run by `pytest` for which another simple `make` target exists:
```bash
mask testpython
```

Note, that this target requires compilation of the `pyext` crate which not only ensures that the
compiled Rust code is up-to-date but also re-generates the Python stub files via the
[`pyo3-stub-gen` crate](https://crates.io/crates/pyo3-stub-gen). These Python type information files
contain the API documentation and therefore must be updated for the doctests to be up-to-date.

## C

Tests for the C API are placed in `tests/c` and they get executed via `ctest` which in turn is built
with `cmake` (see also [here](https://cmake.org/cmake/help/book/mastering-cmake/chapter/Testing%20With%20CMake%20and%20CTest.html)).
Again, there is a simple `make` target for this:
```bash
make testc
```

This in turn requires compilation of the `cext` crate which also ensures that the C header file gets
updated via [`cbindgen`](https://github.com/mozilla/cbindgen).

> **Note:** like `make testrust`, this links `qiskit-sys` against the compiled Qiskit **C** library,
> so `QISKIT_LIB` and `QISKIT_INCLUDE` must be set (see the
> [C installation instructions](../docs/install-c.rst)).

## Coverage

- [ ] figure out how to measure coverage in these different settings and document it here

<!-- vim: set tw=100: -->
