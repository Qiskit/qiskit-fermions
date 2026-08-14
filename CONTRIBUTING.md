# Developer guide

Development of the `qiskit-fermions` package takes place
[on GitHub](https://github.com/Qiskit/qiskit-fermions).

This package is part of the Qiskit stack and a lot of the content from
[its contributing guide](https://github.com/Qiskit/qiskit/blob/main/CONTRIBUTING.md)
also applies here.

As mentioned in the [readme](README.md), the core of this package is implemented in Rust and native
APIs are provided for both, Python and C.

Ideally, any new feature being added should follow this pattern:

1. The feature is implemented directly in the Rust core. It should be tested as part of the crate in
   which it gets implemented and be given some minimal documentation (for internal developers since
   the Rust API is not public).
2. It then gets exposed to the Python API via the `pyext` crate, alongside with tests in
   `tests/python` and proper API documentation.
3. Finally, it also gets exposed to the C API via the `cext` crate, alongside with tests in
   `tests/c` and proper API documentation.

Of course, this overhead may not always be worthwhile and this should be discussed and decided on a
case-by-case basis. When that is the case, a new feature should be implemented in the Python API
directly from where it may later get refactored to the Rust core.

<hr/>

In the following sections we discuss the details of the testing and documentation writing process.
Installation instructions are provided separately [here](docs/install.rst).

## Testing

Please refer to [`tests/README.md`](tests/README.md).

## Documentation

Please refer to [`docs/README.md`](docs/README.md).

## Releasing

Please refer to [`docs/release-process.md`](docs/release-process.md).

<!-- vim: set tw=100: -->
