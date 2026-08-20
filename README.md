<!-- SHIELDS -->
<div align="left">

  [![Release](https://img.shields.io/pypi/v/qiskit-fermions?label=Release&logo=pypi&logoColor=white)](https://pypi.org/project/qiskit-fermions/)
  [![License](https://img.shields.io/github/license/Qiskit/qiskit-fermions?label=License)](LICENSE.txt)
  [![Docs](https://img.shields.io/badge/%F0%9F%93%84%20Docs-stable-blue.svg)](https://quantum.cloud.ibm.com/docs/addons/qiskit-fermions)
  [![Qiskit](https://img.shields.io/badge/Qiskit%20-%20~%3D2.5.0%20-%20%236133BD?logo=Qiskit)](https://github.com/Qiskit/qiskit)
  [![Python](https://img.shields.io/badge/python-3.10%7C3.11%7C3.12%7C3.13%7C3.14-blue.svg)](https://www.python.org/)
  [![rustc](https://img.shields.io/badge/rustc-1.95+-blue.svg)](https://rust-lang.github.io/rfcs/2495-min-rust-version.html)
  ![Platform](https://img.shields.io/badge/%F0%9F%92%BB%20Platform-Linux%20%7C%20macOS%20%7C%20Windows-informational)
  [![Tests](https://github.com/Qiskit/qiskit-fermions/actions/workflows/test_development_versions.yml/badge.svg)](https://github.com/Qiskit/qiskit-fermions/actions/workflows/test_development_versions.yml)
  [![Coverage](https://coveralls.io/repos/github/Qiskit/qiskit-fermions/badge.svg?branch=main)](https://coveralls.io/github/Qiskit/qiskit-fermions?branch=main)
</div>

# Qiskit Fermions

`qiskit-fermions` extends the Qiskit SDK with tools for working on fermionic systems. Within its
scope it provides the following:

- Efficient data structures for the representation and manipulation of fermionic operators in
  different forms
- A framework for implementing operator conversion methods (including fermion-to-qubit encodings)
- A library of efficient implementations of common conversion methods
- A framework and library of gates for expressing fermionic circuits
- A transpilation pipeline integrated with the Qiskit transpiler process to synthesize fermionic
  circuits into qubit-based circuits

Additionally, `qiskit-fermions` integrates with other tools of the ecosystem, such as:

- [ffsim](https://qiskit-community.github.io/ffsim/) for efficient simulation of fermionic circuits

----------------------------------------------------------------------------------------------------

### Documentation

[Documentation](https://quantum.cloud.ibm.com/docs/addons/qiskit-fermions) for this package is
available on IBM Quantum Platform.

----------------------------------------------------------------------------------------------------

### Installation

#### C

Refer to the [C installation instructions](docs/install-c.rst).

#### Python

Install this package via `pip`, when possible:

```bash
pip install 'qiskit-fermions'
```

For more installation information, refer to these [installation instructions](docs/install.rst).

----------------------------------------------------------------------------------------------------

### Get started

Several guides exist to help you get started with this package. For an overview of its breadth of
features, visit the [1D Fermi-Hubbard guide](docs/guides/1d_fermi_hubbard.rst).

----------------------------------------------------------------------------------------------------

### Use case examples

Components of this package have been used in research related to the following
papers:

- The qDRIFT randomized circuit compilation in [^1].
- The fermion-to-qubit synthesis during the transpilation process in [^2].

----------------------------------------------------------------------------------------------------

### Technical discussion

#### Design intentions

This package is deliberately designed to align with Qiskit: it builds on a core
implemented in Rust and provides first-party language bindings to Python and C.
Its API intends to draw parallels to Qiskit in order to seamlessly integrate
into the workflows of users with experience in programming Qiskit.

A core principle to the design of `qiskit-fermions` was the decoupling of its
fermionic circuit representation from its qubitized form. To be more precise:
the fermionic circuits are meaningful by themselves and do not require a mapping
to qubit space to be interpretable.
Furthermore, the fermionic circuit representation cannot make any assumptions
about its fermion-to-qubit encoding applied later on. Consequently, while
Jordan-Wigner retains a dominant position and role, it is not assumed to be the
_default_ fermion-to-qubit encoding.

#### Known issues

As long as the Qiskit C API has not yet reached feature parity with its Python
API, some components of this package remain exclusive to its Python API, too.
This includes the entire circuit library (`qiskit_fermions.circuit`) as
well as transpiler passes (`qiskit_fermions.transpiler`).

#### Future work

- Migrate the circuit library and transpiler passes into the Rust core (and
  provide a C API for interacting with them)
- Extend the library of efficient operator conversion implementations
- Extend the library of efficient operator data structures

----------------------------------------------------------------------------------------------------

### Contributing

The source code is available [on GitHub](https://github.com/Qiskit/qiskit-fermions).

The developer guide is located at
[CONTRIBUTING.md](https://github.com/Qiskit/qiskit-fermions/blob/main/CONTRIBUTING.md)
in the root of this project's repository.
By participating, you are expected to uphold Qiskit's
[code of conduct](https://github.com/Qiskit/qiskit/blob/main/CODE_OF_CONDUCT.md).

----------------------------------------------------------------------------------------------------

### Citing this package

If you use this package in your research, use the [CITATION.bib](CITATION.bib) file in this project’s
repository to cite the appropriate reference(s).

----------------------------------------------------------------------------------------------------

### License

[Apache License 2.0](LICENSE.txt)

----------------------------------------------------------------------------------------------------

### Deprecation policy

This package follows [semantic versioning](https://semver.org/). Breaking changes are made only
occasionally, to improve the user experience. When possible, old interfaces are kept and marked as
deprecated for as long as they can co-exist with the new ones. Each substantial improvement,
breaking change, or deprecation is documented in the
[release notes](https://quantum.cloud.ibm.com/docs/api/qiskit-fermions/release-notes).

----------------------------------------------------------------------------------------------------

### References

[^1]: Samuele Piccinelli, et al., [Quantum chemistry with provable convergence via randomized
sample-based Krylov quantum diagonalization](https://arxiv.org/abs/2508.02578v2), arXiv:2508.02578 [quant-ph].

[^2]: Anthony Gandon, et al., [Stabilizer-based quantum simulation of fermion dynamics with local
qubit encodings](https://arxiv.org/abs/2512.11418v2), arXiv:2512.11418 [quant-ph].

<!-- vim: set tw=100: -->
