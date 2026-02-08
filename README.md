<!-- SHIELDS -->
<div align="left">

  [![License](https://img.shields.io/github/license/Qiskit/qiskit-fermions?label=License)](LICENSE.txt)
  [![Docs](https://img.shields.io/badge/%F0%9F%93%84%20Docs-stable-blue.svg)](https://qiskit.github.io/qiskit-fermions/)
  [![Qiskit](https://img.shields.io/badge/Qiskit%20-%20%3E%3D2.3%20-%20%236133BD?logo=Qiskit)](https://github.com/Qiskit/qiskit)
  [![Python](https://img.shields.io/badge/python-3.10%7C3.11%7C3.12%7C3.13-blue.svg)](https://www.python.org/)
  [![rustc](https://img.shields.io/badge/rustc-1.91+-blue.svg)](https://rust-lang.github.io/rfcs/2495-min-rust-version.html)
  ![Platform](https://img.shields.io/badge/%F0%9F%92%BB%20Platform-Linux%20%7C%20macOS-informational)
  [![Tests](https://github.com/Qiskit/qiskit-fermions/actions/workflows/test_development_versions.yml/badge.svg)](https://github.com/Qiskit/qiskit-fermions/actions/workflows/test_development_versions.yml)
  [![Coverage](https://coveralls.io/repos/github/Qiskit/qiskit-fermions/badge.svg?branch=main)](https://coveralls.io/github/Qiskit/qiskit-fermions?branch=main)
</div>

# Qiskit Fermions

> [!WARNING]
> This package is under active development!
> If you have feedback, please [open an issue](https://github.com/Qiskit/qiskit-fermions/issues/new/choose).

### Table of contents

* [About](#about)
* [Documentation](#documentation)
* [Installation](#installation)
* [Deprecation Policy](#deprecation-policy)
* [Contributing](#contributing)
* [License](#license)

----------------------------------------------------------------------------------------------------

### About

This package extends Qiskit with tools for working on fermionic systems.
It follows a similar design philosophy as Qiskit itself:

- the core functionality is written in Rust
- first-party language bindings are provided for both, Python and C
- the APIs are designed to "feel" similar to Qiskit

The scope of this package includes the following functionality:

- several operator data structures
- a framework to develop operator conversion methods
- a library of common operator converters
- a framework to develop quantum circuit synthesis methods
- a library of common quantum circuit synthesizers

----------------------------------------------------------------------------------------------------

### Documentation

All documentation is available at https://qiskit.github.io/qiskit-fermions/.

----------------------------------------------------------------------------------------------------

### Installation

First, follow the [installation setup instructions](docs/install.rst).
Then, you can follow the language specific installation steps linked below:

#### C

Please refer to the [C installation instructions](docs/install-c.rst).

#### Python

Please refer to the [Python installation instructions](docs/install-py.rst).

----------------------------------------------------------------------------------------------------

### Deprecation Policy

We follow [semantic versioning](https://semver.org/) and are guided by the principles in
[Qiskit's deprecation policy](https://github.com/Qiskit/qiskit/blob/main/DEPRECATION.md).
We may occasionally make breaking changes in order to improve the user experience.
When possible, we will keep old interfaces and mark them as deprecated, as long as they can co-exist with the
new ones.
Each substantial improvement, breaking change, or deprecation will be documented in the
[release notes](https://qiskit.github.io/qiskit-fermions/release-notes.html).

----------------------------------------------------------------------------------------------------

### Contributing

The source code is available [on GitHub](https://github.com/Qiskit/qiskit-fermions).

The developer guide is located at
[CONTRIBUTING.md](https://github.com/Qiskit/qiskit-fermions/blob/main/CONTRIBUTING.md)
in the root of this project's repository.
By participating, you are expected to uphold Qiskit's [code of conduct](https://github.com/Qiskit/qiskit/blob/main/CODE_OF_CONDUCT.md).

We use [GitHub issues](https://github.com/Qiskit/qiskit-fermions/issues/new/choose)
for tracking requests and bugs.

----------------------------------------------------------------------------------------------------

### License

[Apache License 2.0](LICENSE.txt)

<!-- vim: set tw=100: -->
