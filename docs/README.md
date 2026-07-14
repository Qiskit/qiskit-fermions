# Documentation

The documentation for this package is built with [Sphinx](https://www.sphinx-doc.org/en/master/).
Its configuration and static packages reside in `docs/`.
The content is formatted using
[reStructuredText (RST)](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html).
Whenever possible, API documentation should be written directly inside the source code as explained
below.


The whole documentation can be generated using a simple `make` target:
```bash
make docs
```

This includes both, the Python and C API documentation.

To generate a clean build, run the following `make` target first:
```bash
make docsclean
```

## Rust

Even though this package's core is implemented in Rust, we do not provide a stable Rust API.
Thus, there is no public Rust API documentation.

However, Rust code should still be documented properly to ensure that developers working on this
package can understand and navigate the codebase more easily. To this end, we follow the guidelines
set forth by [`rustdoc`](https://doc.rust-lang.org/rustdoc/how-to-write-documentation.html).

## Python

The Python API documentation is structured as follows:

- `docs/pydoc/` contains RST files to configure the overall page layout
- API documentation gets pulled directly from the `python/qiskit_fermions/` source
- components implemented in the `pyext` Rust crate should write their docstrings directly there and
  expose them via the Python stub files (`.pyi`)

## C

The C API documentation gets parsed using [Doxygen](https://www.doxygen.nl/index.html) and then
integrated into the Sphinx documentation using [breathe](https://www.breathe-doc.org/).

This setup is configured via `docs/Doxyfile` and `docs/conf.py`.

The structure of the C API documentation is similar to that of Python:

- `docs/cdoc/` contains RST files to configure the overall page layout (but here this is likely a
  lot more elaborate than in the Python case)
- API documentation gets pulled from the automatically generated C header file
  (`dist/c/include/qiskit_fermions.h`)
- the docstrings included in that header file are written directly in the `cext` Rust crate

Note, that the C API docstrings in the `cext` crate are a mixture of Doxygen and Sphinx
directives. Generally speaking, the following Doxygen directives should be used:

- `@ingroup` to specify which `doxygengroup` (see `docs/cdoc/index.h`) to place this docstring in
- `@brief` for a short description of this function or struct
- `@param` for the description of an argument
- `@return` for the description of the return type

More elaborate explanations as well as code examples should be written inside a `@rst`/`@endrst`
block such that Sphinx can parse and render the contents.

Cross-references to other C API objects can be created only inside RST blocks/files and need to be
correctly pre-fixed, for example like so: ``:c:func:`qf_ferm_op_free` ``.

## Release Notes

> **Note:** Release notes are not being tracked yet. Until the first release is tagged, `main` is an
> unstable development branch and changes do **not** need a `reno` entry. The workflow below applies
> once the first release is tagged.

Release notes are managed by [`reno`](https://pypi.org/project/reno/).
The [Qiskit contributing guide](https://github.com/Qiskit/qiskit/blob/main/CONTRIBUTING.md) contains
a great explanation of working with `reno` which also applies to this package.

<!-- vim: set tw=100: -->
