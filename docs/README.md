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

### Browsing a build offline

The API-reference entries in the sidebar point at the documentation hosted on the IBM Quantum
Platform, because that is what the artifact ingested by the platform needs. In a local build they
therefore navigate away from the build you are reading. To rewrite them to the corresponding pages
inside the same tree, use:
```bash
make docs-local
```

Then open `docs/_build/html/index.html`, or serve the tree with
`python -m http.server -d docs/_build/html`.

Note that this target *modifies* `docs/_build/html` in place, so use plain `make docs` if you need a
tree with the absolute links intact.

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

The `.pyi` stub files are the *only* documentation surface for the compiled `pyext` components:
since that code is native, the docstrings written in the Rust source are extracted by
[`pyo3-stub-gen`](https://crates.io/crates/pyo3-stub-gen) into the stubs, from where Sphinx (and
type checkers, and IDEs) read them. This is why the stubs must be regenerated whenever a `pyext`
docstring changes — see [`tests/README.md`](../tests/README.md) for how they are generated,
committed, and kept in sync.

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

Release notes are managed by [`reno`](https://pypi.org/project/reno/).
The [Qiskit contributing guide](https://github.com/Qiskit/qiskit/blob/main/CONTRIBUTING.md) contains
a great explanation of working with `reno` which also applies to this package.

Any user-facing change needs a note, added with `reno new <slug>` and committed alongside the change.

`reno` files a note under the first release tag reachable from the commit that *added* the file, so
the note has to be committed on the branch whose release it belongs to. In particular, a fix that is
backported to a `stable/X.Y` branch must carry its note on that branch to show up in the
corresponding patch release; adding it only to `main` files it under the next minor instead. The
directory a note lives in has no bearing on this — notes for an already-released minor are collected
into `releasenotes/notes/X.Y/` purely to keep the top level tidy, and `reno` recurses into
subdirectories when scanning.

<!-- vim: set tw=100: -->
