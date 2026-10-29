# This code is a Qiskit project.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import logging
import os
import posixpath
import sys
from importlib.metadata import version as metadata_version

from docutils import nodes

# Reduce logging level to WARNING for the `qiskit` package to avoid unnecessary verbosity during
# documentation generation.
logging.getLogger("qiskit").setLevel(logging.WARNING)

# The following line is required for autodoc to be able to find and import the code whose API should
# be documented.
sys.path.insert(0, os.path.abspath(".."))

project = "Qiskit Fermions"
project_copyright = "2026, Qiskit addons team"
description = "An extension of Qiskit for working with fermionic systems"
author = "Qiskit addons team"
language = "en"
release = metadata_version("qiskit-fermions")

html_theme = "qiskit-ecosystem"

# This allows including custom CSS and HTML templates.
html_theme_options = {
    "dark_logo": "images/qiskit-dark-logo.svg",
    "light_logo": "images/qiskit-light-logo.svg",
    "sidebar_qiskit_ecosystem_member": False,
}
html_static_path = ["_static"]
templates_path = ["_templates"]

# Sphinx should ignore these patterns when building.
exclude_patterns = [
    "_build",
    "_ecosystem_build",
    "_qiskit_build",
    "_pytorch_build",
    "**.ipynb_checkpoints",
    "jupyter_execute",
]

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
    "sphinx.ext.linkcode",
    "sphinx.ext.intersphinx",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_copybutton",
    "sphinx_reredirects",
    "reno.sphinxext",
    "sphinx_design",
    "nbsphinx",
    "qiskit_sphinx_theme",
    "sphinxcontrib.katex",
    "breathe",
    "sphinx_llm.txt",
]

breathe_projects = {"qiskit_fermions": "xml/"}
breathe_default_project = "qiskit_fermions"
breathe_domain_by_extension = {
    "h": "c",
}

copybutton_exclude = ".linenos, .gp, .go"

# The markdown builder behind `sphinx_llm.txt` does not know every docutils node and warns about
# each one it meets, which `-W` turns into an error.  These are the three we currently hit: the
# `abbreviation` nodes that autodoc emits for the bare `*` in signatures, the `caption` of a table of
# download links, and `citation` nodes.  Naming them, rather than silencing every unknown node, keeps
# `-W` effective for any node that is new.  The nodes are still kept as inline code.
# Upstream: liran-funaro/sphinx-markdown-builder#42 and #63 (abbreviation), #76 (caption).
llms_txt_suppress_unknown_node_warnings = ["abbreviation", "caption", "citation", "meta"]
# Only emit `<page>.html.md` next to each `<page>.html`; the default ("auto") also writes a
# redundant `<page>.md` copy of every page.
llms_txt_suffix_mode = "append"
# Static text for the `llms.txt` header (no LLM call involved).
llms_txt_description = (
    "Qiskit Fermions extends Qiskit with tools for working with fermionic systems: operator data "
    "structures, a framework and library for converting operators between representations and to "
    "qubit operators (for example Jordan-Wigner and ternary-tree encodings), and a framework and "
    "library for synthesizing fermionic quantum circuits. The core is written in Rust and exposed "
    "through first-party Python and C APIs."
)

html_last_updated_fmt = "%Y/%m/%d"
html_title = f"{project} {release}"

# This allows RST files to put `|version|` in their file and
# have it updated with the release set in conf.py.
rst_prolog = f"""
.. |version| replace:: {release}
"""

# Options for autodoc. These reflect the values from Qiskit SDK and Runtime.
autosummary_generate = True
autosummary_generate_overwrite = False
autoclass_content = "both"
autodoc_typehints = "description"
autodoc_class_signature = "mixed"
autodoc_default_options = {
    "inherited-members": None,
    "show-inheritance": True,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = False
# Sphinx 9 rewrote autodoc and no longer resolves the module for members of our aliased
# classes (e.g. FermionicRegister = QuantumRegister). Use the legacy implementation until fixed.
# See https://github.com/sphinx-doc/sphinx/issues/14089
autodoc_use_legacy_class_based = True


# This adds numbers to the captions for figures, tables,
# and code blocks.
numfig = True
numfig_format = {"table": "Table %s"}

# Settings for Jupyter notebooks.
nbsphinx_execute = "never"

add_module_names = False

modindex_common_prefix = ["qiskit_fermions."]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    # scipy is the one entry whose target and inventory differ, deliberately.  `docs.scipy.org` is
    # self-hosted (every other entry here sits behind a CDN) and has repeatedly black-holed the docs
    # build: measured 20.8s for `objects.inv` on a good day, then no response at all.  With `-W`
    # the resulting "failed to reach any of the inventories" warning is a hard error, and with `-E`
    # the environment pickle is discarded every run so the inventory is never cached.
    # `static.scipy.org` mirrors `objects.inv` behind a CDN (0.36s, refreshed hourly by a cron job
    # on `docs.scipy.org`) but serves *only* that file -- its HTML 404s -- so it cannot be the
    # target: intersphinx builds link hrefs from the target, not the inventory.  Hence fetch the
    # inventory from the mirror while still pointing readers at the real docs.  The `None` fallback
    # must stay *second*, since locations are tried in order and the first success wins.
    # See https://github.com/scipy/docs.scipy.org/issues/102 for the upstream suggestion.
    "scipy": (
        "https://docs.scipy.org/doc/scipy/",
        ("https://static.scipy.org/doc/scipy/objects.inv", None),
    ),
    "qiskit": ("https://quantum.cloud.ibm.com/docs/api/qiskit/", None),
    "cqiskit": ("https://quantum.cloud.ibm.com/docs/api/qiskit-c/", None),
    "qiskit_addon_sqd": ("https://quantum.cloud.ibm.com/docs/api/qiskit-addon-sqd/", None),
    "pyomo": ("https://pyomo.readthedocs.io/en/stable/", None),
    "ffsim": ("https://qiskit-community.github.io/ffsim/", None),
    "pyscf": ("https://pyscf.org/", None),
}

# Bound each inventory fetch.  Nothing in the mapping above takes more than half a second in
# practice, so this only ever fires for a host that is genuinely unreachable -- where a fast red
# build beats burning the CI job's 30-minute budget on one hung socket.  It also caps the cost of
# the scipy fallback above: without a timeout, falling back to `docs.scipy.org` could hang
# indefinitely.
intersphinx_timeout = 30

plot_working_directory = "."
plot_html_show_source_link = False

# ----------------------------------------------------------------------------------
# Redirects
# ----------------------------------------------------------------------------------

_inlined_apis = []

redirects = {
    "pydoc/qiskit_fermions": "./index.html",
    **{
        f"stubs/{module}.{name}": f"../rydoc/{module}.html#{module}.{name}"
        for module, name in _inlined_apis
    },
}

# ----------------------------------------------------------------------------------
# Source code links
# ----------------------------------------------------------------------------------


def linkcode_resolve(domain, info):
    return None


# ----------------------------------------------------------------------------------
# LLM-friendly Markdown output (`sphinx_llm.txt`)
# ----------------------------------------------------------------------------------


def _adapt_doctree_for_markdown(app, doctree, docname):
    """Adjust the doctree of the Markdown sub-build that `sphinx_llm.txt` runs next to the HTML build.

    Only active in that sub-build, so the HTML output is untouched.  Three things differ:

    * Images.  `sphinx_markdown_builder` writes an image node's `uri` verbatim, but at that point it
      is a path relative to the *source root* (or, for the `plot` directive, to wherever the
      sub-build wrote the figure), so it resolves to nothing from the page that contains it.  The
      HTML build copies every image to `_images/` under a unique name, and the two outputs are merged
      into one tree afterwards, so linking to that copy gives a working relative path.
      Upstream: liran-funaro/sphinx-markdown-builder#33.
    * Comments.  RST comments become `<!-- -->` blocks in the Markdown, and the description scraper
      of `sphinx_llm.txt` only skips the first line of such a block, so a multi-line comment ends up
      as the page's description in `llms.txt`.  They carry nothing for a reader, so drop them.
    * `Bases: ...`.  Autodoc puts this line first in every class entry, so the scraper picks it as
      the description of the page.  Moving it behind the first prose paragraph keeps the inheritance
      information and lets the docstring's own summary line become the description.
    """
    if not app.tags.has("sphinx_llm_markdown"):
        return

    page_dir = posixpath.dirname(docname) or "."
    for node in doctree.findall(nodes.image):
        entry = app.env.images.get(node["uri"])
        if entry is not None:
            node["uri"] = posixpath.relpath(f"_images/{entry[1]}", page_dir)

    for node in list(doctree.findall(nodes.comment)):
        node.parent.remove(node)

    for node in list(doctree.findall(nodes.paragraph)):
        if not node.astext().startswith("Bases: "):
            continue
        parent = node.parent
        position = parent.index(node)
        following = [
            child for child in parent.children[position + 1 :] if isinstance(child, nodes.paragraph)
        ]
        if following:
            parent.remove(node)
            parent.insert(parent.index(following[0]) + 1, node)


def setup(app):
    app.connect("doctree-resolved", _adapt_doctree_for_markdown)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
