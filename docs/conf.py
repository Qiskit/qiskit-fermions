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

import os
import sys
from importlib.metadata import version as metadata_version

# The following line is required for autodoc to be able to find and import the code whose API should
# be documented.
sys.path.insert(0, os.path.abspath(".."))

project = "Qiskit Fermions"
project_copyright = "2025, Qiskit addons team"
description = "Qiskit for Fermions"
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
    "pytest_doctestplus.sphinx.doctestplus",
    "sphinxcontrib.katex",
    "breathe",
]

breathe_projects = {"qiskit_fermions": "xml/"}
breathe_default_project = "qiskit_fermions"
breathe_domain_by_extension = {
    "h": "c",
}

copybutton_exclude = ".linenos, .gp, .go"

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
autoclass_content = "class"
autodoc_typehints = "description"
autodoc_class_signature = "separated"
autodoc_default_options = {
    "inherited-members": None,
    "show-inheritance": True,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = False


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
    "qiskit": ("https://quantum.cloud.ibm.com/docs/api/qiskit/", None),
    "cqiskit": ("https://quantum.cloud.ibm.com/docs/api/qiskit-c/", None),
}

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
