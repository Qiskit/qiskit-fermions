# This code is a Qiskit project.
#
# (C) Copyright IBM 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the documentation markup linter.

Every other script in ``tools/`` validates itself in use: it runs over the real tree and fails
loudly when something is wrong.  ``lint_docs.py`` is the opposite, and that is why it is the one
that gets tests.  Its success condition is silence, so a pattern that quietly stops matching is
indistinguishable from a tree with nothing wrong in it -- the failure mode would be a linter that
has passed every run since the day it broke.

The inputs below are therefore real: each "must flag" case is markup that actually shipped, and
each "must not flag" case is markup that is in the tree right now and is correct.
"""

import importlib.util
import pathlib

import pytest

_PATH = pathlib.Path(__file__).parents[2] / "tools" / "lint_docs.py"
_SPEC = importlib.util.spec_from_file_location("lint_docs", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
lint_docs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(lint_docs)


def _messages(text: str) -> list[str]:
    """Return the message of each finding, for a single chunk of markup."""
    return [finding.message for finding in lint_docs.check_text(text, "example.rst")]


class TestEmbeddedUri:
    """The cross-reference-shaped-as-an-external-link defect, which ``-W`` cannot see."""

    def test_wrapped_target_is_flagged(self):
        """The real defect from ``e61db631``, and the reason the pattern needs ``re.DOTALL``.

        The target was split across two lines, which ``docutils`` joins into a single relative
        URL.  A line-oriented scan sees nothing wrong here.
        """
        text = (
            "   using one of the provided iterator functions. See the section on `term iteration\n"
            "   <term_iteration_and_reconstruction>`_ in the operators guide for details.\n"
        )
        (message,) = _messages(text)
        assert "embedded URI" in message
        # The suggested rewrite has to name the label, since that is the whole fix.
        assert ":ref:`term iteration <term_iteration_and_reconstruction>`" in message

    def test_anonymous_form_is_flagged(self):
        """The real defect from ``c3101c4c``: the same bug with a doubled underscore."""
        text = "See `Transpiling fermionic circuits <transpilation>`__ for details.\n"
        (message,) = _messages(text)
        assert "embedded URI" in message

    def test_line_number_is_the_reference_not_the_file_start(self):
        text = "Intro.\n\nMore text.\n\nSee `a label <some_label>`_ here.\n"
        (finding,) = lint_docs.check_embedded_uri(text, "example.rst")
        assert finding.line == 5

    @pytest.mark.parametrize(
        "text",
        [
            # Real external links: the target is a URI, which is what the syntax is for.
            "`ffsim <https://qiskit-community.github.io/ffsim/>`_",
            "`arXiv:2512.11418v2 <https://arxiv.org/abs/2512.11418v2>`_",
            "the `Rust documentation <https://www.rust-lang.org/tools/install>`__ for",
            # The fixed form of the defect above, plus the cross-reference roles that carry the
            # same angle-bracket payload.  Without the trailing-underscore requirement every role
            # of this shape (the C API pages are full of them) would be a false positive.
            ":ref:`term iteration <term_iteration_and_reconstruction>`",
            ":ref:`operator term grouping <grouping_explanation>`",
            ":class:`.FermionicCircuit <qiskit_fermions.circuit.FermionicCircuit>`",
            ":external+cqiskit:doc:`QkObs <cdoc/qk-obs>`",
            ":c:func:`qf_ferm_op_jordan_wigner`",
            # A substitution reference with a hyperlink target.  This renders as an *internal*
            # reference and is correct; it is also the legitimate construct most likely to be
            # broken by a future tightening of the pattern, so it is pinned here deliberately.
            "- |term_iteration_and_reconstruction|_:\n",
            # Not a reference at all.
            "`a <b>`_foo",
        ],
    )
    def test_legitimate_markup_is_not_flagged(self, text):
        assert lint_docs.check_embedded_uri(text, "example.rst") == []


class TestAltText:
    """Image directives must offer a text alternative."""

    def test_figure_without_alt_is_flagged(self):
        """``.. figure::`` is the form ``2c298333`` moved to.

        This is the single most important case in the file: the validator the documentation
        repository ships for this check does not recognise ``.. figure::`` at all, so it reports
        success on exactly this input.  Matching it is the reason this script exists rather than
        depending on that one.
        """
        (message,) = _messages(".. figure:: images/overview.svg\n\n   A caption.\n")
        assert ".. figure::" in message and ":alt:" in message

    def test_image_without_alt_is_flagged(self):
        """The real defect from ``2c298333``, before the directive was changed to a figure."""
        (message,) = _messages(".. image:: images/overview.svg\n\nThe schematic description\n")
        assert ".. image::" in message

    def test_image_with_unrelated_options_is_flagged(self):
        (message,) = _messages(".. image:: x.svg\n   :width: 400\n   :align: center\n")
        assert ".. image::" in message

    def test_plot_without_alt_or_nofigs_is_flagged(self):
        (message,) = _messages(".. plot::\n   :context: close-figs\n\n   plot_it()\n")
        assert ".. plot::" in message
        # A plot has the extra, legitimate out of rendering no image at all, so say so.
        assert ":nofigs:" in message

    def test_directive_at_end_of_input_is_flagged(self):
        """A directive with nothing after it still has to be checked.

        The upstream validator decides a directive's fate on the *next* loop iteration, so one
        that ends the file is never evaluated and silently passes.
        """
        assert len(_messages(".. image:: a.png\n")) == 1
        assert len(_messages(".. image:: a.png")) == 1

    @pytest.mark.parametrize(
        "text",
        [
            # The real `docs/index.rst` figure, as corrected.
            ".. figure:: images/overview.svg\n   :alt: Schematic description of Qiskit Fermions\n",
            # A plot that renders no image has nothing to describe.
            ".. plot::\n   :nofigs:\n\n   circuit.draw()\n",
            # `:alt:` need not be the first option.
            ".. plot::\n   :context: close-figs\n   :alt: A circuit diagram\n\n   draw()\n",
            # A Rust doc comment, as in `crates/pyext/src/operators/edge_vertex_operator.rs`.
            "/// .. plot::\n///    :alt: An edge-vertex operator\n///\n///    draw()\n",
            # Indented inside another directive, as in `docs/guides/circuit.rst`.
            ".. hint::\n\n      .. plot::\n         :alt: A diagram\n\n         draw()\n",
            # Not a directive.
            "The .. image:: syntax places an image on the page.\n",
        ],
    )
    def test_compliant_markup_is_not_flagged(self, text):
        assert lint_docs.check_alt_text(text, "example.rst") == []


class TestDiscoverFiles:
    """Path handling.  Silence means success here, so finding nothing must never be quiet."""

    def test_missing_path_raises(self):
        with pytest.raises(FileNotFoundError):
            lint_docs.discover_files(["no/such/directory"], None)

    def test_single_file_is_accepted(self):
        assert lint_docs.discover_files([str(_PATH)], None) == [_PATH]

    def test_ignored_trees_are_excluded(self):
        root = _PATH.parents[1]
        assert lint_docs.discover_files([str(root / "tools")], [str(root / "tools")]) == []


def test_repository_is_clean():
    """The whole point: the trees that produce documentation have no findings.

    This is what turns the unit tests above into a guard rather than a demonstration -- it fails
    the moment someone adds a guide with an unresolvable cross-reference or an undescribed image.
    """
    root = _PATH.parents[1]
    files = lint_docs.discover_files(
        [str(root / name) for name in ("crates", "docs", "python")],
        [str(root / "crates" / "qiskit-pyo3-ffi")],
    )
    assert files, "discovered no files to check; the layout must have moved"
    findings = [
        str(finding)
        for file in files
        for finding in lint_docs.check_text(file.read_text(encoding="utf-8"), str(file))
    ]
    assert findings == []
