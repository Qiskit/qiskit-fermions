#!/usr/bin/env python3
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

"""Generate the landing page for the documentation-preview site.

The GitHub Pages site for this repository hosts *previews* rather than the canonical documentation,
which lives on the IBM Quantum Platform.  Several builds coexist there, each in its own directory:
``dev/`` for ``main``, ``stable/<X.Y>/`` for release branches, and ``pr/<N>/`` for pull requests that
carry the preview label.  This script writes the ``index.html`` that lists whatever is currently
present, so the root of the site is navigable instead of a 404.

The layout is two columns, echoing the Furo sidebar of the builds it links to.  The sidebar carries the
logo and the orientation text -- chiefly that these are *not* the released documentation, the one thing a
visitor who arrived here by mistake needs to know before reading anything else -- which leaves the main
column holding nothing but the listing itself.

That listing draws one distinction, between documentation that has been through review and documentation
that has not.  ``main`` and the ``stable/<X.Y>`` branches are listed together as protected branches, and
pull-request previews separately as work in progress.  That split is the reason a reader is here at all:
a preview is worth reading only once you know how provisional it is.

A pull-request entry is additionally labelled with that pull request's title, which this script cannot
know: it reads a directory listing.  The publishing action therefore drops the title into a
``.pr-title`` file inside the preview directory, and this script reads it if it is there.  That keeps
the rule below intact -- the *listing* still comes only from the filesystem -- and keeps the title
self-cleaning, since removing a preview deletes the directory holding it.

The listing is derived from the directories that actually exist on disk, never from a manifest.  That
matters because several jobs publish to the same branch concurrently: a job that regenerated this page
from its own idea of what should exist would drop a directory a competing job had just added.  Reading
the filesystem makes the page correct for whatever state the working tree is in when it runs -- which
is also why the publishing workflow must call this script *inside* its push-retry loop, after replaying
onto the updated branch tip, not once before it.

It also writes an empty ``.nojekyll``.  Pages runs Jekyll by default when serving from a branch, and
Jekyll ignores directories whose names begin with an underscore -- which would silently drop
``_static``, ``_images`` and friends, leaving every page rendered without CSS or images.

Usage:
    python tools/generate_docs_index.py gh-pages
"""

from __future__ import annotations

import argparse
import datetime
import html
import pathlib
import sys

#: Repository that owns the previews, used to link each pull-request entry back to GitHub.
REPO = "Qiskit/qiskit-fermions"

#: Label a pull request must carry for its preview to be built and published, and a link to it on
#: GitHub.  Kept in step with the `if:` guard in `.github/workflows/docs.yml` -- if that label is ever
#: renamed, this is the other place that names it.
PREVIEW_LABEL = "ci: preview docs"
PREVIEW_LABEL_URL = f"https://github.com/{REPO}/labels/ci%3A%20preview%20docs"

#: Where the *released* documentation lives.  This is the `addons` landing page rather than the
#: `api/qiskit-fermions` API reference: the sidebar points a reader at the documentation as a whole, and
#: the API reference is one page within it.
RELEASED_DOCS_URL = "https://quantum.cloud.ibm.com/docs/addons/qiskit-fermions"

#: Written alongside the index; see the module docstring for why this file is essential.
NOJEKYLL = ".nojekyll"

#: The Qiskit logo, borrowed from the development build's Sphinx assets rather than committed here.
#: Two files because the wordmark is hardcoded black in one and white in the other -- neither uses
#: ``currentColor``, so a single file cannot serve both themes; the page shows one and hides the other
#: exactly as the Furo sidebar does.  Depending on another directory's assets is safe only for ``dev/``:
#: ``main`` is published on every merge, so that directory outlives any individual release branch.
LOGO_LIGHT = "dev/_static/images/qiskit-light-logo.svg"
LOGO_DARK = "dev/_static/images/qiskit-dark-logo.svg"

#: Optional file inside a ``pr/<N>/`` directory holding that pull request's title, written by the
#: publishing action because only it knows the title -- this script sees a directory listing and nothing
#: else.  Living *inside* the preview directory is what keeps it self-cleaning: removing a preview is an
#: ``rm -rf pr/<N>``, which takes the title with it, so a stale title cannot outlive its build.  Absent
#: for a preview published before this file existed, which is why every read of it has a fallback.
PR_TITLE_FILE = ".pr-title"

#: Longest pull-request title rendered, in characters.  GitHub allows 256, which would wrap to several
#: lines in a button and push the "on GitHub" link off the row on a narrow screen.
PR_TITLE_MAX = 90

#: Order the sections appear in on the page.  `branches` must stay first: pull-request previews are by
#: far the most numerous, so any order that lets them come first buries the documentation that most
#: visitors are looking for.  Changing this reorders the page, so change it on purpose.
SECTION_ORDER = ("branches", "pr")

PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex">
<title>Qiskit Fermions documentation previews</title>
<style>
/* Two columns: a sidebar carrying the logo and the orientation text, and a main column carrying the
   listing.  Deliberately echoes the Furo layout of the builds this page links to -- same 67em collapse
   point and same 1rem/0.5rem item spacing -- so arriving here and then opening a build does not feel
   like crossing between two unrelated sites. */
:root {{
  color-scheme: light dark;
  --bg: #ffffff;
  --fg: #1a1a1a;
  --muted: #5a5a68;
  --accent: #6929c4;
  --border: #e0e0e0;
  --card: #f7f7f9;
  --sidebar-bg: #f7f7f9;
  --sidebar-width: 18rem;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg: #161616;
    --fg: #f4f4f4;
    --muted: #a8a8b3;
    --accent: #be95ff;
    --border: #393939;
    --card: #212121;
    --sidebar-bg: #1c1c1c;
  }}
}}
* {{ box-sizing: border-box; }}
html, body {{ min-height: 100%; }}
body {{
  margin: 0;
  background: var(--bg);
  color: var(--fg);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  line-height: 1.5;
  display: grid;
  grid-template-columns: var(--sidebar-width) minmax(0, 1fr);
  align-items: start;
}}
.sidebar {{
  position: sticky;
  /* Not `0`: the published page may sit behind a phone's status bar. */
  top: env(safe-area-inset-top, 0px);
  align-self: start;
  max-height: 100vh;
  overflow-y: auto;
  padding: 2rem 1rem;
  background: var(--sidebar-bg);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}}
.brand {{ display: flex; flex-direction: column; gap: .5rem; }}
/* Deliberately no `display` here: `.brand img` is (0,1,1) and would outrank the (0,1,0) class selectors
   below, pinning both logos visible.  Sizing only; the rules below own visibility. */
.brand img {{ width: 100%; max-width: 11rem; height: auto; }}
/* The two logo files differ only in the colour of the wordmark, so exactly one is ever shown.  This
   page has no theme switcher -- unlike the Sphinx builds, which is why they need a `data-theme`
   attribute and this does not -- so the OS preference alone decides, matching the palette above. */
.logo-light {{ display: block; }}
.logo-dark {{ display: none; }}
@media (prefers-color-scheme: dark) {{
  .logo-light {{ display: none; }}
  .logo-dark {{ display: block; }}
}}
.sidebar-text {{ font-size: .875rem; color: var(--muted); }}
.sidebar-text p {{ margin: 0; }}
.sidebar-text p + p {{ margin-top: .75rem; }}
.sidebar-text strong {{ color: var(--fg); }}
.sidebar-text a {{ color: var(--accent); }}
main {{
  min-width: 0;
  max-width: 46rem;
  padding: 2.5rem 2rem 3rem;
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
  /* Fill the viewport so the footer's `margin-top: auto` actually drops it to the bottom.  Without a
     height to push against, a short listing would leave the copyright floating under the last card. */
  min-height: 100vh;
}}
h1 {{ font-size: 1.75rem; margin: 0; font-weight: 600; text-wrap: balance; }}
section {{ display: flex; flex-direction: column; gap: .6rem; }}
h2 {{
  font-size: .8rem; text-transform: uppercase; letter-spacing: .07em;
  color: var(--muted); font-weight: 600;
  margin: 0; padding-bottom: .4rem; border-bottom: 1px solid var(--border);
}}
p.section-note {{ margin: 0; font-size: .875rem; color: var(--muted); }}
p.section-note a {{ color: var(--accent); }}
ul {{ list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: .5rem; }}
li {{ margin: 0; }}
a.entry {{
  display: flex; flex-wrap: wrap; gap: .25rem .75rem; align-items: baseline;
  padding: .7rem .9rem; border: 1px solid var(--border); border-radius: 6px;
  background: var(--card); color: var(--fg); text-decoration: none;
}}
a.entry:hover, a.entry:focus {{ border-color: var(--accent); outline: none; }}
a.entry .name {{ font-weight: 600; }}
a.entry .note {{ color: var(--muted); font-size: .875rem; }}
li.paired {{ display: flex; align-items: center; gap: .75rem; }}
li.paired a.entry {{ flex: 1; }}
a.aside-link {{ color: var(--accent); font-size: .875rem; white-space: nowrap; }}
footer {{
  margin-top: auto; padding-top: 1rem; border-top: 1px solid var(--border);
  font-size: .8125rem; color: var(--muted);
}}
footer p {{ margin: 0; }}
code {{
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: .875em;
}}
/* Furo's own collapse point, so the two layouts reflow at the same width.  The sidebar stops being a
   column and becomes a banner above the content: on a phone the orientation text still has to be read
   before the listing, which is exactly the order the source is already in. */
@media (max-width: 67em) {{
  body {{ grid-template-columns: minmax(0, 1fr); }}
  .sidebar {{
    position: static; max-height: none; overflow: visible;
    border-right: 0; border-bottom: 1px solid var(--border);
    padding: 1.5rem 1.25rem;
  }}
  .brand img {{ max-width: 9rem; }}
  main {{ padding: 2rem 1.25rem 3rem; min-height: 0; }}
}}
</style>
</head>
<body>
<div class="sidebar">
<div class="brand">{logo}</div>
<div class="sidebar-text">
<p><strong>This is not the released documentation.</strong> The documentation for released versions of
<code>{repo}</code> is published on the
<a href="{released_docs}">IBM Quantum Platform</a>.</p>
<p>This site hosts unreleased builds, so these pages may be outdated, incomplete, or describe features
that never ship.</p>
<p>Each button takes you to the documentation build for that branch or pull request, rebuilt each time
the ref changes.</p>
</div>
</div>
<main>
<h1>Qiskit Fermions documentation previews</h1>
{sections}
<footer>
<p>Copyright &#169; {year}, Qiskit addons team</p>
</footer>
</main>
</body>
</html>
"""


def _is_build(path: pathlib.Path) -> bool:
    """Report whether a directory holds a documentation build.

    Args:
        path: Candidate directory.

    Returns:
        ``True`` if it contains an ``index.html``, so half-copied or stale directories are skipped.
    """
    return (path / "index.html").is_file()


def _stable_sort_key(name: str) -> tuple[int, ...]:
    """Return a numeric sort key for a ``stable/<X.Y>`` directory name.

    Args:
        name: The version part of the directory name, e.g. ``"0.10"``.

    Returns:
        A tuple of integers, so ``0.10`` sorts after ``0.9`` rather than before it as a string would.
        Unparsable names sort last, under an empty tuple, rather than raising.
    """
    try:
        return tuple(int(part) for part in name.split("."))
    except ValueError:
        return ()


def _entry(href: str, name: str, note: str = "", aside: tuple[str, str] | None = None) -> str:
    """Render one list entry.

    Args:
        href: Link target, already a safe relative path.
        name: Primary label.
        note: Optional secondary label.
        aside: Optional ``(href, label)`` for a second link rendered beside the card.  Nesting it
            inside the card is not an option -- anchors cannot contain anchors -- so it sits next to
            the card, leaving the card itself a single click through to the documentation.

    Returns:
        An ``<li>`` element.  Every interpolated value is escaped: directory names come from the
        filesystem, so they are attacker-influenced in principle even though only committers can
        create them.
    """
    note_html = f'<span class="note">{html.escape(note)}</span>' if note else ""
    card = (
        f'<a class="entry" href="{html.escape(href, quote=True)}">'
        f'<span class="name">{html.escape(name)}</span>{note_html}</a>'
    )
    if aside is None:
        return f"<li>{card}</li>"
    aside_href, aside_label = aside
    return (
        f'<li class="paired">{card}'
        f'<a class="aside-link" href="{html.escape(aside_href, quote=True)}">'
        f"{html.escape(aside_label)}</a></li>"
    )


def _section(title: str, note: str, entries: list[str], *, escape_note: bool = True) -> str:
    """Wrap entries in a titled section, or return nothing when there are none.

    Args:
        title: Section heading.
        note: One line naming what the entries in this section link to, so a reader knows what they
            are about to open without having to infer it from the heading alone.
        entries: Rendered ``<li>`` elements.
        escape_note: Whether to escape ``note``.  Pass ``False`` only for a note built here in this
            module from already-escaped parts, such as one containing a link; never for a note carrying
            a value that came from the filesystem.

    Returns:
        The section markup, empty when ``entries`` is empty so unused sections do not appear.
    """
    if not entries:
        return ""
    joined = "\n".join(entries)
    note_html = html.escape(note) if escape_note else note
    return (
        f"<section>\n<h2>{html.escape(title)}</h2>\n"
        f'<p class="section-note">{note_html}</p>\n'
        f"<ul>\n{joined}\n</ul>\n</section>"
    )


def _logo(root: pathlib.Path) -> str:
    """Render the Qiskit logo, or nothing when its source build is missing.

    The asset belongs to the development build rather than to this page, so it is only linked when that
    build is actually on disk.  Guarding on the file keeps the page from showing a broken image in the
    one state where ``dev/`` is absent: the very first publication of a release branch to an empty
    site, before ``main`` has ever been deployed.

    Args:
        root: Root of the published site.

    Returns:
        Two ``<img>`` elements, one per theme, or an empty string.
    """
    if not (root / LOGO_LIGHT).is_file() or not (root / LOGO_DARK).is_file():
        return ""
    # `alt` on the first and `alt=""` plus `aria-hidden` on the second: they are one logo shown twice,
    # so announcing both would read the name out twice to a screen reader.
    return (
        f'<img class="logo-light" src="{html.escape(LOGO_LIGHT, quote=True)}" alt="Qiskit logo">'
        f'<img class="logo-dark" src="{html.escape(LOGO_DARK, quote=True)}" alt="" aria-hidden="true">'
    )


def _branches_section(root: pathlib.Path) -> str:
    """Render the section for the repository's protected branches, development build first.

    ``main`` and the ``stable/<X.Y>`` branches are listed together because they are the same kind of
    thing to a reader: branches that only a merged pull request can advance, so their documentation
    describes code that has passed review.  Splitting them left ``main`` alone in a section whose
    heading, note and entry all said "development" and nothing else -- there is only ever one
    development branch, so the grouping carried no information a single entry could not.

    Args:
        root: Root of the published site.

    Returns:
        The section markup, or an empty string when neither kind of build is present.
    """
    entries = []
    if _is_build(root / "dev"):
        entries.append(_entry("dev/", "main", "the development branch"))
    stable_root = root / "stable"
    if stable_root.is_dir():
        versions = sorted(
            (path.name for path in stable_root.iterdir() if path.is_dir() and _is_build(path)),
            key=_stable_sort_key,
            reverse=True,
        )
        # Deliberately "the X.Y release series" rather than anything naming a release as done: a stable
        # branch accumulates backported fixes after X.Y.0 ships, so what it documents is the *next*
        # patch release, not the one already out.
        entries.extend(
            _entry(f"stable/{name}/", f"stable/{name}", f"the {name} release series")
            for name in versions
        )
    return _section(
        "Official branches",
        "Protected branches, which only a merged pull request can advance.",
        entries,
    )


def _pr_title(pr_dir: pathlib.Path) -> str:
    """Read the stored title of the pull request previewed in ``pr_dir``.

    Args:
        pr_dir: A ``pr/<N>/`` directory.

    Returns:
        The title, collapsed onto one line and truncated to ``PR_TITLE_MAX``, or an empty string when
        the file is missing, unreadable or blank.  A missing title is normal rather than exceptional --
        previews published before the sidecar existed have none -- so the caller renders the entry
        without one instead of failing the publication over a cosmetic label.
    """
    try:
        raw = (pr_dir / PR_TITLE_FILE).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""
    # A title is a single line, but it arrives from an API and is written by a shell redirect, so treat
    # any internal whitespace as a separator rather than trusting it to be well formed.
    title = " ".join(raw.split())
    if len(title) > PR_TITLE_MAX:
        # Trim on a word boundary where there is one reasonably close to the limit, so the ellipsis does
        # not land mid-word.
        cut = title[: PR_TITLE_MAX - 1].rstrip()
        if (space := cut.rfind(" ")) > PR_TITLE_MAX * 0.6:
            cut = cut[:space]
        title = cut.rstrip(" ,;:-\u2014") + "\u2026"
    return title


def _pr_section(root: pathlib.Path) -> str:
    """Render the section for pull-request previews, lowest number first.

    Unlike the branch section, this one is rendered even when it is empty.  Its absence would otherwise
    be ambiguous: a reader cannot tell "no pull request is being previewed right now" apart from "this
    site does not preview pull requests at all", and the second reading is wrong.  The empty state says
    which label turns a preview on, so a contributor looking for their own pull request learns what to do
    rather than concluding the feature is broken.

    Args:
        root: Root of the published site.

    Returns:
        The section markup; never empty.
    """
    numbers: list[int] = []
    pr_root = root / "pr"
    if pr_root.is_dir():
        # Numeric sort, and skip anything non-numeric: the publishing workflow only ever creates integer
        # directories here, so a stray name means something is wrong and is better left out of the
        # listing than rendered as a broken link.
        numbers = sorted(
            int(path.name)
            for path in pr_root.iterdir()
            if path.is_dir() and path.name.isdigit() and _is_build(path)
        )
    label = (
        f'the <a href="{html.escape(PREVIEW_LABEL_URL, quote=True)}">'
        f"<code>{html.escape(PREVIEW_LABEL)}</code></a> label"
    )
    if not numbers:
        return (
            "<section>\n<h2>Pull requests</h2>\n"
            f'<p class="section-note">No pull request is being previewed right now. Applying {label} '
            "to a pull request builds its documentation and lists it here.</p>\n</section>"
        )
    return _section(
        "Pull requests",
        f"Work in progress, built from pull requests carrying {label}. These changes are unreviewed "
        "and may never merge.",
        [
            _entry(
                f"pr/{number}/",
                f"#{number}",
                _pr_title(pr_root / str(number)),
                aside=(f"https://github.com/{REPO}/pull/{number}", "on GitHub"),
            )
            for number in numbers
        ],
        escape_note=False,
    )


def build_index(root: pathlib.Path) -> str:
    """Render the landing page for the previews present under ``root``.

    Args:
        root: Root of the published site, i.e. a checkout of the ``gh-pages`` branch.

    Returns:
        The complete HTML document.
    """
    # Rendered in the order given by SECTION_ORDER, which is fixed deliberately rather than emerging
    # from the order these directories happen to be written in.  The branch section must come first:
    # pull-request previews are by far the most numerous, and letting them precede it would bury the
    # entries most visitors actually came for.
    built = {
        "branches": _branches_section(root),
        "pr": _pr_section(root),
    }
    # No "nothing is published" fallback: the pull-request section always renders, so the page is never
    # empty.  `main` is published on every merge, which makes a site with no branch build at all a
    # transient state during the very first deploy rather than something to write copy for.
    body = "\n".join(section for key in SECTION_ORDER if (section := built[key]))
    # The footer year tracks the build date, matching what Sphinx puts in the footer of the builds this
    # page links to.  Regenerating the page is the only way it ever changes, and this script runs on
    # every publish, so it cannot go stale the way a hardcoded year would.
    return PAGE_TEMPLATE.format(
        logo=_logo(root),
        repo=html.escape(REPO),
        released_docs=html.escape(RELEASED_DOCS_URL, quote=True),
        sections=body,
        year=datetime.date.today().year,
    )


def main() -> int:
    """Write ``index.html`` and ``.nojekyll`` into the requested directory.

    Returns:
        A process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "root",
        type=pathlib.Path,
        help="root of the published site, i.e. a checkout of the gh-pages branch",
    )
    args = parser.parse_args()

    if not args.root.is_dir():
        print(f"error: not a directory: {args.root}", file=sys.stderr)
        return 1

    (args.root / "index.html").write_text(build_index(args.root), encoding="utf-8")
    (args.root / NOJEKYLL).touch()
    print(f"Wrote {args.root / 'index.html'} and {args.root / NOJEKYLL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
