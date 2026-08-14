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
import html
import pathlib
import sys

#: Repository that owns the previews, used to link each pull-request entry back to GitHub.
REPO = "Qiskit/qiskit-fermions"

#: Written alongside the index; see the module docstring for why this file is essential.
NOJEKYLL = ".nojekyll"

PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex">
<title>Qiskit Fermions documentation previews</title>
<style>
:root {{
  color-scheme: light dark;
  --bg: #ffffff;
  --fg: #1a1a1a;
  --muted: #5a5a68;
  --accent: #6929c4;
  --border: #e0e0e0;
  --card: #f7f7f9;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg: #161616;
    --fg: #f4f4f4;
    --muted: #a8a8b3;
    --accent: #be95ff;
    --border: #393939;
    --card: #212121;
  }}
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  padding: 2.5rem 1.25rem 4rem;
  background: var(--bg);
  color: var(--fg);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  line-height: 1.5;
}}
main {{ max-width: 46rem; margin: 0 auto; }}
h1 {{ font-size: 1.75rem; margin: 0 0 .35rem; font-weight: 600; }}
p.lede {{ margin: 0 0 2rem; color: var(--muted); }}
h2 {{
  font-size: .8rem; text-transform: uppercase; letter-spacing: .07em;
  color: var(--muted); font-weight: 600;
  margin: 2rem 0 .6rem; padding-bottom: .4rem; border-bottom: 1px solid var(--border);
}}
ul {{ list-style: none; margin: 0; padding: 0; }}
li {{ margin: 0 0 .5rem; }}
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
.aside {{ margin-top: 2.5rem; font-size: .875rem; color: var(--muted); }}
.aside a {{ color: var(--accent); }}
code {{
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: .875em;
}}
</style>
</head>
<body>
<main>
<h1>Qiskit Fermions documentation previews</h1>
<p class="lede">Rendered documentation builds for branches and pull requests of
<code>{repo}</code>.</p>
{sections}
<p class="aside">These are previews and may be outdated or incomplete. The released documentation is
published on the
<a href="https://quantum.cloud.ibm.com/docs/api/qiskit-fermions">IBM Quantum Platform</a>.</p>
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
        Unparseable names sort last, under an empty tuple, rather than raising.
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


def _section(title: str, entries: list[str]) -> str:
    """Wrap entries in a titled section, or return nothing when there are none.

    Args:
        title: Section heading.
        entries: Rendered ``<li>`` elements.

    Returns:
        The section markup, empty when ``entries`` is empty so unused sections do not appear.
    """
    if not entries:
        return ""
    joined = "\n".join(entries)
    return f"<h2>{html.escape(title)}</h2>\n<ul>\n{joined}\n</ul>"


def build_index(root: pathlib.Path) -> str:
    """Render the landing page for the previews present under ``root``.

    Args:
        root: Root of the published site, i.e. a checkout of the ``gh-pages`` branch.

    Returns:
        The complete HTML document.
    """
    sections = []

    if _is_build(root / "dev"):
        sections.append(
            _section(
                "Development",
                [_entry("dev/", "main", "latest development build")],
            )
        )

    stable_root = root / "stable"
    if stable_root.is_dir():
        versions = sorted(
            (path.name for path in stable_root.iterdir() if path.is_dir() and _is_build(path)),
            key=_stable_sort_key,
            reverse=True,
        )
        sections.append(
            _section("Stable", [_entry(f"stable/{name}/", name) for name in versions]),
        )

    pr_root = root / "pr"
    if pr_root.is_dir():
        # Numeric sort, and skip anything non-numeric: the publishing workflow only ever creates
        # integer directories here, so a stray name means something is wrong and is better left out of
        # the listing than rendered as a broken link.
        numbers = sorted(
            int(path.name)
            for path in pr_root.iterdir()
            if path.is_dir() and path.name.isdigit() and _is_build(path)
        )
        sections.append(
            _section(
                "Pull requests",
                [
                    _entry(
                        f"pr/{number}/",
                        f"#{number}",
                        aside=(f"https://github.com/{REPO}/pull/{number}", "on GitHub"),
                    )
                    for number in numbers
                ],
            ),
        )

    body = "\n".join(section for section in sections if section)
    if not body:
        body = "<p>No documentation builds are currently published.</p>"
    return PAGE_TEMPLATE.format(repo=html.escape(REPO), sections=body)


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
