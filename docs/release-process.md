# Release process

This describes how to cut a release of `qiskit-fermions`. It assumes push access to the
[upstream repository](https://github.com/Qiskit/qiskit-fermions) and that the `pypi` deployment
environment is configured for trusted publishing.

Each released minor version gets:

- a `X.Y.Z` tag, which is what actually triggers publication (`.github/workflows/release.yml`);
- a long-lived `stable/X.Y` branch, which receives backported fixes and publishes its own
  documentation to `stable/X.Y/` on the GitHub Pages site.

The tag and the branch are created from the **same commit**, and that is load-bearing — see
[Why the ordering matters](#why-the-ordering-matters).

## Cutting a new minor release

### 1. Prepare the release commit

On a branch off `main`:

```bash
make version-bump VERSION=X.Y.Z
```

This rewrites `python/qiskit_fermions/VERSION.txt` and the `[workspace.package]` version in
`Cargo.toml`, and refreshes `Cargo.lock`. **Stage `Cargo.lock` as well** — it is committed, and
`make lint` runs `cargo metadata --locked`, which fails if it disagrees with `Cargo.toml`.

Open a pull request and merge it. Note the resulting commit on `main`; every later step refers to it
as `$REL`.

### 2. Create the stable branch

```bash
git fetch upstream main
git branch stable/X.Y $REL
git push upstream stable/X.Y
```

Do this *before* pushing the tag. Pushing the branch is also what creates `stable/X.Y/` on the Pages
site: a tag push builds the documentation but deliberately does not publish it (see the ref-to-prefix
mapping in `.github/workflows/docs_deploy.yml`).

This also activates the backport tooling, which is dormant until a `stable/*` branch exists
(`.mergify.yml`, `.github/workflows/backport.yml`).

### 3. Push the tag

```bash
git tag X.Y.Z $REL
git push upstream X.Y.Z
```

Verify `$REL` first: this triggers `release.yml`, which builds the wheels and sdist, creates the
GitHub release, and **publishes to PyPI, which cannot be undone**.

At this point `X.Y.Z`, `main`, and `stable/X.Y` all point at the same commit.

### 4. Post-release housekeeping on `main`

In a follow-up pull request against `main`:

```bash
make version-bump VERSION=X.Y+1.0.dev0   # e.g. 0.2.0.dev0; stage Cargo.lock
```

For the *first* release only, also uncomment `earliest_version` in `releasenotes/config.yaml`. It
must be applied to `stable/X.Y` as well — see below.

Optionally move the release's notes into `releasenotes/notes/X.Y/` to keep the top level tidy. This
is cosmetic; `reno` recurses into subdirectories and the path never changes which release a note
belongs to.

## Backporting a fix

Label the pull request against `main` with `stable backport potential`. Mergify opens the backport
pull request against `stable/X.Y`, and `backport.yml` copies the labels and milestone across.

A backported fix **must carry its release note on the stable branch**. `reno` files a note under the
first release tag reachable from the commit that added the file, so a note that exists only on `main`
is filed under the next minor rather than the patch release.

To publish a patch release, bump the version on `stable/X.Y` and push a `X.Y.Z+1` tag from that
branch. Steps 2 and 4 do not apply — the branch already exists, and `main`'s version is untouched.

## Why the ordering matters

Two constraints are easy to get wrong and neither fails loudly at the point of the mistake.

**`earliest_version` requires a reachable tag.** `releasenotes/config.yaml` sets
`earliest_version`, and `reno` resolves it against the history of whatever branch is being built. If
the tag is not an ancestor of that branch, the documentation build fails outright with
`ValueError: earliest-version set to unknown revision`. Because the tag and `stable/X.Y` are both
created from `$REL`, the tag is reachable from `main` *and* from the stable branch, and the setting
works on both. Enabling it before the tag exists breaks every documentation build, which is why it
belongs in the post-release pull request rather than the release one.

**A publishable artifact must be built from the tag.** `python/qiskit_fermions/version.py` appends a
`+<sha>` local version identifier whenever `HEAD` carries no tag. PyPI rejects local version
identifiers, so never build and upload an artifact from the release branch before tagging.
`release.yml` checks out the tag, so the automated pipeline is unaffected.

## Verifying a release

- Wheel and sdist jobs green, and the GitHub release is marked as a pre-release only if the tag
  carries a pre-release suffix.
- `pip install qiskit-fermions==X.Y.Z` reports exactly `X.Y.Z` — no `+<sha>` suffix.
- Documentation builds pass on both `main` and `stable/X.Y`, the release notes page is headed
  `X.Y.Z`, and `stable/X.Y/` appears on the Pages index.

A locally built release notes page headed `0.0.0` just means the checkout has no tags; fetch them
(`git fetch --tags`) and rebuild.

<!-- vim: set tw=100: -->
