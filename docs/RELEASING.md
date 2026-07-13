# Release procedure

This document defines where release documentation lives, how each document
evolves across releases, the step-by-step procedure for cutting a release,
and when each document must be updated. The audience is maintainers. This
document is evergreen: it describes the process for every release and
should never contain information tied to a single version.

## Branch model

Two long-lived branches with distinct roles. The roles are defined by
purpose, not by who may push:

| | `dev/main` (integration) | `main` (release) |
|---|---|---|
| Purpose | where development happens | promoted snapshot that is always releasable |
| Who writes | lead maintainer directly; all other contributors via PRs targeting `dev/main` | nobody directly — advances only by merging `dev/main` |
| Stability | CI green at all times; API may churn between releases | matches the latest release plus any pending release prep |
| Changelog / migration entries | written here, with the change (see below) | arrive only via the `dev/main` merge |
| Docs site | not deployed | deployed on every push (`docs.yml`) |
| Release tags (`vX.Y.Z`) | never | always — the tag push triggers the PyPI publish (`publish.yml`) |

"Merged" in this document always means **merged into `dev/main`** unless
`main` is named explicitly. The direct-commit privilege on `dev/main` does
not blur the two roles; the line is crossed only if work is committed
directly to `main` or a release is tagged on `dev/main` — neither is ever
done.

### Enforcement

The branch model is convention unless GitHub enforces it. Four mechanisms
back it, each needing a one-time setup in the repository settings:

1. `.github/workflows/branch-flow.yml` fails any PR into `main` whose
   source is not this repository's `dev/main` (forks included). A failing
   check only *blocks* merging if it is marked **required** — in
   Settings → Branches → protection rule for `main`, enable "Require
   status checks to pass" and add the `Branch flow / enforce` check.
2. The same protection rule for `main` should enable "Require a pull
   request before merging", which disables direct pushes to `main`. Note
   this makes the fast-forward push in step 6 below impossible — with
   protection on, always use the `dev/main` → `main` PR path. (Repository
   admins can bypass unless "Do not allow bypassing" is also set.)

3. Tags are not covered by branch protection at all — anyone with write
   access can push a `vX.Y.Z` tag from any commit and trigger the PyPI
   publish. To close this, add a tag ruleset (Settings → Rules → Rulesets)
   restricting creation of `v*` tags to maintainers, and/or add required
   reviewers to the `pypi-production` environment so the final publish
   step needs manual approval regardless of who pushed the tag.
4. `dev/main` gets its own rule implementing the "who writes" row above:
   require a pull request before merging, with the lead maintainer on the
   bypass list (in a ruleset, add them as a bypass actor; in classic
   branch protection, leave "Do not allow bypassing" unchecked so repo
   admins can push directly), and "Require status checks to pass" with
   the `Changelog / check` check added. Result: the maintainer commits
   straight to `dev/main`; every other contributor goes through a PR that
   cannot merge without a changelog entry (or a maintainer-applied
   `no-changelog` label — see "CI guard" below).

Without these settings, anyone with write access can push or merge any
branch straight to `main` (or tag a release); the workflow alone merely
reports a failure.

## Where release documentation lives, and how it grows

| Document | Location | Lifecycle | Canonical? |
|---|---|---|---|
| `CHANGELOG.md` | repo root | **one file forever, append-only** — a new `## [X.Y.Z]` section per release, newest first | yes — single source of truth |
| Migration guides | `docs/migrations/vX-to-vY.md` | **one new file per major release** (e.g. `v1-to-v2.md`); minors/patches never get one | yes |
| `docs/RELEASING.md` | `docs/` | one file forever, evergreen — edited only when the process changes | yes (for process) |
| GitHub Release notes | github.com → Releases | one per release, created from the tag | no — paste from CHANGELOG, link the migration guide |
| Docs site (Quarto) | `packages/pyapprox-tutorials` → GitHub Pages | rebuilt on push to `main` | no — may link to the canonical documents |
| PyPI | project page | renders the package README | no — README links to CHANGELOG/migrations on GitHub |

The split keeps the repo root at a fixed size: `CHANGELOG.md` is the only
release document at the root (tooling and users expect it there), it grows
internally rather than multiplying, and everything else lives under
`docs/` — including migration guides, the only documents that gain a new
file per (major) release. Never create per-version changelog files
(`CHANGELOG-2.0.md`, `NOTES-v2.1.md`, ...); a version's notes are a section
inside `CHANGELOG.md`, and everything else (GitHub Releases, announcements)
is copied or linked from there.

Rationale for repo-tracked markdown as the source of truth: it is
version-controlled with the code, reviewable in PRs, discoverable on
GitHub, and survives doc-toolchain changes (this project has already moved
Sphinx → Quarto). GitHub Releases and PyPI are distribution surfaces, not
sources of truth.

## When notes get written (the important part)

**Changelog entries are written when a change lands on `dev/main`, not
when a release is cut.** Waiting until the tag is being pushed is how
releases ship with no notes: by then, reconstructing months of changes is
a large archaeology job. The rules:

1. **Every change to `dev/main` that is user-facing** (behavior, API,
   dependencies, supported versions, install) **adds a line to the
   `[Unreleased]` section of `CHANGELOG.md` in the same PR or commit**,
   under Added / Changed / Deprecated / Removed / Fixed. Internal
   refactors, CI, and test-only changes are exempt.
2. **Every breaking change updates the migration guide for the upcoming
   major in the same PR or commit that lands it on `dev/main`.** The first
   breaking change after a major release creates the next guide (e.g.
   `docs/migrations/v2-to-v3.md`); later breaking changes append to it.
   The author of the break is the person who knows the old→new mapping
   best, and it is cheapest to write while the diff is open.
3. **At release time you only finalize on `dev/main`**: rename
   `[Unreleased]` to the version, add the date, refresh links — then merge
   `dev/main` into `main` and tag. If you find yourself *writing* notes at
   this stage rather than *retitling* them, the process failed upstream.

The publish workflow enforces (1) at tag time — see "CI guard" below.

### Entry style: how much to write

A changelog entry is **one or two lines per user-visible change**, written
for a *user* deciding whether to upgrade — not for the committer. The test
for length: after reading the entry, the user should know (a) whether it
affects them and (b) what to do about it, without opening the diff. The
test for brevity: if the entry would be equally true of ten other commits
("Fixed bug", "Improved performance", "Updated GP code"), it says nothing.

Rules:

- Name the affected symbol or module in backticks, and for renames or
  removals always give old → new: users search the changelog for the name
  that just broke their import.
- State the user-visible effect, not the implementation. "How" belongs in
  the PR and the code; "what changed for me" belongs here.
- Include the reason only when it changes what the user should do
  (e.g. "deprecated in favor of X — use X for new code").
- Breaking changes: the changelog line says *what* broke; the *how to
  migrate* goes in the migration guide (rule 2 above). Do not inline
  migration instructions longer than one sentence.
- Do not paste commit messages or PR titles verbatim, and do not log
  internal file moves, refactors, or test changes at all.

Calibration examples:

| | Example |
|---|---|
| Too short | `Fixed sparse grid bug.` |
| Too long | `Fixed a bug in the adaptive sparse grid fitter where the priority queue used in the greedy subspace selection loop compared error indicators without normalizing by subspace cost, because the refactor in #412 moved the cost computation after the heap push, which meant that for anisotropic problems with strongly varying evaluation costs the fitter could...` |
| Right | `Fixed \`MultiFidelityAdaptiveSparseGridFitter\` selecting subspaces without cost-normalizing error indicators; anisotropic multi-fidelity fits now converge as documented.` |
| Too short | `Renamed some marginals.` |
| Right | `Renamed \`LognormalMarginal\` → \`LogNormalMarginal\` (breaking; see migration guide).` |

The same calibration applies to migration-guide sections (old → new import
plus a minimal before/after snippet — not an essay) and to GitHub Release
notes (a highlights paragraph plus the pasted changelog section — not a
rewrite).

## Release checklist

Versioning is semver, and the version comes from the git tag via
`hatch-vcs` — there is no version string to bump in any file. The tag push
(on `main`) is the release trigger: `publish.yml` builds all three
packages, verifies the wheels install and import in fresh venvs on Ubuntu
and macOS, uploads to TestPyPI, then publishes to PyPI via trusted
publishing.

Finalize on `dev/main` (a "release prep" commit or PR targeting
`dev/main`):

1. Decide the version. Breaking API change → major; new features → minor;
   fixes only → patch.
2. In `CHANGELOG.md`: rename `[Unreleased]` to `[X.Y.Z] - YYYY-MM-DD`,
   create a fresh empty `[Unreleased]` above it, update the compare links
   at the bottom.
3. If major: confirm `docs/migrations/v(X-1)-to-vX.md` covers every
   breaking change listed in the new changelog section (entries were
   written per-PR; this is a review, not a writing session), and link it
   from the README.
4. Sanity-check `README.md` install/version claims are still true.
5. Push to `dev/main`; wait for CI (tests, lint, architecture) to go
   green there.

Merge to `main`, tag, and publish:

6. Merge `dev/main` into `main`. With branch protection on `main` (see
   "Enforcement" above) this must be a PR:
   ```bash
   gh pr create --base main --head dev/main --title "Release vX.Y.Z"
   ```
   (Only without protection is a direct fast-forward push possible:
   `git checkout main && git merge --ff-only dev/main && git push`.)
7. Wait for CI on `main` to go green (this push also rebuilds and deploys
   the docs site).
8. Tag the merge commit on `main` and push the tag:
   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```
9. Watch the `Publish to PyPI` workflow. It rejects dirty versions
   (`+local` segment) and — with the CI guard below — tags whose version
   has no `CHANGELOG.md` section.

Post-publish:

10. Create the GitHub Release from the tag:
    ```bash
    gh release create vX.Y.Z --title "PyApprox X.Y.Z" \
        --notes-file <(sed -n '/^## \[X.Y.Z\]/,/^## \[/p' CHANGELOG.md | sed '$d')
    ```
    (or paste the changelog section in the web UI). For major releases,
    the first line links to the migration guide.
11. Verify: `pip install pyapprox==X.Y.Z` in a fresh venv; check the docs
    site deployed; check the PyPI page renders.

## CI guard

This step lives in the `build` job of `.github/workflows/publish.yml`,
after checkout and before the build steps. It fails the release if the
tagged version has no changelog section, converting rule (1) above from
convention into a hard gate at the moment it matters:

```yaml
      - name: Require changelog entry for this version
        if: startsWith(github.ref, 'refs/tags/v')
        run: |
          version="${GITHUB_REF_NAME#v}"
          if ! grep -qE "^## \[${version}\]" CHANGELOG.md; then
            echo "::error::CHANGELOG.md has no '## [${version}]' section." \
                 "Finalize the changelog before tagging (see docs/RELEASING.md)."
            exit 1
          fi
```

A companion step in the same job enforces the branch model at tag time:
it fails the publish if the tagged commit is not reachable from `main`
(`git merge-base --is-ancestor "$GITHUB_SHA" origin/main`). Together with
the tag ruleset (Enforcement item 3) this closes the "tag pushed from
dev/main or a feature branch" hole: the ruleset limits *who* may tag,
this guard checks *where* the tag points.

Second guard, at review time: `.github/workflows/changelog.yml` fails any
PR into `dev/**` or `main` that touches `packages/*/src/**` without
touching `CHANGELOG.md`. It is marked **required** on the `dev/main` rule
(Enforcement item 4), so contributor PRs cannot merge without an entry.
Operational notes:

- It is a heuristic: genuinely internal changes (refactors, typing,
  docstrings) will be flagged. The escape hatch is the `no-changelog`
  label. To be clear about what the label does: it does **not** stop
  anyone from adding changelog entries — contributors can and usually
  must add a line to `[Unreleased]`, and that is the normal way to make
  the check pass. The label is a **per-PR waiver**: when a PR is
  genuinely internal (no user-visible effect), forcing an entry would
  produce exactly the noise the entry-style rules prohibit ("internal
  cleanup"), so a maintainer reviews the PR, agrees it needs no entry,
  and applies the label to tell the check to stand down for that one PR.
  Fork contributors cannot label their own PRs, so granting the waiver is
  always a maintainer action during review — which is the point: the
  exemption is signed off, never self-granted.
- The label test is at the job level, not a workflow path filter, on
  purpose: a labeled PR yields a "skipped" job, which GitHub counts as
  passing, whereas a required check whose workflow never runs would block
  the PR forever.
- It only sees PRs. The lead maintainer's direct pushes to `dev/main`
  bypass it; the tag-time guard above is the backstop for those.
