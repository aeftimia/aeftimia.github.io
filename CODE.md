# How I organize code

Check these conventions before adding or reorganizing code.

- **Package reusable code** as an installable library with `pyproject.toml`, `setup.py`, or an equivalent. One-off experiments and tasks belong in scripts under the project root’s `entrypoints/` and import the library.
- **Version the library** for downstream projects. Bump its semantic version in the installation file in the same commit as each library change. For any user-visible change (function, argument, default, format, or behaviour), bump the minor version before 1.0 and the major version thereafter. Use a patch bump for fixes that preserve observable behaviour. Changes confined to entrypoints, tests, or docs need no bump. A version bump carries no compatibility obligation.
- **Reuse and refactor.** Before adding a script or module, look for one to extend or factor. As the code grows, revisit its structure at feature completion or before publishing. Merge similar functions or modules without waiting to be asked.
- **Define each constant once.** Keep pins, revisions, prices, ports, thresholds, multiples, and limits at a single source. Code, scripts, and docs read from it. Where import is impractical, read a shared file or test that a required copy agrees. Instruction files such as `CLAUDE.md` point to the source.
- **Prefer standard libraries** over hand-written equivalents unless the calling code would exceed roughly 1.2 times the hand-written length. Only actively maintained libraries qualify. A commit within the past week or so is a good sign.
- **Parallelize independent work from the start** where practical, including scoring, file processing, tests, and network-bound setup. Prefer vectorized operations or batched requests. For other work, such as subprocesses or individual network calls, use a thread or process pool and guard shared state. Preserve input order and surface each failure.
- **Use lookup or dispatch tables** for known cases. Index a dict by normalized key so each case has one visible row and unknown keys raise. Keep case-specific data such as prices in that row. Avoid CLI defaults for those values and `.get(key, default)` for lookup.
- **Keep `.gitignore` current.** Ignore secrets, local settings, virtual environments, caches, build output, model weights, and setup logs before the first commit that could include them. Untrack any already in git.
- **Keep every resource referenced here in one of my GitHub repositories.** Point shared instructions to the canonical repository copy of each file, script, hook, or template.
- **Push and verify.** Work is complete when you confirm the remote branch contains the commits.

## Prefer clear prototype and research code

Assume my projects are short-lived prototypes or research code, usually maintained for only a few months. Favour clear, direct logic. An established, long-lived project may warrant a different tradeoff.

When an interface, configuration, or data format changes, update its current callers in the same change. Remove obsolete paths, signatures, aliases, shims, and dual formats. Skip speculative migrations. Keep compatibility only if I explicitly request it or a concrete, active external dependency requires it. Name any such dependency and keep the exception narrow.
