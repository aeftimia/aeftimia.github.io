# Notes on how I organize code

- Anything that can be a library should be one, with an installation script (`setup.py` or a modern equivalent such as `pyproject.toml`).
- The actual experiments and tasks are one-off scripts that import from that library. They live in `entrypoints/` at the project root. Nothing reusable belongs in a script.
- Give the library a semantic version and keep it current, since other projects may install it. Bump the version in the installation script in the same commit as any change to the library itself: the minor number (the major number from 1.0 on) for anything a user of the library could notice, such as a changed function, argument, default, file format or behaviour, and the patch number for fixes that leave all of that as it was. Changes only to `entrypoints/`, tests or docs need no bump. A bump records a change; it is not a reason to keep the old behaviour (see below).
- Don't repeat yourself. Before adding a script or module, look for one that already does most of the job and extend or factor it.
- State every constant once. A version pin, model revision, price, port, threshold, multiple or limit lives in one place, and code, scripts and docs read it from there instead of restating it. Where another language or a document cannot import it (a shell script, a README), have it read a shared file or add a test that fails when its copy disagrees. Instruction files such as CLAUDE.md point to the source rather than repeating the number.
- Prefer a standard library to hand-written code for the same job, unless using it would make the code that uses it more than about 1.2 times as long. Count only actively maintained libraries as standard: a commit within the last week or so is a good sign.
- Revisit the architecture as the code grows. When submodules or functions have become similar, combine them into shared code. Do this at natural checkpoints, such as when a feature is done or before publishing, not only when asked.
- Prefer lookup and dispatch tables to branching. When code picks a value or a handler from a set of known cases (hosts, models, modes, kinds), write a dict keyed on the normalized case and index it, so every known case is one visible row and an unlisted one raises. Data that belongs to a case, such as a price, goes in its row, not in a command-line default. Don't read the table with `.get(key, default)`.
- Keep `.gitignore` current. Secrets and local settings, virtual environments, caches, build output, model weights and setup logs don't belong in git. Add the rule before the first commit that could pick such a file up, and untrack any that slipped in.

Before adding or reorganizing code, check it against these notes.

## Prefer clear prototype and research code

Treat my projects as prototypes or research code by default. We rarely maintain them for more than a few months, so backwards compatibility usually does more harm than good. Prefer clear, obvious logic over compatibility layers and the bloat they introduce.

When changing an interface, configuration, or data format, update the code and its current callers together. Remove obsolete paths instead of retaining old signatures, aliases, shims, dual formats, or speculative migration machinery. Do not preserve an old behavior merely because it existed. Keep compatibility only when I explicitly request it or a concrete, active external dependency requires it; identify that dependency and keep the exception narrow. An established, long-lived codebase may justify a different tradeoff, but do not assume that is the situation here.
