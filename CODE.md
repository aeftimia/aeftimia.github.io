# Notes on how I organize code

- Anything that can be a library should be one, with an installation script (`setup.py` or a modern equivalent such as `pyproject.toml`).
- The actual experiments and tasks are one-off scripts that import from that library. They live in `entrypoints/` at the project root. Nothing reusable belongs in a script.
- Don't repeat yourself. Before adding a script or module, look for one that already does most of the job and extend or factor it.
- Revisit the architecture as the code grows. When submodules or functions have become similar, combine them into shared code. Do this at natural checkpoints, such as when a feature is done or before publishing, not only when asked.

Before adding or reorganizing code, check it against these notes.

## Prefer clear prototype and research code

Treat my projects as prototypes or research code by default. We rarely maintain them for more than a few months, so backwards compatibility usually does more harm than good. Prefer clear, obvious logic over compatibility layers and the bloat they introduce.

When changing an interface, configuration, or data format, update the code and its current callers together. Remove obsolete paths instead of retaining old signatures, aliases, shims, dual formats, or speculative migration machinery. Do not preserve an old behavior merely because it existed. Keep compatibility only when I explicitly request it or a concrete, active external dependency requires it; identify that dependency and keep the exception narrow. An established, long-lived codebase may justify a different tradeoff, but do not assume that is the situation here.
