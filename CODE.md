# Notes on how I organize code

- Anything that can be a library should be one, with an installation script (`setup.py` or a modern equivalent such as `pyproject.toml`).
- The actual experiments and tasks are one-off scripts that import from that library. They live in `entrypoints/` at the project root. Nothing reusable belongs in a script.
- Don't repeat yourself. Before adding a script or module, look for one that already does most of the job and extend or factor it.
- Revisit the architecture as the code grows. When submodules or functions have become similar, combine them into shared code. Do this at natural checkpoints, such as when a feature is done or before publishing, not only when asked.

Before adding or reorganizing code, check it against these notes.
