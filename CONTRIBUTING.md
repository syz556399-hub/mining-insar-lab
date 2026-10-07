# Contributing

Create a Python 3.11+ virtual environment and install `requirements-dev.txt`.
Run `ruff check .`, `ruff format --check .`, and `python -m unittest discover -v` before submitting a change.

For scientific changes, explain the equation, units, coordinate/sign convention, assumptions and source. Add an analytical limit, invariance or statistical property test; report its numerical tolerance. Do not present visual similarity or passing numerical tests as evidence of real-world predictive accuracy.

Changing a stochastic algorithm or physical interpretation requires an engine/version change, a CHANGELOG entry and a reproducibility note. Never rewrite old exported parameters to make a new engine appear to reproduce them.

Use synthetic examples for issues and pull requests. Real imagery requires its own provenance and redistribution terms. Please provide a minimal parameter JSON, dependency versions and the expected/observed behavior for a bug report.
