
# Contributing
If you would like to contribute to SynBPS, you are welcome to submit your suggestions, bug reports, or pull requests. Follow the guidelines below to ensure smooth collaboration:

- Before submitting a new feature request or bug report, please check the existing issues to avoid duplicates.
- If you have a new feature idea, open an issue to discuss it with the maintainers and get feedback.
- For bug reports, provide a clear and concise description of the issue, including steps to reproduce it.
- If your contribution requires documentation changes, please update the documentation accordingly.
- Be respectful and considerate towards others in your interactions on the project.

## Development setup and local verification
Before opening a pull request, verify the change locally. The same checks run automatically on every pull request in GitHub Actions (see `.github/workflows/ci.yml`).

Create a virtual environment and install the package in editable mode together with the test tools:

    python -m venv .venv
    source .venv/bin/activate      # Windows: .venv\Scripts\activate
    pip install -e . pytest

Run the test suite:

    pytest -v tests/

To run the tests against every supported Python version installed on your machine, and to build and validate the distribution files, use tox:

    pip install tox
    tox

Individual environments can be selected, for example `tox -e py311` or `tox -e build`.

## Releasing a new version to PyPI
Merging to `main` does **not** publish anything. A release is triggered manually:

1. Bump `version` in `pyproject.toml` and update the "Whats new" section in `README.md` in a pull request, and merge it.
2. On GitHub, open *Releases*, choose *Draft a new release*, create a tag named `v<version>` (for example `v1.2.0`) on `main`, and publish the release.
3. The *Publish package to PyPI* workflow (`.github/workflows/publish.yml`) checks that the tag matches the version in `pyproject.toml`, runs the tests, builds the package and uploads it to PyPI.

If the tag and the package version differ, the workflow stops before anything is uploaded.
