# Contribution Process

## Issues

Bugs, feature requests, and questions go in the
[issue tracker](https://github.com/p2p-ld/numpydantic/issues).

For bugs, please include a minimal example that reproduces the problem,
the versions of numpydantic, pydantic, numpy, and python you are using,
and any array libraries involved (zarr, dask, h5py, etc.).

## Development Environment

numpydantic uses [pdm](https://pdm-project.org) to manage dependencies.

1. [Fork the repository](https://github.com/p2p-ld/numpydantic/fork) on GitHub
2. Clone your fork and make a branch for your changes:

   ```bash
   git clone https://github.com/<your-username>/numpydantic
   cd numpydantic
   git checkout -b my-branch
   ```

3. Install the package and all of its development dependencies:

   ```bash
   pdm install --with :all
   ```

   You can also install only the groups you need with `--with tests` or `--with docs`.

If you add or change a dependency in `pyproject.toml`, update the lockfile with

```bash
pdm run lock
```

CI checks that the lockfile is current.

## Testing

Tests are in `tests` and run with pytest:

```bash
pdm run test
```

To run a subset of the tests, use pytest's `-m` and `-k` options.
The available markers are listed under `[tool.pytest.ini_options]` in `pyproject.toml`.
For example, to run only the zarr tests:

```bash
pdm run pytest -m zarr
```

New features must come with tests.
Bugfixes should come with a test that fails without the fix:
the preferred form for this is to open a draft PR with only the tests added,
so that a CI run with the correctly failing test is visible to the maintainers,
and then the fix for the bug should follow (without changing the test).

See [Testing](./testing.md) for the helpers we use to test combinations of
dtypes, shapes, and interfaces.

## Linting

We use [ruff](https://docs.astral.sh/ruff/) for linting and
[black](https://black.readthedocs.io) for formatting.
To check your code:

```bash
pdm run lint
```

To fix what can be fixed automatically:

```bash
pdm run format
```

CI also runs [codespell](https://github.com/codespell-project/codespell) to check spelling.

## Docs

The docs are built with sphinx. To build them and serve them locally,
rebuilding when files change:

```bash
pdm run docs
```

To build them the way CI does, where warnings are errors:

```bash
pdm run docs-prod
```

## Pull Requests

Please open an issue before opening a pull request,
unless the pull request fixes an unambiguous bug.
That gives us a chance to talk about whether and how a change should be made
before you spend time on it -
we don't want to waste your time,
and most problems with the shape of PRs can be fixed by just talking about them first!

Pull requests should target the `main` branch,
and tests, linting, and docs builds must pass in CI before they are merged.
