# Typechecker Integration

## Mypy Plugin

### Scratch notes

- for zarr you have to do `--follow-untyped-imports` because it doesn't say it's typed.
- enable the pydantic mypy plugin and set `init_typed = true`