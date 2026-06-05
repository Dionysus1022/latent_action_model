# Environment Files

`requirements.txt` is the single dependency snapshot kept at the repository root.
It was generated from the active `.venv` with:

```bash
uv pip freeze --python .venv/bin/python
```

The active virtual environment does not provide `python -m pip`, so older
`requirements_frozen*.txt` files only contained a failed `pip freeze` message.
Those failed snapshots were moved to:

```text
artifacts/failed_requirements_freeze/
```

The generated `environment*.json` files were run-specific machine snapshots.
They were moved to:

```text
artifacts/environment_snapshots/
```

Those snapshots are useful for forensic debugging, but they should not be used
as install inputs. Use `requirements.txt` for reproducing the Python package set.

