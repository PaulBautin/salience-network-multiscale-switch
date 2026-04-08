# `src/logging_utils`

### `setup_manuscript_logger`

```python
setup_manuscript_logger(
    script_name: str,
    project_root: Path,
    args: argparse.Namespace | None = None,
) -> logging.Logger
```

Configure logging to write to both console and `logs/<script_name>.log`.

Attaches a file handler to the root logger so that logging calls from any module are also captured. Each invocation appends a timestamped run header to the log file; the file is never overwritten.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `script_name` | `str` | Base name for the log file (no extension). |
| `project_root` | `Path` | Project root; `logs/` subdirectory is created if absent. |
| `args` | `argparse.Namespace` | Parsed CLI arguments written to the run header. Optional. |

**Returns** `logging.Logger` — named logger for the calling script.

**Example**

```python
logger = setup_manuscript_logger('figure_1a_t1map', project_root, args)
logger.info('Starting analysis')
```
