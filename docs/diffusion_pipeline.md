# Diffusion Planner Pipeline

## One-command training

```bash
.venv/bin/python scripts/train_diffusion_head.py task=cube
```

The task config owns the default raw HDF5 path, optional split paths, output
paths, and LeWM policy path. By default the diffusion pipeline reads the raw
HDF5 directly and skips dataset splitting:

```yaml
pipeline:
  use_raw_dataset: true
```

For cube, the default policy is:

```text
/data/ykz/cube/lewm_epoch_27
```

To override the checkpoint for one run:

```bash
.venv/bin/python scripts/train_diffusion_head.py task=cube task.wm_policy=/data/ykz/cube/lewm_epoch_20
```

To only inspect the resolved pipeline without training:

```bash
.venv/bin/python scripts/train_diffusion_head.py task=cube pipeline.device=cpu pipeline.dry_run=true
```

To restore the old split-first behavior for one run:

```bash
.venv/bin/python scripts/train_diffusion_head.py task=cube pipeline.use_raw_dataset=false
```
