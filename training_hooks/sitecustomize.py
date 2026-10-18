"""Mirror Octo's W&B metrics to a plain JSONL file for course submission."""

from __future__ import annotations

import json
import os
from pathlib import Path


def _json_value(value):
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


log_path = os.environ.get("OCTO_JSONL_LOG")
if log_path:
    import wandb

    _original_log = wandb.log

    def _log(data, *args, **kwargs):
        step = kwargs.get("step")
        record = {
            "step": step,
            "metrics": {key: _json_value(value) for key, value in data.items()},
        }
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return _original_log(data, *args, **kwargs)

    wandb.log = _log
