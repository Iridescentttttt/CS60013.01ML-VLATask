# Octo-Small 1.5 on SimplerEnv (WidowX)

This repository loads a local Octo-Small 1.5 checkpoint and evaluates it in
SimplerEnv's WidowX visual-matching environment.

## Cloud layout

```text
/root/SimplerEnv
/root/octo
/root/models/octo-small-1.5
/root/vlalora
```

## Run one episode

```bash
cd /root/vlalora
bash scripts/run_cloud_once.sh
```

The default task is `widowx_spoon_on_towel`. Results are saved under
`/root/vlalora/results/` as JSONL, summary JSON, and MP4 video files.

## Run the two course tasks

```bash
python scripts/evaluate_octo_simpler.py \
  --checkpoint /root/models/octo-small-1.5 \
  --checkpoint-step 300000 \
  --task widowx_spoon_on_towel \
  --episodes 10 \
  --output-dir results/base_spoon

python scripts/evaluate_octo_simpler.py \
  --checkpoint /root/models/octo-small-1.5 \
  --checkpoint-step 300000 \
  --task widowx_put_eggplant_in_basket \
  --episodes 10 \
  --output-dir results/base_eggplant
```

The wrapper uses SimplerEnv's `OctoInference` adapter. For WidowX it selects
the `bridge_dataset` action statistics, converts Euler rotation deltas to
axis-angle commands, and maps the gripper output to the environment convention.
