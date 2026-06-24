# Datamix synthetic pilot run (~500k tokens)

Short end-to-end test of the pipeline on SLURM before the full 1B run. All artifacts live under `pilot/` and do not overwrite production paths.

## Scale

| Setting | Pilot | Production |
|---------|-------|------------|
| Target tokens | 500k | 1B |
| Prompts | ~600 | ~1.2M |
| Languages | 4 (de, fr, pl, es) | 36 |
| GPU shards | 1 node × 1 GPU (`alpha`, `gpu:1`, `64G`) | 8+ nodes |
| Generator | Datamix 9B | Datamix 9B |

## Files to review

| File | Purpose |
|------|---------|
| `mix_config_pilot.yaml` | Token budget + domain mix |
| `config_datamix_pilot.yaml` | SLURM + vLLM (matches `launch_search.sh`: `alpha`, `gpu:1`, `64G`, `requeue`) |
| `slurm/01_prepare_pilot.sbatch` | CPU: build dataset |
| `slurm/02_validate_create_run.sbatch` | CPU: validate + write `ih_job.slurm` |
| `submit_inference.sh` | Login node: submit GPU job (with confirmation prompt) |
| `slurm/04_postprocess_pilot.sbatch` | CPU: join outputs + token counts |
| `run_pilot_pipeline.sh` | Manual step runner (same stages) |

## Submission order (you submit — not automated here)

```bash
cd /data/horse/ws/hama901h-whittle/inference-hive/examples/datamix_1b_synthetic/pilot
mkdir -p logs

# 1) Build prompts (CPU)
sbatch slurm/01_prepare_pilot.sbatch
# note JOBID from squeue / sacct

# 2) Validate + create inference-hive run (CPU)
sbatch --dependency=afterok:<JOBID> slurm/02_validate_create_run.sbatch

# 3) REVIEW generated SLURM script
less runs/datamix_pilot/ih_job.slurm
less runs/datamix_pilot/ih_config.yaml

# 4) Submit GPU inference (login node, asks for confirmation)
./submit_inference.sh

# 5) Monitor
./run_pilot_pipeline.sh status
# logs: runs/datamix_pilot/logs/

# 6) After inference completes — postprocess (CPU)
sbatch slurm/04_postprocess_pilot.sbatch
```

## Review checklist before `./submit_inference.sh`

- [ ] `config_datamix_pilot.yaml` matches cluster expectations (`alpha`, no account — same as `launch_search.sh`)
- [ ] Datamix model in HF cache snapshot (see `config_datamix_pilot.yaml` `--model` path)
- [ ] `runs/datamix_pilot/ih_job.slurm` loads CUDA module and requests `gpu:1`, `mem=64G`
- [ ] `dataset/datamix-synth-completion` exists and row count ~600
- [ ] `HF_HUB_OFFLINE=1` is OK (model is local)

## SLURM reference

Pilot GPU/CPU jobs follow [`whittle-paper/launch_search.sh`](../../../../whittle-paper/launch_search.sh):

| Parameter | Value |
|-----------|-------|
| Partition | `alpha` |
| GPUs | `gpu:1` (inference job only) |
| CPUs | `cpus-per-task=6` |
| Memory | `64G` |
| Walltime (inference) | `2-00:00:00` |
| Requeue | `--requeue` |
| Logs | `pilot/logs/%x_%j.{out,err}` |

## Manual alternative (no sbatch for steps 1–2)

```bash
./run_pilot_pipeline.sh plan
./run_pilot_pipeline.sh prepare
./run_pilot_pipeline.sh validate
./run_pilot_pipeline.sh create-run
# review, then:
./submit_inference.sh
```

## Outputs

```
pilot/
├── prompts/prompts.jsonl
├── dataset/datamix-synth-completion/
├── ih_outputs/              # inference-hive parquet
├── corpus/
│   ├── synthetic_corpus.jsonl
│   ├── join_stats.json
│   └── tokenized/tokenize_stats.json
└── runs/datamix_pilot/      # ih_job.slurm, logs, progress
```

## Adjusting pilot size

Edit `mix_config_pilot.yaml`:

```yaml
total_tokens: 5_000_000   # e.g. 5M for a longer pilot
```

Re-run step 1 (`prepare`).
