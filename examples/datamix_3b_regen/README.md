# Datamix 3B regeneration (dedup + decontam)

Standalone workspace for a **fresh generation run** with:

- Template-aware **prompt dedup** at `prepare`
- **Output dedup** + **DCLM CORE decontamination** at `postprocess`
- **16 GPUs** (16 inference shards × 1 GPU)
- **`total_tokens: 3_000_000_000`** planned completion tokens (~3.6M prompts)

Uses the shared Python package from `../datamix_1b_synthetic/datamix_synth` and the benchmark index at `../datamix_1b_synthetic/benchmarks/`.

IDs use prefix `regen-` and seed `137` so this run does not collide with the original 1B or top-up jobs.

## Will 3B be enough for ~1B after postprocess?

Your **original 1B run** (QC only, no dedup/decontam):

| Stage | Planned | Kept after postprocess |
|-------|---------|------------------------|
| Main | 1.0B completion tokens | **0.78B** (~60% of docs kept) |
| + top-up | +0.3B | **+0.24B** → **~1.01B** merged |

So historical **token survival ≈ 78%** of `mix_config.total_tokens` (document keep rate ≈ 60%).

With dedup + decontam, expect **65–75%** survival (extra rejections unknown until you run postprocess).

| Planned `total_tokens` | Expected kept (65%) | Expected kept (75%) |
|------------------------|---------------------|---------------------|
| **3B** | **~2.0B** | **~2.25B** |
| 5B | ~3.3B | ~3.75B |

**Recommendation: 3B is enough** for a ~1B target with comfortable margin (similar to how 1.3B planned yielded 1.01B kept). Use **5B** only if you want **~2B+** after filtering or if `prompt_dedup_stats.json` shows large `resample_exhausted` (fewer prompts than budget).

After `prepare`, always check:

```bash
jq '.prompts_written, .resample_exhausted' prompts/prompt_dedup_stats.json
cat budget_plan.json | jq '.total_prompts, .estimated_completion_tokens'
```

## Quick start (SLURM)

```bash
cd inference-hive/examples/datamix_3b_regen
chmod +x run_pipeline.sh submit_inference.sh

# 1. Benchmark index (skip if ../datamix_1b_synthetic/benchmarks/ exists)
#    sbatch slurm/01_prepare.sbatch  # only prepare, see below

# 2. Build deduped prompts (~3.6M planned; may take hours)
sbatch slurm/01_prepare.sbatch

# 3. Validate + inference-hive run dir
PREP=$(sbatch slurm/01_prepare.sbatch | awk '{print $NF}')
sbatch --dependency=afterok:${PREP} slurm/02_validate_create_run.sbatch

# 4. Submit 16 GPU shards (login node, after step 3)
./submit_inference.sh

# 5. After all shards finish
sbatch slurm/04_postprocess.sbatch
```

## Re-postprocess only (existing `datamix_1b_synthetic` outputs)

That is **not** this folder. From the original example:

```bash
cd ../datamix_1b_synthetic
MAIN=$(sbatch slurm/04_postprocess.sbatch | awk '{print $NF}')
sbatch --dependency=afterok:${MAIN} slurm/04_postprocess_topup.sbatch
```

The second job starts **only after** the first succeeds (`afterok`). Both are submitted immediately, but SLURM holds the top-up job until main postprocess completes.

## Layout

```
datamix_3b_regen/
  mix_config.yaml
  config_datamix_3b.yaml
  prompts/          # after prepare
  dataset/
  ih_outputs/       # after inference
  corpus/           # after postprocess
  runs/datamix_3b_regen/
```
