# PACE Linux worker

The official PACE tree at `dc2ef80e` has no whole-suite Docker image. This
Dockerfile packages the existing scorer and installer, including the repaired
original graders at `8076d686`. It does not implement a replacement benchmark
or model inference. Model requests go to the separately owned checkpoint server.

## Build on Prime

Create a clean build context outside the source checkout. Archive this commit's
`evaluations/`, `scripts/pacebench/`, `requirements.txt`, official calibration
CSVs and PlanBench prompt files as `pace.tar`. Copy `evaluations/Dockerfile`
next to it. Do not include credentials, model weights, caches or previous
private evaluation outputs.

```sh
prime images push pace-worker:20261012 --context <context> --private --plain
prime images list --plain
```

The image uses Python 3.11 and CPU Torch. PACE still runs all its installed
benchmark handlers; BFCL retains its separate virtual environment. PlanBench's
real Linux VAL executable must exist or the image build fails.

## Run

Use one Linux worker initially. Model serving concurrency, not available CPU
cores, determines useful request concurrency. Pass `--workers 1` unless the
actual server owner supplies a higher measured capacity.

```sh
python evaluations/score_new_model.py \
  --model <exact-served-model-id> --model-name-out <checkpoint-label> \
  --base-url <endpoint>/v1 \
  --selections <selection.csv> --workers 1 --journal-dir <output>/journal
```

Inject `API_KEY` as a sandbox secret. Never bake it into the image. Use a finite
sandbox lifetime, download the journal and standardized CSVs, then delete only
the owned sandbox. Image storage costs separately; delete the owned image after
the run unless it is the current reusable worker image.

The journal saves raw handler results and scores after each authentic call.
Resume reuses those observations and restores CSVs rather than replaying model
calls. A model/endpoint/selection mismatch fails. Handler failure or missing
grade exits nonzero and retains the failure; neither becomes a zero score.

## Interpretation

The full official C100 selection has 412 weighted rows and 385 distinct source
rows. The existing text projection has 258 weighted rows and 242 distinct
source rows. It excludes 154 visual weighted rows and is not the complete
multimodal PACE panel. Record all exclusions and missing external credentials.
Do not present a zero-filled projection as a measured agentic benchmark score.
Pi Durable can own agent conversations, but these tasks are a deterministic
resumable scorer, not coding-agent conversations. No extra agent loop is used.

Official documentation checked 2026-10-12:

- https://github.com/neulab/pace
- https://docs.primeintellect.ai/sandboxes/overview
- https://docs.primeintellect.ai/sandboxes/images
- https://docs.primeintellect.ai/tutorials-environments/hosted-evaluations
- https://earendil.com/posts/pi-durable/
