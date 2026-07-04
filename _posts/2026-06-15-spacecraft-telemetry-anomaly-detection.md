---
layout: post
title: Spacecraft Telemetry Anomaly Detection
date: 2026-06-15 09:00:00 +0000
image: telemetry.svg
tags: [ML Infrastructure, Aerospace]
description: An end-to-end MLOps platform detecting anomalies across hundreds of channels of real ESA spacecraft telemetry — built so per-channel models stay trainable and servable without infra cost exploding.
---

<a href="https://github.com/loganrudd/spacecraft-telemetry-anomaly-detection" class="button button--primary" target="_blank" rel="noopener">View on GitHub</a>

### Architecture at a glance

```
ESA telemetry (Parquet, ~225 channels)
  → Ray fan-out preprocessing
  → per-channel Telemanom LSTM  (Ray parallel train · Ray Tune ASHA HPO)
  → MLflow registry (@champion)
  → FastAPI + SSE stream replay
  → React / Recharts dashboard

Infra: GKE Autopilot + KubeRay · Cloud Run (scale-to-zero) · Terraform
       · GitHub Actions + Workload Identity Federation
```

### The problem

Real spacecraft telemetry is a fan-out problem before it's a modeling problem. The ESA benchmark used here is roughly 225 channels across three missions and 31 GB of sensor data, and the honest baseline — Telemanom, a small per-channel LSTM forecaster — means training and serving *hundreds* of independent models. The interesting question isn't "what's the best model," it's "how do you keep hundreds of per-channel models trainable, tunable, and servable without the infrastructure cost exploding." I used an off-the-shelf model deliberately, so the platform — not the model — was the work.

### The architecture

Preprocessing and training fan out across channels with Ray Core on KubeRay. Each channel gets its own Telemanom LSTM (2-layer, hidden_dim 80), trained only on nominal data as a one-step-ahead forecaster, with Ray Tune (ASHA) doing per-subsystem hyperparameter search. Models land in an MLflow registry under a `telemanom-{mission}-{channel}` scheme with a `@champion` alias for promotion. Serving is a FastAPI app that replays streams over server-sent events, scoring each tick against EWMA-smoothed residuals and a dynamic threshold, with a React/Recharts dashboard drawing ground-truth vs. predicted anomaly bands. It all runs on GCP: GKE Autopilot + KubeRay for batch compute, Cloud Run (scale-to-zero) for the API and MLflow, Terraform for IaC, and GitHub Actions with Workload Identity Federation for CI/CD.

### What was hard, and what I decided

**Memory under heterogeneous fan-out.** Channels range 50–240 MB compressed, and naive fan-out OOM-killed 4 GB Ray workers. Rather than throw bigger machines at it — the cost-explosion trap — I profiled peak RSS per task and bin-packed by actual footprint: `pd.Categorical` encoding, building arrays with `Categorical.from_codes` instead of Python lists (per-task peak 2.35 GB → 0.5 GB), dropping defensive copies, and Ray `memory=` hints with `max_calls=1` to force reclamation between tasks. Measure peak RSS *before* tuning concurrency.

**The wrong tool on the hot path.** I started with Evidently for drift detection, but its batch reports (~58 ms each) saturated the thread pool at high replay speeds. I swapped to a direct scipy Wasserstein computation (under 1 ms) offloaded with `asyncio.to_thread`, keeping Evidently for the offline path. A great batch tool can be the wrong choice on a latency budget.

**Scale-to-zero makes startup a first-class problem.** Cold starts on a 2 GiB Cloud Run instance OOM'd because every champion model loaded via an unbounded `asyncio.gather`. Fix: deferred background loading behind a bounded semaphore, per-channel bounded queues so one slow channel can't stall the rest, and a single merged SSE connection. `/health` returns early with `status="loading"` while the app already accepts requests.

**Credentials that outlive the job.** GCP ID tokens expire after an hour, so Ray workers on training runs longer than that hit 403s mid-job. I added an in-flight `refresh_mlflow_auth()` called once per epoch — a cheap dict lookup when there's more than 10 minutes left, a refetch only near expiry. Any long-running distributed job that outlives a credential needs refresh, not just auth at startup.

**A fractional GPU is a scheduler hint, not a memory fence.** Scoring got OOM-killed on an L4 at the *exact* `num_gpus=0.125` (8-way) packing that training had run fine at. The trap: Ray's fractional GPU is a logical packing factor — it tells the scheduler how many tasks to co-locate, and fences nothing in VRAM. Training and inference simply have different working sets. Each scoring process held ~3.6 GiB of CUDA context plus a 2048-row activation buffer, so 8 × 3.6 ≈ 29 GiB overran the L4's ~22 GiB usable — while training's smaller forward-pass footprint had packed cleanly at the same factor. Fix: back scoring off to 5-way (`num_gpus=0.2`) and add `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to cut allocator fragmentation under sharing. A fractional-GPU number is a promise the scheduler can't keep — size it against real per-task VRAM, and measure it per phase, not once.

### Infrastructure that fails without telling you

Half the hard bugs here never threw. The infrastructure did something plausible and wrong, and the only signal was a number that looked fine. These are the ones worth remembering.

**"Same project" is not a private network path.** Ray pods on GKE Autopilot read GCS fine but had no route to the MLflow service on Cloud Run — repeated 403-adjacent failures that looked like IAM. Autopilot nodes carry no external IPs, and Private Google Access only covers Google *API* endpoints (`storage.googleapis.com` and friends); a `*.run.app` URL egresses over the public internet even when both services live in the same project. No external IP meant no egress path at all. Fix: a Cloud Router + Cloud NAT in Terraform, so outbound traffic shares one NAT IP without assigning a public IP per node. "It's all Google, so it's all private" is a comforting lie — check which endpoints actually ride Private Google Access before you trust the path.

**Two systems owning one field will fight.** Terraform owned the full `google_cloud_run_v2_service`, image field included — and so did the GitHub Actions pipeline, which set that image on every deploy. Neither knew about the other's last write, so every `terraform apply` after a deploy silently reverted the live service to the placeholder image baked into state, breaking production with a green plan. Fix: `lifecycle { ignore_changes = [template[0].containers[0].image] }` — Terraform owns env vars, scaling, and IAM; the pipeline owns the artifact. When two systems can write the same field, you carve the boundary explicitly or they clobber each other on every run.

**The bug that made "tuned" runs quietly untuned.** The sneakiest bug in the project logged nothing at all. `TUNED=1` scoring runs completed cleanly and reported metrics — identical, it turned out, to the untuned baseline. The entrypoint fetched its config with `gsutil cp gs://.../tuned_configs.json /tmp/ 2>/dev/null || true`, but `gsutil` wasn't in the slim Python image: the copy no-op'd, `|| true` swallowed the failure, and the `if [ -f ... ]` branch never fired. Every "tuned" run scored with untuned defaults — no crash, no error line, pure measurement-integrity corruption. Fix: fetch with `gcsfs` (already a shipped dependency) behind an `exists()` guard, so a missing file is an explicit no-op and a real download failure is a real error. On any correctness-critical path, `|| true` is how you turn a loud failure into a silently wrong number — and a silently wrong number is the most expensive kind.

### On results, honestly

Because the model is a deliberate baseline, the detection numbers are modest — segment-overlap F0.5 ≈ 0.19 fleet-mean on the held-out final 40%, with 30 of 31 labeled channels detected and a few channels above 0.7. I report segment-overlap F0.5 rather than the point-adjust convention common in the SMAP/MSL literature *because* point-adjust inflates scores by crediting an entire anomaly segment for a single detected point. The project is about the platform and the engineering judgment behind it, and honest evaluation is part of that.
