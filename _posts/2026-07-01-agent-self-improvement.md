---
layout: post
title: Agent Self-Improvement
date: 2026-07-01 09:00:00 +0000
image: agent.svg
tags: [Agents, ML Infrastructure]
description: A self-correcting text-to-SQL agent that detects its own performance drift with windowed statistics and learns from its failures — recovery validated by a McNemar test. I owned the drift-detection stage.
---

<a href="https://github.com/loganrudd/agent-self-improvement" class="button button--primary" target="_blank" rel="noopener">View on GitHub</a>

### Architecture at a glance

```
Spider queries
  → Harness    (agent run → TelemetryRecord)
  → Detector   (windowed drift detection → DriftEvent)      ← my stage
  → Correction (teacher generates + verifies SQL → few-shot examples)
  → Viewer     (events.jsonl → recovery curve)

Contracts: frozen Pydantic types between stages · append-only event log
           · feedback = edits to AgentConfig.few_shot_examples
```

### The problem

Production AI agents drift — the same agent that worked last week quietly degrades — and most "self-improving agent" demos can't tell a genuinely bad run from noise. Built at an AI Engineer World's Fair hackathon (24 hours, four people), this is a self-correcting text-to-SQL agent that detects its own degradation and learns from its failures, with the improvement held to a real statistical bar. I owned the **Detector** stage — the drift detection.

### The architecture

Four typed stages with frozen Pydantic contracts between them, over an append-only event log. **Harness** runs the agent on Spider questions and emits `TelemetryRecord`s. **Detector** (mine) consumes those and emits `DriftEvent`s. **Correction** has a stronger teacher model generate and *verify* SQL to mint few-shot examples. **Viewer** turns the event log into a recovery curve. "Feedback" is concretely a modification to `AgentConfig.few_shot_examples` — the loop is data, not weights.

### What was hard, and what I decided

**Distinguishing drift from noise (the Detector).** A single wrong query is noise; sustained degradation is drift. I built the detector as windowed rather than point-wise, running as a small state machine — WARMUP → NORMAL → DRIFTING — so it only fires on a persistent shift in error rate, not one-off failures. That's the difference between an alert you trust and one you mute.

**Proving the agent actually improved.** It's easy to show a number going up and call it learning. We validated recovery with a paired McNemar exact test on 30 held-out hard questions — 0.567 vs. 0.300 accuracy, a gain of 0.267, p = 0.016 — so the improvement is statistically significant, not cherry-picked. To make the test meaningful, we used an intentionally weak agent against a stronger teacher, so any improvement reflects genuine learning from failures rather than the base model already knowing the answer.

**Corrections you can trust.** The teacher's SQL is accepted only if it executes to the correct result set; otherwise it's discarded and the gold query is used instead. Examples are stratified out-of-sample by database, with anti-forgetting anchors, so the agent doesn't overfit to the drift window it just recovered from.
