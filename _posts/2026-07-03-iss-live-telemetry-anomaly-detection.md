---
layout: post
title: "Taking It Live: Anomaly Detection on the ISS Feed"
date: 2026-07-03 09:00:00 +0000
image: iss.svg
tags: [ML Infrastructure, Aerospace]
description: Extending the ESA telemetry platform to a live ISS relay feed — and discovering that the assumptions baked into a clean, archived benchmark quietly break the moment the data stops being continuous.
---

<a href="https://github.com/loganrudd/spacecraft-telemetry-anomaly-detection" class="button button--primary" target="_blank" rel="noopener">View on GitHub</a>

This is a follow-up to [the ESA platform]({% post_url 2026-06-15-spacecraft-telemetry-anomaly-detection %}). That system trained and served hundreds of per-channel anomaly detectors on the ESA benchmark. This post is about pointing the same platform at a live feed from the International Space Station — and everything that broke the moment the data stopped being a clean, archived recording.

The short version: a benchmark can hide its own shape. ESA's data is a regularized historical archive — long, continuous, gap-free. The ISS feed is a raw real-time relay through TDRS that loses signal roughly once per orbit. Every quiet assumption the ESA platform made about continuity turned into a live bug.

### The setup

The ISS extension streams a handful of live channels off the public telemetry relay onto the same 30-second grid the ESA pipeline uses, runs them through the same segment-aware windowing, the same Telemanom LSTM, and the same online EWMA-smoothed threshold detector. Because a live feed has no labeled anomalies, evaluation is fault-injection: synthetically inject drift / spike / flatline faults into the stream and measure whether the detector catches them.

Nothing about the *model* changed. Everything about the *data* did.

### The window that couldn't fit an orbit

First live scoring run: every Ray task completed, then the job failed.

```
ValueError: need at least one array to concatenate
  in predict(): np.concatenate(all_preds)
```

An empty `all_preds` means the DataLoader produced **zero windows** for a channel. The windowing is segment-aware by design: a valid `W`-length window requires `segment_ids[s] == segment_ids[s + span - 1]` — the whole window has to fit inside a single unbroken segment. ISS loss-of-signal (LOS) gaps increment `segment_id`. So if no contiguous run is long enough, no window forms, and `predict()` crashes cryptically on the empty concatenate.

I measured the segment lengths across the test split, and the crash wasn't an edge case — it was structural:

| window size | train windows | test windows | usable test segments |
|---|---|---|---|
| **250** (current) | 1,485 | **0 — crash** | 0 |
| 200 | 1,795 | 37 | 1 |
| 160 | 2,430 | 201 | 10 |
| **128** | 3,538 | **561** | 12 |
| 96 | 4,855 | 996 | 15 |

The longest contiguous run in the test split was **237 samples** — just short of the 251 a `W=250` window needs. Training had survived only by luck: the larger train split happened to contain six rare long segments (max 711), while the smaller test split had none. `W=250` wasn't a little too big; it was fundamentally incompatible with ISS's LOS cadence. And it can't be fixed by collecting more data — segment length is bounded by *when the link drops*, not by total volume.

This directly contradicted a "locked" assumption in my design notes: that `W=250 ≈ 1.35 orbits` of continuous signal was achievable. It isn't. The feed never stays up that long.

### Why ESA tolerated it and ISS doesn't

The natural question is why the exact same code sailed through ESA. So I ran the identical measurement on both:

| | longest contiguous segment | test-split structure | rows windowable at W=250 |
|---|---|---|---|
| **ESA** (channel 22) | 292,807 samples | one unbroken 158,912-sample block | 100% |
| **ISS** (live channel) | 711 samples | 33 fragments, max 237 | 0% |

The ESA test split is a *single continuous recording* ~670× longer than ISS's longest fragment. The difference is provenance, not a parameter I tuned wrong:

- **ISS has structural, per-orbit LOS.** It's a live LEO feed relayed through TDRS. The link drops ~16×/day — once per ~90-minute orbit — from TDRS handovers and the zone-of-exclusion where no relay satellite is in view. Those are genuine dropouts in the stream, so gap detection chops the series into roughly per-orbit segments.
- **ESA effectively has no LOS.** The ESA Anomaly Dataset is a curated, archived benchmark from deep-space missions, downlinked to ground stations as long continuous recordings and regularized before publication. The gap detector finds hundreds of "segments," but two of them hold nearly all the data.

The same `detect_gaps → segment_id → within-segment windowing` logic runs on both. ESA's spans are hundreds of thousands of samples, so any window fits trivially. ISS's are capped near one orbit, so a 1.35-orbit window is longer than the feed ever stays up. (It's also why ESA ships labeled anomalies and ISS can't — one was post-processed and annotated, the other is a raw live feed, which is exactly why evaluation here is fault-injection.)

### The tradeoff you can't design around

LOS strikes about once per orbit, so contiguous segments are *always* shorter than one orbit. That makes two goals mutually exclusive: you cannot have a window that both spans a full orbit **and** fits inside a single segment. `W=128` at the 30-second grid is ~64 minutes ≈ 0.7 orbit — about the most orbital context you can buy while still fitting inside a typical inter-LOS segment, and it yields 561 healthy test windows across 12 segments. A sub-orbital window still captures the dominant dynamics; eclipse entry/exit unfolds within roughly an hour. The only lever for *more* context inside a segment would be a finer sampling grid, not a wider window — but 30 s is already the sweet spot before the slow channels start over-filling.

So the fix is a smaller ISS-specific window (an `SPACECRAFT_MODEL__WINDOW_SIZE=128` override on the ISS jobs; ESA stays at 250 for its long segments), a retrain — scoring enforces `cfg.window_size == saved_window_size`, so 250-trained models can't just be re-scored — and an updated window contract that reflects LOS reality instead of the orbit-count fiction. Plus the obvious robustness fix I should have had from the start: the train path already raised a clear error on an empty loader; the test path didn't. Now it does, and the score runner marks a windowless channel *skipped* rather than failing the whole job on a cryptic `np.concatenate`.

### A detector that was blind between passes

Windowing wasn't the only assumption LOS broke. The online detector has warmup physics, and those physics have to fit the cadence of the data.

The EWMA-smoothed threshold needs to see `threshold_window` live ticks before it produces a finite threshold; until then it sits at `+inf` and nothing can be flagged. Two bugs compounded:

- **It never warmed at startup.** Priming the LSTM input window from the replay slice called `reset()` internally, which zeroed the EWMA/threshold state. At the config default of `threshold_window=250`, that's a ~125-minute warmup after boot before detection was even possible.
- **It re-blinded after every LOS event.** LOS recovery re-primed the same way. With ~16 LOS events/day and acquisition-of-signal (AOS) windows of only ~70 minutes, a 125-minute warmup **exceeds the entire window during which there's signal to detect on**. The detector wasn't slow at startup — it was permanently blind during live operation.

The fix warms the scoring state instead of resetting it (`prime_with_scoring()` steps through `window_size + threshold_window` rows, warming both the input window and the threshold ring buffer), and grows the per-channel history buffer so there's enough to warm from after a recovery. Alongside it, ISS HPO caps `threshold_window` at 70 (~35-minute warmup — it fits inside the first half of a typical AOS window and leaves the second half for actual detection), while ESA's search space is untouched. The lesson generalizes past this codebase: **the warmup cost of your smoother has to be shorter than the shortest continuous window your data gives you.** If it isn't, your detector is off precisely when it matters.

### What the archive hid

None of these were model bugs. They were the same bug wearing three costumes — *a clean benchmark told me something about my data that a live feed does not owe me*:

- Continuous segments long enough for any window (they're capped at one orbit).
- Enough uninterrupted ticks to warm a smoother (the link drops first).
- The luxury of ignoring what happens when the data simply stops (it stops ~16×/day).

Extending a platform from an archive to a live relay is less about scaling the model and more about surfacing every continuity assumption the benchmark let you get away with — and pricing them against the physics of a real orbit. That's the part of the work I find most worth writing down.
