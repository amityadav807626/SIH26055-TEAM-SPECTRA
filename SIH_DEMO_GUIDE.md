# SIH Demo Guide — SPECTRA / SMART-SCAN V3.8

## 1. Opening statement

> “SPECTRA is a software-only synthetic spectrum scanning simulation. The scheduler learns online with a discounted Beta-Bernoulli Thompson Sampling policy: it observes each scan, updates the selected band's posterior, samples candidate priorities, and adapts the next scan.”

## 2. Show Live Console

1. Select **Live Console**.
2. Use 20–30 bands and 260 slots.
3. Keep **Scenario = BASELINE** for the standard run.
4. Click **START LIVE SCAN**.
5. Point out the synthetic spectrum waterfall, selected band, event stream, learned band rewards and the live execution overlay.
6. Point out **Decision Latency: X.X µs (Dwell Valid)** in the header. This is software decision-loop telemetry, not a hardware timing certification.

## 3. Explain the online bandit

Each band begins with a Beta(1,1) prior. A successful synthetic detection increases alpha; a non-detection increases beta. A discount factor reduces the influence of old evidence. Thompson Sampling then draws a probability from each band's current posterior, allowing both exploitation and uncertainty-driven exploration.

Useful jury wording:

> “There is no offline model file. The policy starts with an uninformative prior and learns from the current synthetic scan stream.”

## 4. Show Agile Hopping

1. Select **AGILE HOPPING**.
2. Use a hop interval such as 12 slots.
3. Start the scan again.
4. Explain that the synthetic target channel group changes every N slots according to a seeded pseudo-random sequence.
5. Point out the returned `avg_hop_recovery_time` and `receiver_blind_time_fraction` if viewing the API response.

This scenario is intentionally abstract: it models changing channel occupancy, not a real-world emitter or RF waveform.

## 5. Show Performance

Explain:

- **System P_D** — fraction of all active synthetic band-time opportunities that were actually intercepted.
- **P_D_scanned / recall** — detector recall among observations that were scanned.
- **Precision** — fraction of positive detections that correspond to active synthetic cells.
- **F1** — harmonic balance of precision and recall.
- **P_FA** — false-alarm rate among inactive scanned cells.
- **Mean First-Intercept Delay** — elapsed simulation slots from activity onset to first successful observation.

## 6. Show Benchmark

Click **RUN BASELINE COMPARISON**.

Both strategies use the same seeded synthetic world and detector draws within each trial. The benchmark repeats controlled trials and reports mean ± sample SD.

Say:

> “The standard deviation shows run-to-run variability across controlled synthetic trials. It is not being presented as a confidence interval or as proof of statistical significance by itself.”


## 7. Show the TSRD-derived activity option

The browser **DATA** selector has a third option: **TSRD**. Select it and choose a split such as `TEST SCAN`, then run the live scan or benchmark.

Explain it this way:

> “We keep the scheduler independent of the source format. The selected TSRD files were converted offline into a compact, normalized activity representation. SPECTRA consumes only `band, time_slot, activity`, so the same reward-based scheduler and Round Robin baseline can be evaluated against the derived activity stream.”

The six bundled compact proxies correspond to the two files selected from each TSRD scan/stare train/validation/test split. The raw HDF5 files are not bundled. The `band` value is an abstract scheduler channel bucket, not a claim about physical frequency.

## 8. If a judge asks “Why not Random Forest?”

> “The previous prototype used an offline Random Forest artifact. We replaced that dependency with discounted Thompson Sampling so the scheduler learns directly from the current scan stream. That removes the serialized model and training dataset from the runtime package and makes the adaptive decision loop genuinely online.”

## 9. If a judge asks “Why compare with Round Robin?”

> “Round Robin is a deterministic uniform-scan baseline. It gives us a simple reference for understanding how adaptive allocation changes detection coverage, delay and scan behavior.”

## 10. If a judge asks “Is this real RF?”

> “No. This prototype is intentionally software-only and uses synthetic channel occupancy. It demonstrates scheduling, learning, telemetry and evaluation without making real-world RF measurements.”

## 11. Avoid these claims

Do not claim:

- real RF interception,
- real emitter identification,
- guaranteed EW superiority,
- real-world detection range,
- hardware dwell-time certification,
- operational deployment readiness.

The project is a synthetic software simulation prototype.


## Jury-provided CSV demo

1. Open the SPECTRA browser console.
2. Click **DATA → JURY CSV**.
3. Select the jury CSV.
4. Confirm the record count shown in the header.
5. Run **START SCAN** or **RUN BASELINE COMPARISON**.
6. The dashboard records the source as **Jury Normalized Activity**.

Expected CSV schema: `band,time_slot,activity`.
