# ORDI Improvement Plan

Follow-up work from the evaluation review. Item (a) — Monte Carlo realized-outcome
scoring — is **done** (commit `4ce5188`). The items below remain.

Ordered by impact on the credibility of the paper's claims.

---

## 1. `ground_contact_miss` fault is a no-op (bug) — **FIXED**

**Fix applied (commit TBD):** `_apply` now removes sat→ground edges from
`EpochContactGraph.edges` and `adj` for each epoch in `[start_epoch, end_epoch)`.
`_withdraw` restores them by replaying the stored removed-edge list and
rebuilding `adj`. The worker in `experiments.py` deepcopies `graphs` per job
when any fault is `ground_contact_miss`, preserving the shared object invariant.
E2 re-run confirmed: ground_miss still shows miss_ratio ≈ no_fault, which is
now a genuine measurement (rerouting absorbs the fault) rather than a phantom.

---

## 2. Two modeled failure modes are never exercised — **FIXED**

**Adverse downlink (`π = 0.70`)** — was defined at `reliability.py:20`
(`DEFAULT_DOWNLINK_ADV_PI`), cited in the System Model, but no experiment ever
set it.
- **Fix applied:** `ReliabilityModel` now carries per-aggregator
  `_downlink_overrides` behind a `downlink_pi(agg)` accessor; every downlink
  read (feasibility, ORDI self-processing, realized-MC `down_ok`) routes through
  it. A new `downlink_adverse` fault type sets the override to
  `DEFAULT_DOWNLINK_ADV_PI` for the fault duration, and E2 gains a `downlink_adv`
  scenario that reports it.

**ISL disruption doesn't remove the graph edge** — `injector.py` zeroed
`link_pi` (used only in `z_kv`), but `earliest_arrival` ignores reliability, so a
"disrupted" ISL still carried the tile at full latency. It only lost
probability mass.
- **Fix applied:** `isl_disruption._apply` now also drops the ISL edge (both
  directions) from the epoch graphs for `[start, end)` via a shared
  `_remove_edges`/`_restore_edges` helper (also used by `ground_contact_miss`);
  `_withdraw` replays them. The worker deepcopies `graphs` whenever any fault is
  graph-mutating. E2 now shows a genuine `isl_disruption` miss ratio instead of
  the prior phantom ≈0.

---

## 3. E7 plane-disjoint backup constraint is asserted, not implemented — **FIXED**

**Where:** `ordi/scheduler/ordi.py` (backup selection)

The abstract says "backups are constrained to fault-disjoint helpers and
aggregators... a different orbital plane." The code only required a different
**helper** and **aggregator**; there was no plane check. The paper footnote
(§E7) admitted disjointness was an *emergent measured property* ("greedy scoring
already places 100% of backups in a different plane in this constellation").

**Consequence:** abstract and code disagreed; the headline correlated-failure
result rested on a property that happens to hold for the 6-plane Walker but
isn't guaranteed.

**Fix applied:** implemented the constraint (the stronger paper). `ORDIConfig`
gains a `plane_disjoint_backup` flag (default `False`); when set, the backup
loop rejects any candidate whose helper shares an orbital plane with the
primary's helper. Plane id is parsed from `SAT_<plane>_<idx>` names via
`_plane_of`; unparseable names are treated as disjoint (only both-known-and-equal
is rejected). E7 enables the flag for ORDI only (via per-job `cfg_overrides`),
so nominal experiments keep the emergent-placement framing.

---

## 4. ILP comparison (E8) is a single instance, single seed — **FIXED**

**Where:** `ordi/eval/experiments.py` (`run_E8`).

The "greedy matches ILP" claim rested on one over-constrained 12-sat instance.

**Fix applied:** `run_E8` now sweeps a distribution of small ILP-tractable
instances — two sizes (9 and 12 sats, `_E8_SIZES`) × 12 seeds — via a new
`_run_E8_instance` worker that builds each instance and scores greedy + ILP with
identical stateful lifetime accounting. Instances run concurrently in a
`ProcessPoolExecutor`; `solve_ilp` gained a `threads` param (E8 caps HiGHS to 2
threads per worker since parallelism now comes from the instance sweep). Both
greedy and ILP rows aggregate the full distribution, so the CSV `*_std` columns
report across-instance dispersion and `plot_E8` now draws error bars. Modeled
and realized gaps are both reported. Result: matching mean miss ratios (59.2%),
statistically indistinguishable utility (greedy marginally ahead on realized),
ILP spending 3.6× the ISL traffic and 62% more energy. Abstract, intro, §E8, and
the config table (Seeds 1→12) updated accordingly.

---

## 5. Helper-utilization metric is dimensionally broken — **FIXED**

**Where:** `ordi/eval/metrics.py` (`compute_metrics`).

The old metric divided summed **energy (J)** by **capacity cycles × 1e-9** — a
dimensional mismatch the comment admitted was "approximate".

**Fix applied:** the numerator now sums actual compute cycles
(`tile.compute_ops × len(replicas)` per assignment); the denominator is the
per-satellite compute budget `C_i·epoch_length` summed over the horizon
(`_simulate_stateful` multiplies by `N_EPOCHS` since the final assignment set is
a lifetime record). Both sides are in cycles, so the ratio is dimensionless.
Recomputed Util% is ~5e-5 for ORDI — EO inference is compute-light against the
full constellation budget — with the expected ordering preserved (full-
replication B6–B8 highest, direct-downlink B1 zero). E1 regenerated. Also fixed
a latent crash on the no-faults path (`_parallel_run_algorithm` iterated
`faults` unconditionally, breaking E1/COTS after the graph-mutation change in
`4d9bd6d`).

**Paper follow-up (not yet done):** Table VIII's Util% column still shows the
old ~0.01 values; update from the regenerated `results/E1_core.csv`.

---

## 6. Straggler restore is lossy (minor) — **FIXED**

**Where:** `ordi/faults/injector.py` (`_apply`/`_withdraw`, straggler case).

`_apply` did `C_i *= factor`; `_withdraw` did `C_i /= factor` then
`_throttled_compute_rate()`. If throttle state changed during the fault window
(or straggler windows overlapped), the original `C_i` wasn't recovered exactly.

**Fix applied:** a `_compute_snapshots` dict (fault id → {sat_id: C_i}) captures
the exact pre-fault `C_i` at apply time; `_withdraw` restores it and pops the
entry, rather than inverting the multiply. Verified apply→withdraw round-trips
`C_i` exactly.

---

## 7. Quantify the independence-approximation penalty (research follow-up)

The closed-form `z_kv` (Eq. 2, `reliability.py:107`) assumes independent link/node
failures. With the realized-MC layer now in place (draws shared across a tile's
replicas), the gap between modeled and realized is directly measurable.

**Fix:** report modeled-vs-realized divergence explicitly (a table or an
appendix figure) as the empirical cost of the independence assumption, rather
than only arguing it in prose. The data already exists in the `realized_*`
columns of every results CSV.
