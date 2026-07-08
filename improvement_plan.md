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

## 5. Helper-utilization metric is dimensionally broken

**Where:** `ordi/eval/metrics.py:107-115`

```python
# Convert energy back to compute cycles ... approximate
m.helper_utilization = min(1.0, compute_used / max(total_capacity * 1e-9, 1e-9))
```

Divides summed **energy (J)** by **compute-capacity cycles × 1e-9**; the comment
admits it's approximate. The `Util%` column in Table VIII is not a trustworthy
quantity.

**Fix:** track actual compute cycles (or compute-seconds) used per helper and
divide by `C_i · epoch_length` summed over the horizon. Recompute the Util%
column for Table VIII.

---

## 6. Straggler restore is lossy (minor)

**Where:** `ordi/faults/injector.py:146-151`

`_apply` does `C_i *= factor`; `_withdraw` does `C_i /= factor` then
`_throttled_compute_rate()`. If throttle state changed during the fault window,
the original `C_i` isn't recovered exactly.

**Fix:** snapshot `C_i` at apply time and restore the snapshot on withdraw,
rather than inverting the multiply. Matters for long sweeps with overlapping
faults.

---

## 7. Quantify the independence-approximation penalty (research follow-up)

The closed-form `z_kv` (Eq. 2, `reliability.py:107`) assumes independent link/node
failures. With the realized-MC layer now in place (draws shared across a tile's
replicas), the gap between modeled and realized is directly measurable.

**Fix:** report modeled-vs-realized divergence explicitly (a table or an
appendix figure) as the empirical cost of the independence assumption, rather
than only arguing it in prose. The data already exists in the `realized_*`
columns of every results CSV.
