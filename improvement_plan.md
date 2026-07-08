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

## 2. Two modeled failure modes are never exercised

**Adverse downlink (`π = 0.70`)** — defined at `reliability.py:20`
(`DEFAULT_DOWNLINK_ADV_PI`), cited in the System Model, but no experiment ever
sets it. Dead config.
- **Fix:** add a weather/adverse-downlink scenario (e.g. an E2 fault class or an
  E3 sweep variant) that swaps `default_downlink_pi` to the adverse value for
  targeted aggregators, and report it.

**ISL disruption doesn't remove the graph edge** — `injector.py:80-86` zeroes
`link_pi` (used only in `z_kv`), but `earliest_arrival` ignores reliability, so a
"disrupted" ISL still carries the tile at full latency. It only loses
probability mass.
- **Fix:** make ISL disruption also drop the edge from `E(t)` for the fault
  duration (mirror the plane-outage/helper-failure path that sets `A_i`).
  With the realized-MC layer now in place, the sampled `p=0` will also drop the
  replica in scoring — but feasibility should reflect the outage too.

---

## 3. E7 plane-disjoint backup constraint is asserted, not implemented

**Where:** `ordi/scheduler/ordi.py:270-300` (backup selection)

The abstract says "backups are constrained to fault-disjoint helpers and
aggregators... a different orbital plane." The code only requires a different
**helper** and **aggregator** (`ordi.py:273-276`); there is no plane check. The
paper footnote (§E7) admits disjointness is an *emergent measured property*
("greedy scoring already places 100% of backups in a different plane in this
constellation").

**Consequence:** abstract and code disagree; the headline correlated-failure
result rests on a property that happens to hold for the 6-plane Walker but isn't
guaranteed.

**Fix (choose one):**
- Implement the constraint: under a correlated-failure threat model, reject a
  backup candidate whose helper shares an orbital plane with the primary's
  helper (plane id is parseable from `SAT_<plane>_<idx>` names). Add a config
  flag so it's only active for E7-style scenarios.
- Or soften the abstract to match the code and keep the emergent-property framing.

Implementing it is the stronger paper.

---

## 4. ILP comparison (E8) is a single instance, single seed

**Where:** `ordi/eval/experiments.py:714` (`run_E8`, `seed=0`, one instance);
Table VII confirms "Seeds: 1".

The "greedy matches ILP" claim rests on one over-constrained 12-sat instance.

**Fix:** run a distribution of small instances (multiple seeds, maybe a couple of
sizes within ILP tractability) and report the optimality gap with error bars —
this is the quantified greedy-vs-optimal gap the research plan promised, instead
of an anecdote. Now that realized-MC scoring exists, report both modeled and
realized gaps.

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
