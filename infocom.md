# INFOCOM Submission Gap Analysis

## Bottom line

An INFOCOM submission can rely primarily on software simulation; a physical satellite deployment is not required. ORDI already has a credible evaluation foundation, but the present evidence needs stronger calibration, ablation, statistical validation, and formal analysis before it is competitive for INFOCOM.

The goal should be a **simulation-centered evaluation grounded by measured hardware data**, rather than a purely synthetic simulation.

## Current strengths

- Basilisk/BSK-RL orbital dynamics instead of a static network graph.
- End-to-end modeling of contacts, store-and-forward routing, compute queues, execution, and ground delivery.
- Absolute task deadlines and operational deadline-miss accounting.
- Matched workloads, task releases, deadlines, and fault draws across algorithms.
- Independent fault-rate and correlated orbital-plane outage experiments.
- Selective fault-disjoint replication with online failure-risk learning.
- Communication, energy, helper utilization, and scheduler overhead metrics.
- Baselines covering onboard execution, direct downlink, SECO, and replication policies.
- Small-instance oracle machinery and a reproducible open-source implementation.

## Major gaps

### 1. Empirical calibration

The orbital model is credible, but the compute, communication, workload, energy, and failure parameters need external grounding.

Required evidence:

- Cite or measure satellite-class CPU/GPU/NPU inference throughput.
- Measure representative unsplit and 2-way/4-way inference shards on an embedded device or the closest available platform.
- Report the mapping from measurements to simulated compute rates and energy.
- Source ISL and satellite-ground bandwidth, propagation, elevation, and contact assumptions.
- Ground task size, output size, deadline distribution, and arrival rates in a real Earth-observation workload or published mission profile.
- Explain the selected constellation and ground-station deployment.

A small device microbenchmark is sufficient; a complete satellite prototype is unnecessary.

### 2. Defensible fault model

The 2% E1 fault rate must not appear arbitrary or be presented as one operator's measured reliability.

Separate and document:

- Transient compute or accelerator errors.
- Satellite/node outages.
- Hard assignment failures.
- Link and scheduled-contact failures.
- Stragglers and compute-queue misses.
- Correlated plane, software, or environmental failures.

E2 should demonstrate that the result holds across a broad fault-rate range. E3 should show that fault-domain-disjoint placement, rather than replication alone, improves resilience under correlated outages.

### 3. Stronger baseline fidelity

Minimum comparison set:

- Onboard Only.
- Direct Downlink.
- Full Replication.
- Random Replication.
- A clearly documented and faithful SECO adaptation.
- ORDI without backups.
- ORDI with a fixed backup policy.
- An exact solver or oracle on small instances.

The paper must explicitly document differences between the original SECO model and `SECOAdapted`. Avoid implying exact reproduction when assumptions or workload semantics differ.

### 4. Mechanism ablations

The evaluation must establish which ORDI mechanisms cause the improvement. Disable one component at a time:

- Dynamic 1/2/4-way split selection.
- Contact-aware routing and placement.
- Compute-queue awareness.
- Online failure-risk learning.
- Thompson-sampling exploration.
- Fault-domain-disjoint backup selection.
- Failure/straggler replanning.
- Deadline feasibility pruning.

The central question is whether ORDI wins because of informed adaptation or simply because it consumes additional compute and communication resources.

### 5. Formal problem and analysis

The paper needs a precise joint formulation covering:

- Admission.
- Spatial splitting.
- Helper placement.
- Compute and contact capacity.
- Result routing and ground delivery.
- Absolute deadlines.
- Replica cost and failure probability.

Desirable analytical results:

- Complexity or NP-hardness characterization.
- A feasibility or safe-pruning property.
- A reliability expression or bound for fault-disjoint replicas.
- A bound or monotonicity result for adding redundancy.
- Approximation or empirical optimality gap against an exact solver.

### 6. Statistical confidence

Current seed counts may be questioned, especially with stochastic failures.

Improve the analysis with:

- Paired comparisons because algorithms share matched seeds.
- Bootstrap confidence intervals over seeds and/or tasks.
- Effect sizes and confidence intervals, not only mean and standard deviation.
- More seeds for inexpensive scenarios when possible.
- Per-seed scatter or CDFs to reveal outliers and tail behavior.

The E1 result (ORDI 4.90% versus SECO 9.25% deadline misses) is promising, but the submission should demonstrate that this difference is stable across realizations.

### 7. Online feasibility and scalability

Report more than total experiment runtime:

- Scheduler decision-time p50, p95, and p99.
- Throughput in tasks or tiles scheduled per second.
- Candidate placements explored.
- Exact-search or shortlist expansion counts.
- Peak memory use.
- Scaling with request rate.
- Scaling with constellation size, not only workload on a fixed constellation.

Deadline pruning and context-index reuse should be evaluated as algorithm/runtime mechanisms because scheduling must finish fast enough to be operationally useful.

### 8. Reproducibility and validation

- Pin software and dependency versions.
- Export every experiment configuration with the result.
- Preserve deterministic seeds and matched workload identifiers.
- Add validation tests for resource conservation, no overlapping reservations, deadline accounting, and fault attribution.
- Publish scripts that reproduce tables and figures from raw results.
- Clearly separate generated results and figures from source code commits.

## Recommended evaluation package

### E1: Core performance

Answer whether ORDI improves deadline completion under a nominal 2% random-fault setting.

Report deadline-miss ratio, on-time delivery latency, traffic, energy, helper utilization, and replication cost. Include all core operational baselines.

### E2: Independent fault intensity

Answer whether learned selective replication adapts across failure rates.

Report deadline misses, hard-fault misses, average replicas, traffic, energy, and failure-risk estimate calibration. Include zero-fault results to expose unnecessary redundancy.

### E3: Correlated failures

Answer whether fault-disjoint placement improves resilience beyond random or indiscriminate replication.

Compare ORDI, Full Replication, and Random Replication under zero-, one-, and two-plane outages. Report placement diversity as well as deadline misses.

### E4: Scalability

Answer whether ORDI remains effective and fast as demand grows.

Report deadline misses, scheduler latency, throughput, memory, candidate-search size, and resource saturation at each load. Add a constellation-size sweep if runtime permits.

### Additional experiments

- Component ablations.
- Deadline tightness sensitivity.
- Backup-cost and replica-cap sensitivity.
- Failure-estimator convergence and calibration.
- Stale-state sensitivity.
- Small-instance exact-oracle gap.
- Hardware-calibrated compute microbenchmark.

## Positioning against related work

The strongest positioning is not that replication itself is new. Adaptive replication already exists in terrestrial edge systems, while satellite schedulers already optimize latency, energy, observation coverage, and compute placement.

ORDI's defensible distinction is:

> ORDI jointly selects spatial split width, compute helpers, contact-aware delivery routes, and fault-disjoint backups under absolute end-to-end deadlines. It learns failure-domain risk online while accounting for the compute and communication costs of redundancy.

Relevant direct comparisons include SECO (INFOCOM 2024), Phoenix (INFOCOM 2024), TargetFuse (INFOCOM 2024), LEOEdge (JSAC 2025), Krios (SoCC 2024), and adaptive task-replication work from terrestrial edge computing.

## Priority order

1. Finish and validate E1-E4 with the corrected deadline and latency accounting.
2. Add paired confidence intervals and per-seed results.
3. Complete ORDI component ablations.
4. Validate `SECOAdapted` against the original paper and document every deviation.
5. Calibrate compute latency and energy with a small real-device benchmark.
6. Strengthen the fault-model justification and estimator calibration.
7. Add exact-oracle comparisons and formalize safe deadline pruning.
8. Add constellation-size scalability if computationally feasible.
9. Package all configurations, raw results, and plotting scripts for reproduction.

## Submission readiness criterion

The work is ready for an INFOCOM submission when it can support all of the following claims with direct evidence:

- ORDI lowers operational deadline misses under matched resource and workload assumptions.
- The gain persists across independent and correlated failure regimes.
- Selective, fault-disjoint redundancy outperforms fixed, full, and random replication at comparable cost.
- Each adaptive mechanism contributes measurable value.
- Scheduler decisions complete within an operationally meaningful time budget.
- Results are statistically stable and not dependent on one seed or fault rate.
- Simulation parameters are traceable to measurements or authoritative sources.
- Small-instance results are close to an exact or oracle solution.

