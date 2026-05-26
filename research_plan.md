# ORDI Research Plan
**Orbit-aware Redundant Distributed Inference across LEO Satellites**

---

## Overview

ORDI solves the tile-level distributed EO inference scheduling problem on LEO constellations with runtime fault tolerance. This plan covers simulator construction, algorithm implementation, baseline comparison, fault injection, and evaluation.

---

## Phase 1: Environment & Orbit Infrastructure

**Goal:** Reproducible Python environment + working contact-window generator from real TLEs.

### 1.1 Dependencies
```
skyfield          # orbit propagation from TLE
networkx          # contact graph construction
simpy             # discrete-event simulation
pulp              # LP/MIP solver (ORDI scheduler)
numpy scipy       # numerics
pandas            # results collection
matplotlib        # plotting
tqdm              # progress bars
```

### 1.2 TLE Data
- Use Planet Dove / Flock constellation TLEs (Celestrak `planet.txt`) as the satellite set.
- Supplement with a synthetic Walker-Delta constellation (550 km, 53°, 72 planes × 22 sats) to stress-test scale.
- Ground stations: a representative set of ~10 globally distributed stations (Fairbanks, Svalbard, Punta Arenas, Singapore, Nairobi, …).

### 1.3 Contact Window Computation (`orbit/contacts.py`)
- For each (sat, ground_station) pair: compute rise/set windows over a simulation horizon using Skyfield `find_events`.
- For each (sat_i, sat_j) pair: compute ISL contact windows based on range threshold (≤ 2500 km) and elevation mask.
- Output: `ContactSchedule` — a sorted list of `(t_start, t_end, node_a, node_b, peak_rate_bps)` events.
- Validate: check contact durations match expected ~10 min/pass for LEO–ground links.

### 1.4 Time-Expanded Contact Graph (`orbit/graph.py`)
- Build a `networkx.DiGraph` where each node is `(satellite_id, epoch)` and edges represent feasible contacts within the epoch.
- `earliest_arrival(src, dst, t, data_bits)` → transfer latency using Dijkstra on the time-expanded graph.

---

## Phase 2: Satellite State Model

**Goal:** Implement the per-satellite state vector σ_i(t) = (C_i, B_i, Θ_i, Q_i, A_i).

### 2.1 Satellite State (`sim/satellite.py`)
| Parameter | Source / Model |
|---|---|
| `C_i(t)` compute rate | profiled values: Jetson Orin ~20 TOPS, throttled to 4–8 TOPS in thermal stress |
| `B_i(t)` battery | simple energy balance: solar harvest rate − load draw, typical 20–30 Wh cubesat |
| `Θ_i(t)` thermal | first-order RC model: dΘ/dt = (P_compute + P_comms − P_radiate) / C_thermal |
| `Q_i(t)` queue | running sum of assigned compute cycles not yet completed |
| `A_i(t)` availability | 1 unless B < B_min or Θ > Θ_max or injected failure |

### 2.2 Link Reliability Model (`sim/reliability.py`)
- `π_ij(t)`: Bernoulli model per link type
  - ISL (clear-sky): π = 0.97
  - Ground downlink (clear-sky): π = 0.92
  - Ground downlink (adverse): π = 0.70 (for fault injection)
- `p_kvia(t)` = π_node_i × π_path_ski × π_path_ia × π_down_a

---

## Phase 3: EO Task & Tile Model

**Goal:** Realistic EO inference task generation with per-tile compute/data profiles.

### 3.1 Task Generator (`tasks/generator.py`)
- Draw task arrival process: Poisson with rate λ tasks/orbit.
- Each task: source sat sampled uniformly, release time r_k, deadline D_k = r_k + Δ_deadline (60–600 s).
- Image resolution: 512×512 to 2048×2048 pixels at 3-band uint8.
- Tile grid: 4×4 to 8×8 tiles per image.

### 3.2 Tile Compute Profiles (`tasks/profiles.py`)
Profile four EO workload types against a lightweight CNN (MobileNetV2 / YOLOv5-nano):

| Task | Model | Input (MB/tile) | Output (kB/tile) | Compute (GFLOP/tile) |
|---|---|---|---|---|
| Wildfire/thermal anomaly | MobileNetV2 classifier | ~0.75 | ~1 | ~0.3 |
| Ship detection | YOLOv5-nano | ~0.75 | ~5 | ~0.9 |
| Cloud filtering | lightweight U-Net | ~0.75 | ~50 | ~1.2 |
| Change detection | Siamese CNN | ~1.5 | ~10 | ~1.8 |

Use `time_to_compute = GFLOP / C_i(t)` with profiled GFLOP/s from local hardware benchmarks.

---

## Phase 4: ORDI Scheduler

**Goal:** Implement the rolling-horizon greedy optimizer from Part 4 of the proposal.

### 4.1 Feasibility Checker (`scheduler/feasibility.py`)
For each (k, v, i, a) candidate:
1. Compute L_kvia(t) = ℓ_ski + c_kv/C_i + ℓ_ia + ℓ_down_a
2. Check δ_kvia = L_kvia ≤ τ_k(t) AND A_i AND A_a
3. Compute p_kvia(t)

### 4.2 Greedy Allocation (`scheduler/ordi.py`)
```
For each epoch t:
  1. Update satellite states σ_i(t)
  2. Rebuild contact graph edges E(t)
  3. For each pending task k, tile v:
     a. Enumerate feasible (i, a) candidates
     b. Score by marginal objective: ΔU_kv − λ_E ΔE − λ_C ΔC_comm − λ_R ΔR_rep
     c. Assign primary replica (best score)
     d. Assign backup replica if marginal gain > 0 and r_max_kv > 1
  4. Commit assignments; update Q_i, B_i, link utilization
```

### 4.3 Replanning Triggers (`scheduler/replan.py`)
- Helper failure detected (A_i → 0)
- Missed contact (link unavailable at scheduled time)
- Straggler: tile not returned within 1.5× expected latency
- New high-priority task arrival (u_kv > threshold)

### 4.4 ILP Reference Solver (`scheduler/ilp.py`)
- Use PuLP to formulate the exact MILP for small instances (≤ 5 tasks, ≤ 20 sats).
- Used for: (a) validating greedy optimality gap, (b) ablation experiments.

---

## Phase 5: Baselines

Implement eight baselines for apples-to-apples comparison:

| ID | Name | Description |
|---|---|---|
| B1 | Direct downlink | No onboard inference; downlink raw tiles, process on ground |
| B2 | Onboard-only | Each tile processed only by source satellite, no helpers |
| B3 | Compression-only | Compress tiles before downlink; no distributed compute |
| B4 | Serval-like | Priority-queue bifurcation; single satellite per task |
| B5 | SECO-like | Multi-satellite placement, no redundancy, greedy time-cost |
| B6 | Full replication | Every tile replicated to max r_max helpers |
| B7 | Random replication | Replicate to random feasible helpers |
| B8 | CoCoI-like | MDS coded redundancy (terrestrial model adapted to contact windows) |

---

## Phase 6: Fault Injection

Inject failures during simulation to test ORDI's fault tolerance:

| Fault Type | Implementation |
|---|---|
| ISL disruption | Remove edge from E(t) for a random duration |
| Orbital-plane outage | Set A_i = 0 for all sats in a plane for N epochs |
| Helper failure | Flip A_i → 0 mid-task; trigger replanning |
| Straggler | Scale C_i(t) by 0.1 for random helper during execution |
| Ground-contact miss | Remove downlink window; delay results |
| Battery shortage | Set B_i below B_min; force A_i = 0 |
| Thermal throttling | Set Θ_i > Θ_max; reduce C_i to 25% |

---

## Phase 7: Evaluation Metrics & Experiments

### 7.1 Metrics
- **Deadline miss ratio** (primary): fraction of tiles not delivered before D_k
- **Delivered utility**: Σ u_kv × z_kv (expected) vs. actual delivered utility
- **Partial coverage**: fraction of tiles in a task delivered (useful even if incomplete)
- **Recovery latency**: time from failure detection to result delivered via backup
- **ISL traffic (bits)**: total inter-satellite data transferred
- **Downlink volume (bits)**: total satellite→ground data
- **Energy use (J)**: total helper energy consumed
- **Helper utilization**: fraction of helper compute capacity used

### 7.2 Experiments
| Exp | Variable | Fixed | Goal |
|---|---|---|---|
| E1 | Algorithm (ORDI vs. all 8 baselines) | No faults, 20 sats, 4 tasks | Core performance |
| E2 | Fault type (7 types) | ORDI only | Robustness profile |
| E3 | Fault intensity (0%–50% failure rate) | ORDI vs B5, B6 | Graceful degradation |
| E4 | Constellation size (10–100 sats) | ORDI vs B5 | Scalability |
| E5 | Deadline tightness (60s–600s) | ORDI vs B2, B4 | Deadline sensitivity |
| E6 | λ_R sweep (0.0–2.0) | ORDI | Replication penalty effect |
| E7 | Correlated failures (orbital-plane outage) | ORDI vs B6 | Independence assumption |
| E8 | ILP vs. greedy gap | Small instances | Optimality of greedy |

---

## Phase 8: Implementation Schedule

```
Week 1:  Phase 1 + 2  — orbit infra, contact graph, satellite state model
Week 2:  Phase 3 + 4  — task/tile model, ORDI scheduler (greedy + ILP)
Week 3:  Phase 5      — all 8 baselines
Week 4:  Phase 6 + 7  — fault injection, run all experiments (E1–E8)
Week 5:  Analysis     — plots, tables, paper writeup
```

---

## File Structure

```
pdp-final/
├── research_plan.md
├── proposal.tex
├── reference/
├── ordi/
│   ├── orbit/
│   │   ├── contacts.py       # TLE → contact windows
│   │   └── graph.py          # time-expanded contact graph
│   ├── sim/
│   │   ├── satellite.py      # state vector σ_i(t)
│   │   └── reliability.py    # link/node reliability
│   ├── tasks/
│   │   ├── generator.py      # EO task + tile generation
│   │   └── profiles.py       # per-tile compute/data profiles
│   ├── scheduler/
│   │   ├── ordi.py           # rolling-horizon greedy
│   │   ├── feasibility.py    # δ_kvia, p_kvia computation
│   │   ├── replan.py         # failure/straggler replanning
│   │   └── ilp.py            # exact ILP reference
│   ├── baselines/
│   │   └── baselines.py      # B1–B8
│   ├── faults/
│   │   └── injector.py       # fault injection framework
│   ├── eval/
│   │   ├── metrics.py        # metric computation
│   │   ├── experiments.py    # E1–E8 experiment runner
│   │   └── plots.py          # matplotlib figure generation
│   └── main.py               # CLI entry point
└── results/                  # experiment output CSVs + figures
```
