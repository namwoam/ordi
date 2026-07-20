# ORDI

**Orbit-Aware Redundant Distributed Inference for LEO Earth Observation Constellations**

ORDI is a scheduling research prototype for distributing tiled Earth-observation inference workloads across low-Earth-orbit satellites. It runs as a policy on Basilisk/BSK-RL, using orbit-aware contact graphs, satellite resource state, and selective fault-disjoint replication to improve deadline performance without the cost of replicating every task.

The repository includes ORDI, five core comparison policies, a random-placement control, fault injection, four focused evaluations, plotting utilities, and the accompanying paper.

![ORDI core evaluation](figure/E1_core.png)

## Approach

ORDI schedules each image tile over a rolling horizon. For every epoch it:

1. Builds feasible source-helper-aggregator routes from a time-expanded orbital contact graph.
2. Accounts for compute rate, battery, temperature, queue state, availability, latency, and link reliability.
3. Selects a primary assignment by marginal utility and communication congestion; policies do not estimate joules.
4. Adds backups up to a configurable cap only while their marginal reliability gain exceeds their replication cost, while keeping replicas fault-disjoint. The default cap is one.
5. Replans work affected by helper failures, missed contacts, or stragglers.

Basilisk/BSK-RL models the spacecraft environment. ORDI retains workload generation, contact-graph construction, bandwidth allocation, store-and-forward routing, and seven classes of injected faults.

## Requirements

- Python 3.11
- [uv](https://docs.astral.sh/uv/)
- Optional: [Task](https://taskfile.dev/) for the shorthand commands below

## Setup

Install the locked dependencies:

```bash
uv sync
```

With Task installed, the equivalent command is:

```bash
task setup
```

Verify that the experiment modules import successfully:

```bash
task check
```

## Running Experiments

Run an individual experiment:

```bash
task e1
```

Or invoke the Python entry point directly:

```bash
uv run python -m ordi.main run E1
```

Available evaluations are:

| ID | Evaluation |
| --- | --- |
| E1 | ORDI versus five core baselines under matched random faults |
| E2 | Random-fault intensity sweep |
| E3 | Correlated orbital-plane failures |
| E4 | Constellation scalability |

The former real-data case is excluded until its removed Skyfield/TLE path is
replaced by Basilisk propagation; it is not advertised as runnable in the
meantime.

Run the full evaluation suite and generate every plot:

```bash
task all
```

The focused suite uses eight matched seeds for E1, two matched seeds for E2–E3,
and one seed per E4 constellation size (12, 24, and 36 satellites). To run or
plot all experiments without Task:

```bash
uv run python -m ordi.main run all
uv run python -m ordi.main plot all
```

Experiment CSV files are written to `results/`. Generated plots are written to `figure/`.

## Orbit Propagation

Orbit, eclipse, power, battery, thermal, and spacecraft availability state are
simulated by Basilisk 2.11 through the BSK-RL multi-satellite environment. ORDI
is an ordinary scheduling policy: it consumes BSK-RL/Basilisk state and owns
tile placement, redundancy, bandwidth allocation, and store-and-forward
routing. Compute, ISL transmit/receive, and downlink workloads drive Basilisk
power nodes; reported workload energy comes from those nodes rather than from
policy metadata. Skyfield/SGP4 remains only as an optional independent contact-window
cross-check, not as the mission simulator.

On first use Basilisk downloads its official support data (gravity and SPICE
ephemerides) through `pooch`; set `BSK_SUPPORT_DATA_CACHE` to a writable shared
directory in CI or on a cluster.

## Repository Layout

```text
ordi/
├── algorithms/  # Basilisk-facing policies with one shared schema
├── orbit/       # Contact-window construction and time-expanded graphs
├── sim/         # Basilisk adapter, projected state, reliability, measurements
├── tasks/       # EO task generation and workload profiles
├── faults/      # Fault models and injection
└── eval/        # Experiments, metrics, CSV output, and plotting
figure/          # Generated evaluation figures
paper/           # LaTeX source and compiled paper
research_plan.md # Original system and evaluation plan
Taskfile.yml     # Reproducible development and experiment commands
```

## Paper

The paper is available as [paper/main.pdf](paper/main.pdf), with its LaTeX source in `paper/main.tex` and bibliography in `paper/references.bib`.
