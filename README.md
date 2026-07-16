# ORDI

**Orbit-Aware Redundant Distributed Inference for LEO Earth Observation Constellations**

ORDI is a simulator and scheduling research prototype for distributing tiled Earth-observation inference workloads across low-Earth-orbit satellites. It uses orbit-aware contact graphs, satellite resource state, and selective fault-disjoint replication to improve deadline performance without the cost of replicating every task.

The repository includes the ORDI scheduler, eight comparison baselines, fault injection, an ILP reference solver, ten evaluations, plotting utilities, and the accompanying paper.

![ORDI core evaluation](figure/E1_core.png)

## Approach

ORDI schedules each image tile over a rolling horizon. For every epoch it:

1. Builds feasible source-helper-aggregator routes from a time-expanded orbital contact graph.
2. Accounts for compute rate, battery, temperature, queue state, availability, latency, and link reliability.
3. Selects a primary assignment by marginal utility after energy and communication costs.
4. Adds backups up to a configurable cap only while their marginal reliability gain exceeds their replication cost, while keeping replicas fault-disjoint. The default cap is one.
5. Replans work affected by helper failures, missed contacts, or stragglers.

The simulator models Walker constellations, field-of-view-constrained task arrivals, ground contacts, inter-satellite links, workload-specific compute and data profiles, and seven classes of injected faults.

## Requirements

- Python 3.14
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
| E1 | ORDI versus all eight baselines |
| E2 | Robustness by fault type |
| E3 | Fault-intensity sweep |
| E4 | Constellation scalability |
| E5 | Deadline-tightness sweep |
| E6 | Replication-penalty sweep |
| E7 | Correlated orbital-plane failures |
| E8 | Greedy scheduler versus ILP reference |
| E9 | Maximum-backup cap ablation |
| REAL | Planet/FIRMS/BUPT-1 real-data case study |

Run the full evaluation suite and generate every plot:

```bash
task all
```

The complete suite runs many simulation seeds and may take substantial time. To run or plot all experiments without Task:

```bash
uv run python -m ordi.main run all
uv run python -m ordi.main plot all
```

Experiment CSV files are written to `results/`. Generated plots are written to `figure/`.

## Orbit Propagation

Orbits and contact windows are computed with Skyfield + SGP4.

## Repository Layout

```text
ordi/
├── orbit/       # Orbit propagation, contacts, and time-expanded graphs
├── sim/         # Satellite state, reliability, and COTS measurements
├── tasks/       # EO task generation and workload profiles
├── scheduler/   # ORDI, feasibility checks, replanning, routing, and ILP
├── baselines/   # Eight comparison schedulers
├── faults/      # Fault models and injection
└── eval/        # Experiments, metrics, CSV output, and plotting
figure/          # Generated evaluation figures
paper/           # LaTeX source and compiled paper
research_plan.md # Original system and evaluation plan
Taskfile.yml     # Reproducible development and experiment commands
```

## Paper

The paper is available as [paper/main.pdf](paper/main.pdf), with its LaTeX source in `paper/main.tex` and bibliography in `paper/references.bib`.
