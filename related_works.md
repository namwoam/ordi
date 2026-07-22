# Related Work from Leading Venues

## Scope

ORDI lies at the intersection of orbital edge computing, deadline-aware scheduling, multi-satellite cooperation, and adaptive fault-tolerant replication. No single prior system covers all four dimensions. The most relevant literature therefore falls into four groups:

1. Orbital and satellite edge-computing systems.
2. Multi-satellite Earth-observation scheduling.
3. Deadline-aware edge scheduling.
4. Learning-based task replication and fault tolerance.

This document prioritizes papers from selective systems, networking, and communications venues. Citation counts are intentionally omitted because they vary across indexes and are especially unstable for papers published since 2024.

## Direct satellite-computing systems

### Orbital Edge Computing: Nanosatellite Constellations as a New Class of Computer System

- **Authors:** Bradley Denby and Brandon Lucia
- **Venue:** ACM ASPLOS 2020
- **Link:** <https://doi.org/10.1145/3373376.3378473>
- **Open manuscript:** <https://abstract.ece.cmu.edu/pubs/oec-asplos2020.pdf>

This work establishes orbital edge computing as a distinct computer-systems design space. It argues that large nanosatellite constellations should process sensor data in orbit rather than operate only as bent-pipe collectors. It jointly exposes constraints from orbital motion, sensing quality, downlink capacity, onboard energy, and computation, and introduces computational nanosatellite pipelines that distribute sensing, processing, and communication across a constellation.

**Relationship to ORDI:** This paper provides the architectural motivation for onboard and distributed inference. ORDI addresses a downstream scheduling problem: how to place, split, route, and selectively replicate deadline-constrained inference work in a time-varying constellation. ORDI additionally models compute queues, end-to-end delivery deadlines, execution failures, and correlated fault domains.

### Krios: Scheduling Abstractions and Mechanisms for Enabling a LEO Compute Cloud

- **Authors:** Vaibhav Bhosale, Ada Gavrilovska, and Ketan Bhardwaj
- **Venue:** ACM SoCC 2024
- **Link:** <https://doi.org/10.1145/3698038.3698566>
- **Open manuscript:** <https://vaibhavb007.github.io/papers/krios.pdf>

Krios treats a LEO constellation as an emerging compute cloud. It introduces LEO zones as an orchestration abstraction and provides mechanisms for placing and migrating application instances as satellites move. Its central concern is maintaining application availability without deploying an instance on every satellite or performing excessive handovers.

**Relationship to ORDI:** Krios schedules relatively persistent service instances, whereas ORDI schedules finite Earth-observation inference requests with absolute completion and delivery deadlines. Krios manages availability through instance placement and migration; ORDI chooses spatial split width, compute helpers, contact-aware result routes, and fault-disjoint task backups. The two systems are complementary rather than interchangeable.

### LEOEdge: A Satellite-Ground Cooperation Platform for AI Inference in Large LEO Constellations

- **Authors:** Su Yao, Yiying Lin, Mu Wang, Ke Xu, Mingwei Xu, Changqiao Xu, and Hongke Zhang
- **Venue:** IEEE Journal on Selected Areas in Communications, 2025
- **Link:** <https://doi.org/10.1109/JSAC.2024.3460083>

LEOEdge addresses AI inference across heterogeneous satellite and ground resources. It automatically adapts models to satellite capabilities, uses a layered distributed scheduler to select execution locations, and provides seamless data transmission to prevent predictable satellite movement from interrupting delivery. Its evaluation emphasizes model-search efficiency, execution latency, and delivery latency.

**Relationship to ORDI:** LEOEdge is stronger in hardware-aware neural-model generation and accuracy/latency tradeoffs. ORDI assumes a fixed inference workload and instead focuses on operational reliability: absolute deadlines, spatial task splitting, queue-aware helper placement, online failure-risk learning, fault-disjoint backups, and replanning after failures or stragglers. LEOEdge handles mobility-induced transmission interruption, but it does not center stochastic compute failure or correlated fault-domain replication.

### SECO: Multi-Satellite Edge Computing Enabled Wide-Area and Real-Time Earth Observation Missions

- **Authors:** Zhiwei Zhai, Liekang Zeng, Tao Ouyang, Shuai Yu, Qianyi Huang, and Xu Chen
- **Venue:** IEEE INFOCOM 2024
- **Pages:** 2548-2557
- **Link:** <https://doi.org/10.1109/INFOCOM52122.2024.10621270>

SECO jointly optimizes multi-satellite observation scheduling, image routing, and computation-node selection for wide-area, real-time Earth-observation missions. It uses satellite motion and rotatable cameras to coordinate coverage and cooperative processing in a dynamic, heterogeneous constellation.

**Relationship to ORDI:** SECO is the closest direct algorithmic baseline because both systems coordinate multi-satellite observation processing and delivery. ORDI differs by making per-request deadline completion under failure its primary objective and by adding online risk learning, selective fault-disjoint replication, explicit compute ledgers, and independent/correlated fault evaluation. Any implementation should be labeled `SECOAdapted` unless it faithfully reproduces SECO's original workload, observation, routing, and objective assumptions.

### Resource-Efficient In-Orbit Detection of Earth Objects (TargetFuse)

- **Authors:** Qiyang Zhang et al.
- **Venue:** IEEE INFOCOM 2024
- **Link:** <https://doi.org/10.1109/INFOCOM52122.2024.10621328>

TargetFuse is a satellite-ground collaborative object-detection system designed around constrained onboard energy, computation, and downlink bandwidth. It reduces detection error and improves bandwidth efficiency relative to simple onboard or ground-centric execution strategies.

**Relationship to ORDI:** TargetFuse supports ORDI's premise that neither onboard-only execution nor direct downlink is generally sufficient. Its main contribution is application-specific satellite-ground inference efficiency rather than failure-aware multi-satellite scheduling. ORDI can use Onboard Only and Direct Downlink as corresponding operational baselines while citing TargetFuse as motivation for cooperative placement.

### In-Orbit Processing or Not? Sunlight-Aware Task Scheduling for Energy-Efficient Space Edge Computing Networks (Phoenix)

- **Venue:** IEEE INFOCOM 2024
- **Link:** <https://doi.org/10.1109/INFOCOM52122.2024.10621268>

Phoenix studies whether tasks should be processed in orbit under satellite energy constraints and time-varying solar-energy availability. It contributes an energy-aware scheduling formulation for deciding when onboard computation is beneficial.

**Relationship to ORDI:** Phoenix emphasizes energy harvesting and the process-in-orbit decision. ORDI models energy as a scheduling cost but primarily optimizes deadline reliability under contact, queue, and fault uncertainty. Phoenix is relevant to ORDI's energy model and is a candidate for future sunlight/battery-aware extensions.

### Satellite Edge Computing for Real-Time and Very-High-Resolution Earth Observation

- **Authors:** Israel Leyva-Mayorga et al.
- **Venue:** IEEE Transactions on Communications, 2023
- **Link:** <https://doi.org/10.1109/TCOMM.2023.3296584>
- **Preprint:** <https://arxiv.org/abs/2212.12912>

This work formulates satellite mobile edge computing for real-time, very-high-resolution Earth observation. It jointly considers onboard processing, communication, energy consumption, and inter-satellite data routing to reduce the burden of downlinking raw imagery.

**Relationship to ORDI:** It provides a strong communications-level foundation for ORDI's joint compute-and-delivery model. ORDI adds online arrivals, compute contention, explicit deadlines, adaptive splitting, and failure-aware redundancy.

## Deadline-aware edge scheduling

### Dependent Task Scheduling and Offloading for Minimizing Deadline Violation Ratio in Mobile Edge Computing Networks

- **Authors:** Shumei Liu et al.
- **Venue:** IEEE Journal on Selected Areas in Communications, 2023
- **Link:** <https://doi.org/10.1109/JSAC.2022.3233532>

This work schedules dynamically arriving dependent tasks in mobile edge networks with deadline-violation ratio as the primary reliability objective. It models task dependencies and develops scheduling, migration, and merging mechanisms for applications represented as directed acyclic graphs.

**Relationship to ORDI:** It supports deadline-miss ratio as an algorithm-neutral primary metric rather than relying only on average latency. ORDI applies the deadline-reliability objective to orbital contact opportunities, satellite compute queues, result delivery, and fault-tolerant replication.

### TODG: Distributed Task Offloading with Delay Guarantees for Edge Computing

- **Authors:** Sheng Yue, Ju Ren, Nan Qiao, Yongmin Zhang, Hongbo Jiang, Yaoxue Zhang, and Yuanyuan Yang
- **Venue:** IEEE Transactions on Parallel and Distributed Systems, 2022
- **Link:** <https://doi.org/10.1109/TPDS.2021.3123535>
- **Preprint:** <https://arxiv.org/abs/2101.02772>

TODG studies distributed online computation offloading with heterogeneous tasks, stochastic communication resources, and hard delay constraints. It decomposes a long-term stochastic optimization problem into slot-level decisions and provides delay guarantees and an optimality analysis.

**Relationship to ORDI:** TODG is useful theoretical precedent for distributed online decisions and hard-delay feasibility. ORDI operates over predictable but intermittent orbital contacts, jointly reserves compute and routes, and uses deadline pruning to avoid work that cannot complete on time.

## Adaptive task replication and fault tolerance

### Task Replication for Vehicular Edge Computing: A Combinatorial Multi-Armed Bandit Approach

- **Authors:** Yuxuan Sun, Jinhui Song, Sheng Zhou, Xueying Guo, and Zhisheng Niu
- **Venue:** IEEE GLOBECOM 2018
- **Preprint:** <https://arxiv.org/abs/1807.05718>

This paper learns the delay performance of dynamic vehicular edge workers and sends a task to multiple workers to improve completion latency and service reliability. Its combinatorial multi-armed-bandit formulation balances exploration with exploitation in a changing network.

**Relationship to ORDI:** It is direct precedent for learning helper quality and using selective replication rather than a fixed worker. ORDI replaces vehicular proximity with predicted orbital contacts and extends the decision to deadline feasibility, spatial splitting, routing, and correlated satellite fault domains.

### Distributed Task Replication for Vehicular Edge Computing: Performance Analysis and Learning-Based Algorithm

- **Authors:** Yuxuan Sun, Sheng Zhou, and Zhisheng Niu
- **Venue:** IEEE Transactions on Wireless Communications, 2021
- **Preprint:** <https://arxiv.org/abs/2002.08833>

This work analyzes how replica count affects delay and failure probability in dynamic edge systems. It derives a near-optimal replica count from task arrival rate, worker density, service capacity, and erasure probability, and shows that excessive replication can overload compute queues and perform worse than a smaller replication factor.

**Relationship to ORDI:** This is the strongest theoretical motivation for ORDI's limited, adaptive redundancy. ORDI similarly charges replication cost and avoids full replication, but learns fault-domain outcomes online and requires backup placements to remain feasible over time-varying compute and contact resources.

## Adjacent satellite-network systems and methodology

### StarryNet: Empowering Researchers to Evaluate Futuristic Integrated Space and Terrestrial Networks

- **Venue:** USENIX NSDI 2023
- **Link:** <https://www.usenix.org/conference/nsdi23/presentation/lai-zeqi>

StarryNet provides large-scale emulation for integrated satellite-terrestrial networks. It recreates time-varying LEO topology and network behavior to support repeatable evaluation of protocols and applications.

**Relationship to ORDI:** StarryNet is not a competing scheduler, but it establishes an evaluation standard beyond abstract graph simulation. ORDI currently uses Basilisk/BSK-RL for orbital dynamics and its own compute/contact ledger. Cross-validating network behavior or replaying ORDI placements in an emulator such as StarryNet would strengthen external validity.

### Celestial: Virtual Software System Testbeds for the LEO Edge

- **Venue:** ACM/IFIP Middleware 2022
- **Link:** <https://doi.org/10.1145/3528535.3531517>

Celestial provides virtual testbeds for evaluating software systems deployed across moving LEO satellite constellations. It combines orbital dynamics with virtualized application execution and changing network conditions.

**Relationship to ORDI:** Celestial is relevant as an independent emulation environment for validating scheduling decisions and application-level timing. ORDI's simulator is more specialized to tiled Earth-observation workloads, resource reservations, and fault injection.

## Summary of overlap and gap

| Work | Orbit-aware | Compute placement | Hard deadlines | Task splitting | Adaptive replication | Learned failure risk | Correlated fault domains | End-to-end ground delivery |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Orbital Edge Computing | Yes | Conceptual | Limited | Pipeline | No | No | No | Yes |
| Krios | Yes | Yes | No | No | Service placement | No | No | Availability-oriented |
| LEOEdge | Yes | Yes | Latency-oriented | Model adaptation | No | No | No | Yes |
| SECO | Yes | Yes | Real-time objective | Cooperative processing | No | No | No | Yes |
| TargetFuse | Yes | Satellite/ground | Latency-oriented | Application-specific | No | No | No | Yes |
| Phoenix | Yes | Yes | Timing constraints | No | No | No | No | Yes |
| JSAC deadline scheduling | No | Yes | Yes | DAG tasks | No | No | No | Edge delivery |
| Vehicular task replication | Mobility-aware | Yes | Completion deadline | No | Yes | Worker-delay learning | No | First-result completion |
| **ORDI** | **Yes** | **Yes** | **Yes** | **1/2/4-way spatial** | **Yes** | **Yes** | **Yes** | **Yes** |

The table should be treated as a high-level taxonomy, not as a claim that prior systems omit every secondary mechanism. Final paper language should be checked against each full paper.

## Defensible ORDI positioning

Replication, deadline-aware scheduling, and satellite edge computing are each established independently. ORDI should therefore avoid claiming novelty for any one of these ideas in isolation.

A defensible contribution statement is:

> ORDI is an online scheduler for LEO Earth-observation inference that jointly selects spatial split width, compute helpers, contact-aware delivery routes, and fault-disjoint backups under absolute end-to-end deadlines. Unlike satellite schedulers centered on latency, energy, or nominal mobility, ORDI learns failure-domain risk from execution outcomes and admits redundancy only when its marginal reliability benefit justifies its compute and communication cost.

The novelty must be supported experimentally by showing that:

- ORDI improves deadline completion over SECO and conventional execution paths.
- Selective replication outperforms full and random replication at comparable resource cost.
- Fault-disjoint placement matters under correlated plane outages.
- Online risk learning adapts across fault intensities without access to the injector's rate.
- Dynamic splitting and queue/contact awareness contribute independently of replication.

## Suggested related-work narrative

Orbital edge-computing systems establish the value of processing sensor data within a constellation rather than downlinking all raw observations. Orbital Edge Computing characterizes the physical constraints of computational nanosatellites, while Krios and LEOEdge provide orchestration and satellite-ground inference mechanisms for moving, heterogeneous LEO resources. Recent INFOCOM systems—including SECO, TargetFuse, and Phoenix—jointly optimize subsets of observation coverage, compute placement, routing, energy, and delivery latency. These systems primarily address nominal mobility and resource efficiency rather than deadline reliability under stochastic and correlated execution failures.

Deadline-aware mobile-edge schedulers optimize violation probability under dynamic arrivals and constrained communication, but do not model predictable orbital contacts or satellite fault domains. Separately, learning-based replication in vehicular edge computing shows that selective redundancy can reduce tail latency and failure probability, while excessive replication creates queue contention. ORDI combines these lines of work: it performs deadline-feasible split, placement, and routing decisions over an orbital contact graph and selectively adds fault-disjoint backups using online failure-risk estimates.

## Core bibliography for an INFOCOM submission

The following papers should be treated as mandatory citations:

1. Denby and Lucia, **Orbital Edge Computing**, ASPLOS 2020.
2. Zhai et al., **SECO**, INFOCOM 2024.
3. Bhosale et al., **Krios**, SoCC 2024.
4. Yao et al., **LEOEdge**, IEEE JSAC 2025.
5. Zhang et al., **TargetFuse**, INFOCOM 2024.
6. **Phoenix**, INFOCOM 2024.
7. Leyva-Mayorga et al., **Satellite Edge Computing for Real-Time and Very-High-Resolution Earth Observation**, IEEE Transactions on Communications 2023.
8. Liu et al., **Dependent Task Scheduling and Offloading for Minimizing Deadline Violation Ratio**, IEEE JSAC 2023.
9. Sun et al., **Task Replication for Vehicular Edge Computing**, GLOBECOM 2018.
10. Sun et al., **Distributed Task Replication for Vehicular Edge Computing**, IEEE Transactions on Wireless Communications 2021.
