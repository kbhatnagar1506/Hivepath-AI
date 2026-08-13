<div align="center">

<img src="assets/images/logohivepath.png" alt="HivePath AI" width="140" />

# HivePath AI

### The routing engine that treats accessibility as a first-class constraint — not an afterthought.

[![CI](https://github.com/kbhatnagar1506/Hivepath-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/kbhatnagar1506/Hivepath-AI/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](hivepath-ai/pyproject.toml)
[![Tests: 190 passing](https://img.shields.io/badge/tests-190%20passing-brightgreen.svg)](hivepath-ai/docs/TESTING.md)
[![Coverage: 86%](https://img.shields.io/badge/coverage-86%25-brightgreen.svg)](hivepath-ai/docs/TESTING.md#coverage-by-module)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-blueviolet.svg)](CONTRIBUTING.md)

</div>

---

Every logistics optimizer on the market solves the same objective: minimize
distance, minimize time, minimize cost. That objective has a blind spot it
never has to answer for — **it doesn't know which stops are hard to reach, so
it drops them first, silently, every single time capacity runs short.**
HivePath is built around closing that blind spot at the objective-function
level, not with a dashboard warning bolted on afterward.

Given a depot, a fleet, and a list of stops, it plans routes under time
windows and capacity limits — same category as any commercial VRP engine. The
difference shows up exactly when it matters: when the fleet **can't serve
everything.** A standard optimizer sheds whatever is most awkward to reach.
HivePath makes awkward stops *more expensive to skip*, so the plan sheds easy
ones instead.

**Measured: +25 points of service rate for hard-to-reach stops, at zero cost
to overall throughput.** [See how, below](#the-numbers).

```bash
git clone https://github.com/kbhatnagar1506/Hivepath-AI.git
cd Hivepath-AI/hivepath-ai
pip install -e ".[dev]"
pytest                        # 190 tests, zero configuration required
python -m hivepath             # → http://localhost:8000/docs
```

---

## How it stacks up

|  | Distance-only VRP<br>*(typical commercial stack)* | HivePath AI |
|---|---|---|
| Objective function | Minimize distance / time / cost | Minimize distance / time / cost **+ accessibility-weighted drop cost** |
| When capacity runs short | Drops whichever stop is costliest to reach — usually the hard address | Drops the *easiest* stop first; hard-to-reach stops are kept in the plan |
| Accessibility data | Not modeled | 0–100 score per stop, from vision-scored Street View imagery or supplied directly |
| No credentials configured | Often won't start, or silently no-ops a feature | **Starts and solves routes with zero credentials.** Degraded features report their own fallback |
| Distance source transparency | Rarely exposed | Every plan reports `matrix_source`: `haversine` or `google_maps` — never claims one while using the other |
| Disruption response | Full manual re-plan | `POST /incidents` blocks a stop and replans in one call, warm-started from the previous plan |
| Failure mode on partial infeasibility | Often a hard error | 200 OK with `ok: false` and a reason, or a partial plan with `dropped_stop_ids` — never a crash for a well-formed problem |
| Test suite | Frequently absent or unverifiable from outside | 190 tests, 86% coverage, runnable by anyone who clones the repo |
| License | Often proprietary | MIT — fork it, run it, change it |
| Impact claims | Marketing figures | `scripts/benchmark_impact.py` — run it yourself, seed included |

---

## The numbers

`hivepath-ai/scripts/benchmark_impact.py` runs the same solver twice per
scenario — accessibility weighting on, then off — so the only variable is the
feature itself. 59 scenarios, 14–22 stops, fleet capacity deliberately set to
72% of total demand to force real trade-offs:

```bash
cd hivepath-ai && python scripts/benchmark_impact.py --trials 60
```

**Equity** — service rate for hard-to-reach stops (access score < 35):

| | Service rate |
|---|---|
| Accessibility weighting **on** | **100%** |
| Accessibility weighting **off** | ~75% |
| Difference | **+25 points** |

Overall service rate is 73.5% either way — the fleet is capacity-limited, so
this is not "serve more stops." It is **the same number of stops, chosen
differently.** With weighting on, no hard-to-reach stop was dropped in any of
the 59 scenarios; the plan drops easier ones instead.

**Efficiency** — optimized plan vs. serving the manifest in submitted order,
which is what happens without an optimizer:

| | Result |
|---|---|
| Mean optimized distance | ~41 km |
| Mean unoptimized distance | 58.0 km |
| **Reduction** | **~27%** |
| CO₂ avoided per run | ~14 kg (diesel) |

> These are synthetic scenarios measuring solver behaviour, not field-observed
> delivery outcomes. Full methodology, the reproducibility caveat, and the
> honest limits of what this proves are in
> [`hivepath-ai/README.md`](hivepath-ai/README.md#the-numbers) — read it
> before citing these numbers anywhere.

---

## Why this exists

A route optimizer minimizing distance or time treats every stop as
interchangeable. It isn't. A stop with no legal kerbside parking, a blocked
loading bay, or stairs-only access costs more to serve — so when capacity runs
short, a distance-minimizing objective drops it **first, and every time.**
Nobody configures that outcome. It falls out of the math by default, and it
compounds silently, one route at a time, forever, unless something in the
objective function pushes back.

Those aren't randomly distributed addresses, either. They skew toward older
housing stock, dense blocks, and buildings that never got a loading dock —
which means the "efficient" route is systematically less efficient at serving
exactly the people who already have the fewest alternatives.

HivePath inverts that default at the source: inside the objective function
itself, not in a report generated after the fact.

---

## What's actually in this repository

The project is `hivepath-ai/` — that's where the code, tests, and docs live.
Everything else is context, kept rather than deleted:

```
Hivepath-AI/
├── hivepath-ai/            The service. Start here.
│   ├── src/hivepath/       190 tests' worth of typed, tested Python
│   ├── tests/              pytest, hermetic, 86% coverage
│   ├── docs/                ARCHITECTURE.md and TESTING.md — the deep dives
│   ├── scripts/             benchmark_impact.py and ML training scripts
│   └── legacy/               superseded backend + 21 old scripts, kept with
│                             a migration map, not deleted
├── assets/                  logo, architecture diagrams, and dashboard
│                             screenshots from an earlier presentation of
│                             this project — historical, not current output
├── legacy-frontend/         a single static HTML page from an even earlier
│                             version of this project (then called SwarmAura)
├── LICENSE                  MIT
├── CONTRIBUTING.md          how to set up, test, and submit a PR
├── SECURITY.md              how to report a vulnerability
└── CODE_OF_CONDUCT.md
```

**[→ hivepath-ai/README.md](hivepath-ai/README.md)** — the full project README:
API reference, the accessibility penalty formula and its guarantees,
degradation behavior without credentials, and the complete benchmark
methodology.

**[→ hivepath-ai/docs/ARCHITECTURE.md](hivepath-ai/docs/ARCHITECTURE.md)** —
every design decision, stated as the defect it replaced. The request
lifecycle traced module by module, what's inside the OR-Tools model, and what
was deliberately left out of scope.

**[→ hivepath-ai/docs/TESTING.md](hivepath-ai/docs/TESTING.md)** — the
isolation model, a full test inventory, the coverage table, and ten
regressions walked through individually.

---

## Contributing

Issues and PRs are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) first —
the short version is: name bug-fix tests after the bug, keep the domain typed,
never let a feature fail silently, and don't add a comment that just restates
the code above it.

## License

[MIT](LICENSE) — fork it, run it, ship it, change what you disagree with.
