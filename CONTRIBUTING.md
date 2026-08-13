# Contributing to HivePath AI

Thanks for considering it. This project is a routing engine with a specific
opinion — accessibility belongs in the objective function, not a dashboard
bolted on afterward — and the codebase is held to the same standard the README
claims: typed boundaries, no silent fallbacks, and every regression fixed with
a named test.

The real project lives in [`hivepath-ai/`](hivepath-ai/); this is a landing
page for a repo that also carries some non-code assets and a legacy snapshot.
All the commands below run from that directory.

## Setup

```bash
cd hivepath-ai
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env      # optional — every value has a working default
pytest                    # 190 tests, should be green with zero configuration
```

No credentials are required to develop, test, or run the service. If you want
to exercise the Google Maps or vision-model code paths, add real keys to
`.env` — see [`README.md`](hivepath-ai/README.md#configuration) for what each
one unlocks and how the service behaves without it.

## Before you open a PR

```bash
pytest                                              # must pass
ruff check src tests                                # must be clean
python -m coverage run --source=src/hivepath -m pytest && \
  python -m coverage report                         # shouldn't regress
```

All three run in CI on every PR; a red check blocks merge.

## What a good PR looks like here

**If you're fixing a bug, name the test after the bug**, not the feature. This
codebase's entire test suite grew out of real defects in the system it
replaced, and [`docs/TESTING.md`](hivepath-ai/docs/TESTING.md) explains why:
`test_penalty_is_nonzero_at_default_weight` tells you what broke and how, in a
way `test_penalties_2` never will. Put the mechanism in a one-line docstring,
not just the symptom.

**If your change touches the network**, it goes through
`hivepath.integrations`, and it's tested by mocking at that boundary — never
by making a real call in the test suite. See
`test_falls_back_when_google_client_raises` in
[`tests/test_distance.py`](hivepath-ai/tests/test_distance.py) for the shape.

**If a feature can degrade**, it should degrade visibly. A missing credential,
a missing checkpoint, an upstream API error — all of these should downgrade to
a documented fallback and say so in the response, not fail silently and not
crash. [`docs/ARCHITECTURE.md`](hivepath-ai/docs/ARCHITECTURE.md#1-design-principles)
has the reasoning; `Plan.telemetry` is the pattern to follow.

**Keep the domain typed.** Don't pass a `dict` across a module boundary where
a dataclass in `hivepath.domain` already exists. This isn't a style
preference — an untyped `dict` boundary is exactly how a past version of this
service ended up calling a solver with a keyword argument it never declared,
and nothing caught it until production.

**No comments explaining what the code does.** A comment should explain a
non-obvious *why* — a constraint, an invariant, a workaround — never restate
what's already legible from the code. If you'd delete a comment and nothing
would be lost, it shouldn't be there.

## Reporting a bug

Open an issue with:
- what you ran (request payload, preset, or command),
- what you expected,
- what actually happened, including the exact error or a `telemetry` block if
  it's a solver/routing issue.

If it's a security issue — a credential handling gap, an injection point,
anything that shouldn't wait for the public issue queue — see
[`SECURITY.md`](SECURITY.md) instead.

## Proposing a feature

Open an issue first if the change is more than a few files. This is a small,
opinionated codebase and it's easier to agree on the shape of something before
code exists than after.

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). Be direct
about the work, be decent about the person doing it.
