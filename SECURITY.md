# Security Policy

## Reporting a vulnerability

Please use **[GitHub's private vulnerability reporting](https://github.com/kbhatnagar1506/Hivepath-AI/security/advisories/new)**
(Security tab → *Report a vulnerability*) rather than a public issue. If that's
not available to you, open an issue with as little detail as possible and ask
for a private channel.

Include what you found, how to reproduce it, and what you think the impact is.
There's no bug bounty here — this is a personal open-source project — but
you'll be credited in the fix, and I'll turn around a response quickly.

## Supported versions

This project has no version branches yet; only the latest commit on `main` is
supported. If you find something, check it still reproduces against `main`
before reporting.

## Scope

In scope:
- Anything in `hivepath-ai/src/hivepath/` — the running service.
- Dependency vulnerabilities that are actually reachable from this codebase's
  usage of them, not just present in `requirements.txt`.
- Credential handling: how `.env` values flow through `Settings`, whether a
  secret can leak into a log line, a response body, or a traceback.

Out of scope:
- `hivepath-ai/legacy/` — dead code, kept for reference, not imported by
  anything that runs. See [`legacy/README.md`](hivepath-ai/legacy/README.md).
- The Next.js dashboard under `integrated_dashboard/` — a separate,
  unmaintained frontend not wired to the current API.
- Findings that require an attacker to already control your `.env` file or
  your deployment environment.

## What this project does about it structurally

A few properties worth knowing if you're evaluating this for your own
deployment, not just reporting a bug:

- **Every credential is a `pydantic.SecretStr`**, read from exactly one place
  ([`config.py`](hivepath-ai/src/hivepath/config.py)). It cannot appear in a
  `repr()`, and there's a test (`test_secrets_are_not_exposed_in_repr`) that
  fails the build if that ever regresses.
- **No feature fails open on a missing credential.** Accessibility analysis
  without an OpenAI key returns `503`, not a fabricated neutral score dressed
  up as a real assessment. See
  [`docs/ARCHITECTURE.md`](hivepath-ai/docs/ARCHITECTURE.md#6-accessibility-assessment-policy-and-the-scale-conversion).
- **`.env` is gitignored**, and only `.env.example` — containing no real
  values — is tracked. If you're forking or deploying this, never commit your
  own `.env`.

## A note on this repository's own history

An earlier, pre-restructure version of this codebase had a Google Maps API key
hardcoded directly in three source files. It has been removed from the working
tree and purged from git history via `git filter-repo` before this repository
was made public, so it should not be recoverable from any clone made after
that point. If you're working from a mirror or fork taken *before* the purge,
assume that key is exposed in its history and should be treated as
compromised regardless of whether it has been rotated on the Google Cloud
side — don't take that on faith from this document or any other; verify
directly in the Google Cloud Console if it matters to you.

If you find another hardcoded credential anywhere in this codebase — purged
history or not — please report it through the channel above. That's exactly
the class of issue this policy exists for.
