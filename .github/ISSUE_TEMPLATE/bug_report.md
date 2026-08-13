---
name: Bug report
about: Something in the service didn't behave the way the README or code says it should
title: ""
labels: bug
---

**What happened**
A clear description of the behavior you saw.

**What you expected**
What the README, docstring, or a test led you to expect instead.

**Reproduction**
Minimal steps or request payload to reproduce it:

```bash
curl -X POST localhost:8000/api/v1/optimize/routes -d '{ ... }'
```

**Response / traceback**
The actual response body, or the full traceback if it's a 500. If it's a
routing-quality issue rather than a crash, include the `telemetry` block from
the plan — it tells us which code path actually ran.

**Environment**
- HivePath version / commit: `git rev-parse HEAD`
- Python version: `python --version`
- OS:
- Installed extras (`ml`, `vlm`, neither):

**Anything else**
Config, `.env` values with secrets redacted, or anything else that might be
relevant.
