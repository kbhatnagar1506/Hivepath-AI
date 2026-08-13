## What this changes and why

<!-- One or two sentences. If it fixes a bug, describe the failure mode, not
     just "fixes bug". -->

## How it was verified

- [ ] `pytest` passes locally
- [ ] `ruff check src tests` is clean
- [ ] New/changed behavior has a test — and if it's a bug fix, the test is
      named after what would have failed before the fix
- [ ] If this touches network-calling code, the test mocks at the
      `hivepath.integrations` boundary rather than reaching the network

## Does this change the API contract?

<!-- New field, new endpoint, changed response shape, changed default
     behavior. If yes, describe what a caller of the current API would
     experience. If no, delete this section. -->

## Anything the reviewer should look at closely

<!-- Optional. A tricky invariant, a tradeoff you're not fully sure about,
     a place you'd want a second opinion. -->
