# Contributing

## Branching Model

This repository uses a GitFlow-lite workflow.

- `main`
  Production-ready history only. Merge here from `release/*` and `hotfix/*`.
- `develop`
  Default integration branch for day-to-day development.
- `feature/<topic>`
  Branch from `develop` and open the PR back to `develop`.
- `release/<version>`
  Branch from `develop` when preparing a release. Stabilization-only changes go here, and the PR targets `main`.
- `hotfix/<topic>`
  Branch from `main` for urgent production fixes. The PR targets `main`, then the fix must be merged back into `develop`.

## Pull Request Targets

- `feature/*` -> `develop`
- `release/*` -> `main`
- `hotfix/*` -> `main`

After a `release/*` or `hotfix/*` PR lands on `main`, merge the resulting `main` commit back into `develop` so the integration branch stays current.

## CI Expectations

GitHub Actions is split into three validation layers:

- Fast Tests
  Coverage-backed unit and integration suite (`make test-cov`) on every supported push and PR.
- Sample Graphs
  In-process graph runner plus the in-process API smoke flow (`make test-samples`).
- SeekDB Validation
  Real backend checkpoint smoke plus live HTTP e2e coverage (`make test-seekdb`) on PRs to `develop` or `main`, and on pushes to `develop`, `main`, `release/*`, and `hotfix/*`. CI runs this in a Docker matrix across `seekdb`, `oceanbase`, and `mysql`, while local quick verification should prefer embedded SeekDB via `scripts/seekdb_embed_launcher.py`.

Keep local commands aligned with CI when possible:

- `make test`
- `make test-cov`
- `make test-samples`
- `make test-e2e`
- `make test-seekdb`
