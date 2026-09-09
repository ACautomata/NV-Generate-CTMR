# NV-Generate-CTMR — Agent Guide

## Agent skills

### Issue tracker

Issues are tracked in this repo's GitHub Issues (`ACautomata/NV-Generate-CTMR`), managed via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default five-role triage vocabulary (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` at the repo root + `docs/adr/`. See `docs/agents/domain.md`.

### Server deployment

Deploying to the shared gauss GPU server, or launching training / validation there (environment setup, run directories, claiming idle GPUs without disturbing other users)? See [`skills/deploy_gauss.md`](skills/deploy_gauss.md).

### Experiment releases

After any experiment finishes (training, inference, evaluation), publish its outputs — checkpoints, raw records, typical example images, one-off scripts — to a GitHub Release instead of committing them. See [`skills/publish_experiment.md`](skills/publish_experiment.md).
