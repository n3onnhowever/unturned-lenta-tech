# Artifact Policy

This public repository is intentionally clean.

## Included

- Frontend source.
- Backend source.
- Docker Compose runtime definition.
- Scripts and docs.
- Only required runtime weights:
  - `price_tag_merged_internal_best.pt`
  - `FSRCNN_x4.pb`

## Excluded

- Raw videos.
- Runtime uploads.
- Generated frames/crops.
- Generated CSV/JSON results.
- Logs.
- SQLite databases.
- Docker cache.
- `node_modules`.
- Frontend `build` output.
- Alternative research weights not needed by Docker runtime.

## Why

The repository should be cloneable, reviewable, and safe to publish. Runtime outputs can be regenerated locally and should not be committed.
