# Active geometry-map code

Governing plan: [Geometry Map Experiment](<../../Geometry Map Experiment.md>). Primary model/task: frozen JEPA-WM, MetaWorld Reach-Wall.

| Stage | Files |
|---|---|
| Data, labels, and splits | `discover_metaworld_shards.py`, `build_manifest.py`, `protocol.py` |
| Model loading and capture | `model_loader.py`, `canary_capture.py`, `capture_shard.py`, `validate_capture.py` |
| Geometry screens | `screen_pooled.py`, `screen_nonlinear.py`, `screen_spatial.py`, `screen_emergence_geometry.py` |
| Matched action patching | `causal_action_patching.py`, `summarize_action_patching.py` |
| On-policy replay bank | `collect_on_policy_bank.py`, `validate_on_policy_bank.py`, `build_snapshot_manifest.py` |
| Later Push-T replication assets | `prepare_pusht_assets.py` |

The [frozen bank manifest](../../docs/manifests/geometry-map-reach-wall-v1/on_policy_bank_manifest.json) separates development episodes 0–11 from held paired-evaluation episodes 12–23. Protected outcomes cannot select the site, subspace, operator, or dose.

Use cached analysis and snapshot forks to validate planner effects before complete steered episodes. A decoded variable or latent-transfer effect alone does not establish better planning or task success. HMM routing and the broader Frankenstein modules remain conditional.
