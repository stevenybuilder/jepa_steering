# Archive provenance

The reset places the complete Git tree from commit
`bf8d9b47214718be3e851573bde68f84b69eba7d` under
`archive/2026-09-07-workspace/`.

Original root tree SHA: `fa6f500f9abd8385b68ae1d1404a9b7a569387c4`.
The archived subtree must have exactly that SHA. Reusing the Git tree preserves
all tracked file contents and relative directory structure without rewriting results.

The archive includes the original README and ignore rules. Historical absolute local
paths, remote-host references, and links to previously uncommitted archives are left
unchanged for provenance; moving them does not make those resources available.

Heavy trajectory tensors, checkpoints, most videos, and previously ignored archives
were not included in the original Git commit and therefore cannot be recovered by
this operation. No claim about their current cloud-storage availability is made.

The active model_loader.py is an explicitly retained copy of the former
scripts/geometry_map/model_loader.py. Other archived code is not imported at runtime.
An isolated branch and pull request make the reset reviewable without rewriting history.
