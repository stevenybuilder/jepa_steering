#!/usr/bin/env bash
set -euo pipefail

# Read-only search. This never creates or rents an instance.
vastai search offers \
  'reliability>=0.99 num_gpus=1 gpu_ram>=24 compute_cap>=800 disk_space>=100 inet_down>=200' \
  --storage 100 \
  --order 'dph' \
  --limit 20 \
  --raw
