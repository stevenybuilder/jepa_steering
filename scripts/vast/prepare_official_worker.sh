#!/usr/bin/env bash
set -euo pipefail

# Prepare one already-rented Vast worker for either MetaWorld or Push-T. This script
# does not rent or stop instances and never prints the Hugging Face token.
if [[ $# -ne 1 || ( "$1" != "metaworld" && "$1" != "pusht" ) ]]; then
  echo "Usage: $0 metaworld|pusht" >&2
  exit 2
fi
dataset="$1"
workspace_root="${JEPA_WORKSPACE_ROOT:-/workspace}"
project_dir="${workspace_root}/jepa_steering"
runtime_root="${JEPA_DURABLE_ROOT:-${workspace_root}/jepa-runtime}"
secret_env="${JEPA_SECRET_ENV:-/root/.jepa.env}"

if ! command -v ffmpeg >/dev/null || [[ "$(ffmpeg -version 2>/dev/null | sed -n '1p')" != *"ffmpeg version 7."* ]]; then
  if [[ ! -x /opt/conda/bin/conda ]]; then
    echo "FFmpeg 7 is missing and /opt/conda/bin/conda is unavailable." >&2
    exit 1
  fi
  /opt/conda/bin/conda install -y -c conda-forge ffmpeg=7.1
fi

bash "${workspace_root}/jepa-bootstrap.sh"
source "${workspace_root}/jepa-python/bin/activate"
if [[ ! -f "${secret_env}" ]]; then
  echo "Missing local worker secret file: ${secret_env}" >&2
  exit 1
fi
set -a
source "${secret_env}"
set +a
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set in the worker secret file." >&2
  exit 1
fi

export JEPAWM_DSET="${runtime_root}/data"
export JEPAWM_LOGS="${runtime_root}/runs"
export JEPAWM_HOME="${project_dir}/vendor"
export JEPAWM_CKPT="${runtime_root}/data/checkpoints"
mkdir -p "${JEPAWM_DSET}" "${JEPAWM_LOGS}" "${JEPAWM_CKPT}"
python "${project_dir}/vendor/jepa-wms/setup_macros.py"
python "${project_dir}/vendor/jepa-wms/src/scripts/download_data.py" \
  --dataset "${dataset}" --dataset-root "${JEPAWM_DSET}"

if [[ "${dataset}" == "metaworld" ]]; then
  checkpoint_name="mw_jepa-wm.pth.tar"
  checkpoint_url="https://dl.fbaipublicfiles.com/jepa-wms/mw_jepa-wm.pth.tar"
  checkpoint_sha="c3297772c7af4e84c28bd0c0937e22398a888848305014bb3ab3f1508a310bd8"
  data_root="${JEPAWM_DSET}/Metaworld/data"
else
  checkpoint_name="pt_jepa-wm.pth.tar"
  checkpoint_url="https://dl.fbaipublicfiles.com/jepa-wms/pt_jepa-wm.pth.tar"
  checkpoint_sha="9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb"
  data_root="${JEPAWM_DSET}/pusht_noise"
fi
checkpoint_path="${JEPAWM_CKPT}/${checkpoint_name}"
if [[ ! -f "${checkpoint_path}" ]] || \
    [[ "$(sha256sum "${checkpoint_path}" | awk '{print $1}')" != "${checkpoint_sha}" ]]; then
  partial_path="${checkpoint_path}.partial.$$.download"
  trap 'rm -f "${partial_path}"' EXIT
  curl --fail --location --retry 5 --retry-all-errors \
    --output "${partial_path}" "${checkpoint_url}"
  if [[ "$(sha256sum "${partial_path}" | awk '{print $1}')" != "${checkpoint_sha}" ]]; then
    echo "Checkpoint checksum mismatch after download." >&2
    exit 1
  fi
  mv "${partial_path}" "${checkpoint_path}"
  trap - EXIT
fi

if [[ ! -d "${data_root}" ]]; then
  echo "Downloaded dataset root is missing: ${data_root}" >&2
  exit 1
fi
echo "Prepared ${dataset}: data_root=${data_root} checkpoint=${checkpoint_path} sha256=${checkpoint_sha}"
