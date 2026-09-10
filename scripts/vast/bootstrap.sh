#!/usr/bin/env bash
set -euo pipefail

# Run inside an already-authorized Vast SSH instance. It does not rent compute,
# download study datasets/checkpoints, or start a benchmark.
workspace_root="${JEPA_WORKSPACE_ROOT:-/workspace}"
project_dir="${workspace_root}/jepa_steering"
environment_dir="${workspace_root}/jepa-python"
runtime_root="${JEPA_DURABLE_ROOT:-${workspace_root}/jepa-runtime}"
project_branch="${JEPA_STEERING_BRANCH:-offline-study-reset}"
project_url="${JEPA_STEERING_URL:-https://github.com/stevenybuilder/jepa_steering.git}"
vendor_url="https://github.com/facebookresearch/jepa-wms.git"
vendor_commit="13cf1d9c7e476f53c17714d2e0f1dc239a883ce0"

if ! command -v git >/dev/null || ! command -v python3 >/dev/null; then
  echo "The instance image must provide git and Python 3."
  exit 1
fi
if ! command -v ffmpeg >/dev/null || [[ "$(ffmpeg -version | sed -n '1p')" != *"ffmpeg version 7."* ]]; then
  echo "FFmpeg 7.x is required by upstream; install it in the image before dataset execution."
  exit 1
fi
if ! command -v uv >/dev/null; then
  python3 -m pip install --no-cache-dir 'uv==0.8.15'
fi

python_base_dir="${workspace_root}/jepa-python-base"
python_interpreter="${JEPA_PYTHON_INTERPRETER:-}"
if [[ -z "${python_interpreter}" && -x "${python_base_dir}/bin/python" ]]; then
  python_interpreter="${python_base_dir}/bin/python"
fi
if [[ -z "${python_interpreter}" ]] && command -v python3.10 >/dev/null; then
  python_interpreter="$(command -v python3.10)"
fi
if [[ -z "${python_interpreter}" && -x /opt/conda/bin/conda ]]; then
  # Several Vast hosts have fast conda mirrors but cannot reliably reach the
  # GitHub-hosted python-build-standalone archive used by `uv python install`.
  # Prefer the image's package channel so bootstrap is not coupled to that route.
  /opt/conda/bin/conda create -y -p "${python_base_dir}" python=3.10 pip
  python_interpreter="${python_base_dir}/bin/python"
fi
if [[ -z "${python_interpreter}" ]]; then
  uv python install 3.10
  python_interpreter="$(uv python find 3.10)"
fi
if [[ ! -x "${environment_dir}/bin/python" ]]; then
  uv venv --python "${python_interpreter}" "${environment_dir}"
fi
source "${environment_dir}/bin/activate"

if [[ -f "${project_dir}/pyproject.toml" && ! -d "${project_dir}/.git" ]]; then
  echo "Using pre-staged project source snapshot at ${project_dir}."
elif [[ ! -d "${project_dir}/.git" ]]; then
  git clone --branch "${project_branch}" --single-branch "${project_url}" "${project_dir}"
else
  git -C "${project_dir}" fetch origin "${project_branch}"
  git -C "${project_dir}" switch "${project_branch}"
  git -C "${project_dir}" merge --ff-only "origin/${project_branch}"
fi

if [[ -f "${project_dir}/vendor/jepa-wms/pyproject.toml" && ! -d "${project_dir}/vendor/jepa-wms/.git" ]]; then
  if [[ ! -f "${project_dir}/vendor/jepa-wms/.source-commit" ]] || \
      [[ "$(<"${project_dir}/vendor/jepa-wms/.source-commit")" != "${vendor_commit}" ]]; then
    echo "Pre-staged upstream snapshot does not match pinned commit ${vendor_commit}." >&2
    exit 1
  fi
  echo "Using pre-staged upstream snapshot at pinned commit ${vendor_commit}."
elif [[ ! -d "${project_dir}/vendor/jepa-wms/.git" ]]; then
  git clone --filter=blob:none "${vendor_url}" "${project_dir}/vendor/jepa-wms"
  git -C "${project_dir}/vendor/jepa-wms" fetch origin "${vendor_commit}"
  git -C "${project_dir}/vendor/jepa-wms" checkout --detach "${vendor_commit}"
else
  git -C "${project_dir}/vendor/jepa-wms" fetch origin "${vendor_commit}"
  git -C "${project_dir}/vendor/jepa-wms" checkout --detach "${vendor_commit}"
fi

uv pip install -e "${project_dir}/vendor/jepa-wms"
uv pip install -e "${project_dir}"
export JEPAWM_DSET="${JEPAWM_DSET:-${runtime_root}/data}"
export JEPAWM_LOGS="${JEPAWM_LOGS:-${runtime_root}/runs}"
export JEPAWM_HOME="${JEPAWM_HOME:-${project_dir}/vendor}"
export JEPAWM_CKPT="${JEPAWM_CKPT:-${runtime_root}/data/checkpoints}"
mkdir -p "${JEPAWM_DSET}" "${JEPAWM_LOGS}" "${JEPAWM_CKPT}"
python "${project_dir}/vendor/jepa-wms/setup_macros.py"
python -c 'import sys, torch; assert sys.version_info[:2] == (3, 10); assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
cd "${project_dir}"
python -m unittest discover -s tests -v

echo "Bootstrap complete. Transfer audited data/checkpoints separately, then run the bounded benchmark commands in docs/GPU_EFFICIENCY.md."
