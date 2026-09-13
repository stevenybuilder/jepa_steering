# Source before every receiving or scientific command, passing extracted root.
campaign_root=${1:?Pass absolute campaign root}
export PATH="$campaign_root/python/bin:$PATH"
export PYTHONPATH="$campaign_root/src:$campaign_root/vendor/jepa-wms"
export JEPAWM_DSET="$campaign_root/data" JEPAWM_LOGS="$campaign_root/logs"
export JEPAWM_HOME="$campaign_root/vendor" JEPAWM_CKPT="$campaign_root/checkpoints"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl SDL_VIDEODRIVER=dummy
export MUJOCO_PY_MUJOCO_PATH="$campaign_root/preparation-runtime/mujoco210"
export MUJOCO_PY_FORCE_CPU=1 D4RL_SUPPRESS_IMPORT_ERROR=1
export LD_LIBRARY_PATH="$MUJOCO_PY_MUJOCO_PATH/bin:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
