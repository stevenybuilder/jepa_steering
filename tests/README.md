# Tests

`unit/` contains the portable CPU suite used by CI. Its subdirectories follow the
code's subjects: models, interventions, planning, experiments, runtime, and
analysis. Tests cover numerical behavior, statistical summaries, protocol checks,
source identities, and figure generation.

`integration/` contains checks that are selected explicitly. Some require the
complete pinned upstream checkout, fitted banks, simulator dependencies, or
private archived inputs. Running the unit suite does not establish full
integration coverage.

## Run the CI suite locally

Install the project with `pip install -e '.[analysis,test]'`. A few analysis tests
verify the exact source bytes used by the archived experiments. Restore that
small source fixture before running them:

```bash
mkdir -p vendor/frozen-study
git archive 40fa529afc782165f5f610e6dac436931e429772 src/offline_study analysis/mechanism \
  | tar -x -C vendor/frozen-study
export JEPA_FROZEN_SOURCE_ROOT="$PWD/vendor/frozen-study/src/offline_study"
python -m pytest tests/unit
```

The crop-parity test also needs the pinned JEPA-WM `transforms.py` fixture;
[CI](../.github/workflows/tests.yml) records its upstream commit and path. No
checkpoint, cloud account, or GPU is needed for the unit suite.

Run an integration module only after restoring its documented dependencies, for
example `python -m pytest tests/integration/planning/test_planning_contract.py`.
Tests for removed routing, training-engineering, and advancement workflows were
retired with those modules. Existing experimental results were not changed.
