"""Frozen head067 repeat on previously seen DEVELOPMENT8..11, never confirmation."""
import copy
import hashlib
import sys
from pathlib import Path


FROZEN = {"block": 4, "dose": 1, "family": "branch_scale", "head": 3,
          "id": "broad0-067", "mask": "all", "pulse": 2, "rank": 1,
          "sign": -1, "site": "attention_preproj"}


def validate_repeat(episodes, candidates):
    if not episodes or len(set(episodes)) != len(episodes) or any(e not in (8, 9, 10, 11) for e in episodes):
        raise ValueError("Only unique predeclared seen DEVELOPMENT8..11")
    if candidates != [FROZEN]:
        raise ValueError("Repeat must preserve exactly frozen head067; no changed dose/site/operator")


def augment_protocol(value):
    value = copy.deepcopy(value)
    value.update(phase="seen_development_frozen067_repeat_v1", held_data_opened=False,
                 interpretation="Exploratory repeatability on seen development states; not untouched confirmation or prespecified combined significance",
                 prior_exposure={"all_8_11": "Original baseline packaging, initial H6 physical truth and old P3 rank/time fixed-action diagnostics",
                                 "10": "Also previous physical and Sonar diagnostics",
                                 "late_contexts": "8/9 captured;10 failed preserved;11 not attempted",
                                 "current_candidate_search": "All512 candidate search and selection used0..7, not8..11"},
                 primary="Native ever-success; final-success/reward/final-distance and matched sham comparisons secondary; all12 episodes retained")
    value["source_sha256"][Path(__file__).name] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return value


def run():
    import protocol
    import run_residual_search_full_v1 as original
    previous_validate, previous_write = original.validate_design, protocol.write_json_atomic
    def writer(path, value):
        if Path(path).name == "protocol.json" and value.get("phase") == "adaptive_development_full99":
            # Mutate this run's reporting dictionary so the final report contains
            # the same frozen declaration as its initial protocol receipt.
            enriched = augment_protocol(value)
            value.clear(); value.update(enriched)
        return previous_write(path, value)
    original.validate_design = validate_repeat
    protocol.write_json_atomic = writer
    try:
        original.main()
    finally:
        original.validate_design = previous_validate
        protocol.write_json_atomic = previous_write


if __name__ == "__main__":
    run()
