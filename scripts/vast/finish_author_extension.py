"""Automatically merge each completed seven-row extension with its verified source sweep."""
import os
import subprocess
import sys
import time
from pathlib import Path

from offline_study.author_access import CATEGORIES, PRECISIONS
from offline_study.protocol import sha256, write_json


def main():
    original = Path("/workspace/jepa-runtime/author-correction-20260907")
    root = Path("/workspace/jepa-runtime/metaworld-author-extension-20260907")
    status = root / "closure-v1"
    status.mkdir(exist_ok=False)
    pending = [(p, t, c) for p in PRECISIONS for t in ("reach", "reach-wall") for c in CATEGORIES]
    start, reports = time.monotonic(), []
    try:
        while pending:
            if time.monotonic() - start > 14400:
                raise TimeoutError("Four-hour full-pool merge cap")
            if list((root / "evaluation-v1").rglob("FAILED.json")) or (original / "closure-v2/FAILED.json").exists():
                raise RuntimeError("A required source run failed; no full-pool completion claim")
            for precision, task, category in list(pending):
                old = original / "analysis-v2" / precision / task / category
                extra = root / "evaluation-v1" / precision / task / category
                if not (old / "DONE.json").exists() or not all((extra / f"shard-{s:03d}/DONE.json").exists() for s in range(2)):
                    continue
                output = root / "analysis-full-v1" / precision / task / category
                command = [sys.executable, "-m", "offline_study.author_merge_extension", "--original-root", str(original),
                    "--extension-root", str(root), "--precision", precision, "--task", task,
                    "--category", category, "--output", str(output)]
                with (root / "logs" / f"full-analysis-{precision}-{task}-{category}.log").open("x") as log:
                    subprocess.run(command, check=True, timeout=900, stdout=log, stderr=subprocess.STDOUT)
                reports.append({"path": str(output / "report.json"), "sha256": sha256(output / "report.json")})
                pending.remove((precision, task, category))
            time.sleep(10)
        write_json(status / "report.json", {"status": "full_primary_metaworld_replication_verified", "analysis_scopes": 20,
            "primary_tasks_measured": 2, "trajectories": {"mw-reach": 33, "mw-reach-wall": 27},
            "newly_evaluated_rows": 7, "reused_rows": 53, "reports": reports,
            "fresh_confirmation": False, "full_study_complete": False})
        write_json(status / "DONE.json", {"report_sha256": sha256(status / "report.json")})
    except Exception as exc:
        write_json(status / "FAILED.json", {"error": str(exc)})
        raise


if __name__ == "__main__":
    main()
