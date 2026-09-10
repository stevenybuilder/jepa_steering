"""Hard compute stop for new50531754 only if verified closeout has not finished.

Never deletes an unarchived disk. Normal completion destroys the fully archived
disposable worker earlier through finish_decision_diagnostic.py.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2] / "artifacts/offline_study/decision-closeout-20260910-v1"
VAST = "/Users/stevenyang/.local/bin/vastai"
DEADLINE = datetime(2026, 9, 11, 3, 0, tzinfo=timezone.utc).timestamp()


def main():
    while time.time() < DEADLINE:
        if (ROOT / "DONE.json").exists():
            return
        time.sleep(min(30, max(.1, DEADLINE - time.time())))
    instances = json.loads(subprocess.check_output([VAST, "show", "instances", "--raw"], text=True, timeout=90))
    owned = [row for row in instances if row["id"] == 50531754]
    if not owned:
        return
    if owned[0]["label"] != "jepa-decision-bounded-20260910" or owned[0]["geolocation"] != "Michigan, US":
        raise ValueError("Backstop refuses a mismatched instance identity")
    result = json.loads(subprocess.check_output([VAST, "stop", "instance", "50531754", "--raw"], text=True, timeout=90))
    with (ROOT / "BUDGET_STOP.json").open("x") as output:
        json.dump({"utc": datetime.now(timezone.utc).isoformat(), "instance": 50531754,
                   "stop_response": result, "disk_destroyed": False,
                   "reason": "hard compute deadline; preserve source storage if closeout failed"}, output, indent=2)


if __name__ == "__main__":
    main()
