"""Read-only public DROID source catalogue; no downloads, model calls or selection.

Use the same trajectory.h5 existence rule as JEPA-WM's generate_droid_paths.py.
Retain success AND failure directories. An enumeration does not identify the
paper's 8,000-example manifest or authorize a substituted training population.
"""
import argparse
import json
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path, PurePosixPath

from .protocol import sha256, write_json


PREFIX = "robotics/droid_raw/1.0.1/"
API = "https://storage.googleapis.com/storage/v1/b/gresearch/o"


def validate_object(row):
    name = row["name"]
    path = PurePosixPath(name)
    if (not name.startswith(PREFIX) or path.name != "trajectory.h5" or ".." in path.parts or
            not str(row["generation"]).isdigit() or int(row["size"]) <= 0 or not row.get("md5Hash")):
        raise ValueError("Unexpected public DROID object metadata")
    return dict(row)


def fetch_page(token=None):
    args = {"prefix": PREFIX, "matchGlob": PREFIX + "**/trajectory.h5", "maxResults": 1000,
            "fields": "items(name,generation,size,md5Hash),nextPageToken"}
    if token:
        args["pageToken"] = token
    url = API + "?" + urllib.parse.urlencode(args)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read(4 * 1024**2 + 1)
            if len(data) > 4 * 1024**2:
                raise ValueError("Unexpected catalogue page size")
            return json.loads(data)
        except (TimeoutError, urllib.error.URLError):
            if attempt == 3:
                raise
            time.sleep(attempt + 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    write_json(args.output / "protocol.json", {"role": "public_DROID_catalogue_provenance_only",
        "bucket": "gresearch", "prefix": PREFIX, "file_eligibility": "trajectory.h5 exists, as native generator",
        "success_only_filter": False, "source_sha256": sha256(Path(__file__)),
        "bounds": {"seconds": 900, "pages": 1000, "objects": 150000},
        "downloaded_videos": 0, "model_calls": 0, "fitting_or_training_population_selected": False,
        "paper_8000_manifest_reconstructed": False, "confirmation_authorized": False})
    try:
        rows, tokens, pages, token = {}, set(), {}, None
        for page in range(1000):
            if time.monotonic() - started > 900:
                raise TimeoutError("Bounded catalogue audit timed out; preserve partial pages")
            response = fetch_page(token)
            path = args.output / f"page-{page:04d}.json"
            write_json(path, response)
            pages[path.name] = sha256(path)
            for row in response.get("items", []):
                item = validate_object(row)
                if item["name"] in rows:
                    raise ValueError("Duplicate or inconsistent pagination object")
                rows[item["name"]] = item
            if len(rows) > 150000:
                raise ValueError("Unexpected corpus expansion, audit scope exceeded")
            progress = {"pages": page + 1, "objects": len(rows), "seconds": time.monotonic() - started}
            write_json(args.output / "progress.json", progress)
            print(json.dumps(progress), flush=True)
            token = response.get("nextPageToken")
            if not token:
                break
            if token in tokens:
                raise ValueError("Repeated catalogue cursor")
            tokens.add(token)
        else:
            raise ValueError("Catalogue page bound reached before EOF")
        if not rows:
            raise ValueError("Empty public source catalogue")
        ordered = [rows[name] for name in sorted(rows)]
        write_json(args.output / "catalogue.json", ordered)
        labs = Counter(row["name"][len(PREFIX):].split("/")[0] for row in ordered)
        roles = Counter(row["name"][len(PREFIX):].split("/")[1] for row in ordered)
        write_json(args.output / "report.json", {"status": "public_DROID_source_catalogue_complete",
            "protocol_sha256": sha256(args.output / "protocol.json"),
            "catalogue_sha256": sha256(args.output / "catalogue.json"), "page_files_sha256": pages,
            "source_trajectory_objects": len(rows), "by_lab": dict(labs), "by_source_folder": dict(roles),
            "sum_trajectory_h5_bytes_not_video_bytes": sum(int(row["size"]) for row in ordered),
            "seconds": time.monotonic() - started, "paper_8000_manifest_reconstructed": False,
            "downloaded_videos": 0, "model_calls": 0, "fitting_or_training_population_selected": False})
        write_json(args.output / "DONE.json", {"report_sha256": sha256(args.output / "report.json")})
    except Exception as exc:
        write_json(args.output / "FAILED.json", {"error": str(exc), "partial_pages_preserved": True})
        raise


if __name__ == "__main__":
    main()
