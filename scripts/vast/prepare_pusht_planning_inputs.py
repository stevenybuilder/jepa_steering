"""Stage the pinned full Push-T archive and released checkpoint, CPU/network only."""
import concurrent.futures
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import time
import urllib.request
import urllib.parse
import zipfile

ROOT = Path("/workspace/jepa-runtime/pusht-planning-assets-20260908-v2")
DATA_REV = "6116f042ae7ae4c8e3f1fd2f194f432615664182"
MODEL_REV = "9b9c41ef249466630dbf1a20e78391865d07b3b9"
ASSETS = (("datasets", DATA_REV, "pusht/pusht_noise.zip",
           "442f5dee246edf670964ed7bdecd248683cd6d00580fa0e4d458abb53f92da08"),
          ("models", MODEL_REV, "jepa_wm_pusht.pth.tar",
           "9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb"))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for b in iter(lambda: stream.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    token = sys.stdin.readline().strip()
    if not token.startswith("hf_"):
        raise ValueError("Pass the existing authorized HF token via stdin; never log or save it")
    headers = {"Authorization": "Bearer " + token}
    class SafeRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, response_headers, newurl):
            result = super().redirect_request(req, fp, code, msg, response_headers, newurl)
            if result is not None and urllib.parse.urlsplit(req.full_url).netloc != urllib.parse.urlsplit(newurl).netloc:
                result.remove_header("Authorization")
            return result
    opener = urllib.request.build_opener(SafeRedirect())
    ROOT.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        files = []
        for kind, revision, filename, expected in ASSETS:
            request = urllib.request.Request(f"https://huggingface.co/api/{kind}/facebook/jepa-wms/revision/{revision}?blobs=true", headers=headers)
            with opener.open(request, timeout=60) as r:
                metadata = json.load(r)
            entry = next(s for s in metadata["siblings"] if s["rfilename"] == filename)
            if metadata["sha"] != revision or entry["lfs"]["sha256"] != expected:
                raise ValueError("Pinned official asset changed")
            prefix = "datasets/" if kind == "datasets" else ""
            files.append({"kind": kind, "revision": revision, "filename": filename,
                "sha256": expected, "bytes": entry["size"],
                "url": f"https://huggingface.co/{prefix}facebook/jepa-wms/resolve/{revision}/{filename}"})
        write("protocol.json", {"role": "official_pusht_input_staging_no_evaluation", "assets": files,
            "script_sha256": digest(Path(__file__)), "scientific_outcomes_accessed": False})

        def download(entry):
            folder = ROOT / "downloads" / entry["kind"]
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / Path(entry["filename"]).name
            partial = target.with_suffix(target.suffix + ".partial")
            prior = ROOT.with_name("pusht-planning-assets-20260908-v1") / "downloads" / entry["kind"] / target.name
            if prior.is_file() and prior.stat().st_size == entry["bytes"] and digest(prior) == entry["sha256"]:
                shutil.copyfile(prior, target)
                return target
            request = urllib.request.Request(entry["url"], headers=headers)
            with opener.open(request, timeout=120) as r, partial.open("xb") as out:
                shutil.copyfileobj(r, out, 4 << 20)
            if partial.stat().st_size != entry["bytes"] or digest(partial) != entry["sha256"]:
                raise ValueError("Downloaded asset differs from pinned metadata")
            partial.replace(target)
            print(json.dumps({"verified_asset": entry["filename"], "bytes": entry["bytes"]}), flush=True)
            return target

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            archive, checkpoint = list(pool.map(download, files))
        data = ROOT / "data"
        data.mkdir()
        with zipfile.ZipFile(archive) as z:
            members = z.infolist()
            names = [m.filename for m in members]
            expanded = sum(m.file_size for m in members)
            for member in members:
                name = PurePosixPath(member.filename)
                if (name.is_absolute() or ".." in name.parts or "\\" in member.filename or
                        stat.S_ISLNK(member.external_attr >> 16)):
                    raise ValueError("Unsafe archive path")
            if len(names) != len(set(names)) or expanded > 64 * 1024**3:
                raise ValueError("Duplicate archive member or unexpected expansion")
            if shutil.disk_usage(data).free < expanded + 8 * 1024**3:
                raise ValueError("Insufficient disk space with 8GiB reserve")
            required = {f"pusht_noise/{pool}/" + name for pool in ("train", "val") for name in
                        ("states.pth", "velocities.pth", "rel_actions.pth", "seq_lengths.pkl")}
            required |= {f"pusht_noise/val/obses/episode_{i:03d}.mp4" for i in range(21)}
            required.add("pusht_noise/train/obses/episode_10810.mp4")
            if not required <= set(names):
                raise ValueError("Official archive structure differs: " + repr(sorted(required - set(names))[:8]))
            manifest = {}
            for member in members:
                z.extract(member, data)
                if not member.is_dir():
                    path = data / member.filename
                    manifest[member.filename] = {"bytes": path.stat().st_size, "sha256": digest(path)}
            write("files.json", manifest)
        write("report.json", {"status": "full_pinned_pusht_inputs_verified_and_extracted",
            "protocol_sha256": digest(ROOT / "protocol.json"), "files_sha256": digest(ROOT / "files.json"),
            "archive_sha256": digest(archive), "checkpoint_sha256": digest(checkpoint),
            "files": len(manifest), "extracted_bytes": expanded, "seconds": time.monotonic() - started,
            "fresh_confirmation": False, "planning_executed": False, "loader_parity_checked": False})
        write("DONE.json", {"report_sha256": digest(ROOT / "report.json")})
        print(json.dumps({"status": "inputs_ready_not_planning_complete", "files": len(manifest)}), flush=True)
    except Exception as exc:
        write("FAILED.json", {"error": str(exc), "planning_executed": False})
        raise


if __name__ == "__main__":
    main()
