"""Download the complete pinned author pool; token is read from stdin, never logged."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import sys
import urllib.request
import urllib.parse

DATA_REV = "6116f042ae7ae4c8e3f1fd2f194f432615664182"
MODEL_REV = "9b9c41ef249466630dbf1a20e78391865d07b3b9"
MODEL_SHA = "c3297772c7af4e84c28bd0c0937e22398a888848305014bb3ab3f1508a310bd8"
ROOT = Path("/workspace/jepa-runtime/fixed-response-assets-20260908-v1")


class CredentialSafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and urllib.parse.urlsplit(req.full_url).netloc != urllib.parse.urlsplit(newurl).netloc:
            redirected.remove_header("Authorization")
        return redirected


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    token = sys.stdin.readline().strip()
    if not token.startswith("hf_"):
        raise ValueError("Provide the existing authorized HF token via stdin")
    headers = {"Authorization": "Bearer " + token}
    def metadata(kind, revision):
        request = urllib.request.Request("https://huggingface.co/api/" + kind +
            "/facebook/jepa-wms/revision/" + revision + "?blobs=true", headers=headers)
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.load(response)
        if result["sha"] != revision:
            raise ValueError("HF revision changed")
        return result
    data = metadata("datasets", DATA_REV)
    model = metadata("models", MODEL_REV)
    files = []
    for repo_type, revision, siblings in (("datasets/", DATA_REV, data["siblings"]),
                                          ("", MODEL_REV, model["siblings"])):
        for row in siblings:
            name = row["rfilename"]
            if ((repo_type and name.startswith("metaworld/data/") and name.endswith(".parquet")) or
                    (not repo_type and name == "jepa_wm_metaworld.pth.tar")):
                files.append({"path": name, "bytes": row["size"], "sha256": row["lfs"]["sha256"],
                    "url": "https://huggingface.co/" + repo_type + "facebook/jepa-wms/resolve/" + revision + "/" + name})
    if len(files) != 127 or next(r for r in files if r["path"].endswith(".tar"))["sha256"] != MODEL_SHA:
        raise ValueError("Expected exact 126-shard data release and pinned model")
    ROOT.mkdir(parents=True, exist_ok=True)
    expected = json.dumps({"data_revision": DATA_REV, "model_revision": MODEL_REV, "files": files}, indent=2)
    manifest = ROOT / "DOWNLOAD_MANIFEST.json"
    if manifest.exists() and manifest.read_text() != expected:
        raise ValueError("Existing input manifest differs")
    manifest.write_text(expected)
    def download(row):
        path = ROOT / row["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
                raise ValueError("Existing source file changed: " + row["path"])
            return
        temporary = path.with_suffix(path.suffix + ".partial")
        request = urllib.request.Request(row["url"], headers=headers)
        opener = urllib.request.build_opener(CredentialSafeRedirect())
        with opener.open(request, timeout=180) as response, temporary.open("wb") as output:
            while chunk := response.read(1024**2):
                output.write(chunk)
        if temporary.stat().st_size != row["bytes"] or digest(temporary) != row["sha256"]:
            raise ValueError("Source download checksum failed: " + row["path"])
        temporary.replace(path)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for i, _ in enumerate(pool.map(download, files), 1):
            if i % 16 == 0:
                print(json.dumps({"verified_files": i, "total": len(files)}), flush=True)
    receipt = {"status": "all_126_parquet_and_checkpoint_bytes_verified", "files": len(files),
               "bytes": sum(row["bytes"] for row in files), "manifest_sha256": digest(manifest),
               "fit_or_evaluation_has_run": False}
    (ROOT / "DONE.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
