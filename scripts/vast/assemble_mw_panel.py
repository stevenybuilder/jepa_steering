#!/usr/bin/env python3
"""Assemble one MetaWorld component-extension panel tree from per-host tarballs.

Input: a directory of tarballs collected from the boxes (results/..., engineering/...).
Output: <out>/results/<task>/<arm>/shard-NN and <out>/engineering/<device-uuid>/<task>/...
Rules: macOS '._*' sidecars and '*.MOVED' markers are dropped; the same results shard or
engineering directory appearing in several tarballs must be byte-identical (else error);
prints the host/device map per (task, arm, slot) and the missing units. No science here.
"""
import argparse, hashlib, json, re, shutil, sys, tarfile, tempfile
from pathlib import Path

TASKS = ("reach", "reach-wall")
ARMS = ("native", "visual_only", "action_condition_only", "joint")
SLOTS = range(8)
SKIP = ("reach-shard00.tgz",)          # NV0 salvage: engineering receipt never preserved -> unusable
PREFIXES = ("mw-", "reach-slot", "reachwall-")   # MetaWorld tarballs only


def dir_digest(path):
    h = hashlib.sha256()
    for f in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(str(f.relative_to(path)).encode()); h.update(hashlib.sha256(f.read_bytes()).digest())
    return h.hexdigest()


def clean(root):
    for p in list(root.rglob("._*")) + list(root.rglob("*.MOVED")) + list(root.rglob(".DS_Store")):
        p.unlink()


def place(src, dst, kind, origin, seen):
    digest = dir_digest(src)
    key = str(dst)
    if dst.exists():
        if seen.get(key, (None,))[0] != digest:
            raise SystemExit(f"CONFLICT {kind} {dst.name}: {origin} differs from {seen.get(key, ('?', '?'))[1]}")
        return "dup-identical"
    shutil.copytree(src, dst)
    seen[key] = (digest, origin)
    return "placed"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tarballs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--manifest", type=Path)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}")
    (a.out / "results").mkdir(parents=True); (a.out / "engineering").mkdir()
    seen, shard_origin, eng_origin = {}, {}, {}
    tars = sorted(p for p in a.tarballs.iterdir() if p.name.endswith(".tgz") and p.name.startswith(PREFIXES) and p.name not in SKIP)
    for tgz in tars:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            with tarfile.open(tgz) as t:
                for m in t.getmembers():
                    if m.name.startswith(("/", "..")) or "/../" in m.name:
                        raise SystemExit(f"unsafe member {m.name} in {tgz.name}")
                t.extractall(tmp)
            clean(tmp)
            for task in TASKS:
                for arm in ARMS:
                    d = tmp / "results" / task / arm
                    for shard in sorted(d.glob("shard-*")) if d.is_dir() else []:
                        if not shard.is_dir():
                            continue
                        dev = json.loads((shard / "protocol.json").read_text())["device_uuid"]
                        state = place(shard, a.out / "results" / task / arm / shard.name, "shard", tgz.name, seen)
                        shard_origin.setdefault((task, arm, shard.name), []).append((tgz.name, dev, state))
                eng = tmp / "engineering"
                for dev_dir in sorted(eng.glob("*")) if eng.is_dir() else []:
                    src = dev_dir / task
                    if src.is_dir():
                        # drivers named the dir after nvidia-smi ("GPU-<uuid>"); the frozen analysis looks up
                        # engineering_root/<launch device_uuid>/<task>, and device_uuid() is the bare CUDA uuid.
                        bare = json.loads((src / "report.json").read_text())["device_uuid"]
                        if dev_dir.name not in (bare, "GPU-" + bare):
                            raise SystemExit(f"engineering dir {dev_dir.name} does not match its report device_uuid {bare} ({tgz.name})")
                        state = place(src, a.out / "engineering" / bare / task, "engineering", tgz.name, seen)
                        eng_origin.setdefault((bare, task), []).append((tgz.name, dev_dir.name, state))
    # report
    missing, rows = [], []
    for task in TASKS:
        for arm in ARMS:
            for s in SLOTS:
                name = f"shard-{s:02d}"
                o = shard_origin.get((task, arm, name))
                if not o:
                    missing.append(f"{task}/{arm}/{name}"); continue
                dev = o[0][1]
                has_eng = (a.out / "engineering" / dev / task / "report.json").exists()
                rows.append((task, arm, s, dev, ",".join(x[0] for x in o), "eng-ok" if has_eng else "ENGINEERING-MISSING"))
    print(f"tarballs: {len(tars)}  shards placed: {len(shard_origin)}/64  engineering dirs: {len(eng_origin)}")
    for r in rows:
        print("  %-10s %-22s slot %d  %s  %s  %s" % r)
    print("MISSING (%d): %s" % (len(missing), " ".join(missing) if missing else "none"))
    bad = [r for r in rows if r[5] != "eng-ok"]
    print("SHARDS WITHOUT ENGINEERING RECEIPT DIR (%d): %s" % (len(bad), " ".join(f"{r[0]}/{r[1]}/shard-{r[2]:02d}@{r[3]}" for r in bad) or "none"))
    if a.manifest:
        a.manifest.write_text(json.dumps({"tarballs": [t.name for t in tars], "shards": {"/".join((k[0], k[1], k[2])): v for k, v in shard_origin.items()},
                                          "engineering": {"/".join(k): v for k, v in eng_origin.items()}, "missing": missing}, indent=1))
    sys.exit(1 if missing or bad else 0)


if __name__ == "__main__":
    main()
