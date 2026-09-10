"""Read-only selector for the already-complete, four-scope navigation geometry panel.

Returns names compatible with snapshot_live_results.HASH_FILES and tar -T. This
does not copy, archive, upload, import scientific code, or recompute statistics.
The self-contained function may be sent via inspect.getsource to remote stdin.
Production pins were read back on Indiana on 2026-09-08; no current repo source
is substituted. Recheck the selected manifest after any separately authorized
transfer, then use backup_results_to_google.verify_archive for full readback.
Raw reports embed the evaluation-source hash. Original aggregate schemas do not
embed an analysis-source hash: its existing snapshot is independently pinned,
not presented as a newly proven aggregate-to-producer receipt.
"""


def select_files(root, *, _test_bindings=None):
    """Select exact frozen evidence; private binding injection is for CPU fixtures.

    The CLI and production callers must use the pinned defaults. Large outcome
    arrays are only hashed; completed aggregate content is never used to select
    an arm, precision, shard, or endpoint. Paths embedded in original receipts
    remain anchored to their original /workspace/jepa-runtime location.
    """
    import hashlib
    import json
    import stat
    from pathlib import Path

    bindings = {
        "sources": {
            "navigation-evaluation-code-v1": (
                "abb50325c78027ecba580c9eb5b222ac0735513b28f2e5cde12681f406b24c68",
                "b9e744749e1a3a288860ba0ee40acd18f1e396a4a44939ecc6aa9579fd0361ca", 123),
            "navigation-analysis-code-v1": (
                "f2ff6ef8dfa860286987f071f3213c019aa200158bfd5bce38edfb5b1827ec9c",
                "a4bc409cb1e87f91291e06c117383301df8c2044e9ad91900c27b1c87f4ccb19", 80),
        },
        "cohorts": {
            "wall": "6d7482f29ba232cb84032e46ba3d9f35251cbfabfc4110f6123a45bcd69e56af",
            "pointmaze": "57b9a3d181e4508fdc0c14a036144f9fbe80a3ea9dfb847df4655bb15b89b8b4",
        },
        # protocol, fit receipt, aggregate report; every precision is mandatory.
        "scopes": {
            "bfloat16/wall": (
                "fdefd9ece07075863cc7dca1461932f2cb0eea8370b00cff0308164cb311b5eb",
                "d5f71f92bfd9d69a0de9383f15d1181edd35f5aa88d71fb3cc919649854cde7c",
                "643247c9860778d46c7deea714e045de15d4a401d8bc1b96a506f405d2def9bd"),
            "bfloat16/pointmaze": (
                "deb8968c73984379ace32729bc7f5b275f5aad4307efcb7b1a2e97e18fc2c089",
                "87e8cd02f8e372a29aa804b2bd43762ac78b1aaf31507c56985cac20318df091",
                "34797270e44abf8937457060b8fda437fe02bb46726e1acf7e6b49e4beb0bfde"),
            "float32/wall": (
                "09d8124e8458f0a8d013a38f4438244c50dcd80dc5cea7ffb1b38127cee56a73",
                "03a3e5157c92dd0caa240f026a604766551379d8f946a402115c0049a4dbfa13",
                "3f5315532e50c29bd794996dfc29f416e00cf7b717e0e4031cee6be736687077"),
            "float32/pointmaze": (
                "7e453e87d02e78b37715c777f83de52f4b7254d621f9fdd34221b0223bc861fb",
                "40e3a3e9bb7666e653532db786794ac26f6f4e885ac30025e93e1df156ceb57b",
                "550fc44b126621b221d5abca79e718e96953de0455a4b4c2947ac4ce585b6d95"),
        },
        "parity": {
            "bfloat16": "901b307f3cd571f182eb6745b402fd3f97b2c910e1d8ff20a8288ccd9dd0416c",
            "float32": "1c4519c7a41da17e29737c3275bb40c4c4bba8d7b1f6ab0a3713a3ee947b770e",
        },
    }
    if _test_bindings is not None:
        bindings = _test_bindings
    root = Path(root).absolute()
    chosen, hashes = set(), {}
    arms = {"native", "zero_dose", "equal_anchor_linear", "cubic",
            "projected_cubic", "reflected_curvature", "matched_random"}
    category = "action_response_geometry"
    complete = "verified_author_corrected_development_complete"

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def safe(path, directory=False):
        relative = path.relative_to(root)
        require(not root.is_symlink(), "Symlink root")
        cursor = root
        for part in relative.parts:
            require(part not in (".", "..") and not part.startswith(".")
                    and not any(c in part for c in "\r\n\0"), "Unsafe member name")
            cursor = cursor / part
            require(not cursor.is_symlink(), "Symlink member or ancestor: " + str(cursor))
        require(path.is_dir() if directory else path.is_file(), "Missing member: " + str(path))
        if not directory:
            require(stat.S_ISREG(path.stat().st_mode), "Nonregular member")
            require(path.stat().st_size <= 128 << 20, "Unexpectedly large member")
        return str(relative)

    def sha(path):
        name = safe(path)
        chosen.add(name)
        if name not in hashes:
            h = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(4 << 20), b""):
                    h.update(block)
            hashes[name] = h.hexdigest()
        return hashes[name]

    def read(path):
        sha(path)
        return json.loads(path.read_text())

    def matches(record, expected, label):
        require(all(record.get(k) == v for k, v in expected.items()), "Binding mismatch: " + label)

    def no_failure(path):
        safe(path, directory=True)
        require(not (path / "FAILED.json").exists() and not (path / "FAILED.json").is_symlink(),
                "Failed evidence root: " + str(path))

    for label in ("navigation-evaluation-code-v1", "navigation-analysis-code-v1"):
        base = root / label
        safe(base, directory=True)
        harness, tree, count = hashlib.sha256(), hashlib.sha256(), 0
        for path in sorted(base.rglob("*")):
            require(not path.is_symlink(), "Symlink in source snapshot")
            relative = path.relative_to(base)
            if "__pycache__" in relative.parts or path.suffix == ".log":
                continue
            if path.is_dir():
                continue
            # Explicit source allowlist: no credentials, datasets or checkpoints.
            require(path.suffix in (".py", ".sh") and not any(
                token in path.name.lower() for token in ("credential", "secret", ".pem", ".key")),
                "Unexpected or credential-like source member")
            sha(path)
            data = path.read_bytes()
            tree.update(str(relative).encode() + b"\0" + data)
            if relative.parent == Path("src/offline_study") and path.suffix == ".py":
                harness.update(path.name.encode() + b"\0" + data)
            count += 1
        require((harness.hexdigest(), tree.hexdigest(), count) == tuple(bindings["sources"][label]),
                "Original source snapshot changed: " + label)

    for task in ("wall", "pointmaze"):
        base = root / "navigation-offline-cohorts-20260907-v1" / task
        no_failure(base)
        cohort = read(base / "cohort.json")
        frozen = read(base / "FROZEN.json")
        require(sha(base / "cohort.json") == bindings["cohorts"][task], "Cohort changed")
        matches(cohort, {"task": task, "fresh_confirmation": False}, "cohort role")
        matches(frozen, {"cohort_sha256": bindings["cohorts"][task], "model_runs": 0,
                        "confirmation_authorized": False,
                        "input_files_sha256": sha(base / "input_files.json")}, "cohort freeze")
        matches(cohort, {"source_input_files_sha256": sha(base / "input_files.json"),
                         "source_manifest_sha256": sha(base / "source_manifest.json")}, "cohort inputs")

    for precision in ("bfloat16", "float32"):
        for task in ("wall", "pointmaze"):
            scope = precision + "/" + task
            protocol_hash, receipt_hash, aggregate_hash = bindings["scopes"][scope]
            fit = root / "navigation-fits-20260907-v1" / scope / category
            raw = root / "navigation-comparisons-20260907-v1" / scope / category
            aggregate = root / "navigation-analysis-20260907-v1" / scope / category
            for path in (fit, raw, aggregate):
                no_failure(path)
            require(sha(fit.parent / "PARITY.json") == bindings["parity"][precision], "Parity changed")
            require(sha(fit / "protocol.json") == protocol_hash, "Frozen protocol changed")
            require(sha(fit / "fit_receipt.json") == receipt_hash, "Frozen fit receipt changed")
            receipt, protocol = read(fit / "fit_receipt.json"), read(fit / "protocol.json")
            common = {"task": task, "category": category, "precision": precision,
                      "cohort_sha256": bindings["cohorts"][task]}
            matches(receipt, {**common, "status": "author_train_fit_only_complete"}, "fit")
            matches(read(fit / "DONE.json"), {"status": "author_protocol_frozen",
                    "cohort_sha256": common["cohort_sha256"], "protocol_sha256": protocol_hash,
                    "fit_receipt_sha256": receipt_hash,
                    "operator_bank_sha256": sha(fit / "operator_bank.pt")}, "fit DONE")
            matches(protocol, {"status": "frozen", "category": category,
                    "evaluation_split": "development", "fit_receipt_sha256": receipt_hash,
                    "checkpoint_sha256": receipt["checkpoint_sha256"]}, "protocol")
            matches(protocol["author_validation"], {"cohort_sha256": common["cohort_sha256"],
                                                   "precision": precision}, "protocol cohort")
            require(len(protocol["arms"]) == 7 and {a["name"] for a in protocol["arms"]} == arms,
                    "Frozen seven-arm registry changed")
            shared = {**common, "protocol_sha256": protocol_hash, "fit_receipt_sha256": receipt_hash,
                      "protected_outcomes_accessed": False, "fresh_confirmation": False}
            require({p.name for p in raw.glob("shard-*")} == {f"shard-{i}" for i in range(8)},
                    "Missing or extra geometry shard")
            inputs = {}
            for index in range(8):
                shard = raw / f"shard-{index}"
                no_failure(shard)
                done = read(shard / "DONE.json")
                require(done.get("status") == "author_offline_shard_complete", "Incomplete shard DONE")
                for name in ("report", "window_metrics", "selection", "protocol", "mechanism_diagnostics"):
                    path = shard / (name + ".json")
                    digest = sha(path)
                    require(done.get(name + "_sha256") == digest, "Shard checksum mismatch: " + name)
                    inputs[str(Path("/workspace/jepa-runtime") / path.relative_to(root))] = digest
                require(sha(shard / "protocol.json") == protocol_hash, "Shard protocol changed")
                matches(read(shard / "report.json"), {**shared, "status": "author_offline_shard_complete",
                        "checkpoint_sha256": receipt["checkpoint_sha256"], "shard_index": index,
                        "shard_count": 8, "zero_dose_identity": True, "native_instrumentation_fidelity": True,
                        "source_sha256": bindings["sources"]["navigation-evaluation-code-v1"][0]}, "shard")
            require(sha(aggregate / "report.json") == aggregate_hash, "Aggregate report changed")
            matches(read(aggregate / "DONE.json"), {"status": complete,
                    "report_sha256": aggregate_hash}, "aggregate DONE")
            matches(read(aggregate / "report.json"), {**shared, "status": complete,
                    "input_sha256": inputs, "rollout_trajectories": 192 if task == "wall" else 200,
                    "prefix_rollouts_per_arm": 4224 if task == "wall" else 24400}, "aggregate")
    require(sum((root / name).stat().st_size for name in chosen) <= 2 << 30, "Archive exceeds 2 GiB raw bound")
    return sorted(chosen)


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="Existing evidence root; no network operations")
    print(json.dumps(select_files(parser.parse_args().root), sort_keys=True))
