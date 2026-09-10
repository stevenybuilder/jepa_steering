import json, sys
d = json.load(open(sys.argv[1]))
r = lambda x: None if x is None else round(x, 4)
for goal in ("progress", "brake"):
    res = d["results"][goal]; u = res["unsteered"]
    print(goal, "unsteered margin", {k: r(v["mean"]) for k, v in u["margin"].items()}, "safe", {k: r(v["mean"]) for k, v in u["safe_choice"].items()})
    for name, e in res.items():
        c = e["config"]
        if c["kind"] not in ("main", "euclidean_twin", "random_matched", "ungated_on", "wrong_block"):
            continue
        mc = {k: (r(v["margin_change"]["mean"]), r(v["margin_change"]["ci_low"]), r(v["margin_change"]["ci_high"])) for k, v in e["vs_unsteered"].items()}
        dm = e["did"][e["did_primary"]["contrast"]]["margin"]
        sc = {k: r(v["mean"]) for k, v in e["safe_choice"].items()}
        print(f"  {c['source'][:22]:>22s} {c['mode']:>10s} b={c['beta']} {c['kind']:>14s} site={c['sites'][0]} safe={sc} marginD={mc} DiD_margin={r(dm['mean'])} [{r(dm['ci_low'])},{r(dm['ci_high'])}] p={r(dm['sign_flip_p'])} editRMS={r((e.get('activation_edit_relative_rms') or {}).get('mean'))}")
