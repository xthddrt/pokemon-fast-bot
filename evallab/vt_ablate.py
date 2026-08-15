"""ENCODER VALUE TEST -- what is the net actually using?

Same methodology as ENCODER_AUDIT.md, retargeted from CE-on-outcome to
BRIER-on-label_p and run per band on the held-out test split.

  1. PERMUTATION IMPORTANCE BY FEATURE GROUP -- shuffle every column of one
     group across rows with ONE shared permutation (destroys the association,
     preserves the group's joint marginal), average over 3 permutations, report
     dBrier = Brier(shuffled) - Brier(intact), per band.
  2. FIRST-LAYER WEIGHT NORM per group, against a random-init control. A group
     sitting at its init norm is untrained, i.e. unused.
  3. DIRECTED SENSITIVITY PROBES on the relational blocks: edit one feature in
     a way with an unambiguous sign and check the prediction moves that way.

  python vt_ablate.py --arm enc2|old --ckpt run.pt --out abl.json
"""
import argparse
import json
import re

import numpy as np
import torch

import vt_lib as V

REPS = 3


# ----------------------------------------------------------- groups ---------
def enc2_groups():
    import enc2
    nm = enc2.DEFAULT_LAYOUT.names
    ix = lambda pat: np.array([i for i, n in enumerate(nm) if re.search(pat, n)], np.int64)
    g = {
        "mon.hp_frac": ix(r"^mon\d+\.hp_frac$"),
        "mon.maxhp": ix(r"^mon\d+\.maxhp$"),
        "mon.alive": ix(r"^mon\d+\.alive$"),
        "mon.stats": ix(r"^mon\d+\.(attack|defense|sp_atk|sp_def|speed)$"),
        "mon.status": ix(r"^mon\d+\.(status_|sleep_rest_turns)"),
        "mon.pp+disabled": ix(r"^mon\d+\.(pp_frac|disabled)\d$"),
        "mon.reveal_bits": ix(r"^mon\d+\.seen_"),
        "mon.capabilities": ix(r"^mon\d+\.cap_"),
        "mon.tera+rage": ix(r"^mon\d+\.(terastallized|times_attacked)$"),
        "mon.typing(multi-hot)": ix(r"^mon\d+\.type_(eff|own)_"),
        "side.hazards": ix(r"^side\d\.(stealth_rock|spikes|toxic_spikes|sticky_web)$"),
        "side.screens": ix(r"^side\d\.(reflect|light_screen|aurora_veil|tailwind|safeguard|mist)"),
        "side.boosts": ix(r"^side\d\.boost_"),
        "side.volatiles": ix(r"^side\d\.(vol_|dur_|substitute_hp_frac|toxic_counter)"),
        "side.move_state": ix(r"^side\d\.(lock_|just_switched_in|force_|last_move_failed|wish_|future_sight_)"),
        "side.resources": ix(r"^side\d\.(tera_available|n_fainted_revivable|best_revival_target_score|times_revived)$"),
        "REL.ko_matrix": ix(r"^rel\.ko_now_"),
        "REL.speed+priority": ix(r"^rel\.(pairspeed|active_speed|order_|prio_|revenge_|mprio)"),
        "CF.setup(+1)": ix(r"^cf_mon\d+\."),
        "CF.tera+sweep(side)": ix(r"^cf_side\d\."),
        "global.field": ix(r"^glob\."),
    }
    # COMPOSITES -- the decisive question is not one block at a time. Permutation
    # importance is not additive when blocks are redundant, and enc2's KO/setup
    # blocks are COMPUTED FROM hp, so shuffling hp alone leaves its signal intact
    # inside them. These joint shuffles are what actually separate "counting
    # resources" from "using the completed comparisons".
    seen = np.concatenate(list(g.values()))
    assert len(seen) == len(set(seen.tolist())), "groups overlap"
    missing = sorted(set(range(len(nm))) - set(seen.tolist()))
    assert not missing, "ungrouped: %s" % [nm[i] for i in missing[:10]]
    # embedding id groups: mon ids are 12 x 9 (item, ability, t1, t2, tera, m0..3)
    idg = {"emb.item": [0], "emb.ability": [1], "emb.types": [2, 3],
           "emb.tera_type": [4], "emb.moves": [5, 6, 7, 8]}
    ids = {k: np.array([s * 9 + c for s in range(12) for c in v], np.int64)
           for k, v in idg.items()}
    ids["emb.side_move_state"] = np.arange(108, 112, dtype=np.int64)
    return g, ids


def addon_groups(add):
    """The ADDITIVE arm's extra columns, split into the blocks that were added
    and then into sub-blocks, so the report can say whether each one still
    carries signal once it sits BESIDE arm A's own per-mon columns (rather than
    replacing them, which is what enc2 did).

    -> ({name: idx into the selected `am` columns}, {name: idx into `ar`})."""
    lay = V.load_addon_layout()
    mc, rc = V.addon_cols(add)
    mpos = {int(c): i for i, c in enumerate(mc)}
    rpos = {int(c): i for i, c in enumerate(rc)}
    mn, rn = lay["mon_names"], lay["rest_names"]
    mg, rg = {}, {}
    for g, (a, b) in lay["groups"].items():
        pre, name = g.split(".")
        sel = [(mpos if pre == "mon" else rpos).get(c) for c in range(a, b)]
        sel = [s for s in sel if s is not None]
        if sel:
            (mg if pre == "mon" else rg)["ADD." + name] = np.array(sel, np.int64)
    # sub-blocks of the two big ones, by column name
    def sub(names, pos, pat):
        v = [pos[i] for i, n in enumerate(names) if re.search(pat, n) and i in pos]
        return np.array(v, np.int64)
    for nm2, pat in (("ADD.ko/us->them", r"ko_now_us_to_them"),
                     ("ADD.ko/them->us", r"ko_now_them_to_us"),
                     ("ADD.speed/order+margin", r"(pairspeed|active_speed|order_)"),
                     ("ADD.speed/priority", r"(prio_|revenge_|mprio)"),
                     ("ADD.tera/2-feature", r"(tera_best_value|tera_enabled_sweep)"),
                     ("ADD.tera/aggregates", r"(best_sweep_1|free_turn|answers_to)")):
        s = sub(rn, rpos, pat)
        if len(s):
            rg[nm2] = s
    for nm2, pat in (("ADD.setup/sweep+mtk", r"cf_mon\d+\.(s1_sweep|s1_mtk)"),
                     ("ADD.setup/counts", r"cf_mon\d+\.(s1_ohko_count|s1_outspeed_count)"),
                     ("ADD.setup/move present", r"cf_mon\d+\.setup_")):
        s = sub(mn, mpos, pat)
        if len(s):
            mg[nm2] = s
    return mg, rg


def old_groups():
    mf = {  # per-mon numeric, 73 columns
        "mon.hp_frac": [0], "mon.level": [1], "mon.stats": list(range(2, 7)),
        "mon.maxhp": [7], "mon.status": list(range(8, 15)),
        "mon.boosts": list(range(15, 20)), "mon.tera+active": [20, 21],
        "mon.pp+disabled": list(range(22, 30)), "mon.sleep_rest": [30, 31],
        "mon.rage+actions+once": [32, 33, 34], "mon.weight": [35],
        "mon.typing": list(range(36, 72)), "mon.tera_stellar": [72],
    }
    sf = {  # per-side numeric, 99 columns
        "side.hazards": [0, 1, 2, 3], "side.screens": [4, 5, 6, 7, 8],
        "side.alive": [9], "side.wish+healwish+teraused": [10, 11, 12],
        "side.extras": list(range(13, 39)),
        "side.volatiles": list(range(39, 73)), "side.paradox": list(range(73, 78)),
        "side.durations": list(range(78, 91)), "side.v5_action_state": list(range(91, 99)),
    }
    gf = {"global.field": list(range(18))}
    idg = {"emb.species": [0], "emb.item": [1], "emb.ability": [2],
           "emb.moves": [3, 4, 5, 6], "emb.tera_type": [7], "emb.last_item": [8]}
    return mf, sf, gf, idg


# ------------------------------------------------------ permutation ---------
def run_perm(arm, net, meta, idx, rng, add="", drop=""):
    """-> {group: {band: dBrier}}"""
    base_batch = V.Arm(arm, add=add, drop=drop)
    b0 = base_batch.batch(idx)
    y, band = meta["label_p"][idx], meta["band"][idx]

    def brier(b):
        with torch.no_grad():
            p = np.concatenate([torch.sigmoid(net({k: v[i:i + 8192] for k, v in b.items()})).numpy()
                                for i in range(0, len(idx), 8192)])
        return {"all": float(np.mean((p - y) ** 2)),
                **{bd: float(np.mean((p[band == bd] - y[band == bd]) ** 2)) for bd in V.BANDS}}

    base = brier(b0)
    out = {"_intact": base}
    if arm == "enc2":
        num, ids = enc2_groups()
        num["COMPOSITE resources(hp+alive+status+boosts)"] = np.concatenate(
            [num[k] for k in ("mon.hp_frac", "mon.alive", "mon.status", "side.boosts")])
        num["COMPOSITE all derived (REL+CF)"] = np.concatenate(
            [num[k] for k in ("REL.ko_matrix", "REL.speed+priority", "CF.setup(+1)",
                              "CF.tera+sweep(side)")])
        jobs = [("feats", k, v) for k, v in num.items()] + [("ids", k, v) for k, v in ids.items()]
    else:
        mf, sf, gf, idg = old_groups()
        if drop:
            km, ks, kg, dli = V.prune_keep(drop)
            rm = {int(c): i for i, c in enumerate(km)}
            rs = {int(c): i for i, c in enumerate(ks)}
            rg = {int(c): i for i, c in enumerate(kg)}
            mf = {k: [rm[c] for c in v if c in rm] for k, v in mf.items()}
            sf = {k: [rs[c] for c in v if c in rs] for k, v in sf.items()}
            gf = {k: [rg[c] for c in v if c in rg] for k, v in gf.items()}
            mf = {k: v for k, v in mf.items() if v}
            sf = {k: v for k, v in sf.items() if v}
            gf = {k: v for k, v in gf.items() if v}
            if dli:
                idg.pop("emb.last_item", None)
        jobs = ([("mon_f", k, np.array(v)) for k, v in mf.items()]
                + [("side_f", k, np.array(v)) for k, v in sf.items()]
                + [("g", k, np.array(v)) for k, v in gf.items()]
                + [("mon_ids", k, np.array(v)) for k, v in idg.items()])
        if add:
            mg, rg = addon_groups(add)
            jobs += [("am", k, v) for k, v in mg.items()]
            jobs += [("ar", k, v) for k, v in rg.items()]
            jobs.append(("amar", "JOINT: every added block", None))
            # arm A's own resource block, for the same joint reading as §7 of
            # the value test: do the added blocks matter beside it, or is it
            # still just counting HP?
            jobs.append(("mon_f", "JOINT armA resources(hp+status+boosts)",
                         np.array(mf["mon.hp_frac"] + mf["mon.status"] + mf["mon.boosts"])))
    n = len(idx)
    for kind, name, cols in jobs:
        acc = {k: 0.0 for k in base}
        for r in range(REPS):
            pm = rng.permutation(n)
            b = dict(b0)
            if kind in ("feats", "ids"):
                key = "feats" if kind == "feats" else "ids"
                t = b0[key].clone()
                t[:, cols] = t[pm][:, cols]
                b[key] = t
            elif kind == "g":
                t = b0["g"].clone(); t[:, cols] = t[pm][:, cols]; b["g"] = t
            elif kind == "side_f":
                for k in ("sf1", "sf2"):
                    t = b0[k].clone(); t[:, cols] = t[pm][:, cols]; b[k] = t
            elif kind == "am":
                t = b0["am"].clone(); t[:, :, cols] = t[pm][:, :, cols]; b["am"] = t
            elif kind == "ar":
                t = b0["ar"].clone(); t[:, cols] = t[pm][:, cols]; b["ar"] = t
            elif kind == "amar":
                for k in ("am", "ar"):
                    if k in b0:
                        b[k] = b0[k][pm].clone()
            elif kind == "mon_f":
                for k in ("a1_f", "a2_f"):
                    t = b0[k].clone(); t[:, cols] = t[pm][:, cols]; b[k] = t
                for k in ("b1_f", "b2_f"):
                    t = b0[k].clone(); t[:, :, cols] = t[pm][:, :, cols]; b[k] = t
            else:  # mon_ids
                for k in ("a1_ids", "a2_ids"):
                    t = b0[k].clone(); t[:, cols] = t[pm][:, cols]; b[k] = t
                for k in ("b1_ids", "b2_ids"):
                    t = b0[k].clone(); t[:, :, cols] = t[pm][:, :, cols]; b[k] = t
            s = brier(b)
            for k in acc:
                acc[k] += (s[k] - base[k]) / REPS
        ncol = 0 if cols is None else len(cols) * (
            2 if kind == "side_f" else 12 if kind in ("mon_f", "mon_ids", "am") else 1)
        out[name] = {**acc, "n_cols": int(ncol)}
        print("  %-26s dBrier all=%+.5f early=%+.5f mid=%+.5f late=%+.5f"
              % (name, acc["all"], acc["early"], acc["mid"], acc["late"]), flush=True)
    return out


# ------------------------------------------------------ weight norms --------
def weight_norms(arm, net, net_init):
    def colnorm(W):
        return W.detach().pow(2).sum(0).sqrt().numpy()

    out = {}
    if arm == "enc2":
        num, ids = enc2_groups()
        if hasattr(net, "mon_cols"):
            # SHARED encoder: a per-mon column lands in the shared per-mon MLP's
            # first layer (at the same position for all twelve slots), everything
            # else lands in the trunk's first layer after the 12 (or 4) mon
            # vectors. Map every feature column to the matrix it actually enters.
            mc, rest = V.mon_and_rest_cols(V.load_layout())
            pos = {}
            for k in range(12):
                for j, c in enumerate(mc[k]):
                    pos[int(c)] = (0, j)
            off = (12 if net.mode == "shared" else 4) * net.mon[0].out_features
            for j, c in enumerate(rest):
                pos[int(c)] = (1, off + j)
            W = (colnorm(net.mon[0].weight), colnorm(net.trunk[0].weight))
            W0 = (colnorm(net_init.mon[0].weight), colnorm(net_init.trunk[0].weight))
            for k, v in num.items():
                p = [pos[int(c)] for c in v]
                t = float(np.mean([W[a][b] for a, b in p]))
                i0 = float(np.mean([W0[a][b] for a, b in p]))
                out[k] = {"trained": t, "init": i0, "ratio": t / i0}
        else:
            w, w0 = colnorm(net.trunk[0].weight), colnorm(net_init.trunk[0].weight)
            for k, v in num.items():
                out[k] = {"trained": float(w[v].mean()), "init": float(w0[v].mean()),
                          "ratio": float(w[v].mean() / w0[v].mean())}
        # embedding tables: mean row norm over rows actually used
        for k in ids:
            t = k.split(".")[1]
            tab = {"item": "item", "ability": "ability", "types": "ptype",
                   "tera_type": "ptype", "moves": "move", "side_move_state": "move"}[t]
            e, e0 = net.emb[tab].weight, net_init.emb[tab].weight
            out[k] = {"trained": float(e.detach().norm(dim=1).mean()),
                      "init": float(e0.detach().norm(dim=1).mean()),
                      "ratio": float(e.detach().norm(dim=1).mean() / e0.detach().norm(dim=1).mean())}
    else:
        mf, sf, gf, idg = old_groups()
        # mon MLP input = [species32,item12,ability12,moves64,tera8,item12, feats73]
        wm, wm0 = colnorm(net.mon[0].weight), colnorm(net_init.mon[0].weight)
        off = 32 + 12 + 12 + 64 + 8 + 12
        for k, v in mf.items():
            v = np.array(v) + off
            out[k] = {"trained": float(wm[v].mean()), "init": float(wm0[v].mean()),
                      "ratio": float(wm[v].mean() / wm0[v].mean())}
        emb_span = {"emb.species": (0, 32), "emb.item": (32, 44), "emb.ability": (44, 56),
                    "emb.moves": (56, 120), "emb.tera_type": (120, 128), "emb.last_item": (128, 140)}
        for k, (a, b) in emb_span.items():
            out[k] = {"trained": float(wm[a:b].mean()), "init": float(wm0[a:b].mean()),
                      "ratio": float(wm[a:b].mean() / wm0[a:b].mean())}
        wt, wt0 = colnorm(net.trunk[0].weight), colnorm(net_init.trunk[0].weight)
        H = net.mon[-2].out_features
        s1 = 4 * H
        for k, v in sf.items():
            v = np.array(v)
            cols = np.concatenate([s1 + v, s1 + 99 + v])
            out[k] = {"trained": float(wt[cols].mean()), "init": float(wt0[cols].mean()),
                      "ratio": float(wt[cols].mean() / wt0[cols].mean())}
        v = np.arange(18) + s1 + 198
        out["global.field"] = {"trained": float(wt[v].mean()), "init": float(wt0[v].mean()),
                               "ratio": float(wt[v].mean() / wt0[v].mean())}
    return out


# ------------------------------------------------------ directed probes -----
def probes(net, meta, idx):
    """enc2 only. Each probe edits ONE feature with an unambiguous sign and
    reports the mean change in P(side one wins) and the sign-agreement rate.
    CAVEAT, stated: editing one column leaves the rest of the vector unchanged
    and therefore mutually inconsistent -- this measures the net's response to
    the column, not to a real position change."""
    import enc2
    nm = enc2.DEFAULT_LAYOUT.names
    col = {n: i for i, n in enumerate(nm)}
    data = V.Arm("enc2")
    b0 = data.batch(idx)

    def pred(b):
        with torch.no_grad():
            return np.concatenate([torch.sigmoid(net({k: v[i:i + 8192] for k, v in b.items()})).numpy()
                                   for i in range(0, len(idx), 8192)])

    p0 = pred(b0)
    res = {}

    def run(name, edit, mask_fn, expect):
        f = b0["feats"].clone()
        m = mask_fn(b0["feats"])
        if m.sum() < 50:
            res[name] = {"n": int(m.sum()), "note": "too few rows"}
            return
        edit(f, m)
        b = dict(b0); b["feats"] = f
        d = (pred(b) - p0)[m.numpy()]
        res[name] = {"n": int(m.sum()), "expect": expect, "mean_dp": float(d.mean()),
                     "median_dp": float(np.median(d)),
                     "sign_ok": float(np.mean(d > 0 if expect > 0 else d < 0))}
        print("  %-42s n=%6d expect%+d  mean_dp=%+.4f  sign_ok=%.3f"
              % (name, res[name]["n"], expect, d.mean(), res[name]["sign_ok"]), flush=True)

    # 1. our active gains a guaranteed sweep at +1
    c = col["cf_mon0.s1_sweep"]
    run("setup sweep flag: OUR active 0->1", lambda f, m: f.__setitem__((m, c), 1.0),
        lambda f: (f[:, c] == 0) & (f[:, col["mon0.alive"]] > 0), +1)
    c6 = col["cf_mon6.s1_sweep"]
    run("setup sweep flag: THEIR active 0->1", lambda f, m: f.__setitem__((m, c6), 1.0),
        lambda f: (f[:, c6] == 0) & (f[:, col["mon6.alive"]] > 0), -1)

    # 2. our active's KO on their active upgraded 2HKO -> OHKO
    o, t = col["rel.ko_now_us_to_them_00_ohko"], col["rel.ko_now_us_to_them_00_2hko"]
    def up(f, m):
        f[m, t] = 0.0; f[m, o] = 1.0
    run("KO(our active -> their active) 2HKO->OHKO", up, lambda f: f[:, t] > 0.5, +1)
    o2, t2 = col["rel.ko_now_them_to_us_00_ohko"], col["rel.ko_now_them_to_us_00_2hko"]
    def up2(f, m):
        f[m, t2] = 0.0; f[m, o2] = 1.0
    run("KO(their active -> our active) 2HKO->OHKO", up2, lambda f: f[:, t2] > 0.5, -1)

    # 3. speed: our active becomes the faster of the active pair
    sp = col["rel.pairspeed_00"]
    mg = col["rel.active_speed_margin"]
    def fast(f, m):
        f[m, sp] = 1.0
        f[m, mg] = f[m, mg].abs()
    run("active pair speed: we go from slower to faster", fast,
        lambda f: (f[:, sp] < 0.5) & (f[:, col["mon0.alive"]] > 0) & (f[:, col["mon6.alive"]] > 0), +1)

    # 4. tera option value
    tv, ts = col["cf_side0.tera_best_value"], col["cf_side0.tera_enabled_sweep"]
    run("tera: OUR tera-enabled sweep 0->1", lambda f, m: f.__setitem__((m, ts), 1.0),
        lambda f: (f[:, ts] == 0) & (f[:, col["side0.tera_available"]] > 0.5), +1)
    tv1, ts1 = col["cf_side1.tera_best_value"], col["cf_side1.tera_enabled_sweep"]
    run("tera: THEIR tera-enabled sweep 0->1", lambda f, m: f.__setitem__((m, ts1), 1.0),
        lambda f: (f[:, ts1] == 0) & (f[:, col["side1.tera_available"]] > 0.5), -1)
    run("tera: OUR best tera value +0.5", lambda f, m: f.__setitem__((m, tv), f[m, tv] + 0.5),
        lambda f: f[:, col["side0.tera_available"]] > 0.5, +1)

    # 5. control: our active loses half its HP (must move down, and a lot)
    h = col["mon0.hp_frac"]
    run("CONTROL our active hp_frac -> half", lambda f, m: f.__setitem__((m, h), f[m, h] * 0.5),
        lambda f: f[:, h] > 0.5, -1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=0, help="subsample test rows (0 = all)")
    a = ap.parse_args()
    torch.set_num_threads(4)
    ck = torch.load(a.ckpt, weights_only=False)
    r = ck["res"]
    cap, seed = tuple(r["cap"]), r["seed"]
    kw = {k: r[k] for k in ("arch", "mon_depth", "depth", "slot_emb", "add", "drop") if k in r}
    add, drop = kw.get("add", ""), kw.get("drop", "")
    net = V.build_net(a.arm, cap, seed, **kw)
    net.load_state_dict(ck["sd"])
    net.eval()
    net_init = V.build_net(a.arm, cap, seed + 991, **kw)

    meta = V.load_meta()
    ix, _ = V.split_idx(meta)
    idx = ix["test"]
    if a.n:
        idx = np.sort(np.random.default_rng(3).choice(idx, a.n, replace=False))
    rng = np.random.default_rng(11)
    print("== permutation importance (%s, cap=%s, %d test rows) ==" % (a.arm, cap, len(idx)))
    perm = run_perm(a.arm, net, meta, idx, rng, add=add, drop=drop)
    print("== weight norms ==")
    wn = weight_norms(a.arm, net, net_init) if not add else {}
    for k in sorted(wn, key=lambda k: -wn[k]["ratio"]):
        v = wn[k]
        print("  %-26s trained=%.4f init=%.4f  ratio=%.3f" % (k, v["trained"], v["init"], v["ratio"]), flush=True)
    pr = {}
    if a.arm == "enc2":
        print("== directed probes ==")
        pr = probes(net, meta, idx)
    json.dump({"arm": a.arm, "cap": list(cap), "ckpt": a.ckpt, "perm": perm,
               "wnorm": wn, "probes": pr, "res": ck["res"]}, open(a.out, "w"), indent=1)
    print("wrote " + a.out)


if __name__ == "__main__":
    main()
