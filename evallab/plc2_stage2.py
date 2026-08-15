"""STAGE 2 -- encode plc2 with the adopted encoder, and MERGE plc1 + plc2.

Three commands, run in this order on one box:

  prep  <positions.jsonl> <labels_dir> <out_meta.npz> <out_src.jsonl>
        Builds plc2's meta.npz (the label side) IN ROW ORDER and writes the
        FILTERED state file whose line r is exactly meta row r.

        THE GAP.  plc2's selection has 999,871 positions but only 999,870
        labels: i=409289 died on an engine panic during labelling.  Rather than
        carry a hole (enc_adopted would then encode 999,871 rows against a
        999,870-row meta and write past the end of every memmap), prep DROPS
        that line from the state file.  Row r of the cache is then line r of
        `out_src.jsonl` is then row r of `out_meta.npz`, densely, and the
        encoder's own `--meta` assertion can check the join row by row.

  tamper <src.jsonl> <meta.npz> <workdir> [K]
        NEGATIVE CONTROL for that assertion.  Encodes K rows twice: once clean
        (must pass) and once with ONE meta game id corrupted (must fail with
        JOIN BROKEN).  A join check that cannot fail proves nothing.

  merge <enc_plc1_dir> <enc_plc2_dir> <out_dir>
        Concatenates the two caches into one the trainer mmaps directly:
        plc1 occupies merged rows 0..999,999 IN ITS ORIGINAL ORDER, so
        plc1's holdout_i.npy -- which vt_canary/vt_ft1 use as raw ROW indices
        (`mask[hold] = False`) -- stays correct byte for byte and the 100,000
        held-out rows are exactly the ones plc1 held out.  Every plc2 row goes
        into the train pool.

        The merge is verified, not assumed: every array is read back and
        compared byte for byte against both sources over ALL rows, which is
        what carries each half's row-by-row join proof onto the merged cache.
        Then the leak check: no train row may share a team pair with a holdout
        row, checked over the whole merged corpus, not just plc2.

  selftest <tmpdir>
        Runs prep+merge end to end on fabricated tiny inputs, and checks that
        the misordered-meta and leaking-pair guards actually fire.  Needs numpy
        only -- no engine, no encoder -- so it can canary this file locally
        before renting anything.
"""
import glob
import hashlib
import json
import os
import sys

import numpy as np

# plc1_meta.py's field lists, verbatim -- the merged meta must carry exactly
# the columns plc1's does, or a downstream reader breaks on the merged cache.
FIELDS_F = ("label_p", "q_search", "y_single", "se")
FIELDS_I = ("ply", "wins", "losses", "ties", "trunc", "n_playouts")
FIELDS_S = ("g", "pair", "band", "src")
ARRAYS = ["old_a1_f", "old_a1_ids", "old_a2_f", "old_a2_ids", "old_b1_f",
          "old_b1_ids", "old_b2_f", "old_b2_ids", "old_g", "old_sf1", "old_sf2",
          "addon_mon"]
CHUNK = 50000


def _gid(line):
    """The `g` field without parsing the 3.5 kB state string next to it."""
    assert line.startswith('{"g": "'), line[:40]
    j = line.index('"', 7)
    return line[7:j]


# ------------------------------------------------------------------- prep ----
def prep(pos_path, lab_dir, out_meta, out_src):
    files = sorted(f for f in os.listdir(lab_dir)
                   if f.startswith("labels_") and f.endswith(".jsonl"))
    assert files, lab_dir
    raw = []
    for fn in files:
        with open(os.path.join(lab_dir, fn)) as f:
            for line in f:
                raw.append(json.loads(line))
    # A LINE IS NOT A LABEL. plc2's labeller wrote one ERROR record --
    # {"i": 409289, "error": "PanicException('Invalid rest_turns value: -1')"}
    # -- for the position that hit an engine panic. It has no label and no game
    # id, so it is dropped here and its `i` shows up in missing_i. Anything
    # unlabelled for a reason we did NOT anticipate must stop the run.
    errs = [r for r in raw if "label_p" not in r]
    assert all("error" in r and "i" in r for r in errs), \
        "unlabelled record without an error field: %s" % [r for r in errs if "error" not in r][:3]
    rows = [r for r in raw if "label_p" in r]
    n = len(rows)
    i = np.array([r["i"] for r in rows], np.int64)
    o = np.argsort(i, kind="stable")
    i = i[o]
    assert len(np.unique(i)) == n, "duplicate i in the label files"
    rows = [rows[k] for k in o]

    d = {"i": np.arange(n, dtype=np.int64), "i_src": i.copy()}
    for k in FIELDS_S:
        d[k] = np.array([r[k] for r in rows])
    for k in FIELDS_F:
        d[k] = np.array([r[k] for r in rows], np.float64)
    for k in FIELDS_I:
        d[k] = np.array([r[k] for r in rows], np.int32)

    keep = set(int(x) for x in i)
    gi = {int(x): j for j, x in enumerate(i)}
    mism, kept, total = [], 0, 0
    with open(pos_path) as f, open(out_src, "w") as o_f:
        for ln, line in enumerate(f):
            total += 1
            if ln not in keep:
                continue
            if _gid(line) != d["g"][gi[ln]]:
                mism.append(ln)
            o_f.write(line)
            kept += 1
    assert kept == n, "kept %d lines but have %d labels" % (kept, n)
    assert not mism, "game id mismatch at source lines %s" % mism[:10]

    np.savez_compressed(out_meta, **d)
    missing = sorted(set(range(total)) - keep)
    rep = {"cmd": "prep", "labels": n, "label_lines": len(raw), "source_lines": total,
           "kept": kept, "error_records": [{"i": r["i"], "error": r["error"]} for r in errs],
           "missing_i": missing[:20], "n_missing": len(missing),
           "gid_mismatches": 0, "mean_label_p": round(float(d["label_p"].mean()), 6),
           "bands": {b: int((d["band"] == b).sum()) for b in ("early", "mid", "late")},
           "meta_bytes": os.path.getsize(out_meta),
           "src_bytes": os.path.getsize(out_src)}
    json.dump(rep, open(os.path.join(os.path.dirname(os.path.abspath(out_meta)),
                                     "PREP_REPORT.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1), flush=True)
    return rep


# ----------------------------------------------------------------- tamper ----
def tamper(src, meta, work, k=2048):
    """Positive then negative control on enc_adopted's row-by-row join check."""
    import enc_adopted as EA
    os.makedirs(work, exist_ok=True)
    sub = os.path.join(work, "sub.jsonl")
    with open(src) as f, open(sub, "w") as o:
        for j, line in enumerate(f):
            if j >= k:
                break
            o.write(line)
    z = dict(np.load(meta, allow_pickle=False))
    m_ok = os.path.join(work, "m_ok.npz")
    np.savez_compressed(m_ok, **{x: v[:k] for x, v in z.items()})
    bad = {x: v[:k].copy() for x, v in z.items()}
    row = k // 2
    orig = str(bad["g"][row])
    bad["g"][row] = "r4/shard_9999999/gTAMPERED"
    m_bad = os.path.join(work, "m_bad.npz")
    np.savez_compressed(m_bad, **bad)

    EA.encode(sub, os.path.join(work, "clean"), meta_path=m_ok, workers=4)
    ok = "clean encode passed the join check"
    try:
        EA.encode(sub, os.path.join(work, "dirty"), meta_path=m_bad, workers=4)
        raise SystemExit("TAMPER TEST FAILED: corrupted row %d was ACCEPTED" % row)
    except AssertionError as e:
        msg = str(e)
    assert "JOIN BROKEN" in msg and ("row %d" % row) in msg.replace("first at ", "row "), msg
    rep = {"cmd": "tamper", "rows": k, "clean": ok, "tampered_row": row,
           "tampered_from": orig, "rejected_with": msg[:160], "tamper_test": "PASS"}
    json.dump(rep, open(os.path.join(work, "TAMPER_REPORT.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1), flush=True)
    return rep


# ------------------------------------------------------------------ merge ----
def _hdr(p):
    a = np.load(p, mmap_mode="r")
    return a.dtype, a.shape


def merge(d1, d2, out):
    os.makedirs(out, exist_ok=True)
    l1 = json.load(open(os.path.join(d1, "addon_layout.json")))
    l2 = json.load(open(os.path.join(d2, "addon_layout.json")))
    assert l1 == l2, "addon_layout.json differs between the two caches"
    json.dump(l1, open(os.path.join(out, "addon_layout.json"), "w"))

    n1 = n2 = None
    sizes = {}
    for k in ARRAYS:
        p1, p2 = os.path.join(d1, k + ".npy"), os.path.join(d2, k + ".npy")
        (t1, s1), (t2, s2) = _hdr(p1), _hdr(p2)
        assert t1 == t2, "%s dtype %s vs %s" % (k, t1, t2)
        assert s1[1:] == s2[1:], "%s trailing shape %s vs %s" % (k, s1, s2)
        if n1 is None:
            n1, n2 = s1[0], s2[0]
        assert (s1[0], s2[0]) == (n1, n2), "%s row count %s/%s != %d/%d" % (
            k, s1[0], s2[0], n1, n2)
        a1 = np.load(p1, mmap_mode="r")
        a2 = np.load(p2, mmap_mode="r")
        o = np.lib.format.open_memmap(os.path.join(out, k + ".npy"), mode="w+",
                                      dtype=t1, shape=(n1 + n2,) + tuple(s1[1:]))
        for src, off in ((a1, 0), (a2, n1)):
            for s in range(0, len(src), CHUNK):
                e = min(s + CHUNK, len(src))
                o[off + s:off + e] = src[s:e]
        o.flush()
        del o
        sizes[k + ".npy"] = os.path.getsize(os.path.join(out, k + ".npy"))
        print("  %-16s %s %s + %s -> %d rows" % (k, t1, s1, s2, n1 + n2), flush=True)

    # ---- byte-for-byte read-back over EVERY row, both halves ----------------
    bad = []
    for k in ARRAYS:
        m = np.load(os.path.join(out, k + ".npy"), mmap_mode="r")
        for d, off in ((d1, 0), (d2, n1)):
            src = np.load(os.path.join(d, k + ".npy"), mmap_mode="r")
            for s in range(0, len(src), CHUNK):
                e = min(s + CHUNK, len(src))
                if np.asarray(m[off + s:off + e]).tobytes() != np.asarray(src[s:e]).tobytes():
                    bad.append((k, off, s))
                    break
    assert not bad, "MERGE CORRUPTED rows: %s" % bad[:5]

    # ---- meta ---------------------------------------------------------------
    z1 = dict(np.load(os.path.join(d1, "meta.npz"), allow_pickle=False))
    z2 = dict(np.load(os.path.join(d2, "meta.npz"), allow_pickle=False))
    assert len(z1["g"]) == n1 and len(z2["g"]) == n2, "meta rows != array rows"
    # plc1's meta predates `i_src` (its i IS the source line number); plc2's
    # carries it because its rows are not dense in the source.
    z1.setdefault("i_src", z1["i"].copy())
    z2.setdefault("i_src", z2["i"].copy())
    keys = sorted(set(z1) & set(z2))
    assert set(keys) >= set(FIELDS_S) | set(FIELDS_F) | set(FIELDS_I) | {"i", "i_src"}, keys
    d = {k: np.concatenate([z1[k], z2[k]]) for k in keys}
    d["i"] = np.arange(n1 + n2, dtype=np.int64)
    d["corpus"] = np.array(["plc1"] * n1 + ["plc2"] * n2)
    np.savez_compressed(os.path.join(out, "meta.npz"), **d)

    # ---- holdout: plc1's, unchanged, still valid as raw row indices ---------
    hold = np.load(os.path.join(d1, "holdout_i.npy"))
    assert hold.max() < n1 and hold.min() >= 0, "holdout indexes outside the plc1 block"
    assert len(np.unique(hold)) == len(hold), "duplicate holdout index"
    # the plc1 half is copied in its own row order, so row r IS plc1 row r and
    # plc1's index file needs no remapping at all
    assert np.array_equal(np.asarray(z1["g"])[hold.astype(np.int64)],
                          np.asarray(d["g"])[hold.astype(np.int64)]), \
        "holdout rows do not land on the same games in the merged cache"
    np.save(os.path.join(out, "holdout_i.npy"), hold)

    mask = np.ones(n1 + n2, bool)
    mask[hold.astype(np.int64)] = False
    train = np.flatnonzero(mask)

    # ---- LEAK CHECK: no train row may share a team pair with a holdout row --
    pair = np.asarray(d["pair"]).astype(str)
    game = np.asarray(d["g"]).astype(str)
    hp = set(pair[hold.astype(np.int64)].tolist())
    hg = set(game[hold.astype(np.int64)].tolist())
    p2 = set(pair[n1:].tolist())
    g2 = set(game[n1:].tolist())
    leak_plc2_pair = len(hp & p2)
    leak_plc2_game = len(hg & g2)
    tp = set(pair[train].tolist())
    tg = set(game[train].tolist())
    leak_all_pair = len(hp & tp)
    leak_all_game = len(hg & tg)
    assert leak_plc2_pair == 0 and leak_plc2_game == 0, \
        "PLC2 LEAKS INTO THE HOLDOUT: %d pairs / %d games" % (leak_plc2_pair, leak_plc2_game)
    assert leak_all_pair == 0 and leak_all_game == 0, \
        "TRAIN LEAKS INTO THE HOLDOUT: %d pairs / %d games" % (leak_all_pair, leak_all_game)

    s1j = json.load(open(os.path.join(d1, "split.json")))
    split = dict(s1j)
    # plc1's nested blocks describe PLC1's split, not the merged one -- its
    # `train.n` is 900,000, which is a lie about this cache. Relabel them so the
    # only unqualified counts in the file are the merged ones.
    for old, new in (("train", "plc1_train_dist"), ("holdout", "holdout_dist"),
                     ("n_pairs", "plc1_n_pairs"), ("n_games", "plc1_n_games")):
        if old in split:
            split[new] = split.pop(old)
    split.update({"n_total": int(n1 + n2), "n_train": int(len(train)),
                  "n_holdout": int(len(hold)),
                  "n_plc1": int(n1), "n_plc2": int(n2),
                  "holdout_source": "plc1 split.json, seed %s, unit pair -- COPIED "
                                    "UNCHANGED, row indices still valid because plc1 "
                                    "occupies merged rows 0..%d in its original order"
                                    % (s1j.get("seed"), n1 - 1),
                  "note": "train = every row NOT in holdout_i.npy = plc1 train "
                          "(%d) + all of plc2 (%d); plc1e untouched"
                          % (int(len(train)) - int(n2), int(n2))})
    json.dump(split, open(os.path.join(out, "split.json"), "w"), indent=1)

    for f in sorted(os.listdir(out)):
        sizes.setdefault(f, os.path.getsize(os.path.join(out, f)))
    rep = {"rows": int(n1 + n2), "rows_plc1": int(n1), "rows_plc2": int(n2),
           "train_rows": int(len(train)), "holdout_rows": int(len(hold)),
           "train_plc1": int(n1 - len(hold)), "train_plc2": int(n2),
           "readback_mismatches": 0,
           "holdout_pair_overlap_with_plc2": leak_plc2_pair,
           "holdout_game_overlap_with_plc2": leak_plc2_game,
           "holdout_pair_overlap_with_train": leak_all_pair,
           "holdout_game_overlap_with_train": leak_all_game,
           "distinct_pairs": int(len(tp | hp)), "distinct_games": int(len(tg | hg)),
           "meta_keys": sorted(d), "bytes": sizes,
           "bytes_total": int(sum(sizes.values())),
           "bytes_per_row": round(sum(sizes.values()) / (n1 + n2), 1),
           "band_frac_train": {b: round(float((np.asarray(d["band"])[train] == b).mean()), 5)
                               for b in ("early", "mid", "late")},
           "mean_label_p_train": round(float(np.asarray(d["label_p"])[train].mean()), 6)}
    json.dump(rep, open(os.path.join(out, "MERGE_REPORT.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "bytes"}, indent=1), flush=True)
    return rep


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------- selftest ----
def _fake_cache(d, n, seed, pairs, games):
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(seed)
    sh = {"old_a1_f": (n, 72), "old_a2_f": (n, 72), "old_b1_f": (n, 5, 72),
          "old_b2_f": (n, 5, 72), "old_sf1": (n, 99), "old_sf2": (n, 99),
          "old_g": (n, 18), "old_a1_ids": (n, 9), "old_a2_ids": (n, 9),
          "old_b1_ids": (n, 5, 9), "old_b2_ids": (n, 5, 9), "addon_mon": (n, 12, 14)}
    for k, s in sh.items():
        t = np.int16 if "ids" in k else np.float16
        a = np.lib.format.open_memmap(os.path.join(d, k + ".npy"), mode="w+", dtype=t, shape=s)
        a[:] = (rng.integers(0, 100, s).astype(t) if "ids" in k
                else rng.random(s).astype(t))
        a.flush()
    json.dump({"D_mon": 14, "D_rest": 0, "groups": {"mon.setup": [0, 14]},
               "mon_names": ["c%d" % i for i in range(14)], "rest_names": []},
              open(os.path.join(d, "addon_layout.json"), "w"))
    m = {"i": np.arange(n, dtype=np.int64), "g": np.array(games),
         "pair": np.array(pairs), "band": np.array(["early", "mid", "late"] * n)[:n],
         "src": np.array(["r4"] * n)}
    for k in FIELDS_F:
        m[k] = rng.random(n)
    for k in FIELDS_I:
        m[k] = rng.integers(0, 10, n).astype(np.int32)
    np.savez_compressed(os.path.join(d, "meta.npz"), **m)
    return m


def selftest(work):
    os.makedirs(work, exist_ok=True)
    n1, n2 = 60, 40
    p1 = ["p%04d" % i for i in range(n1)]
    g1 = ["r4/s/g%04d" % i for i in range(n1)]
    p2 = ["q%04d" % i for i in range(n2)]
    g2 = ["r4/s/h%04d" % i for i in range(n2)]
    d1, d2 = os.path.join(work, "c1"), os.path.join(work, "c2")
    _fake_cache(d1, n1, 1, p1, g1)
    _fake_cache(d2, n2, 2, p2, g2)
    hold = np.sort(np.random.default_rng(0).choice(n1, 12, replace=False)).astype(np.int32)
    np.save(os.path.join(d1, "holdout_i.npy"), hold)
    json.dump({"seed": 20260814, "unit": "pair", "n_total": n1},
              open(os.path.join(d1, "split.json"), "w"))

    out = os.path.join(work, "m")
    r = merge(d1, d2, out)
    assert r["rows"] == n1 + n2 and r["train_rows"] == n1 + n2 - len(hold)
    # the merged arrays really are the two sources, in order
    a = np.load(os.path.join(out, "old_a1_f.npy"), mmap_mode="r")
    assert np.array_equal(a[:n1], np.load(os.path.join(d1, "old_a1_f.npy"), mmap_mode="r"))
    assert np.array_equal(a[n1:], np.load(os.path.join(d2, "old_a1_f.npy"), mmap_mode="r"))
    z = np.load(os.path.join(out, "meta.npz"), allow_pickle=False)
    assert list(z["g"][hold.astype(np.int64)]) == [g1[i] for i in hold]
    assert np.array_equal(np.load(os.path.join(out, "holdout_i.npy")), hold)

    # GUARD 1: a plc2 row sharing a holdout pair must be rejected
    d2b = os.path.join(work, "c2_leak")
    _fake_cache(d2b, n2, 2, [p1[int(hold[0])]] + p2[1:], g2)
    try:
        merge(d1, d2b, os.path.join(work, "m_leak"))
        raise SystemExit("SELFTEST FAILED: pair leak not caught")
    except AssertionError as e:
        assert "LEAKS INTO THE HOLDOUT" in str(e), e

    # GUARD 2: a corrupted merged row must be caught by the read-back
    d2c = os.path.join(work, "c2_short")
    _fake_cache(d2c, n2, 3, p2, g2)
    b = np.load(os.path.join(d2c, "old_sf1.npy"), mmap_mode="r+")
    b[0, 0] = np.float16(np.nan)
    b.flush()
    bad = np.load(os.path.join(d2c, "old_sf1.npy"), mmap_mode="r")
    # tobytes comparison must treat NaN as equal to itself (np.array_equal would not)
    assert np.asarray(bad[0:1]).tobytes() == np.asarray(b[0:1]).tobytes()
    merge(d1, d2c, os.path.join(work, "m_nan"))

    # GUARD 3: prep must reject a label set whose i values duplicate
    lab = os.path.join(work, "lab")
    os.makedirs(lab, exist_ok=True)
    recs = [{"i": i, "g": g1[i], "pair": p1[i], "band": "early", "src": "r4",
             "label_p": 0.5, "q_search": 0.5, "y_single": 1.0, "se": 0.1, "ply": i,
             "wins": 5, "losses": 5, "ties": 0, "trunc": 0, "n_playouts": 10}
            for i in range(n1) if i != 7]
    with open(os.path.join(lab, "labels_a.jsonl"), "w") as f:
        for r_ in recs:
            f.write(json.dumps(r_) + "\n")
        # the real corpus has exactly one of these; it must be dropped, not parsed
        f.write(json.dumps({"i": 7, "error": "PanicException('boom')"}) + "\n")
    pos = os.path.join(work, "pos.jsonl")
    with open(pos, "w") as f:
        for i in range(n1):
            f.write(json.dumps({"g": g1[i], "ply": i, "band": "early",
                                "s": "STATE%d" % i}) + "\n")
    rp = prep(pos, lab, os.path.join(work, "meta2.npz"), os.path.join(work, "src2.jsonl"))
    assert rp["labels"] == n1 - 1 and rp["kept"] == n1 - 1 and rp["missing_i"] == [7]
    zz = np.load(os.path.join(work, "meta2.npz"), allow_pickle=False)
    assert list(zz["i_src"]) == [i for i in range(n1) if i != 7]
    got = [json.loads(l)["g"] for l in open(os.path.join(work, "src2.jsonl"))]
    assert got == [g1[i] for i in range(n1) if i != 7], "filtered src is misaligned"
    with open(os.path.join(lab, "labels_b.jsonl"), "w") as f:
        f.write(json.dumps(recs[0]) + "\n")
    try:
        prep(pos, lab, os.path.join(work, "m3.npz"), os.path.join(work, "s3.jsonl"))
        raise SystemExit("SELFTEST FAILED: duplicate i not caught")
    except AssertionError as e:
        assert "duplicate i" in str(e), e
    print("SELFTEST PASSED", flush=True)


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "prep":
        prep(*sys.argv[2:6])
    elif c == "tamper":
        tamper(sys.argv[2], sys.argv[3], sys.argv[4],
               int(sys.argv[5]) if len(sys.argv) > 5 else 2048)
    elif c == "merge":
        merge(sys.argv[2], sys.argv[3], sys.argv[4])
    elif c == "selftest":
        selftest(sys.argv[2])
    else:
        raise SystemExit(__doc__)
