"""LOSSLESS position encoder, production-quality for the lab (pair A, full info).

This is `valuenet/lossless_encoder.py` (the reference implementation and the
executable form of LOSSLESS_ENCODING_SPEC.md) turned into something that can
encode millions of positions: a columnar batch parser plus a fully vectorised
numpy encoder. It shares every CONSTANT and the DECODER with the reference
module, so there is exactly one definition of the layout and one decoder, and
`ll_gate.py` proves this implementation against the reference parser.

WHAT IS DIFFERENT FROM THE REFERENCE
  reference: state string -> dict-of-dicts -> ~400 scalar numpy stores
  here:      list[state string] -> COLUMNAR numpy arrays -> vectorised writes
  The layout, the divisors, the column order and the decoder are IDENTICAL --
  `VARIANTS["full"]` is asserted equal to `lossless_encoder.L` column for column.

TWO VARIANTS (spec section 5.6), selected by `variant=`:
  "full"  1,863 numeric columns = 12*121 + 2*196 + 19   (EXACT + DERIVED)
  "lean"  1,206 numeric columns = 12*68  + 2*186 + 18   (EXACT only)
Both round-trip: the decoder reads EXACT columns only, so DERIVED is a warm
start for the net, never a carrier of information. "lean" is narrower than
today's shipped encoder (1,260) AND provably lossless.

THE FOUR STRUCTURAL FIXES THE SPEC REQUIRED, all present here:
  1. `active_index` is a 6-way one-hot and the bench is in PARTY order (slot 0 =
     active, slots 1..5 = the other party members ascending). No BENCH_SORT. Two
     switch successors of one root therefore encode DIFFERENTLY -- asserted by
     `ll_gate.py switchsib`.
  2. `last_used_move` carries its switch target (`lum_switch_slot`, 6 columns)
     and its `UnslottedMove(Choices)` payload (a move embedding id), so the
     33.34%-of-side-vectors loss and the Python-asserts/Rust-folds divergence
     both disappear.
  3. `active_move_actions` and `times_attacked` are UNCLIPPED (/64). The clipped
     forms survive only as DERIVED columns, because they are what the mechanics
     actually gate on.
  4. `switch_out_move_second_saved_move` is a move EMBEDDING ID, not a bool
     (`has_banked_move` is kept alongside it as a convenience column), and ALL
     107 `PokemonVolatileStatus` variants get a bit, not 34.

ASSUMPTIONS, STATED
  * The state string is the definition of "the full state" (spec section 2.1
    lists the 23 struct fields `serialize` drops; no encoder can see them).
  * Vocabulary injectivity is required for losslessness. A miss is recorded and
    reported, never silently folded into UNK -- the gate treats one as failure.
  * The parser assumes engine-cased (upper) names on the hot path and falls back
    to `.upper()` on a miss, so a lower-cased producer costs speed, not
    correctness. Proven by the gate, which compares every parsed field against
    `lossless_encoder.parse_state` (itself validated against the real bindings).
  * float32 is the encoding dtype. `lldataset.py` stores the training cache as
    float16 for memory; that is a STORAGE choice and `ll_gate.py fp16` measures
    exactly which columns it costs (see ENCODER_BUILD.md).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "valuenet"))
import lossless_encoder as LE  # noqa: E402

# --- everything closed-enum comes from the reference module; no second copy ---
TYPE_ORDER, STATUS_ORDER, NATURE_ORDER = LE.TYPE_ORDER, LE.STATUS_ORDER, LE.NATURE_ORDER
WEATHER_ORDER, TERRAIN_ORDER = LE.WEATHER_ORDER, LE.TERRAIN_ORDER
VOLATILE_ORDER, N_VOLATILES = LE.VOLATILE_ORDER, LE.N_VOLATILES
SIDE_CONDITION_FIELDS, DURATION_FIELDS = LE.SIDE_CONDITION_FIELDS, LE.DURATION_FIELDS
TYPE_IX, STATUS_IX, NATURE_IX = LE.TYPE_IX, LE.STATUS_IX, LE.NATURE_IX
WEATHER_IX, TERRAIN_IX, VOLATILE_IX = LE.WEATHER_IX, LE.TERRAIN_IX, LE.VOLATILE_IX
LUM_KINDS = ("none", "move", "switch", "struggle", "unslotted")
LUM_KIND_IX = {k: i for i, k in enumerate(LUM_KINDS)}
N_MON_IDS, N_SIDE_IDS = 15, 2
N_SC, N_DUR = len(SIDE_CONDITION_FIELDS), len(DURATION_FIELDS)


# ===========================================================================
# 1. column layout, per variant
# ===========================================================================
class Variant:
    """A column layout. "full" is `lossless_encoder.L` verbatim (asserted);
    "lean" is its EXACT subset in the same relative order."""

    def __init__(self, name, keep_derived):
        self.name = name
        self.keep_derived = keep_derived

        def pick(names, kinds):
            return [n for n, k in zip(names, kinds) if keep_derived or k == "EXACT"]

        self.mon_feat = pick(LE.L.mon_feat, LE.L.mon_kind)
        self.side_feat = pick(LE.L.side_feat, LE.L.side_kind)
        self.glob_feat = pick(LE.L.glob_feat, LE.L.glob_kind)
        self.M = {n: i for i, n in enumerate(self.mon_feat)}
        self.S = {n: i for i, n in enumerate(self.side_feat)}
        self.G = {n: i for i, n in enumerate(self.glob_feat)}
        self.NM, self.NS, self.NG = len(self.mon_feat), len(self.side_feat), len(self.glob_feat)
        self.N_FEATS = 12 * self.NM + 2 * self.NS + self.NG
        self.N_IDS = 12 * N_MON_IDS + 2 * N_SIDE_IDS
        # index of each of this variant's columns inside the FULL layout, so a
        # lean vector can be re-expanded for the (single, shared) decoder.
        self.mon_to_full = np.array([LE.L.MON_IX[n] for n in self.mon_feat], np.int64)
        self.side_to_full = np.array([LE.L.SIDE_IX[n] for n in self.side_feat], np.int64)
        self.glob_to_full = np.array([LE.L.GLOB_IX[n] for n in self.glob_feat], np.int64)
        # contiguous column blocks used by the vectorised writer
        self.mon_blocks = {
            "status": self._blk(self.M, ["status_%s" % s for s in STATUS_ORDER]),
            "stellar": self._blk(self.M, ["stellar_boosted_%s" % t for t in TYPE_ORDER]),
            "revmask": self._blk(self.M, ["reveal_mask_bit%d" % b for b in range(8)]),
            "ev": self._blk(self.M, ["ev%d" % i for i in range(6)]),
            "pp": self._blk(self.M, ["pp%d" % i for i in range(4)]),
            "disabled": self._blk(self.M, ["disabled%d" % i for i in range(4)]),
        }
        if keep_derived:
            self.mon_blocks.update({
                "ppfrac": self._blk(self.M, ["pp_frac%d" % i for i in range(4)]),
                "type_eff": self._blk(self.M, ["type_eff_%s" % t for t in TYPE_ORDER[:18]]),
                "type_own": self._blk(self.M, ["type_own_%s" % t for t in TYPE_ORDER[:18]]),
                "boost": self._blk(self.M, ["boost_%s" % n for n in
                                            ("atk", "def", "spa", "spd", "spe")]),
            })
        self.side_blocks = {
            "active": self._blk(self.S, ["active_index_%d" % i for i in range(6)]),
            "sc": self._blk(self.S, ["sc_%s" % n for n in SIDE_CONDITION_FIELDS]),
            "vol": self._blk(self.S, ["vol_%s" % v for v in VOLATILE_ORDER]),
            "dur": self._blk(self.S, ["dur_%s" % n for n in DURATION_FIELDS]),
            "boost": self._blk(self.S, ["attack_boost", "defense_boost",
                                        "special_attack_boost", "special_defense_boost",
                                        "speed_boost", "accuracy_boost", "evasion_boost"]),
            "fs_src": self._blk(self.S, ["fs_source_slot%d" % i for i in range(6)]),
            "flags": self._blk(self.S, ["force_switch", "force_trapped", "slow_uturn_move",
                                        "baton_passing", "shed_tailing", "revival_blessing",
                                        "last_move_failed"]),
            "lum_kind": self._blk(self.S, ["lum_kind_%s" % k for k in LUM_KINDS]),
            "lum_move": self._blk(self.S, ["lum_move_slot%d" % i for i in range(4)]),
            "lum_switch": self._blk(self.S, ["lum_switch_slot%d" % i for i in range(6)]),
        }
        if keep_derived:
            self.side_blocks["paradox"] = self._blk(
                self.S, ["paradox_%s" % s for s in LE.PARADOX_STATS])
        self.glob_blocks = {
            "weather": self._blk(self.G, ["weather_%s" % w for w in WEATHER_ORDER]),
            "terrain": self._blk(self.G, ["terrain_%s" % t for t in TERRAIN_ORDER]),
        }

    @staticmethod
    def _blk(ix, names):
        """Assert the named columns are contiguous and return (start, stop)."""
        a = [ix[n] for n in names]
        assert a == list(range(a[0], a[0] + len(a))), "non-contiguous block %s" % names[:3]
        return (a[0], a[0] + len(a))


VARIANTS = {"full": Variant("full", True), "lean": Variant("lean", False)}

# The full variant MUST be the reference layout, column for column. If this ever
# fires, the two files have drifted and every round-trip number is void.
assert VARIANTS["full"].mon_feat == LE.L.mon_feat
assert VARIANTS["full"].side_feat == LE.L.side_feat
assert VARIANTS["full"].glob_feat == LE.L.glob_feat
assert VARIANTS["full"].N_FEATS == LE.L.N_FEATS_TOTAL == 1863
assert VARIANTS["lean"].N_FEATS == 1206, VARIANTS["lean"].N_FEATS

# ---- max-pp table, resolved to vocab ids once (DERIVED pp_frac only) --------
_MOVE_MAXPP_CACHE = {}


def move_maxpp_table(vocab):
    """arr[move_vocab_id] = max pp. Exact iff every MAX_PP key is in the vocab,
    which is asserted here rather than assumed -- a missing key would make the
    DERIVED pp_frac column silently disagree with the reference encoder."""
    key = id(vocab)
    t = _MOVE_MAXPP_CACHE.get(key)
    if t is None:
        d = vocab.d["move"]
        missing = [k for k in LE.MAX_PP if k not in d]
        assert not missing, "MAX_PP keys absent from the move vocab: %s" % missing[:5]
        t = np.zeros(max(d.values()) + 1, np.float64)
        for name, i in d.items():
            t[i] = LE.MAX_PP.get(name, 0)
        _MOVE_MAXPP_CACHE[key] = t
    return t


# ===========================================================================
# 2. columnar batch parser  (state strings -> numpy, no per-state dicts)
# ===========================================================================
# fixed field order inside the per-mon integer row (34 wide)
_MI = {n: i for i, n in enumerate(
    ["hp", "maxhp", "level", "attack", "defense", "special_attack",
     "special_defense", "speed", "status", "rest_turns", "sleep_turns",
     "times_attacked", "active_move_actions", "stellar", "revmask",
     "terastallized", "revealed", "illusion_broken", "known", "opbau"]
    + ["disabled%d" % i for i in range(4)]
    + ["ev%d" % i for i in range(6)]
    + ["pp%d" % i for i in range(4)])}
N_MI = len(_MI)
# fixed field order inside the per-side integer row (23 + 19 sc + 13 dur)
_SI = {n: i for i, n in enumerate(
    ["active_index", "substitute_health", "attack_boost", "defense_boost",
     "special_attack_boost", "special_defense_boost", "speed_boost",
     "accuracy_boost", "evasion_boost", "wish0", "wish1", "fs0", "fs1",
     "force_switch", "force_trapped", "slow_uturn_move", "baton_passing",
     "shed_tailing", "revival_blessing", "last_move_failed", "times_revived",
     "lum_kind", "lum_slot"]
    + ["sc%d" % i for i in range(N_SC)] + ["dur%d" % i for i in range(N_DUR)])}
N_SI = len(_SI)
_SC0, _DUR0 = _SI["sc0"], _SI["dur0"]
_GI = {n: i for i, n in enumerate(
    ["weather", "weather_turns", "terrain", "terrain_turns", "trick_room",
     "trick_room_turns", "team_preview"])}
N_GI = len(_GI)

_DEFAULT_EVS = (85,) * 6


def _lookup(vocab, table):
    """Hot-path categorical id: raw dict hit, falling back to the reference
    Vocab (which upper-cases and RECORDS the miss) so nothing is ever silently
    folded into UNK."""
    d = vocab.d[table]

    def g(name, _d=d, _t=table, _v=vocab):
        i = _d.get(name)
        return vocab.id(_t, name) if i is None else i
    return g


def parse_batch(states, vocab):
    """list[state string] -> columnar dict. Mirrors lossless_encoder.parse_state
    field index for field index; the gate proves the two agree on every field of
    every state."""
    n = len(states)
    v_spec, v_item = _lookup(vocab, "species"), _lookup(vocab, "item")
    v_abil, v_move = _lookup(vocab, "ability"), _lookup(vocab, "move")
    tix, six_, nix = TYPE_IX, STATUS_IX, NATURE_IX
    vix, wix, terix = VOLATILE_IX, WEATHER_IX, TERRAIN_IX
    lkix = LUM_KIND_IX

    mi, mid, mw, si, sid, gi, volflat = [], [], [], [], [], [], []
    r = 0                                     # running (state, side) pair index
    for s in states:
        top = s.split("/")
        for sidx in (0, 1):
            f = top[sidx].split("=")
            for pi in range(6):
                p = f[pi].split(",")
                nf = len(p)
                evs = p[12]
                ev = tuple(int(x) for x in evs.split(";")) if evs != "" else _DEFAULT_EVS
                mv = [p[22 + i].split(";") for i in range(4)]
                mi.extend((
                    int(p[6]), int(p[7]), int(p[1]), int(p[13]), int(p[14]),
                    int(p[15]), int(p[16]), int(p[17]), six_[p[18]],
                    int(p[19]), int(p[20]),
                    int(p[31]) if nf > 31 else 0,
                    int(p[34]) if nf > 34 else 0,
                    int(p[35]) if nf > 35 else 0,
                    int(p[36]) if nf > 36 else 0,
                    p[26] == "true",
                    p[28] == "true" if nf > 28 else True,
                    p[29] == "true" if nf > 29 else False,
                    p[30] == "true" if nf > 30 else True,
                    p[33] == "true" if nf > 33 else False,
                    mv[0][1] == "true", mv[1][1] == "true",
                    mv[2][1] == "true", mv[3][1] == "true",
                    ev[0], ev[1], ev[2], ev[3], ev[4], ev[5],
                    int(mv[0][2]), int(mv[1][2]), int(mv[2][2]), int(mv[3][2])))
                mid.extend((
                    v_spec(p[0]), v_item(p[10]), v_abil(p[8]), v_abil(p[9]),
                    v_item(p[32] if nf > 32 else "NONE"), nix[p[11]], tix[p[27]],
                    tix[p[2]], tix[p[3]], tix[p[4]], tix[p[5]],
                    v_move(mv[0][0]), v_move(mv[1][0]), v_move(mv[2][0]), v_move(mv[3][0])))
                mw.append(float(p[21]))

            lum = f[28].split(":")
            if lum[0] == "switch":
                lk, ls, lc = 2, int(lum[1]), None
            elif lum[1] == "none":
                lk, ls, lc = 0, 0, None
            elif lum[1] == "struggle":
                lk, ls, lc = 3, 0, None
            elif lum[1] == "unslotted":
                lk, ls, lc = 4, 0, (lum[2] if len(lum) > 2 else "NONE")
            else:
                lk, ls, lc = 1, int(lum[1]), None
            nsf = len(f)
            dur = f[9].split(";")
            row = [
                int(f[6]), int(f[10]), int(f[11]), int(f[12]), int(f[13]),
                int(f[14]), int(f[15]), int(f[16]), int(f[17]), int(f[18]),
                int(f[19]), int(f[20]), int(f[21]),
                f[22] == "true", f[27] == "true", f[29] == "true",
                f[24] == "true", f[25] == "true", f[26] == "true",
                (f[31] == "true") if nsf > 31 else False,
                int(f[30]) if nsf > 30 else 0, lk, ls]
            row.extend(int(x) for x in f[7].split(";"))
            row.extend(int(dur[i]) if i < len(dur) and dur[i] != "" else 0
                       for i in range(N_DUR))
            si.extend(row)
            sid.extend((v_move(f[23]), v_move(lc) if lc is not None else 0))
            base = r * N_VOLATILES
            for x in f[8].split(":"):
                if x != "":
                    j = vix.get(x)
                    volflat.append(base + (vix[x.upper()] if j is None else j))
            r += 1

        w, t, tr = top[2].split(";"), top[3].split(";"), top[4].split(";")
        gi.extend((wix[w[0]], int(w[1]), terix[t[0]], int(t[1]),
                   tr[0] == "true", int(tr[1]), top[5] == "true"))

    vol = np.zeros(n * 2 * N_VOLATILES, bool)
    if volflat:
        vol[np.asarray(volflat, np.int64)] = True
    return {
        "n": n,
        "mi": np.asarray(mi, np.int32).reshape(n, 2, 6, N_MI),
        "mid": np.asarray(mid, np.int32).reshape(n, 2, 6, N_MON_IDS),
        "mw": np.asarray(mw, np.float64).reshape(n, 2, 6),
        "si": np.asarray(si, np.int32).reshape(n, 2, N_SI),
        "sid": np.asarray(sid, np.int32).reshape(n, 2, N_SIDE_IDS),
        "vol": vol.reshape(n, 2, N_VOLATILES),
        "gi": np.asarray(gi, np.int32).reshape(n, N_GI),
    }


# ===========================================================================
# 3. vectorised encoder
# ===========================================================================
def _slot_order(active):
    """order[n, side, slot] = party index. slot 0 = active, 1..5 = the rest in
    ascending party order (spec section 5.1 rule 2). Bijective given
    active_index, which is why every PokemonIndex payload round-trips."""
    n = active.shape[0]
    o = np.empty((n, 2, 6), np.int64)
    o[:, :, 0] = active
    c = np.broadcast_to(np.arange(5, dtype=np.int64), (n, 2, 5))
    a = active[:, :, None]
    o[:, :, 1:] = np.where(c < a, c, c + 1)
    return o


def _party_to_slot(party, active):
    """Inverse of _slot_order, elementwise."""
    return np.where(party == active, 0, np.where(party < active, party + 1, party))


def _gather(x, order):
    """(n,2,6[,k]) in party order -> (n,12[,k]) in slot order."""
    if x.ndim == 3:
        out = np.take_along_axis(x, order, axis=2)
    else:
        out = np.take_along_axis(x, order[..., None], axis=2)
    return out.reshape((x.shape[0], 12) + x.shape[3:])


def _scatter(dst, blk, idx, val):
    """One-hot write of `val` into dst[..., blk[0]+idx]."""
    np.put_along_axis(dst, (blk[0] + idx)[..., None], val[..., None], axis=-1)


def encode_columnar(C, vocab, variant="full"):
    """Columnar batch -> (ids int32[n, N_IDS], feats float32[n, N_FEATS])."""
    V = VARIANTS[variant]
    n = C["n"]
    M, S, G = V.M, V.S, V.G
    MB, SB, GB = V.mon_blocks, V.side_blocks, V.glob_blocks
    mi, si, gi = C["mi"], C["si"], C["gi"]
    order = _slot_order(si[:, :, _SI["active_index"]])

    # ---------------- per-mon ----------------
    m = _gather(mi, order).astype(np.float64)          # (n,12,N_MI)
    mid = _gather(C["mid"], order)                     # (n,12,15)
    mw = _gather(C["mw"], order)                       # (n,12)
    maxhp = m[:, :, _MI["maxhp"]]
    occ = maxhp != 0
    o = occ.astype(np.float64)
    mon = np.zeros((n, 12, V.NM), np.float32)
    ids = np.zeros((n, V.N_IDS), np.int32)
    ids[:, :12 * N_MON_IDS] = (mid * occ[:, :, None]).reshape(n, -1)

    def put(name, val):
        mon[:, :, M[name]] = val

    put("hp", m[:, :, _MI["hp"]] / LE.D_HP * o)
    put("maxhp", maxhp / LE.D_HP * o)
    put("level", m[:, :, _MI["level"]] / LE.D_LEVEL * o)
    for nm in ("attack", "defense", "special_attack", "special_defense", "speed"):
        put(nm, m[:, :, _MI[nm]] / LE.D_STAT * o)
    put("weight_kg", mw / LE.D_WEIGHT * o)
    a, b = MB["ev"]
    mon[:, :, a:b] = m[:, :, _MI["ev0"]:_MI["ev0"] + 6] / LE.D_EV * o[:, :, None]
    _scatter(mon, MB["status"], m[:, :, _MI["status"]].astype(np.int64), o)
    put("rest_turns", m[:, :, _MI["rest_turns"]] / LE.D_REST * o)
    put("sleep_turns", m[:, :, _MI["sleep_turns"]] / LE.D_SLEEP * o)
    ta = m[:, :, _MI["times_attacked"]]
    ama = m[:, :, _MI["active_move_actions"]]
    put("times_attacked", ta / LE.D_TIMES_ATTACKED * o)          # UNCLIPPED
    put("active_move_actions", ama / LE.D_MOVE_ACTIONS * o)      # UNCLIPPED
    for nm, k in (("terastallized", "terastallized"), ("revealed", "revealed"),
                  ("illusion_broken", "illusion_broken"), ("known", "known"),
                  ("once_per_battle_ability_used", "opbau")):
        put(nm, m[:, :, _MI[k]] * o)
    bits = np.arange(20, dtype=np.int64)
    a, b = MB["stellar"]
    mon[:, :, a:b] = ((m[:, :, _MI["stellar"]].astype(np.int64)[:, :, None] >> bits) & 1) * o[:, :, None]
    a, b = MB["revmask"]
    mon[:, :, a:b] = ((m[:, :, _MI["revmask"]].astype(np.int64)[:, :, None] >> bits[:8]) & 1) * o[:, :, None]
    pp = m[:, :, _MI["pp0"]:_MI["pp0"] + 4]
    a, b = MB["pp"]
    mon[:, :, a:b] = pp / LE.D_PP * o[:, :, None]
    a, b = MB["disabled"]
    mon[:, :, a:b] = m[:, :, _MI["disabled0"]:_MI["disabled0"] + 4] * o[:, :, None]
    put("slot_occupied", o)

    slot = np.broadcast_to(np.arange(12) % 6, (n, 12))
    if V.keep_derived:
        den = np.where(occ, maxhp, 1.0)
        put("hp_frac", np.maximum(m[:, :, _MI["hp"]], 0.0) / den * o)
        mp = move_maxpp_table(vocab)[mid[:, :, 11:15]]
        a, b = MB["ppfrac"]
        mon[:, :, a:b] = np.where(mp > 0, np.minimum(np.maximum(pp, 0.0), mp)
                                  / np.where(mp > 0, mp, 1.0), 0.0) * o[:, :, None]
        t1, t2 = mid[:, :, 7].astype(np.int64), mid[:, :, 8].astype(np.int64)
        tt = mid[:, :, 6].astype(np.int64)
        tera = m[:, :, _MI["terastallized"]] > 0.5
        stellar = tt == TYPE_IX["STELLAR"]
        use_tera = tera & ~stellar
        # 20 scratch columns so STELLAR/TYPELESS (indices 18,19) land outside
        # the 18 real ones, exactly as the reference's `TYPE_IX[t] < 18` guard.
        for blk, e1, e2 in (("type_eff", np.where(use_tera, tt, t1), np.where(use_tera, tt, t2)),
                            ("type_own", t1, t2)):
            scratch = np.zeros((n, 12, 20), np.float32)
            _scatter(scratch, (0, 20), e1, o)
            _scatter(scratch, (0, 20), e2, o)
            a, b = MB[blk]
            mon[:, :, a:b] = scratch[:, :, :18]
        put("tera_stellar", (tera & stellar) * o)
        put("is_active", (slot == 0).astype(np.float64))
        put("slot_index", slot / 5.0)
        put("perspective_side_two", np.broadcast_to(
            (np.arange(12) >= 6).astype(np.float64), (n, 12)))
        sb = si[:, :, _SI["attack_boost"]:_SI["attack_boost"] + 5] / LE.D_BOOST
        a, b = MB["boost"]
        mon[:, 0, a:b] = sb[:, 0] * o[:, 0:1]
        mon[:, 6, a:b] = sb[:, 1] * o[:, 6:7]
        put("log_weight", np.log1p(mw) / 7.0 * o)
        put("times_attacked_capped", np.minimum(ta, 6) / 6.0 * o)
        put("active_move_actions_capped", np.minimum(ama, 2) / 2.0 * o)

    # ---------------- per-side ----------------
    side = np.zeros((n, 2, V.NS), np.float32)
    act = si[:, :, _SI["active_index"]].astype(np.int64)
    _scatter(side, SB["active"], act, np.ones((n, 2)))
    a, b = SB["sc"]
    side[:, :, a:b] = si[:, :, _SC0:_SC0 + N_SC] / LE.D_SIDECOND
    a, b = SB["vol"]
    vol = C["vol"]
    side[:, :, a:b] = vol
    a, b = SB["dur"]
    side[:, :, a:b] = si[:, :, _DUR0:_DUR0 + N_DUR] / LE.D_DURATION
    side[:, :, S["substitute_health"]] = si[:, :, _SI["substitute_health"]] / LE.D_SUB
    a, b = SB["boost"]
    side[:, :, a:b] = si[:, :, _SI["attack_boost"]:_SI["attack_boost"] + 7] / LE.D_BOOST
    side[:, :, S["wish_turns"]] = si[:, :, _SI["wish0"]] / LE.D_WISH_TURNS
    side[:, :, S["wish_amount"]] = si[:, :, _SI["wish1"]] / LE.D_WISH_AMOUNT
    side[:, :, S["future_sight_turns"]] = si[:, :, _SI["fs0"]] / LE.D_FS_TURNS
    _scatter(side, SB["fs_src"], _party_to_slot(si[:, :, _SI["fs1"]].astype(np.int64), act),
             np.ones((n, 2)))
    a, b = SB["flags"]
    side[:, :, a:b] = si[:, :, [_SI[k] for k in
                               ("force_switch", "force_trapped", "slow_uturn_move",
                                "baton_passing", "shed_tailing", "revival_blessing",
                                "last_move_failed")]]
    side[:, :, S["times_revived"]] = si[:, :, _SI["times_revived"]] / LE.D_TIMES_REVIVED
    lk = si[:, :, _SI["lum_kind"]].astype(np.int64)
    ls = si[:, :, _SI["lum_slot"]].astype(np.int64)
    _scatter(side, SB["lum_kind"], lk, np.ones((n, 2)))
    # conditional one-hots: a scratch column absorbs the write when the kind
    # does not apply, so no python-level branch is needed.
    sc4 = np.zeros((n, 2, 5), np.float32)
    _scatter(sc4, (0, 5), np.where(lk == 1, ls, 4), np.ones((n, 2)))
    a, b = SB["lum_move"]
    side[:, :, a:b] = sc4[:, :, :4]
    sc6 = np.zeros((n, 2, 7), np.float32)
    _scatter(sc6, (0, 7), np.where(lk == 2, _party_to_slot(ls, act), 6), np.ones((n, 2)))
    a, b = SB["lum_switch"]
    side[:, :, a:b] = sc6[:, :, :6]
    banked = C["sid"][:, :, 0]
    ids[:, 12 * N_MON_IDS:] = C["sid"].reshape(n, -1)
    side[:, :, S["has_banked_move"]] = banked != vocab.id("move", "NONE")

    if V.keep_derived:
        hp12 = _gather(mi[:, :, :, _MI["hp"]], order).reshape(n, 2, 6)
        mh12 = _gather(mi[:, :, :, _MI["maxhp"]], order).reshape(n, 2, 6)
        side[:, :, S["alive"]] = ((hp12 > 0) & (mh12 > 0)).sum(axis=2) / 6.0
        ter12 = _gather(mi[:, :, :, _MI["terastallized"]], order).reshape(n, 2, 6)
        side[:, :, S["tera_used"]] = ter12.any(axis=2)
        per = np.zeros((n, 2), np.float64)
        for name, k in LE.PERISH.items():
            per = np.maximum(per, vol[:, :, VOLATILE_IX[name]] * k)
        side[:, :, S["perish_count"]] = per / 4.0
        a, b = SB["paradox"]
        for i, st in enumerate(LE.PARADOX_STATS):
            side[:, :, a + i] = (vol[:, :, VOLATILE_IX["PROTOSYNTHESIS" + st]]
                                 | vol[:, :, VOLATILE_IX["QUARKDRIVE" + st]])
        tt_ix = [VOLATILE_IX[v] for v in sorted(LE.TWO_TURN)]
        side[:, :, S["two_turn_move"]] = vol[:, :, tt_ix].any(axis=2)
        # active mon is slot 0 of each side after the gather
        den = mh12[:, :, 0] / 4.0
        sub = si[:, :, _SI["substitute_health"]]
        has_sub = vol[:, :, VOLATILE_IX["SUBSTITUTE"]] & (den > 0)
        side[:, :, S["substitute_fraction"]] = np.where(
            has_sub, np.clip(sub / np.where(den > 0, den, 1.0), 0.0, 1.0), 0.0)

    # ---------------- global ----------------
    glob = np.zeros((n, V.NG), np.float32)
    _scatter(glob, GB["weather"], gi[:, _GI["weather"]].astype(np.int64), np.ones(n))
    wt = gi[:, _GI["weather_turns"]]
    glob[:, G["weather_turns"]] = (wt + 1) / LE.D_WEATHER_TURNS
    _scatter(glob, GB["terrain"], gi[:, _GI["terrain"]].astype(np.int64), np.ones(n))
    glob[:, G["terrain_turns"]] = (gi[:, _GI["terrain_turns"]] + 1) / LE.D_WEATHER_TURNS
    glob[:, G["trick_room"]] = gi[:, _GI["trick_room"]]
    glob[:, G["trick_room_turns"]] = (gi[:, _GI["trick_room_turns"]] + 1) / LE.D_TR_TURNS
    glob[:, G["team_preview"]] = gi[:, _GI["team_preview"]]
    if V.keep_derived:
        glob[:, G["weather_permanent"]] = (gi[:, _GI["weather"]] != WEATHER_IX["NONE"]) & (wt < 0)

    feats = np.concatenate(
        [mon.reshape(n, 12 * V.NM), side.reshape(n, 2 * V.NS), glob], axis=1)
    return ids, feats


def encode_states(states, vocab, variant="full", chunk=8192):
    """list[state string] -> (ids int32[n,·], feats float32[n,·]). THE entry
    point; `chunk` bounds peak memory so a million-row call is safe."""
    if len(states) <= chunk:
        return encode_columnar(parse_batch(states, vocab), vocab, variant)
    parts = [encode_columnar(parse_batch(states[i:i + chunk], vocab), vocab, variant)
             for i in range(0, len(states), chunk)]
    return (np.concatenate([p[0] for p in parts]),
            np.concatenate([p[1] for p in parts]))


# ===========================================================================
# 4. decoder -- the reference decoder, reached by re-expanding a lean vector
# ===========================================================================
def to_full(feats, variant):
    """lean (or full) feature rows -> full-layout rows. DERIVED columns are left
    at 0; the decoder reads EXACT columns only, which is the whole claim."""
    V = VARIANTS[variant]
    if V.keep_derived:
        return feats
    feats = np.atleast_2d(feats)
    n = feats.shape[0]
    F = VARIANTS["full"]
    out = np.zeros((n, F.N_FEATS), np.float32)
    mon = feats[:, :12 * V.NM].reshape(n, 12, V.NM)
    side = feats[:, 12 * V.NM:12 * V.NM + 2 * V.NS].reshape(n, 2, V.NS)
    glob = feats[:, 12 * V.NM + 2 * V.NS:]
    omon = np.zeros((n, 12, F.NM), np.float32)
    oside = np.zeros((n, 2, F.NS), np.float32)
    oglob = np.zeros((n, F.NG), np.float32)
    omon[:, :, V.mon_to_full] = mon
    oside[:, :, V.side_to_full] = side
    oglob[:, V.glob_to_full] = glob
    out[:] = np.concatenate(
        [omon.reshape(n, 12 * F.NM), oside.reshape(n, 2 * F.NS), oglob], axis=1)
    return out


def decode(ids_row, feats_row, vocab, variant="full"):
    """One row -> the same dict shape `lossless_encoder.parse_state` produces."""
    return LE.decode(np.asarray(ids_row).ravel(),
                     to_full(np.asarray(feats_row), variant).ravel(), vocab)


def widths(variant="full"):
    V = VARIANTS[variant]
    return {"variant": V.name, "mon_feats": V.NM, "side_feats": V.NS,
            "global_feats": V.NG, "feats_total": V.N_FEATS, "ids_total": V.N_IDS}


if __name__ == "__main__":
    for v in ("full", "lean"):
        print(widths(v))
