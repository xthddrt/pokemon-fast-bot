"""FROZEN pre-change copy of ENCODER 2 -- the evaluator encoder of EVALUATOR_SPEC.md, implemented as written.

  §1 per-Pokemon block x12    §2 per-side block x2     §3 relational block
  §4 counterfactual block     §5 global block

WHAT IS REUSED AND WHAT IS NEW
  The state-string PARSER is `llencoder.parse_batch` -- a columnar, vectorised
  parser proven field-for-field against `lossless_encoder.parse_state` by
  `ll_gate.py`. Nothing about parsing is re-derived here. Everything above the
  parse is new: this is a different, leaner, RELATIONAL feature set, not a
  lossless one, and it shares no column with the lossless layout.
  The damage kernel and the randbats pool come from `dmgtab.py` / `rbpool.py`.

THE STATIC / DYNAMIC SPLIT (the spec's cost rule)
  Inside one search the twelve Pokemon never change: species, moves, item,
  ability, base stats, types and tera type are fixed. Everything that depends
  only on those is `StaticCtx`, built ONCE per search. Per leaf we recompute
  only what HP, boosts, status, volatiles and field state change -- and the
  spec's own trick makes that cheap: the static half stores best damage as a
  fraction of the target's FULL hp, split by category, so a leaf applies
      dmg = static_phys * atk_boost / def_boost * reflect * burn      (physical)
          | static_spec * spa_boost / spd_boost * light_screen        (special)
  and a KO count is one division. `encode_states(..., share_static=True)` is
  that path; `enc2_gate.py cost` measures both halves.

DESIGN RULES HONOURED (spec "Design rules")
  1. No pre-aggregation: every per-mon and per-pair quantity keeps its own
     column. The 13 capability bools live on the MON, not the side.
  2. Completed comparisons only: KO counts, speed order and sweep flags are
     finished booleans/one-hots, never raw stats for the net to subtract.
  3. Nothing that is a deterministic function of something else encoded:
     species, level, EVs, nature, weight and base types have no columns.
  4. Nothing unreachable in gen9 randbats: the volatile columns are exactly
     `rbpool.reachable_list()` (44 of the engine's 107), and Gravity is dropped
     from §5 because no randbats move or ability sets it.
  5. Resources priced by best use: tera is 2 columns per side, not a matrix.
  Fainted mons: excluded from every matchup, sweep, killability and capability
  sum (their bools read 0); they still count in `n_fainted_revivable`.

ASSUMPTIONS, STATED (each one is a judgement the spec left open)
  * KO counts use the MAXIMUM damage roll ("can this kill"), the same roll the
    engine's `calculate_damage` returns. The strict sweep conjunction instead
    uses the MINIMUM roll, so "sweep" means GUARANTEED sweep.
  * The KO matrix is CURRENT-hp only. The full-hp (intrinsic-matchup) version
    the spec also asks for was dropped: "can I kill it now" is what nearly every
    decision turns on, and the intrinsic matchup is largely recoverable from the
    stats, types and moves already encoded. `Layout(ko_versions=2)` restores it.
  * Damage is computed as TWO 6x6 CROSS-SIDE blocks, never a 12x12: of the 144
    ordered slot pairs, 72 are same-side and no column ever read them.
  * `mons-required-to-kill` counts an opponent as able to act if it is faster,
    or holds a damaging priority move, or survives the setup mon's best hit.
  * A bench Pokemon has no boosts and no volatiles of its own (they belong to
    the active), but does get side-wide effects: Tailwind, screens, weather.
  * Accuracy is not modelled anywhere; a 70%-accurate OHKO reads as an OHKO.
  * §3 says "all 36 ordered pairs". 36 is 6x6, which covers one direction only
    and would tell the net who we kill but not who kills us. BOTH directions are
    encoded (see `Layout.ko_directions`); set it to 1 to get the literal
    reading. This is flagged in ENCODER2_BUILD.md, not decided silently.
"""

import os
import sys
import time

import numpy as np

LAB = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LAB)

import labenv  # noqa: F401,E402  (pins sys.path and the engine/encoder flags)
import lossless_encoder as LE  # noqa: E402
import llencoder as LL  # noqa: E402
import dmgtab as DT  # noqa: E402
import rbpool  # noqa: E402

_MI, _SI, _GI = LL._MI, LL._SI, LL._GI
STATUS_ORDER = LE.STATUS_ORDER
WEATHER_ORDER, TERRAIN_ORDER = LE.WEATHER_ORDER, LE.TERRAIN_ORDER
N_TYPES, TYPELESS = DT.N_TYPES, DT.TYPELESS

# Design rule 4: HARSHSUN and HEAVYRAIN need Primordial Sea / Desolate Land,
# neither of which is in the gen9 randbats ability pool, so they get no column.
# They fold onto SUN / RAIN (which is what they are, only stronger) rather than
# being dropped, so an unexpected one can never be silently lost.
WEATHER_COLS = [w for w in WEATHER_ORDER if w not in ("HARSHSUN", "HEAVYRAIN")]
# Weather changes damage (Sun/Rain by 50%) and defensive stats (Sand: Rock SpD,
# Snow: Ice Def), and it CHANGES INSIDE A SEARCH, so the static damage table is
# built once per weather variant and the leaf selects one. Five variants, built
# once per search; a leaf pays one gather. Measured: without this, sun/rain
# positions were 25%% wrong on the pool-wide gate.
WEATHER_VARIANTS = ["NONE", "SUN", "RAIN", "SAND", "SNOW"]
_WV = {"NONE": 0, "SUN": 1, "HARSHSUN": 1, "RAIN": 2, "HEAVYRAIN": 2,
       "SAND": 3, "SNOW": 4, "HAIL": 0}
WEATHER_VAR_MAP = np.array([_WV[w] for w in WEATHER_ORDER], np.int64)
WEATHER_MAP = np.array([WEATHER_COLS.index({"HARSHSUN": "SUN",
                                            "HEAVYRAIN": "RAIN"}.get(w, w))
                        for w in WEATHER_ORDER], np.int64)

# ---- §2 volatile columns: exactly the reachable set, gate 1 ----------------
# FROZEN REFERENCE COPY of enc2.py (pre-change), used only by enc2_equiv.py
# to prove bit-identity. The volatile list is PINNED so that changing
# rbpool.py cannot move the reference.
VOL_COLS = ['CONFUSION', 'CUDCHEW', 'CURSE', 'DESTINYBOND', 'DISABLE', 'ENCORE', 'FLASHFIRE', 'FLINCH', 'GLAIVERUSH', 'HEALBLOCK', 'LEECHSEED', 'LIGHTSCREEN', 'LOCKEDMOVE', 'MAGNETRISE', 'METEORBEAM', 'MUSTRECHARGE', 'NORETREAT', 'PARTIALLYTRAPPED', 'PROTECT', 'PROTOSYNTHESISATK', 'PROTOSYNTHESISDEF', 'PROTOSYNTHESISSPA', 'PROTOSYNTHESISSPD', 'PROTOSYNTHESISSPE', 'QUARKDRIVEATK', 'QUARKDRIVEDEF', 'QUARKDRIVESPA', 'QUARKDRIVESPD', 'QUARKDRIVESPE', 'REFLECT', 'ROOST', 'SALTCURE', 'SLOWSTART', 'SOLARBEAM', 'SPARKLINGARIA', 'SUBSTITUTE', 'TAUNT', 'THROATCHOP', 'TRANSFORM', 'TRAPPED', 'TRUANT', 'TYPECHANGE', 'UNBURDEN', 'YAWN']
VOL_IX = np.array([LE.VOLATILE_IX[v] for v in VOL_COLS], np.int64)
# durations, restricted to volatiles that are reachable
DUR_COLS = [d for d in LE.DURATION_FIELDS
            if d.upper().replace("_", "") in set(VOL_COLS)]
DUR_IX = np.array([LE.DURATION_FIELDS.index(d) for d in DUR_COLS], np.int64)
SC = {n: i for i, n in enumerate(LE.SIDE_CONDITION_FIELDS)}

D_STAT_EFF = 2.0 * LE.D_STAT     # effective stats can be 4x the raw stat
KO_BUCKETS = ("ohko", "2hko", "3hko", "4plus", "never")
MTK_BUCKETS = ("1", "2", "3", "4plus", "never")

# ---------------------------------------------------------------------------
# capability move sets (§2, 13 bools per mon)
# ---------------------------------------------------------------------------
CAP_NAMES = ["hazard_removal", "hazard_setting", "recovery", "priority",
             "setup", "phazing", "haze_unaware", "trapping", "trick_room",
             "weather_setter", "terrain_setter", "revival", "healing_wish"]
CAP_MOVES = {
    "hazard_removal": {"RAPIDSPIN", "DEFOG", "MORTALSPIN", "TIDYUP", "COURTCHANGE"},
    "hazard_setting": {"STEALTHROCK", "SPIKES", "TOXICSPIKES", "STICKYWEB",
                       "CEASELESSEDGE", "STONEAXE"},
    "recovery": {"RECOVER", "ROOST", "SOFTBOILED", "SLACKOFF", "MOONLIGHT",
                 "MORNINGSUN", "SYNTHESIS", "SHOREUP", "MILKDRINK", "REST",
                 "WISH", "PAINSPLIT", "STRENGTHSAP", "JUNGLEHEALING",
                 "LUNARBLESSING", "PURIFY", "HEALORDER"},
    "phazing": {"WHIRLWIND", "ROAR", "DRAGONTAIL", "CIRCLETHROW"},
    "haze_unaware": {"HAZE", "CLEARSMOG"},
    "trapping": {"BLOCK", "MEANLOOK", "SPIDERWEB", "JAWLOCK", "OCTOLOCK",
                 "THOUSANDWAVES", "ANCHORSHOT", "SPIRITSHACKLE"},
    "trick_room": {"TRICKROOM"},
    "weather_setter": {"SUNNYDAY", "RAINDANCE", "SANDSTORM", "SNOWSCAPE",
                       "CHILLYRECEPTION", "HAIL"},
    "terrain_setter": {"ELECTRICTERRAIN", "GRASSYTERRAIN", "MISTYTERRAIN",
                       "PSYCHICTERRAIN"},
    "revival": {"REVIVALBLESSING"},
    "healing_wish": {"HEALINGWISH", "LUNARDANCE"},
}
CAP_ABILITIES = {
    "recovery": {"REGENERATOR"},
    "haze_unaware": {"UNAWARE"},
    "trapping": {"ARENATRAP", "SHADOWTAG", "MAGNETPULL"},
    "weather_setter": {"DROUGHT", "DRIZZLE", "SANDSTREAM", "SNOWWARNING",
                       "ORICHALCUMPULSE", "DESOLATELAND", "PRIMORDIALSEA",
                       "SANDSPIT"},
    "terrain_setter": {"ELECTRICSURGE", "GRASSYSURGE", "MISTYSURGE",
                       "PSYCHICSURGE", "HADRONENGINE", "SEEDSOWER"},
}
PRANKSTER, GALEWINGS, TRIAGE = "PRANKSTER", "GALEWINGS", "TRIAGE"
WEATHER_SPEED_ABILITY = {"CHLOROPHYLL": ("SUN", "HARSHSUN"),
                         "SWIFTSWIM": ("RAIN", "HEAVYRAIN"),
                         "SANDRUSH": ("SAND",), "SLUSHRUSH": ("SNOW", "HAIL")}


# ===========================================================================
# 1. vocab-indexed tables (built once per vocab, cached)
# ===========================================================================
class Tables:
    """Every per-move / per-ability / per-item quantity, as arrays indexed by
    the vocab id the parser produces. One gather replaces every dict lookup."""

    _cache = {}

    def __new__(cls, vocab):
        t = cls._cache.get(id(vocab))
        if t is None:
            t = super().__new__(cls)
            t._init(vocab)
            cls._cache[id(vocab)] = t
        return t

    def _init(self, vocab):
        mt = DT.moves()
        self.mt = mt
        dmove, dabil, ditem = vocab.d["move"], vocab.d["ability"], vocab.d["item"]
        nm, na, ni = max(dmove.values()) + 1, max(dabil.values()) + 1, max(ditem.values()) + 1
        # ---- moves -------------------------------------------------------
        loc = np.zeros(nm, np.int32)
        for name, i in dmove.items():
            loc[i] = mt.ix.get(name.upper(), 0)
        self.mv_local = loc
        for f in ("bp", "type", "cat", "prio", "flags", "secondary", "hits",
                  "hits_dice", "bp_kind", "unmodelled", "off_is_def",
                  "def_is_phys", "off_is_target", "pp"):
            setattr(self, "mv_" + f, getattr(mt, f)[loc])
        self.mv_boosts = mt.boosts[loc]                     # (nm, 7)
        self.mv_damaging = mt.damaging[loc]
        # setup move: a status move that raises its user's atk / spa / spe.
        off = self.mv_boosts[:, [0, 2, 4]].max(axis=1)
        self.mv_is_setup = (self.mv_cat == 2) & (off >= 1)
        self.mv_setup_rank = np.where(self.mv_is_setup,
                                      self.mv_boosts[:, :5].clip(0).sum(axis=1)
                                      + 0.5 * off, 0.0)
        # capability bits per move
        self.mv_cap = np.zeros((nm, len(CAP_NAMES)), np.float32)
        for name, i in dmove.items():
            u = name.upper()
            for c, cn in enumerate(CAP_NAMES):
                if u in CAP_MOVES.get(cn, ()):
                    self.mv_cap[i, c] = 1.0
        heal = mt.heals[loc]
        self.mv_cap[:, CAP_NAMES.index("recovery")] = np.maximum(
            self.mv_cap[:, CAP_NAMES.index("recovery")], (heal > 0) & (self.mv_cat == 2))
        self.mv_cap[:, CAP_NAMES.index("setup")] = self.mv_is_setup
        self.mv_cap[:, CAP_NAMES.index("priority")] = (self.mv_prio > 0) & self.mv_damaging
        self.mv_is_status = self.mv_cat == 2
        self.mv_is_heal = heal > 0
        # ---- abilities ---------------------------------------------------
        anames = [""] * na
        for name, i in dabil.items():
            anames[i] = name.upper()
        self.ab = {k: v for k, v in DT.ability_codes(anames).items()}
        self.ab_cap = np.zeros((na, len(CAP_NAMES)), np.float32)
        for i, u in enumerate(anames):
            for c, cn in enumerate(CAP_NAMES):
                if u in CAP_ABILITIES.get(cn, ()):
                    self.ab_cap[i, c] = 1.0
        self.ab_prankster = np.array([u == PRANKSTER for u in anames], np.int8)
        self.ab_galewings = np.array([u == GALEWINGS for u in anames], np.int8)
        self.ab_triage = np.array([u == TRIAGE for u in anames], np.int8)
        self.ab_quickfeet = np.array([u == "QUICKFEET" for u in anames], np.int8)
        self.ab_weather_speed = np.zeros((na, len(WEATHER_ORDER)), np.int8)
        for i, u in enumerate(anames):
            for w in WEATHER_SPEED_ABILITY.get(u, ()):
                self.ab_weather_speed[i, LE.WEATHER_IX[w]] = 1
        self.ab_surgesurfer = np.array([u == "SURGESURFER" for u in anames], np.int8)
        self.ab_slowstart = np.array([u == "SLOWSTART" for u in anames], np.int8)
        # Supreme Overlord scales with fainted allies -> dynamic, not static.
        # It was 63%% of the residual damage error before it was modelled.
        self.ab_supreme = np.array([u == "SUPREMEOVERLORD" for u in anames], np.int8)
        # ---- items -------------------------------------------------------
        inames = [""] * ni
        for name, i in ditem.items():
            inames[i] = name.upper()
        self.it = {k: v for k, v in DT.item_codes(inames).items()}
        self.it_scarf = np.array([u == "CHOICESCARF" for u in inames], np.int8)
        self.it_choice = np.array([u in ("CHOICESCARF", "CHOICEBAND", "CHOICESPECS")
                                   for u in inames], np.int8)
        self.it_boots = np.array([u == "HEAVYDUTYBOOTS" for u in inames], np.int8)


# ===========================================================================
# 2. layout
# ===========================================================================
class Layout:
    """Column names in order. The order IS the input vector (spec:
    'flat, per-slot, no pooling'); nothing here is ever sorted."""

    def __init__(self, ko_directions=2, ko_versions=1):
        assert ko_directions in (1, 2) and ko_versions in (1, 2)
        self.ko_directions = ko_directions
        # ko_versions=1 keeps the CURRENT-hp matrix only. "Can I kill it now" is
        # what nearly every decision turns on; the intrinsic full-hp matchup is
        # largely recoverable from the stats, types and moves the net already
        # has, and it cost 360 columns. Set to 2 to restore it.
        self.ko_versions = ko_versions
        m = ["hp_frac", "maxhp", "attack", "defense", "sp_atk", "sp_def", "speed"]
        m += ["status_" + s.lower() for s in STATUS_ORDER]
        # DROPPED here, all confirmed dead by the pool-wide census:
        #   is_active     -- a deterministic function of the slot index
        #   slot_occupied -- always 1 (gen9 randbats teams are always six)
        #   toxic_counter -- the engine stores it per SIDE, so the ten bench
        #                    slots were structurally zero; it now lives on the
        #                    side block (2 columns instead of 12).
        m += ["sleep_rest_turns", "times_attacked", "terastallized", "alive"]
        m += ["pp_frac%d" % i for i in range(4)]
        m += ["disabled%d" % i for i in range(4)]
        m += ["seen_species", "seen_item", "seen_ability", "seen_tera"]
        m += ["seen_move%d" % i for i in range(4)]
        m += ["cap_" + c for c in CAP_NAMES]
        self.mon = m

        s = ["stealth_rock", "spikes", "toxic_spikes", "sticky_web",
             "reflect", "reflect_turns", "light_screen", "light_screen_turns",
             "aurora_veil", "aurora_veil_turns", "tailwind", "tailwind_turns",
             "safeguard", "mist"]
        s += ["boost_" + b for b in ("atk", "def", "spa", "spd", "spe", "acc", "eva")]
        s += ["vol_" + v.lower() for v in VOL_COLS]
        s += ["dur_" + d for d in DUR_COLS]
        s += ["toxic_counter", "substitute_hp_frac", "wish_turns", "wish_amount",
              "future_sight_turns", "future_sight_source"]
        s += ["lock_choice", "lock_encore", "lock_disable", "lock_multiturn",
              "just_switched_in", "force_switch", "force_trapped",
              "last_move_failed"]
        s += ["tera_available", "n_fainted_revivable",
              "best_revival_target_score", "times_revived"]
        self.side = s

        r = []
        dirs = ["us_to_them"] if ko_directions == 1 else ["us_to_them", "them_to_us"]
        for ver in ("now", "full")[:ko_versions]:
            for d in dirs:
                for i in range(6):
                    for j in range(6):
                        for b in KO_BUCKETS:
                            r.append("ko_%s_%s_%d%d_%s" % (ver, d, i, j, b))
        r += ["pairspeed_%d%d" % (i, j) for i in range(6) for j in range(6)]
        r += ["active_speed_margin", "active_speed_tie"]
        r += ["order_us%d_them%d" % (i, j) for i in range(4) for j in range(4)]
        r += ["prio_outranks_us", "prio_outranks_them",
              "revenge_priority_on_us", "revenge_priority_on_them"]
        r += ["mprio_s%d_m%d" % (k, i) for k in range(12) for i in range(4)]
        self.rel = r

        c = ["setup_present"]
        c += ["setup_boost_" + b for b in ("atk", "def", "spa", "spd", "spe")]
        for lv in (1, 2):
            c += ["s%d_ohko_count" % lv, "s%d_outspeed_count" % lv, "s%d_sweep" % lv]
            c += ["s%d_mtk_%s" % (lv, b) for b in MTK_BUCKETS]
        self.cf_mon = c
        self.cf_side = ["best_sweep_1", "best_sweep_2", "free_turn",
                        "answers_to_best_threat", "tera_best_value",
                        "tera_enabled_sweep"]

        g = ["weather_" + w.lower() for w in WEATHER_COLS] + ["weather_turns"]
        g += ["terrain_" + t.lower() for t in TERRAIN_ORDER] + ["terrain_turns"]
        g += ["trick_room", "trick_room_turns"]
        self.glob = g

        self.NM, self.NS = len(m), len(s)
        self.NR, self.NCM, self.NCS = len(r), len(self.cf_mon), len(self.cf_side)
        self.NG = len(g)
        self.N = (12 * self.NM + 2 * self.NS + self.NR + 12 * self.NCM
                  + 2 * self.NCS + self.NG)
        self.MON_IDS = ["item", "ability", "type1", "type2", "tera_type",
                        "move0", "move1", "move2", "move3"]
        self.SIDE_IDS = ["locked_move", "charging_move"]
        self.N_IDS = 12 * len(self.MON_IDS) + 2 * len(self.SIDE_IDS)
        # offsets
        o = 0
        self.O_MON, o = o, o + 12 * self.NM
        self.O_SIDE, o = o, o + 2 * self.NS
        self.O_REL, o = o, o + self.NR
        self.O_CFM, o = o, o + 12 * self.NCM
        self.O_CFS, o = o, o + 2 * self.NCS
        self.O_GLOB, o = o, o + self.NG
        assert o == self.N
        self.names = self._names()

    def _names(self):
        out = []
        for k in range(12):
            out += ["mon%d.%s" % (k, n) for n in self.mon]
        for k in range(2):
            out += ["side%d.%s" % (k, n) for n in self.side]
        out += ["rel." + n for n in self.rel]
        for k in range(12):
            out += ["cf_mon%d.%s" % (k, n) for n in self.cf_mon]
        for k in range(2):
            out += ["cf_side%d.%s" % (k, n) for n in self.cf_side]
        out += ["glob." + n for n in self.glob]
        assert len(out) == self.N
        return out

    def block_sizes(self):
        return {"per_mon(x12)": 12 * self.NM, "per_side(x2)": 2 * self.NS,
                "relational": self.NR, "counterfactual_mon(x12)": 12 * self.NCM,
                "counterfactual_side(x2)": 2 * self.NCS, "global": self.NG,
                "TOTAL_numeric": self.N, "embedding_ids": self.N_IDS}


DEFAULT_LAYOUT = Layout()


# ===========================================================================
# 3. the static half -- once per search
# ===========================================================================
# damage channels: (category, offensive stat, defensive stat) -- see
# StaticCtx._pair_damage for why five and not two.
N_CHAN = 5
CHAN_PHYSICAL = np.array([1, 0, 0, 1, 1], bool)      # burn / Reflect apply
CHAN_SPECIAL = ~CHAN_PHYSICAL                        # Light Screen applies


def _channel_of(T, mvi):
    """move vocab ids -> damage channel, -1 for anything that cannot damage."""
    cat = T.mv_cat[mvi]
    ch = np.where(cat == 0, 0, np.where(cat == 1, 1, -1))
    ch = np.where((cat == 1) & (T.mv_def_is_phys[mvi] > 0), 2, ch)
    ch = np.where((cat == 0) & (T.mv_off_is_def[mvi] > 0), 3, ch)
    ch = np.where((cat == 0) & (T.mv_off_is_target[mvi] > 0), 4, ch)
    return ch


def _sides_of(v):
    """(n,12,...) slot-major -> (attacker (n,2,6,...), defender (n,2,6,...)).

    Block d holds side d's six mons as ATTACKERS and side 1-d's six as
    DEFENDERS, so the defender view is the side-swapped one. Both are views."""
    r = v.reshape((v.shape[0], 2, 6) + v.shape[2:])
    return r, r[:, ::-1]


def _channel_damage(dmg, att_stage, def_stage):
    """Apply the boost ratios each channel actually scales with.

    `dmg` is (n,2,6,6,5) static fractions of the defender's full HP -- the two
    CROSS-SIDE blocks only (see `StaticCtx._pair_damage`). `att_stage` /
    `def_stage` are (n,2,6,5) boost stages in atk/def/spa/spd/spe order, already
    split so that index [b] is the attacker / defender side of block b.
    -> (physical (n,2,6,6), special (n,2,6,6)) before burn/screens."""
    a_atk = _boost(att_stage[:, :, :, 0])[:, :, :, None]
    a_spa = _boost(att_stage[:, :, :, 2])[:, :, :, None]
    a_def = _boost(att_stage[:, :, :, 1])[:, :, :, None]
    d_atk = _boost(def_stage[:, :, :, 0])[:, :, None, :]
    d_def = _boost(def_stage[:, :, :, 1])[:, :, None, :]
    d_spd = _boost(def_stage[:, :, :, 3])[:, :, None, :]
    c0 = dmg[..., 0] * a_atk / d_def
    c1 = dmg[..., 1] * a_spa / d_spd
    c2 = dmg[..., 2] * a_spa / d_def
    c3 = dmg[..., 3] * a_def / d_def
    c4 = dmg[..., 4] * d_atk / d_def
    phys = np.maximum(np.maximum(c0, c3), c4)
    spec = np.maximum(c1, c2)
    return phys, spec, np.stack([c0, c1, c2, c3, c4], -1)


def _bucket_onehot(count, n_buckets, out, never_mask):
    """count in {1..}, bucketed 1,2,3,4+,never -> one-hot written into `out`."""
    b = np.clip(count.astype(np.int64), 1, n_buckets - 1) - 1
    b = np.where(never_mask, n_buckets - 1, b)
    np.put_along_axis(out, b[..., None], 1.0, axis=-1)
    return out


class StaticCtx:
    """Everything that does not change while a search deepens.

    Shapes carry a leading n (states). When `share_static=True` the caller
    builds this from ONE state and broadcasts, which is what a real search does.
    """

    __slots__ = ("n", "occ", "maxhp", "level", "stats", "weight", "t1", "t2",
                 "tera", "dt1", "dt2", "mv", "mv_ok", "mv_present", "prio",
                 "dmg", "dmg_prio", "dmg_tera",
                 "def_tera_mult", "cap", "setup_boost", "setup_present",
                 "base_speed", "scarf", "ability", "item", "A", "D", "T",
                 "pp_max", "eff_now")

    def __init__(self, C, T, order=None):
        n = C["n"]
        self.n, self.T = n, T
        # PARTY order, not slot order: the active can change inside a search
        # (that is what a switch is), and slot order depends on active_index.
        # A static context built in slot order silently decays the moment the
        # search switches. `reorder()` maps to slot order per leaf.
        m = C["mi"].reshape(n, 12, -1).astype(np.float32)       # (n,12,N_MI)
        mid = C["mid"].reshape(n, 12, -1).astype(np.int64)      # (n,12,15)
        self.weight = C["mw"].reshape(n, 12).astype(np.float32)
        self.maxhp = m[:, :, _MI["maxhp"]]
        self.occ = occ_ = self.maxhp > 0
        self.level = m[:, :, _MI["level"]]
        self.stats = np.stack([m[:, :, _MI[k]] for k in
                               ("attack", "defense", "special_attack",
                                "special_defense", "speed")], -1)     # (n,12,5)
        self.t1, self.t2 = mid[:, :, 7].astype(np.int16), mid[:, :, 8].astype(np.int16)
        self.tera = mid[:, :, 6].astype(np.int16)
        # poke-engine does NOT rewrite `types` when a mon terastallizes: it sets
        # `terastallized` and leaves the type list alone, applying the tera type
        # at damage time. So the DEFENSIVE typing has to be derived here, or
        # every KO count against a terastallized mon is wrong (measured: this
        # was a 100%-of-damage error on tera'd Steel types).
        tera_on = m[:, :, _MI["terastallized"]] > 0
        use_tera = tera_on & (self.tera != DT.TIX["STELLAR"])
        self.dt1 = np.where(use_tera, self.tera, self.t1).astype(np.int16)
        self.dt2 = np.where(use_tera, np.int16(DT.TYPELESS), self.t2).astype(np.int16)
        self.item, self.ability = mid[:, :, 1], mid[:, :, 2]
        self.mv = mid[:, :, 11:15]                                    # (n,12,4)
        not_disabled = m[:, :, _MI["disabled0"]:_MI["disabled0"] + 4] == 0
        # mv_present: the slot holds a real, usable move (status moves included)
        # mv_ok:      ... and it deals damage. The priority-bracket columns use
        # mv_present, because Prankster only ever matters on a status move.
        has_pp = m[:, :, _MI["pp0"]:_MI["pp0"] + 4] > 0
        self.mv_present = ((T.mv_local[self.mv] != 0) & not_disabled & has_pp
                           & occ_[:, :, None])
        self.mv_ok = T.mv_damaging[self.mv] & not_disabled & has_pp
        self.pp_max = np.maximum(T.mv_pp[self.mv], 1.0)

        # --- priority brackets, incl. Prankster / Gale Wings / Triage -----
        pr = T.mv_prio[self.mv].astype(np.float32)
        ab = self.ability
        pr = pr + (T.ab_prankster[ab][:, :, None] * T.mv_is_status[self.mv])
        flying = T.mv_type[self.mv] == DT.TIX["FLYING"]
        pr = pr + (T.ab_galewings[ab][:, :, None] * flying)
        pr = pr + 3 * (T.ab_triage[ab][:, :, None] * T.mv_is_heal[self.mv])
        self.prio = pr                                                # (n,12,4)

        # --- the static damage table: best damage as a fraction of the
        #     defender's FULL hp, per category, with no boosts, no screens,
        #     no weather, no burn and no Multiscale (all applied per leaf).
        #     Only the FIELD changes between weather variants, so the kernel's
        #     attacker/defender/move inputs are built once per tera variant
        #     rather than once per (tera, weather) pair -- at n=1 (one battle
        #     per search) this half is numpy call-overhead bound, not FLOP bound.
        kn, kt = self._kernel_inputs(m, False), self._kernel_inputs(m, True)
        dm, dp, dt = [], [], []
        for w in WEATHER_VARIANTS:
            a, b = self._pair_damage(kn, w)
            dm.append(a)
            dp.append(b)
            dt.append(self._pair_damage(kt, w)[0])
        self.dmg = np.stack(dm, 1)             # (n, 5, 2, 6, 6, N_CHAN)
        self.dmg_prio = np.stack(dp, 1)
        self.dmg_tera = np.stack(dt, 1)
        # defender-side tera: type multiplier of each attacker's best move type
        # if the DEFENDER terastallizes (used to price defensive tera)
        self.def_tera_mult = self._def_tera_mult()

        # --- capabilities (13 bools per mon), zeroed for empty slots ------
        cap = T.mv_cap[self.mv].max(axis=2) + T.ab_cap[ab]
        self.cap = np.clip(cap, 0.0, 1.0) * self.occ[:, :, None]

        # --- setup profile -------------------------------------------------
        rank = np.where(self.mv_ok | T.mv_is_setup[self.mv],
                        T.mv_setup_rank[self.mv], 0.0)
        best = rank.argmax(axis=2)                                    # (n,12)
        bm = np.take_along_axis(self.mv, best[:, :, None], axis=2)[:, :, 0]
        self.setup_present = ((np.take_along_axis(rank, best[:, :, None], axis=2)[:, :, 0] > 0)
                              & self.occ).astype(np.float32)
        self.setup_boost = (T.mv_boosts[bm][:, :, :5].astype(np.float32)
                            * self.setup_present[:, :, None])         # (n,12,5)

        # --- static speed: the stat plus the modifiers that do not move ----
        self.scarf = T.it_scarf[self.item].astype(np.float32)
        self.base_speed = self.stats[:, :, 4] * np.where(self.scarf > 0, 1.5, 1.0)

    # ------------------------------------------------------------------
    def _sides(self, m, tera_attacker):
        T = self.T
        A = {"level": self.level, "atk": self.stats[:, :, 0],
             "spa": self.stats[:, :, 2], "defense": self.stats[:, :, 1],
             "weight": self.weight, "stab1": self.t1, "stab2": self.t2,
             "tera_type": self.tera,
             "terastallized": np.int8(1) if tera_attacker else
                              m[:, :, _MI["terastallized"]].astype(np.int8),
             "hp": self.maxhp}
        D = {"defense": self.stats[:, :, 1], "spd": self.stats[:, :, 3],
             "atk": self.stats[:, :, 0], "maxhp": self.maxhp, "hp": self.maxhp,
             "t1": self.dt1, "t2": self.dt2, "weight": self.weight}
        for k, v in T.ab.items():
            A[k] = v[self.ability]
            D[k] = v[self.ability]
        for k, v in T.it.items():
            A[k] = v[self.item]
            D[k] = v[self.item]
        D["ab_multiscale"] = np.int8(0)     # applied per leaf (needs current hp)
        return A, D

    def _kernel_inputs(self, m, tera_attacker):
        """The damage kernel's attacker / defender / move arrays, shaped for the
        two cross-side blocks. Field-independent, so it is built once per tera
        variant and reused across the five weather variants.

        Axes: (n, block, attacker, move, defender). Every per-mon array is
        (n,12); `_sides_of` splits it into the attacker view (axis 2) and the
        side-swapped defender view (axis 4)."""
        T = self.T
        A0, D0 = self._sides(m, tera_attacker)
        A = {k: (_sides_of(v)[0][:, :, :, None, None]
                 if isinstance(v, np.ndarray) and v.ndim == 2 else v)
             for k, v in A0.items()}
        D = {k: (_sides_of(v)[1][:, :, None, None, :]
                 if isinstance(v, np.ndarray) and v.ndim == 2 else v)
             for k, v in D0.items()}
        mvi = self.mv
        dice = T.it["it_loaded_dice"][self.item] > 0
        M = {"bp": T.mv_bp[mvi], "type": T.mv_type[mvi], "cat": T.mv_cat[mvi],
             "flags": T.mv_flags[mvi], "secondary": T.mv_secondary[mvi],
             "hits": np.where(dice[:, :, None], T.mv_hits_dice[mvi], T.mv_hits[mvi]),
             "bp_kind": T.mv_bp_kind[mvi], "unmodelled": T.mv_unmodelled[mvi],
             "off_is_def": T.mv_off_is_def[mvi], "def_is_phys": T.mv_def_is_phys[mvi],
             "off_is_target": T.mv_off_is_target[mvi]}
        M = {k: _sides_of(v)[0][:, :, :, :, None] for k, v in M.items()}
        return dict(A=A, D=D, M=M,
                    def_maxhp=_sides_of(self.maxhp)[1][:, :, None, None, :],
                    usable=_sides_of(self.mv_ok & self.occ[:, :, None])[0][..., None],
                    chan=_sides_of(_channel_of(T, mvi))[0][..., None],
                    prio=_sides_of(self.prio)[0][..., None])

    def _pair_damage(self, K, weather="NONE"):
        """-> (dmg[n,2,6,6,5] fraction of defender full hp, prio[n,2,6,6,5]).

        TWO CROSS-SIDE BLOCKS, not a 12x12. Of the 144 ordered slot pairs, 72
        are same-side ("our mon attacks our mon") and no column ever reads them,
        so computing them was half the damage work -- and that waste was
        inherited by every consumer (the +1/+2 setup counterfactuals, the tera
        counterfactual, all five weather variants). Block 0 is side 0 attacking
        side 1, block 1 is side 1 attacking side 0; the real work is
        6 attackers x 4 moves x 6 defenders x 2 directions = 288 move
        evaluations, and that is now exactly what runs.

        FIVE CHANNELS, not two. The spec's cheap runtime path multiplies the
        static damage by boost ratios per leaf; that only works if the static
        value is split by WHICH stats it used, because the boosts that scale it
        differ. In the randbats pool that is five real cases:

          ch  category  offensive stat        defensive stat   example
          0   physical  attacker Attack       target Defense   Knock Off
          1   special   attacker Sp. Atk      target Sp. Def   Hydro Pump
          2   special   attacker Sp. Atk      target Defense   Psyshock  (251 sets)
          3   physical  attacker DEFENCE      target Defense   Body Press(162 sets)
          4   physical  TARGET's Attack       target Defense   Foul Play ( 89 sets)

        Folding 2-4 into channel 0/1 made Body Press damage wrong by up to 4x
        under Iron Defense, which is exactly the position the feature exists to
        describe."""
        n = self.n
        # a terastallized defender presents ONE type; here the defender is not
        # terastallized (that is `def_tera_mult`), so we use its real types.
        F = {"sun": np.int8(weather == "SUN"), "rain": np.int8(weather == "RAIN"),
             "sand": np.int8(weather == "SAND"), "snow": np.int8(weather == "SNOW")}
        raw = DT.raw_damage(K["A"], K["D"], K["M"], F)      # (n,2,6,4,6)
        frac = raw / np.maximum(K["def_maxhp"], 1.0)
        frac = np.where(K["usable"], frac, 0.0)
        chan, prio = K["chan"], K["prio"]
        out = np.zeros((n, 2, 6, 6, N_CHAN), np.float32)
        pri = np.zeros((n, 2, 6, 6, N_CHAN), np.float32)
        for c in range(N_CHAN):
            f = np.where(chan == c, frac, -1.0)
            b = f.argmax(axis=3)
            out[..., c] = np.maximum(np.take_along_axis(
                f, b[:, :, :, None, :], axis=3)[:, :, :, 0, :], 0.0)
            pri[..., c] = np.take_along_axis(
                np.broadcast_to(prio, f.shape), b[:, :, :, None, :],
                axis=3)[:, :, :, 0, :]
        return out, pri

    def _def_tera_mult(self):
        """mult[n, block, att, def] = (type effectiveness of att's best move
        against def AFTER def terastallizes) / (before). Prices defensive tera.
        Cross-side blocks only, like the damage table."""
        T = self.T
        mvi = self.mv
        mt = _sides_of(T.mv_type[mvi])[0][..., None]            # (n,2,6,4,1)
        d_t1 = _sides_of(self.t1)[1][:, :, None, None, :]
        d_t2 = _sides_of(self.t2)[1][:, :, None, None, :]
        d_tera = _sides_of(self.tera)[1][:, :, None, None, :]
        eff_before = DT.CHART_P[mt, d_t1] * DT.CHART_P[mt, d_t2]
        eff_after = DT.CHART_P[mt, d_tera]
        usable = _sides_of(self.mv_ok & self.occ[:, :, None])[0][..., None]
        eb = np.where(usable, eff_before, -1.0)
        b = eb.argmax(axis=3)
        before = np.maximum(np.take_along_axis(
            eb, b[:, :, :, None, :], axis=3)[:, :, :, 0, :], 0.0)
        after = np.take_along_axis(
            eff_after, b[:, :, :, None, :], axis=3)[:, :, :, 0, :]
        return (after / np.maximum(before, 1e-3)).astype(np.float32)

    # per-mon fields (n,12,...) and per-pair fields with two 12-axes
    _MON_FIELDS = ("occ", "maxhp", "level", "stats", "weight", "t1", "t2",
                   "tera", "dt1", "dt2", "mv", "mv_ok", "mv_present", "prio",
                   "cap", "setup_boost", "setup_present", "base_speed",
                   "scarf", "ability", "item", "pp_max")
    _PAIR_FIELDS = ("dmg", "dmg_prio", "dmg_tera", "def_tera_mult")

    def reorder(self, pidx):
        """Party order -> this leaf's canonical slot order, PER-MON fields only.
        `pidx[b, slot]` is the party index (0-11, side-major) in that slot. The
        pair tables are left alone and reordered by `pair()` after the weather
        variant has been picked -- gathering all five variants per leaf cost
        4.7x the whole dynamic half."""
        out = StaticCtx.__new__(StaticCtx)
        out.n, out.T = self.n, self.T
        for f in self._MON_FIELDS:
            v = getattr(self, f)
            ix = pidx.reshape(pidx.shape + (1,) * (v.ndim - 2))
            setattr(out, f, np.take_along_axis(v, ix, axis=1))
        for f in self._PAIR_FIELDS:
            setattr(out, f, getattr(self, f))
        return out

    @staticmethod
    def pair(v, oidx, wsel=None):
        """(n[,5],2,6,6[,C]) in party order -> (n,2,6,6[,C]) in slot order,
        selecting weather variant `wsel` on the way if the field has one.

        `oidx[b, side, slot]` is the WITHIN-SIDE party index in that slot
        (`llencoder._slot_order`). Block d's attacker axis is reordered by side
        d's permutation and its defender axis by side 1-d's."""
        if wsel is not None:
            v = v[np.arange(v.shape[0]), wsel]
        tail = (1,) * (v.ndim - 4)
        att = oidx.reshape(oidx.shape + (1,) + tail)               # (n,2,6,1)
        dfn = oidx[:, ::-1].reshape((oidx.shape[0], 2, 1, 6) + tail)
        v = np.take_along_axis(v, att, axis=2)
        return np.take_along_axis(v, dfn, axis=3)

    def broadcast(self, n):
        """Reuse one battle's static context for n leaves (the search path)."""
        if self.n == n:
            return self
        out = StaticCtx.__new__(StaticCtx)
        for f in StaticCtx.__slots__:
            v = getattr(self, f, None)
            if isinstance(v, np.ndarray) and v.shape[:1] == (self.n,):
                v = np.broadcast_to(v, (n,) + v.shape[1:])
            setattr(out, f, v)
        out.n = n
        return out


# ===========================================================================
# 4. the dynamic half -- once per leaf
# ===========================================================================
def encode_columnar(C, vocab, S=None, L=DEFAULT_LAYOUT, debug=None):
    """Columnar parse -> (ids int32[n,N_IDS], feats float32[n,N]).

    `S` is a prebuilt StaticCtx (the search path); None builds one here.
    `debug`, if a dict, receives the intermediates the gates check against the
    live engine (damage in HP points, effective speeds, KO counts).
    """
    T = Tables(vocab)
    n = C["n"]
    mi, si, gi = C["mi"], C["si"], C["gi"]
    order = LL._slot_order(si[:, :, _SI["active_index"]])
    # slot -> party index, side-major (side 1's party k is global index 6 + k)
    pidx = np.concatenate([order[:, 0], order[:, 1] + 6], axis=1)
    S = (StaticCtx(C, T) if S is None else S.broadcast(n)).reorder(pidx)

    m = LL._gather(mi, order).astype(np.float32)
    mid = LL._gather(C["mid"], order).astype(np.int64)
    vol = C["vol"]                                             # (n,2,107)
    occ, maxhp = S.occ, S.maxhp
    hp = np.maximum(m[:, :, _MI["hp"]], 0.0)
    hp_frac = np.where(occ, hp / np.maximum(maxhp, 1.0), 0.0)
    alive = (hp > 0) & occ
    status = m[:, :, _MI["status"]].astype(np.int64)
    is_active = np.broadcast_to((np.arange(12) % 6 == 0).astype(np.float32), (n, 12))
    # side-level things, exploded to the 12 slots of the owning side
    sboost = si[:, :, _SI["attack_boost"]:_SI["attack_boost"] + 7].astype(np.float32)
    scond = si[:, :, LL._SC0:LL._SC0 + len(LE.SIDE_CONDITION_FIELDS)].astype(np.float32)
    sdur = si[:, :, LL._DUR0:LL._DUR0 + len(LE.DURATION_FIELDS)].astype(np.float32)
    trick_room = gi[:, _GI["trick_room"]].astype(bool)
    weather = gi[:, _GI["weather"]].astype(np.int64)
    terrain = gi[:, _GI["terrain"]].astype(np.int64)

    def spread(x):
        """(n,2,...) -> (n,12,...): side 0 to slots 0-5, side 1 to slots 6-11."""
        return np.repeat(x, 6, axis=1)

    boost12 = np.zeros((n, 12, 7), np.float32)
    boost12[:, 0] = sboost[:, 0]
    boost12[:, 6] = sboost[:, 1]                 # boosts belong to the ACTIVE
    scond12 = spread(scond)
    vol12 = spread(vol)
    active12 = is_active > 0

    # ================= effective speed (§3 layer 2) =====================
    spe = S.base_speed.copy()
    para = (status == STATUS_ORDER.index("PARALYZE"))
    qf = T.ab_quickfeet[S.ability] > 0
    spe = spe * np.where(para, np.where(qf, 1.5, 0.5),
                         np.where(qf & (status > 0), 1.5, 1.0))
    spe = spe * _boost(boost12[:, :, 4])
    spe = spe * np.where(scond12[:, :, SC["tailwind"]] > 0, 2.0, 1.0)
    ws = T.ab_weather_speed[S.ability]                            # (n,12,8)
    spe = spe * np.where(np.take_along_axis(ws, weather[:, None, None]
                                            .repeat(12, 1), axis=2)[:, :, 0] > 0, 2.0, 1.0)
    spe = spe * np.where((T.ab_surgesurfer[S.ability] > 0)
                         & (terrain[:, None] == LE.TERRAIN_IX["ELECTRICTERRAIN"]), 2.0, 1.0)
    spe = spe * np.where(active12 & (vol12[:, :, LE.VOLATILE_IX["SLOWSTART"]]), 0.5, 1.0)
    spe = spe * np.where(active12 & (vol12[:, :, LE.VOLATILE_IX["UNBURDEN"]]), 2.0, 1.0)
    spe = spe * np.where(active12 & (vol12[:, :, LE.VOLATILE_IX["PROTOSYNTHESISSPE"]]
                                     | vol12[:, :, LE.VOLATILE_IX["QUARKDRIVESPE"]]), 1.5, 1.0)
    spe = np.maximum(spe, 1.0)
    # Trick Room inverts the comparison, not the number: fold it into a key.
    skey = np.where(trick_room[:, None], -spe, spe)

    # ================= dynamic damage (the spec's multiplier path) =======
    # static fraction of FULL hp -> fraction of CURRENT hp, with boosts,
    # screens, burn, weather-independent items and Multiscale applied.
    # Everything here is (n,2,6,6): the two CROSS-SIDE blocks, never 12x12.
    # Attacker-side quantities take axis 2, defender-side quantities axis 3.
    burn = _sides_of(status == STATUS_ORDER.index("BURN"))[0][..., None]
    guts = _sides_of(T.ab["ab_guts"][S.ability] > 0)[0][..., None]
    infil = _sides_of(T.ab["ab_infiltrator"][S.ability] > 0)[0][..., None]
    refl = _sides_of((scond12[:, :, SC["reflect"]] > 0)
                     | (scond12[:, :, SC["aurora_veil"]] > 0))[1][:, :, None, :] & ~infil
    lscr = _sides_of((scond12[:, :, SC["light_screen"]] > 0)
                     | (scond12[:, :, SC["aurora_veil"]] > 0))[1][:, :, None, :] & ~infil
    ms = _sides_of((T.ab["ab_multiscale"][S.ability] > 0)
                   & (hp >= maxhp))[1][:, :, None, :]

    stage = boost12[:, :, :5]
    stage_def = _sides_of(stage)[1]                        # (n,2,6,5)
    wsel = WEATHER_VAR_MAP[weather]                        # (n,)
    dmg_static = S.pair(S.dmg, order, wsel)                # (n,2,6,6,N_CHAN)
    prio_static = S.pair(S.dmg_prio, order, wsel)
    dmg_tera_static = S.pair(S.dmg_tera, order, wsel)
    def_tera_mult = S.pair(S.def_tera_mult, order)         # (n,2,6,6)
    ms_mult = np.where(ms, 0.5, 1.0)
    # Supreme Overlord: +10%% per fainted ally, capped at +50%%
    n_fainted_ally = np.repeat(np.stack([((~alive) & occ)[:, :6].sum(1),
                                         ((~alive) & occ)[:, 6:].sum(1)], 1), 6, axis=1)
    so_mult = _sides_of(np.where(T.ab_supreme[S.ability] > 0,
                                 1.0 + 0.1 * np.minimum(n_fainted_ally, 5),
                                 1.0))[0][..., None]
    burn_mult = np.where(burn & ~guts, 0.5, 1.0) * np.where(
        guts & _sides_of(status > 0)[0][..., None], 1.5, 1.0)
    refl_mult, lscr_mult = np.where(refl, 0.5, 1.0), np.where(lscr, 0.5, 1.0)

    def combine(dmg, att_stage):
        p, sp, ch = _channel_damage(dmg, _sides_of(att_stage)[0], stage_def)
        return (p * burn_mult * refl_mult * ms_mult * so_mult,
                sp * lscr_mult * ms_mult * so_mult, ch)

    phys, spec, chan_d = combine(dmg_static, stage)
    dmg_full = np.maximum(phys, spec)                    # fraction of FULL hp
    # priority of whichever channel actually wins
    chan_scaled = chan_d * np.where(CHAN_PHYSICAL, burn_mult[..., None] * refl_mult[..., None],
                                    lscr_mult[..., None]) * ms_mult[..., None]
    dmg_prio = np.take_along_axis(prio_static, chan_scaled.argmax(axis=-1)[..., None],
                                  axis=-1)[..., 0]
    # attacker must be alive; defender must be alive to be a target
    alive_a, alive_d = _sides_of(alive)
    live_pair = alive_a[..., None] & alive_d[:, :, None, :]
    dmg_full = np.where(live_pair, dmg_full, 0.0)

    hpf12 = np.maximum(hp_frac, 1e-6)                      # (n,12)
    hpf_def = _sides_of(hpf12)[1][:, :, None, :]           # (n,2,1,6)
    ko_now = np.where(dmg_full > 0, np.ceil(hpf_def / np.maximum(dmg_full, 1e-9)), 99.0)
    never_now = ~(dmg_full > 0) | ~live_pair

    # ================= assemble ==========================================
    feats = np.zeros((n, L.N), np.float32)
    ids = np.zeros((n, L.N_IDS), np.int32)

    # ---- ids -------------------------------------------------------------
    mon_ids = np.stack([mid[:, :, 1], mid[:, :, 2], S.t1, S.t2, S.tera,
                        S.mv[:, :, 0], S.mv[:, :, 1], S.mv[:, :, 2], S.mv[:, :, 3]], -1)
    ids[:, :12 * 9] = (mon_ids * occ[:, :, None]).reshape(n, -1)

    # ---- §1 per-mon ------------------------------------------------------
    mon = np.zeros((n, 12, L.NM), np.float32)
    M = {k: i for i, k in enumerate(L.mon)}
    mon[:, :, M["hp_frac"]] = hp_frac
    mon[:, :, M["maxhp"]] = maxhp / LE.D_HP
    # EFFECTIVE stats (spec §1: "computed, all modifiers"). The divisor is
    # 2 x D_STAT, not D_STAT: a +6 boost multiplies by 4, and the lossless
    # encoder's D_STAT is sized for RAW stats only, so /D_STAT put boosted
    # columns outside [-1, 1] (caught by the corpus census).
    eff_stats = S.stats * _boost(boost12[:, :, [0, 1, 2, 3, 4]])
    for i, k in enumerate(("attack", "defense", "sp_atk", "sp_def")):
        mon[:, :, M[k]] = np.minimum(eff_stats[:, :, i] / D_STAT_EFF, 1.0) * occ
    mon[:, :, M["speed"]] = np.minimum(spe / D_STAT_EFF, 1.0) * occ
    np.put_along_axis(mon[:, :, M["status_none"]:M["status_none"] + 7],
                      status[:, :, None], occ[:, :, None].astype(np.float32), axis=-1)
    mon[:, :, M["sleep_rest_turns"]] = np.maximum(
        m[:, :, _MI["sleep_turns"]], m[:, :, _MI["rest_turns"]]) / LE.D_SLEEP
    mon[:, :, M["times_attacked"]] = m[:, :, _MI["times_attacked"]] / LE.D_TIMES_ATTACKED
    mon[:, :, M["terastallized"]] = m[:, :, _MI["terastallized"]] * occ
    mon[:, :, M["alive"]] = alive
    pp = m[:, :, _MI["pp0"]:_MI["pp0"] + 4]
    mon[:, :, M["pp_frac0"]:M["pp_frac0"] + 4] = np.clip(pp / S.pp_max, 0.0, 1.0) * occ[:, :, None]
    mon[:, :, M["disabled0"]:M["disabled0"] + 4] = \
        m[:, :, _MI["disabled0"]:_MI["disabled0"] + 4] * occ[:, :, None]
    rev = m[:, :, _MI["revmask"]].astype(np.int64)
    bits = np.arange(8)
    mon[:, :, M["seen_species"]:M["seen_species"] + 8] = \
        ((rev[:, :, None] >> bits) & 1) * occ[:, :, None]
    mon[:, :, M["cap_" + CAP_NAMES[0]]:M["cap_" + CAP_NAMES[0]] + len(CAP_NAMES)] = \
        S.cap * alive[:, :, None]          # dead mons' capability bools read 0
    feats[:, L.O_MON:L.O_MON + 12 * L.NM] = mon.reshape(n, -1)

    # ---- §2 per-side -----------------------------------------------------
    side = np.zeros((n, 2, L.NS), np.float32)
    SS = {k: i for i, k in enumerate(L.side)}
    for name, col, div in (("stealth_rock", "stealth_rock", 1.0),
                           ("spikes", "spikes", 3.0),
                           ("toxic_spikes", "toxic_spikes", 2.0),
                           ("sticky_web", "sticky_web", 1.0),
                           ("reflect", "reflect", 1.0),
                           ("light_screen", "light_screen", 1.0),
                           ("aurora_veil", "aurora_veil", 1.0),
                           ("tailwind", "tailwind", 1.0),
                           ("safeguard", "safeguard", 1.0),
                           ("mist", "mist", 1.0)):
        v = scond[:, :, SC[col]]
        side[:, :, SS[name]] = np.minimum(v, div) / div if div > 1 else (v > 0)
        if name + "_turns" in SS:
            side[:, :, SS[name + "_turns"]] = v / 8.0
    side[:, :, SS["boost_atk"]:SS["boost_atk"] + 7] = sboost / LE.D_BOOST
    # the engine stores the toxic counter per SIDE (only the active can hold
    # one), so it is a side column, not twelve mon columns
    side[:, :, SS["toxic_counter"]] = scond[:, :, SC["toxic_count"]] / LE.D_SIDECOND
    side[:, :, SS["vol_" + VOL_COLS[0].lower()]:
         SS["vol_" + VOL_COLS[0].lower()] + len(VOL_COLS)] = vol[:, :, VOL_IX]
    if DUR_COLS:
        side[:, :, SS["dur_" + DUR_COLS[0]]:
             SS["dur_" + DUR_COLS[0]] + len(DUR_COLS)] = sdur[:, :, DUR_IX] / LE.D_DURATION
    act_maxhp = np.stack([maxhp[:, 0], maxhp[:, 6]], 1)
    side[:, :, SS["substitute_hp_frac"]] = np.clip(
        si[:, :, _SI["substitute_health"]].astype(np.float32)
        / np.maximum(act_maxhp / 4.0, 1.0), 0.0, 1.0)
    side[:, :, SS["wish_turns"]] = si[:, :, _SI["wish0"]] / LE.D_WISH_TURNS
    side[:, :, SS["wish_amount"]] = si[:, :, _SI["wish1"]] / LE.D_WISH_AMOUNT
    side[:, :, SS["future_sight_turns"]] = si[:, :, _SI["fs0"]] / LE.D_FS_TURNS
    side[:, :, SS["future_sight_source"]] = LL._party_to_slot(
        si[:, :, _SI["fs1"]].astype(np.int64),
        si[:, :, _SI["active_index"]].astype(np.int64)) / 5.0
    # move state, encoded by meaning
    lum_kind = si[:, :, _SI["lum_kind"]]
    lum_slot = si[:, :, _SI["lum_slot"]].astype(np.int64)
    act_item = np.stack([S.item[:, 0], S.item[:, 6]], 1)
    act_abil = np.stack([S.ability[:, 0], S.ability[:, 6]], 1)
    used_move = lum_kind == 1
    side[:, :, SS["lock_choice"]] = (T.it_choice[act_item] > 0) & used_move
    side[:, :, SS["lock_encore"]] = vol[:, :, LE.VOLATILE_IX["ENCORE"]]
    dis12 = m[:, :, _MI["disabled0"]:_MI["disabled0"] + 4]
    side[:, :, SS["lock_disable"]] = np.stack(
        [dis12[:, 0].max(axis=1), dis12[:, 6].max(axis=1)], 1) > 0
    side[:, :, SS["lock_multiturn"]] = vol[:, :, LE.VOLATILE_IX["LOCKEDMOVE"]]
    side[:, :, SS["just_switched_in"]] = lum_kind == 2
    for k in ("force_switch", "force_trapped", "last_move_failed"):
        side[:, :, SS[k]] = si[:, :, _SI[k]]
    tera_used = np.stack([m[:, :6, _MI["terastallized"]].max(axis=1),
                          m[:, 6:, _MI["terastallized"]].max(axis=1)], 1)
    side[:, :, SS["tera_available"]] = 1.0 - (tera_used > 0)
    fainted = (~alive) & occ
    side[:, :, SS["n_fainted_revivable"]] = np.stack(
        [fainted[:, :6].sum(1), fainted[:, 6:].sum(1)], 1) / 6.0
    side[:, :, SS["times_revived"]] = si[:, :, _SI["times_revived"]] / LE.D_TIMES_REVIVED

    # locked / charging move embedding ids
    lock_id = np.where(used_move, np.take_along_axis(
        np.stack([S.mv[:, 0], S.mv[:, 6]], 1), np.clip(lum_slot, 0, 3)[:, :, None],
        axis=2)[:, :, 0], 0)
    # charging_move_id: only SOLARBEAM and METEORBEAM are reachable two-turn
    # moves in gen9 randbats (REACHABLE_VOLATILES.md), so the volatile names
    # the move and we resolve it back to the active's own move slot.
    charge_id = np.zeros((n, 2), np.int64)
    mv2 = np.stack([S.mv[:, 0], S.mv[:, 6]], 1)                  # (n,2,4)
    for cname in ("SOLARBEAM", "METEORBEAM"):
        cv = vol[:, :, LE.VOLATILE_IX[cname]]
        match = T.mv_local[mv2] == DT.moves().ix[cname]
        pick = np.where(match.any(axis=2),
                        np.take_along_axis(mv2, match.argmax(axis=2)[:, :, None],
                                           axis=2)[:, :, 0], 0)
        charge_id = np.where(cv > 0, pick, charge_id)
    ids[:, 12 * 9:] = np.stack([lock_id, charge_id], -1).reshape(n, -1)
    feats[:, L.O_SIDE:L.O_SIDE + 2 * L.NS] = side.reshape(n, -1)

    # ---- §3 relational ---------------------------------------------------
    rel = np.zeros((n, L.NR), np.float32)
    o = 0
    ours, theirs = slice(0, 6), slice(6, 12)
    pair_speed = _first(skey[:, ours, None], skey[:, None, theirs])   # (n,6,6)
    assert L.ko_versions == 1, "the full-hp KO matrix was dropped (see §10)"
    for kob, nev in ((ko_now, never_now),):
        for d in range(L.ko_directions):
            # block 0 is us->them; block 1 is them->us, transposed so the row
            # index stays OUR mon in both directions.
            sub = kob[:, 0] if d == 0 else kob[:, 1].transpose(0, 2, 1)
            nv = nev[:, 0] if d == 0 else nev[:, 1].transpose(0, 2, 1)
            oh = np.zeros((n, 6, 6, 5), np.float32)
            _bucket_onehot(sub, 5, oh, nv)
            rel[:, o:o + 180] = oh.reshape(n, -1)
            o += 180
    rel[:, o:o + 36] = pair_speed.reshape(n, -1)
    o += 36
    v_us, v_them = spe[:, 0], spe[:, 6]
    marg = np.tanh(np.log(np.maximum(v_us, 1.0) / np.maximum(v_them, 1.0)))
    rel[:, o] = np.where(trick_room, -marg, marg)
    rel[:, o + 1] = v_us == v_them
    o += 2
    # 16 move-order bits: my move i vs their move j
    pu, pt = S.prio[:, 0], S.prio[:, 6]
    spd_first = _first(skey[:, 0], skey[:, 6])
    order16 = np.where(pu[:, :, None] != pt[:, None, :],
                       (pu[:, :, None] > pt[:, None, :]).astype(np.float32),
                       spd_first[:, None, None])
    rel[:, o:o + 16] = (order16 * S.mv_present[:, 0][:, :, None]
                        * S.mv_present[:, 6][:, None, :]).reshape(n, -1)
    o += 16
    # derived priority flags
    best_pu = np.where(S.mv_present[:, 0], pu, -9).max(axis=1)
    best_pt = np.where(S.mv_present[:, 6], pt, -9).max(axis=1)
    rel[:, o] = best_pu > np.maximum(best_pt, 0)
    rel[:, o + 1] = best_pt > np.maximum(best_pu, 0)
    # revenge: an opposing mon has a priority move that kills our active
    # (the opposing ACTIVE is index 0 of the other side's block)
    prio_kill_them = ((dmg_prio[:, 0, :, 0] > 0)
                      & (dmg_full[:, 0, :, 0] >= hpf12[:, 6][:, None])
                      & alive[:, ours]).any(axis=1)
    prio_kill_us = ((dmg_prio[:, 1, :, 0] > 0)
                    & (dmg_full[:, 1, :, 0] >= hpf12[:, 0][:, None])
                    & alive[:, theirs]).any(axis=1)
    rel[:, o + 2] = prio_kill_us
    rel[:, o + 3] = prio_kill_them
    o += 4
    rel[:, o:o + 48] = (S.prio / 7.0 * S.mv_present).reshape(n, -1)  # gen9 range -7..+5
    o += 48
    assert o == L.NR, (o, L.NR)
    feats[:, L.O_REL:L.O_REL + L.NR] = rel

    # ---- §4 counterfactual ----------------------------------------------
    cfm = np.zeros((n, 12, L.NCM), np.float32)
    CM = {k: i for i, k in enumerate(L.cf_mon)}
    cfm[:, :, CM["setup_present"]] = S.setup_present * alive
    cfm[:, :, CM["setup_boost_atk"]:CM["setup_boost_atk"] + 5] = \
        S.setup_boost * alive[:, :, None] / 6.0     # Belly Drum is +6
    # The counterfactual block is per-SLOT against that slot's SIX opponents,
    # so it works in the same (n,2,6,6) cross shape: [block, my slot, opponent].
    # `incx` is the transposed view -- damage coming INTO my slot from each of
    # the six opponents -- which is block 1-b of the same two tensors.
    cfmx = cfm.reshape(n, 2, 6, L.NCM)
    incx = dmg_full[:, ::-1].transpose(0, 1, 3, 2)          # (n,2,6,6)
    incx_prio = dmg_prio[:, ::-1].transpose(0, 1, 3, 2)
    opp = np.broadcast_to(_sides_of(alive)[1][:, :, None, :], (n, 2, 6, 6))
    n_opp = np.broadcast_to(_sides_of(alive)[1].sum(axis=2)[:, :, None], (n, 2, 6))
    hpf_opp = _sides_of(hpf12)[1][:, :, None, :]            # (n,2,1,6)
    hpf_mine = _sides_of(hpf12)[0][:, :, :, None]           # (n,2,6,1)
    hpfr_mine = _sides_of(hp_frac)[0]                       # (n,2,6)
    alive_r, setup_r = _sides_of(alive)[0], _sides_of(S.setup_present)[0]
    for lv in (1, 2):
        # the setup move's own boost profile, applied lv times on top of the
        # boosts already on the board
        st = np.clip(stage + S.setup_boost * lv, -6, 6)
        p2, s2_, _ = combine(dmg_static, st)
        d2 = np.where(live_pair, np.maximum(p2, s2_), 0.0)
        spe2 = spe * (_boost(st[:, :, 4]) / _boost(stage[:, :, 4]))
        skey2 = np.where(trick_room[:, None], -spe2, spe2)
        skey2_mine = _sides_of(skey2)[0][:, :, :, None]     # (n,2,6,1)
        skey_opp = _sides_of(skey)[1][:, :, None, :]        # (n,2,1,6)
        ohko = ((d2 >= hpf_opp) & opp).sum(axis=3)
        outspd_n = ((_first(skey2_mine, skey_opp) > 0) & opp).sum(axis=3)
        # their priority answer: an opponent with a damaging priority move that
        # kills the setup mon outright
        prio_ans = ((incx_prio > 0) & (incx >= hpf_mine) & opp).any(axis=3)
        # The sweep flag is the one the spec says should dominate an evaluation,
        # so it is the GUARANTEED version: the OHKOs must land on the MINIMUM
        # damage roll, not the maximum the KO matrix uses.
        ohko_min = ((d2 * DT.MIN_ROLL >= hpf_opp) & opp).sum(axis=3)
        sweep = ((ohko_min == n_opp) & (n_opp > 0) & (outspd_n == n_opp) & ~prio_ans
                 & alive_r & (setup_r > 0))
        # mons required to kill it: only opponents that get to act
        acts = opp & ((_first(skey_opp, skey2_mine) > 0) | (incx_prio > 0)
                      | (d2 < hpf_opp))
        odmg = np.where(acts, incx, 0.0)
        cum = np.cumsum(-np.sort(-odmg, axis=3), axis=3)
        need = (cum < hpfr_mine[:, :, :, None]).sum(axis=3) + 1
        never = cum[:, :, :, -1] < hpfr_mine
        base = CM["s%d_ohko_count" % lv]
        cfmx[:, :, :, base] = ohko / 6.0
        cfmx[:, :, :, base + 1] = outspd_n / 6.0
        cfmx[:, :, :, base + 2] = sweep
        oh = np.zeros((n, 2, 6, 5), np.float32)
        _bucket_onehot(need, 5, oh, never)
        cfmx[:, :, :, base + 3:base + 8] = oh * alive_r[:, :, :, None]
        if lv == 1:
            sweep1 = sweep.reshape(n, 12)
        else:
            sweep2 = sweep.reshape(n, 12)
    feats[:, L.O_CFM:L.O_CFM + 12 * L.NCM] = cfm.reshape(n, -1)

    # ---- §4 per-side aggregates + tera ----------------------------------
    cfs = np.zeros((n, 2, L.NCS), np.float32)
    CS = {k: i for i, k in enumerate(L.cf_side)}
    cfs[:, 0, CS["best_sweep_1"]] = sweep1[:, :6].max(axis=1)
    cfs[:, 1, CS["best_sweep_1"]] = sweep1[:, 6:].max(axis=1)
    cfs[:, 0, CS["best_sweep_2"]] = sweep2[:, :6].max(axis=1)
    cfs[:, 1, CS["best_sweep_2"]] = sweep2[:, 6:].max(axis=1)
    # free turn: does THEIR active survive MY best move (spec §4, literal).
    # Side s attacks in block s; the opposing active is defender index 0.
    cfs[:, 0, CS["free_turn"]] = np.clip(hp_frac[:, 6] - dmg_full[:, 0, 0, 0], 0.0, 1.0)
    cfs[:, 1, CS["free_turn"]] = np.clip(hp_frac[:, 0] - dmg_full[:, 1, 0, 0], 0.0, 1.0)
    # answers to their best threat
    alivex = _sides_of(alive)[0]
    hpf12x = _sides_of(hpf12)[0]
    hp_fracx = _sides_of(hp_frac)[0]
    skeyx = _sides_of(skey)[0]
    rows = np.arange(n)
    for s in (0, 1):
        mine = slice(0, 6) if s == 0 else slice(6, 12)
        # block 1-s is "they attack me": [their attacker, my defender]
        threat_score = ((dmg_full[:, 1 - s] >= hpf12x[:, s][:, None, :])
                        & alivex[:, 1 - s][:, :, None]
                        & alivex[:, s][:, None, :]).sum(axis=2)
        worst = threat_score.argmax(axis=1)          # index within the OTHER side
        kill_it = (dmg_full[:, s][rows, :, worst]
                   >= hpf12x[:, 1 - s][rows, worst][:, None])
        faster_than_it = _first(skeyx[:, s], skeyx[:, 1 - s][rows, worst][:, None]) > 0
        survive_it = dmg_full[:, 1 - s][rows, worst, :] < hp_fracx[:, s]
        ans = (kill_it & (faster_than_it | survive_it) & alive[:, mine]).sum(axis=1)
        cfs[:, s, CS["answers_to_best_threat"]] = ans / 6.0
    # tera: exactly two features per side (spec §4 -- priced as an option, not
    # as a per-mon matrix)
    phys_te, spec_te, _ = combine(dmg_tera_static, stage)
    dmg_tera_full = np.where(live_pair, np.maximum(phys_te, spec_te), 0.0)
    for s in (0, 1):
        mine = slice(0, 6) if s == 0 else slice(6, 12)
        avail = side[:, s, SS["tera_available"]] > 0
        untera = alive[:, mine] & (m[:, mine, _MI["terastallized"]] == 0)
        # offensive: extra damage on the opposing active (defender index 0 of
        # block s), 1.0 if it turns a non-kill into a kill
        d_no = dmg_full[:, s, :, 0]
        d_te = dmg_tera_full[:, s, :, 0]
        tgt = hpf12x[:, 1 - s][:, 0][:, None]
        off_gain = np.where((d_te >= tgt) & (d_no < tgt), 1.0,
                            np.clip(d_te - d_no, 0.0, 1.0))
        # defensive: the opposing active's damage into me, scaled by the type
        # multiplier my tera type would impose
        inc = dmg_full[:, 1 - s, 0, :]
        inc_te = inc * def_tera_mult[:, 1 - s, 0, :]
        mine_hp = hp_frac[:, mine]
        def_gain = np.where((inc >= mine_hp) & (inc_te < mine_hp), 1.0,
                            np.clip((inc - inc_te) / np.maximum(mine_hp, 1e-6), 0.0, 1.0))
        gain = np.where(untera, np.maximum(off_gain, def_gain), 0.0)
        cfs[:, s, CS["tera_best_value"]] = np.where(avail, gain.max(axis=1), 0.0)
        # tera-enabled sweep: tera alone, or tera composed with one setup
        # tera alone (it turns a non-kill into a kill on the active) composed
        # with either an existing +1 sweep or a setup move it could use
        tera_sweep = (untera & (off_gain >= 1.0)
                      & (sweep1[:, mine] | (S.setup_present[:, mine] > 0)))
        cfs[:, s, CS["tera_enabled_sweep"]] = np.where(
            avail, tera_sweep.max(axis=1), 0.0)
    # best revival target score: the strongest fainted mon, priced by how many
    # living opponents it would OHKO at full HP. This deliberately reads the
    # UNMASKED damage: every other matchup feature excludes fainted mons (spec
    # design rule), but Revival Blessing is precisely the question "what is the
    # dead mon worth", so masking it made the column a constant zero.
    full_ohko = np.maximum(phys, spec) >= 1.0            # (n,2,6,6), unmasked
    for s in (0, 1):
        mine = slice(0, 6) if s == 0 else slice(6, 12)
        f = fainted[:, mine]
        score = (full_ohko[:, s] & alivex[:, 1 - s][:, None, :]).sum(axis=2) / 6.0
        side[:, s, SS["best_revival_target_score"]] = np.where(f, score, 0.0).max(axis=1)
    feats[:, L.O_SIDE:L.O_SIDE + 2 * L.NS] = side.reshape(n, -1)
    feats[:, L.O_CFS:L.O_CFS + 2 * L.NCS] = cfs.reshape(n, -1)

    # ---- §5 global -------------------------------------------------------
    glob = np.zeros((n, L.NG), np.float32)
    G = {k: i for i, k in enumerate(L.glob)}
    np.put_along_axis(glob[:, G["weather_none"]:G["weather_none"] + len(WEATHER_COLS)],
                      WEATHER_MAP[weather][:, None], 1.0, axis=-1)
    glob[:, G["weather_turns"]] = (gi[:, _GI["weather_turns"]] + 1) / LE.D_WEATHER_TURNS
    np.put_along_axis(glob[:, G["terrain_none"]:G["terrain_none"] + len(TERRAIN_ORDER)],
                      terrain[:, None], 1.0, axis=-1)
    glob[:, G["terrain_turns"]] = (gi[:, _GI["terrain_turns"]] + 1) / LE.D_WEATHER_TURNS
    glob[:, G["trick_room"]] = trick_room
    glob[:, G["trick_room_turns"]] = (gi[:, _GI["trick_room_turns"]] + 1) / LE.D_TR_TURNS
    feats[:, L.O_GLOB:L.O_GLOB + L.NG] = glob

    if debug is not None:
        # OFF the hot path: re-expand the two cross blocks to the 12x12 view the
        # gates index into. Same-side entries are structurally zero (they are
        # not computed, and no column reads them).
        wide = np.zeros((n, 12, 12), np.float32)
        wide[:, :6, 6:] = dmg_full[:, 0]
        wide[:, 6:, :6] = dmg_full[:, 1]
        debug.update(dmg_points=wide * maxhp[:, None, :], dmg_frac=wide,
                     dmg_cross=dmg_full, speed=spe, speed_key=skey,
                     hp_frac=hp_frac, alive=alive, ko_now=ko_now,
                     never=never_now, sweep1=sweep1, sweep2=sweep2,
                     cap=S.cap, prio=S.prio, setup_boost=S.setup_boost, static=S)
    return ids, feats


def _boost(stage):
    return DT.BOOST_MULT_TABLE[np.clip(stage, -6, 6).astype(np.int64) + 6]


def _first(a, b):
    """1.0 where speed-key a acts before b, 0.5 on a tie, 0.0 otherwise.
    The key already carries Trick Room as a sign flip."""
    return np.where(a > b, 1.0, np.where(a == b, 0.5, 0.0)).astype(np.float32)


# ===========================================================================
# 5. entry points
# ===========================================================================
def encode_states(states, vocab, layout=DEFAULT_LAYOUT, chunk=2048,
                  share_static=False, debug=None):
    """list[state string] -> (ids, feats). `share_static=True` builds the
    StaticCtx from the FIRST state only and reuses it for all of them; that is
    the search path, and it requires every state to hold the same 12 Pokemon."""
    S = None
    if share_static:
        S = StaticCtx(LL.parse_batch(states[:1], vocab), Tables(vocab))
    parts = []
    for i in range(0, len(states), chunk):
        C = LL.parse_batch(states[i:i + chunk], vocab)
        parts.append(encode_columnar(C, vocab, S=S, L=layout, debug=debug))
    if len(parts) == 1:
        return parts[0]
    return (np.concatenate([p[0] for p in parts]),
            np.concatenate([p[1] for p in parts]))


def build_static(states, vocab):
    """StaticCtx for a set of states that share their twelve Pokemon. Built in
    PARTY order, so it stays valid across switches inside the search."""
    return StaticCtx(LL.parse_batch(states, vocab), Tables(vocab))


if __name__ == "__main__":
    L = DEFAULT_LAYOUT
    for k, v in L.block_sizes().items():
        print("%-28s %6d" % (k, v))
    print("volatile columns: %d (of %d engine variants)"
          % (len(VOL_COLS), LE.N_VOLATILES))
