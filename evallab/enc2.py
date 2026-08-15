"""ENCODER 2 -- the evaluator encoder of EVALUATOR_SPEC.md, implemented as written.

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
  spec's own trick makes that cheap: the static half stores damage as a
  fraction of the target's FULL hp, PER MOVE, so a leaf applies
      dmg = static * off_boost / def_boost * screen * burn * Multiscale * ...
  takes the maximum over the moves it may actually use, and a KO count is one
  division. `encode_states(..., share_static=True)` is that path;
  `enc2_gate.py cost` measures both halves.

  THE STATIC CONTEXT CANNOT GO STALE except on a change of the twelve Pokemon
  themselves. It reads no `disabled`, no `pp` and no `terastallized`: the first
  two mask the per-move table in the dynamic half, and tera is an axis of that
  table (12 = 6 party slots x {not tera'd, tera'd} on each mon axis), selected
  by the same gather that does the slot permutation. Measured over 895 real MCTS
  transitions those three fired on 16.2% / 2.7% / 0.0% of children, and a
  rebuild would have cost 13-40% of the whole per-leaf budget for `disabled`
  alone. See ENCODER2_BUILD.md section 11.3.

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
  * ONE setup counterfactual level (+1). The +2 level was 8 of the 22 per-mon
    counterfactual columns and 17% of the dynamic dataflow; +1 already carries
    the sweep flag the spec says should dominate. See ENCODER2_BUILD.md 11.2.
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
VOL_COLS = rbpool.reachable_list()
VOL_IX = np.array([LE.VOLATILE_IX[v] for v in VOL_COLS], np.int64)
# durations, restricted to volatiles that are reachable
DUR_COLS = [d for d in LE.DURATION_FIELDS
            if d.upper().replace("_", "") in set(VOL_COLS)]
DUR_IX = np.array([LE.DURATION_FIELDS.index(d) for d in DUR_COLS], np.int64)
SC = {n: i for i, n in enumerate(LE.SIDE_CONDITION_FIELDS)}
# the four volatiles the effective-speed layer reads, and they belong to the
# ACTIVE only -- `np.repeat`ing all 107 out to twelve slots to read four bits
# cost 1,284 element-ops per leaf.
VIX_SLOWSTART = LE.VOLATILE_IX["SLOWSTART"]
VIX_UNBURDEN = LE.VOLATILE_IX["UNBURDEN"]
VIX_PROTO_SPE = LE.VOLATILE_IX["PROTOSYNTHESISSPE"]
VIX_QUARK_SPE = LE.VOLATILE_IX["QUARKDRIVESPE"]
VIX_ENCORE = LE.VOLATILE_IX["ENCORE"]
VIX_LOCKEDMOVE = LE.VOLATILE_IX["LOCKEDMOVE"]

D_STAT_EFF = 2.0 * LE.D_STAT     # effective stats can be 4x the raw stat
KO_BUCKETS = ("ohko", "2hko", "3hko", "4plus", "never")
MTK_BUCKETS = ("1", "2", "3", "4plus", "never")
SETUP_LEVELS = (1,)              # the +2 counterfactual was cut; see Layout
# DT.TYPES is 19 long: 18 real types + STELLAR. STELLAR is a tera MARKER, never
# a defensive typing, so the multi-hots are 18 wide and DT.TYPELESS (= 19) and
# STELLAR both fall outside them -- same convention as encoder.py:_type_bits.
N_REAL_TYPES = 18
assert DT.TYPES[N_REAL_TYPES] == "STELLAR" and DT.TYPELESS == DT.N_TYPES

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
        # TYPING, as the SHIPPED encoder represents it: two 18-bit multi-hots,
        # the effective (tera-aware) DEFENSIVE typing and the own typing that
        # still drives STAB after a tera. enc2 originally carried typing only as
        # `type1`/`type2` EMBEDDING IDS, and the value test measured those dead
        # (half their init norm, +0.0003 permutation dBrier) while the shipped
        # encoder's multi-hot earned +0.0084. The ids are STATIC per slot, so a
        # tera moves nothing in them; the effective multi-hot is DYNAMIC, which
        # is the information that was being lost. STELLAR and TYPELESS set no
        # bit, exactly as `encoder.py:_type_bits` does.
        m += ["type_eff_" + t.lower() for t in DT.TYPES[:N_REAL_TYPES]]
        m += ["type_own_" + t.lower() for t in DT.TYPES[:N_REAL_TYPES]]
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
        # ONE setup level (+1). The +2 level was 8 of the 22 per-mon columns and
        # 17% of the whole dynamic dataflow -- the best value-per-column cut
        # available -- and +1 already carries the sweep flag the spec says should
        # dominate an evaluation. Removed, not disabled: nothing computes it.
        for lv in SETUP_LEVELS:
            c += ["s%d_ohko_count" % lv, "s%d_outspeed_count" % lv, "s%d_sweep" % lv]
            c += ["s%d_mtk_%s" % (lv, b) for b in MTK_BUCKETS]
        self.cf_mon = c
        self.cf_side = ["best_sweep_1", "free_turn",
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
        self._index()

    def _index(self):
        """Every column index the dynamic half needs, resolved ONCE per layout.

        `M`/`SS`/`CM`/`CS`/`G` used to be rebuilt as python dicts on every leaf
        (~185 entries per leaf, for a path that runs 130k times a second)."""
        self.M = {k: i for i, k in enumerate(self.mon)}
        self.SS = {k: i for i, k in enumerate(self.side)}
        self.CM = {k: i for i, k in enumerate(self.cf_mon)}
        self.CS = {k: i for i, k in enumerate(self.cf_side)}
        self.G = {k: i for i, k in enumerate(self.glob)}
        M, SS, CM, CS, G = self.M, self.SS, self.CM, self.CS, self.G
        self.I_MON = {k: M[k] for k in
                      ("hp_frac", "maxhp", "attack", "speed", "status_none",
                       "sleep_rest_turns", "times_attacked", "terastallized",
                       "alive", "pp_frac0", "disabled0", "seen_species")}
        self.I_CAP = M["cap_" + CAP_NAMES[0]]
        self.I_STAT4 = np.array([M[k] for k in ("attack", "defense",
                                                "sp_atk", "sp_def")], np.int64)
        # §2 side conditions: value columns and their /8 turn columns, resolved
        # to flat index arrays so the ten-iteration python loop with dict
        # lookups (189 us at n=1, for 412 element-ops) becomes two array ops.
        sc_spec = (("stealth_rock", 1.0), ("spikes", 3.0), ("toxic_spikes", 2.0),
                   ("sticky_web", 1.0), ("reflect", 1.0), ("light_screen", 1.0),
                   ("aurora_veil", 1.0), ("tailwind", 1.0), ("safeguard", 1.0),
                   ("mist", 1.0))
        self.SC_SRC = np.array([SC[k] for k, _ in sc_spec], np.int64)
        self.SC_DST = np.array([SS[k] for k, _ in sc_spec], np.int64)
        self.SC_DIV = np.array([d for _, d in sc_spec], np.float32)
        turns = [k for k, _ in sc_spec if k + "_turns" in SS]
        self.SCT_SRC = np.array([SC[k] for k in turns], np.int64)
        self.SCT_DST = np.array([SS[k + "_turns"] for k in turns], np.int64)
        self.I_SIDE = {k: SS[k] for k in
                       ("boost_atk", "toxic_counter", "substitute_hp_frac",
                        "wish_turns", "wish_amount", "future_sight_turns",
                        "future_sight_source", "lock_choice", "lock_encore",
                        "lock_disable", "lock_multiturn", "just_switched_in",
                        "tera_available", "n_fainted_revivable", "times_revived",
                        "best_revival_target_score",
                        "vol_" + VOL_COLS[0].lower())}
        if DUR_COLS:
            self.I_SIDE["dur0"] = SS["dur_" + DUR_COLS[0]]
        # every remaining `side[:, :, col] = si[:, :, field] / D` column, as one
        # gather + one divide + one scatter instead of seven of each
        si_cols = (("wish_turns", "wish0", LE.D_WISH_TURNS),
                   ("wish_amount", "wish1", LE.D_WISH_AMOUNT),
                   ("future_sight_turns", "fs0", LE.D_FS_TURNS),
                   ("times_revived", "times_revived", LE.D_TIMES_REVIVED),
                   ("force_switch", "force_switch", 1.0),
                   ("force_trapped", "force_trapped", 1.0),
                   ("last_move_failed", "last_move_failed", 1.0))
        self.SI_DST = np.array([SS[a] for a, _, _ in si_cols], np.int64)
        self.SI_SRC = np.array([_SI[b] for _, b, _ in si_cols], np.int64)
        self.SI_DIV = np.array([d for _, _, d in si_cols], np.float64)
        self.I_CFM = {k: CM[k] for k in ("setup_present", "setup_boost_atk")}
        self.I_CFM_S = [CM["s%d_ohko_count" % lv] for lv in SETUP_LEVELS]
        self.I_CFS = {k: CS[k] for k in CS}
        self.I_G = {k: G[k] for k in G}

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
# StaticCtx._pair_damage. A move's channel says WHICH stats scale it, and is
# what makes the leaf's boost multipliers correct.
N_CHAN = 5
CHAN_PHYSICAL = np.array([1, 0, 0, 1, 1], bool)      # burn / Reflect apply


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


# Boost stages live ONLY on the two active slots (`boost12` is written at slots
# 0 and 6 and nowhere else), so on the DEFENDER axis every boost ratio is
# exactly 1.0 off index 0 -- and multiplying or dividing by exactly 1.0 is the
# identity in IEEE754. That is why `combine()` applies the defender ratios to
# one column of six instead of to all thirty-six pairs: the bench x bench
# quadrant, 50 of the 72 pairs, is boost-free by construction, not by luck.
_BITS8 = np.arange(8)
# atk / spa / spa / def -- the attacker stat each of damage channels 0..3 scales
# with, as an index into the (atk, def, spa, spd, spe) boost vector
_NUM_STAT = np.array([0, 2, 2, 1])


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
                 "tera", "mv", "prio", "chan", "mv_dmg", "mv_real", "mv_phys",
                 "dmg", "eff_te",
                 "cap", "setup_boost", "setup_present",
                 "base_speed", "scarf", "ability", "item", "T", "pp_max",
                 "level", "weight", "pack", "pack_spec")

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
        self.item, self.ability = mid[:, :, 1], mid[:, :, 2]
        self.mv = mid[:, :, 11:15]                                    # (n,12,4)
        # NOTHING here reads `disabled`, `pp` or `terastallized`. Those three
        # are what used to invalidate the shared static context, and they are
        # not rare: over 895 real MCTS transitions a terastallization happened
        # on 16.2% and a new disable on 2.7%, so at a Rust static cost of
        # 42-127 us a rebuild would have eaten 13-40% (disable) and far more
        # (tera) of the entire 8.4 us/leaf budget. `disabled`/`pp` now mask the
        # PER-MOVE damage table in the dynamic half (a per-channel maximum
        # cannot be un-taken once a move is disabled, which is the real reason
        # the table is now per move); tera is carried as an extra axis of the
        # table and selected per leaf. Nothing in here can go stale.
        self.mv_real = (T.mv_local[self.mv] != 0) & occ_[:, :, None]
        self.mv_dmg = T.mv_damaging[self.mv]
        self.pp_max = np.maximum(T.mv_pp[self.mv], 1.0)
        self.chan = _channel_of(T, self.mv)                           # (n,12,4)
        self.mv_phys = CHAN_PHYSICAL[np.clip(self.chan, 0, N_CHAN - 1)]

        # --- priority brackets, incl. Prankster / Gale Wings / Triage -----
        pr = T.mv_prio[self.mv].astype(np.float32)
        ab = self.ability
        pr = pr + (T.ab_prankster[ab][:, :, None] * T.mv_is_status[self.mv])
        flying = T.mv_type[self.mv] == DT.TIX["FLYING"]
        pr = pr + (T.ab_galewings[ab][:, :, None] * flying)
        pr = pr + 3 * (T.ab_triage[ab][:, :, None] * T.mv_is_heal[self.mv])
        self.prio = pr                                                # (n,12,4)

        # --- the static damage table: damage as a fraction of the defender's
        #     FULL hp, PER MOVE, with no boosts, no screens, no weather, no
        #     burn and no Multiscale (all applied per leaf). Only the FIELD
        #     changes between weather variants, so the kernel's inputs are built
        #     ONCE and reused across all five -- at n=1 (one battle per search)
        #     this half is numpy call-overhead bound, not FLOP bound.
        K = self._kernel_inputs()
        self.dmg = np.stack([self._pair_damage(K, w) for w in WEATHER_VARIANTS], 1)
        #     -> (n, 5 weather, 2 blocks, 12 attackers, 4 moves, 12 defenders)
        #        where the 12 is 6 party slots x {not terastallized, tera'd}.
        # defender-side tera: type effectiveness of each attacker's moves
        # against each defender before and after the DEFENDER terastallizes
        # (used to price defensive tera). The best-move pick is made per leaf,
        # for the two ACTIVE attackers only -- the only rows any column reads.
        self.eff_te = self._def_tera_eff()

        # --- capabilities (13 bools per mon), zeroed for empty slots ------
        cap = T.mv_cap[self.mv].max(axis=2) + T.ab_cap[ab]
        self.cap = np.clip(cap, 0.0, 1.0) * self.occ[:, :, None]

        # --- setup profile -------------------------------------------------
        # `mv_setup_rank` is already 0 for every non-setup move, so the old
        # `where(mv_ok | is_setup, rank, 0)` was the identity -- which is why
        # the setup profile is disable- and PP-independent, bit for bit.
        rank = T.mv_setup_rank[self.mv]
        best = rank.argmax(axis=2)                                    # (n,12)
        bm = np.take_along_axis(self.mv, best[:, :, None], axis=2)[:, :, 0]
        self.setup_present = ((np.take_along_axis(rank, best[:, :, None], axis=2)[:, :, 0] > 0)
                              & self.occ).astype(np.float32)
        self.setup_boost = (T.mv_boosts[bm][:, :, :5].astype(np.float32)
                            * self.setup_present[:, :, None])         # (n,12,5)

        # --- static speed: the stat plus the modifiers that do not move ----
        self.scarf = T.it_scarf[self.item].astype(np.float32)
        self.base_speed = self.stats[:, :, 4] * np.where(self.scarf > 0, 1.5, 1.0)
        self._pack()

    # ------------------------------------------------------------------
    def _kernel_inputs(self):
        """The damage kernel's attacker / defender / move arrays.

        Axes: (n, block, attacker 6, ATTACKER-TERA 2, move 4, defender 6,
        DEFENDER-TERA 2). Field-independent, so it is built once and reused
        across all five weather variants.

        The two tera axes are the whole point. poke-engine does NOT rewrite
        `types` on a terastallization -- it sets a flag and applies the tera
        type at damage time -- so tera moves both the attacker's STAB and the
        defender's typing, and the old context baked the root's tera state in.
        Carrying both as axes costs 2x the kernel's element count in a half
        that is call-bound rather than FLOP-bound, and makes the static context
        exactly tera-independent: a leaf indexes the entries its own state
        calls for. It also hands the +1 tera counterfactual out of the same
        table for free (index the tera'd attacker rows).

        Broadcasting does the duplication: only `terastallized` and the
        defender's two type columns actually carry a tera axis."""
        T = self.T
        att = lambda v: _sides_of(v)[0][:, :, :, None, None, None, None]  # noqa: E731
        dfn = lambda v: _sides_of(v)[1][:, :, None, None, None, :, None]  # noqa: E731
        A = {"level": att(self.level), "atk": att(self.stats[:, :, 0]),
             "spa": att(self.stats[:, :, 2]), "defense": att(self.stats[:, :, 1]),
             "weight": att(self.weight), "stab1": att(self.t1),
             "stab2": att(self.t2), "tera_type": att(self.tera),
             "hp": att(self.maxhp),
             "terastallized": np.array([0, 1], np.int8).reshape(1, 1, 1, 2, 1, 1, 1)}
        # defender types: index 0 is the real typing, index 1 the typing it
        # presents once terastallized (Stellar keeps its own types).
        use_t = self.tera != DT.TIX["STELLAR"]
        dt1 = np.stack([self.t1, np.where(use_t, self.tera, self.t1)], -1)
        dt2 = np.stack([self.t2, np.where(use_t, np.int16(DT.TYPELESS), self.t2)], -1)
        D = {"defense": dfn(self.stats[:, :, 1]), "spd": dfn(self.stats[:, :, 3]),
             "atk": dfn(self.stats[:, :, 0]), "maxhp": dfn(self.maxhp),
             "hp": dfn(self.maxhp), "weight": dfn(self.weight),
             "t1": _sides_of(dt1.astype(np.int16))[1][:, :, None, None, None, :, :],
             "t2": _sides_of(dt2.astype(np.int16))[1][:, :, None, None, None, :, :]}
        for k, v in T.ab.items():
            A[k] = att(v[self.ability])
            D[k] = dfn(v[self.ability])
        for k, v in T.it.items():
            A[k] = att(v[self.item])
            D[k] = dfn(v[self.item])
        D["ab_multiscale"] = np.int8(0)     # applied per leaf (needs current hp)
        mvi = self.mv
        dice = T.it["it_loaded_dice"][self.item] > 0
        M = {"bp": T.mv_bp[mvi], "type": T.mv_type[mvi], "cat": T.mv_cat[mvi],
             "flags": T.mv_flags[mvi], "secondary": T.mv_secondary[mvi],
             "hits": np.where(dice[:, :, None], T.mv_hits_dice[mvi], T.mv_hits[mvi]),
             "bp_kind": T.mv_bp_kind[mvi], "unmodelled": T.mv_unmodelled[mvi],
             "off_is_def": T.mv_off_is_def[mvi], "def_is_phys": T.mv_def_is_phys[mvi],
             "off_is_target": T.mv_off_is_target[mvi]}
        M = {k: _sides_of(v)[0][:, :, :, None, :, None, None] for k, v in M.items()}
        return dict(A=A, D=D, M=M,
                    def_maxhp=dfn(self.maxhp),
                    usable=_sides_of(self.mv_dmg & self.occ[:, :, None])[0]
                    [:, :, :, None, :, None, None])

    def _pair_damage(self, K, weather="NONE"):
        """-> per-MOVE damage as a fraction of the defender's full hp,
        (n, 2, 12 attackers, 4 moves, 12 defenders), where 12 = 6 party slots x
        {not terastallized, terastallized}.

        TWO CROSS-SIDE BLOCKS, not a 12x12. Of the 144 ordered slot pairs, 72
        are same-side ("our mon attacks our mon") and no column ever reads them.
        Block 0 is side 0 attacking side 1, block 1 is side 1 attacking side 0.

        PER MOVE, not per damage CHANNEL. The old table pre-took the maximum
        over each attacker's moves within each of five stat channels (physical,
        special, Psyshock, Body Press, Foul Play -- they differ in WHICH stats
        scale them, which is what makes the leaf's boost multipliers correct).
        That maximum cannot be un-taken, so a disabled move -- Choice lock
        disables three of four -- forced a full static rebuild. Keeping the four
        moves is both exact under any dynamic mask and CHEAPER: four moves per
        attacker where there were five channels, and no argmax loop here."""
        F = {"sun": np.int8(weather == "SUN"), "rain": np.int8(weather == "RAIN"),
             "sand": np.int8(weather == "SAND"), "snow": np.int8(weather == "SNOW")}
        raw = DT.raw_damage(K["A"], K["D"], K["M"], F)   # (n,2,6,2,4,6,2)
        frac = raw / np.maximum(K["def_maxhp"], 1.0)
        frac = np.where(K["usable"], frac, 0.0)
        n = self.n
        return frac.reshape(n, 2, 12, 4, 12)

    def _def_tera_eff(self):
        """Type effectiveness of every attacker MOVE against every defender,
        before and after the DEFENDER terastallizes. (n,2,6,4,6) each; the
        best-move pick is deferred to the leaf, which needs the two ACTIVE
        attackers' rows and nothing else (12 of the 72 rows), and which is the
        only place the dynamic disable/PP mask is known."""
        T = self.T
        mt = _sides_of(T.mv_type[self.mv])[0][..., None]        # (n,2,6,4,1)
        d_t1 = _sides_of(self.t1)[1][:, :, None, None, :]
        d_t2 = _sides_of(self.t2)[1][:, :, None, None, :]
        d_tera = _sides_of(self.tera)[1][:, :, None, None, :]
        # stacked so the leaf gathers ONE array, not two
        return np.stack([DT.CHART_P[mt, d_t1] * DT.CHART_P[mt, d_t2],
                         DT.CHART_P[mt, d_tera]], -1)           # (n,2,6,4,6,2)

    # Per-mon fields the DYNAMIC half reads, in slot order. `level`, `weight`
    # and `scarf` are deliberately absent: they feed the damage kernel and the
    # base speed inside the static build and no column reads them per leaf.
    _MON_FIELDS = ("occ", "maxhp", "stats", "t1", "t2", "tera", "mv", "prio",
                   "chan", "mv_dmg", "mv_real", "mv_phys", "cap", "setup_boost",
                   "setup_present", "base_speed", "ability", "item", "pp_max")
    _PAIR_FIELDS = ("dmg", "eff_te")

    def _pack(self):
        """Group the per-mon fields by dtype into a handful of (n,12,K) blocks.

        `reorder()` runs on EVERY leaf, and it was nineteen `take_along_axis`
        calls moving 632 elements -- 130 us at n=1, all of it dispatch. Packing
        by dtype makes it five, without touching a single value (each field
        comes back as a VIEW at its own offset, same dtype, same bytes)."""
        groups = {}
        for k in self._MON_FIELDS:
            v = getattr(self, k)
            groups.setdefault(v.dtype.str, []).append(k)
        arrs, specs = [], []
        for _dt, keys in sorted(groups.items()):
            parts, spec, o = [], [], 0
            for k in keys:
                v = getattr(self, k)
                w = 1 if v.ndim == 2 else v.shape[2]
                parts.append(v[:, :, None] if v.ndim == 2 else v)
                spec.append((k, o, w if v.ndim > 2 else 0))
                o += w
            arrs.append(np.concatenate(parts, axis=2))
            specs.append(tuple(spec))
        self.pack = tuple(arrs)
        self.pack_spec = tuple(specs)

    def reorder(self, pidx):
        """Party order -> this leaf's canonical slot order, PER-MON fields only.
        `pidx[b, slot]` is the party index (0-11, side-major) in that slot. The
        pair tables are left alone and reordered by `pair()` after the weather
        variant has been picked -- gathering all five variants per leaf cost
        4.7x the whole dynamic half."""
        out = StaticCtx.__new__(StaticCtx)
        out.n, out.T = self.n, self.T
        ix = pidx[:, :, None]
        for arr, spec in zip(self.pack, self.pack_spec):
            g = np.take_along_axis(arr, ix, axis=1)
            for k, o, w in spec:
                setattr(out, k, g[:, :, o] if w == 0 else g[:, :, o:o + w])
        for f in self._PAIR_FIELDS:
            setattr(out, f, getattr(self, f))
        return out

    @staticmethod
    def pair(v, aix, dix):
        """(n,2,12,4,12) party-order damage table for THIS leaf's weather ->
        (n,2,6,4,6) in this leaf's slot order.

        `aix[b, side, slot]` = 2 * (within-side party index) + (that mon is
        terastallized), and `dix` is the same thing for the opposing side. One
        gather therefore does the slot permutation AND the tera selection, so
        tera costs the dynamic half exactly nothing."""
        v = np.take_along_axis(v, aix[:, :, :, None, None], axis=2)
        return np.take_along_axis(v, dix[:, :, None, None, :], axis=4)

    @staticmethod
    def active_row(v, oidx):
        """(n,2,6,4,6,2) -> (n,2,4,6,2) for the ACTIVE attacker only, defenders
        in slot order. The only rows `tera_best_value` reads -- 12 of 72."""
        v = np.take_along_axis(v, oidx[:, :, :1, None, None, None], axis=2)[:, :, 0]
        return np.take_along_axis(v, oidx[:, ::-1, None, :, None], axis=3)

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
        out.pack = tuple(np.broadcast_to(a, (n,) + a.shape[1:]) for a in self.pack)
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
    sboost = si[:, :, _SI["attack_boost"]:_SI["attack_boost"] + 7].astype(np.float32)
    scond = si[:, :, LL._SC0:LL._SC0 + len(LE.SIDE_CONDITION_FIELDS)].astype(np.float32)
    sdur = si[:, :, LL._DUR0:LL._DUR0 + len(LE.DURATION_FIELDS)].astype(np.float32)
    trick_room = gi[:, _GI["trick_room"]].astype(bool)
    weather = gi[:, _GI["weather"]].astype(np.int64)
    terrain = gi[:, _GI["terrain"]].astype(np.int64)
    # `disabled` and `pp` are read HERE, not in the static context: a new
    # disable (2.7% of real transitions -- Choice lock and Encore both set it)
    # must never cost a static rebuild.
    dis4 = m[:, :, _MI["disabled0"]:_MI["disabled0"] + 4]
    pp4 = m[:, :, _MI["pp0"]:_MI["pp0"] + 4]
    live_mv = (dis4 == 0) & (pp4 > 0)                          # (n,12,4)
    mv_present = S.mv_real & live_mv     # a real, usable move (status included)

    boost12 = np.zeros((n, 12, 7), np.float32)
    boost12[:, 0] = sboost[:, 0]
    boost12[:, 6] = sboost[:, 1]                 # boosts belong to the ACTIVE

    # ================= effective speed (§3 layer 2) =====================
    spe = S.base_speed.copy()
    para = (status == STATUS_ORDER.index("PARALYZE"))
    qf = T.ab_quickfeet[S.ability] > 0
    spe = spe * np.where(para, np.where(qf, 1.5, 0.5),
                         np.where(qf & (status > 0), 1.5, 1.0))
    spe = spe * _boost(boost12[:, :, 4])
    spe = spe * np.where(T.ab_weather_speed[S.ability, weather[:, None]] > 0, 2.0, 1.0)
    spe = spe * np.where((T.ab_surgesurfer[S.ability] > 0)
                         & (terrain[:, None] == LE.TERRAIN_IX["ELECTRICTERRAIN"]), 2.0, 1.0)
    # Tailwind is a SIDE condition and the four speed volatiles belong to the
    # ACTIVE. Both used to be `np.repeat`-ed to twelve slots first. Every factor
    # here is an exact power of two except the last, so applying them in this
    # order is bit-identical to applying them one slot-array at a time.
    spe6 = spe.reshape(n, 2, 6)
    spe6 *= np.where(scond[:, :, SC["tailwind"]] > 0, 2.0, 1.0)[:, :, None]
    spe6[:, :, 0] *= (np.where(vol[:, :, VIX_SLOWSTART], 0.5, 1.0)
                      * np.where(vol[:, :, VIX_UNBURDEN], 2.0, 1.0)
                      * np.where(vol[:, :, VIX_PROTO_SPE]
                                 | vol[:, :, VIX_QUARK_SPE], 1.5, 1.0))
    spe = np.maximum(spe, 1.0)
    # Trick Room inverts the comparison, not the number: fold it into a key.
    skey = np.where(trick_room[:, None], -spe, spe)

    # ================= dynamic damage (the spec's multiplier path) =======
    # PER-MOVE static fraction of FULL hp -> fraction of CURRENT hp, with the
    # disable/PP mask, boosts, screens, burn, Multiscale and Supreme Overlord
    # applied. Everything is (n,2,6,4,6): the two CROSS-SIDE blocks, per move.
    # Attacker-side quantities take axis 2, defender-side quantities axis 4.
    chan_a = _sides_of(S.chan)[0]                              # (n,2,6,4)
    phys_a = _sides_of(S.mv_phys)[0]
    prio_a = _sides_of(S.prio)[0]
    okm = _sides_of(S.mv_dmg & live_mv & occ[:, :, None])[0]
    stage = boost12[:, :, :5]
    st_a, st_d = _sides_of(stage)                              # (n,2,6,5) each
    # Defender boost ratios are exactly 1.0 off the opposing ACTIVE, so only
    # that ONE column of six needs them -- `boost12` is written at slots 0 and 6
    # and nowhere else, which makes the bench x bench quadrant (50 of the 72
    # pairs) boost-free by construction, not by luck.
    db = _boost(st_d[:, :, 0])[:, :, None, None, :]         # (n,2,1,1,5)
    c5 = chan_a[..., None]
    num_d0 = np.where(c5 == 4, db[..., 0:1], 1.0)           # (n,2,6,4,1)
    den_d0 = np.where(c5 == 1, db[..., 3:4], db[..., 1:2])

    guts_a = _sides_of(T.ab["ab_guts"][S.ability] > 0)[0]
    infil_a = _sides_of(T.ab["ab_infiltrator"][S.ability] > 0)[0]
    # A screen belongs to the DEFENDING side and is uniform across its six mons,
    # so `refl`/`lscr` are per (block, attacker) once Infiltrator is folded in --
    # they were being materialised as 6x6.
    veil = scond[:, :, SC["aurora_veil"]] > 0
    refl_d = ((scond[:, :, SC["reflect"]] > 0) | veil)[:, ::-1, None] & ~infil_a
    lscr_d = ((scond[:, :, SC["light_screen"]] > 0) | veil)[:, ::-1, None] & ~infil_a
    burn_mult = (np.where(_sides_of(status == STATUS_ORDER.index("BURN"))[0]
                          & ~guts_a, 0.5, 1.0)
                 * np.where(guts_a & _sides_of(status > 0)[0], 1.5, 1.0))
    # burn/Guts and the screen are both per (block, attacker) and only apply on
    # the matching category, so they fold into ONE per-move factor. Every screen
    # factor is a power of two, so the fold is exact.
    br4 = np.where(phys_a, burn_mult[:, :, :, None], 1.0) * np.where(
        np.where(phys_a, refl_d[:, :, :, None], lscr_d[:, :, :, None]), 0.5, 1.0)
    ms_d = _sides_of((T.ab["ab_multiscale"][S.ability] > 0) & (hp >= maxhp))[1]
    # Supreme Overlord: +10%% per fainted ally, capped at +50%%
    fainted12 = (~alive) & occ
    n_fainted_ally = np.repeat(np.stack([fainted12[:, :6].sum(1),
                                         fainted12[:, 6:].sum(1)], 1), 6, axis=1)
    so_mult = _sides_of(np.where(T.ab_supreme[S.ability] > 0,
                                 1.0 + 0.1 * np.minimum(n_fainted_ally, 5), 1.0))[0]
    # Multiscale (defender) and Supreme Overlord (attacker) fold together too.
    msso = so_mult[:, :, :, None] * np.where(ms_d, 0.5, 1.0)[:, :, None, :]

    tera_on = m[:, :, _MI["terastallized"]] > 0
    aix = order * 2 + tera_on.reshape(n, 2, 6)      # slot -> (party, tera) pair
    dix = aix[:, ::-1]
    dmg_w = S.dmg[np.arange(n), WEATHER_VAR_MAP[weather]]  # weather variant
    dmg_moves = S.pair(dmg_w, aix, dix)                    # (n,2,6,4,6)

    def combine(dm, numm, def0=False):
        """per-move damage table + masked attacker numerator -> scaled damage.

        `def0=True` keeps only the OPPOSING ACTIVE as defender, which is all the
        tera counterfactual ever reads."""
        x = dm * numm                              # mask + attacker boost ratio
        x0 = x if def0 else x[:, :, :, :, :1]      # defender boosts: active only
        x0 *= num_d0
        x0 /= den_d0
        x = x * br4[..., None]                     # -> float64, exactly as before
        return x * (msso[:, :, :, None, :1] if def0 else msso[:, :, :, None, :])

    chan_c = np.clip(chan_a, 0, 3)

    def numerator(sa):
        """masked per-move attacker boost ratio, (n,2,6,4,1). Channels 0..3 take
        the attacker's atk / spa / spa / def; channel 4 (Foul Play) takes the
        TARGET's attack, which is defender-side and lands in `num_d0`."""
        num = np.take_along_axis(_boost(sa[..., _NUM_STAT]), chan_c, axis=-1)
        return np.where(okm, np.where(chan_a == 4, 1.0, num), 0.0)[..., None]

    numm_now = numerator(st_a)                 # reused by the tera counterfactual
    x_now = combine(dmg_moves, numm_now)
    bmv = x_now.argmax(axis=3)                             # (n,2,6,6)
    dmg_full = np.take_along_axis(x_now, bmv[:, :, :, None, :], axis=3)[:, :, :, 0, :]
    dmg_prio = np.take_along_axis(np.broadcast_to(prio_a[..., None], x_now.shape),
                                  bmv[:, :, :, None, :], axis=3)[:, :, :, 0, :]
    full_ohko = dmg_full >= 1.0            # unmasked: Revival Blessing reads it
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
    M = L.M                                  # resolved once per layout, not per leaf
    occ1 = occ[:, :, None]
    mon[:, :, M["hp_frac"]] = hp_frac
    mon[:, :, M["maxhp"]] = maxhp / LE.D_HP
    # EFFECTIVE stats (spec §1: "computed, all modifiers"). The divisor is
    # 2 x D_STAT, not D_STAT: a +6 boost multiplies by 4, and the lossless
    # encoder's D_STAT is sized for RAW stats only, so /D_STAT put boosted
    # columns outside [-1, 1] (caught by the corpus census).
    ia = M["attack"]
    mon[:, :, ia:ia + 4] = np.minimum(
        S.stats[:, :, :4] * _boost(stage[:, :, :4]) / D_STAT_EFF, 1.0) * occ1
    mon[:, :, ia + 4] = np.minimum(spe / D_STAT_EFF, 1.0) * occ
    np.put_along_axis(mon[:, :, M["status_none"]:M["status_none"] + 7],
                      status[:, :, None], occ1.astype(np.float32), axis=-1)
    mon[:, :, M["sleep_rest_turns"]] = np.maximum(
        m[:, :, _MI["sleep_turns"]], m[:, :, _MI["rest_turns"]]) / LE.D_SLEEP
    mon[:, :, M["times_attacked"]] = m[:, :, _MI["times_attacked"]] / LE.D_TIMES_ATTACKED
    mon[:, :, M["terastallized"]] = m[:, :, _MI["terastallized"]] * occ
    mon[:, :, M["alive"]] = alive
    mon[:, :, M["pp_frac0"]:M["pp_frac0"] + 4] = np.clip(pp4 / S.pp_max, 0.0, 1.0) * occ1
    mon[:, :, M["disabled0"]:M["disabled0"] + 4] = dis4 * occ1
    rev = m[:, :, _MI["revmask"]].astype(np.int64)
    mon[:, :, M["seen_species"]:M["seen_species"] + 8] = \
        ((rev[:, :, None] >> _BITS8) & 1) * occ1
    mon[:, :, L.I_CAP:L.I_CAP + len(CAP_NAMES)] = \
        S.cap * alive[:, :, None]          # dead mons' capability bools read 0
    # effective (tera-aware) defensive typing + own typing, 18 bits each. The
    # scratch buffer is 20 wide so DT.TYPELESS (19) and STELLAR (18) have a
    # column to land in that is then sliced away -- i.e. they set no bit.
    use_t = S.tera != DT.TIX["STELLAR"]
    et1 = np.where(tera_on & use_t, S.tera, S.t1).astype(np.int64)[:, :, None]
    et2 = np.where(tera_on & use_t, np.int16(DT.TYPELESS), S.t2).astype(np.int64)[:, :, None]
    tb = np.zeros((n, 12, DT.N_TYPES + 1), np.float32)
    occf = occ1.astype(np.float32)
    np.put_along_axis(tb, et1, occf, axis=-1)
    np.put_along_axis(tb, et2, occf, axis=-1)
    i_te = L.M["type_eff_" + DT.TYPES[0].lower()]
    mon[:, :, i_te:i_te + N_REAL_TYPES] = tb[:, :, :N_REAL_TYPES]
    tb[:] = 0.0
    np.put_along_axis(tb, S.t1.astype(np.int64)[:, :, None], occf, axis=-1)
    np.put_along_axis(tb, S.t2.astype(np.int64)[:, :, None], occf, axis=-1)
    i_to = L.M["type_own_" + DT.TYPES[0].lower()]
    mon[:, :, i_to:i_to + N_REAL_TYPES] = tb[:, :, :N_REAL_TYPES]
    feats[:, L.O_MON:L.O_MON + 12 * L.NM] = mon.reshape(n, -1)

    # ---- §2 per-side -----------------------------------------------------
    side = np.zeros((n, 2, L.NS), np.float32)
    SS = L.SS
    # The ten hazard/screen columns and their four turn counters, as two array
    # ops. This was a ten-iteration python loop with dict lookups: 189 us at
    # n=1 for 412 element-ops, the worst work-per-microsecond in the encoder.
    # min(v, 1)/1 is exactly (v > 0) for the integer-valued counters, so the
    # /1, /2 and /3 columns share one expression.
    side[:, :, L.SC_DST] = np.minimum(scond[:, :, L.SC_SRC], L.SC_DIV) / L.SC_DIV
    side[:, :, L.SCT_DST] = scond[:, :, L.SCT_SRC] / 8.0
    side[:, :, SS["boost_atk"]:SS["boost_atk"] + 7] = sboost / LE.D_BOOST
    # the engine stores the toxic counter per SIDE (only the active can hold
    # one), so it is a side column, not twelve mon columns
    side[:, :, SS["toxic_counter"]] = scond[:, :, SC["toxic_count"]] / LE.D_SIDECOND
    side[:, :, SS["vol_" + VOL_COLS[0].lower()]:
         SS["vol_" + VOL_COLS[0].lower()] + len(VOL_COLS)] = vol[:, :, VOL_IX]
    if DUR_COLS:
        side[:, :, SS["dur_" + DUR_COLS[0]]:
             SS["dur_" + DUR_COLS[0]] + len(DUR_COLS)] = sdur[:, :, DUR_IX] / LE.D_DURATION
    act_maxhp = maxhp.reshape(n, 2, 6)[:, :, 0]
    side[:, :, SS["substitute_hp_frac"]] = np.clip(
        si[:, :, _SI["substitute_health"]].astype(np.float32)
        / np.maximum(act_maxhp / 4.0, 1.0), 0.0, 1.0)
    side[:, :, L.SI_DST] = si[:, :, L.SI_SRC] / L.SI_DIV
    side[:, :, SS["future_sight_source"]] = LL._party_to_slot(
        si[:, :, _SI["fs1"]].astype(np.int64),
        si[:, :, _SI["active_index"]].astype(np.int64)) / 5.0
    # move state, encoded by meaning
    lum_kind = si[:, :, _SI["lum_kind"]]
    lum_slot = si[:, :, _SI["lum_slot"]].astype(np.int64)
    act_item = S.item.reshape(n, 2, 6)[:, :, 0]
    used_move = lum_kind == 1
    side[:, :, SS["lock_choice"]] = (T.it_choice[act_item] > 0) & used_move
    side[:, :, SS["lock_encore"]] = vol[:, :, VIX_ENCORE]
    side[:, :, SS["lock_disable"]] = dis4.reshape(n, 2, 6, 4)[:, :, 0].max(axis=2) > 0
    side[:, :, SS["lock_multiturn"]] = vol[:, :, VIX_LOCKEDMOVE]
    side[:, :, SS["just_switched_in"]] = lum_kind == 2

    side[:, :, SS["tera_available"]] = 1.0 - tera_on.reshape(n, 2, 6).any(axis=2)
    fainted = fainted12
    side[:, :, SS["n_fainted_revivable"]] = \
        fainted.reshape(n, 2, 6).sum(axis=2) / 6.0

    # locked / charging move embedding ids
    lock_id = np.where(used_move, np.take_along_axis(
        S.mv.reshape(n, 2, 6, 4)[:, :, 0], np.clip(lum_slot, 0, 3)[:, :, None],
        axis=2)[:, :, 0], 0)
    # charging_move_id: only SOLARBEAM and METEORBEAM are reachable two-turn
    # moves in gen9 randbats (REACHABLE_VOLATILES.md), so the volatile names
    # the move and we resolve it back to the active's own move slot.
    charge_id = np.zeros((n, 2), np.int64)
    mv2 = S.mv.reshape(n, 2, 6, 4)[:, :, 0]                      # (n,2,4)
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
    skey6 = skey.reshape(n, 2, 6)
    pair_speed = _first(skey6[:, 0, :, None], skey6[:, 1, None, :])   # (n,6,6)
    assert L.ko_versions == 1, "the full-hp KO matrix was dropped (see §10)"
    # block 0 is us->them; block 1 is them->us, transposed so the row index
    # stays OUR mon in both directions. Both directions bucket in one pass.
    nd = L.ko_directions
    sub = np.stack([ko_now[:, 0], ko_now[:, 1].transpose(0, 2, 1)][:nd], 1)
    nv = np.stack([never_now[:, 0], never_now[:, 1].transpose(0, 2, 1)][:nd], 1)
    oh = np.zeros((n, nd, 6, 6, 5), np.float32)
    _bucket_onehot(sub, 5, oh, nv)
    rel[:, o:o + 180 * nd] = oh.reshape(n, -1)
    o += 180 * nd
    rel[:, o:o + 36] = pair_speed.reshape(n, -1)
    o += 36
    v_us, v_them = spe[:, 0], spe[:, 6]
    marg = np.tanh(np.log(np.maximum(v_us, 1.0) / np.maximum(v_them, 1.0)))
    rel[:, o] = np.where(trick_room, -marg, marg)
    rel[:, o + 1] = v_us == v_them
    o += 2
    # 16 move-order bits: my move i vs their move j
    prio2 = S.prio.reshape(n, 2, 6, 4)[:, :, 0]                  # (n,2,4)
    pres2 = mv_present.reshape(n, 2, 6, 4)[:, :, 0]
    pu, pt = prio2[:, 0], prio2[:, 1]
    spd_first = _first(skey6[:, 0, 0], skey6[:, 1, 0])
    order16 = np.where(pu[:, :, None] != pt[:, None, :],
                       (pu[:, :, None] > pt[:, None, :]).astype(np.float32),
                       spd_first[:, None, None])
    rel[:, o:o + 16] = (order16 * pres2[:, 0][:, :, None]
                        * pres2[:, 1][:, None, :]).reshape(n, -1)
    o += 16
    # derived priority flags
    best_p = np.where(pres2, prio2, -9).max(axis=2)              # (n,2)
    best_pu, best_pt = best_p[:, 0], best_p[:, 1]
    rel[:, o] = best_pu > np.maximum(best_pt, 0)
    rel[:, o + 1] = best_pt > np.maximum(best_pu, 0)
    # revenge: an opposing mon has a priority move that kills our active
    # (the opposing ACTIVE is index 0 of the other side's block)
    alive6 = alive.reshape(n, 2, 6)
    prio_kill = ((dmg_prio[:, :, :, 0] > 0)
                 & (dmg_full[:, :, :, 0] >= hpf12.reshape(n, 2, 6)[:, ::-1, 0][:, :, None])
                 & alive6).any(axis=2)                           # (n,2)
    rel[:, o + 2] = prio_kill[:, 1]                              # they kill us
    rel[:, o + 3] = prio_kill[:, 0]
    o += 4
    rel[:, o:o + 48] = (S.prio / 7.0 * mv_present).reshape(n, -1)  # gen9 range -7..+5
    o += 48
    assert o == L.NR, (o, L.NR)
    feats[:, L.O_REL:L.O_REL + L.NR] = rel

    # ---- §4 counterfactual ----------------------------------------------
    cfm = np.zeros((n, 12, L.NCM), np.float32)
    CM = L.CM
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
    alive_x, alive_o = _sides_of(alive)                     # mine / opposing
    opp = alive_o[:, :, None, :]
    n_opp = alive_o.sum(axis=2)[:, :, None]
    hpf_x, hpf_o = _sides_of(hpf12)
    hpf_opp = hpf_o[:, :, None, :]                          # (n,2,1,6)
    hpf_mine = hpf_x[:, :, :, None]                         # (n,2,6,1)
    hpfr_mine = _sides_of(hp_frac)[0]                       # (n,2,6)
    alive_r, setup_r = alive_x, _sides_of(S.setup_present)[0]
    skey_opp = _sides_of(skey)[1][:, :, None, :]            # (n,2,1,6)
    # their priority answer and the acting set do not depend on the setup level
    prio_ans = ((incx_prio > 0) & (incx >= hpf_mine) & opp).any(axis=3)
    untera = alive_x & ~tera_on.reshape(n, 2, 6)
    for lv in SETUP_LEVELS:
        # the setup move's own boost profile, applied lv times on top of the
        # boosts already on the board
        st = np.clip(stage + S.setup_boost * lv, -6, 6)
        num1 = numerator(_sides_of(st)[0])
        d2 = combine(dmg_moves, num1)
        d2 = np.where(live_pair, d2.max(axis=3), 0.0)
        spe2 = spe * (_boost(st[:, :, 4]) / _boost(stage[:, :, 4]))
        skey2 = np.where(trick_room[:, None], -spe2, spe2)
        skey2_mine = _sides_of(skey2)[0][:, :, :, None]     # (n,2,6,1)
        ohko = ((d2 >= hpf_opp) & opp).sum(axis=3)
        outspd_n = ((_first(skey2_mine, skey_opp) > 0) & opp).sum(axis=3)
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
        sweep1 = sweep
        # ---- tera-enabled sweep, the SAME STRICT CONJUNCTION ---------------
        # The previous implementation asked a different and much weaker
        # question -- "does tera turn a non-kill on the opposing ACTIVE into a
        # kill, for a mon that owns a setup move" -- which is a flag for
        # NEEDING tera, and it measured with the WRONG SIGN on real labels
        # (E[win | our flag] = 0.311 against 0.349 with tera available and the
        # flag off; and the same direction for BOTH sides, i.e. incoherent).
        # It now asks what EVALUATOR_SPEC §4 states: does terastallizing create
        # a mon that OHKOs EVERY living opponent on the minimum roll, outspeeds
        # every living opponent, and has no priority answer.
        #
        # "alone, or composed with one setup" is ONE evaluation, not two: `st`
        # already carries +1 only where a setup move exists (`setup_boost` is
        # zero elsewhere), and setup boosts are non-negative, so the +1 branch
        # dominates the +0 branch exactly where the +0 branch is available.
        # Speed is untouched by tera, so `outspd_n` is reused as it stands.
        d2t = combine(S.pair(dmg_w, order * 2 + 1, dix), num1)
        d2t = np.where(live_pair, d2t.max(axis=3), 0.0)
        tera_sweep = ((((d2t * DT.MIN_ROLL >= hpf_opp) & opp).sum(axis=3) == n_opp)
                      & (n_opp > 0) & (outspd_n == n_opp) & ~prio_ans & untera)
    feats[:, L.O_CFM:L.O_CFM + 12 * L.NCM] = cfm.reshape(n, -1)

    # ---- §4 per-side aggregates + tera ----------------------------------
    cfs = np.zeros((n, 2, L.NCS), np.float32)
    CS = L.CS
    cfs[:, :, CS["best_sweep_1"]] = sweep1.max(axis=2)
    # free turn: does THEIR active survive MY best move (spec §4, literal).
    # Side s attacks in block s; the opposing active is defender index 0.
    cfs[:, :, CS["free_turn"]] = np.clip(
        hp_frac.reshape(n, 2, 6)[:, ::-1, 0] - dmg_full[:, :, 0, 0], 0.0, 1.0)
    # answers to their best threat. Both sides at once: block 1-s is "they
    # attack me", which is just the block axis reversed.
    dfr = dmg_full[:, ::-1]                     # dfr[:, s] = "they attack me"
    skey_x = _sides_of(skey)[0]
    threat = ((dfr >= hpf_x[:, :, None, :]) & alive_o[:, :, :, None]
              & alive_x[:, :, None, :]).sum(axis=3)
    worst = threat.argmax(axis=2)               # (n,2), index within the OTHER side
    w4 = worst[:, :, None, None]
    kill_it = (np.take_along_axis(dmg_full, w4, axis=3)[:, :, :, 0]
               >= np.take_along_axis(hpf_o, worst[:, :, None], axis=2))
    faster_than_it = _first(skey_x, np.take_along_axis(
        _sides_of(skey)[1], worst[:, :, None], axis=2)) > 0
    survive_it = np.take_along_axis(dfr, w4, axis=2)[:, :, 0, :] < hpfr_mine
    cfs[:, :, CS["answers_to_best_threat"]] = \
        (kill_it & (faster_than_it | survive_it) & alive_x).sum(axis=2) / 6.0
    # tera: exactly two features per side (spec §4 -- priced as an option, not
    # as a per-mon matrix). Both the offensive and the defensive half read only
    # the OPPOSING ACTIVE, so the tera damage table is combined against defender
    # 0 alone -- one column of six -- and `def_tera_mult` only ever needed the
    # two active attackers' rows, which is why the best-move pick moved here.
    dmg_tera_full = np.where(live_pair[:, :, :, :1],
                             combine(S.pair(dmg_w, order * 2 + 1, dix[:, :, :1]),
                                     numm_now, def0=True).max(axis=3), 0.0)[:, :, :, 0]
    ete = StaticCtx.active_row(S.eff_te, order)            # (n,2,4,6,2)
    eb = np.where(okm[:, :, 0][:, :, :, None], ete[..., 0], -1.0)
    bt = eb.argmax(axis=2)[:, :, None, :]                  # (n,2,1,6)
    ba = np.take_along_axis(ete, bt[..., None], axis=2)[:, :, 0]      # (n,2,6,2)
    before = np.maximum(np.take_along_axis(eb, bt, axis=2)[:, :, 0], 0.0)
    def_tera_mult = (ba[..., 1] / np.maximum(before, 1e-3)).astype(np.float32)
    avail = side[:, :, SS["tera_available"]] > 0
    d_no = dmg_full[:, :, :, 0]
    tgt = hpf_o[:, :, :1]
    off_gain = np.where((dmg_tera_full >= tgt) & (d_no < tgt), 1.0,
                        np.clip(dmg_tera_full - d_no, 0.0, 1.0))
    # defensive: the opposing active's damage into me, scaled by the type
    # multiplier my tera type would impose
    inc = dfr[:, :, 0, :]
    inc_te = inc * def_tera_mult[:, ::-1]
    mine_hp = hp_frac.reshape(n, 2, 6)
    def_gain = np.where((inc >= mine_hp) & (inc_te < mine_hp), 1.0,
                        np.clip((inc - inc_te) / np.maximum(mine_hp, 1e-6), 0.0, 1.0))
    gain = np.where(untera, np.maximum(off_gain, def_gain), 0.0)
    cfs[:, :, CS["tera_best_value"]] = np.where(avail, gain.max(axis=2), 0.0)
    # tera-enabled sweep: computed in the counterfactual block above, where the
    # setup-level boosts and the sweep conjunction already live.
    cfs[:, :, CS["tera_enabled_sweep"]] = np.where(avail, tera_sweep.max(axis=2), 0.0)
    # best revival target score: the strongest fainted mon, priced by how many
    # living opponents it would OHKO at full HP. This deliberately reads the
    # UNMASKED damage: every other matchup feature excludes fainted mons (spec
    # design rule), but Revival Blessing is precisely the question "what is the
    # dead mon worth", so masking it made the column a constant zero.
    score = (full_ohko & alive_o[:, :, None, :]).sum(axis=3) / 6.0
    side[:, :, SS["best_revival_target_score"]] = \
        np.where(fainted.reshape(n, 2, 6), score, 0.0).max(axis=2)
    feats[:, L.O_SIDE:L.O_SIDE + 2 * L.NS] = side.reshape(n, -1)
    feats[:, L.O_CFS:L.O_CFS + 2 * L.NCS] = cfs.reshape(n, -1)

    # ---- §5 global -------------------------------------------------------
    glob = np.zeros((n, L.NG), np.float32)
    G = L.G
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
                     never=never_now, sweep1=sweep1.reshape(n, 12),
                     mv_present=mv_present,
                     cap=S.cap, prio=S.prio, setup_boost=S.setup_boost, static=S)
    return ids, feats


def _boost(stage):
    """Boost stage -> stat multiplier. `clip(x,-6,6).astype(int)+6` and
    `clip(x.astype(int)+6, 0, 12)` agree on every integer-valued stage, and the
    second is one numpy call cheaper -- which matters, because this runs on
    every leaf about ten times."""
    return DT.BOOST_MULT_TABLE[np.clip(stage.astype(np.int64) + 6, 0, 12)]


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
