"""Per-decision turn table for an archived ladder game.

    python3 ladder-games/analysis/turn_table.py [game_dir_or_battle_id]

With no argument it uses the MOST RECENTLY ARCHIVED game. The argument may be a
full archive dir, a bare dir name, or any substring of one (e.g. a battle id).

Output is one markdown row per DECISION (not per turn -- a pivot or a faint
replacement makes a second decision inside the same turn, and both are shown
with the same turn number):

    | Turn | Top 1 | Top 2 | Opp top 1 | Opp top 2 | Opp actual |

Each move cell is two lines: the move on line 1, `share%, score` on line 2.
`--timing` adds the per-decision search wall clock and budget.

WHERE THE NUMBERS COME FROM
  battle.log.gz, written by fp/search/selection.py at INFO:
    * `WorldStats <w>`     our per-world arms:  move share%/avg_score/±within-world sd
    * `OppWorldStats <w>`  the OPPONENT MODEL's per-world arms, same shape minus the sd
    * `Policy <w>: ... sample_chance_multiplier=` the world's posterior weight
  Shares and scores here are POOLED over the sampled worlds exactly the way
  `_aggregate_results` pools them -- share is sample-chance-weighted and
  renormalised, score is the sample-chance-weighted mean over the worlds that
  visited the arm -- so Top 1 is the arm the bot actually played under
  --selection-argmax-only (the one exception is a turn where the tera gate
  overrode the argmax; the `played` value is read from the ARGMAX-ONLY line, so
  the join stays correct there).

  `Opp actual` is the move the opponent REALLY made, joined from
  opponent_ledger.jsonl (battle_modifier stamps the observed action against the
  prediction stashed by selection.py). It is bolded when it appears in the
  model's top 2. Rows with `—` in the opponent columns are our own forced
  switches, where side_two had no decision to make.

SCORES ARE v8-ABSOLUTE. Under the v8 nets `PE_NN_REWARD=absolute`, so a score
IS the net's P(win) from our side and 0.50 is even. Under a relative-reward net
(v6 and earlier) the same column is a sigmoid around the ROOT eval and is not a
win probability -- do not compare the two across net generations.
"""
import gzip
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LADDER = os.path.dirname(HERE)
ROOT = os.path.dirname(LADDER)
LEDGER = os.path.join(ROOT, "opponent_ledger.jsonl")

RE_TURN = re.compile(r"\|turn\|(\d+)")
RE_POLICY = re.compile(r"Policy (\d+): .* sample_chance_multiplier=([\d.]+)")
# (?<!Opp) matters: "OppWorldStats" contains "WorldStats", so without it every
# opponent line is also parsed as ours and our side comes out empty.
RE_WS = re.compile(r"(?<!Opp)WorldStats (\d+): (.*)")
RE_OWS = re.compile(r"OppWorldStats (\d+): (.*)")
RE_ARM_OURS = re.compile(r"(.+) ([\d.]+)%/(-?[\d.]+)/")   # ours carries the ±sd
RE_ARM_OPP = re.compile(r"(.+) ([\d.]+)%/(-?[\d.]+)$")
RE_CHOICE = re.compile(r"selection.*?: (.+?) with")
RE_TIMING = re.compile(
    r"TurnTiming: elapsed=([\d.]+)s budget_per_world=(\d+)ms worlds=(\d+) waves=(\d+)"
)
RE_PLAYER = re.compile(r"^\|player\|(p[12])\|([^|]+)\|")
RE_TERA = re.compile(r"^\|-terastallize\|(p[12])a")
RE_GATE = re.compile(r"Tera gate: (BLOCKED|allowed) (\S+)")


def find_game(arg=None):
    games = os.path.join(LADDER, "games")
    dirs = sorted(
        (d for d in os.listdir(games) if os.path.isdir(os.path.join(games, d))),
        key=lambda d: os.path.getmtime(os.path.join(games, d)),
        reverse=True,
    )
    if not dirs:
        raise SystemExit("no archived games in %s" % games)
    if not arg:
        return os.path.join(games, dirs[0])
    if os.path.isdir(arg):
        return arg
    matches = [d for d in dirs if arg in d]
    if not matches:
        raise SystemExit("no archived game matching %r" % arg)
    return os.path.join(games, matches[0])


def parse(game_dir):
    """battle.log.gz -> list of decisions, in play order."""
    path = os.path.join(game_dir, "battle.log.gz")
    if not os.path.isfile(path):
        raise SystemExit("no battle.log.gz in %s" % game_dir)
    decisions, cur, turn = [], None, 0
    players, tera_turns = {}, {}
    with gzip.open(path, "rt", errors="replace") as fh:
        for line in fh:
            m = RE_PLAYER.match(line)
            if m:
                players.setdefault(m.group(1), m.group(2).strip())
            m = RE_TERA.match(line)
            if m:
                tera_turns.setdefault(m.group(1), turn)
            m = RE_TURN.search(line)
            if m:
                turn = int(m.group(1))
            m = RE_POLICY.search(line)
            if m:
                # the first Policy line of a decision opens it; `turn` is
                # whatever the protocol last announced, which is this decision's
                if cur is None:
                    cur = {"turn": turn, "chance": {}, "ws": {}, "ows": {}}
                cur["chance"][int(m.group(1))] = float(m.group(2))
            if cur is not None:
                m = RE_OWS.search(line)
                if m:
                    cur["ows"][int(m.group(1))] = _arms(m.group(2), RE_ARM_OPP)
                    continue
                m = RE_WS.search(line)
                if m:
                    cur["ws"][int(m.group(1))] = _arms(m.group(2), RE_ARM_OURS)
                if "ARGMAX-ONLY" in line:
                    m = RE_CHOICE.search(line)
                    if m:
                        cur["choice"] = m.group(1)
                m = RE_GATE.search(line)
                if m:
                    cur["gate"] = (m.group(1), m.group(2))
                m = RE_TIMING.search(line)
                if m:
                    # TurnTiming is the last line of a decision: close it out
                    cur["elapsed_ms"] = int(float(m.group(1)) * 1000)
                    cur["budget_ms"] = int(m.group(2))
                    cur["worlds"] = int(m.group(3))
                    cur["waves"] = int(m.group(4))
                    decisions.append(cur)
                    cur = None
    return decisions, players, tera_turns


def _arms(blob, pattern):
    out = {}
    for part in blob.split(" | "):
        m = pattern.match(part)
        if m:
            out[m.group(1)] = (float(m.group(2)) / 100.0, float(m.group(3)))
    return out


def pool(worlds, chance, top=2):
    """Sample-chance-weighted pooling, mirroring selection._aggregate_results.

    share  = sum_w chance_w * share_w(c), renormalised over the arms present
    score  = sum_w chance_w * avg_w(c) / sum_w chance_w   (worlds that visited c)
    """
    share, ssum, wsum = {}, {}, {}
    for w, arms in worlds.items():
        c = chance.get(w, 1.0 / max(len(worlds), 1))
        for name, (s, avg) in arms.items():
            share[name] = share.get(name, 0.0) + c * s
            ssum[name] = ssum.get(name, 0.0) + c * avg
            wsum[name] = wsum.get(name, 0.0) + c
    total = sum(share.values()) or 1.0
    ranked = sorted(
        ((n, share[n] / total, ssum[n] / wsum[n]) for n in share), key=lambda x: -x[1]
    )
    return ranked[:top]


def cell(ranked, i):
    # "No Move" is the engine's forced-wait pseudo-arm, not a decision
    if i >= len(ranked) or ranked[i][0] == "No Move":
        return "—"
    name, share, score = ranked[i]
    return "%s<br>%.0f%%, %.2f" % (name, share * 100, score)


def ledger_rows(battle_tag_fragment):
    if not os.path.isfile(LEDGER):
        return []
    rows = []
    for line in open(LEDGER):
        if battle_tag_fragment in line:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            # only rows carrying a prediction correspond to a search decision
            if r.get("pred"):
                rows.append(r)
    return rows


def main(argv):
    show_timing = "--timing" in argv
    args = [a for a in argv if not a.startswith("--")]
    game_dir = find_game(args[0] if args else None)
    decisions, players, tera_turns = parse(game_dir)
    if not decisions:
        raise SystemExit("no decisions parsed from %s (log level below INFO?)" % game_dir)

    meta = {}
    meta_path = os.path.join(game_dir, "meta.json")
    if os.path.isfile(meta_path):
        meta = json.load(open(meta_path))
    account = meta.get("account", "")
    opp_side = next(
        (side for side, name in players.items() if name != account), "p1"
    )
    opp_tera_turn = tera_turns.get(opp_side)
    battle_id = (meta.get("battle_tag") or os.path.basename(game_dir)).split("-")
    frag = next((p for p in battle_id if p.isdigit() and len(p) > 6), "")
    acts = ledger_rows(frag) if frag else []

    print("# %s" % os.path.basename(game_dir))
    if meta:
        print(
            "\n%s vs %s (%s) — **%s** · our elo %s · [replay](%s)"
            % (
                meta.get("account", "?"), meta.get("opponent", "?"),
                meta.get("opp_elo", "?"), meta.get("result", "?"),
                meta.get("our_elo", "?"), meta.get("replay_url", ""),
            )
        )
    print()
    head = "| Turn | Top 1 | Top 2 | Opp top 1 | Opp top 2 | Opp actual |"
    sep = "|---|---|---|---|---|---|"
    if show_timing:
        head += " Search |"
        sep += "---|"
    print(head)
    print(sep)

    li, hits, hits1, scored = 0, 0, 0, 0
    for d in decisions:
        # sequential join: the ledger row for this decision must agree on BOTH
        # the turn and the move we played, else it belongs to another decision.
        # One prediction can produce SEVERAL observation rows: the opponent's
        # move, then pivot switch-ins ("switch") and faint replacements
        # ("forced_switch"), the latter sometimes stamped with the PREVIOUS
        # turn number. A strict 1:1 join deadlocks on any of them and blanks
        # the whole rest of the game, so: skip stale rows from earlier turns,
        # join the first same-turn same-choice row, then consume its
        # follow-up switch rows as "⇒ target".
        def _tn(r):
            t = r.get("turn")
            return t if isinstance(t, int) else -1

        while li < len(acts) and _tn(acts[li]) < d["turn"]:
            li += 1
        actual, actual_event = "—", None
        if (
            li < len(acts)
            and _tn(acts[li]) == d["turn"]
            and acts[li].get("our_choice") == d.get("choice")
        ):
            actual = acts[li].get("actual") or "—"
            actual_event = acts[li].get("event")
            li += 1
            while (
                li < len(acts)
                and acts[li].get("event") in ("switch", "forced_switch")
                and _tn(acts[li]) == d["turn"]
                and acts[li].get("our_choice") == d.get("choice")
            ):
                follow = acts[li].get("actual") or ""
                if follow.startswith("switch "):
                    actual += " ⇒ " + follow.split(" ", 1)[1]
                li += 1
        # the ledger's observed action lacks the tera annotation; the
        # protocol's |-terastallize| event carries it. Without this, turn 18
        # of 2665372016 showed "thunderbolt" while the model's top-1 was
        # thunderbolt-tera — a perfect call scored as a miss.
        if (
            opp_tera_turn is not None
            and d["turn"] == opp_tera_turn
            and actual not in ("—",)
            and not actual.startswith("switch ")
            and "-tera" not in actual
        ):
            actual = actual.replace(" ⇒", "-tera ⇒") if " ⇒" in actual else actual + "-tera"
        ours = pool(d["ws"], d["chance"])
        theirs = pool(d["ows"], d["chance"])
        names = [t[0] for t in theirs]
        shown = actual
        if actual_event == "move" and names and names[0] != "No Move":
            scored += 1
            if actual in names:
                hits += 1
                shown = "**%s**" % actual
                if actual == names[0]:
                    hits1 += 1
        c_top1, c_top2 = cell(ours, 0), cell(ours, 1)
        gate = d.get("gate")
        if gate:
            verdict, arm = gate
            tag = " ⊘gate" if verdict == "BLOCKED" else " ✓gate"
            if ours and ours[0][0] == arm:
                c_top1 = c_top1.replace("<br>", tag + "<br>", 1)
            elif len(ours) > 1 and ours[1][0] == arm:
                c_top2 = c_top2.replace("<br>", tag + "<br>", 1)
            if verdict == "BLOCKED" and d.get("choice") and (
                not ours or d["choice"] != ours[0][0]
            ):
                c_top1 += "<br>→ played **%s**" % d["choice"]
        row = "| %s | %s | %s | %s | %s | %s |" % (
            d["turn"], c_top1, c_top2,
            cell(theirs, 0), cell(theirs, 1), shown,
        )
        if show_timing:
            row += " %dms/%d |" % (d["elapsed_ms"], d["budget_ms"] * d["waves"])
        print(row)

    print()
    print(
        "%d decisions · opponent model: top-1 %d/%d (%.0f%%), top-2 %d/%d (%.0f%%)"
        % (
            len(decisions), hits1, scored, 100.0 * hits1 / max(scored, 1),
            hits, scored, 100.0 * hits / max(scored, 1),
        )
    )
    if show_timing:
        tot = sum(d["elapsed_ms"] for d in decisions)
        print(
            "search: total %.1fs, mean %.2fs, worlds %s"
            % (tot / 1000.0, tot / 1000.0 / len(decisions),
               "/".join(str(x) for x in sorted({d["worlds"] for d in decisions})))
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
