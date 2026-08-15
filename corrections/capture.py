"""Capture a ruling: game + turn + ruled move -> ledger row with the recorded
worlds of that decision (HAMMER_SPEC.md Part 2, corrections/capture.py).

  .venv python capture.py <game-dir-or-name> <turn> <ruled move>
      [--note "..."] [--decision N] [--ledger PATH] [--force]

Turn -> decision via battle.log.gz (Nth 'Searching for a move using MCTS'
block = worlds.jsonl decision N; a turn can hold several decisions, e.g. a
post-KO forced switch). When ambiguous, the decision where the ruled move is
legal wins; --decision overrides.
"""

import argparse
import json
import sys

import common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("game")
    ap.add_argument("turn", type=int)
    ap.add_argument("ruled_move", nargs="+")
    ap.add_argument("--note", default="")
    ap.add_argument("--decision", type=int, default=None)
    ap.add_argument("--ledger", default=common.LEDGER)
    ap.add_argument("--force", action="store_true",
                    help="append even if the ruled move is not among the "
                         "engine's legal options for the recorded worlds")
    args = ap.parse_args()

    game_dir = common.resolve_game_dir(args.game)
    ruled = common.norm_choice(" ".join(args.ruled_move))
    decisions = common.parse_decisions(game_dir)
    worlds = common.load_worlds(game_dir)

    in_turn = [d for d in decisions if d["turn"] == args.turn]
    if not in_turn:
        raise SystemExit("no decisions found in turn %d (turns run 1..%d)"
                         % (args.turn, max(d["turn"] for d in decisions)))

    # legality per candidate decision, from the engine's own root enumeration
    legal = {}
    for d in in_turn:
        rows = worlds.get(d["decision"])
        if not rows:
            continue
        s1_opts, _ = common.root_options(rows[0]["state"])
        legal[d["decision"]] = (ruled in s1_opts, s1_opts)

    if args.decision is not None:
        pick = next((d for d in in_turn if d["decision"] == args.decision),
                    None)
        if pick is None:
            raise SystemExit("--decision %d is not in turn %d (candidates: %s)"
                             % (args.decision, args.turn,
                                [d["decision"] for d in in_turn]))
    else:
        eligible = [d for d in in_turn if legal.get(d["decision"], (False,))[0]]
        if len(eligible) == 1:
            pick = eligible[0]
        elif not eligible:
            pick = in_turn[0]
        else:
            pick = eligible[0]
            print("NOTE: %d decisions in turn %d allow %r; taking decision %d "
                  "(use --decision to override): %s"
                  % (len(eligible), args.turn, ruled, pick["decision"],
                     [(d["decision"], d["choice"]) for d in in_turn]))

    ok, s1_opts = legal.get(pick["decision"], (False, []))
    if not ok:
        msg = ("ruled move %r is not a legal option of decision %d "
               "(turn %d). Legal: %s" % (ruled, pick["decision"], args.turn,
                                         s1_opts))
        if not args.force:
            raise SystemExit(msg + "  (--force to append anyway)")
        print("WARNING: " + msg)

    rows = worlds[pick["decision"]]
    ctx = common.opp_context(rows[0]["state"])
    game_name = game_dir.rstrip("/").split("/")[-1]
    row = {
        "id": "%s_d%02d" % (game_name, pick["decision"]),
        "game": game_name,
        "turn": args.turn,
        "decision_idx": pick["decision"],
        "ruled_move": ruled,
        "played_move": pick["choice"],
        "states": [{"world": r["world"], "chance": r["chance"],
                    "state": r["state"]} for r in rows],
        "note": args.note,
        "ts": common.now_ts(),
        "net": common.game_net(game_dir),
        **ctx,
    }

    dupes = [r for r in common.read_ledger(args.ledger) if r["id"] == row["id"]]
    if dupes and not args.force:
        raise SystemExit("ledger already has %s (--force to append a second "
                         "ruling for the same decision)" % row["id"])

    common.append_ledger(row, args.ledger)
    print(json.dumps({k: row[k] for k in
                      ("id", "turn", "decision_idx", "ruled_move",
                       "played_move", "net", "opp_alive", "opp_unrevealed",
                       "opp_tera_used")}, indent=1))
    print("captured %d worlds -> %s" % (len(rows), args.ledger))
    return 0


if __name__ == "__main__":
    sys.exit(main())
