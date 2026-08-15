# ladder-games — real ladder game archive

**DO NOT DELETE OR WIPE THIS DIRECTORY.** It is the permanent, append-only record
of real ladder games, started 2026-08-08. Games played before this archive existed
are deliberately not backfilled. No cleanup script may touch this tree.

## Structure

```
ladder-games/
  README.md            this file
  run_game.sh          plays ONE ranked game with the canonical config + archives it
  archive_game.py      called by run_game.sh; copies logs, fetches replay, indexes
  index.jsonl          one JSON line per game, append-only
  LOSSES.md            loss register; Sally adds a one-line assessment per loss
  games/
    <YYYY-MM-DD>_<battle-id>_<opponent>_<W|L>/
      meta.json        tag, opponent, result, timestamps, run flags
      replay.json      omniscient server replay (exact HP both sides, Elos)
      worlds.jsonl     one row per sampled world per decision:
                       {decision, world, chance, state} — `state` is the full
                       engine StateString, re-runnable at any budget/net
      search.log       per-decision search internals, [d N]-prefixed:
                       Budget, World cap, Policy/WorldStats/OppWorldStats,
                       PairTable, ARGMAX/Tera gate lines, TurnTiming, Choice
      protocol.log     raw server websocket messages (includes |request| JSON)
      battle.log.gz    the complete raw foul-play log, gzipped (source of
                       truth; everything above is derived from it)
```

## Usage

```bash
bash /Users/sallyliu/pokemon-fast-bot/ladder-games/run_game.sh
```

Plays exactly one gen9randombattleblitz ladder game as "fable foul play" with the
current champion (valuenet_v6ref_nopuct) and the tera gate active
(per-mon 0.001, visit frac 1/3, +0.002 opp-tera-held), then archives it here.

## Reviewing a game

```bash
python3 /Users/sallyliu/pokemon-fast-bot/ladder-games/analysis/turn_table.py --timing
```

One markdown row per DECISION: our top 2 arms and the opponent model's top 2,
each as `move` / `share%, score`, plus the move the opponent actually made
(bolded when the model had it in its top 2) and a top-1/top-2 accuracy line.
`--timing` adds per-decision search wall clock vs budget. With no argument it
takes the most recent archived game; otherwise pass a battle id, a dir name, or
any substring of one. Under a v8 net the score column is absolute P(win).

## Process (agreed 2026-08-08)

One game at a time. Sally personally assesses each loss and records a one-line
improvement thought in LOSSES.md.
