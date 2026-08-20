/* Reference corpus of REAL Pokemon Showdown gen9randombattle teams.
 *
 *   node gen_ps_reference.js [N_TEAMS] > ps_reference.json
 *
 * The authority for audit_team_legality.py. Deliberately the JS generator and
 * not fp/search/ps_teams.py: several open sampling findings are defects IN
 * that Python port, so grading our sampler against it would hide exactly the
 * bugs this pass exists to find.
 *
 * Emits, per speciesId: the levels PS ever assigns, and every complete set
 * signature it ever produces (sorted moves | item | ability | tera) with
 * counts. Plus TEAM-level distributions -- hazard setters, hazard removers,
 * screens, duplicate species -- which is what finding #13 (worlds containing
 * teams PS can never build) and #16 (hazard roles under-produced) need.
 */
const PS = process.env.PS_ROOT || '/Users/sallyliu/pokemon-fast-bot/pokemon-showdown';
const {Teams} = require(PS + '/dist/sim/teams.js');
const {Dex} = require(PS + '/dist/sim/dex.js');
const N = parseInt(process.argv[2] || '20000', 10);

const norm = s => (s || '').toLowerCase().replace(/[^a-z0-9]/g, '');
// Tracked PER MOVE, not bucketed: finding #13's claim is about STEALTH ROCK
// specifically ("P(>=2 Stealth Rock setters) = 0.000 in real PS teams"), and a
// coarse "hazard setter" bucket cannot test it -- PS happily builds a team with
// Stealth Rock on one mon and Spikes on another.
const TRACKED = ['stealthrock', 'spikes', 'toxicspikes', 'stickyweb',
                 'rapidspin', 'defog', 'mortalspin', 'tidyup',
                 'lightscreen', 'reflect', 'auroraveil'];

const species = Object.create(null);
const teamStats = {n: N, dupSpecies: 0, perMove: Object.create(null)};
for (const mv of TRACKED) teamStats.perMove[mv] = Object.create(null);
const bump = (o, k) => { o[k] = (o[k] || 0) + 1; };

for (let i = 0; i < N; i++) {
  const team = Teams.generate('gen9randombattle');
  const seen = new Set();
  const cnt = Object.create(null);
  for (const mv of TRACKED) cnt[mv] = 0;
  for (const m of team) {
    const id = norm(m.speciesId || m.species);
    if (seen.has(id)) teamStats.dupSpecies++;
    seen.add(id);
    const moves = m.moves.map(norm).sort();
    const sig = [moves.join(','), norm(m.item) || 'none',
                 norm(m.ability), norm(m.teraType)].join('|');
    let rec = species[id];
    if (!rec) rec = species[id] = {n: 0, levels: {}, sets: {}};
    rec.n++;
    bump(rec.levels, m.level);
    bump(rec.sets, sig);
    for (const mv of moves) if (mv in cnt) cnt[mv]++;
  }
  for (const mv of TRACKED) bump(teamStats.perMove[mv], cnt[mv]);
}
// COSMETIC formes only (Gastrodon-East, Maushold-Four, Vivillon-*): same
// stats/types/moves as the base, so they must collapse to it for pool
// membership and for Species Clause. Battle formes (Zacian-Crowned,
// Ogerpon-*) are deliberately NOT aliased -- they have different stats, and
// collapsing them is precisely the defect this audit is hunting.
const cosmetic = Object.create(null);
const sameShape = (a, b) => {
  const sa = a.baseStats, sb = b.baseStats;
  return ['hp','atk','def','spa','spd','spe'].every(k => sa[k] === sb[k]) &&
         (a.types || []).join(',') === (b.types || []).join(',');
};
for (const sp of Dex.species.all()) {
  for (const cf of (sp.cosmeticFormes || [])) cosmetic[norm(cf)] = norm(sp.id);
  // A forme whose base stats AND types match its baseSpecies is cosmetic IN
  // EFFECT (Maushold-Four, Dudunsparce-Three-Segment) even when PS does not
  // list it under cosmeticFormes -- the engine cannot tell them apart, so
  // treating them as distinct species would flag legal teams. Zacian-Crowned
  // and friends DIFFER in stats, so they stay separate: collapsing those is
  // the very bug being measured.
  if (!sp.baseSpecies || sp.baseSpecies === sp.name) continue;
  const base = Dex.species.get(sp.baseSpecies);
  if (base && base.exists && sameShape(sp, base)) cosmetic[norm(sp.id)] = norm(base.id);
}
process.stdout.write(JSON.stringify({teams: N, species, teamStats, cosmetic}));
