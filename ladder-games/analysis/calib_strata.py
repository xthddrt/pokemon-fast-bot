import json, re, os
from multiprocessing import Pool
GAMES="/Users/sallyliu/pokemon-fast-bot/ladder-games/games"
POL=re.compile(r"\[d (\d+)\] INFO\s+Policy \d+: .*avg_score=([\d.]+)")
rows=[json.loads(l) for l in open("/Users/sallyliu/pokemon-fast-bot/ladder-games/analysis/features.jsonl")]
def one(r):
    evs={}
    try:
        for l in open(os.path.join(GAMES,r["game"],"search.log"),errors="replace"):
            m=POL.match(l)
            if m: evs.setdefault(int(m.group(1)),[]).append(float(m.group(2)))
    except OSError: return None
    y=1.0 if r["result"]=="W" else 0.0
    return {"elo":r.get("opp_elo") or 0, "fin":(r.get("our_left")==0 or r.get("opp_left")==0),
            "pts":[(sum(v)/len(v),y) for v in evs.values()]}
if __name__=="__main__":
    with Pool(6) as p: gs=[g for g in p.map(one,rows) if g]
    strata=[("opp<1400",lambda g:g["elo"]<1400),("1400-1699",lambda g:1400<=g["elo"]<1700),("opp>=1700",lambda g:g["elo"]>=1700),
            ("opp>=1700 AND finished",lambda g:g["elo"]>=1700 and g["fin"])]
    B=[(0,.2),(.2,.3),(.3,.4),(.4,.5),(.5,.6),(.6,.7),(.7,1.01)]
    hdr="bucket      " + "".join(f"{s[0]:>24s}" for s in strata)
    print(hdr)
    for lo,hi in B:
        line=f"{lo:.1f}-{hi:.1f}     "
        for _,f in strata:
            b=[y for g in gs if f(g) for e,y in g["pts"] if lo<=e<hi]
            line += f"{(f'{sum(b)/len(b):.2f} (n={len(b)})' if b else '-'):>24s}"
        print(line)
    for name,f in strata:
        n=sum(1 for g in gs if f(g)); print(f"{name}: {n} games")
