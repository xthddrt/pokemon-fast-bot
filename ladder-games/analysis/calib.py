import json, re, os, random
from multiprocessing import Pool
GAMES="/Users/sallyliu/pokemon-fast-bot/ladder-games/games"
POL=re.compile(r"\[d (\d+)\] INFO\s+Policy \d+: .*avg_score=([\d.]+)")
rows=[json.loads(l) for l in open("/Users/sallyliu/pokemon-fast-bot/ladder-games/analysis/features.jsonl")]
random.seed(1); samp=random.sample(rows,300)
def one(r):
    evs={}
    try:
        for l in open(os.path.join(GAMES,r["game"],"search.log"),errors="replace"):
            m=POL.match(l)
            if m: evs.setdefault(int(m.group(1)),[]).append(float(m.group(2)))
    except OSError: return []
    y=1.0 if r["result"]=="W" else 0.0
    n=max(evs)+1 if evs else 1
    return [(sum(v)/len(v),y,d/n) for d,v in evs.items()]
if __name__=="__main__":
    with Pool(6) as p: pts=[x for sub in p.map(one,samp) for x in sub]
    print(f"calibration over {len(pts)} decisions (300 games): predicted -> realized")
    for lo,hi in [(0,.2),(.2,.3),(.3,.4),(.4,.5),(.5,.6),(.6,.7),(.7,.8),(.8,1.01)]:
        b=[y for e,y,_ in pts if lo<=e<hi]
        if b: print(f"  eval {lo:.1f}-{hi:.1f}: n={len(b):5d}  realized {sum(b)/len(b):.2f}")
    print("\nmid-game only (20-70% through game):")
    for lo,hi in [(0,.2),(.2,.3),(.3,.4),(.4,.5),(.5,.6),(.6,.7),(.7,.8),(.8,1.01)]:
        b=[y for e,y,f in pts if lo<=e<hi and .2<=f<=.7]
        if b: print(f"  eval {lo:.1f}-{hi:.1f}: n={len(b):5d}  realized {sum(b)/len(b):.2f}")
