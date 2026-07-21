"""
STAGE 2 -- ROSTERING (assign pairings to named captains, one base) [LEAN]
Assigns each Stage-1 pairing to exactly one qualified home-based captain.

MODEL
  x[p,c]=1 if captain c flies pairing p (binary, only where eligible)
  un[p] =1 if pairing p left unassigned (costed slack -> always feasible)
HARD
  (A) cover : sum_c x[p,c] + un[p] == 1
  (B) rest  : a captain's pairings may not overlap (span + 12h rest)
  (C) avail : hard-off (KUR/U/SU) on any day a pairing touches -> x excluded
SOFT (minimise)
  (S1) off-claims balance |days_off - target|   [README KPI]
  (S2) off-request (O_*) violations
  (S3) unassigned pairings (big penalty)
"""
import os, time, json
from collections import defaultdict
import pandas as pd
from pyscipopt import Model, quicksum

DATA_DIR = r"C:\Users\Mukhil\OneDrive\Desktop\python\jupyter\Airline_scheduling"
HARD_OFF_CODES = {'KUR', 'U', 'SU'}
SOFT_OFF_CODES = {'O_L', 'O_M', 'O_TX', 'O_TZ', 'O_U'}
W_CLAIM, W_REQ, BIG_UNASSIGNED = 10.0, 5.0, 1.0e6
# A reserve captain absorbing a gap costs MORE than a normal line assignment (0)
# but FAR less than leaving the pairing uncovered. So the solver prefers line
# captains, then reserves, then -- only if reserves are exhausted -- uncovered.
RESERVE_COVER_COST = 1.0e3
MIN_REST_BETWEEN_PAIRINGS = 720
DAYS_IN_MONTH = 31          # May; used to pro-rate the monthly standby figure


def load_standby(base, start, end):
    """
    standby.csv gives MONTHLY reserve-shift counts per base (e.g. DUS May = 344).
    Each shift is 12h. We pro-rate to the rostering window:
        shifts_in_window = round(monthly / DAYS_IN_MONTH * window_days)
    Returns an integer reserve-shift budget for the whole window.
    """
    p = os.path.join(DATA_DIR, "standby.csv")
    if not os.path.exists(p):
        return 0
    df = pd.read_csv(p, sep=';')
    df['base'] = df['base'].astype(str).str.strip().str.upper()
    row = df[df['base'] == base]
    if row.empty:
        return 0
    monthly = float(row.iloc[0]['MAY'])       # window is in May
    window_days = (pd.to_datetime(end).date() - pd.to_datetime(start).date()).days + 1
    return int(round(monthly / DAYS_IN_MONTH * window_days))

def load_captains(base):
    hb = pd.read_csv(os.path.join(DATA_DIR, "home_bases.csv"))
    hb['home_base'] = hb['home_base'].astype(str).str.strip().str.upper()
    return sorted(hb[hb['home_base'] == base]['captain_id'].astype(int).tolist())

def load_off_requests(start):
    df = pd.read_csv(os.path.join(DATA_DIR, "off_requests_202505.csv"), sep=';')
    df['bd'] = pd.to_datetime(df['Begin Date'], format='%d.%m.%y %H:%M')
    df['ed'] = pd.to_datetime(df['End Date'], format='%d.%m.%y %H:%M')
    b0 = pd.to_datetime(start); out = defaultdict(list)
    for r in df.itertuples():
        out[int(r.captain_id)].append((r.Code,
            (r.bd-b0).total_seconds()/60, (r.ed-b0).total_seconds()/60))
    return out

def load_off_claims(base_caps):
    df = pd.read_csv(os.path.join(DATA_DIR, "off_claims_202505.csv"), sep=';')
    df.columns = [c.strip() for c in df.columns]; s = set(base_caps)
    return {int(r.captain_id): int(r.Count) for r in df.itertuples() if int(r.captain_id) in s}

def _day(mn): return int(mn // 1440)

def solve_rostering(pairings, base, start, end, time_limit=300, verbose=True):
    horizon = (pd.to_datetime(end).date() - pd.to_datetime(start).date()).days + 1
    days = list(range(horizon))
    caps = load_captains(base); offreq = load_off_requests(start); claims = load_off_claims(caps)
    hard_off, soft_off = defaultdict(set), defaultdict(set)
    for cid, reqs in offreq.items():
        for code, b, e in reqs:
            for d in range(_day(b), _day(max(b, e-1))+1):
                if 0 <= d < horizon:
                    (hard_off if code in HARD_OFF_CODES else soft_off)[cid].add(d)
    p_days = {p['pairing_id']: set(range(_day(p['start_min']), _day(p['end_min'])+1)) for p in pairings}
    p_span = {p['pairing_id']: (p['start_min'], p['end_min']) for p in pairings}
    P = [p['pairing_id'] for p in pairings]

    m = Model(f"roster_{base}")
    if not verbose: m.hideOutput()
    x = {}
    for pid in P:
        for c in caps:
            if not (p_days[pid] & hard_off[c]):
                x[pid, c] = m.addVar(vtype='B', name=f"x_{pid}_{c}")
    by_cap = defaultdict(list)
    for (pid, c) in x: by_cap[c].append(pid)

    reserve_budget = load_standby(base, start, end)

    un = {}
    res = {}                       # res[pid] = 1 if pairing pid covered by a reserve
    for pid in P:
        un[pid]  = m.addVar(vtype='B', name=f"un_{pid}")
        res[pid] = m.addVar(vtype='B', name=f"res_{pid}")
        # each pairing: one line captain, OR a reserve, OR uncovered
        m.addCons(quicksum(x[pid, c] for c in caps if (pid, c) in x)
                  + res[pid] + un[pid] == 1, name=f"cov_{pid}")

    # finite reserve capacity (pro-rated monthly standby shifts for this window)
    m.addCons(quicksum(res.values()) <= reserve_budget, name="reserve_budget")

    for c in caps:
        plist = sorted(by_cap[c], key=lambda q: p_span[q][0])
        for i in range(len(plist)):
            a = plist[i]; a_end = p_span[a][1] + MIN_REST_BETWEEN_PAIRINGS; grp = [x[a, c]]
            for j in range(i+1, len(plist)):
                b = plist[j]
                if p_span[b][0] >= a_end: break
                grp.append(x[b, c])
            if len(grp) > 1: m.addCons(quicksum(grp) <= 1, name=f"rest_{c}_{a}")

    fly = {}
    for c in caps:
        for d in days:
            tt = [x[pid, c] for pid in by_cap[c] if d in p_days[pid]]
            if tt:
                f = m.addVar(vtype='B', name=f"fly_{c}_{d}")
                m.addCons(f <= quicksum(tt))
                for t in tt: m.addCons(f >= t)
                fly[c, d] = f

    dev = []
    for c in caps:
        worked = quicksum(fly[c, d] for d in days if (c, d) in fly)
        target = min(claims.get(c, horizon), horizon)
        dp = m.addVar(vtype='C', lb=0, name=f"dp_{c}"); dn = m.addVar(vtype='C', lb=0, name=f"dn_{c}")
        m.addCons((horizon - worked) - target == dp - dn, name=f"dev_{c}")
        dev += [dp, dn]

    req = [fly[c, d] for c in caps for d in soft_off[c] if (c, d) in fly]
    m.setObjective(W_CLAIM*quicksum(dev) + W_REQ*quicksum(req)
                   + RESERVE_COVER_COST*quicksum(res.values())
                   + BIG_UNASSIGNED*quicksum(un.values()), "minimize")
    m.setRealParam("limits/time", time_limit)
    t0 = time.perf_counter(); m.optimize(); secs = time.perf_counter()-t0
    sol = m.getBestSol()

    assign = {}                    # pid -> captain_id | "RESERVE" | None
    for pid in P:
        who = next((c for c in caps if (pid, c) in x and sol[x[pid, c]] > 0.5), None)
        if who is None and sol[res[pid]] > 0.5:
            who = "RESERVE"
        assign[pid] = who
    n_reserve = sum(1 for v in assign.values() if v == "RESERVE")
    n_un = sum(1 for v in assign.values() if v is None)
    cap_rows = []
    for c in caps:
        my = [pid for pid in by_cap[c] if sol[x[pid, c]] > 0.5]
        wd = set()
        for pid in my: wd |= p_days[pid]
        cap_rows.append(dict(captain_id=c, pairings=len(my), days_worked=len(wd),
                             days_off=horizon-len(wd), off_target=min(claims.get(c, horizon), horizon),
                             soft_req_days=len(soft_off[c]), hard_off_days=len(hard_off[c])))
    meta = dict(base=base, window=f"{start}..{end}", horizon_days=horizon, captains=len(caps),
                pairings=len(P), assigned_line=len(P)-n_reserve-n_un,
                reserve_budget=reserve_budget, reserve_used=n_reserve, unassigned=n_un,
                status=m.getStatus(), objective=round(m.getObjVal(), 1), secs=round(secs, 1))
    return assign, cap_rows, meta

if __name__ == "__main__":
    d = json.load(open("dus_pairings.json")); pairings = d["pairings"]
    base = d["meta"]["base"]; start, end = d["meta"]["window"].split("..")
    assign, caps, meta = solve_rostering(pairings, base, start, end,
        time_limit=float(os.environ.get("CPP_TIME", 180)), verbose=False)
    print(json.dumps(meta, indent=2))
