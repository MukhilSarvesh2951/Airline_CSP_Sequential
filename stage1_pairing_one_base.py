"""

STAGE 1 -PAIRING  (single base, branch-and-price)

Generates a proven-optimal set of crew pairings for ONE base over a date window.
This is the input to Stage 2 (rostering), which assigns pairings to named captains.

"""
import os, time, heapq
from collections import defaultdict
import numpy as np, pandas as pd, networkx as nx
from pyscipopt import Model, Pricer, Conshdlr, Branchrule, SCIP_RESULT, SCIP_PARAMSETTING

# EASA / ops  
MCT, MIN_REST, CHECK_IN, CHECK_OUT, SECURITY_BUFFER = 45, 720, 70, 30, 60
MAX_SIT_TIME, MAX_LAYOVER_TIME, MAX_LEGS_PER_DUTY, MAX_DUTIES_PER_PAIRING = 360, 2160, 10, 3
LAYOVER_BASE_COST, TRANSFER_FIX_COST, TRANSFER_VAR_COST = 120.0, 45.0, 0.40
PAIRING_FIX_COST, COST_PER_FLIGHT_MINUTE, ARTIFICIAL_COST = 250.0, 1.0, 1.0e5
AIRCRAFT_FAMILY = ['319', '320', '321']

MAX_LABELS_PER_NODE, COLS_PER_BASE, COLUMNS_PER_ROUND, RC_EPS = 20, 40, 150, -1e-6
# Same column pool management fix as the multi-base solver :
# a column at ~0 LP value for this many consecutive rounds gets permanently
# fixed to ub=0. Safe (not just fast): if a path is ever attractive again under
# later duals, the pricer just regenerates it as a new column. This matters
# less here than in the multi-base file since a single base's reachable-leg
# network is much smaller, but it costs nothing to have and protects you if you
# later push a single base out to a full week.
STALE_ROUNDS_LIMIT = int(os.environ.get("CPP_STALE_LIMIT", 30))
DATA_DIR = r"C:\Users\Mukhil\OneDrive\Desktop\python\jupyter\Airline_scheduling"


# data loaders (window-parameterised) 
def load_fdt_limits(path):
    df = pd.read_csv(path)
    sc = [c for c in df.columns if str(c).strip().isdigit()]
    out = []
    for _, r in df.iterrows():
        sh = pd.to_timedelta(str(r['flight_duty_time_start_time'])).total_seconds()/3600
        eh = pd.to_timedelta(str(r['flight_duty_time_end_time'])).total_seconds()/3600
        out.append(((sh, eh), {int(c): int(float(r[c])*60) for c in sc}))
    return out

def build_fdt_lookup(limits):
    t = np.full((1440, MAX_LEGS_PER_DUTY+1), 720, dtype=np.int32)
    for m in range(1440):
        h = m/60
        for (sh, eh), sm in limits:
            if (h >= sh or h <= eh) if sh > eh else (sh <= h <= eh):
                for n in range(1, MAX_LEGS_PER_DUTY+1):
                    t[m, n] = sm.get(n, sm[max(sm)])
                break
    return t

def load_flights(path, start, end):
    df = pd.read_csv(path)
    df['SCHEDULED_DEPARTURE_TIME'] = pd.to_datetime(df['SCHEDULED_DEPARTURE_TIME'])
    df['SCHEDULED_ARRIVAL_TIME']   = pd.to_datetime(df['SCHEDULED_ARRIVAL_TIME'])
    s, e = pd.to_datetime(start).date(), pd.to_datetime(end).date()
    df = df[(df['SCHEDULED_DEPARTURE_TIME'].dt.date >= s) &
            (df['SCHEDULED_DEPARTURE_TIME'].dt.date <= e)].copy()
    df['AIRCRAFT_TYPE'] = df['AIRCRAFT_TYPE'].astype(str)
    df = df[df['AIRCRAFT_TYPE'].isin(AIRCRAFT_FAMILY)].copy()
    base0 = pd.to_datetime(start)
    df['DEP_MIN'] = (df['SCHEDULED_DEPARTURE_TIME']-base0).dt.total_seconds()/60
    df['ARR_MIN'] = (df['SCHEDULED_ARRIVAL_TIME']-base0).dt.total_seconds()/60
    df['BLOCK'] = df['ARR_MIN']-df['DEP_MIN']
    df = df[df['BLOCK'] > 0].sort_values('DEP_MIN').reset_index(drop=True)
    df['LEG_ID'] = df['LEG_ID'].astype(str)
    return df

def load_ground(path):
    df = pd.read_csv(path); gt = {}
    for _, r in df.iterrows():
        a, b = str(r['Dep Ap']).strip().upper(), str(r['Arr Ap']).strip().upper()
        t = int(float(r['avg_duration'])*60); gt[(a, b)] = t; gt[(b, a)] = t
    return gt

#This method builds a directed graph of the flight legs and their connections based on the given constraints and parameters.
# It takes in a dataframe of flight legs, a ground transfer time dictionary, a base airport code, and a list of allowed flight legs. 
# The method creates nodes for each flight leg and adds edges between them based on the layover and transfer constraints. 
# It also adds source and sink nodes for the base airport and connects them to the appropriate flight legs. The resulting directed graph is returned.
def build_network(df, gt, base, allowed):
    sub = df[df['LEG_ID'].isin(allowed)]
    G = nx.DiGraph()
    for r in sub.itertuples():
        G.add_node(r.LEG_ID, kind='FLIGHT', origin=r.DEPARTURE_AIRPORT, dest=r.ARRIVAL_AIRPORT,
                   dep=float(r.DEP_MIN), arr=float(r.ARR_MIN), block=float(r.BLOCK),
                   cost=COST_PER_FLIGHT_MINUTE*float(r.BLOCK))
    G.add_node(f"SRC_{base}", kind='SOURCE', base=base)
    G.add_node(f"SNK_{base}", kind='SINK', base=base)
    legs = [r.LEG_ID for r in sub.itertuples()]
    dep, arr = sub['DEP_MIN'].to_numpy(), sub['ARR_MIN'].to_numpy()
    for i in range(len(legs)):
        u, ai = legs[i], arr[i]; du = G.nodes[u]['dest']
        hi = np.searchsorted(dep, ai+MAX_LAYOVER_TIME, side='right')
        for j in range(i+1, hi):
            v = legs[j]; gap = dep[j]-ai
            if gap < MCT: continue
            ov = G.nodes[v]['origin']
            if du == ov:
                if gap <= MAX_SIT_TIME:
                    G.add_edge(u, v, type='TURN', gap=gap, cost=0.0, rest=False)
                elif MIN_REST <= gap <= MAX_LAYOVER_TIME:
                    G.add_edge(u, v, type='LAYOVER', gap=gap, cost=LAYOVER_BASE_COST, rest=True)
            else:
                t = gt.get((du, ov))
                if t is None: continue
                need = t+SECURITY_BUFFER; tc = TRANSFER_FIX_COST+TRANSFER_VAR_COST*t
                if need <= gap <= MAX_SIT_TIME:
                    G.add_edge(u, v, type='TRANSFER', gap=gap, cost=tc, rest=False, transit=t)
                elif max(MIN_REST, need) <= gap <= MAX_LAYOVER_TIME:
                    G.add_edge(u, v, type='LAYOVER_TRANSFER', gap=gap,
                               cost=LAYOVER_BASE_COST+tc, rest=True, transit=t)
    for lg in legs:
        o, d = G.nodes[lg]['origin'], G.nodes[lg]['dest']
        if o == base: G.add_edge(f"SRC_{base}", lg, type='START_DUTY', cost=0.0, rest=False)
        elif (base, o) in gt:
            t = gt[(base, o)]
            G.add_edge(f"SRC_{base}", lg, type='START_DUTY_TRANSFER',
                       cost=TRANSFER_FIX_COST+TRANSFER_VAR_COST*t, rest=False, transit=t)
        if d == base: G.add_edge(lg, f"SNK_{base}", type='END_DUTY', cost=0.0, rest=False)
        elif (d, base) in gt:
            t = gt[(d, base)]
            G.add_edge(lg, f"SNK_{base}", type='END_DUTY_TRANSFER',
                       cost=TRANSFER_FIX_COST+TRANSFER_VAR_COST*t, rest=False, transit=t)
    return G
#This function takes a directed graph of flight legs and their connections, a base airport code, and a list of flight legs, 
# and returns the set of flight legs that are reachable from the source node to the sink node in the graph.
def reachable_from(G, base, legs):
    src, snk = f"SRC_{base}", f"SNK_{base}"
    if src not in G or snk not in G: return set()
    return (nx.descendants(G, src) & nx.ancestors(G, snk)) & set(legs)

# Label class for storing information about each label in the branch-and-price algorithm
class Label:
    __slots__ = ('rc','cost','node','duty_legs','n_duties','fdt_start','visited','parent')
    def __init__(s, rc, cost, node, dl, nd, fs, vis, par):
        s.rc, s.cost, s.node, s.duty_legs, s.n_duties, s.fdt_start, s.visited, s.parent = \
            rc, cost, node, dl, nd, fs, vis, par
    def __lt__(s, o): return s.rc < o.rc

# RCSPP class for solving the resource-constrained shortest path problem using a label-setting algorithm.
class RCSPP:
    def __init__(s, G, base, fdt, conshdlr=None):
        s.G, s.base, s.fdt, s.conshdlr = G, base, fdt, conshdlr
    def _fdt_ok(s, fs, arr, n):
        if n > MAX_LEGS_PER_DUTY: return False
        return (arr+CHECK_OUT)-fs <= s.fdt[int(fs) % 1440, n]
    def solve(s, duals, max_cols=COLUMNS_PER_ROUND):
        found = s._search(duals, COLS_PER_BASE*4); found.sort(key=lambda x: x[3])
        picked, hit = [], set()
        for it in found:
            if len(picked) >= max_cols: break
            if not set(it[0]) & hit: picked.append(it); hit.update(it[0])
        for it in found:
            if len(picked) >= max_cols: break
            if it not in picked: picked.append(it)
        return picked
    def _search(s, duals, max_cols):
        G, base = s.G, s.base; src, snk = f"SRC_{base}", f"SNK_{base}"
        if src not in G: return []
        ch = s.conshdlr
        buckets = defaultdict(list); pq, res = [], []
        for _, lg, e in G.out_edges(src, data=True):
            if ch and lg in ch.enforced_in:
                continue  # lg must be immediately preceded by something else -> can't start here
            nd = G.nodes[lg]; fs = nd['dep']-CHECK_IN
            if fs < 0 or not s._fdt_ok(fs, nd['arr'], 1): continue
            cost = PAIRING_FIX_COST+e['cost']+nd['cost']; rc = cost-duals.get(lg, 0.0)
            lab = Label(rc, cost, lg, 1, 1, fs, frozenset((lg,)), None)
            heapq.heappush(pq, (rc, id(lab), lab))
        while pq:
            rc, _, lab = heapq.heappop(pq)
            if len(res) >= max_cols: break
            if G.has_edge(lab.node, snk) and not (ch and lab.node in ch.enforced_out):
                e = G.edges[lab.node, snk]; ca = G.nodes[lab.node]['arr']+e.get('transit', 0)
                if s._fdt_ok(lab.fdt_start, ca, lab.duty_legs):
                    trc = lab.rc+e['cost']
                    if trc < RC_EPS: res.append((s._path(lab), base, lab.cost+e['cost'], trc))
            enforced_out_node = ch.enforced_out.get(lab.node) if ch else None
            for _, v, e in G.out_edges(lab.node, data=True):
                if G.nodes[v]['kind'] != 'FLIGHT' or v in lab.visited: continue
                if enforced_out_node and v != enforced_out_node: continue
                if ch and ch.forbidden_edges.get((lab.node, v), 0) > 0: continue
                nv = G.nodes[v]
                if e['rest']:
                    if lab.n_duties >= MAX_DUTIES_PER_PAIRING: continue
                    ndd, dl, fs = lab.n_duties+1, 1, nv['dep']-CHECK_IN
                else:
                    if lab.duty_legs >= MAX_LEGS_PER_DUTY: continue
                    ndd, dl, fs = lab.n_duties, lab.duty_legs+1, lab.fdt_start
                if not s._fdt_ok(fs, nv['arr'], dl): continue
                nrc = lab.rc+e['cost']+nv['cost']-duals.get(v, 0.0)
                nco = lab.cost+e['cost']+nv['cost']
                bk = buckets[(v, ndd)]
                if any(ol.rc <= nrc and ol.duty_legs <= dl and ol.fdt_start >= fs for ol in bk): continue
                if len(bk) >= MAX_LABELS_PER_NODE:
                    if nrc >= bk[-1].rc: continue
                    bk.pop()
                nl = Label(nrc, nco, v, dl, ndd, fs, lab.visited | {v}, lab)
                bk.append(nl); bk.sort(key=lambda l: l.rc); heapq.heappush(pq, (nrc, id(nl), nl))
        return res
    @staticmethod
    def _path(lab):
        p = []
        while lab is not None: p.append(lab.node); lab = lab.parent
        return list(reversed(p))

class RyanFosterConshdlr(Conshdlr):
    '''
    Custom constraint handler for Ryan-Foster branching on flight legs.
    '''
    def __init__(self, cols):
        super().__init__()
        self.cols = cols
        self.forbidden_edges = defaultdict(int)
        self.enforced_out = {}
        self.enforced_in = {}
        self.leg_to_cols = defaultdict(set)
        self.edge_to_cols = defaultdict(set)
        self.var_name_to_idx = {}

    def register_column(self, idx, var, path):
        self.var_name_to_idx[var.name] = idx
        for lg in path:
            self.leg_to_cols[lg].add(idx)
        for i in range(len(path) - 1):
            self.edge_to_cols[(path[i], path[i + 1])].add(idx)

    def consactive(self, constraint):
        u, v, together = constraint.data['u'], constraint.data['v'], constraint.data['together']
        if together:
            self.enforced_out[u] = v
            self.enforced_in[v] = u
        else:
            self.forbidden_edges[(u, v)] += 1

    def consdeactive(self, constraint):
        u, v, together = constraint.data['u'], constraint.data['v'], constraint.data['together']
        if together:
            if u in self.enforced_out: del self.enforced_out[u]
            if v in self.enforced_in: del self.enforced_in[v]
        else:
            self.forbidden_edges[(u, v)] -= 1
            if self.forbidden_edges[(u, v)] <= 0: del self.forbidden_edges[(u, v)]

    def consprop(self, constraints, nusefulconss, nmarkconss, proptiming):
        result = SCIP_RESULT.DIDNOTFIND
        for cons in constraints:
            u, v, together = cons.data['u'], cons.data['v'], cons.data['together']
            candidates = (self.leg_to_cols.get(u, ()) | self.leg_to_cols.get(v, ())) if together \
                         else self.edge_to_cols.get((u, v), ())
            for idx in candidates:
                var, path, base, cost = self.cols[idx]
                if self.model.getVarUbLocal(var) <= 0.5: continue
                violates = False
                if together:
                    if u in path:
                        pos = path.index(u)
                        if pos == len(path) - 1 or path[pos + 1] != v: violates = True
                    if not violates and v in path:
                        pos = path.index(v)
                        if pos == 0 or path[pos - 1] != u: violates = True
                else:
                    violates = True
                if violates:
                    self.model.chgVarUb(var, 0.0)
                    result = SCIP_RESULT.REDUCEDDOM
        return {"result": result}

    def conscheck(self, constraints, solution, checkintegrality, checklprows, printreason, completely):
        return {"result": SCIP_RESULT.FEASIBLE}
    def consenfolp(self, constraints, nusefulconss, solinfeasible):
        return {"result": SCIP_RESULT.FEASIBLE}
    def consenfops(self, constraints, nusefulconss, solinfeasible, objinfeasible):
        return {"result": SCIP_RESULT.FEASIBLE}
    def conslock(self, constraint, locktype, nlockspos, nlocksneg):
        pass


class RyanFosterBranchrule(Branchrule):
    def __init__(self, cols, conshdlr):
        super().__init__()
        self.cols = cols
        self.conshdlr = conshdlr
        self.n_branches = 0

    def branchexeclp(self, allowaddcons):
        lpcands, lpcandscols, frac, ncands, npriocands, nfracimplvars = self.model.getLPBranchCands()
        if not lpcands:
            return {"result": SCIP_RESULT.DIDNOTRUN}
        edge_flow = defaultdict(float)
        for var, f in zip(lpcands, frac):
            idx = self.conshdlr.var_name_to_idx.get(var.name)
            if idx is None: continue
            path = self.cols[idx][1]
            for i in range(len(path) - 1):
                edge_flow[(path[i], path[i + 1])] += f
        best_edge, best_diff = None, 1.0
        for edge, flow in edge_flow.items():
            if 1e-6 < flow < 1 - 1e-6:
                diff = abs(flow - 0.5)
                if diff < best_diff:
                    best_diff, best_edge = diff, edge
        if best_edge is None:
            return {"result": SCIP_RESULT.DIDNOTRUN}
        u, v = best_edge
        node0 = self.model.createChild(0.0, self.model.getLocalEstimate())
        node1 = self.model.createChild(0.0, self.model.getLocalEstimate())
        cons0 = self.model.createCons(self.conshdlr, f"sep_{u}_{v}", initial=True, separate=False,
                                       enforce=False, check=False, propagate=True, local=True,
                                       modifiable=False, dynamic=False, removable=True, stickingatnode=False)
        cons0.data = {'u': u, 'v': v, 'together': False}
        self.model.addConsNode(node0, cons0)
        cons1 = self.model.createCons(self.conshdlr, f"tog_{u}_{v}", initial=True, separate=False,
                                       enforce=False, check=False, propagate=True, local=True,
                                       modifiable=False, dynamic=False, removable=True, stickingatnode=False)
        cons1.data = {'u': u, 'v': v, 'together': True}
        self.model.addConsNode(node1, cons1)
        self.n_branches += 1
        return {"result": SCIP_RESULT.BRANCHED}

# CrewPricer class for pricing new columns in the branch-and-price algorithm.
class CrewPricer(Pricer):
    def __init__(s, rcspp, cover, cols, verbose=False):
        super().__init__()
        s.rcspp, s.cover_cons, s.cols = rcspp, cover, cols
        s.rounds = s.total_added = 0
        s.converged = False
        s.verbose = verbose
        # column pool management (see STALE_ROUNDS_LIMIT)
        s.active_idx = list(range(len(cols)))
        s.zero_streak = {}
        s.pruned = 0
        s._pruning_enabled = True

    def pricerinit(s):
        for lg in s.cover_cons: s.cover_cons[lg] = s.model.getTransformedCons(s.cover_cons[lg])

    def _prune_stale_columns(s):
        if not s._pruning_enabled or s.model.getNNodes() > 1:
            return
        still_active = []
        for idx in s.active_idx:
            v = s.cols[idx][0]
            try:
                val = s.model.getVal(v)
            except Exception:
                s._pruning_enabled = False
                return
            streak = s.zero_streak.get(idx, 0) + 1 if val < 1e-7 else 0
            s.zero_streak[idx] = streak
            if streak >= STALE_ROUNDS_LIMIT:
                s.model.chgVarUbGlobal(v, 0.0)
                s.pruned += 1
                s.zero_streak.pop(idx, None)
            else:
                still_active.append(idx)
        s.active_idx = still_active

    def pricerredcost(s):
        s.rounds += 1
        s._prune_stale_columns()
        duals = {lg: s.model.getDualsolLinear(c) for lg, c in s.cover_cons.items()}
        new = s.rcspp.solve(duals)
        if not new:
            s.converged = True
            return {'result': SCIP_RESULT.SUCCESS}
        for path, base, cost, rc in new:
            idx = len(s.cols)
            v = s.model.addVar(vtype='B', lb=0.0, ub=1.0, obj=cost, pricedVar=True, name=f"p{idx}")
            for lg in path: s.model.addConsCoeff(s.cover_cons[lg], v, 1.0)
            s.cols.append((v, path, base, cost))
            s.active_idx.append(idx)
            if s.rcspp.conshdlr is not None:
                s.rcspp.conshdlr.register_column(idx, v, path)
        s.total_added += len(new)
        return {'result': SCIP_RESULT.SUCCESS}

    def pricerfarkas(s):
        # artificial columns guarantee feasibility; RyanFosterConshdlr never
        # touches art_* variables, so this stays a safe no-op (same reasoning
        # as crew_pairing_bp_scip_pooled.py)
        return {'result': SCIP_RESULT.SUCCESS}

# This function is responsible for solving the crew pairing problem for a given base airport and date window. It takes in the base airport code, start and end dates, time limit for optimization. 
# The function loads flight data, builds a directed graph of flight legs and their connections, and sets up a branch-and-price model using SCIP. 
# It generates pairings by solving the resource-constrained shortest path problem and returns the resulting pairings along with metadata about the solution.
def solve_pairings(base, start, end, time_limit=600, verbose=True):
    """Returns (pairings, meta). pairings = [{pairing_id, legs, cost, ...}]."""
    fdt = build_fdt_lookup(load_fdt_limits(os.path.join(DATA_DIR, "fdt_limits.csv")))
    df  = load_flights(os.path.join(DATA_DIR, "flight_schedule.csv"), start, end)
    gt  = load_ground(os.path.join(DATA_DIR, "ground_transportation_times.csv"))
    legs = [r.LEG_ID for r in df.itertuples()]

    # build the directed graph of flight legs and their connections
    G = build_network(df, gt, base, set(legs))
    reach = sorted(reachable_from(G, base, legs))
    if verbose:
        print(f"  [pairing] {base} {start}..{end}: {len(df)} legs total, "
              f"{len(reach)} reachable from {base}")

    # set up the branch-and-price model
    m = Model(f"pair_{base}"); m.hideOutput()
    m.setPresolve(SCIP_PARAMSETTING.OFF); m.setSeparating(SCIP_PARAMSETTING.OFF)
    m.setIntParam("presolving/maxrestarts", 0)
    cols, cover = [], {}

    # add artificial columns to guarantee feasibility of the LP relaxation
    for lg in reach:
        # vtype='B' (not 'C'): during LP relaxation SCIP still treats this as
        # continuous in [0,1], so column generation behaves identically; the
        # difference only shows up once SCIP needs an integral solution, which
        # is what lets Ryan-Foster branching below actually engage instead of
        # silently never firing (the bug found and fixed in the multi-base file).
        a = m.addVar(vtype='B', lb=0.0, obj=ARTIFICIAL_COST, name=f"art_{lg}")
        cover[lg] = m.addCons(a == 1.0, name=f"cover_{lg}", separate=False, modifiable=True)

    rf_conshdlr = RyanFosterConshdlr(cols)
    m.includeConshdlr(rf_conshdlr, "RyanFoster", "Ryan-Foster constraint handler",
                      chckpriority=100000, enfopriority=100000, propfreq=1)
    rf_branchrule = RyanFosterBranchrule(cols, rf_conshdlr)
    #commented out the branch rule inclusion to avoid potential issues with SCIP's branching behavior. Uncomment if needed.
    #m.includeBranchrule(rf_branchrule, "RyanFosterBranchrule", "Ryan-Foster Branching",priority=100000, maxdepth=-1, maxbounddist=1.0)

    rcspp = RCSPP(G, base, fdt, rf_conshdlr) # rcspp.solve() returns [(path, base, cost, rc), ...]
    uncov = set(reach) # legs that still need to be covered by pairings

    #Iteration to generate pairings until all reachable legs are covered or no new pairings can be found
    for _ in range(40):
        if not uncov: break
        new = rcspp.solve({lg: ARTIFICIAL_COST for lg in uncov})
        if not new: break
        prog = False
        for path, b, cost, _ in new:
            if not (set(path) & uncov): continue
            idx = len(cols)
            v = m.addVar(vtype='B', lb=0.0, ub=1.0, obj=cost, name=f"p{idx}") # new pairing variable
            for lg in path: m.addConsCoeff(cover[lg], v, 1.0)
            cols.append((v, path, b, cost))
            rf_conshdlr.register_column(idx, v, path)
            uncov -= set(path); prog = True
        if not prog: break
    pricer = CrewPricer(rcspp, cover, cols, verbose=False) # 
    m.includePricer(pricer, "P", "rcspp")
    m.setRealParam("limits/time", time_limit)
    t0 = time.perf_counter(); m.optimize(); secs = time.perf_counter()-t0

    sol = m.getBestSol()
    chosen = [(p, c) for (v, p, b, c) in cols if sol[v] > 0.5]
    # attach absolute clock times for the rostering stage
    tmap = {r.LEG_ID: (float(r.DEP_MIN), float(r.ARR_MIN)) for r in df.itertuples()}
    pairings = []
    for i, (path, cost) in enumerate(sorted(chosen, key=lambda x: -x[1]), 1):
        dep = min(tmap[l][0] for l in path); arr = max(tmap[l][1] for l in path)
        pairings.append(dict(pairing_id=i, base=base, legs=list(path), n_legs=len(path),
                             cost=round(cost, 2), start_min=dep, end_min=arr))
    covered = set(l for p, _ in chosen for l in p)
    meta = dict(base=base, window=f"{start}..{end}", legs_total=len(df),
                legs_reachable=len(reach), legs_covered=len(covered),
                n_pairings=len(pairings), crew_cost=round(sum(c for _, c in chosen), 2),
                status=m.getStatus(), converged=pricer.converged, secs=round(secs, 1),
                pricing_rounds=pricer.rounds, columns_generated=pricer.total_added,
                columns_pruned=pricer.pruned, ryan_foster_branches=rf_branchrule.n_branches)
    if verbose:
        print(f"  [pairing] {len(pairings)} pairings, "
              f"{len(covered)}/{len(reach)} covered, EUR {meta['crew_cost']:,.0f}, "
              f"{'optimal' if pricer.converged else m.getStatus()}, {secs:.1f}s")
    return pairings, meta


if __name__ == "__main__":
    import json
    base = os.environ.get("CPP_BASE", "DUS")
    start = os.environ.get("CPP_START", "2025-05-01")
    end   = os.environ.get("CPP_END", "2025-05-03")
    pairings, meta = solve_pairings(base, start, end,
                                    time_limit=float(os.environ.get("CPP_TIME", 9000)))
    print(json.dumps(meta, indent=2))

    # NOTE: the module docstring promised "(with CPP_EXPORT=1) writes
    # pairings_<base>.csv" but this was never actually implemented before.
    # Added here: writes one row per pairing, with legs pipe-joined so it
    # stays a flat CSV rather than needing a nested/JSON column.
    if os.environ.get("CPP_EXPORT"):
        out_path = f"pairings_{base}.csv"
        rows = [{**{k: v for k, v in p.items() if k != 'legs'},
                 "legs": "|".join(p["legs"])} for p in pairings]
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"\nExported: {out_path}  ({len(rows)} pairings)")
