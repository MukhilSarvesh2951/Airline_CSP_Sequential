"""
AIRLINE CREW PAIRING PROBLEM  --  BRANCH-AND-PRICE  (SCIP / PySCIPOpt)
Master Problem : Set Covering  (every flight leg covered by >= 1 pairing)
Pricing Problem: Resource-Constrained Shortest Path (RCSPP) via label-setting
                 on an explicit NetworkX DiGraph of flight legs.
Solver         : SCIP 10 via PySCIPOpt  (native Pricer plugin -> Branch-Price-&-Cut)

"""

import os
import time
import heapq
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
from pyscipopt import Model, Pricer, Conshdlr, Branchrule, SCIP_RESULT, SCIP_PARAMSETTING, quicksum

# 1. OPERATIONAL PARAMETERS  (EASA + project spec)
MCT                     = 45      # Minimum Connection Time (min)
MIN_REST                = 720     # 12 h minimum rest between duties (min)
CHECK_IN                = 70      # Check-in before first departure of duty (min)
CHECK_OUT               = 30      # Check-out after last arrival of duty (min)
SECURITY_BUFFER         = 60      # Extra buffer for ground transfers (min)
MAX_SIT_TIME            = 360     # 6 h max sit within a duty (min)
MAX_LAYOVER_TIME        = 2160    # 36 h max layover (min)
MAX_LEGS_PER_DUTY       = 10      # EASA sector cap
MAX_DUTIES_PER_PAIRING  = 3       # Pairing spans at most 3 duties

LAYOVER_BASE_COST       = 120.0   # Hotel cost per layover (EUR)
TRANSFER_FIX_COST       = 45.0    # Fixed ground-transfer cost (EUR)
TRANSFER_VAR_COST       = 0.40    # EUR per minute of ground transit
PAIRING_FIX_COST        = 250.0   # Cost of opening one pairing (crew duty pay)
COST_PER_FLIGHT_MINUTE  = 1.0     # Block-hour proxy cost

ARTIFICIAL_COST         = 1.0e5   # Big-M for artificial (slack) columns

START_DATE = os.environ.get("CPP_START", '2025-04-27')
END_DATE   = os.environ.get("CPP_END", '2025-04-28')
# Ryan-Foster branching toggle (registry-based encoding; see RyanFosterConshdlr)
RYAN_FOSTER = os.environ.get("CPP_RYAN_FOSTER", "1") != "0"
AIRCRAFT_FAMILY = ['319', '320', '321']   # A320 family only

# Column generation controls 
MAX_LABELS_PER_NODE   = int(os.environ.get("CPP_MAX_LABELS", 30))
COLS_PER_BASE         = int(os.environ.get("CPP_COLS_PER_BASE", 60))
COLUMNS_PER_ROUND     = int(os.environ.get("CPP_COLS_PER_ROUND", 400))
RC_EPS                = -1e-6  # reduced-cost tolerance
# Column pool management (ported from crew_pairing_bp_scip_pooled.py): a column
# at ~0 LP value for this many consecutive rounds gets its ub fixed to 0. Needed
# here because with vtype='B' the run now enters branch-and-price (many nodes),
# and without pruning the 50k+ column pool that made late rounds crawl at ~37s
# each would make the B&B tree impractical. Safe: a pruned path is regenerated
# by the pricer if it becomes attractive again.
STALE_ROUNDS_LIMIT    = int(os.environ.get("CPP_STALE_LIMIT", 30))

DATA_DIR = os.environ.get("CPP_DATA_DIR",
    r"C:\Users\Mukhil\OneDrive\Desktop\python\jupyter\Airline_scheduling")



# 2. DATA LOADING
def load_fdt_limits(path):
    """
    Parse EASA FDT table.
    Returns list of ((start_hour, end_hour), {n_legs: max_fdt_minutes}).
    Row = check-in time window; columns 1..10 = number of sectors.
    """
    df = pd.read_csv(path)
    sector_cols = [c for c in df.columns if c.strip().isdigit()]
    limits = []
    for _, row in df.iterrows():
        sh = pd.to_timedelta(str(row['flight_duty_time_start_time'])).total_seconds() / 3600.0
        eh = pd.to_timedelta(str(row['flight_duty_time_end_time'])).total_seconds() / 3600.0
        smap = {int(c): int(float(row[c]) * 60) for c in sector_cols}
        limits.append(((sh, eh), smap))
    return limits


def build_fdt_lookup(limits):
    """
    Pre-compute a dense lookup table: fdt[minute_of_day][n_legs] -> max FDT (min).
    Called O(millions) of times inside the pricer, so we flatten it once into a
    NumPy array. Shape (1440, 11); index 0 of axis 1 is unused.
    """
    table = np.full((1440, MAX_LEGS_PER_DUTY + 1), 720, dtype=np.int32)
    for m in range(1440):
        h = m / 60.0
        for (sh, eh), smap in limits:
            hit = (h >= sh or h <= eh) if sh > eh else (sh <= h <= eh)
            if hit:
                for n in range(1, MAX_LEGS_PER_DUTY + 1):
                    table[m, n] = smap.get(n, smap[max(smap)])
                break
    return table


def load_flights(path):
    """Load, filter to window + A320 family, convert times to absolute minutes."""
    df = pd.read_csv(path)
    df['SCHEDULED_DEPARTURE_TIME'] = pd.to_datetime(df['SCHEDULED_DEPARTURE_TIME'])
    df['SCHEDULED_ARRIVAL_TIME']   = pd.to_datetime(df['SCHEDULED_ARRIVAL_TIME'])

    s, e = pd.to_datetime(START_DATE).date(), pd.to_datetime(END_DATE).date()
    df = df[(df['SCHEDULED_DEPARTURE_TIME'].dt.date >= s) &
            (df['SCHEDULED_DEPARTURE_TIME'].dt.date <= e)].copy()

    df['AIRCRAFT_TYPE'] = df['AIRCRAFT_TYPE'].astype(str)
    df = df[df['AIRCRAFT_TYPE'].isin(AIRCRAFT_FAMILY)].copy()

    base = pd.to_datetime(START_DATE)
    df['DEP_MIN'] = (df['SCHEDULED_DEPARTURE_TIME'] - base).dt.total_seconds() / 60.0
    df['ARR_MIN'] = (df['SCHEDULED_ARRIVAL_TIME']   - base).dt.total_seconds() / 60.0
    df['BLOCK']   = df['ARR_MIN'] - df['DEP_MIN']

    df = df[df['BLOCK'] > 0]                                   # drop corrupt rows
    df = df.sort_values('DEP_MIN').reset_index(drop=True)
    df['LEG_ID'] = df['LEG_ID'].astype(str)
    return df


def load_ground(path):
    """Symmetric map (apt_a, apt_b) -> ground transit minutes."""
    df = pd.read_csv(path)
    gt = {}
    for _, r in df.iterrows():
        a = str(r['Dep Ap']).strip().upper()
        b = str(r['Arr Ap']).strip().upper()
        t = int(float(r['avg_duration']) * 60)
        gt[(a, b)] = t
        gt[(b, a)] = t
    return gt


def load_bases(path):
    """Crew home bases, taken from the real roster file (NOT hardcoded)."""
    df = pd.read_csv(path)
    return sorted(df['home_base'].astype(str).str.strip().str.upper().unique())


# 3. NETWORK CONSTRUCTION
def build_network(df, gt, bases):
    """
    Explicit NetworkX DiGraph.
      Nodes: one per flight leg (+ SOURCE_<base>, SINK_<base>)
      Edges: TURN | LAYOVER | TRANSFER | LAYOVER_TRANSFER | START_* | END_*
    Edge cost = *connection* cost only. Flight block cost is on the NODE and is
    added by the pricer when the leg is consumed, which keeps the RCSPP correct.
    """
    G = nx.DiGraph()

    for r in df.itertuples():
        G.add_node(r.LEG_ID, kind='FLIGHT',
                   origin=r.DEPARTURE_AIRPORT, dest=r.ARRIVAL_AIRPORT,
                   dep=float(r.DEP_MIN), arr=float(r.ARR_MIN),
                   block=float(r.BLOCK),
                   cost=COST_PER_FLIGHT_MINUTE * float(r.BLOCK))

    for b in bases:
        G.add_node(f"SRC_{b}",  kind='SOURCE', base=b)
        G.add_node(f"SNK_{b}",  kind='SINK',   base=b)

    legs = [r.LEG_ID for r in df.itertuples()]
    dep  = df['DEP_MIN'].to_numpy()
    arr  = df['ARR_MIN'].to_numpy()

    # --- flight -> flight -----------------------------------------------------
    # df is sorted by DEP_MIN, so for leg i we only scan forward, and we stop as
    # soon as dep[j] - arr[i] exceeds MAX_LAYOVER_TIME (searchsorted bound).
    for i in range(len(legs)):
        u, ai = legs[i], arr[i]
        du, ou = G.nodes[u]['dest'], G.nodes[u]['origin']
        hi = np.searchsorted(dep, ai + MAX_LAYOVER_TIME, side='right')
        for j in range(i + 1, hi):
            v = legs[j]
            gap = dep[j] - ai
            if gap < MCT:
                continue
            ov = G.nodes[v]['origin']

            if du == ov:                                   # same airport
                if gap <= MAX_SIT_TIME:
                    G.add_edge(u, v, type='TURN', gap=gap, cost=0.0, rest=False)
                elif MIN_REST <= gap <= MAX_LAYOVER_TIME:
                    G.add_edge(u, v, type='LAYOVER', gap=gap,
                               cost=LAYOVER_BASE_COST, rest=True)
            else:                                          # ground transfer
                t = gt.get((du, ov))
                if t is None:
                    continue
                need = t + SECURITY_BUFFER
                tc = TRANSFER_FIX_COST + TRANSFER_VAR_COST * t
                if need <= gap <= MAX_SIT_TIME:
                    G.add_edge(u, v, type='TRANSFER', gap=gap,
                               cost=tc, rest=False, transit=t)
                elif max(MIN_REST, need) <= gap <= MAX_LAYOVER_TIME:
                    G.add_edge(u, v, type='LAYOVER_TRANSFER', gap=gap,
                               cost=LAYOVER_BASE_COST + tc, rest=True, transit=t)

    # --- source/sink ----------------------------------------------------------
    for lg in legs:
        o, d = G.nodes[lg]['origin'], G.nodes[lg]['dest']
        for b in bases:
            if o == b:
                G.add_edge(f"SRC_{b}", lg, type='START_DUTY', cost=0.0, rest=False)
            elif (b, o) in gt:
                t = gt[(b, o)]
                G.add_edge(f"SRC_{b}", lg, type='START_DUTY_TRANSFER',
                           cost=TRANSFER_FIX_COST + TRANSFER_VAR_COST * t,
                           rest=False, transit=t)
            if d == b:
                G.add_edge(lg, f"SNK_{b}", type='END_DUTY', cost=0.0, rest=False)
            elif (d, b) in gt:
                t = gt[(d, b)]
                G.add_edge(lg, f"SNK_{b}", type='END_DUTY_TRANSFER',
                           cost=TRANSFER_FIX_COST + TRANSFER_VAR_COST * t,
                           rest=False, transit=t)
    return G


# 4. RCSPP PRICING SUBPROBLEM  (label-setting on the DiGraph)
class Label:
    """
    A partial pairing.
      rc        : reduced cost so far  (real cost - sum of duals of covered legs)
      cost      : true cost so far  (what goes in the objective)
      node      : current flight leg
      duty_legs : legs flown in the CURRENT duty
      n_duties  : duties opened so far
      fdt_start : check-in clock (absolute min) of the current duty
      visited   : frozenset of legs (prevents cycles / double-cover)
      parent    : back-pointer for path reconstruction
      dominated : flag for lazy deletion
    """
    __slots__ = ('rc', 'cost', 'node', 'duty_legs', 'n_duties',
                 'fdt_start', 'visited', 'parent', 'dominated')

    def __init__(self, rc, cost, node, duty_legs, n_duties, fdt_start, visited, parent):
        self.rc = rc; self.cost = cost; self.node = node
        self.duty_legs = duty_legs; self.n_duties = n_duties
        self.fdt_start = fdt_start; self.visited = visited; self.parent = parent
        self.dominated = False

    def __lt__(self, other):          # for heapq tie-breaks
        return self.rc < other.rc


class RCSPP:
    """Label-setting RCSPP. Enforces EASA FDT at EVERY edge expansion."""

    def __init__(self, G, bases, fdt_table, leg_row, conshdlr=None):
        self.G = G
        self.bases = bases
        self.fdt = fdt_table
        self.leg_row = leg_row              # LEG_ID -> master-constraint index
        self.legs = [n for n, a in G.nodes(data=True) if a['kind'] == 'FLIGHT']
        self.conshdlr = conshdlr
        try:
            self.rev_topo = list(reversed(list(nx.topological_sort(G))))
        except nx.NetworkXUnfeasible:
            self.rev_topo = []
        self.min_rc = {}

    # ---- EASA feasibility check ------------------------------------------
    def _fdt_ok(self, fdt_start, arr_time, n_legs):
        """Is a duty that checked in at fdt_start, now has n_legs, and whose last
        leg lands at arr_time, still inside the EASA FDT envelope?"""
        if n_legs > MAX_LEGS_PER_DUTY:
            return False
        elapsed = (arr_time + CHECK_OUT) - fdt_start
        cap = self.fdt[int(fdt_start) % 1440, n_legs]
        return elapsed <= cap

    #  main solve 
    def solve(self, duals, max_cols=10, heuristic_limit=None):
        """
        Solve RCSPP from every base.
        Return list of (path, base, cost, rc)
        """
        self.min_rc = {n: float('inf') for n in self.G.nodes()}
        for b in self.bases:
            self.min_rc[f"SNK_{b}"] = 0.0
            
        for u in self.rev_topo:
            if u.startswith("SNK_"):
                continue
            best = float('inf')
            for _, v, e in self.G.out_edges(u, data=True):
                step_rc = e.get('cost', 0.0)
                if self.G.nodes[v]['kind'] == 'FLIGHT':
                    step_rc += self.G.nodes[v].get('cost', 0.0) - duals.get(v, 0.0)
                if self.min_rc[v] + step_rc < best:
                    best = self.min_rc[v] + step_rc
            self.min_rc[u] = best

        found = []
        for base in self.bases:
            if self.min_rc.get(f"SRC_{base}", float('inf')) >= -1e-6:
                continue
            found.extend(self._solve_from_base(base, duals, max_cols, heuristic_limit))
        
        # sort by reduced cost
        found.sort(key=lambda x: x[3])
        
        # Diversify: prefer columns that cover legs not yet hit this round. A pure
        # "most negative rc" pick returns near-duplicate paths over the same few
        # hot legs, which makes the root LP crawl. This spreads dual pressure.
        picked, hit = [], set()
        for item in found:
            if len(picked) >= max_cols:
                break
            if not set(item[0]) & hit:
                picked.append(item)
                hit.update(item[0])
        for item in found:                       # backfill with the best leftovers
            if len(picked) >= max_cols:
                break
            if item not in picked:
                picked.append(item)
        return picked

    def _solve_from_base(self, base, duals, max_cols, heuristic_limit=None):
        """
        Label-setting algorithm for a specific base.
        """
        G, src, snk = self.G, f"SRC_{base}", f"SNK_{base}"
        if src not in G:
            return []

        # buckets[(node, n_duties)] -> list of non-dominated labels
        buckets = defaultdict(list)
        pq = []                                  # best-first on reduced cost
        results = []

        #  seed: one label per leg reachable from the base source 
        for _, lg, e in G.out_edges(src, data=True):
            if self.conshdlr and lg in self.conshdlr.enforced_in:
                continue # Cannot start duty at lg if it MUST be preceded by something else
            nd = G.nodes[lg]
            fdt_start = nd['dep'] - CHECK_IN          # check-in clock starts here
            if fdt_start < 0:
                continue
            if not self._fdt_ok(fdt_start, nd['arr'], 1):
                continue
            cost = PAIRING_FIX_COST + e['cost'] + nd['cost']
            rc   = cost - duals.get(lg, 0.0)
            lab  = Label(rc, cost, lg, 1, 1, fdt_start, frozenset((lg,)), None)
            heapq.heappush(pq, (rc, id(lab), lab))

        while pq:
            rc, _, lab = heapq.heappop(pq)
            if lab.dominated: continue
            if len(results) >= max_cols:
                break

            # 1. try to CLOSE the pairing back at the SAME base 
            if G.has_edge(lab.node, snk):
                if not (self.conshdlr and lab.node in self.conshdlr.enforced_out):
                    e = G.edges[lab.node, snk]
                    close_arr = G.nodes[lab.node]['arr'] + (e.get('transit', 0))
                    if self._fdt_ok(lab.fdt_start, close_arr, lab.duty_legs):
                        tot_cost = lab.cost + e['cost']
                        tot_rc   = lab.rc   + e['cost']
                        if tot_rc < RC_EPS:
                            results.append((self._path(lab), base, tot_cost, tot_rc))

            #  2. EXTEND 
            enforced_out_node = self.conshdlr.enforced_out.get(lab.node) if self.conshdlr else None
            for _, v, e in G.out_edges(lab.node, data=True):
                if enforced_out_node and v != enforced_out_node:
                    continue
                if self.conshdlr and self.conshdlr.enforced_in.get(v) and self.conshdlr.enforced_in.get(v) != lab.node:
                    continue
                if self.conshdlr and (lab.node, v) in self.conshdlr.forbidden_edges:
                    continue
                if G.nodes[v]['kind'] != 'FLIGHT' or v in lab.visited:
                    continue
                nv = G.nodes[v]

                if e['rest']:                       # ---- LAYOVER: new duty ----
                    if lab.n_duties >= MAX_DUTIES_PER_PAIRING:
                        continue
                    n_duties  = lab.n_duties + 1
                    duty_legs = 1
                    fdt_start = nv['dep'] - CHECK_IN
                else:                               # ---- TURN/TRANSFER -------
                    if lab.duty_legs >= MAX_LEGS_PER_DUTY:
                        continue
                    n_duties  = lab.n_duties
                    duty_legs = lab.duty_legs + 1
                    fdt_start = lab.fdt_start

                if not self._fdt_ok(fdt_start, nv['arr'], duty_legs):
                    continue

                ncost = lab.cost + e['cost'] + nv['cost']
                nrc   = lab.rc   + e['cost'] + nv['cost'] - duals.get(v, 0.0)

                #  A* Bound Pruning 
                if nrc + self.min_rc.get(v, float('inf')) >= -1e-6:
                    continue

                # dominance / bucket pruning 
                key = (v, n_duties)
                bk  = buckets[key]
                # dominated if an existing label has <= rc, <= duty_legs, and a
                # later-or-equal fdt_start (i.e. more remaining FDT budget)
                dom = False
                for ol in bk:
                    if ol.rc <= nrc and ol.duty_legs <= duty_legs and ol.fdt_start >= fdt_start:
                        dom = True
                        break
                if dom:
                    continue
                if len(bk) >= MAX_LABELS_PER_NODE:
                    if nrc >= bk[-1].rc:
                        continue
                    worst_ol = bk.pop()
                    worst_ol.dominated = True
                
                if heuristic_limit and len(bk) >= heuristic_limit:
                    if nrc >= bk[-1].rc:
                        continue
                    worst_ol = bk.pop()
                    worst_ol.dominated = True

                buckets[key] = bk

                nl = Label(nrc, ncost, v, duty_legs, n_duties, fdt_start,
                           lab.visited | {v}, lab)
                bk.append(nl)
                bk.sort(key=lambda l: l.rc)
                heapq.heappush(pq, (nrc, id(nl), nl))

        return results

    @staticmethod
    def _path(lab):
        p = []
        while lab is not None:
            p.append(lab.node)
            lab = lab.parent
        return list(reversed(p))


# 5. SCIP PRICER PLUGIN  (this is what makes it Branch-and-PRICE)
class CrewPricer(Pricer):
    """
    SCIP calls pricerredcost() at every B&B node after the LP is solved.
    We read duals, run the RCSPP, and inject any negative-reduced-cost pairing
    as a new variable with pricedVar=True, wired into its cover constraints.
    """

    def __init__(self, rcspp, cover_cons, cols, verbose=True):
        super().__init__()
        self.rcspp = rcspp
        self.cover_cons = cover_cons      # LEG_ID -> SCIP constraint
        self.cols = cols                  # list of (var, path, base, cost)
        self.rounds = 0
        self.total_added = 0
        self.verbose = verbose
        # column pool management state
        self.active_idx = list(range(len(cols)))
        self.zero_streak = {}
        self.pruned = 0
        self._pruning_enabled = not RYAN_FOSTER   # global ub-fixing is unsafe under RF branching

    def pricerinit(self):
        """Constraints must be mapped to their transformed counterparts."""
        for lg in self.cover_cons:
            self.cover_cons[lg] = self.model.getTransformedCons(self.cover_cons[lg])

    def _prune_stale_columns(self):
        # only at root (see pooled.py reasoning); heuristic pruning shouldn't
        # tighten a global bound based on one subtree's duals
        if not self._pruning_enabled or self.model.getNNodes() > 1:
            return
        still = []
        for idx in self.active_idx:
            v = self.cols[idx][0]
            try:
                val = self.model.getVal(v)
            except Exception:
                self._pruning_enabled = False
                return
            streak = self.zero_streak.get(idx, 0) + 1 if val < 1e-7 else 0
            self.zero_streak[idx] = streak
            if streak >= STALE_ROUNDS_LIMIT:
                self.model.chgVarUbGlobal(v, 0.0)
                self.pruned += 1
                self.zero_streak.pop(idx, None)
            else:
                still.append(idx)
        self.active_idx = still

    def pricerredcost(self):
        self.rounds += 1
        self._prune_stale_columns()
        t0 = time.perf_counter()

        duals = {lg: self.model.getDualsolLinear(self.cover_cons[lg]) for lg in self.rcspp.legs}

        # Ryan-Foster: rebuild this node's decisions BEFORE pricing, so the
        # RCSPP cannot regenerate a column the branch just forbade.
        if getattr(self.rcspp, 'conshdlr', None) is not None:
            self.rcspp.conshdlr.refresh_from_node()

        # Multi-Phase Pricing: Fast Pass
        new = self.rcspp.solve(duals, max_cols=COLUMNS_PER_ROUND, heuristic_limit=1)
        
        # Exact Pass (if Fast Pass fails)
        if not new:
            if self.verbose:
                print(f"  [pricer r{self.rounds:>3}] Fast pass failed, running Exact pass...")
            new = self.rcspp.solve(duals, max_cols=COLUMNS_PER_ROUND, heuristic_limit=None)

        dt = time.perf_counter() - t0
        if not new:
            if self.verbose:
                print(f"  [pricer r{self.rounds:>3}] no negative rc column "
                      f"-> LP optimal for this node   ({dt:.2f}s)")
            return {'result': SCIP_RESULT.SUCCESS}

        for path, base, cost, rc in new:
            idx = len(self.cols)
            v = self.model.addVar(vtype='B', lb=0.0, ub=1.0, obj=cost,
                                  pricedVar=True,
                                  name=f"p{idx}_{base}")
            for lg in path:
                self.model.addConsCoeff(self.cover_cons[lg], v, 1.0)
            self.cols.append((v, path, base, cost))
            self.active_idx.append(idx)
            if getattr(self.rcspp, "conshdlr", None) is not None:
                self.rcspp.conshdlr.register_column(idx, v, path)

        self.total_added += len(new)
        if self.verbose:
            best = new[0]
            print(f"  [pricer r{self.rounds:>3}] +{len(new):>2} cols | "
                  f"best rc {best[3]:>10.2f} | legs {len(best[0]):>2} @ {best[1]} | "
                  f"total {self.total_added:>4} | active {len(self.active_idx):>5} | "
                  f"pruned {self.pruned:>5} | {dt:.2f}s")

        return {'result': SCIP_RESULT.SUCCESS}

    def pricerfarkas(self):
        """Artificial columns guarantee LP feasibility, so Farkas pricing is a no-op."""
        return {'result': SCIP_RESULT.SUCCESS}


# 5.5 RYAN-FOSTER BRANCHING COMPONENTS
class RyanFosterConshdlr(Conshdlr):
    def __init__(self, cols):
        super().__init__()
        self.cols = cols
        self.forbidden_edges = defaultdict(int)
        self.enforced_out = {}
        self.enforced_in = {}
        # incremental indices so consprop / branch rule never scan every column
        self.leg_to_cols = defaultdict(set)
        self.edge_to_cols = defaultdict(set)
        self.var_name_to_idx = {}
        # node number -> (u, v, 'DIFFER'|'TOGETHER').  Registry-based encoding:
        # nothing is attached to the node itself (pyscipopt 6.x addConsNode only
        # accepts ExprCons), state is rebuilt by walking the ancestor chain.
        self.node_decision = {}

    def refresh_from_node(self):
        """Rebuild forbidden/enforced state for the CURRENT B&B node."""
        self.forbidden_edges = defaultdict(int)
        self.enforced_out = {}
        self.enforced_in = {}
        if self.model is None:
            return
        try:
            node = self.model.getCurrentNode()
        except Exception:
            return
        while node is not None:
            d = self.node_decision.get(node.getNumber())
            if d is not None:
                u, v, kind = d
                if kind == 'TOGETHER':
                    self.enforced_out[u] = v
                    self.enforced_in[v] = u
                else:
                    self.forbidden_edges[(u, v)] += 1
            try:
                node = node.getParent()
            except Exception:
                break

    def register_column(self, idx, var, path):
        self.var_name_to_idx[var.name] = idx
        for lg in path:
            self.leg_to_cols[lg].add(idx)
        for i in range(len(path) - 1):
            self.edge_to_cols[(path[i], path[i + 1])].add(idx)

    def consactive(self, constraint):
        # No-op by design: no constraint objects are created any more.
        return

    def consdeactive(self, constraint):
        return

    def consprop(self, constraints, nusefulconss, nmarkconss, proptiming):
        """Fix out every column violating the decisions in force at this node.
        Also catches columns priced AFTER the branch was taken."""
        result = SCIP_RESULT.DIDNOTFIND
        self.refresh_from_node()
        decisions = [(u, v, True) for u, v in self.enforced_out.items()]
        decisions += [(u, v, False) for (u, v) in list(self.forbidden_edges)]
        for u, v, together in decisions:
            candidates = (self.leg_to_cols.get(u, set()) | self.leg_to_cols.get(v, set())) \
                         if together else self.edge_to_cols.get((u, v), set())
            for idx in candidates:
                var, path, base, cost = self.cols[idx]
                if var.getUbLocal() <= 0.5:
                    continue
                violates = False
                if together:
                    if u in path:
                        pos = path.index(u)
                        if pos == len(path) - 1 or path[pos+1] != v:
                            violates = True
                    if not violates and v in path:
                        pos = path.index(v)
                        if pos == 0 or path[pos-1] != u:
                            violates = True
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
            if idx is None:
                continue
            path = self.cols[idx][1]
            for i in range(len(path)-1):
                edge_flow[(path[i], path[i+1])] += f

        self.conshdlr.refresh_from_node()
        best_edge = None
        best_diff = 1.0
        for edge, flow in edge_flow.items():
            if not (1e-6 < flow < 1 - 1e-6):
                continue
            u_, v_ = edge
            if self.conshdlr.enforced_out.get(u_) == v_:
                continue                      # already decided TOGETHER
            if edge in self.conshdlr.forbidden_edges:
                continue                      # already decided DIFFER
            diff = abs(flow - 0.5)
            if diff < best_diff:
                best_diff = diff
                best_edge = edge
                    
        if best_edge is None:
            return {"result": SCIP_RESULT.DIDNOTRUN}
            
        u, v = best_edge

        est = self.model.getLocalEstimate()
        node0 = self.model.createChild(0.0, est)
        node1 = self.model.createChild(0.0, est)
        # Registry encoding: record the decision against each child's node
        # number. consprop and the pricer rebuild it via refresh_from_node().
        self.conshdlr.node_decision[node0.getNumber()] = (u, v, 'DIFFER')
        self.conshdlr.node_decision[node1.getNumber()] = (u, v, 'TOGETHER')

        self.n_branches += 1
        return {"result": SCIP_RESULT.BRANCHED}


# 6. MASTER PROBLEM + DRIVER
def structurally_uncoverable(G, legs, bases):
    """
    Legs with NO source->leg->sink path in the graph. These can never be flown by
    any pairing (typically late legs at outstations that cannot be brought home
    inside the 2-day horizon / 3-duty / 36h-layover envelope). They are a property
    of the DATA, not a solver failure -- the artificial columns absorb them, and we
    report them explicitly rather than hiding them inside the objective.
    """
    fwd, bwd = set(), set()
    for b in bases:
        if f"SRC_{b}" in G:
            fwd |= nx.descendants(G, f"SRC_{b}")
        if f"SNK_{b}" in G:
            bwd |= nx.ancestors(G, f"SNK_{b}")
    ok = fwd & bwd
    return [lg for lg in legs if lg not in ok]


def build_master(G, legs, bases, fdt_table, verbose=True):
    m = Model("CrewPairing_SetCovering")
    m.setPresolve(SCIP_PARAMSETTING.OFF)     # required: presolve breaks pricing
    m.setSeparating(SCIP_PARAMSETTING.OFF)
    m.setIntParam("presolving/maxrestarts", 0)

    cols = []
    cover = {}

    # --- artificial (slack) columns: guarantee an initial feasible RMP --------
    for lg in legs:
        # vtype='B' (was 'C'): during LP relaxation SCIP still relaxes this to
        # [0,1] so column generation is unchanged, but it gives branch-and-price
        # an integrality target. This is THE fix for the ~61% coverage artifact:
        # previously the fractional LP solution was read through a >0.5 filter
        # with no branching, so legs covered only by sub-0.5 pairings looked
        # 'uncovered'. With binary vars, Ryan-Foster branching resolves those
        # fractions into a real integer roster.
        a = m.addVar(vtype='B', lb=0.0, obj=ARTIFICIAL_COST, name=f"art_{lg}")
        c = m.addCons(a == 1.0, name=f"cover_{lg}",
                      separate=False, modifiable=True)   # modifiable -> pricer may add coeffs
        cover[lg] = c

    leg_row = {lg: i for i, lg in enumerate(legs)}
    
    # Initialize and include Ryan-Foster branching
    rf_conshdlr = RyanFosterConshdlr(cols)
    # needscons=False is REQUIRED: no constraint objects exist any more, so with
    # the default (True) SCIP would never call consprop.
    m.includeConshdlr(rf_conshdlr, "RyanFoster", "Ryan-Foster constraint handler",
                      chckpriority=100000, enfopriority=100000, propfreq=1,
                      needscons=False)

    rf_branchrule = RyanFosterBranchrule(cols, rf_conshdlr)
    if RYAN_FOSTER:
        m.includeBranchrule(rf_branchrule, "RyanFosterBranchrule", "Ryan-Foster Branching",
                            priority=100000, maxdepth=-1, maxbounddist=1.0)
    
    rcspp = RCSPP(G, bases, fdt_table, leg_row, rf_conshdlr)

    pricer = CrewPricer(rcspp, cover, cols, verbose)
    m.includePricer(pricer, "CrewPricer", "RCSPP pricer for crew pairings")
    return m, pricer, cols, cover, rf_branchrule


def main():
    t_all = time.perf_counter()

    print("=" * 78)
    print("AIRLINE CREW PAIRING  --  BRANCH-AND-PRICE (SCIP + NetworkX RCSPP)")
    print("=" * 78)

    #  1. DATA 
    print("\n[1/5] Loading data ...")
    df    = load_flights(os.path.join(DATA_DIR, "flight_schedule.csv"))
    gt    = load_ground(os.path.join(DATA_DIR, "ground_transportation_times.csv"))
    bases = load_bases(os.path.join(DATA_DIR, "home_bases.csv"))
    fdt   = build_fdt_lookup(load_fdt_limits(os.path.join(DATA_DIR, "fdt_limits.csv")))
    print(f"      flights (A320 family, {START_DATE}..{END_DATE}) : {len(df)}")
    print(f"      ground transfer pairs                      : {len(gt)//2}")
    print(f"      crew bases (from home_bases.csv)           : {bases}")

    #  2. NETWORK -
    print("\n[2/5] Building leg-based network ...")
    t0 = time.perf_counter()
    G = build_network(df, gt, bases)
    et = defaultdict(int)
    for _, _, a in G.edges(data=True):
        et[a['type']] += 1
    print(f"      nodes {G.number_of_nodes():>6}   edges {G.number_of_edges():>7}"
          f"   ({time.perf_counter()-t0:.1f}s)")
    for k in sorted(et):
        print(f"        {k:<22} {et[k]:>7}")

    #  3. MASTER 
    legs = [r.LEG_ID for r in df.itertuples()]
    dead = structurally_uncoverable(G, legs, bases)
    print(f"      structurally uncoverable legs              : {len(dead)}"
          f"  (no SRC->leg->SNK path; absorbed by artificials)")

    print("\n[3/5] Building Restricted Master Problem (set covering) ...")
    m, pricer, cols, cover, rf_branchrule = build_master(G, legs, bases, fdt)
    print(f"      cover constraints : {len(cover)}")
    print(f"      artificial columns: {len(legs)}  (cost {ARTIFICIAL_COST:,.0f} each)")

    # 4. SOLVE
    print("\n[4/5] Branch-and-Price (SCIP drives B&B; pricer generates columns)")
    print("-" * 78)
    m.setRealParam("limits/time", float(os.environ.get("CPP_TIME_LIMIT", 600)))
    m.optimize()
    print("-" * 78)

    # 5. OUTPUT 
    print(f"\n[5/5] Results")
    print(f"      status            : {m.getStatus()}")
    print(f"      pricing rounds    : {pricer.rounds}")
    print(f"      columns generated : {pricer.total_added}")
    print(f"      columns pruned    : {pricer.pruned}  (active: {len(pricer.active_idx)})")
    print(f"      B&B nodes         : {m.getNNodes()}")
    print(f"      Ryan-Foster branches : {rf_branchrule.n_branches}")

    if m.getNSols() == 0:
        print("      no solution found within limits.")
        return

    sol = m.getBestSol()

    art_used = sum(1 for v in m.getVars()
                   if v.name.startswith("art_") and sol[v] > 1e-6)
    chosen = [(v, p, b, c) for (v, p, b, c) in cols if sol[v] > 0.5]
    crew_cost = sum(c for (_v, _p, _b, c) in chosen)

    print(f"      optimality gap    : {m.getGap()*100:.4f} %")
    print(f"      uncovered legs    : {art_used}  "
          f"(of which {len(dead)} structurally impossible)")
    print(f"\n      TRUE CREW COST    : EUR {crew_cost:,.2f}   <-- the real KPI")
    print(f"      penalty (Big-M)   : EUR {m.getObjVal()-crew_cost:,.2f}  "
          f"({art_used} artificials x {ARTIFICIAL_COST:,.0f})")
    print(f"      LP objective      : EUR {m.getObjVal():,.2f}")

    chosen = [(v, p, b, c) for (v, p, b, c) in cols if sol[v] > 0.5]
    chosen.sort(key=lambda x: -x[3])

    print(f"\n=== SELECTED PAIRINGS ({len(chosen)}) ===")
    covered = set()
    for i, (v, path, base, cost) in enumerate(chosen, 1):
        covered.update(path)
        seq = []
        for j, lg in enumerate(path):
            n = G.nodes[lg]
            seq.append(f"{n['origin']}->{n['dest']}")
            if j + 1 < len(path):
                e = G.edges[lg, path[j + 1]]
                seq.append(f"[{e['type'][:3]}]")
        print(f"\n  Pairing {i:>3} | base {base} | legs {len(path):>2} | EUR {cost:>9,.2f}")
        print(f"     route : {' '.join(seq)}")
        print(f"     legs  : {' -> '.join(path)}")

    coverable = len(legs) - len(dead)
    print(f"\n  coverage: {len(covered)}/{len(legs)} of all legs "
          f"({100*len(covered)/len(legs):.1f}%)")
    print(f"  coverage: {len(covered)}/{coverable} of COVERABLE legs "
          f"({100*len(covered)/max(coverable,1):.1f}%)   <-- the meaningful figure")

    #  6. CSV EXPORT (CPP_EXPORT=1) 
    # Columns are chosen to match exactly what stage2_rostering.solve_rostering()
    # expects per pairing (pairing_id, base, legs, n_legs, cost, start_min,
    # end_min), so a per-base file can be fed straight into the rostering stage
    # without any reshaping. start_min/end_min are minutes from START_DATE 00:00,
    # taken from the flight nodes' dep/arr, matching stage 1's own convention.
    if os.environ.get("CPP_EXPORT"):
        stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")

        pair_rows = []
        for i, (v, path, base, cost) in enumerate(chosen, 1):
            deps = [G.nodes[lg]['dep'] for lg in path]
            arrs = [G.nodes[lg]['arr'] for lg in path]
            pair_rows.append({
                "pairing_id": i,
                "base":       base,
                "n_legs":     len(path),
                "cost":       round(cost, 2),
                "start_min":  min(deps),
                "end_min":    max(arrs),
                "route":      " ".join(f"{G.nodes[lg]['origin']}->{G.nodes[lg]['dest']}"
                                       for lg in path),
                "legs":       "|".join(path),
            })
        pdf = pd.DataFrame(pair_rows)

        all_path = f"pairings_ALL_{START_DATE}_{END_DATE}_{stamp}.csv"
        pdf.to_csv(all_path, index=False)
        print(f"\n  Exported: {all_path}  ({len(pdf)} pairings, all bases)")

        # per-base files, each directly consumable by stage2_rostering.py
        for b in sorted(pdf['base'].unique()):
            sub = pdf[pdf['base'] == b].copy()
            # renumber so pairing_id is 1..n within each base file
            sub['pairing_id'] = range(1, len(sub) + 1)
            bp = f"pairings_{b}_{START_DATE}_{END_DATE}_{stamp}.csv"
            sub.to_csv(bp, index=False)
            print(f"            {bp}  ({len(sub)} pairings)")

        # uncovered / structurally-impossible legs, for the write-up
        uncovered_legs = sorted(set(legs) - covered)
        dead_set = set(dead)
        udf = pd.DataFrame([{
            "leg_id":   lg,
            "origin":   G.nodes[lg]['origin'],
            "dest":     G.nodes[lg]['dest'],
            "dep_min":  G.nodes[lg]['dep'],
            "arr_min":  G.nodes[lg]['arr'],
            "reason":   "structurally_impossible" if lg in dead_set else "not_selected",
        } for lg in uncovered_legs])
        upath = f"uncovered_legs_{START_DATE}_{END_DATE}_{stamp}.csv"
        udf.to_csv(upath, index=False)
        print(f"            {upath}  ({len(udf)} legs)")

        # run summary (one row) so results are comparable across runs
        sdf = pd.DataFrame([{
            "start_date": START_DATE, "end_date": END_DATE,
            "status": m.getStatus(),
            "pricing_rounds": pricer.rounds,
            "columns_generated": pricer.total_added,
            "columns_pruned": pricer.pruned,
            "columns_active": len(pricer.active_idx),
            "bnb_nodes": m.getNNodes(),
            "ryan_foster_branches": rf_branchrule.n_branches,
            "pairings_selected": len(chosen),
            "legs_total": len(legs),
            "legs_coverable": coverable,
            "legs_covered": len(covered),
            "coverage_pct_coverable": round(100*len(covered)/max(coverable, 1), 2),
            "true_crew_cost_eur": round(crew_cost, 2),
            "lp_objective_eur": round(m.getObjVal(), 2),
            "artificials_used": art_used,
            "wall_time_s": round(time.perf_counter()-t_all, 1),
        }])
        spath = f"summary_{START_DATE}_{END_DATE}_{stamp}.csv"
        sdf.to_csv(spath, index=False)
        print(f"            {spath}")

    print(f"\nTotal wall time: {time.perf_counter()-t_all:.1f}s")


if __name__ == "__main__":
    main()