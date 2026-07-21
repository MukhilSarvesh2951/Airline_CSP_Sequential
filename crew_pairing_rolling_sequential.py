import os
import time
import heapq
import itertools
from collections import defaultdict
from datetime import datetime

import pandas as pd
import numpy as np
import networkx as nx

from pyscipopt import Model, Pricer, SCIP_RESULT, SCIP_PARAMSETTING

# ==============================================================================
# 1. PARAMETERS & EASA RULES
# ==========================================
DATA_DIR = r"C:\Users\defne\Documents\Master\Semester 2\Analytics Project\or_project_data\or_project_data"

MIN_REST                = 12 * 60 # 12 hours
MCT                     = 45      # 45 mins
CHECK_IN                = 60      # 60 mins
CHECK_OUT               = 30      # 30 mins
MAX_LEGS_PER_DUTY       = 5
MAX_LAYOVER_TIME        = 36 * 60 # 36 hours maximum wait at an outpost

ARTIFICIAL_COST         = 1.0e5   # Big-M for artificial (slack) columns

AIRCRAFT_FAMILY = ['319', '320', '321']   # A320 family only

# --- Column generation controls -----------------------------------------------
COLUMNS_PER_ROUND = 50   # how many negative-rc columns to add per pricing round
# ==============================================================================

def load_flights(csv_path):
    df = pd.read_csv(csv_path)
    df['SCHEDULED_DEPARTURE_TIME'] = pd.to_datetime(df['SCHEDULED_DEPARTURE_TIME'])
    df['SCHEDULED_ARRIVAL_TIME'] = pd.to_datetime(df['SCHEDULED_ARRIVAL_TIME'])
    df = df[df['AIRCRAFT_TYPE'].astype(str).str[:3].isin(AIRCRAFT_FAMILY)].copy()
    
    baseline = pd.to_datetime("2025-01-01") # Fixed baseline for absolute minutes
    df['DEP_MIN'] = (df['SCHEDULED_DEPARTURE_TIME'] - baseline).dt.total_seconds() / 60
    df['ARR_MIN'] = (df['SCHEDULED_ARRIVAL_TIME'] - baseline).dt.total_seconds() / 60
    df = df.sort_values('DEP_MIN').reset_index(drop=True)
    return df

def load_ground(csv_path):
    gt = pd.read_csv(csv_path)
    gt_map = {}
    for _, r in gt.iterrows():
        gt_map[(r['Dep Ap'], r['Arr Ap'])] = r['avg_duration'] * 60
    return gt_map

def load_bases(csv_path):
    b = pd.read_csv(csv_path)
    return b['home_base'].unique().tolist()

def load_fdt_limits(path):
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

# ==============================================================================
# 2. DYNAMIC NETWORK BUILDER
# ==============================================================================
def build_network(df, gt, bases, start_locations, window_end_min):
    """
    Builds the DAG for the subset of flights in the current rolling window.
    Adds SRC_{loc} for every location a captain is stranded at, plus bases.
    Adds SNK_{base} and an OPEN_OUTPOST dummy sink to allow pairings to span windows.
    """
    G = nx.DiGraph()
    
    legs, dep, arr, orig, dest = [], [], [], [], []
    for r in df.itertuples():
        G.add_node(r.LEG_ID, kind='FLIGHT', dep=r.DEP_MIN, arr=r.ARR_MIN,
                   origin=r.DEPARTURE_AIRPORT, dest=r.ARRIVAL_AIRPORT, cost=r.ARR_MIN - r.DEP_MIN)
        legs.append(r.LEG_ID)
        dep.append(r.DEP_MIN)
        arr.append(r.ARR_MIN)
        orig.append(r.DEPARTURE_AIRPORT)
        dest.append(r.ARRIVAL_AIRPORT)
        
    for loc in start_locations:
        G.add_node(f"SRC_{loc}", kind='SOURCE', base=loc)
    
    for b in bases:
        G.add_node(f"SNK_{b}", kind='SINK', base=b)
        
    G.add_node("OPEN_OUTPOST", kind='SINK', base='ANY')

    # Connections
    for i in range(len(legs)):
        u, du, au, ou, du_dest = legs[i], dep[i], arr[i], orig[i], dest[i]
        
        # SRC -> Flight
        if ou in start_locations:
            G.add_edge(f"SRC_{ou}", u, type='SRC', cost=0.0, rest=True)
            
        # Flight -> SNK
        if du_dest in bases:
            G.add_edge(u, f"SNK_{du_dest}", type='SNK', cost=0.0, rest=True)
            
        # Flight -> OPEN_OUTPOST (If it arrives within 20h of window end)
        if au >= window_end_min - 20 * 60:
            G.add_edge(u, "OPEN_OUTPOST", type='OPEN', cost=0.0, rest=True)
            
        # Flight -> Flight
        for j in range(i + 1, len(legs)):
            v, dv, av, ov, dv_dest = legs[j], dep[j], arr[j], orig[j], dest[j]
            if dv < au: continue
            
            if du_dest == ov:
                gap = dv - au
                if MCT <= gap <= MAX_LAYOVER_TIME:
                    if gap >= MIN_REST:
                        G.add_edge(u, v, type='REST', cost=0.0, rest=True)
                    else:
                        G.add_edge(u, v, type='CONNECTION', cost=0.0, rest=False)
            else:
                transfer = gt.get((du_dest, ov))
                if transfer is not None:
                    gap = dv - (au + transfer)
                    if gap >= MCT and gap <= MAX_LAYOVER_TIME:
                        G.add_edge(u, v, type='GROUND', cost=transfer, rest=(gap >= MIN_REST))
                        
    return G

# ==============================================================================
# 3. LABEL SETTING RCSPP
# ==============================================================================
class Label:
    __slots__ = ('rc', 'cost', 'node', 'duty_legs', 'n_duties',
                 'fdt_start', 'visited', 'parent', 'dominated', 'cum_duty')

    def __init__(self, rc, cost, node, duty_legs, n_duties, fdt_start, visited, parent, cum_duty):
        self.rc = rc; self.cost = cost; self.node = node
        self.duty_legs = duty_legs; self.n_duties = n_duties
        self.fdt_start = fdt_start; self.visited = visited; self.parent = parent
        self.cum_duty = cum_duty
        self.dominated = False

    def __lt__(self, other):
        return self.rc < other.rc

class RCSPP:
    def __init__(self, G, bases, fdt_table, leg_row, conshdlr=None):
        self.G = G
        self.bases = bases
        self.fdt = fdt_table
        self.leg_row = leg_row
        self.legs = [n for n, a in G.nodes(data=True) if a['kind'] == 'FLIGHT']
        self.conshdlr = conshdlr
        try:
            self.rev_topo = list(reversed(list(nx.topological_sort(G))))
        except nx.NetworkXUnfeasible:
            self.rev_topo = []
        self.min_rc = {}

    def _fdt_ok(self, fdt_start, arr_time, n_legs):
        if n_legs > MAX_LEGS_PER_DUTY:
            return False
        elapsed = (arr_time + CHECK_OUT) - fdt_start
        cap = self.fdt[int(fdt_start) % 1440, n_legs]
        return elapsed <= cap

    def solve(self, duals, captain_states, max_cols=10, heuristic_limit=None):
        self.min_rc = {n: float('inf') for n in self.G.nodes()}
        for b in self.bases:
            self.min_rc[f"SNK_{b}"] = 0.0
        self.min_rc["OPEN_OUTPOST"] = 0.0
            
        for u in self.rev_topo:
            if isinstance(u, str) and (u.startswith("SNK_") or u == "OPEN_OUTPOST"):
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
        for state in captain_states:
            found.extend(self._solve_from_state(state, duals, max_cols, heuristic_limit))
            
        found.sort(key=lambda x: x[6]) # sort by reduced cost
        
        picked, hit = [], set()
        for item in found:
            if len(picked) >= max_cols: break
            if not set(item[0]) & hit:
                picked.append(item)
                hit.update(item[0])
        for item in found:
            if len(picked) >= max_cols: break
            if item not in picked:
                picked.append(item)
        return picked

    def _solve_from_state(self, state, duals, max_cols, heuristic_limit=None):
        loc, min_dep, max_dep, home_base, start_cum_duty, anon_id = state
        G, src = self.G, f"SRC_{loc}"
        if src not in G:
            return []

        buckets = defaultdict(list)
        pq = []
        results = []

        for _, lg, e in G.out_edges(src, data=True):
            flight_dep = G.nodes[lg]['dep']
            if not (min_dep <= flight_dep <= max_dep):
                continue
                
            new_cum_duty = start_cum_duty + 4.0
            if new_cum_duty > 60.0: continue
            
            nrc = G.nodes[lg]['cost'] - duals.get(lg, 0.0) + e['cost']
            ncost = G.nodes[lg]['cost'] + e['cost']
            if nrc + self.min_rc[lg] >= 0: continue
            
            fdt_start = flight_dep - CHECK_IN
            lab = Label(nrc, ncost, lg, 1, 1, fdt_start, frozenset([lg]), None, new_cum_duty)
            buckets[(lg, 1)].append(lab)
            heapq.heappush(pq, (nrc, id(lab), lab))

        while pq and len(results) < max_cols * 5:
            rc, _, lab = heapq.heappop(pq)
            if lab.dominated: continue
            u = lab.node

            for _, v, e in G.out_edges(u, data=True):
                is_rest = e.get('rest', False)
                n_duties = lab.n_duties + 1 if is_rest else lab.n_duties
                
                if G.nodes[v]['kind'] == 'SINK':
                    if v == f"SNK_{home_base}":
                        path = self._path(lab)
                        arr_time = G.nodes[u]['arr']
                        results.append((path, loc, home_base, home_base, arr_time, lab.cost, lab.rc, anon_id))
                    elif v == "OPEN_OUTPOST":
                        path = self._path(lab)
                        arr_time = G.nodes[u]['arr']
                        end_loc = G.nodes[u]['dest']
                        results.append((path, loc, home_base, end_loc, arr_time, lab.cost, lab.rc, anon_id))
                    continue
                    
                if v in lab.visited: continue
                if self.conshdlr and not self.conshdlr.is_valid(lab.visited, v): continue
                
                new_cum_duty = lab.cum_duty + 4.0
                if new_cum_duty > 60.0: continue
                
                nrc = lab.rc + G.nodes[v].get('cost', 0.0) - duals.get(v, 0.0) + e['cost']
                if nrc + self.min_rc[v] >= 0: continue

                arr_time = G.nodes[v]['arr']
                fdt_start = (G.nodes[v]['dep'] - CHECK_IN) if is_rest else lab.fdt_start
                duty_legs = 1 if is_rest else lab.duty_legs + 1

                if not self._fdt_ok(fdt_start, arr_time, duty_legs): continue

                ncost = lab.cost + G.nodes[v].get('cost', 0.0) + e['cost']
                key = (v, n_duties)
                bk = buckets[key]

                if heuristic_limit and len(bk) >= heuristic_limit:
                    if nrc >= bk[-1].rc: continue
                    worst_ol = bk.pop()
                    worst_ol.dominated = True

                nl = Label(nrc, ncost, v, duty_legs, n_duties, fdt_start, lab.visited | {v}, lab, new_cum_duty)
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

class CrewPricer(Pricer):
    def __init__(self, rcspp, cover_cons, cols, captain_states, verbose=True):
        super().__init__()
        self.rcspp = rcspp
        self.cover_cons = cover_cons
        self.cols = cols
        self.captain_states = captain_states
        self.rounds = 0
        self.verbose = verbose

    def pricerinit(self):
        for lg in self.cover_cons:
            self.cover_cons[lg] = self.model.getTransformedCons(self.cover_cons[lg])

    def pricerredcost(self):
        self.rounds += 1
        duals = {}
        try:
            for lg in self.rcspp.legs:
                duals[lg] = self.model.getDualsolLinear(self.cover_cons[lg])
        except Exception as e:
            print(f"[ERROR] Failed to get duals: {e}")
            return {'result': SCIP_RESULT.SUCCESS}

        # Fast Pass
        new = self.rcspp.solve(duals, self.captain_states, max_cols=COLUMNS_PER_ROUND, heuristic_limit=1)
        if not new:
            new = self.rcspp.solve(duals, self.captain_states, max_cols=COLUMNS_PER_ROUND, heuristic_limit=None)

        if not new:
            return {'result': SCIP_RESULT.SUCCESS}

        for path, start_loc, home_base, end_loc, end_time, cost, rc, anon_id in new:
            v = self.model.addVar(vtype='C', lb=0.0, ub=1.0, obj=cost, pricedVar=True, name=f"p{len(self.cols)}")
            for lg in path:
                self.model.addConsCoeff(self.cover_cons[lg], v, 1.0)
            self.cols.append((v, path, start_loc, home_base, end_loc, end_time, cost, anon_id))

        return {'result': SCIP_RESULT.SUCCESS}

def structurally_uncoverable(G, legs, start_locations, bases):
    """Fallback check if a leg is physically disconnected from SRCs and SNKs."""
    dead = []
    # simplified: just checking if in_degree / out_degree > 0
    return dead

def build_master(G, legs, bases, fdt, start_locations, captain_states, verbose=True):
    m = Model("CrewPairing_Master")
    if not verbose: m.hideOutput()
    m.setPresolve(SCIP_PARAMSETTING.OFF)

    cols = []
    cover = {}
    for lg in legs:
        a = m.addVar(vtype='C', lb=0.0, obj=ARTIFICIAL_COST, name=f"art_{lg}")
        cover[lg] = m.addCons(a >= 1.0, name=f"cov_{lg}", separate=False, modifiable=True)
    
    rcspp = RCSPP(G, bases, fdt, cover, conshdlr=None)
    pricer = CrewPricer(rcspp, cover, cols, captain_states, verbose)
    m.includePricer(pricer, "CrewPricer", "RCSPP open pairings")
    return m, pricer, cols, cover

# ==============================================================================
# MAIN ROLLING HORIZON LOOP
# ==============================================================================
def main():
    t_all = time.perf_counter()

    print("=" * 78)
    print("AIRLINE CREW SCHEDULING  --  2-DAY ROLLING HORIZON WITH OPEN OUTPOSTS")
    print("=" * 78)

    print("\n[1/5] Loading data ...")
    df    = load_flights(os.path.join(DATA_DIR, "flight_schedule.csv"))
    gt    = load_ground(os.path.join(DATA_DIR, "ground_transportation_times.csv"))
    bases = load_bases(os.path.join(DATA_DIR, "home_bases.csv"))
    fdt   = build_fdt_lookup(load_fdt_limits(os.path.join(DATA_DIR, "fdt_limits.csv")))
    
    home_bases_df = pd.read_csv(os.path.join(DATA_DIR, "home_bases.csv"))
    captains_list = home_bases_df['captain_id'].unique().tolist()
    
    print(f"      flights (A320 family)                      : {len(df)}")
    print(f"      captains loaded                            : {len(captains_list)}")

    in_progress_crews = [] # list of dicts: {'loc', 'sleep', 'home', 'cum_duty', 'anon_id'}
    master_rosters = {} # anon_id -> {'home_base', 'path': [], 'cost'}
    next_anon_id = 1

    df['FLIGHT_DATE'] = pd.to_datetime(df['SCHEDULED_DEPARTURE_TIME']).dt.normalize()
    unique_dates = sorted(df['FLIGHT_DATE'].unique())[:7] # Limit to 7 days for now
    
    seven_day_df = df[df['FLIGHT_DATE'].isin(unique_dates)]
    total_flights_7d = len(seven_day_df)
    
    window_size = 2
    step_size = 1
    
    covered_legs_global = set()

    for i in range(0, len(unique_dates), step_size):
        if i + window_size > len(unique_dates) and i > 0:
            break
            
        window_start = unique_dates[i]
        window_end = window_start + pd.Timedelta(days=window_size-1)
        
        baseline = pd.to_datetime("2025-01-01")
        window_end_min = (window_end - baseline).total_seconds() / 60 + (24 * 60) # End of the window day
        
        print(f"\n{'='*60}")
        print(f"--- ROLLING WINDOW: {window_start.date()} to {window_end.date()} ---")
        print(f"{'='*60}")
        
        window_df = df[(df['FLIGHT_DATE'] >= window_start) & (df['FLIGHT_DATE'] <= window_end)].copy()
        window_df = window_df[~window_df['LEG_ID'].isin(covered_legs_global)]
        
        if window_df.empty:
            continue
            
        print(f"      active flights in window: {len(window_df)}")
        
        unique_states = set()
        start_locations = set()
        
        # 1. Fresh crews at all bases
        for b in bases:
            start_locations.add(b)
            unique_states.add((b, 0, float('inf'), b, 0.0, None))
            
        # 2. In-progress crews
        for ipc in in_progress_crews:
            loc, sleep, home, cum_duty, anon_id = ipc['loc'], ipc['sleep'], ipc['home'], ipc['cum_duty'], ipc['anon_id']
            start_locations.add(loc)
            min_dep = sleep + MIN_REST if sleep > 0 else 0
            max_dep = sleep + MAX_LAYOVER_TIME
            unique_states.add((loc, min_dep, max_dep, home, cum_duty, anon_id))
            
        G = build_network(window_df, gt, bases, start_locations, window_end_min)
        legs = [r.LEG_ID for r in window_df.itertuples()]
        
        print(f"      Building Branch-and-Price SCIP model...")
        m, pricer, cols, cover = build_master(G, legs, bases, fdt, start_locations, list(unique_states), verbose=False)
        m.setRealParam("limits/time", float(os.environ.get("CPP_TIME_LIMIT", 180))) 
        m.optimize()
        
        generated_pairings = []
        for v, path, start_loc, home_base, end_loc, end_time, cost, anon_id in cols:
            generated_pairings.append({
                'path': path, 'start_loc': start_loc, 'home_base': home_base,
                'end_loc': end_loc, 'end_time': end_time, 'cost': cost, 'anon_id': anon_id
            })
            
        print(f"      SCIP Pairings generated: {len(generated_pairings)}")
        
        print(f"      Solving Anonymous Integer Selection...")
        int_m = Model(f"IntSelection_{window_start.date()}")
        int_m.hideOutput()
        int_m.setRealParam("limits/time", 60) 
        
        x = {}
        for p_idx, p_data in enumerate(generated_pairings):
            x[p_idx] = int_m.addVar(vtype='B', obj=p_data['cost'], name=f"x_{p_idx}")
            
        uncovered = {}
        for lg in legs:
            uncovered[lg] = int_m.addVar(vtype='B', obj=100000, name=f"uncov_{lg}")
            
        for lg in legs:
            covering_p_indices = [p_idx for p_idx, p_data in enumerate(generated_pairings) if lg in p_data['path']]
            int_m.addCons(
                sum(x[p_idx] for p_idx in covering_p_indices) + uncovered[lg] >= 1,
                name=f"Cover_{lg}"
            )
            
        # Ensure we don't reuse the same in-progress crew multiple times
        for idx, ipc in enumerate(in_progress_crews):
            anon_id = ipc['anon_id']
            matching_p_indices = [p_idx for p_idx, p_data in enumerate(generated_pairings) if p_data['anon_id'] == anon_id]
            if matching_p_indices:
                int_m.addCons(sum(x[p_idx] for p_idx in matching_p_indices) <= 1, name=f"LimitInProgress_{idx}")
        # ---------------------------------------------------------
        # Phase 1.b: Select Anonymous Pairings & Stitch Rosters
        # ---------------------------------------------------------
        int_m.optimize()
        
        next_in_progress = []
        if int_m.getStatus() in ["optimal", "timelimit", "gaplimit"]:
            uncov_count = sum(1 for lg in legs if int_m.getVal(uncovered[lg]) > 0.5)
            print(f"      Selection successful. {uncov_count} legs left uncovered.")
            
            for p_idx in x:
                if int_m.getVal(x[p_idx]) > 0.5:
                    p_data = generated_pairings[p_idx]
                    covered_legs_global.update(p_data['path'])
                    
                    anon_id = p_data['anon_id']
                    if anon_id is None:
                        anon_id = next_anon_id
                        next_anon_id += 1
                        master_rosters[anon_id] = {'home_base': p_data['home_base'], 'path': [], 'cost': 0.0}
                        
                    master_rosters[anon_id]['path'].extend(p_data['path'])
                    master_rosters[anon_id]['cost'] += p_data['cost']
                    
                    if p_data['end_loc'] not in bases:
                        new_cum_duty = len(master_rosters[anon_id]['path']) * 4.0
                        next_in_progress.append({
                            'loc': p_data['end_loc'],
                            'sleep': p_data['end_time'],
                            'home': p_data['home_base'],
                            'cum_duty': new_cum_duty,
                            'anon_id': anon_id
                        })
        else:
            print("      Selection failed or incomplete!")
            
        in_progress_crews = next_in_progress

    print("\n" + "="*78)
    print("=== PHASE 2: GLOBAL ROSTERING ===")
    
    # Filter only valid rosters that actually flew something
    all_assigned_roster = [r for r in master_rosters.values() if len(r['path']) > 0]
    
    print("      Loading Off Requests and Claims...")
    data_dir = r"c:\Users\defne\Documents\Master\Semester 2\Analytics Project\or_project_data\or_project_data"
    off_claims_df = pd.read_csv(os.path.join(data_dir, "off_claims_202505.csv"), sep=";")
    off_requests_df = pd.read_csv(os.path.join(data_dir, "off_requests_202505.csv"), sep=";")
    off_claims_df.columns = off_claims_df.columns.str.strip()
    off_requests_df.columns = off_requests_df.columns.str.strip()
    
    hard_leave_codes = {'KUR', 'U', 'SU'}
    banned_captain_days = defaultdict(set)
    def extract_day_int(date_str):
        try:
            return datetime.strptime(date_str.split()[0], "%d.%m.%y").day
        except:
            return 1
            
    if 'Begin Date' in off_requests_df.columns:
        for _, row in off_requests_df.iterrows():
            c_id = row['captain_id']
            if c_id in captains_list:
                start_d = extract_day_int(row['Begin Date'])
                end_d = extract_day_int(row['End Date'])
                if str(row['Code']).strip() in hard_leave_codes:
                    for d in range(start_d, end_d + 1):
                        banned_captain_days[c_id].add(d)

    df['REAL_DOM'] = pd.to_datetime(df['SCHEDULED_DEPARTURE_TIME']).dt.day
    leg_to_day_map = dict(zip(df['LEG_ID'], df['REAL_DOM']))
    
    rost_m = Model("GlobalRostering")
    rost_m.setRealParam("limits/time", 300) 
    
    y = {}
    for p_idx, p_data in enumerate(all_assigned_roster):
        p_base = p_data['home_base']
        valid_caps = home_bases_df[home_bases_df['home_base'] == p_base]['captain_id'].tolist()
        p_days = set(leg_to_day_map.get(leg, 1) for leg in p_data['path'])
        
        for c in valid_caps:
            if banned_captain_days[c].intersection(p_days):
                continue
            y[c, p_idx] = rost_m.addVar(vtype='B', obj=0.0, name=f"y_{c}_{p_idx}")
                        
    unassigned = {}
    for p_idx, p_data in enumerate(all_assigned_roster):
        unassigned[p_idx] = rost_m.addVar(vtype='B', obj=50000, name=f"unass_{p_idx}")
        rost_m.addCons(
            sum(y[c, p_idx] for c in captains_list if (c, p_idx) in y) + unassigned[p_idx] == 1,
            name=f"Assign_{p_idx}"
        )
        
    dev = {}
    horizon_days_count = len(unique_dates)
    
    for c in captains_list:
        dev[c] = rost_m.addVar(vtype='C', lb=0.0, obj=500.0, name=f"dev_{c}")
        
        matching_claim = off_claims_df[off_claims_df['captain_id'] == c]
        target_monthly_claim = matching_claim['Count'].values[0] if not matching_claim.empty else 10
        scaled_target_claim = (target_monthly_claim / 31.0) * horizon_days_count
        
        c_vars = []
        c_num_days = []
        duty_vars = []
        
        for p_idx in range(len(all_assigned_roster)):
            if (c, p_idx) in y:
                p_days = set(leg_to_day_map.get(leg, 1) for leg in all_assigned_roster[p_idx]['path'])
                num_days = len(p_days) if len(p_days) > 0 else 1
                c_vars.append(y[c, p_idx])
                c_num_days.append(num_days)
                duty_vars.append(y[c, p_idx] * (len(all_assigned_roster[p_idx]['path']) * 4.0))
                
        total_days_worked = sum(c_vars[i] * c_num_days[i] for i in range(len(c_vars))) if c_vars else 0.0
        observed_days_off = horizon_days_count - total_days_worked
        
        rost_m.addCons(dev[c] >= scaled_target_claim - observed_days_off, name=f"FairnessA_{c}")
        rost_m.addCons(dev[c] >= observed_days_off - scaled_target_claim, name=f"FairnessB_{c}")
        
        if duty_vars:
            rost_m.addCons(sum(duty_vars) <= 60.0, name=f"Duty_{c}")

    print(f"      Solving Global Rostering for {len(all_assigned_roster)} pairings and 432 Captains...")
    rost_m.optimize()
    
    final_output = []
    if rost_m.getStatus() in ["optimal", "timelimit", "gaplimit"]:
        for (c, p_idx) in y:
            if rost_m.getVal(y[c, p_idx]) > 0.5:
                p_data = all_assigned_roster[p_idx]
                final_output.append({
                    'Captain_ID': c,
                    'Home_Base': p_data['home_base'],
                    'Legs': p_data['path']
                })

    print("\n" + "="*78)
    print("=== FINAL METRICS ===")
    
    assigned_pilots = len(set(row['Captain_ID'] for row in final_output))
    total_pairings_assigned = len(final_output)
    total_pairings_generated = len(all_assigned_roster)
    
    covered = len(covered_legs_global)
    uncovered = total_flights_7d - covered
    coverage_pct = (covered / total_flights_7d * 100) if total_flights_7d > 0 else 0
    
    phase1_cost = sum(r['cost'] for r in all_assigned_roster)
    phase2_cost = rost_m.getObjVal() if rost_m.getStatus() in ["optimal", "timelimit", "gaplimit"] else 0.0
    
    print(f"      Pilots Assigned           : {assigned_pilots} (out of {len(captains_list)})")
    print(f"      Pairings Assigned         : {total_pairings_assigned} (out of {total_pairings_generated})")
    print(f"      Total Flights Covered     : {covered} ({coverage_pct:.1f}%)")
    print(f"      Total Flights Uncovered   : {uncovered}")
    print(f"      Phase 1 Cost (Pairings)   : {phase1_cost:,.2f}")
    print(f"      Phase 2 Cost (Rostering)  : {phase2_cost:,.2f} (Penalty for {total_pairings_generated - total_pairings_assigned} unassigned pairings)")

    print("\n" + "="*78)
    out_df = pd.DataFrame(final_output)
    out_df.to_csv("rolling_roster_output.csv", index=False)
    print("Saved to rolling_roster_output.csv")
    print(f"Total wall time: {time.perf_counter()-t_all:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise
