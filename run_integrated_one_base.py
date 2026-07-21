"""
INTEGRATED PAIRING + ROSTERING  

Stage 1 : branch-and-price  -> optimal set of crew pairings for the base
Stage 2 : assignment MIP     -> pairings assigned to NAMED captains


USAGE
    CPP_BASE=DUS CPP_START=2025-05-01 CPP_END=2025-05-02 python run_integrated.py
    (add CPP_EXPORT=1 to write the roster CSVs)

WINDOW SCALING
    The rostering MIP is small and fast; the binding stage is PAIRING. Same
    scaling story as before: 2 days is easy, longer windows may need the pairing
    time limit raised. Set CPP_PAIR_TIME (seconds) and CPP_ROSTER_TIME.
"""
import os, json, time
import pandas as pd

import stage1_pairing_one_base as s1
import stage2_rostering_one_base as s2

#This function runs the integrated pairing and rostering process for a given base, start and end date, pair_time, roster_time and export flag. 
# It first calls the solve_pairings function from stage1_pairing_one_base.py to generate the optimal set of crew pairings for the base and 
# then calls the solve_rostering function from stage2_rostering_one_base.py to assign the pairings to named captains. The results are printed to the console and optionally exported to CSV files.
def run(base, start, end, pair_time=600, roster_time=300, export=False):
    t0 = time.perf_counter()
    print("=" * 78)
    print(f"INTEGRATED PAIRING + ROSTERING  --  {base}  {start}..{end}")
    print("=" * 78)

    #  STAGE 1: PAIRING 
    print("\n[Stage 1] Pairing (branch-and-price) ...")
    pairings, pmeta = s1.solve_pairings(base, start, end, time_limit=pair_time)
    print(f"          -> {pmeta['n_pairings']} pairings, "
          f"EUR {pmeta['crew_cost']:,.0f}, "
          f"{'OPTIMAL' if pmeta['converged'] else pmeta['status']}")
    
    # save pairings immediately, so a later rostering failure doesn't lose them
    if export:
        stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
        pd.DataFrame([{
            "pairing_id": p['pairing_id'], "base": base, "n_legs": p['n_legs'],
            "cost_eur": p['cost'], "start_min": p['start_min'],
            "end_min": p['end_min'], "legs": "|".join(p['legs']),
        } for p in pairings]).to_csv(f"pairings_{base}_{stamp}.csv", index=False)
        print(f"  [saved] pairings_{base}_{stamp}.csv ({len(pairings)} rows)")

    # STAGE 2: ROSTERING 
    print("\n[Stage 2] Rostering (assignment MIP) ...")
    assign, cap_rows, rmeta = s2.solve_rostering(
        pairings, base, start, end, time_limit=roster_time, verbose=False)

    horizon = rmeta['horizon_days']
    active = [c for c in cap_rows if c['pairings'] > 0]
    idle   = [c for c in cap_rows if c['pairings'] == 0]

    # off-claims KPI: mean absolute deviation from target
    devs = [abs(c['days_off'] - c['off_target']) for c in cap_rows]
    mad = sum(devs) / len(devs) if devs else 0.0
    perfect = sum(1 for d in devs if d == 0)

    # soft off-request honouring
    soft_total = soft_hit = 0
    for c in cap_rows:
        cid = c['captain_id']
    # recompute request honouring from the solution
    soft_req_days = sum(c['soft_req_days'] for c in cap_rows)

    print(f"          -> status {rmeta['status']}, {rmeta['secs']}s")
    print("\n" + "=" * 78)
    print("RESULTS")
    print("=" * 78)
    print(f"  PAIRING")
    print(f"    pairings generated     : {pmeta['n_pairings']}")
    print(f"    crew cost              : EUR {pmeta['crew_cost']:,.2f}")
    print(f"    legs covered           : {pmeta['legs_covered']}/{pmeta['legs_reachable']} "
          f"reachable")
    print(f"    optimality gap         : {pmeta.get('gap_pct', 'N/A')}%")
    print(f"    ryan-foster branches   : {pmeta.get('ryan_foster_branches', 0)}")
    print(f"  ROSTERING")
    print(f"    captains at {base}         : {rmeta['captains']}")
    print(f"    assigned to line captains : {rmeta['assigned_line']}/{pmeta['n_pairings']}")
    print(f"    covered by RESERVE pool   : {rmeta['reserve_used']}/{rmeta['reserve_budget']} "
          f"(pro-rated from monthly standby)")
    print(f"    still UNCOVERED           : {rmeta['unassigned']}"
          f"   <-- gap beyond line + reserve capacity")
    print(f"    captains used          : {len(active)}/{rmeta['captains']}")
    print(f"    off-claims MAD         : {mad:.2f} days   "
          f"({perfect}/{len(cap_rows)} captains hit target exactly)")
    print(f"    soft off-request days  : {soft_req_days} requested")

    if rmeta['unassigned'] > 0:
        full_avail = sum(1 for c in cap_rows if c['hard_off_days'] == 0)
        print(f"\n  NOTE ON UNCOVERED PAIRINGS")
        print(f"    {rmeta['unassigned']} pairings remain uncovered even after using the")
        print(f"    full reserve budget ({rmeta['reserve_used']} shifts). Most pairings span")
        print(f"    the whole {horizon}-day window; only {full_avail}/{rmeta['captains']} captains are")
        print(f"    fully available, and the pro-rated reserve pool covers "
              f"{rmeta['reserve_budget']} more.")
        print(f"    The residual {rmeta['unassigned']} is a genuine crew-capacity shortfall for")
        print(f"    this window -- it would require additional reserves or schedule change.")

    print(f"\n  wall time: {time.perf_counter()-t0:.1f}s")

    # EXPORT of files
    if export:
        stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
        pid2legs = {p['pairing_id']: p for p in pairings}

        roster_rows = []
        for pid, cid in assign.items():
            p = pid2legs[pid]
            roster_rows.append({
                "pairing_id":  pid,
                "captain_id":  cid if cid is not None else "UNASSIGNED",
                "base":        base,
                "n_legs":      p['n_legs'],
                "cost_eur":    p['cost'],
                "start_min":   p['start_min'],
                "end_min":     p['end_min'],
                "legs":        "|".join(p['legs']),
            })
        pd.DataFrame(roster_rows).to_csv(f"roster_{base}_{stamp}.csv", index=False)
        pd.DataFrame(cap_rows).to_csv(f"captains_{base}_{stamp}.csv", index=False)
        pd.DataFrame([{**pmeta, **{f"roster_{k}": v for k, v in rmeta.items()}}]) \
          .to_csv(f"summary_{base}_{stamp}.csv", index=False)
        print(f"\n  Exported: roster_{base}_{stamp}.csv, captains_{base}_{stamp}.csv, "
              f"summary_{base}_{stamp}.csv")

    return pairings, assign, cap_rows, pmeta, rmeta


if __name__ == "__main__":
    base  = os.environ.get("CPP_BASE", "DUS")
    start = os.environ.get("CPP_START", "2025-05-01")
    end   = os.environ.get("CPP_END", "2025-05-03")
    run(base, start, end,
        pair_time=float(os.environ.get("CPP_PAIR_TIME", 9000)),
        roster_time=float(os.environ.get("CPP_ROSTER_TIME", 300)),
        export=bool(os.environ.get("CPP_EXPORT"))) #calls the intergrated pairing and rostering function with the base, start and end date, pair_time, roster_time and export flag
