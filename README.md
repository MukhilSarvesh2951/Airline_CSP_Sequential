# Airline Crew Scheduling — Sequential Pairing and Rostering

Operations Research Practice Project, RWTH Aachen Business School (WS 2025/26), in
collaboration with **Lufthansa Industry Solutions AS GmbH** / Eurowings.

The project solves the **Airline Crew Scheduling Problem** in the standard two-stage
industry decomposition:

1. **Crew Pairing** — build minimum-cost, legally feasible sequences of flight legs
   ("pairings") that start and end at a crew base. Solved by **branch-and-price**:
   a Set Partitioning master problem, with columns generated on demand by a
   Resource-Constrained Shortest Path (RCSPP) label-setting pricer over a NetworkX
   flight network, enforcing EASA Flight Duty Time limits.
2. **Crew Rostering** — assign the resulting pairings to *named* captains, respecting
   rest rules, absences, off-day claims and a finite standby/reserve pool. Solved as
   an assignment MIP.

Both stages use **SCIP** via `PySCIPOpt`.

---

##  Data licence — read before sharing

> The data is intended for use **only** within the scope of the Operations Research
> Practice Project in Winter Semester 2025/26 and must not be shared outside of it.
> Any use of the data outside this project requires **written permission from
> Lufthansa Industry Solutions AS GmbH**.
>
> Contacts: Bartlomiej Jezierski <bartlomiej.jezierski@lhind.dlh.de>,
> Joseph Doetsch <joseph.doetsch@lhind.dlh.de>

**This repository must remain private** while it contains the `.csv` inputs below.
If it is ever made public, remove the data files first and add them to `.gitignore`.

---

## Repository layout

```
Airline_CSP/
├── crew_pairing_RF_all_bases.py        # main pairing solver (all bases, branch-and-price)
├── crew_pairing_rolling_sequential.py  # rolling-horizon variant (price-and-branch)
├── stage1_pairing_one_base.py          # single-base pairing (Stage 1 of the pipeline)
├── stage2_rostering_one_base.py        # single-base rostering MIP (Stage 2)
├── run_integrated_one_base.py          # driver: runs Stage 1 -> Stage 2
├── Rostering_2day_ALL_BASES.ipynb      # notebook: 2-day rostering, all bases
├── Rostering_1week_DUS.ipynb           # notebook: 1-week rostering, DUS
└── data/                               # LHIND input data (see below)
```

---

## Data description

All input files live in `data/` (path configurable via `CPP_DATA_DIR`).
Note the **mixed delimiters** — some files are comma-separated, some semicolon-separated.

| File | Sep | Rows | Description |
|---|---|---|---|
| `flight_schedule.csv` | `,` | 52,885 | All flights to be crewed. `LEG_ID`, departure/arrival airport, scheduled departure/arrival times, `AIRCRAFT_TYPE`. Times are **CET** (relevant for FDT limits). Solvers filter to the A320 family (`319`, `320`, `321`) and to the requested date window. |
| `home_bases.csv` | `,` | 432 | Every captain and their home base. Eight bases appear: BER, CGN, DUS, HAJ, HAM, MUC, NUE, STR. |
| `fdt_limits.csv` | `,` | 14 | Maximum **Flight Duty Time** in hours, as a function of duty start time (a time window) and number of legs in the duty (columns `1`…`10`). This is a step function of two variables — the reason automatic Dantzig-Wolfe decomposition (GCG) was unsuitable and the pricer was written by hand. |
| `ground_transportation_times.csv` | `,` | 62 | Average ground transfer duration (hours) between airport pairs. Enables deadheading by ground between nearby airports. |
| `codes.csv` | `;` | 7 | Absence code dictionary. **Hard** (captain unavailable): `KUR` cure/rehab, `U` vacation, `SU` special leave. **Soft** (off *requests*, not entitlements, but should be granted at a high rate): `O_L`, `O_M`, `O_TX`, `O_TZ`, `O_U`. |
| `off_requests_202505.csv` (also `202506`, `202507`) | `;` | 2,339 | Per-captain absence and off-request intervals with begin/end date and code. Times are **UTC**. |
| `off_claims_202505.csv` (also `202506`, `202507`) | `;` | 403 | Number of off-days each captain is entitled to that month. Days lost to `KUR`/`U`/`SU` do **not** count toward this. The deviation between claims and actual assigned off-days is a project KPI, minimised in Stage 2. |
| `standby.csv` | `;` | 5 | Monthly reserve-shift count per base (12 h per shift). Stage 2 pro-rates this to the rostering window to size the reserve pool. Note only 5 bases are listed, not all 8. |

---

## Python files

### `crew_pairing_RF_all_bases.py` — main pairing solver

Solves crew pairing for **all bases simultaneously** in one shared master problem.
This is the primary solver and the recommended entry point.

- **Master problem: Set Partitioning (SPP)** — each leg covered *exactly* once:
  ```python
  c = m.addCons(a == 1.0, name=f"cover_{lg}", ...)
  ```
  **To switch to Set Covering (SCP)**, change `== 1.0` to `>= 1.0` on that line.
  SCP permits a leg to be covered by more than one pairing, which relaxes the model
  and usually makes it easier to solve, at the cost of allowing redundant coverage.
  This is not purely cosmetic: SPP is the tighter formulation, and the artificial
  (Big-M) columns and `pricerfarkas` behaviour should be re-checked when switching,
  since feasibility recovery differs between the two senses.
- **Pricer**: RCSPP label-setting over a NetworkX DiGraph, with dominance pruning
  per `(node, n_duties)` bucket and EASA FDT checked at every label extension.
  Uses a two-phase strategy: a fast heuristic pass, falling back to an exact pass
  only when the fast pass finds no improving column.
- **Variables**: `vtype='B'` — genuine branch-and-price. (An earlier `vtype='C'`
  version solved only the LP relaxation; see *Known issues* below.)
- **Branching**: Ryan-Foster pairwise branching is implemented (constraint handler +
  branch rule + pricer-side edge enforcement) and gated behind the `RYAN_FOSTER`
  flag. When disabled, SCIP's default variable branching is used.
- **Column pool management**: columns idle at zero LP value for `CPP_STALE_LIMIT`
  consecutive pricing rounds are fixed to `ub=0` to stop the LP growing without
  bound on long runs.

### `crew_pairing_rolling_sequential.py` — rolling-horizon variant

Splits a long planning window into overlapping slices solved in sequence, carrying
partially-built pairings across slice boundaries. Uses `vtype='C'` during column
generation and then re-solves as a binary integer program over the frozen column pool
(**price-and-branch**), rather than pricing inside the branch-and-bound tree.
Faster than full branch-and-price, but heuristic: no columns are generated after the
pool is frozen.

`DATA_DIR` in this file is hardcoded to a different machine's path — update before running.

### `stage1_pairing_one_base.py` — Stage 1 (single base)

Same modelling approach as the main solver, restricted to one base's reachable legs.
Set Partitioning (`== 1.0`), `vtype='B'`, 
Importable as `solve_pairings(base, start, end) -> (pairings, meta)`.

### `stage2_rostering_one_base.py` — Stage 2 (single base)

Assignment MIP mapping Stage-1 pairings to named captains.

- `x[p,c] = 1` if captain `c` flies pairing `p` (created only where eligible)
- `res[p] = 1` if the pairing is absorbed by the **reserve/standby pool**
- `un[p] = 1` if the pairing is left uncovered (heavily penalised slack)

**Hard constraints:** every pairing goes to exactly one line captain, a reserve, or
uncovered; a captain's pairings may not overlap (span + 12 h rest); captains with a
hard absence (`KUR`/`U`/`SU`) on any day a pairing touches are excluded; total reserve
usage is capped by the pro-rated standby budget.

**Objective (minimise):** off-claim deviation `|days_off − target|`, soft off-request
violations, reserve usage, and uncovered pairings — weighted so the solver prefers
line captains → reserves → uncovered, in that order.

### `run_integrated_one_base.py` — pipeline driver

Runs Stage 1 then feeds its real output into Stage 2, printing a combined KPI summary
and optionally exporting CSVs.



### Notebooks

- `Rostering_2day_ALL_BASES.ipynb` — 2-day rostering across all bases
- `Rostering_1week_DUS.ipynb` — 1-week rostering for DUS

---

## Usage

Requirements: `pyscipopt`, `pandas`, `numpy`, `networkx`.

```bash
pip install pyscipopt pandas numpy networkx
```

All configuration is via environment variables. PowerShell:

```powershell
$env:CPP_DATA_DIR = ".\data"
$env:CPP_START    = "2025-05-01"
$env:CPP_END      = "2025-05-02"
$env:CPP_TIME_LIMIT = 1000
$env:CPP_EXPORT   = 1
python crew_pairing_RF_all_bases.py
```

Bash:

```bash
CPP_DATA_DIR=./data CPP_START=2025-05-01 CPP_END=2025-05-02 \
CPP_TIME_LIMIT=900 CPP_EXPORT=1 python crew_pairing_RF_all_bases.py
```

### Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `CPP_DATA_DIR` | `./data` | Directory holding the input CSVs |
| `CPP_START` / `CPP_END` | `2025-04-27` / `2025-04-28` | Planning window (inclusive) |
| `CPP_TIME_LIMIT` | `7000` | SCIP wall-clock limit (seconds) |
| `CPP_EXPORT` | unset | Set to `1` to write result CSVs |
| `CPP_STALE_LIMIT` | `3000` | Rounds a column may sit at zero before being pruned. High = effectively off. **Lower only for long runs (many hundreds of rounds)** — pruning too early makes the pricer regenerate columns it just discarded and can prevent convergence. |
| `CPP_COLS_PER_ROUND` | `400` | Max columns injected per pricing call |
| `CPP_COLS_PER_BASE` | `60` | Negative-reduced-cost columns kept per base |
| `CPP_MAX_LABELS` | `30`–`40` | Dominance bucket size in the RCSPP |
| `CPP_RYAN_FOSTER` | unset | Set to `1` to enable Ryan-Foster branching |
| `CPP_BASE` | `DUS` | Base to solve (single-base pipeline only) |
| `CPP_PAIR_TIME` / `CPP_ROSTER_TIME` | `9000` / `300` | Per-stage time limits in the pipeline |

---

## Results

Written to the working directory when `CPP_EXPORT=1`. Files are timestamped
(`YYYYMMDD_HHMM`) so successive runs don't overwrite each other.

| File | Contents |
|---|---|
| `pairings_ALL_<start>_<end>_<stamp>.csv` | Every selected pairing, all bases: `pairing_id`, `base`, `n_legs`, `cost`, `start_min`, `end_min`, `route`, `legs` |
| `pairings_<BASE>_..._<stamp>.csv` | One file per base, `pairing_id` renumbered from 1. Column layout matches what `stage2_rostering.solve_rostering()` expects, so it can be fed straight into rostering |
| `uncovered_legs_<start>_<end>_<stamp>.csv` | Uncovered legs with airports/times and a `reason` column: `structurally_impossible` (no legal SRC→leg→SNK path exists) vs `not_selected` |
| `summary_<start>_<end>_<stamp>.csv` | One row per run: status, pricing rounds, columns generated/pruned/active, B&B nodes, coverage %, crew cost, wall time |
| `roster_<BASE>_<stamp>.csv` | Stage 2: one row per pairing with its assigned `captain_id`, or `RESERVE`, or `UNASSIGNED` |
| `captains_<BASE>_<stamp>.csv` | Per-captain summary: pairings flown, days worked/off, off-target, absence days |

`start_min` / `end_min` are **minutes from midnight on `CPP_START`**, not clock times.

### Reading the coverage numbers

Two figures are reported, and the second is the meaningful one:

- *coverage of all legs* — includes legs no legal pairing can ever cover
- *coverage of COVERABLE legs* — excludes structurally impossible legs

A leg is **structurally impossible** when the network contains no path
`SRC_base → leg → SNK_base` under the MCT / rest / FDT rules — typically a late leg
at an outstation with no way home inside the window. These are absorbed by artificial
columns and are a property of the data and window length, not a solver failure.
They largely disappear as the window lengthens.


### Reference result — 2025-05-01 to 2025-05-02, all bases

| Metric | Value |
|---|---|
| Legs in window (A320 family) | 861 |
| Structurally uncoverable | 112 |
| Coverage of coverable legs | **747 / 749 (99.7 %)** |
| Pairings selected | 180 |
| True crew cost | **EUR 171,421.60** |
| Optimality gap | **0.0019 %** |
| Pricing rounds / columns | 359 / 11,129 |
| Status | `timelimit` at 900 s |

`timelimit` here is cosmetic: at a 0.0019 % gap the solution is optimal for practical
purposes; the remaining budget was spent trying to *prove* the last sliver. Set
`limits/gap` if you would prefer a clean early exit.

---


