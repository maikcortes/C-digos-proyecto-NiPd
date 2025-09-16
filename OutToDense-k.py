# -*- coding: utf-8 -*-
"""
promote_best_Tx_to_scf.py — v1.3
• Picks best among *_T1 / *_T2 (most negative H, then E).
• Uses optimized coordinates from the final coordinates block in OUT.
• Copies per-atom flags VERBATIM (only when present on that line).
• Copies CELL_PARAMETERS verbatim from the matching source *.in (same stem) or from the .out.
• Writes SCF inputs/jobs named strictly as: <System>_<Plane>.{in,lsf,out}
  - System = first token in original name, normalized to explicit counts (NiPd3 → Ni1Pd3, Pd → Pd1).
  - Plane  = first 3-digit block in the name (e.g., 200 / 111 / 220). Files without a plane are skipped.

Requires: pip install ase
"""

from pathlib import Path
import re
from typing import Dict, Tuple, List, Optional
from collections import defaultdict
import numpy as np
from ase.io import read

# ───────────── User settings ─────────────
SRC_DIR  = Path(r"G:\My Drive\Work\UNAM\Doctorado\Proyecto\Resultados\Nanoparticles\QE\Supercell\SurfaceEnergy\311_1x1\relax")
DEST_DIR = Path(r"G:\My Drive\Work\UNAM\Doctorado\Proyecto\Resultados\Nanoparticles\QE\Supercell\SurfaceEnergy\311_1x1\SCF")

KPOINTS_FINE: Tuple[int,int,int] = (20, 10, 1)

PSEUDO_DIR  = r"/tmpu/mdach_g/mdach/PP"
PSEUDO: Dict[str, str] = {"Ni": "Ni.upf", "Pd": "Pd.upf"}

QUEUE_NAME = "q_hpc"
NCORES     = 24
USER_EMAIL = "ccuan@pceim.unam.mx"
PW_MODULES = [
    "tbb/2021.6.0",
    "compiler-rt/2022.1.0",
    "intel/2022.1.0",
    "mpi/intel-2021.6.0",
    "mkl/2022.1.0",
    "quantum/7.3",
]
MPI_LAUNCH = "mpirun -np {n} pw.x -inp {inp} > {out}"

END_OK = "JOB DONE"

# QE physics — hard-coded to SCF
CALCULATION_QE  = "scf"
NSTEP_QE        = 512
MAX_SECONDS_QE  = "8.64000e+99"
OUTDIR_QE       = "."
RESTART_MODE_QE = "from_scratch"

DEGAUSS_QE       = "0.0146997171"
IBRAV_QE         = 0
NSPIN_QE         = 2
NOSYM_QE         = ".true."
OCCUPATIONS_QE   = "smearing"
SMEARING_QE      = "methfessel-paxton"
START_MAG_QE     = "1"
VDW_CORR_QE      = "DFT-D3"
DFTD3_VERSION_QE = 6
ECUTRHO_QE       = "4.80000e+02"
ECUTWFC_QE       = "6.00000e+01"

# electrons
CONV_THR_ELEC     = "6.40000e-09"
DIAGO_DAVID_NDIM  = 12
DIAGONALIZATION   = "david"
ELECTRON_MAXSTEP  = 512
MIXING_BETA       = "0.2"
MIXING_MODE       = "local-TF"
MIXING_NDIM       = 12
STARTINGPOT       = "atomic"
STARTINGWFC       = "atomic"

# ───────────── Regex helpers ─────────────
TOTAL_RES = [
    re.compile(r"!\s*total\s+energy\s*=\s*(-?\d+\.\d+)", re.I),
    re.compile(r"\btotal\s+energy\s*=\s*(-?\d+\.\d+)", re.I),
    re.compile(r"\bEtot\s*=\s*(-?\d+\.\d+)", re.I),
]
ENTH_RES  = [
    re.compile(r"\bfinal\s*enthalpy\s*=\s*(-?\d+\.\d+)", re.I),
    re.compile(r"\benthalpy\s*new\s*=\s*(-?\d+\.\d+)", re.I),
]
RX_Tx = re.compile(r"(.*)_T[12](?=(_|$))", re.I)

RX_BEGIN = re.compile(r"^\s*Begin\s+final\s+coordinates\s*$", re.I | re.M)
RX_END   = re.compile(r"^\s*End\s+final\s+coordinates\s*$", re.I | re.M)

# ATOMIC_POSITIONS: accept {angstrom} / (angstrom) / none
RX_ATPOS_HDR = re.compile(r"^\s*ATOMIC_POSITIONS(?:\s*(?:\{([^}]*)\}|\(([^)]*)\)))?\s*$", re.I | re.M)
RX_ATOM = re.compile(
    r"^\s*([A-Za-z][A-Za-z0-9]*)\s+"
    r"([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)"
    r"(?:\s+([01])\s+([01])\s+([01]))?\s*$"
)

# CELL_PARAMETERS header
RX_CELL_HDR_ANY = re.compile(r"^\s*CELL_PARAMETERS(?:\s*(?:\{[^}]*\}|\([^)]*\)))?\s*$", re.I | re.M)
RX_VEC_LINE = re.compile(r"^\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*$")

# Optional &SYSTEM flags mirrored if present in OUT
RX_CONSTR_MAGN = re.compile(r"\bconstrained_magnetization\s*=\s*([\"']?[^\"'\s]+[\"']?)", re.I)

SECTION_STARTERS = re.compile(
    r"^\s*(CELL_PARAMETERS|K_POINTS|Begin\s+final\s+coordinates|End\s+final\s+coordinates|&\w+|ATOMIC_SPECIES|ATOMIC_POSITIONS)\b",
    re.I | re.M
)

# ───────────── Naming helpers ─────────────
RX_PLANE = re.compile(r"(^|_)(\d{3})(_|$)")
RX_ELEMCOUNT = re.compile(r"([A-Z][a-z]?)(\d*)")

def normalize_system_token(token: str) -> str:
    """Make 'Pd' -> 'Pd1', 'NiPd3' -> 'Ni1Pd3', 'Ni2Pd2' -> unchanged."""
    parts = RX_ELEMCOUNT.findall(token)
    if not parts:
        return token
    out = []
    for el, num in parts:
        out.append(el + (num if num else "1"))
    return "".join(out)

def canonical_base_from_stem(stem: str) -> Optional[str]:
    """
    From an original stem like:
      'NiPd3_Ni1_slab_200_T1_relax' or 'Pd_CeldaUnitaria_slab_200_T2_relax'
    produce 'Ni1Pd3_200' or 'Pd1_200'.
    """
    # System = first token before first underscore
    first_tok = stem.split("_", 1)[0]
    system = normalize_system_token(first_tok)

    # Plane = first 3-digit run anywhere in name
    m = RX_PLANE.search(stem)
    if not m:
        return None
    plane = m.group(2)

    return f"{system}_{plane}"

# ───────────── Core helpers ─────────────
def ensure_dirs() -> None:
    DEST_DIR.mkdir(parents=True, exist_ok=True)

def normalize_cell_block(cell_block: Optional[List[str]]) -> Optional[List[str]]:
    """Ensure: ['CELL_PARAMETERS {...}', '  v1', '  v2', '  v3']."""
    if not cell_block:
        return None
    block = [ln.rstrip() for ln in cell_block if ln.strip()]
    if not block:
        return None
    if not block[0].strip().lower().startswith("cell_parameters"):
        vecs = [ln for ln in block if RX_VEC_LINE.match(ln)][:3]
        if len(vecs) != 3: return None
        return ["CELL_PARAMETERS {angstrom}"] + vecs
    header = block[0]
    vecs = [ln for ln in block[1:] if RX_VEC_LINE.match(ln)][:3]
    if len(vecs) != 3: return None
    return [header] + vecs

def energetics(outf: Path):
    txt = outf.read_text(errors="ignore")
    finished = (END_OK in txt)
    def _last(res):
        for rx in res:
            vals = [float(x) for x in rx.findall(txt)]
            if vals: return vals[-1]
        return None
    return (_last(TOTAL_RES), _last(ENTH_RES), finished)

def starting_magn(outf: Path) -> Dict[str, float]:
    txt = outf.read_text(errors="ignore")
    species, magn = [], {}
    if "ATOMIC_SPECIES" in txt:
        after = txt.split("ATOMIC_SPECIES", 1)[1]
        for l in after.splitlines()[1:]:
            if not l.strip(): break
            parts = l.split()
            if parts: species.append(parts[0])
    for idx, val in re.findall(r"starting_magnetization\((\d+)\)\s*=\s*([0-9.eE+-]+)", txt):
        i = int(idx) - 1
        if 0 <= i < len(species):
            try: magn[species[i]] = float(val)
            except ValueError: pass
    return magn

def read_cell_block_verbatim(src_in: Path) -> Optional[List[str]]:
    """Return LAST CELL_PARAMETERS block from a source *.in."""
    if not src_in.exists(): return None
    txt = src_in.read_text(encoding="utf-8", errors="ignore")
    matches = list(RX_CELL_HDR_ANY.finditer(txt))
    if not matches: return None
    m = matches[-1]
    header_line = txt[m.start():m.end()].splitlines()[0].rstrip()
    vec_lines: List[str] = []
    for ln in txt[m.end():].splitlines():
        if not ln.strip():
            if len(vec_lines) >= 3: break
            continue
        if RX_VEC_LINE.match(ln):
            vec_lines.append(ln.rstrip())
            if len(vec_lines) == 3: break
        else:
            if SECTION_STARTERS.match(ln): break
    if len(vec_lines) != 3: return None
    return [header_line] + vec_lines

def read_cell_block_from_out(outf: Path) -> Optional[List[str]]:
    """Return LAST CELL_PARAMETERS block from a QE *.out (verbatim)."""
    if not outf.exists(): return None
    txt = outf.read_text(encoding="utf-8", errors="ignore")
    matches = list(RX_CELL_HDR_ANY.finditer(txt))
    if not matches: return None
    m = matches[-1]
    header_line = txt[m.start():m.end()].splitlines()[0].rstrip()
    vec_lines: List[str] = []
    for ln in txt[m.end():].splitlines():
        if not ln.strip():
            if len(vec_lines) >= 3: break
            continue
        if RX_VEC_LINE.match(ln):
            vec_lines.append(ln.rstrip())
            if len(vec_lines) == 3: break
        else:
            if SECTION_STARTERS.match(ln): break
    if len(vec_lines) != 3: return None
    return [header_line] + vec_lines

def extra_system_flags(outf: Path) -> Dict[str,str]:
    txt = outf.read_text(errors="ignore")
    flags = {}
    m = RX_CONSTR_MAGN.search(txt)
    if m: flags["constrained_magnetization"] = m.group(1)
    return flags

def tgroup_key(stem: str) -> str:
    return RX_Tx.sub(r"\1_", stem)

def choose_best(records: List[Tuple[Path, Optional[float], Optional[float]]]) -> Optional[Path]:
    best, best_rank = None, None
    for p, E, H in records:
        rank = (float("inf") if H is None else H, float("inf") if E is None else E)
        if (best_rank is None) or (rank < best_rank):
            best_rank, best = rank, p
    return best

def _slice_final_region(txt: str) -> str:
    b = RX_BEGIN.search(txt); e = RX_END.search(txt)
    if b and e and e.start() > b.end(): return txt[b.end():e.start()]
    return txt

def _find_all_blocks(txt: str, header_rx) -> List[Tuple[str, List[str]]]:
    """Return list of (unit_tag, lines) for each block (ATOMIC_POSITIONS or CELL_PARAMETERS)."""
    blocks: List[Tuple[str, List[str]]] = []
    for m in header_rx.finditer(txt):
        unit = (m.group(1) or m.group(2) or "").strip().lower() if header_rx is RX_ATPOS_HDR else ""
        start = m.end()
        end_match = SECTION_STARTERS.search(txt, pos=start)
        end = end_match.start() if end_match else len(txt)
        body = txt[start:end]
        lines = [ln for ln in body.splitlines() if ln.strip()]
        blocks.append((unit, lines))
    return blocks

# --- positions/flags from OUT ---
def parse_final_positions_and_flags(outf: Path, nat_expect: int, cell_A: np.ndarray):
    """
    Returns (symbols, positions_cart_A, flags_tokens) from OUT.
    flags_tokens[i] is None (no flags) or a string like "0 0 0".
    """
    txt = outf.read_text(errors="ignore")
    region = _slice_final_region(txt)
    blocks = _find_all_blocks(region, RX_ATPOS_HDR) or _find_all_blocks(txt, RX_ATPOS_HDR)
    if not blocks: return None

    for unit, lines in reversed(blocks):
        symbols: List[str] = []
        coords: List[List[float]] = []
        flags_tokens: List[Optional[str]] = []
        for ln in lines:
            m = RX_ATOM.match(ln)
            if not m: continue
            sym = m.group(1)
            x, y, z = float(m.group(2)), float(m.group(3)), float(m.group(4))
            fx, fy, fz = m.group(5), m.group(6), m.group(7)
            flags = None if fx is None else f"{fx} {fy} {fz}"
            symbols.append(sym); coords.append([x,y,z]); flags_tokens.append(flags)
        if len(symbols) != nat_expect: continue

        unit = (unit or "angstrom").lower()
        pos_A = np.asarray(coords, float)
        if unit in ("angstrom", "angs", ""): pass
        elif unit == "bohr": pos_A *= 0.529177210903
        elif unit.startswith("crystal"): pos_A = pos_A @ cell_A
        elif unit == "alat": continue
        else: continue
        return symbols, pos_A, flags_tokens
    return None

# ───────────── Writers ─────────────
def write_pw_in(symbols: List[str],
                positions_A: np.ndarray,
                cell_A: np.ndarray,
                fn: Path,
                kpoints: Tuple[int,int,int],
                magn_map: Dict[str,float],
                pseudo_map: Dict[str,str],
                flags_tokens: Optional[List[Optional[str]]],
                extra_sys: Optional[Dict[str,str]] = None,
                cell_block_verbatim: Optional[List[str]] = None) -> None:
    """Write QE SCF input with verbatim flags and CELL_PARAMETERS if available."""
    def _norm_cell(block: Optional[List[str]]) -> Optional[List[str]]:
        return normalize_cell_block(block) if block else None

    # species in first-appearance order
    species: List[str] = []
    for s in symbols:
        if s not in species:
            species.append(s)

    from ase.data import atomic_masses, atomic_numbers
    masses = []
    for s in species:
        try: masses.append(float(atomic_masses[atomic_numbers[s]]))
        except Exception: masses.append(1.0)

    L: List[str] = []
    A = L.append

    # ---- CONTROL ----
    A("&CONTROL")
    A(f"  title           = '{fn.stem}'")
    A(f"  calculation     = '{CALCULATION_QE}'")
    A(f"  pseudo_dir      = '{PSEUDO_DIR}'")
    A(f"  prefix          = '{fn.stem}'")
    A(f"  outdir          = '{OUTDIR_QE}'")
    A(f"  restart_mode    = '{RESTART_MODE_QE}'")
    etot_thr_ry = len(symbols) * 7.3498645e-7
    A(f"  etot_conv_thr   = {format(etot_thr_ry, '.6e').replace('e','d')}")
    A("  forc_conv_thr   = 7.8d-4")
    A("/\n")

    # ---- SYSTEM ----
    A("&SYSTEM")
    if extra_sys:
        for k, v in extra_sys.items():
            A(f"    {k} = {v}")
    A(f"    degauss                   =  {DEGAUSS_QE}")
    A(f"    ibrav                     =  {IBRAV_QE}")
    A(f"    nat                       =  {len(symbols)}")
    A(f"    nspin                     =  {NSPIN_QE}")
    A(f"    ntyp                      =  {len({s for s in symbols})}")
    A(f"    nosym                     =  {NOSYM_QE}")
    A(f"    occupations               = \"{OCCUPATIONS_QE}\"")
    A(f"    smearing                  = \"{SMEARING_QE}\"")
    for i, s in enumerate(dict.fromkeys(symbols), 1):
        val = magn_map.get(s)
        A(f"    starting_magnetization({i}) =  {START_MAG_QE if val is None else f'{val:.6f}'}")
    A(f"    vdw_corr                  =  \"{VDW_CORR_QE}\"")
    A(f"    dftd3_version             =  {DFTD3_VERSION_QE}")
    A(f"    ecutrho                   =  {ECUTRHO_QE}")
    A(f"    ecutwfc                   =  {ECUTWFC_QE}")
    A("    input_dft                 = 'PBE'")
    A("    assume_isolated           = '2D'")
    A("    noinv                     = .true.")
    A("/\n")

    # ---- ELECTRONS ----
    A("&ELECTRONS")
    A(f"    conv_thr         =  {CONV_THR_ELEC}")
    A(f"    diagonalization  =  \"{DIAGONALIZATION}\"")
    A(f"    mixing_beta      =  {MIXING_BETA}")
    A(f"    mixing_mode      =  \"{MIXING_MODE}\"")
    A(f"    mixing_ndim      =  {MIXING_NDIM}")
    A("/\n")

    # ---- ATOMIC_SPECIES ----
    species = []
    for s in symbols:
        if s not in species:
            species.append(s)
    A("ATOMIC_SPECIES")
    for s in species:
        m = float(atomic_masses[atomic_numbers[s]]) if s in atomic_numbers else 1.0
        pseudo = pseudo_map.get(s, f"{s}.upf")
        A(f"  {s} {m:.4f} {pseudo}")
    A("")

    # ---- ATOMIC_POSITIONS ----
    A("ATOMIC_POSITIONS (angstrom)")
    for i, s in enumerate(symbols):
        x, y, z = positions_A[i]
        if flags_tokens and i < len(flags_tokens) and flags_tokens[i]:
            A(f"  {s:<2} {x: .10f} {y: .10f} {z: .10f} {flags_tokens[i]}")
        else:
            A(f"  {s:<2} {x: .10f} {y: .10f} {z: .10f}")
    A("")

    # ---- CELL_PARAMETERS ----
    nb = normalize_cell_block(cell_block_verbatim)
    if nb:
        for ln in nb: A(ln)
        A("")
    else:
        A("CELL_PARAMETERS {angstrom}")
        for v in cell_A:
            A(f"  {v[0]: .10f} {v[1]: .10f} {v[2]: .10f}")
        A("")

    # ---- K_POINTS ----
    A("K_POINTS {automatic}")
    A(f"  {kpoints[0]} {kpoints[1]} {kpoints[2]}  0 0 0")

    fn.write_text("\n".join(L), encoding="utf-8")


def write_lsf(lsf_path: Path, inp_name: str):
    # Bare job name; ptile=16
    stem_short = inp_name.split('/')[-1].split('\\')[-1].replace(".in","")
    job = f"{stem_short}"
    mods = "\n".join([f"module load {m}" for m in PW_MODULES])
    out_name = f"{stem_short}.out"
    payload = f"""#!/bin/bash
#BSUB -J {job}
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -q {QUEUE_NAME}
#BSUB -n {NCORES}
#BSUB -R "span[ptile={NCORES}]"
#BSUB -B
#BSUB -N
#BSUB -u {USER_EMAIL}
#

{mods}

{MPI_LAUNCH.format(n=NCORES, inp=inp_name, out=out_name)}
"""
    lsf_path.write_text(payload, encoding="utf-8")

def convert_to_unix(p: Path):
    p.write_text(p.read_text(encoding="utf-8"), newline="\n", encoding="utf-8")

# ───────────── Main ─────────────
def main():
    ensure_dirs()
    outs = sorted(SRC_DIR.glob("*.out"))
    print(f"[info] Found {len(outs)} OUT files in: {SRC_DIR}")

    # 1) group by T1/T2 siblings and pick best per group
    groups = defaultdict(list)
    for outf in outs:
        E, H, finished = energetics(outf)
        if not finished:
            print(f"[skip] {outf.name} (not finished)")
            continue
        stem = outf.stem
        key = tgroup_key(stem) if RX_Tx.search(stem) else stem
        groups[key].append((outf, E, H))

    chosen: List[Path] = []
    for key, recs in groups.items():
        pick = choose_best(recs)
        if pick:
            print(f"[pick] {key} -> {pick.name}")
            chosen.append(pick)

    # 2) build new inputs with canonical names
    n_ok = 0
    for outf in chosen:
        # Canonical short name: <System>_<Plane>
        canon = canonical_base_from_stem(outf.stem)
        if not canon:
            print(f"[warn] Cannot determine plane (###) in '{outf.name}' — skipping.")
            continue

        # Get nat + fallback cell from ASE (in case coords were CRYSTAL)
        try:
            at = read(outf, format="espresso-out", index=-1)
            nat_expect = len(at)
            cell_A_fallback = at.cell.array
        except Exception as e:
            print(f"[warn] Cannot read final geometry via ASE for {outf.name}: {e}")
            continue

        # final positions + per-atom flags (verbatim)
        parsed = parse_final_positions_and_flags(outf, nat_expect=nat_expect, cell_A=cell_A_fallback)
        if parsed:
            symbols, pos_A, flags_tokens = parsed
        else:
            symbols = [a.symbol for a in at]
            pos_A   = at.get_positions()
            flags_tokens = [None] * len(symbols)
            print(f"[note] No parseable final ATOMIC_POSITIONS in {outf.name}; using ASE coords (no flags).")

        # mags + extra &SYSTEM flags copied from OUT
        magn_map = starting_magn(outf)
        extras   = extra_system_flags(outf)

        # destinations use canonical base only
        in_path  = DEST_DIR / f"{canon}.in"
        lsf_path = DEST_DIR / f"{canon}.lsf"

        # Prefer CELL_PARAMETERS from OUT, else from matching IN beside OUT
        cell_block = read_cell_block_from_out(outf)
        if cell_block is None:
            src_in = outf.with_suffix(".in")
            cell_block = read_cell_block_verbatim(src_in)
        cell_block = normalize_cell_block(cell_block)

        # write files
        write_pw_in(
            symbols, pos_A, cell_A_fallback, in_path,
            KPOINTS_FINE, magn_map, PSEUDO, flags_tokens,
            extra_sys=extras, cell_block_verbatim=cell_block
        )
        write_lsf(lsf_path, in_path.name)
        convert_to_unix(in_path)
        convert_to_unix(lsf_path)

        n_ok += 1
        print(f"[ok] {outf.name} → {in_path.name}")

    print(f"[done] Wrote {n_ok} SCF inputs to: {DEST_DIR}")


if __name__ == "__main__":
    main()
